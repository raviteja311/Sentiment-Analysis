"""What drove a transformer prediction: Integrated Gradients over its tokens.

The linear model's decision is a sum of per-term weights, so
:mod:`src.inference.explain` can show it exactly. RoBERTa's is not, and its
attention weights are not an explanation either: they say where a layer looked,
not what changed the output. Integrated Gradients (Sundararajan et al., 2017)
answers the question that matters, how much each token moved the score of the
predicted class, by integrating the gradient along a straight path from a
neutral baseline to the actual input::

    attribution_i = (x_i - x'_i) * mean over a in (0, 1] of dF(x' + a(x - x')) / dx_i

``x`` is the token embeddings, ``x'`` the baseline (every content token
replaced by ``<pad>``, the special tokens kept), and ``F`` the logit of the
explained class. The method's defining property, completeness, is that the
attributions sum to ``F(x) - F(x')``; the gap left over is reported as
``convergence_delta`` and shrinks as ``steps`` grows. A large delta means the
numbers should not be trusted, so the check travels with every explanation.

Implemented directly on the embedding layer rather than through captum: it is
the whole algorithm, and one fewer locked dependency.
"""

from __future__ import annotations

from dataclasses import dataclass

from src.config import LABELS, ROBERTA_CONFIG
from src.utils.preprocessing import preprocess_tweet

# On the served model 32 steps left a completeness gap of 0.6 logits and 128
# under 0.03 (under 1% of the explained difference), at about a second a tweet
# on a GTX 1650.
DEFAULT_STEPS = 128

# RoBERTa's byte-level BPE marks a token that starts a new word with this
# character (an encoded leading space).
WORD_START = "Ġ"


@dataclass(frozen=True)
class TokenAttribution:
    token: str
    # Contribution to the explained class's logit. Positive pushes towards the
    # class, negative away from it.
    score: float


@dataclass(frozen=True)
class Attribution:
    label: str
    # Sub-word tokens as the model saw them, special tokens excluded.
    tokens: list[TokenAttribution]
    # The same scores summed back into whitespace-separated words.
    words: list[TokenAttribution]
    # F(x) - F(baseline) minus the attributions' sum. Near 0 when converged.
    convergence_delta: float


def integrated_gradients(
    model,
    tokenizer,
    text: str,
    target: int | None = None,
    steps: int = DEFAULT_STEPS,
    max_length: int = ROBERTA_CONFIG.max_len,
) -> Attribution:
    """Attribute ``text``'s logit for ``target`` to its tokens.

    ``text`` must already be preprocessed the way the model was trained;
    :func:`explain_roberta` does that for the served predictor. ``target``
    defaults to the predicted class.
    """
    import torch

    device = next(model.parameters()).device
    encoded = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
    input_ids = encoded["input_ids"].to(device)
    attention_mask = encoded["attention_mask"].to(device)

    embedding_layer = model.get_input_embeddings()
    special = torch.tensor(
        tokenizer.get_special_tokens_mask(
            input_ids[0].tolist(), already_has_special_tokens=True
        ),
        dtype=torch.bool,
        device=device,
    )
    baseline_ids = torch.where(
        special, input_ids[0], torch.tensor(tokenizer.pad_token_id, device=device)
    ).unsqueeze(0)

    with torch.no_grad():
        inputs = embedding_layer(input_ids)
        baseline = embedding_layer(baseline_ids)

    def logits_for(embeddings):
        mask = attention_mask.expand(embeddings.shape[0], -1)
        return model(inputs_embeds=embeddings, attention_mask=mask).logits

    with torch.no_grad():
        endpoint_logits = logits_for(torch.cat([inputs, baseline]))
    if target is None:
        target = int(endpoint_logits[0].argmax())

    # Midpoint Riemann sum: alphas at the centres of `steps` equal intervals.
    alphas = (torch.arange(steps, device=device, dtype=inputs.dtype) + 0.5) / steps
    path = baseline + alphas.view(-1, 1, 1) * (inputs - baseline)
    path.requires_grad_(True)
    # Gradients with respect to the path only; the weights need none.
    (gradients,) = torch.autograd.grad(logits_for(path)[:, target].sum(), path)

    average_gradient = gradients.mean(dim=0)
    scores = ((inputs[0] - baseline[0]) * average_gradient).sum(dim=-1)

    expected = float(endpoint_logits[0, target] - endpoint_logits[1, target])
    delta = expected - float(scores.sum())

    tokens = tokenizer.convert_ids_to_tokens(input_ids[0].tolist())
    keep = (~special).tolist()
    token_scores = [
        TokenAttribution(token=token, score=float(score))
        for token, score, kept in zip(tokens, scores.tolist(), keep, strict=True)
        if kept
    ]
    return Attribution(
        label=LABELS[target],
        tokens=token_scores,
        words=merge_into_words(token_scores),
        convergence_delta=delta,
    )


def merge_into_words(tokens: list[TokenAttribution]) -> list[TokenAttribution]:
    """Sum sub-word scores into words, so "unbelievable" reads as one unit.

    Attributions are additive, so a word's score is the sum of its pieces'.
    """
    words: list[TokenAttribution] = []
    for token in tokens:
        piece = token.token
        if words and not piece.startswith(WORD_START):
            last = words[-1]
            words[-1] = TokenAttribution(last.token + piece, last.score + token.score)
        else:
            words.append(TokenAttribution(piece.lstrip(WORD_START), token.score))
    return words


def explain_roberta(
    predictor, text: str, target: int | None = None, steps: int = DEFAULT_STEPS
) -> Attribution:
    """Integrated Gradients for the served transformer.

    Applies the predictor's own preprocessing first, so the tokens explained
    are the tokens the model actually classified.
    """
    cleaned = preprocess_tweet(text, predictor.preprocessing)
    return integrated_gradients(
        predictor._model, predictor._tokenizer, cleaned, target=target, steps=steps
    )
