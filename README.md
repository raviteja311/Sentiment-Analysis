# Sentiment Analysis

Three-class sentiment classification for tweets - **negative**, **neutral**, **positive** -
with four models trained on [`cardiffnlp/tweet_eval`](https://huggingface.co/datasets/cardiffnlp/tweet_eval)
(`sentiment` config). Served as a FastAPI service and a Streamlit app, with a
containerised CPU image and CI.

## Results

Measured on the full test split (12,284 tweets). Every figure here is generated
by `make evaluate` and read from `reports/metrics/*.json` - **do not edit this
table by hand**, regenerate it.

| Model | Split | Examples | Accuracy | Macro F1 | Macro recall | F1 negative | F1 neutral | F1 positive |
|---|---|---|---|---|---|---|---|---|
| Twitter-RoBERTa (base, not fine-tuned) | test | 12284 | 0.7246 | 0.7240 | 0.7276 | 0.7447 | 0.7150 | 0.7125 |
| Twitter-RoBERTa (base, not fine-tuned) | validation | 2000 | 0.8055 | 0.7996 | 0.8080 | 0.7704 | 0.7864 | 0.8418 |
| cardiffnlp/twitter-roberta-base-sentiment-latest (base, not fine-tuned) | validation | 2000 | 0.7715 | 0.7610 | 0.7769 | 0.7116 | 0.7494 | 0.8220 |
| cardiffnlp/twitter-roberta-base-sentiment-latest (base, not fine-tuned) | test | 12284 | 0.7234 | 0.7255 | 0.7363 | 0.7450 | 0.7054 | 0.7262 |
| Twitter-RoBERTa | validation | 2000 | 0.7945 | 0.7864 | n/a | 0.7539 | 0.7637 | 0.8417 |
| Twitter-RoBERTa | test | 12284 | 0.7109 | 0.7124 | n/a | 0.7328 | 0.6954 | 0.7091 |
| Bi-LSTM | validation | 2000 | 0.6705 | 0.6450 | 0.6359 | 0.5590 | 0.6947 | 0.6814 |
| Bi-LSTM | test | 12284 | 0.6435 | 0.6304 | 0.6188 | 0.6140 | 0.6756 | 0.6015 |
| Bi-GRU | validation | 2000 | 0.6825 | 0.6580 | 0.6521 | 0.5676 | 0.7001 | 0.7062 |
| Bi-GRU | test | 12284 | 0.6387 | 0.6300 | 0.6245 | 0.6220 | 0.6615 | 0.6066 |
| Logistic Regression | validation | 2000 | 0.6575 | 0.6390 | 0.6649 | 0.5450 | 0.6505 | 0.7216 |
| Logistic Regression | test | 12284 | 0.5848 | 0.5825 | 0.5970 | 0.6029 | 0.5781 | 0.5665 |

The transformer leads by a clear margin. The two recurrent models are within
half a point of each other - treat them as equivalent - and beat the linear
baseline by 5-6 points of accuracy. Both initialise their embeddings from
GloVe Twitter vectors (95.8% vocabulary coverage), which is worth roughly
+0.03 macro F1 and +0.05 positive-class F1 over learning them from scratch.

The rows marked "base, not fine-tuned" are checkpoints exactly as published,
scored by `python -m src.evaluate --include-base`: the transformer's base, and
`twitter-roberta-base-sentiment-latest` as a candidate replacement for it.
Both are already fine-tuned on this dataset, and both score **above** our
fine-tuned run on test; the model card says what that means and why the
served transformer is left as it is. Macro recall is TweetEval's official metric
for this task; `n/a` marks records written before it was reported.
See [MODEL_CARD.md](MODEL_CARD.md) for limitations and intended use.

Each record in `reports/metrics/` stores the hyperparameters, the split sizes and
the class distribution alongside the scores, so any number above can be traced
to what produced it.

## Models

| Key | Model | Framework | Artifact |
|---|---|---|---|
| `lr` | TF-IDF + Logistic Regression | scikit-learn | `models/lr/pipeline.joblib` |
| `lstm` | Bi-LSTM over GloVe Twitter embeddings | Keras 3 | `models/lstm/model_final.keras` |
| `gru` | Bi-GRU over GloVe Twitter embeddings | Keras 3 | `models/gru/model_final.keras` |
| `roberta` | Fine-tuned `cardiffnlp/twitter-roberta-base-sentiment` | Transformers | `models/roberta/` |

The transformer was keyed as `bert` until the name was corrected to match the
checkpoint - it is **RoBERTa** (`model_type: roberta`, byte-pair tokenizer).
`"model": "bert"` is still accepted as a deprecated alias and resolves to
`roberta`, so existing requests keep working.

## Dataset

| Split | Examples |
|---|---|
| train | 45,615 |
| validation | 2,000 |
| test | 12,284 |

The training split is imbalanced: neutral 20,673, positive 17,849, negative
7,093. The dataset downloads automatically on the first training or evaluation
run; it is not needed for inference.

## Quick start

Requires **Python 3.12** (3.11 is also covered by CI).

```bash
git clone https://github.com/raviteja311/Sentiment-Analysis.git
cd Sentiment-Analysis
python -m venv .venv && .venv/Scripts/activate   # Linux/macOS: source .venv/bin/activate
pip install --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements/inference-cpu.lock           # or: make install
make fetch-weights                               # 574 MB, see Weights below
```

Keep the `--extra-index-url`. Without it pip downloads the default PyTorch
wheel, which bundles CUDA and pulls in several gigabytes of NVIDIA libraries
that a CPU install never uses. On macOS and Linux ARM the requirements file
selects the plain `tensorflow` package automatically, because `tensorflow-cpu`
has no wheels for those platforms.

### Weights

The large weights are **not in git**. They live on the Hugging Face Hub at
[RAVITEJA311/sentiment-analysis-models](https://huggingface.co/RAVITEJA311/sentiment-analysis-models)
and are fetched on demand:

```bash
make fetch-weights                        # all of them, 574 MB
python -m src.artifacts --models lstm     # or just one
```

Point it elsewhere with `SENTIMENT_MODELS_REPO`.

Downloads are pinned and verified. `models/remote_manifest.json` records the
Hub commit that matches the committed tokenizers and calibration files, and the
sha256 of every fetched file; a download that does not match is deleted and
reported. `SENTIMENT_MODELS_REVISION` overrides the pin. After publishing new
weights, `make publish-weights` rewrites the manifest so the pin moves with
them; `python -m src.artifacts --write-manifest` does the same for weights
already on the Hub.

Skipping this step is not fatal. The Logistic Regression pipeline, the
tokenizers and the transformer's config and vocabulary are small enough to stay
in git, so a plain clone still serves predictions from `lr`; the neural models
report themselves unavailable with the command that fixes it, the UI hides them,
and the API answers 503 rather than 500.

### API

```bash
make api          # or: uvicorn api.main:app --host 0.0.0.0 --port 8000
```

| Endpoint | Purpose |
|---|---|
| `GET /health` | liveness; answers even when no model is usable |
| `GET /ready` | readiness; 503 when no model can be served |
| `GET /models` | every model: availability, reason, version and per-file digests |
| `POST /predict` | classify one text |
| `POST /predict/batch` | classify up to 256 texts |

```bash
curl -X POST localhost:8000/predict \
  -H 'Content-Type: application/json' \
  -d '{"text":"this is fantastic","model":"roberta"}'
```

```json
{
  "model": "roberta",
  "version": "a378eddd7113",
  "label": "positive",
  "confidence": 0.9474,
  "probabilities": {"negative": 0.0210, "neutral": 0.0315, "positive": 0.9474}
}
```

`version` is a digest of the artifact that produced the prediction - the
weights, the tokenizer and the calibration file - so a logged prediction stays
traceable after a retrain or a re-fetch.

`GET /models` reports the version **actually being served**. A predictor is
cached once loaded, so replacing weights on disk does not change what answers
requests; the response therefore also carries `loaded`, `version_on_disk` and
`stale`, which tells you a restart is needed rather than quietly reporting a
version nothing is using.

`confidence` is **calibrated**, not a raw softmax - see
[Calibration](#calibration).

Interactive docs are at `/docs`.

Models load on first use, so the process is up quickly and a broken artifact
cannot stop it from starting. Three optional variables tune serving:

| Variable | Default | Purpose |
|---|---|---|
| `PRELOAD_MODELS` | unset | keys to load at startup, e.g. `lr,roberta` or `all`; one that fails to load is logged and skipped |
| `MAX_CONCURRENT_INFERENCE` | unset (unlimited) | cap on simultaneous forward passes; the endpoints run in a 40-thread pool, which oversubscribes a CPU |
| `TORCH_NUM_THREADS` | torch's default | intra-op threads for the transformer, applied once at load |

### Logging

The service logs one JSON object per line. Every request is assigned an
`X-Request-ID` (an inbound one is honoured if it is safe to log) which is
echoed in the response and attached to every log line produced while serving it,
so a prediction can be traced from the access line to the model version that
produced it:

```json
{"timestamp": "2026-09-22T10:17:20.105Z", "level": "INFO", "logger": "api.access",
 "message": "request", "request_id": "ad56b98b50aa", "method": "POST",
 "path": "/predict", "status": 200, "duration_ms": 1.59}
```

**The text being classified is never logged** - only its length. Set
`LOG_FORMAT=plain` for readable local output and `LOG_LEVEL` to change verbosity.

### Rate limiting

Callers get `60/minute` by default, **shared across endpoints** so the budget
cannot be spent twice by alternating between `/predict` and `/predict/batch`.
Exceeding it returns **429** with a `Retry-After` header, and the remaining
budget is reported in `X-RateLimit-*`. Health, readiness, `/models` and the
OpenAPI docs are exempt - throttling a liveness probe gets a healthy container
restarted, and throttling `/docs` just makes the service look broken.

That limit counts requests, and a batch is one request however many texts it
carries. A second budget therefore counts **texts**: `/predict` costs 1,
`/predict/batch` costs one per text, and a batch that would overrun the budget
is rejected whole with **429** without consuming anything. A batch larger than
the entire budget says so in its `detail`, since waiting would not help.

| Variable | Default | Purpose |
|---|---|---|
| `RATE_LIMIT` | `60/minute` | request limit, or `off` to disable |
| `TEXT_RATE_LIMIT` | `1024/minute` | text limit (a batch of N costs N), or `off` to disable |
| `RATE_LIMIT_STORAGE_URI` | `memory://` | set to `redis://...` to share counters across workers |
| `TRUST_PROXY_HEADERS` | unset | use `X-Forwarded-For` for the caller's identity |

Two limits of this, worth knowing before relying on it. With the in-memory
store each worker counts separately, so N workers give a caller N times the
limit; use Redis to share state. And `TRUST_PROXY_HEADERS` should only be set
when a proxy you control overwrites that header - otherwise a caller sets it
themselves and bypasses the limit.

**This is a backstop, not the real defence.** Rate limiting belongs at the edge,
in an ingress or API gateway, where it applies before a request reaches any
application process.

### Streamlit app

```bash
make ui           # or: streamlit run app/streamlit_app.py
```

Besides the prediction and its probabilities, the app shows the selected
model's test-split confusion matrix, read from its record in
`reports/metrics/`, and for the Logistic Regression model the terms that
pushed the text towards or away from its label: each term's tf-idf value
times its coefficient for the predicted class. The neural models get no such
explanation, because their decisions are not sums of per-term weights and a
made-up one would be worse than none.

### Docker

```bash
docker build --target cpu -t sentiment-analysis-api:local .
docker run --rm -p 8000:8000 sentiment-analysis-api:local
docker compose up --build      # API on 8000, UI on 8501
```

The image carries only the small artifacts, so it serves `lr` out of the box and
stays under a gigabyte. Bake the rest in at build time, or mount them at run
time as docker-compose does:

```bash
docker build --target cpu --build-arg FETCH_MODELS="lstm gru roberta" .
```

## Calibration

Raw softmax outputs are overconfident: before calibration the transformer
averaged 0.915 confidence while being right 71.1% of the time. `make calibrate`
fits a temperature per model on the validation split, adopts it only if it also
helps on held-out validation folds, and stores it in
`models/<key>/calibration.json`; the predictor applies it, so the API, the UI
and the evaluation all report the same calibrated numbers. This table is
printed by `python -m src.calibration --table-only` - regenerate it, do not
edit it.

| Model | Temperature | ECE before | ECE after | Mean confidence before | after | Accuracy |
|---|---|---|---|---|---|---|
| roberta | 1.99 | 0.2039 | 0.0865 | 0.915 | 0.796 | 0.7109 |
| lstm | 1.00 (not adopted) | 0.0312 | 0.0312 | 0.672 | 0.672 | 0.6435 |
| gru | 1.12 | 0.0319 | 0.0238 | 0.656 | 0.634 | 0.6387 |
| lr | 1.00 (not adopted) | 0.0442 | 0.0442 | 0.613 | 0.613 | 0.5848 |

Fitted on validation, measured on test. Temperature scaling is monotonic, so no
prediction changes and the results table above is unaffected - only the spread
of the probabilities moves.

The linear baseline and the Bi-LSTM are left uncalibrated. The temperature is
constrained to at least 1.0, so calibration can only soften confidence, and a
candidate is adopted only when it also lowers ECE on validation folds it was
not fitted to. For the linear model the bounded fit lands on exactly 1.0: it
is underconfident on validation and overconfident on test (61.3% confidence,
58.5% accuracy), because validation is the easier split, and sharpening it
would overstate confidence. For the retrained Bi-LSTM the candidate (1.11)
gained nothing held out, and its raw ECE of 0.031 is already the best of the
non-transformer models, so it too is served at 1.0.

## Retraining

```bash
pip install -r requirements/train.txt
make train-lr        # ~5 min, CPU
make train-lstm      # ~6 min, CPU
make train-gru       # ~6 min, CPU
make train-roberta   # ~2h15m on a GTX 1650 with mixed precision
make evaluate        # scores every available model, regenerates the table above
python -m src.evaluate --include-base   # also the base checkpoint, not fine-tuned
make calibrate       # refits temperature scaling after retraining
make behaviour       # CheckList-style probes: negation, entities, emoji, invariances
make experiments     # class-weight, decision-bias and label-shift comparisons, validation only
```

`make behaviour` asks targeted questions the aggregate metrics cannot: does
negating a sentence change its label, does swapping one username or URL for
another leave it alone, does a heart read as positive. It prints a pass rate
per check and writes `reports/metrics/behaviour_<model>.json`. It is a report,
not a CI gate, because model behaviour can be measured but not guaranteed.

Training writes both the artifact and a metrics record to `reports/metrics/`.
`make evaluate` merges its results into those records rather than replacing
them, so the training provenance survives. Publish retrained weights with
`make publish-weights` (needs `huggingface-cli login`).

The transformer can be fine-tuned from another checkpoint without replacing
the served one. `cardiffnlp/twitter-roberta-base-sentiment-latest` is the same
architecture trained by the same group on a larger, more recent tweet corpus:

```bash
python -m src.evaluate --include-base \
    --base-model cardiffnlp/twitter-roberta-base-sentiment-latest   # zero-shot, no GPU needed
python -m src.training.train_roberta \
    --base-model cardiffnlp/twitter-roberta-base-sentiment-latest \
    --out-dir reports/experiments/scratch/roberta_latest             # the fine-tuning run
```

The variant's weights, record and history land under `--out-dir`; compare its
validation scores in `metrics.json` with `reports/metrics/roberta.json`, and
only if it wins change `RobertaConfig.base_model` and retrain into the served
location. Test is consulted for the final table, not for the choice.

Preprocessing is versioned. Each model trains with the spec named for it in
`TRAIN_PREPROCESSING` (`src/config.py`) and records it in
`models/<key>/preprocessing.json`; serving reads that file, so a change to the
training spec does not alter what an already trained model is shown. An
artifact without the file was trained with `glove-v1`, the original behaviour.
The specs themselves are in `src/utils/preprocessing.py`.

TensorFlow has no native Windows GPU support from 2.11 onward, so the LSTM and
GRU train on CPU there regardless of what hardware is present. The transformer
uses CUDA when available and enables mixed precision automatically, which is
what keeps it inside 4 GB of VRAM.

## Tests and quality

```bash
make test         # 480 tests
make lint         # ruff + black
make lock         # regenerate requirements/*.lock after editing requirements/*.txt
```

Dependencies are locked. `requirements/*.txt` name the direct dependencies and
their reasons; `requirements/*.lock`, generated by `uv pip compile` through
`make lock`, pin the whole closure so that transitive packages such as
starlette cannot change under a fresh install. The locks are universal: one
file serves Linux, Windows and macOS through environment markers, with torch
taken from the CPU index where a `+cpu` wheel exists. `make install`, CI and
the Docker image all install from the locks; edit a `.txt`, run `make lock`,
and commit both.

Tests skip themselves rather than fail when a model's weights are absent, so the
suite is meaningful on a clone where `make fetch-weights` has not been run. CI runs lint, the test suite on
Python 3.11 and 3.12, and a container smoke test that asserts real predictions
from the image and a 503 when no artifacts are present.

## Project structure

```
.
├── api/                  # FastAPI service and structured logging
├── app/                  # Streamlit UI
├── models/               # Small artifacts; large weights fetched from the Hub
├── reports/metrics/      # Generated metrics records
├── requirements/         # base, inference-cpu, train, dev
├── src/
│   ├── artifacts.py      # fetches the large weights
│   ├── calibration.py    # temperature scaling
│   ├── config.py         # labels, paths, hyperparameters
│   ├── data.py           # dataset loading
│   ├── embeddings.py     # pretrained GloVe vectors
│   ├── evaluate.py       # evaluation and the results table
│   ├── inference/        # shared predictor and model versioning
│   ├── models/           # Keras architectures
│   ├── training/         # training entry points
│   └── utils/            # preprocessing, metrics, IO
├── scripts/              # publish_weights.py
├── tests/
├── Dockerfile
├── docker-compose.yml
└── Makefile
```

## License

See [LICENSE](LICENSE).
