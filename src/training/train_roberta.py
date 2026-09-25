"""Fine-tune the Twitter-RoBERTa checkpoint on tweet_eval sentiment.

The tokenizer is byte-pair (vocab.json + merges.txt) rather than BERT's
WordPiece. Both the script and the directory were called "bert" until the name
was corrected; `config.MODEL_ALIASES` keeps the old key working.

Two things changed after the first training run:

* The saved config carried HuggingFace's placeholder labels (LABEL_0, LABEL_1,
  LABEL_2), so anyone loading the checkpoint with transformers.pipeline() got
  meaningless class names. Real label names are now written into the config.
* The run trained for 3 epochs and kept 2 checkpoints, at 498 MB each. Its own
  trainer_state.json shows epoch 1 was the best (macro F1 0.7806) while
  validation loss climbed from 0.5146 to 0.9767 by epoch 3 - the extra epochs
  bought overfitting and a gigabyte of duplicated weights.

Run it as a module. Without arguments it trains the configured base into
models/roberta/ and writes the served record::

    python -m src.training.train_roberta

To try another base checkpoint without touching the served artifact, train a
variant into a scratch directory and compare its validation scores with the
served record's; test is consulted only for the final table::

    python -m src.training.train_roberta \\
        --base-model cardiffnlp/twitter-roberta-base-sentiment-latest \\
        --out-dir reports/experiments/scratch/roberta_latest
"""

import argparse
import logging
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    EarlyStoppingCallback,
    Trainer,
    TrainingArguments,
)

from src.config import (
    ID2LABEL,
    LABEL2ID,
    METRICS_DIR,
    NUM_LABELS,
    PROJECT_ROOT,
    REPORTS_DIR,
    ROBERTA_CONFIG,
    ROBERTA_DIR,
    SEED,
    TRAIN_PREPROCESSING,
)
from src.data import describe, load_raw_dataset
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics
from src.utils.preprocessing import preprocess_tweet, write_artifact_spec

LOGGER = logging.getLogger(__name__)


def preprocess_examples(examples, spec: str):
    return {
        "text": [preprocess_tweet(text, spec) for text in examples["text"]],
        "label": examples["label"],
    }


def relative_to_project(path: str | None) -> str | None:
    """A path as recorded in a metrics file: relative to the project root.

    The Trainer records the checkpoint it kept as an absolute path, which
    once put a Windows user directory into a committed report. A report is
    read on other machines, so it names files the way the repository does.
    """
    if path is None:
        return None
    resolved = Path(path).resolve()
    try:
        return resolved.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return resolved.as_posix()


def training_history(state) -> dict:
    """Per-epoch validation scores from a Trainer's state, for the report.

    ``state`` is ``trainer.state``: its ``log_history`` mixes training-loss
    lines with evaluation lines, and only the latter are kept. The result is
    the shape of reports/metrics/roberta_training_history.json.
    """
    evaluations = [entry for entry in state.log_history if "eval_loss" in entry]
    return {
        "model": "roberta",
        "best_metric": state.best_metric,
        "best_model_checkpoint": relative_to_project(state.best_model_checkpoint),
        "epoch": state.epoch,
        "global_step": state.global_step,
        "evaluations": evaluations,
    }


def output_paths(out_dir: Path | None) -> dict[str, Path]:
    """Where a run writes: the served locations, or all under ``out_dir``.

    A variant run must not overwrite the served weights, their metrics
    record or the training history, so everything it writes goes under one
    scratch directory.
    """
    if out_dir is None:
        return {
            "model": ROBERTA_DIR,
            "report": METRICS_DIR / "roberta.json",
            "history": METRICS_DIR / "roberta_training_history.json",
            "logs": REPORTS_DIR / "tb_logs" / "roberta",
        }
    out_dir = Path(out_dir)
    return {
        "model": out_dir,
        "report": out_dir / "metrics.json",
        "history": out_dir / "training_history.json",
        "logs": out_dir / "tb_logs",
    }


def trainer_metrics(eval_pred):
    """Scalar metrics for Trainer's own logging and checkpoint selection."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(config=ROBERTA_CONFIG, out_dir: Path | None = None):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    paths = output_paths(out_dir)
    model_dir = paths["model"]
    model_dir.mkdir(parents=True, exist_ok=True)

    spec = TRAIN_PREPROCESSING["roberta"]
    dataset = load_raw_dataset()
    dataset = dataset.map(preprocess_examples, batched=True, fn_kwargs={"spec": spec})

    LOGGER.info("Loading tokenizer and model: %s", config.base_model)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(
        config.base_model,
        num_labels=NUM_LABELS,
        # Without these two the saved config keeps HuggingFace's LABEL_0/1/2
        # placeholders and the meaning of each index lives nowhere but in our
        # own source code.
        id2label={index: name for index, name in ID2LABEL.items()},
        label2id=dict(LABEL2ID),
    )

    def tokenize(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding=False,
            max_length=config.max_len,
        )

    LOGGER.info("Tokenizing...")
    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")

    training_args = TrainingArguments(
        output_dir=str(model_dir),
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=config.batch_size,
        per_device_eval_batch_size=config.batch_size,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        weight_decay=config.weight_decay,
        logging_steps=50,
        # One intermediate checkpoint, not two. Each is ~498 MB and duplicates
        # weights that the final save writes anyway.
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        # Keep TensorBoard event files out of models/, where they were being
        # committed alongside the weights.
        logging_dir=str(paths["logs"]),
        seed=SEED,
        push_to_hub=False,
        # Mixed precision roughly halves activation memory. On a 4 GB card that
        # is the difference between fine-tuning at batch 8 and an OOM.
        fp16=torch.cuda.is_available(),
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=trainer_metrics,
        callbacks=[
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience)
        ],
    )

    LOGGER.info("Training...")
    trainer.train()

    # The per-epoch validation curve is what justifies the epoch count in
    # config; kept next to the metrics so it survives deleting the checkpoint.
    save_json(training_history(trainer.state), paths["history"])
    LOGGER.info("Wrote training history: %s", paths["history"])

    metrics = {}
    for split in ("validation", "test"):
        output = trainer.predict(tokenized[split])
        preds = np.argmax(output.predictions, axis=-1)
        metrics[split] = compute_metrics(output.label_ids, preds)
        LOGGER.info(
            "%s: accuracy=%.4f macro F1=%.4f",
            split,
            metrics[split]["accuracy"],
            metrics[split]["f1_macro"],
        )

    LOGGER.info("Saving model and tokenizer to %s", model_dir)
    trainer.save_model(str(model_dir))
    tokenizer.save_pretrained(str(model_dir))
    write_artifact_spec("roberta", spec, directory=model_dir)

    report = build_report(
        model="roberta",
        hyperparameters={**asdict(config), "preprocessing": spec},
        dataset=describe(dataset),
        metrics=metrics,
    )
    save_json(report, paths["report"])
    LOGGER.info("Wrote metrics: %s", paths["report"])

    return report


def parse_args(argv=None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--base-model",
        default=ROBERTA_CONFIG.base_model,
        help="checkpoint to fine-tune from (default: the configured base)",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help=(
            "write the weights, record and history here instead of the served "
            "locations; use for a variant that must not replace models/roberta"
        ),
    )
    return parser.parse_args(argv)


def run(argv=None) -> int:
    args = parse_args(argv)
    config = replace(ROBERTA_CONFIG, base_model=args.base_model)
    if args.out_dir is None and config.base_model != ROBERTA_CONFIG.base_model:
        LOGGER.warning(
            "Training %s into the served location. Prefer --out-dir for a variant, "
            "so the served artifact and its record are not replaced before the "
            "comparison on validation.",
            config.base_model,
        )
    main(config, out_dir=args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(run())
