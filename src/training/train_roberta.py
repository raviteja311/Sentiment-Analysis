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
"""

import logging
from dataclasses import asdict
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
)
from src.data import describe, load_raw_dataset
from src.utils.io import save_json
from src.utils.metrics import build_report, compute_metrics
from src.utils.preprocessing import preprocess_tweet

LOGGER = logging.getLogger(__name__)


def preprocess_examples(examples):
    return {
        "text": [preprocess_tweet(text) for text in examples["text"]],
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


def trainer_metrics(eval_pred):
    """Scalar metrics for Trainer's own logging and checkpoint selection."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(config=ROBERTA_CONFIG):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    ROBERTA_DIR.mkdir(parents=True, exist_ok=True)

    dataset = load_raw_dataset()
    dataset = dataset.map(preprocess_examples, batched=True)

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
        output_dir=str(ROBERTA_DIR),
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
        logging_dir=str(REPORTS_DIR / "tb_logs" / "roberta"),
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
    history_path = METRICS_DIR / "roberta_training_history.json"
    save_json(training_history(trainer.state), history_path)
    LOGGER.info("Wrote training history: %s", history_path)

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

    LOGGER.info("Saving model and tokenizer to %s", ROBERTA_DIR)
    trainer.save_model(str(ROBERTA_DIR))
    tokenizer.save_pretrained(str(ROBERTA_DIR))

    report = build_report(
        model="roberta",
        hyperparameters=asdict(config),
        dataset=describe(dataset),
        metrics=metrics,
    )
    report_path = METRICS_DIR / "roberta.json"
    save_json(report, report_path)
    LOGGER.info("Wrote metrics: %s", report_path)

    return report


if __name__ == "__main__":
    main()
