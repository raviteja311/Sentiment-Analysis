"""Fine-tune the Twitter-RoBERTa checkpoint on tweet_eval sentiment.

The script is named train_bert.py and writes to models/bert/ for historical
reasons. The model is RoBERTa: models/bert/config.json reports model_type
"roberta", and the tokenizer is byte-pair (vocab.json + merges.txt) rather than
BERT's WordPiece.

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

import numpy as np
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
    BERT_CONFIG,
    BERT_DIR,
    ID2LABEL,
    LABEL2ID,
    METRICS_DIR,
    NUM_LABELS,
    REPORTS_DIR,
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


def trainer_metrics(eval_pred):
    """Scalar metrics for Trainer's own logging and checkpoint selection."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }


def main(config=BERT_CONFIG):
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    BERT_DIR.mkdir(parents=True, exist_ok=True)

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
        output_dir=str(BERT_DIR),
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
        logging_dir=str(REPORTS_DIR / "tb_logs" / "bert"),
        seed=SEED,
        push_to_hub=False,
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

    LOGGER.info("Saving model and tokenizer to %s", BERT_DIR)
    trainer.save_model(str(BERT_DIR))
    tokenizer.save_pretrained(str(BERT_DIR))

    report = build_report(
        model="bert",
        hyperparameters=asdict(config),
        dataset=describe(dataset),
        metrics=metrics,
    )
    report_path = METRICS_DIR / "bert.json"
    save_json(report, report_path)
    LOGGER.info("Wrote metrics: %s", report_path)

    return report


if __name__ == "__main__":
    main()
