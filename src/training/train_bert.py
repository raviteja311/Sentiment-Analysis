
import os
from pathlib import Path
import numpy as np
import torch

from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding,
)
from sklearn.metrics import accuracy_score, f1_score
from src.utils.preprocessing import preprocess_tweet


MODEL_NAME = "cardiffnlp/twitter-roberta-base-sentiment"  
OUT_DIR = "models/bert"
NUM_LABELS = 3
BATCH_SIZE = 8       
EPOCHS = 3
LR = 2e-5
WEIGHT_DECAY = 0.01
MAX_LEN = 128
SEED = 42
FP16 = False           
# ---------------------------

def set_seed(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def preprocess_examples(examples):
    texts = [preprocess_tweet(t) for t in examples["text"]]
    return {"text": texts, "label": examples["label"]}

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "f1_macro": f1_score(labels, preds, average="macro"),
    }

def main():
    set_seed(SEED)
    Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

    print("Loading dataset (tweet_eval: sentiment)...")
    ds = load_dataset("cardiffnlp/tweet_eval", "sentiment")
 
    ds = ds.map(preprocess_examples, batched=True)

    print("Loading tokenizer & model:", MODEL_NAME)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_NAME, num_labels=NUM_LABELS)

    def tokenize_fn(batch):
        return tokenizer(batch["text"], truncation=True, padding=False, max_length=MAX_LEN)

    print("Tokenizing dataset...")
    tokenized = ds.map(tokenize_fn, batched=True, remove_columns=["text"])
    tokenized = tokenized.rename_column("label", "labels")
    tokenized.set_format(type="torch")

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    training_args = TrainingArguments(
        output_dir=OUT_DIR,
        eval_strategy="epoch",
        save_strategy="epoch",
        per_device_train_batch_size=BATCH_SIZE,
        per_device_eval_batch_size=BATCH_SIZE,
        num_train_epochs=EPOCHS,
        learning_rate=LR,
        weight_decay=WEIGHT_DECAY,
        logging_steps=50,
        save_total_limit=2,
        load_best_model_at_end=True,
        metric_for_best_model="f1_macro",
        greater_is_better=True,
        fp16=FP16,
        seed=SEED,
        push_to_hub=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        eval_dataset=tokenized["validation"],
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    print("Starting training...")
    trainer.train()

    print("Evaluating on test set...")
    metrics = trainer.evaluate(eval_dataset=tokenized["test"])
    print("Test metrics:", metrics)

    print("Saving model & tokenizer to", OUT_DIR)
    trainer.save_model(OUT_DIR)
    tokenizer.save_pretrained(OUT_DIR)
    print("Saved. Use src/models/bert_wrapper.py to run inference.")

if __name__ == "__main__":
    main()
