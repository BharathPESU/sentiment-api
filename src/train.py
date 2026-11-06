# src/train.py
"""Fine-tune DistilBERT, evaluate it, log to MLflow, and save the model."""
import json
from pathlib import Path

import mlflow
import numpy as np
import pandas as pd
import yaml
from datasets import Dataset
from sklearn.metrics import accuracy_score, f1_score
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)

params = yaml.safe_load(Path("params.yaml").read_text())["train"]
MODEL_DIR = Path("models/sentiment")
ID2LABEL = {0: "NEGATIVE", 1: "POSITIVE"}
LABEL2ID = {v: k for k, v in ID2LABEL.items()}

# 1. Load prepared data and tokenize it
train_ds = Dataset.from_pandas(pd.read_csv("data/train.csv"))
test_ds = Dataset.from_pandas(pd.read_csv("data/test.csv"))
tok = AutoTokenizer.from_pretrained(params["model_name"])

def tokenize(batch):
    return tok(batch["text"], truncation=True, max_length=params["max_length"])

train_ds = train_ds.map(tokenize, batched=True)
test_ds = test_ds.map(tokenize, batched=True)

# 2. Build the model. id2label stores readable names in the model config.
model = AutoModelForSequenceClassification.from_pretrained(
    params["model_name"], num_labels=2, id2label=ID2LABEL, label2id=LABEL2ID
)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, preds), "f1": f1_score(labels, preds)}

args = TrainingArguments(
    output_dir="outputs",
    num_train_epochs=params["epochs"],
    per_device_train_batch_size=params["batch_size"],
    per_device_eval_batch_size=params["batch_size"] * 2,
    eval_strategy="epoch",    # older transformers: rename to evaluation_strategy
    save_strategy="no",       # we save the final model ourselves below
    logging_steps=50,
    report_to="none",         # MLflow logging is handled manually below
    seed=42,
)

# 3. Train, evaluate, and log to MLflow
mlflow.set_experiment("sentiment-analysis")
with mlflow.start_run(run_name=f"distilbert-{params['train_size']}"):
    mlflow.log_params(params)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=test_ds,
        data_collator=DataCollatorWithPadding(tok),
        compute_metrics=compute_metrics,
    )
    trainer.train()
    raw = trainer.evaluate()
    metrics = {
        "accuracy": round(raw["eval_accuracy"], 4),
        "f1": round(raw["eval_f1"], 4),
    }
    mlflow.log_metrics(metrics)

    # 4. Save the model and tokenizer
    trainer.save_model(str(MODEL_DIR))
    tok.save_pretrained(str(MODEL_DIR))

# 5. Write metrics to a file so DVC can read them (Lab 5)
Path("metrics.json").write_text(json.dumps(metrics, indent=2))
print("Saved model to", MODEL_DIR, "metrics:", metrics)