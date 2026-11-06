# src/prepare_data.py
"""Download IMDB, take a reproducible subset, and write CSV files."""
from pathlib import Path

import yaml
from datasets import load_dataset

params = yaml.safe_load(Path("params.yaml").read_text())["train"]
Path("data").mkdir(exist_ok=True)

ds = load_dataset("stanfordnlp/imdb")  # if this name fails, try "imdb"

train = (ds["train"].shuffle(seed=42)
         .select(range(params["train_size"])).to_pandas())
test = (ds["test"].shuffle(seed=42)
        .select(range(params["test_size"])).to_pandas())

train[["text", "label"]].to_csv("data/train.csv", index=False)
test[["text", "label"]].to_csv("data/test.csv", index=False)
print(f"Wrote {len(train)} train and {len(test)} test rows")