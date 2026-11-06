# src/train_baseline.py
import mlflow
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.pipeline import make_pipeline

train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")

mlflow.set_experiment("sentiment-analysis")
with mlflow.start_run(run_name="tfidf-logreg-baseline"):
    pipe = make_pipeline(
        TfidfVectorizer(max_features=20000, ngram_range=(1, 2)),
        LogisticRegression(max_iter=1000),
    )
    pipe.fit(train["text"], train["label"])
    preds = pipe.predict(test["text"])

    mlflow.log_params({"model": "tfidf-logreg", "max_features": 20000})
    mlflow.log_metrics({
        "accuracy": accuracy_score(test["label"], preds),
        "f1": f1_score(test["label"], preds),
    })
    print("Baseline accuracy:", accuracy_score(test["label"], preds))