# src/app/main.py
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from prometheus_fastapi_instrumentator import Instrumentator
from transformers import pipeline

from .schemas import PredictRequest, PredictResponse

# Configuration comes from environment variables, with safe defaults
MODEL_DIR = os.getenv("MODEL_DIR", "models/sentiment")
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

logging.basicConfig(level=LOG_LEVEL)
logger = logging.getLogger("sentiment-api")

classifier = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once at startup, not on every request."""
    global classifier
    logger.info("Loading model from %s", MODEL_DIR)
    classifier = pipeline("text-classification", model=MODEL_DIR, tokenizer=MODEL_DIR)
    logger.info("Model loaded")
    yield
    classifier = None


app = FastAPI(title="Sentiment API", version="1.0.0", lifespan=lifespan)

# Adds request count and latency metrics, exposed at /metrics
Instrumentator().instrument(app).expose(app)


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": classifier is not None}


@app.post("/predict", response_model=PredictResponse)
def predict(req: PredictRequest):
    if classifier is None:
        raise HTTPException(status_code=503, detail="Model not loaded yet")
    # truncation=True stops very long reviews from exceeding the model's limit
    result = classifier(req.text, truncation=True)[0]
    return PredictResponse(
        label=result["label"].lower(),          # "POSITIVE" -> "positive"
        score=round(float(result["score"]), 4),
    )