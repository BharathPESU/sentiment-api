# tests/test_api.py
import pytest
from fastapi.testclient import TestClient

from src.app.main import app

pytestmark = pytest.mark.model  # these tests need models/sentiment


def test_health_ok():
    with TestClient(app) as client:  # the context manager runs the lifespan
        r = client.get("/health")
        assert r.status_code == 200
        assert r.json()["status"] == "ok"


def test_predict_returns_label_and_score():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "An absolutely wonderful film"})
        assert r.status_code == 200
        body = r.json()
        assert body["label"] in {"positive", "negative"}
        assert 0.0 <= body["score"] <= 1.0


def test_empty_text_returns_422():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": ""})
        assert r.status_code == 422


def test_long_text_is_truncated_not_crashing():
    with TestClient(app) as client:
        r = client.post("/predict", json={"text": "great movie " * 1000})
        assert r.status_code == 200