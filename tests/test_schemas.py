# tests/test_schemas.py
import pytest
from pydantic import ValidationError

from src.app.schemas import PredictRequest


def test_valid_request_is_accepted():
    assert PredictRequest(text="good film").text == "good film"


def test_empty_text_is_rejected():
    with pytest.raises(ValidationError):
        PredictRequest(text="")