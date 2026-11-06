# src/app/schemas.py
from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    # Reject empty strings, and cap length to protect the server
    text: str = Field(..., min_length=1, max_length=50000)


class PredictResponse(BaseModel):
    label: str    # "positive" or "negative"
    score: float  # confidence between 0 and 1