from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    """Request body for JSON prediction calls."""

    tweet: str = Field(..., min_length=1)
    keyword: Optional[str] = ""


class PredictResponse(BaseModel):
    """Prediction result returned by the API and rendered by the UI."""

    probability: float
    label: int
    label_name: str
    threshold: float
    backend: str
    warnings: List[Dict[str, Any]] = Field(default_factory=list)


class PredictionLog(BaseModel):
    """Stored prediction record returned by the logs endpoint."""

    id: int
    created_at: str
    tweet: str
    keyword: Optional[str] = None
    final_text: str
    probability: float
    label: int
    label_name: str
    threshold: float
    backend: str
    warnings: List[Dict[str, Any]] = Field(default_factory=list)
