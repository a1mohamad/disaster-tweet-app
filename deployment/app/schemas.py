from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field

class PredictRequest(BaseModel):
    tweet: str = Field(..., min_length=1)
    keyword: Optional[str] = ""

class PredictResponse(BaseModel):
    probability: float
    label: int
    label_name: str
    threshold: float
    warnings: List[Dict[str, Any]] = []


class PredictionLog(BaseModel):
    id: int
    created_at: str
    tweet: str
    keyword: Optional[str] = None
    final_text: str
    probability: float
    label: int
    label_name: str
    threshold: float
    warnings: List[Dict[str, Any]] = []
