from pydantic import BaseModel
from datetime import datetime
from typing import Optional


class UserResponse(BaseModel):
    id: int
    name: str
    image_paths: Optional[str] = None
    created_at: datetime

    model_config = {"from_attributes": True}


class VerifyResponse(BaseModel):
    identity: Optional[str] = None
    similarity: float
    decision: str


class MetricsResponse(BaseModel):
    far: float
    frr: float
    eer: float
    threshold: float


class MessageResponse(BaseModel):
    message: str
