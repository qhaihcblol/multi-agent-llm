from typing import Any

from pydantic import BaseModel, Field


class Citation(BaseModel):
    id: str
    text: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    score: float
