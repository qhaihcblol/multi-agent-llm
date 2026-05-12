from pydantic import BaseModel, Field

from .citation import Citation


class Point(BaseModel):
    id: str = Field(..., description="Unique point ID")
    text: str = Field(..., min_length=1)
    citations: list[Citation] = Field(default_factory=list)
    grounding_score: float | None = None
    