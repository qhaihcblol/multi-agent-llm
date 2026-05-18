from pydantic import BaseModel, Field

from .citation import Citation


class Point(BaseModel):
    id: str = Field(..., description="Unique point ID")
    node_id: str = Field(..., description="Source node ID")
    text: str = Field(..., description="Extracted point text")
    citations: list[Citation] = Field(default_factory=list)
    grounding_score: float | None = None
    abstain: bool = False
