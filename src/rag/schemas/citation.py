from pydantic import BaseModel, Field

MetadataValue = str | int | float | bool


class Citation(BaseModel):
    doc_id: str
    chunk_id: str
    chunk_index: int

    text: str
    score: float

    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
