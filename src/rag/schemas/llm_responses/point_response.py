from pydantic import BaseModel


class PointResponse(BaseModel):
    text: str
    source_indices: list[int] = []
    abstain: bool = False
