from pydantic import BaseModel
from typing import Any


class Citation(BaseModel):
    id: str
    document: str
    metadata: dict[str, Any] = {}
    distance: float
