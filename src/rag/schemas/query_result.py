from dataclasses import dataclass, field

from .citation import Citation
from .retrieved_chunk import RetrievedChunk


@dataclass(slots=True)
class QueryResult:
    question: str
    answer: str
    citations: list[Citation] = field(default_factory=list)
    chunks: list[RetrievedChunk] = field(default_factory=list)
    system_prompt: str = ""
    prompt: str = ""
