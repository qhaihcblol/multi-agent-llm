from dataclasses import dataclass, field
from typing import Any


@dataclass(slots=True)
class Chunk:
    id: str
    text: str
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def preview(self, max_length: int = 100) -> str:
        if len(self.text) <= max_length:
            return self.text
        return self.text[:max_length] + "..."