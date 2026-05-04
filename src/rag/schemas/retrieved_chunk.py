from dataclasses import dataclass, field

MetadataValue = str | int | float | bool


@dataclass(slots=True)
class RetrievedChunk:
    id: str
    text: str
    score: float
    metadata: dict[str, MetadataValue] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.text)

    def preview(self, max_length: int = 100) -> str:
        return (
            self.text
            if len(self.text) <= max_length
            else self.text[:max_length] + "..."
        )
