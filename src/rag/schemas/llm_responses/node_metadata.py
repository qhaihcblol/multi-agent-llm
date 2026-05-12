from pydantic import BaseModel, Field


class NodeMetadata(BaseModel):
    domains: list[str] = Field(default_factory=list)
    scopes: list[str] = Field(default_factory=list)
    description: str | None = None
