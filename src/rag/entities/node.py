from ..components.retriever import Retriever
from ..schemas.retrieved_chunk import RetrievedChunk


class Node:
    def __init__(
        self,
        id: str,
        domains: list[str],
        description: str | None,
        scopes: list[str],
        doc_id: str,
        retriever: Retriever,
        name: str | None = None,
    ) -> None:
        self.id = id
        self.domains = domains
        self.description = description
        self.scopes = scopes
        self.doc_id = doc_id
        self.retriever = retriever

        self.name = name or id

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievedChunk]:
        return self.retriever.retrieve(
            query=query, top_k=top_k, where={"doc_id": self.doc_id}
        )
