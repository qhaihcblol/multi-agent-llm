from ..components.retriever import Retriever
from ..schemas.retrieved_chunk import RetrievedChunk
from ..schemas.llm_responses.node_metadata import NodeMetadata

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

    @classmethod
    def from_metadata(
        cls,
        metadata: NodeMetadata,
        id: str,
        doc_id: str,
        retriever: Retriever,
        name: str | None = None,
    ) -> "Node":
        return cls(
            id=id,
            doc_id=doc_id,
            retriever=retriever,
            name=name,
            domains=metadata.domains,
            scopes=metadata.scopes,
            description=metadata.description,
        )
