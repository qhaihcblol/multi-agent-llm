from ..components.retriever import Retriever


class Node:
    def __init__(
        self,
        id: str,
        domains: str,
        description: str,
        scopes: str,
        doc_id: str,
        retriever: Retriever,
        name: str | None = None,
    ):
        self.id = id
        self.domains = domains
        self.description = description
        self.scopes = scopes
        self.doc_id = doc_id
        self.retriever = retriever

        self.name = name or id

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.retrieve(
            query=query, top_k=top_k, where={"doc_id": self.doc_id}
        )
