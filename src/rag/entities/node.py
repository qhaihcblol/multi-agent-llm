from uuid import UUID
from ..components.retriever import Retriever


class Node:
    id: UUID
    name: str
    domain: str
    description: str
    scope: str
    
    
    def __init__(self, doc_id: str, retriever: Retriever):
        self.doc_id = doc_id
        self.retriever = retriever

    def retrieve(self, query: str, top_k: int = 5):
        return self.retriever.retrieve(
            query=query, top_k=top_k, where={"doc_id": self.doc_id}
        )
        