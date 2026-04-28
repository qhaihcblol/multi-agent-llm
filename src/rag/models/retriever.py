from chromadb.api.types import QueryResult, Where

from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Where | None = None,
    ) -> QueryResult:
        query_embedding = self.embedder.embed_query(query)
        return self.vector_store.search(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )
