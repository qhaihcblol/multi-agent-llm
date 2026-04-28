from typing import Any

from .embedder import Embedder
from .vector_store import VectorStore


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder):
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(self, query: str, top_k: int = 5, where: dict[str, Any] | None = None):
        # Embedding query
        query_embedding = self.embedder.embed_query(query)
        # Search
        results = self.vector_store.search(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )
        return results
    
