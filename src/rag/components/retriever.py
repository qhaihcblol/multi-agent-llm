from chromadb.api.types import Where

from .embedder import Embedder
from .vector_store import VectorStore
from ..schemas.retrieved_chunk import RetrievedChunk


class Retriever:
    def __init__(self, vector_store: VectorStore, embedder: Embedder) -> None:
        self.vector_store = vector_store
        self.embedder = embedder

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        where: Where | None = None,
    ) -> list[RetrievedChunk]:
        query_embedding = self.embedder.embed_query(query)

        result = self.vector_store.search(
            query_embeddings=query_embedding,
            n_results=top_k,
            where=where,
        )

        return self._to_retrieved_chunks(result)

    def _to_retrieved_chunks(self, result) -> list[RetrievedChunk]:
        ids = result.get("ids", [[]])[0]
        documents = result.get("documents", [[]])[0]
        metadatas = result.get("metadatas", [[]])[0]
        distances = result.get("distances", [[]])[0]

        chunks: list[RetrievedChunk] = []

        for i in range(len(ids)):
            chunks.append(
                RetrievedChunk(
                    id=ids[i],
                    text=documents[i],
                    metadata=metadatas[i] or {},
                    score=1 - distances[i] if distances else 0.0,
                )
            )

        # Sort chunks by score in descending order
        chunks.sort(key=lambda chunk: chunk.score, reverse=True)

        return chunks
