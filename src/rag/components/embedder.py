import numpy as np
from sentence_transformers import SentenceTransformer

from ..schemas.chunk import Chunk


class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def _encode(self, texts: str | list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        return self._encode(docs)

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self.embed_documents([chunk.text for chunk in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        return self._encode(query)

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        return self._encode(queries)
