from typing import ClassVar

import numpy as np
from sentence_transformers import SentenceTransformer
from ..schemas.chunk import Chunk


class Embedder:
    _instances: ClassVar[dict[str, "Embedder"]] = {}

    def __new__(
        cls,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ) -> "Embedder":
        key = f"{model_name}:{device}"
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._init(model_name, device, batch_size)
            cls._instances[key] = instance
        return cls._instances[key]

    def _init(self, model_name: str, device: str | None, batch_size: int) -> None:
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed_documents(self, docs: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            docs,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False,
        )
        return embeddings

    def embed_chunks(self, chunks: list[Chunk]) -> np.ndarray:
        return self.embed_documents([c.text for c in chunks])

    def embed_query(self, query: str) -> np.ndarray:
        embedding = self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )
        return embedding

    def embed_queries(self, queries: list[str]) -> np.ndarray:
        embeddings = self.model.encode(
            queries,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
        )
        return embeddings
