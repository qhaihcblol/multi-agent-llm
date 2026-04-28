from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    _instances = {}

    def __new__(cls, model_name="all-MiniLM-L6-v2", device=None, batch_size=64):
        key = f"{model_name}:{device}"
        if key not in cls._instances:
            instance = super().__new__(cls)
            instance._init(model_name, device, batch_size)
            cls._instances[key] = instance
        return cls._instances[key]

    def _init(self, model_name, device, batch_size):
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
