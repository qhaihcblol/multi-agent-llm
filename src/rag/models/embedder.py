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


if __name__ == "__main__":
    embedder = Embedder(device="cpu")

    # Test single query
    q_emb = embedder.embed_query("hello world")
    print("Query embedding dtype:", q_emb.dtype)
    print("Query embedding shape:", q_emb.shape)

    # Test batch documents
    docs = ["hello world", "multi agent rag", "embedding test"]
    d_emb = embedder.embed_documents(docs)
    print("Docs embedding dtype:", d_emb.dtype)
    print("Docs embedding shape:", d_emb.shape)

    # Test multiple queries
    queries = ["what is rag?", "how embedding works?"]
    qs_emb = embedder.embed_queries(queries)
    print("Queries embedding dtype:", qs_emb.dtype)
    print("Queries embedding shape:", qs_emb.shape)
