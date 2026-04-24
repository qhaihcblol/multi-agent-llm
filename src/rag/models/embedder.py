from sentence_transformers import SentenceTransformer
import numpy as np


class Embedder:
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: str | None = None,
        batch_size: int = 64,
    ):
        self.model = SentenceTransformer(model_name, device=device)
        self.batch_size = batch_size

    def embed_documents(self, texts: list[str]) -> np.ndarray:
        return self.model.encode(
            texts,
            normalize_embeddings=True,
            convert_to_numpy=True,
            batch_size=self.batch_size,
            show_progress_bar=False
        )

    def embed_query(self, query: str) -> np.ndarray:
        return self.model.encode(
            query, normalize_embeddings=True, convert_to_numpy=True
        )

# if __name__ == "__main__":
#     embedder = Embedder(model_name="all-MiniLM-L6-v2", device="cpu",batch_size=64)
#     documents = [
#         "The cat is on the roof.",
#         "The dog is in the garden.",
#         "The bird is flying in the sky."
#     ]
#     query = "Where is the cat?"
#     doc_embeddings = embedder.embed_documents(documents)
#     query_embedding = embedder.embed_query(query)
#     print("Document Embeddings:", doc_embeddings)
#     print("Query Embedding:", query_embedding)