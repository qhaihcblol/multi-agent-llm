import chromadb
import numpy as np
from embedder import Embedder


class VectorStore:
    def __init__(self, collection_name: str, persist_dir: str = "./chroma_db"):
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(collection_name)
        self.embedder = Embedder()

    def add(self, documents: list[str], ids: list[str], metadatas: list[dict] = None):
        embeddings = self.embedder.embed_documents(documents).tolist()
        self.collection.add(
            ids=ids,
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas or [{} for _ in documents],
        )

    def query(self, query: str, top_k: int = 5) -> list[dict]:
        embedding = self.embedder.embed_query(query).tolist()
        results = self.collection.query(query_embeddings=[embedding], n_results=top_k)
        return [
            {"id": id, "document": doc, "score": 1 - distance}
            for id, doc, distance in zip(
                results["ids"][0],
                results["documents"][0],
                results["distances"][0],
            )
        ]
