import chromadb
import numpy as np
from chromadb.config import Settings
from typing import Any


class VectorStore:
    def __init__(
        self, collection_name: str = "rag_store", persist_dir: str = "./chroma_db"
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": "cosine"},
        )

    def add(
        self,
        embeddings: np.ndarray,
        metadatas: list[dict],
        ids: list[str] | None = None,
    ) -> None:
        if ids is None:
            ids = [str(i) for i in range(len(embeddings))]
        self.collection.add(
            ids=ids,
            embeddings=embeddings.tolist(),
            metadatas=metadatas, 
        )
