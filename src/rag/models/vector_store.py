import chromadb
import numpy as np
from chromadb.api.types import (
    OneOrMany,
    Metadata,
    ID,
    Embedding,
    Document,
    Where,
)
from chromadb.config import Settings


class VectorStore:
    def __init__(
        self,
        collection_name: str = "rag_store",
        persist_dir: str = "./chroma_db",
        distance_metric: str = "cosine",
    ):
        self.client = chromadb.PersistentClient(
            path=persist_dir,
            settings=Settings(anonymized_telemetry=False),
        )
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            metadata={"hnsw:space": distance_metric},
        )

    def add(
        self,
        ids: OneOrMany[ID],
        embeddings: OneOrMany[Embedding],
        metadatas: OneOrMany[Metadata] | None = None,
        documents: OneOrMany[Document] | None = None,
    ) -> None:

        if isinstance(embeddings, np.ndarray) and embeddings.ndim == 1:
            embeddings = [embeddings]

        self.collection.add(
            ids=ids,
            embeddings=embeddings,
            metadatas=metadatas,
            documents=documents,
        )

    def search(
        self,
        query_embeddings: OneOrMany[Embedding],
        n_results: int = 5,
        where: Where | None = None,
        
    ):
        if isinstance(query_embeddings, np.ndarray) and query_embeddings.ndim == 1:
            query_embeddings = [query_embeddings]


        return self.collection.query(
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
        )
