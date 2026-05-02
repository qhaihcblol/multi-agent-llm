from src.rag.components.chunker import Chunker
from src.rag.components.embedder import Embedder
from src.rag.components.vector_store import VectorStore
from src.rag.orchestrator.ingestor import Ingestor


def main():
    
    chunker = Chunker(
        chunk_size=600,
        chunk_overlap=60,
    )

    embedder = Embedder(model_name="all-MiniLM-L6-v2", device="cpu")

    vector_store = VectorStore(
        collection_name="test_collection_2", persist_dir="./data/embeddings/chroma_db"
    )

    # 2. init ingestor
    ingestor = Ingestor(chunker=chunker, embedder=embedder, vector_store=vector_store)

    # 3. ingest file
    doc_path = "data/processed_docs/doc2.txt"
    ingestor.ingest(doc_path)

    print("Ingestion done!")


if __name__ == "__main__":
    main()
