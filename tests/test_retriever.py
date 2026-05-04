from src.rag.components.vector_store import VectorStore
from src.rag.components.retriever import Retriever
from src.rag.components.embedder import Embedder

def main():
    embedder = Embedder(device="cpu")
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_dir="./data/embeddings/chroma_db",
    )
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    query = "How did Betty Ford and Eleanor Roosevelt break the traditional boundaries of the role of First Lady to become influential figures in national debates?"

    results = retriever.retrieve(query, top_k=3)
    print("Query:", query)

    if not results:
        print("No relevant documents found.")
        return
    for i, chunk in enumerate(results, 1):
        print(f"--- Result {i} ---")
        print(f"ID: {chunk.id}")
        print(f"Score: {chunk.score:.4f}")
        print(f"Text: {chunk.text}")
        print(f"Metadata: {chunk.metadata}")
        print()
if __name__ == "__main__":
    main()
