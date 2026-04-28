from pathlib import Path

from src.rag.models.chunker import Chunker
from src.rag.models.embedder import Embedder
from src.rag.models.vector_store import VectorStore
from src.rag.models.retriever import Retriever


def main():
    doc_path = Path("data/raw_docs/doc1.txt")
    text = doc_path.read_text(encoding="utf-8")

    # 1. Chunk
    chunker = Chunker(chunk_size=600, chunk_overlap=80)
    chunks = chunker.split(
        text=text,
        doc_id="doc1",
        metadata={"source": str(doc_path)},
    )
    print(f"Total chunks: {len(chunks)}")

    # 2. Embed
    embedder = Embedder(device="cpu")
    embeddings = embedder.embed_documents([chunk.text for chunk in chunks])

    # 3. Store
    vector_store = VectorStore(collection_name="doc1_demo", )
    vector_store.add(
        ids=[chunk.id for chunk in chunks],
        embeddings=embeddings,
        metadatas=[chunk.metadata for chunk in chunks],
        documents=[chunk.text for chunk in chunks],
    )

    # 4. Retrieve
    retriever = Retriever(vector_store=vector_store, embedder=embedder)
    results = retriever.retrieve("Why was Betty Ford controversial?", top_k=3)

    print("\nTop results:")
    for i, (doc, meta, dist) in enumerate(
        zip(
            results["documents"][0],
            results["metadatas"][0],
            results["distances"][0],
        ),
        start=1,
    ):
        print(f"\nResult {i}")
        print("Distance:", dist)
        print("Metadata:", meta)
        print("Text:", doc)


if __name__ == "__main__":
    main()
