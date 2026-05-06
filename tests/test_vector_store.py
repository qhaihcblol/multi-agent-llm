import random
from src.rag.components.vector_store import VectorStore
from src.rag.schemas.chunk import Chunk


def main():
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_dir="./data/embeddings/chroma_db",
    )

    data = vector_store.collection.get(include=["documents", "metadatas"])

    ids = data.get("ids", [])
    docs = data.get("documents", [])
    metas = data.get("metadatas", [])

    if not ids:
        print("Không tìm thấy chunk")
        return

    if docs is None or metas is None:
        print("Không có dữ liệu đầy đủ")
        return

    chunks = [
        Chunk(id=cid, text=doc, metadata=meta or {}) # type: ignore
        for cid, doc, meta in zip(ids, docs, metas)
    ]

    sample_size = min(5, len(chunks))
    sampled_chunks = random.sample(chunks, sample_size)

    print(f"Total chunks: {len(chunks)} | Sampling: {sample_size}\n")

    for i, chunk in enumerate(sampled_chunks, 1):
        print(f"--- Chunk {i} ---")
        print(f"id: {chunk.id}")
        print(f"doc_id: {chunk.metadata.get('doc_id')}")
        print(f"chunk_index: {chunk.metadata.get('chunk_index')}")
        print(f"length: {len(chunk)}")
        print(f"text: {chunk.preview(200).replace('\\n', ' ')}")
        print()


if __name__ == "__main__":
    main()
