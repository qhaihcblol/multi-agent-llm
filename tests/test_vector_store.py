from src.rag.components.vector_store import VectorStore


def main():
    vs = VectorStore(
        collection_name="test_collection",
        persist_dir="./data/embeddings/chroma_db",
    )

    target_id = "doc2_c59b3bfa_00018"

    data = vs.collection.get(
        ids=[target_id],
        include=["documents", "metadatas", "embeddings"],
    )

    if not data["ids"]:
        print("Không tìm thấy chunk")
        return

    docs = data["documents"] or []
    metas = data["metadatas"] or []
    embs = data["embeddings"]

    if docs is None or metas is None or embs is None:
        print("Không có dữ liệu đầy đủ")
        return

    text = docs[0]
    meta = metas[0]
    emb = embs[0]

    print(f"ID: {target_id}")
    print(f"doc_id: {meta.get('doc_id')}, chunk: {meta.get('chunk_index')}")
    print(f"text: {text[:200].replace('\n', ' ')}...")

    # shape
    print(f"embedding dim: ({len(emb)})")


if __name__ == "__main__":
    main()
