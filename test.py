import json
from pathlib import Path
from src.rag.models import Chunker


def main():
    file_path = Path("data/raw_docs/doc1.txt")
    output_path = Path("data/chunks/doc1_chunks.json")

    # đảm bảo folder tồn tại
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # đọc file
    text = file_path.read_text(encoding="utf-8")

    # init chunker
    chunker = Chunker(chunk_size=600, chunk_overlap=80)

    # split
    chunks = chunker.split(
        text=text,
        doc_id="doc1",
        metadata={"source": str(file_path)},
    )

    # convert sang dict để lưu JSON
    chunks_data = [
        {
            "id": c.id,
            "text": c.text,
            "metadata": c.metadata,
        }
        for c in chunks
    ]

    # save
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(chunks_data, f, ensure_ascii=False, indent=2)

    print(f"Saved {len(chunks)} chunks to {output_path}")

    # debug thử vài chunk đầu
    for c in chunks[:3]:
        print("=" * 50)
        print(c.id)
        print(c.metadata)
        print(c.preview())


if __name__ == "__main__":
    main()
