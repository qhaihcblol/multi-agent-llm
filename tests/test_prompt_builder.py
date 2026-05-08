from src.rag.schemas.retrieved_chunk import RetrievedChunk
from src.rag.components.prompt_builder import PromptBuilder
from src.rag.schemas.citation import Citation


def main():
    builder = PromptBuilder()
    # Mock retrieved chunks
    chunks = [
        RetrievedChunk(
            id="c1",
            text="GAN is widely used for image generation.",
            score=0.92,
            metadata={"doc_id": "paper1", "chunk_index": 1},
        ),
        RetrievedChunk(
            id="c2",
            text="Diffusion models achieve higher quality but slower inference.",
            score=0.95,
            metadata={"doc_id": "paper2", "chunk_index": 3},
        ),
        RetrievedChunk(
            id="c3",
            text="Face swapping can be done using encoder-decoder architectures.",
            score=0.89,
            metadata={"doc_id": "blog1", "chunk_index": 2},
        ),
    ]

    question = "What methods are used for face generation?"
    system_prompt, prompt, citations = builder.build(question, chunks)

    print("===== SYSTEM PROMPT =====\n")
    print(system_prompt)
    print("===== GENERATED PROMPT =====\n")
    print(prompt)

    print("\n===== CITATIONS =====\n")
    for i, citation in enumerate(citations, start=1):
        print(f"[{i}]")
        print(f"  doc_id      : {citation.doc_id}")
        print(f"  chunk_index : {citation.chunk_index}")
        print(f"  score       : {citation.score:.4f}")
        print(f"  text        : {citation.text[:100]}...")
        print("-" * 40)


if __name__ == "__main__":
    main()
