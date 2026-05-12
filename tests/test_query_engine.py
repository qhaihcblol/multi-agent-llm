import os

from src.rag.components.embedder import Embedder
from src.rag.components.vector_store import VectorStore
from src.rag.components.retriever import Retriever

from src.rag.components.generator import Generator
from src.rag.components.prompt_builder import PromptBuilder
from src.llm.clients.openai import OpenAIClient
from src.llm.configs.openai import OpenAIConfig

from src.rag.orchestrator.query_engine import QueryEngine


def main():
    embedder = Embedder(device="cpu")
    vector_store = VectorStore(
        collection_name="test_collection",
        persist_dir="./data/embeddings/chroma_db",
    )
    # One
    retriever = Retriever(embedder=embedder, vector_store=vector_store)

    config = OpenAIConfig.from_env()
    llm_client = OpenAIClient(config=config)

    # Two
    generator = Generator(
        llm_client=llm_client, model_name="gpt-4o", temperature=0.2, max_tokens=1500
    )

    # Three
    prompt_builder = PromptBuilder()

    query_engine = QueryEngine(
        retriever=retriever,
        generator=generator,
        prompt_builder=prompt_builder,
    )

    question = "How did Betty Ford and Eleanor Roosevelt break the traditional boundaries of the role of First Lady to become influential figures in national debates?"
    # question = "What is Covid-1"
    result = query_engine.ask(question=question, top_k=10)

    print("\n" + "=" * 80)
    print("SYSTEM PROMPT")
    print("=" * 80)
    print(result.system_prompt)

    print("\n" + "=" * 80)
    print("PROMPT")
    print("=" * 80)
    print(result.prompt)

    print("\n" + "=" * 80)
    print("QUESTION")
    print("=" * 80)
    print(result.question)

    print("\n" + "=" * 80)
    print("ANSWER")
    print("=" * 80)
    print(result.answer)

    print("\n" + "=" * 80)
    print("CITATIONS")
    print("=" * 80)
    for i, citation in enumerate(result.citations, start=1):
        print(f"=== Citation {i} ===")
        print(f"Doc ID     : {citation.doc_id}")
        print(f"Chunk ID   : {citation.chunk_id}")
        print(f"Chunk Index: {citation.chunk_index}")
        print(f"Score      : {citation.score:.4f}")
        print(f"Metadata   : {citation.metadata}")
        print("Text:")
        print(citation.text)
        print()
        
    print("\n" + "=" * 80)
    print("QUERY COMPLETED")
    print("=" * 80)


if __name__ == "__main__":
    main()
