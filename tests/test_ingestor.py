from src.rag.components.chunker import Chunker
from src.rag.components.embedder import Embedder
from src.rag.components.vector_store import VectorStore
from src.rag.components.generator import Generator
from src.rag.components.prompt_builder import PromptBuilder

from src.llm.clients.openai import OpenAIClient
from src.llm.configs.openai import OpenAIConfig

from src.rag.orchestrator.ingestor import Ingestor


def main():

    chunker = Chunker(
        chunk_size=600,
        chunk_overlap=60,
    )

    embedder = Embedder(model_name="all-MiniLM-L6-v2", device="cpu")

    vector_store = VectorStore(
        collection_name="test_collection", persist_dir="./data/embeddings/chroma_db"
    )
    config = OpenAIConfig.from_env()
    llm_client = OpenAIClient(config=config)

    # Two
    generator = Generator(
        llm_client=llm_client, model_name="gpt-4o", temperature=0.2, max_tokens=1500
    )
    prompt_builder = PromptBuilder()

    # 2. init ingestor
    ingestor = Ingestor(
        chunker=chunker,
        embedder=embedder,
        vector_store=vector_store,
        generator=generator,
        prompt_builder=prompt_builder,
        storage_path="./data/node_metadata.json",
    )

    # 3. ingest file
    doc_path = "data/processed_docs/doc1.txt"
    doc_id = ingestor.ingest(doc_path=doc_path, name=None)
    print(f"Document ingested with doc_id: {doc_id}")
    
    doc_path = "data/processed_docs/doc2.txt"
    doc_id = ingestor.ingest(doc_path=doc_path, name=None)
    print(f"Document ingested with doc_id: {doc_id}")
    
    print("Ingestion done!")


if __name__ == "__main__":
    main()
