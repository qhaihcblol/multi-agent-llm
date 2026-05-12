import json
import uuid
from pathlib import Path

from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..components.vector_store import VectorStore
from ..entities.registration_agent import RegistrationAgent
from ..schemas.chunk import Chunk


class Ingestor:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        registration_agent: RegistrationAgent | None = None,
        generator: Generator | None = None,
        prompt_builder: PromptBuilder | None = None,
        storage_path: str | Path | None = None,
        chunks_storage_dir: str | Path = "data/chunks",
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.registration_agent = registration_agent or self._build_registration_agent(
            generator=generator,
            prompt_builder=prompt_builder,
            storage_path=storage_path,
        )
        self.chunks_storage_dir = Path(chunks_storage_dir)
        self.chunks_storage_dir.mkdir(parents=True, exist_ok=True)

    def _build_registration_agent(
        self,
        generator: Generator | None,
        prompt_builder: PromptBuilder | None,
        storage_path: str | Path | None,
    ) -> RegistrationAgent:
        if generator is None or prompt_builder is None or storage_path is None:
            raise ValueError(
                "Either registration_agent must be provided, or generator, "
                "prompt_builder, and storage_path must all be set."
            )
        return RegistrationAgent(
            generator=generator,
            prompt_builder=prompt_builder,
            storage_path=storage_path,
        )

    def _load_document(self, doc_path: str) -> str:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {doc_path}")
        return path.read_text(encoding="utf-8")

    def _create_doc_id(self, doc_path: str) -> str:
        return f"{Path(doc_path).stem}_{uuid.uuid4().hex[:8]}"

    def ingest(self, doc_path: str, name: str | None = None) -> str:
        text = self._load_document(doc_path)
        doc_id = self._create_doc_id(doc_path)
        chunks = self.chunker.split(text=text, doc_id=doc_id)
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)
        self._save_chunks(chunks, doc_id)
        
        self.registration_agent.register(
            doc_id=doc_id,
            chunks=chunks,
            name=name or Path(doc_path).stem,
        )
        return doc_id

    def _save_chunks(self, chunks: list[Chunk], doc_id: str) -> None:
        chunk_data = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        output_path = self.chunks_storage_dir / f"{doc_id}_chunks.json"
        output_path.write_text(
            json.dumps(chunk_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
