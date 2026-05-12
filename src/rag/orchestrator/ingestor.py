import json
import uuid
from pathlib import Path

from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..components.vector_store import VectorStore
from ..schemas.chunk import Chunk
from ..schemas.llm_responses.node_metadata import NodeMetadata


class Ingestor:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        prompt_builder: PromptBuilder,
        storage_path: str | Path,
        chunks_storage_dir: str | Path = "data/chunks",
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.prompt_builder = prompt_builder

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunks_storage_dir = Path(chunks_storage_dir)
        self.chunks_storage_dir.mkdir(parents=True, exist_ok=True)

    def _load_document(self, doc_path: str) -> str:
        path = Path(doc_path)
        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        if not path.is_file():
            raise ValueError(f"Not a file: {doc_path}")
        return path.read_text(encoding="utf-8")

    def _create_doc_id(self, doc_path: str) -> str:
        return f"{Path(doc_path).stem}_{uuid.uuid4().hex[:8]}"

    def _create_node_id(self, doc_id: str) -> str:
        return f"node_{doc_id}"

    def ingest(self, doc_path: str, name: str | None = None) -> str:
        text = self._load_document(doc_path)
        doc_id = self._create_doc_id(doc_path)
        chunks = self.chunker.split(text=text, doc_id=doc_id)
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)
        self._save_chunks(chunks, doc_id)

        system_prompt = self.prompt_builder.build_registration_system_prompt()
        user_prompt = self.prompt_builder.build_registration_user_prompt(chunks)
        node_metadata = self.generator.parse(system_prompt, user_prompt, NodeMetadata)

        node = {
            "id": self._create_node_id(doc_id),
            "doc_id": doc_id,
            "name": name or Path(doc_path).stem,
            "domains": node_metadata.domains,
            "scopes": node_metadata.scopes,
            "description": node_metadata.description,
        }
        self._save_node_metadata(node)
        return doc_id

    def _save_node_metadata(self, node: dict) -> None:
        if self.storage_path.exists():
            try:
                existing = json.loads(self.storage_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError:
                existing = []
        else:
            existing = []

        existing.append(node)
        self.storage_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

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
