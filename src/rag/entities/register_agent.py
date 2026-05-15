import json
import uuid
from pathlib import Path
from typing import Any

from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..components.retriever import Retriever
from ..components.vector_store import VectorStore
from ..schemas.chunk import Chunk
from ..schemas.llm_responses.node_metadata import NodeMetadata
from .node import Node


class RegisterAgent:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.retriever = Retriever(vector_store=vector_store, embedder=embedder)

    def _load_document(self, doc_path: Path) -> str:
        if not doc_path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")
        if not doc_path.is_file():
            raise ValueError(f"Not a file: {doc_path}")
        return doc_path.read_text(encoding="utf-8")

    def _create_doc_id(self, doc_path: Path) -> str:
        return f"{doc_path.stem}_{uuid.uuid4().hex[:8]}"

    def _save_chunks(
        self, chunks: list[Chunk], doc_id: str, chunks_storage_dir: Path
    ) -> None:
        chunk_data = [
            {
                "id": chunk.id,
                "text": chunk.text,
                "metadata": chunk.metadata,
            }
            for chunk in chunks
        ]
        chunks_storage_dir.mkdir(parents=True, exist_ok=True)
        output_path = chunks_storage_dir / f"{doc_id}_chunks.json"

        output_path.write_text(
            json.dumps(chunk_data, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    # Ingest the document
    def ingest(
        self, doc_path: Path, chunks_storage_dir: Path
    ) -> tuple[str, list[Chunk]]:
        text = self._load_document(doc_path)
        doc_id = self._create_doc_id(doc_path)
        chunks = self.chunker.split(text=text, doc_id=doc_id)
        embeddings = self.embedder.embed_chunks(chunks)
        self.vector_store.add_chunks(chunks, embeddings)
        self._save_chunks(chunks, doc_id, chunks_storage_dir)
        return doc_id, chunks

    def _create_node_id(self, doc_id: str) -> str:
        return f"node_{doc_id}"

    def _extract_node_metadata(self, chunks: list[Chunk]) -> NodeMetadata:
        system_prompt = self.prompt_builder.build_registration_system_prompt()
        user_prompt = self.prompt_builder.build_registration_user_prompt(chunks)
        return self.generator.parse(system_prompt, user_prompt, NodeMetadata)

    def register(
        self,
        doc_id: str,
        chunks: list[Chunk],
        storage_path: Path,
        name: str | None = None,
    ) -> Node:
        if not doc_id.strip():
            raise ValueError("doc_id must not be empty.")
        if not chunks:
            raise ValueError("chunks must not be empty.")

        node_id = self._create_node_id(doc_id)
        node_metadata = self._extract_node_metadata(chunks)
        node = Node.from_metadata(
            metadata=node_metadata,
            id=node_id,
            doc_id=doc_id,
            retriever=self.retriever,
            name=name,
        )
        self._save_node_metadata(node.to_dict(), storage_path)
        return node

    def _load_existing_nodes(self, storage_path: Path) -> list[dict[str, Any]]:
        if not storage_path.exists():
            return []
        try:
            existing = json.loads(storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in node metadata file: {storage_path}"
            ) from exc

        if not isinstance(existing, list):
            raise ValueError(
                f"Node metadata file must contain a JSON array: {storage_path}"
            )
        return existing

    def _save_node_metadata(self, node: dict[str, Any], storage_path: Path) -> None:
        existing = self._load_existing_nodes(storage_path)

        existing.append(node)
        storage_path.parent.mkdir(parents=True, exist_ok=True)
        storage_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
