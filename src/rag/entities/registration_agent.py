import json
from pathlib import Path
from typing import Any

from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..schemas.chunk import Chunk
from ..schemas.llm_responses.node_metadata import NodeMetadata


class RegistrationAgent:
    def __init__(
        self,
        generator: Generator,
        prompt_builder: PromptBuilder,
        storage_path: str | Path,
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder

        self.storage_path = Path(storage_path)
        self.storage_path.parent.mkdir(parents=True, exist_ok=True)

    def _create_node_id(self, doc_id: str) -> str:
        return f"node_{doc_id}"

    def _extract_node_metadata(self, chunks: list[Chunk]) -> NodeMetadata:
        system_prompt = self.prompt_builder.build_registration_system_prompt()
        user_prompt = self.prompt_builder.build_registration_user_prompt(chunks)
        return self.generator.parse(system_prompt, user_prompt, NodeMetadata)

    def _build_node(
        self, doc_id: str, node_metadata: NodeMetadata, name: str | None = None
    ) -> dict[str, Any]:
        return {
            "id": self._create_node_id(doc_id),
            "doc_id": doc_id,
            "name": name or doc_id,
            "domains": node_metadata.domains,
            "scopes": node_metadata.scopes,
            "description": node_metadata.description,
        }

    def register(
        self, doc_id: str, chunks: list[Chunk], name: str | None = None
    ) -> str:
        if not doc_id.strip():
            raise ValueError("doc_id must not be empty.")
        if not chunks:
            raise ValueError("chunks must not be empty.")

        node_metadata = self._extract_node_metadata(chunks)
        node = self._build_node(doc_id=doc_id, node_metadata=node_metadata, name=name)
        self._save_node_metadata(node)
        return doc_id

    def _load_existing_nodes(self) -> list[dict[str, Any]]:
        if not self.storage_path.exists():
            return []
        try:
            existing = json.loads(self.storage_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON in node metadata file: {self.storage_path}"
            ) from exc

        if not isinstance(existing, list):
            raise ValueError(
                f"Node metadata file must contain a JSON array: {self.storage_path}"
            )
        return existing

    def _save_node_metadata(self, node: dict[str, Any]) -> None:
        existing = self._load_existing_nodes()

        existing.append(node)
        self.storage_path.write_text(
            json.dumps(existing, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
