from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.vector_store import VectorStore
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from pathlib import Path
import uuid


class Ingestor:
    def __init__(
        self,
        chunker: Chunker,
        embedder: Embedder,
        vector_store: VectorStore,
        generator: Generator,
        prompt_builder: PromptBuilder,
        storage_path: str | Path,
    ):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store
        self.generator = generator
        self.prompt_builder = prompt_builder

        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)

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

    def _register_node(self):
        # Placeholder for node registration logic
        pass

    def ingest(self, doc_path: str):
        text = self._load_document(doc_path)
        doc_id = self._create_doc_id(doc_path)

        chunks = self.chunker.split(text=text, doc_id=doc_id)
        embeddings = self.embedder.embed_chunks(chunks)

        self.vector_store.add_chunks(chunks, embeddings)
