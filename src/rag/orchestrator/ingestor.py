from ..components.chunker import Chunker
from ..components.embedder import Embedder
from ..components.vector_store import VectorStore

from pathlib import Path
import uuid


class Ingestor:
    def __init__(self, chunker: Chunker, embedder: Embedder, vector_store: VectorStore):
        self.chunker = chunker
        self.embedder = embedder
        self.vector_store = vector_store

    def _load_document(self, doc_path: str) -> str:
        path = Path(doc_path)

        if not path.exists():
            raise FileNotFoundError(f"Document not found: {doc_path}")

        if not path.is_file():
            raise ValueError(f"Not a file: {doc_path}")

        return path.read_text(encoding="utf-8")

    def _create_doc_id(self, doc_path: str) -> str:
        return f"{Path(doc_path).stem}_{uuid.uuid4().hex[:8]}"

    def ingest(self, doc_path: str):
        text = self._load_document(doc_path)
        doc_id = self._create_doc_id(doc_path)

        chunks = self.chunker.split(text=text, doc_id=doc_id)
        embeddings = self.embedder.embed_chunks(chunks)
    
        self.vector_store.add_chunks(chunks, embeddings)
