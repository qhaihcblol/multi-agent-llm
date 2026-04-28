from typing import Any, List

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..schemas.chunk import Chunk


class Chunker:
    def __init__(
        self,
        chunk_size: int = 400,
        chunk_overlap: int = 80,
        separators: list[str] | None = None,
    ) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.separators = separators or ["\n\n", "\n", ". ", ", ", " ", ""]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
        )

    def split(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> List[Chunk]:
        """
        Split a document into Chunk objects.
        """
        metadata = metadata or {}

        raw_chunks = self._splitter.split_text(text)

        chunks: list[Chunk] = []

        for i, chunk_text in enumerate(raw_chunks):
            chunk_id = f"{doc_id}_{i:05d}"

            chunks.append(
                Chunk(
                    id=chunk_id,
                    text=chunk_text.strip(),
                    metadata={
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_index": i,
                    },
                )
            )

        return chunks
