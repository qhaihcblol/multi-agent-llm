from typing import Any

from langchain_text_splitters import RecursiveCharacterTextSplitter

from ..schemas import Chunk


class Chunker:
    def __init__(
        self,
        chunk_size: int = 600,
        chunk_overlap: int = 60,
        separators: list[str] | None = None,
        min_chunk_size: int = 200,
        max_merged_chunk_size: int | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")
        if chunk_overlap < 0 or chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be >= 0 and < chunk_size")
        if min_chunk_size < 0:
            raise ValueError("min_chunk_size must be >= 0")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.min_chunk_size = min_chunk_size
        self.max_merged_chunk_size = (
            max_merged_chunk_size
            if max_merged_chunk_size is not None
            else chunk_size + min_chunk_size
        )
        if self.max_merged_chunk_size <= 0:
            raise ValueError("max_merged_chunk_size must be > 0")

        self.separators = separators or [
            "\n\n",
            "\n",
            ". ",
            "? ",
            "! ",
            " ",
            "",
        ]

        self._splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.chunk_size,
            chunk_overlap=self.chunk_overlap,
            separators=self.separators,
            length_function=len,
            add_start_index=True,
        )

    def _normalize_bounds(self, start: int, end: int, source_text: str) -> tuple[int, int]:
        if start < 0 or end < 0:
            return -1, -1

        while start < end and source_text[start].isspace():
            start += 1
        while end > start and source_text[end - 1].isspace():
            end -= 1

        if (
            start > 0
            and start + 1 < end
            and source_text[start] in ".;,:"
            and source_text[start + 1] == " "
            and source_text[start - 1].isalnum()
        ):
            start += 2

        return start, end

    def _merge_short_chunks(
        self, raw_chunks: list[tuple[str, int, int]], source_text: str
    ) -> list[tuple[str, int, int]]:
        if self.min_chunk_size == 0:
            return raw_chunks

        merged: list[tuple[str, int, int]] = []

        for chunk_text, start, end in raw_chunks:
            if not merged:
                merged.append((chunk_text, start, end))
                continue

            if len(chunk_text) >= self.min_chunk_size:
                merged.append((chunk_text, start, end))
                continue

            prev_text, prev_start, prev_end = merged[-1]
            if prev_start >= 0 and prev_end >= 0 and start >= 0 and end >= 0:
                new_start = min(prev_start, start)
                new_end = max(prev_end, end)
                combined_text = source_text[new_start:new_end]
            else:
                separator = ""
                if not prev_text.endswith((" ", "\n")) and not chunk_text.startswith(
                    (" ", "\n")
                ):
                    separator = " "
                combined_text = prev_text + separator + chunk_text
                new_start = prev_start
                new_end = -1

            if len(combined_text) <= self.max_merged_chunk_size:
                merged[-1] = (combined_text, new_start, new_end)
            else:
                merged.append((chunk_text, start, end))

        return merged

    def split(
        self,
        text: str,
        doc_id: str,
        metadata: dict[str, Any] | None = None,
    ) -> list[Chunk]:
        metadata = metadata or {}
        docs = self._splitter.create_documents([text])

        raw_chunks: list[tuple[str, int, int]] = []
        for doc in docs:
            start = int(doc.metadata.get("start_index", -1))
            end = start + len(doc.page_content) if start >= 0 else -1
            start, end = self._normalize_bounds(start, end, text)
            if start >= 0 and end >= 0:
                chunk_text = text[start:end]
            else:
                chunk_text = doc.page_content
            raw_chunks.append((chunk_text, start, end))

        merged_chunks = self._merge_short_chunks(raw_chunks, text)

        chunks: list[Chunk] = []
        for i, (chunk_text, start, end) in enumerate(merged_chunks):
            chunks.append(
                Chunk(
                    id=f"{doc_id}_{i:05d}",
                    text=chunk_text,
                    metadata={
                        **metadata,
                        "doc_id": doc_id,
                        "chunk_index": i,
                        "start": start,
                        "end": end,
                        "length": len(chunk_text),
                    },
                )
            )

        return chunks
