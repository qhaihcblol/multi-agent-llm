from ..schemas.retrieved_chunk import RetrievedChunk
from ..schemas.citation import Citation


class PromptBuilder:
    def _build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
        citations: list[Citation] = []
        for chunk in chunks:
            metadata = chunk.metadata or {}
            citations.append(
                Citation(
                    doc_id=str(metadata.get("doc_id", "unknown")),
                    chunk_id=chunk.id,
                    chunk_index=int(metadata.get("chunk_index", -1)),
                    text=chunk.text,
                    score=chunk.score,
                    metadata=metadata,
                )
            )
        return citations

    def _build_context(self, chunks: list[RetrievedChunk]) -> str:
        if not chunks:
            return "No context provided."

        blocks = []
        for i, chunk in enumerate(chunks, start=1):
            metadata = chunk.metadata or {}
            doc_id = metadata.get("doc_id", "unknown")
            chunk_index = metadata.get("chunk_index", "unknown")
            block = f"[{i}] (doc: {doc_id}, chunk: {chunk_index})\n" f"{chunk.text}"
            blocks.append(block)

        return "\n\n---\n\n".join(blocks)

    def _build_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a reliable AI assistant.",
                "",
                "Rules:",
                "- Answer ONLY using the provided context",
                "- Cite sources using [number]",
                "- Do NOT hallucinate",
                '- If unsure, say "I don\'t know"',
            ]
        )

    def _build_prompt(self, question: str, context: str) -> str:
        return "\n".join(
            [
                "Context:",
                context,
                "",
                "Question:",
                question,
                "",
                "Answer:",
            ]
        )

    def build(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, str, list[Citation]]:
        chunks = sorted(chunks, key=lambda x: x.score, reverse=True)

        context = self._build_context(chunks)
        citations = self._build_citations(chunks)
        system_prompt = self._build_system_prompt()
        prompt = self._build_prompt(question, context)

        return system_prompt, prompt, citations
