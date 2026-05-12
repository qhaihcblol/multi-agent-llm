from ..schemas.retrieved_chunk import RetrievedChunk
from ..schemas.citation import Citation
from ..schemas.chunk import Chunk


class PromptBuilder:
    def build_citations(self, chunks: list[RetrievedChunk]) -> list[Citation]:
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

    def build_context(self, chunks: list[RetrievedChunk]) -> str:
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

    def build_system_prompt(self) -> str:
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

    def build_prompt(self, question: str, context: str) -> str:
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

        context = self.build_context(chunks)
        citations = self.build_citations(chunks)
        system_prompt = self.build_system_prompt()
        prompt = self.build_prompt(question, context)

        return system_prompt, prompt, citations

    def build_registration_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a retrieval metadata extraction system.",
                "",
                "Extract metadata from document excerpts.",
                "",
                "Rules:",
                "- Use only explicitly supported information.",
                "- Do not infer unsupported specialization.",
                "- Be concise and precise.",
                "- Prefer conservative outputs when information is limited.",
                "- Avoid vague or generic wording.",
                "- Output must follow the requested schema exactly.",
            ]
        )

    def build_registration_user_prompt(self, chunks: list[Chunk]) -> str:
        selected_chunks = chunks[:5]
        formatted_chunks: list[str] = []
        for index, chunk in enumerate(selected_chunks, start=1):
            text = chunk.text.strip()
            if not text:
                continue
            formatted_chunks.append(f"[Chunk {index}]\n{text}")
        document_context = "\n\n".join(formatted_chunks)
        return "\n".join(
            [
                "Extract the following fields from the document:",
                "",
                "- domain:",
                "High-level knowledge category using short stable lowercase labels.",
                "",
                "- scope:",
                "Specific knowledge coverage and specialization boundaries.",
                "",
                "- description:",
                "Concise semantic summary of the document knowledge.",
                "",
                "Document:",
                "",
                document_context,
            ]
        )
        
