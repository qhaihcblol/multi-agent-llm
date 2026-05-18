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

    # Default RAG system prompt
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

    # Default RAG user prompt
    def build_user_prompt(self, question: str, context: str) -> str:
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

    # Default RAG build
    def build(
        self, question: str, chunks: list[RetrievedChunk]
    ) -> tuple[str, str, list[Citation]]:

        context = self.build_context(chunks)
        citations = self.build_citations(chunks)
        system_prompt = self.build_system_prompt()
        prompt = self.build_user_prompt(question, context)

        return system_prompt, prompt, citations

    # Registration system prompt
    def build_registration_system_prompt(self) -> str:
        return "\n".join(
            [
                "You are a retrieval metadata extraction system.",
                "",
                "Rules:",
                "- Use only information explicitly present in the text.",
                "- Do not infer or hallucinate missing information.",
                "- Be concise and precise.",
                "- Avoid generic or decorative wording.",
                "- If information is missing, return the most conservative valid output.",
            ]
        )

    # Registration user prompt
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
                "Extract metadata from the document below.",
                "",
                "Return the following fields:",
                "",
                "- domains:",
                "  High-level stable categories used for indexing.",
                "  Must be broad and reusable (e.g. technology, science, business, law, education, health).",
                "  Do not include specific topics or entities.",
                "",
                "- scopes:",
                "  Specific topical focus of the document.",
                "  Includes concrete subjects, named entities, events, systems, or time ranges when present.",
                "  This is document-specific and may vary per document.",
                "",
                "- description:",
                "  1–2 sentences describing the central idea or analytical insight.",
                "  Focus on meaning, argument, or intent rather than surface topic listing.",
                "  Must not repeat scopes or domains.",
                "",
                "Important separation rules:",
                "- domains = stable classification layer",
                "- scopes = document-specific focus layer",
                "- description = interpretive summary layer",
                "",
                "Document:",
                "",
                document_context,
            ]
        )

    # Stage 1 system prompt
    def build_create_point_system_prompt(self):
        return "\n".join(
            [
                "You are a point extraction system for retrieval-augmented generation.",
                "",
                "Rules:",
                "- Produce exactly one atomic point.",
                "- The point must be directly relevant to the question.",
                "- Keep the point concise, specific, and factual.",
                "- Prefer a focused paraphrase of the best supporting evidence over a broad summary.",
                "- Do not merge multiple claims into one point.",
                "- Use only information explicitly supported by the provided context.",
                "- If the context does not support a relevant point, return an empty text.",
                "- When text is empty, return no source indices.",
                "- Source indices must refer only to the numbered context blocks.",
            ]
        )

    def build_create_point_user_prompt(self, question: str, context: str) -> str:
        return "\n".join(
            [
                "Extract the single best point from the context that helps answer the question.",
                "",
                "Context:",
                context,
                "",
                "Question:",
                question,
                "",
                "Response requirements:",
                "- Return one concise point only.",
                "- If the context does not support a relevant point, set text to an empty string and source_indices to an empty list.",
                "- If text is not empty, include only the source indices that directly support it.",
                "- Use the smallest possible set of source indices.",
                "- Do not include sources that are only loosely related.",
                "- Favor the most relevant chunk over wider context when possible.",
            ]
        )
