import uuid
import numpy as np

from ..components.embedder import Embedder
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..schemas.citation import Citation
from ..schemas.llm_responses.point_response import PointResponse
from ..schemas.point import Point
from ..schemas.retrieved_chunk import RetrievedChunk
from .node import Node


class PointExtractorAgent:
    def __init__(
        self,
        generator: Generator,
        prompt_builder: PromptBuilder,
        embedder: Embedder,
        retrieval_threshold: float = 0.5,
        grounding_threshold: float = 0.7,
    ) -> None:
        self.generator = generator
        self.prompt_builder = prompt_builder
        self.embedder = embedder

        self.retrieval_threshold = retrieval_threshold
        self.grounding_threshold = grounding_threshold

    def extract_points(
        self,
        nodes: list[Node],
        question: str,
        top_k: int = 5,
    ) -> list[Point]:
        points: list[Point] = []

        system_prompt = self.prompt_builder.build_create_point_system_prompt()

        for node in nodes:
            retrieved_chunks = node.retrieve(
                query=question,
                top_k=top_k,
            )

            retrieved_chunks = self._filter_retrieved_chunks(retrieved_chunks)

            if not retrieved_chunks:
                continue

            context = self.prompt_builder.build_context(retrieved_chunks)

            user_prompt = self.prompt_builder.build_create_point_user_prompt(
                question=question,
                context=context,
            )

            response = self.generator.parse(
                system_prompt,
                user_prompt,
                PointResponse,
            )

            point_text = response.text.strip()

            citations = self.prompt_builder.build_citations(retrieved_chunks)

            support_citations = self._select_support_citations(
                citations,
                response.source_indices,
            )

            grounding_score = self._compute_grounding_score(
                point_text,
                support_citations,
            )

            abstain = (
                response.abstain
                or not point_text
                or not support_citations
                or grounding_score < self.grounding_threshold
            )

            points.append(
                Point(
                    id=f"point_{uuid.uuid4()}_{node.id}",
                    node_id=node.id,
                    text=point_text,
                    citations=support_citations,
                    grounding_score=grounding_score,
                    abstain=abstain,
                )
            )

        return points

    def _filter_retrieved_chunks(
        self,
        chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        return [chunk for chunk in chunks if chunk.score >= self.retrieval_threshold]

    def _compute_grounding_score(
        self,
        point_text: str,
        citations: list[Citation],
    ) -> float:
        if not point_text.strip():
            return 0.0

        citation_text = "\n".join(
            citation.text.strip() for citation in citations if citation.text.strip()
        )

        if not citation_text:
            return 0.0

        point_emb = self.embedder.embed_query(point_text)

        citation_emb = self.embedder.embed_query(citation_text)

        return float(np.dot(point_emb, citation_emb))

    @staticmethod
    def _select_support_citations(
        citations: list[Citation],
        source_indices: list[int],
    ) -> list[Citation]:
        selected_citations: list[Citation] = []
        seen_indices: set[int] = set()

        for index in source_indices:
            if index in seen_indices:
                continue

            if not 1 <= index <= len(citations):
                continue

            seen_indices.add(index)

            selected_citations.append(citations[index - 1])

        return selected_citations
