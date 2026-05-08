from chromadb.api.types import Where

from ..components.retriever import Retriever
from ..components.generator import Generator
from ..components.prompt_builder import PromptBuilder
from ..schemas.query_result import QueryResult


class QueryEngine:
    def __init__(
        self,
        retriever: Retriever,
        generator: Generator,
        prompt_builder: PromptBuilder,
    ) -> None:
        self.retriever = retriever
        self.generator = generator
        self.prompt_builder = prompt_builder

    def query(
        self,
        question: str,
        top_k: int = 5,
        where: Where | None = None,
    ) -> QueryResult:
        question = question.strip()
        if not question:
            raise ValueError("question must not be empty.")
        if top_k <= 0:
            raise ValueError("top_k must be > 0.")

        chunks = self.retriever.retrieve(query=question, top_k=top_k, where=where)
        system_prompt, prompt, citations = self.prompt_builder.build(question, chunks)
        answer = self.generator.generate(system_prompt=system_prompt, prompt=prompt)

        return QueryResult(
            question=question,
            answer=answer,
            citations=citations,
            chunks=chunks,
            system_prompt=system_prompt,
            prompt=prompt,
        )

    def ask(
        self,
        question: str,
        top_k: int = 5,
        where: Where | None = None,
    ) -> QueryResult:
        return self.query(question=question, top_k=top_k, where=where)
