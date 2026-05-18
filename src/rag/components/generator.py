from typing import TypeVar, Type
from pydantic import BaseModel
from ...llm.clients.base import BaseLLMClient

T = TypeVar("T", bound=BaseModel)


class Generator:

    def __init__(self, llm_client: BaseLLMClient) -> None:
        self.llm = llm_client

    def create(
        self,
        system_prompt: str,
        prompt: str,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        return self.llm.create(
            system_prompt=system_prompt,
            prompt=prompt,
            temperature=temperature,
            max_tokens=max_tokens,
        )

    def parse(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        return self.llm.parse(
            system_prompt=system_prompt,
            prompt=prompt,
            response_model=response_model,
            temperature=temperature,
            max_tokens=max_tokens,
        )
