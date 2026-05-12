from abc import ABC, abstractmethod
from typing import TypeVar, Type
from pydantic import BaseModel

T = TypeVar("T", bound=BaseModel)


class LLMError(RuntimeError):
    """Base error for all LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when the client is configured incorrectly."""


class LLMGenerationError(LLMError):
    """Raised when text generation fails."""


class BaseLLMClient(ABC):

    @abstractmethod
    def create(
        self,
        system_prompt: str,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a raw text response."""

    @abstractmethod
    def parse(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        """Generate a structured response validated against response_model."""
