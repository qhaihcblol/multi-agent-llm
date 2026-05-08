from abc import ABC, abstractmethod


class LLMError(RuntimeError):
    """Base error for all LLM client failures."""


class LLMConfigurationError(LLMError):
    """Raised when the client is configured incorrectly."""


class LLMGenerationError(LLMError):
    """Raised when text generation fails."""


class BaseLLMClient(ABC):
    @abstractmethod
    def generate(
        self,
        system_prompt: str,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate a text response from a prompt."""
