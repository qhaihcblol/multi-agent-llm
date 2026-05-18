from typing import TypeVar, Type
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from .base import BaseLLMClient, LLMConfigurationError, LLMGenerationError
from ..configs.openai import OpenAIConfig

T = TypeVar("T", bound=BaseModel)


class OpenAIClient(BaseLLMClient):
    def __init__(self, config: OpenAIConfig) -> None:
        if not config.api_key:
            raise LLMConfigurationError("OpenAIConfig.api_key must not be empty.")

        self.default_model = config.model
        self.default_temperature = config.temperature
        self.default_max_tokens = config.max_tokens

        self.client = OpenAI(
            api_key=config.api_key,
            base_url=config.base_url,
            timeout=config.timeout,
        )

    def _resolve_params(
        self,
        model: str | None,
        temperature: float | None,
        max_tokens: int | None,
    ) -> tuple[str, float, int]:
        """Merge call-level overrides with instance defaults."""
        model = model or self.default_model
        if not model:
            raise LLMConfigurationError("No model was provided for generation.")

        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        return model, temperature, max_tokens

    @staticmethod
    def _validate_prompt(prompt: str) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty.")
        return prompt

    def create(
        self,
        system_prompt: str,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = self._validate_prompt(prompt)
        model, temperature, max_tokens = self._resolve_params(
            model, temperature, max_tokens
        )

        try:
            response = self.client.responses.create(
                instructions=system_prompt,
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI request failed: {exc}") from exc

        return response.output_text

    def parse(
        self,
        system_prompt: str,
        prompt: str,
        response_model: Type[T],
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> T:
        prompt = self._validate_prompt(prompt)
        model, temperature, max_tokens = self._resolve_params(
            model, temperature, max_tokens
        )

        try:
            response = self.client.responses.parse(
                instructions=system_prompt,
                model=model,
                input=prompt,
                temperature=temperature,
                max_output_tokens=max_tokens,
                text_format=response_model,
            )
        except ValidationError as exc:
            # Model response did not match the expected schema
            raise LLMGenerationError(
                f"Response did not match schema {response_model.__name__}: {exc}"
            ) from exc
        except Exception as exc:
            raise LLMGenerationError(f"OpenAI request failed: {exc}") from exc

        if response.output_parsed is None:
            raise LLMGenerationError(
                "Model refused to generate structured output "
                "(likely triggered safety filter)."
            )

        return response.output_parsed
