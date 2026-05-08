from openai import OpenAI

from .base import BaseLLMClient, LLMConfigurationError, LLMGenerationError
from ..configs.openai import OpenAIConfig
from openai.types.responses import Response


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

    def generate(
        self,
        system_prompt: str,
        prompt: str,
        model: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        prompt = prompt.strip()
        if not prompt:
            raise ValueError("prompt must not be empty.")

        model = model or self.default_model
        if not model:
            raise LLMConfigurationError("No model was provided for generation.")

        temperature = self.default_temperature if temperature is None else temperature
        max_tokens = self.default_max_tokens if max_tokens is None else max_tokens

        if max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")
        if not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

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

        text = self._extract_text(response)
        if not text:
            raise LLMGenerationError("OpenAI returned an empty response.")

        return text

    def _extract_text(self, response: Response) -> str:
        if getattr(response, "output_text", None):
            return response.output_text.strip()

        output = getattr(response, "output", []) or []
        fragments: list[str] = []

        for item in output:
            content = getattr(item, "content", []) or []

            for block in content:
                if getattr(block, "type", None) == "output_text":
                    text = getattr(block, "text", None)
                    if text:
                        fragments.append(text)

        return "".join(fragments).strip()