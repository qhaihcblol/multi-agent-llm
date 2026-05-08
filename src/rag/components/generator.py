from ...llm.clients.base import BaseLLMClient


class Generator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model_name: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> None:
        if temperature is not None and not 0.0 <= temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")
        if max_tokens is not None and max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")

        self.llm = llm_client
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens

    def generate(self, system_prompt: str, prompt: str) -> str:
        return self.llm.generate(
            system_prompt=system_prompt,
            prompt=prompt,
            model=self.model_name,
            temperature=self.temperature,
            max_tokens=self.max_tokens,
        )
