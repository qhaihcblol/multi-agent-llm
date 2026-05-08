from ...llm.clients.base import BaseLLMClient


class Generator:
    def __init__(
        self,
        llm_client: BaseLLMClient,
        model_name: str = "gpt-4o",
        temperature: float = 0.7,
        max_tokens: int = 2048,
    ) -> None:
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
