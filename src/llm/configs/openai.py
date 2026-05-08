import os
from dataclasses import dataclass


@dataclass(frozen=True)
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4o"
    base_url: str | None = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0

    def __post_init__(self) -> None:
        if not self.api_key.strip():
            raise ValueError("OPENAI_API_KEY is required.")

        if not 0.0 <= self.temperature <= 2.0:
            raise ValueError("temperature must be between 0.0 and 2.0.")

        if self.max_tokens <= 0:
            raise ValueError("max_tokens must be > 0.")

        if self.timeout <= 0:
            raise ValueError("timeout must be > 0.")

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        def _get_float(name: str, default: float) -> float:
            try:
                val = os.getenv(name, str(default)).strip()
                return float(val) if val else default
            except ValueError as e:
                raise ValueError(f"{name} must be a valid float.") from e

        def _get_int(name: str, default: int) -> int:
            try:
                val = os.getenv(name, str(default)).strip()
                return int(val) if val else default
            except ValueError as e:
                raise ValueError(f"{name} must be a valid integer.") from e

        return cls(
            api_key=os.getenv("OPENAI_API_KEY", "").strip(),
            model=os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o",
            base_url=(os.getenv("OPENAI_BASE_URL", "").strip() 
                     or "https://api.openai.com/v1") or None,
            temperature=_get_float("OPENAI_TEMPERATURE", 0.2),
            max_tokens=_get_int("OPENAI_MAX_TOKENS", 2048),
            timeout=_get_float("OPENAI_TIMEOUT", 60.0),
        )
