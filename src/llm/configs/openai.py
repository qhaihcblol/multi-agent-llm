import os
from dataclasses import dataclass


@dataclass(slots=True, frozen=True)
class OpenAIConfig:
    api_key: str
    model: str = "gpt-4o"
    base_url: str | None = "https://api.openai.com/v1"
    temperature: float = 0.2
    max_tokens: int = 2048
    timeout: float = 60.0

    @classmethod
    def from_env(cls) -> "OpenAIConfig":
        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY is required.")

        model = os.getenv("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

        raw_base_url = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1").strip()
        base_url = raw_base_url or None

        temperature = _get_float_env("OPENAI_TEMPERATURE", 0.2)
        max_tokens = _get_int_env("OPENAI_MAX_TOKENS", 2048)
        timeout = _get_float_env("OPENAI_TIMEOUT", 60.0)

        if not 0.0 <= temperature <= 2.0:
            raise ValueError("OPENAI_TEMPERATURE must be between 0.0 and 2.0.")
        if max_tokens <= 0:
            raise ValueError("OPENAI_MAX_TOKENS must be > 0.")
        if timeout <= 0:
            raise ValueError("OPENAI_TIMEOUT must be > 0.")

        return cls(
            api_key=api_key,
            model=model,
            base_url=base_url,
            temperature=temperature,
            max_tokens=max_tokens,
            timeout=timeout,
        )


def _get_float_env(name: str, default: float) -> float:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return float(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be a float.") from exc


def _get_int_env(name: str, default: int) -> int:
    raw_value = os.getenv(name)
    if raw_value is None or not raw_value.strip():
        return default
    try:
        return int(raw_value)
    except ValueError as exc:
        raise ValueError(f"{name} must be an integer.") from exc
