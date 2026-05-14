from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv


DEFAULT_LLM_TIMEOUT_SECONDS = 300.0


def _env_bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


@dataclass(frozen=True)
class LLMSettings:
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    timeout_seconds: float = DEFAULT_LLM_TIMEOUT_SECONDS
    max_tokens: int = 1200
    use_json_response_format: bool = False

    @classmethod
    def from_env(cls) -> "LLMSettings":
        load_dotenv()

        base_url = os.getenv("INTERNAL_LLM_BASE_URL", "").strip()
        api_key = os.getenv("INTERNAL_LLM_API_KEY", "").strip()
        model = os.getenv("INTERNAL_LLM_MODEL", "").strip()

        missing = [
            name
            for name, value in {
                "INTERNAL_LLM_BASE_URL": base_url,
                "INTERNAL_LLM_API_KEY": api_key,
                "INTERNAL_LLM_MODEL": model,
            }.items()
            if not value
        ]
        if missing:
            joined = ", ".join(missing)
            raise ValueError(f"Missing required environment variables: {joined}")

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            timeout_seconds=float(
                os.getenv("LLM_TIMEOUT_SECONDS", str(int(DEFAULT_LLM_TIMEOUT_SECONDS)))
            ),
            max_tokens=int(os.getenv("LLM_MAX_TOKENS", "1200")),
            use_json_response_format=_env_bool("LLM_USE_JSON_RESPONSE_FORMAT", False),
        )
