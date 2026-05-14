from __future__ import annotations

import json
import os
from typing import Any
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
    extra_body: dict[str, Any] | None = None

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
            extra_body=_extra_body_from_env(model),
        )


def _extra_body_from_env(model: str) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}

    raw_json = os.getenv("LLM_EXTRA_BODY_JSON", "").strip()
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError("LLM_EXTRA_BODY_JSON must be a JSON object")
        extra_body.update(parsed)

    if _env_bool("LLM_QWEN_DISABLE_THINKING", _looks_like_qwen(model)):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError("LLM_EXTRA_BODY_JSON.chat_template_kwargs must be a JSON object")
        chat_template_kwargs.setdefault("enable_thinking", False)

    return extra_body or None


def _looks_like_qwen(model: str) -> bool:
    return "qwen" in model.casefold()
