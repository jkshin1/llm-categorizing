from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv


DEFAULT_LLM_TIMEOUT_SECONDS = 300.0
DEFAULT_LLM_MAX_TOKENS = 1200
DEFAULT_GLM_MAX_TOKENS = 2048
DEFAULT_QWEN_THINKING_MAX_TOKENS = 4096
SUPPORTED_PROVIDER_PROFILES = {"auto", "qwen", "glm", "generic"}


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
    max_tokens: int = DEFAULT_LLM_MAX_TOKENS
    use_json_response_format: bool = False
    extra_body: dict[str, Any] | None = None
    provider_profile: str = "generic"

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

        provider_profile = _provider_profile_from_env(model)

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(os.getenv("LLM_TEMPERATURE", "0")),
            timeout_seconds=float(
                os.getenv("LLM_TIMEOUT_SECONDS", str(int(DEFAULT_LLM_TIMEOUT_SECONDS)))
            ),
            max_tokens=_max_tokens_from_env(model, provider_profile),
            use_json_response_format=_env_bool("LLM_USE_JSON_RESPONSE_FORMAT", False),
            extra_body=_extra_body_from_env(model, provider_profile),
            provider_profile=provider_profile,
        )


def _provider_profile_from_env(model: str) -> str:
    requested = os.getenv("LLM_PROVIDER_PROFILE", "auto").strip().casefold()
    if requested not in SUPPORTED_PROVIDER_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_PROFILES))
        raise ValueError(f"LLM_PROVIDER_PROFILE must be one of: {supported}")
    if requested != "auto":
        return requested
    if _looks_like_qwen(model):
        return "qwen"
    if _looks_like_glm(model):
        return "glm"
    return "generic"


def _extra_body_from_env(model: str, provider_profile: str) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}

    raw_json = _profile_extra_body_raw(provider_profile)
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"{_profile_extra_body_env_name(provider_profile)} must be a JSON object")
        extra_body.update(parsed)

    if _qwen_thinking_disabled(model, provider_profile):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError(
                f"{_profile_extra_body_env_name(provider_profile)}.chat_template_kwargs must be a JSON object"
            )
        chat_template_kwargs.setdefault("enable_thinking", False)
    elif _qwen_thinking_enabled(model, provider_profile):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError(
                f"{_profile_extra_body_env_name(provider_profile)}.chat_template_kwargs must be a JSON object"
            )
        chat_template_kwargs.setdefault("enable_thinking", True)

    return extra_body or None


def _max_tokens_from_env(model: str, provider_profile: str) -> int:
    thinking_tokens = int(
        os.getenv("LLM_QWEN_THINKING_MAX_TOKENS", str(DEFAULT_QWEN_THINKING_MAX_TOKENS))
    )
    raw = _profile_max_tokens_raw(provider_profile)
    if raw:
        parsed = int(raw)
        if _qwen_thinking_enabled(model, provider_profile):
            return max(parsed, thinking_tokens)
        return parsed
    if _qwen_thinking_enabled(model, provider_profile):
        return thinking_tokens
    if provider_profile == "glm":
        return DEFAULT_GLM_MAX_TOKENS
    return DEFAULT_LLM_MAX_TOKENS


def _profile_max_tokens_raw(provider_profile: str) -> str:
    if provider_profile == "qwen":
        return os.getenv("LLM_QWEN_MAX_TOKENS", "").strip() or os.getenv("LLM_MAX_TOKENS", "").strip()
    if provider_profile == "glm":
        return os.getenv("LLM_GLM_MAX_TOKENS", "").strip() or os.getenv("LLM_MAX_TOKENS", "").strip()
    return os.getenv("LLM_MAX_TOKENS", "").strip()


def _profile_extra_body_raw(provider_profile: str) -> str:
    if provider_profile == "qwen":
        return os.getenv("LLM_QWEN_EXTRA_BODY_JSON", "").strip() or os.getenv("LLM_EXTRA_BODY_JSON", "").strip()
    if provider_profile == "glm":
        return os.getenv("LLM_GLM_EXTRA_BODY_JSON", "").strip()
    return os.getenv("LLM_EXTRA_BODY_JSON", "").strip()


def _profile_extra_body_env_name(provider_profile: str) -> str:
    if provider_profile == "qwen":
        return "LLM_QWEN_EXTRA_BODY_JSON or LLM_EXTRA_BODY_JSON"
    if provider_profile == "glm":
        return "LLM_GLM_EXTRA_BODY_JSON"
    return "LLM_EXTRA_BODY_JSON"


def _qwen_thinking_disabled(model: str, provider_profile: str) -> bool:
    return provider_profile == "qwen" and _env_bool("LLM_QWEN_DISABLE_THINKING", True)


def _qwen_thinking_enabled(model: str, provider_profile: str) -> bool:
    return provider_profile == "qwen" and not _qwen_thinking_disabled(model, provider_profile)


def _looks_like_qwen(model: str) -> bool:
    return "qwen" in model.casefold()


def _looks_like_glm(model: str) -> bool:
    return "glm" in model.casefold()
