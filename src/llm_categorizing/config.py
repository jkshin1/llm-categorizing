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
SUPPORTED_ENDPOINT_PROFILES = {"auto", "internal", "alibaba"}
SUPPORTED_PROVIDER_PROFILES = {"auto", "qwen", "glm", "generic"}


class _ScopedEnv:
    def __init__(self, role: str | None = None) -> None:
        self.role = (role or "").strip().upper()
        self.prefix = f"{self.role}_" if self.role else ""

    def get(self, name: str, default: str = "") -> str:
        if self.prefix:
            scoped = os.getenv(f"{self.prefix}{name}")
            if scoped is not None:
                return scoped
        return os.getenv(name, default)

    def bool(self, name: str, default: bool = False) -> bool:
        raw = self.get(name, "")
        if raw == "":
            return default
        return raw.strip().lower() in {"1", "true", "yes", "y", "on"}

    def label(self, name: str) -> str:
        if not self.prefix:
            return name
        return f"{self.prefix}{name} or {name}"


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
    endpoint_profile: str = "internal"
    provider_profile: str = "generic"

    @classmethod
    def from_env(cls, role: str | None = None) -> "LLMSettings":
        load_dotenv()
        env = _ScopedEnv(role)

        endpoint_profile = _endpoint_profile_from_env(env)
        base_url, api_key, model, resolved_endpoint_profile = _endpoint_settings_from_env(
            endpoint_profile,
            env,
        )

        provider_profile = _provider_profile_from_env(model, env)

        return cls(
            base_url=base_url,
            api_key=api_key,
            model=model,
            temperature=float(env.get("LLM_TEMPERATURE", "0")),
            timeout_seconds=float(
                env.get("LLM_TIMEOUT_SECONDS", str(int(DEFAULT_LLM_TIMEOUT_SECONDS)))
            ),
            max_tokens=_max_tokens_from_env(model, provider_profile, env),
            use_json_response_format=env.bool("LLM_USE_JSON_RESPONSE_FORMAT", False),
            extra_body=_extra_body_from_env(model, provider_profile, env),
            endpoint_profile=resolved_endpoint_profile,
            provider_profile=provider_profile,
        )


def _endpoint_profile_from_env(env: _ScopedEnv) -> str:
    requested = env.get("LLM_ENDPOINT_PROFILE", "auto").strip().casefold()
    if requested not in SUPPORTED_ENDPOINT_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_ENDPOINT_PROFILES))
        raise ValueError(f"{env.label('LLM_ENDPOINT_PROFILE')} must be one of: {supported}")
    return requested


def _endpoint_settings_from_env(
    endpoint_profile: str,
    env: _ScopedEnv,
) -> tuple[str, str, str, str]:
    if endpoint_profile == "auto":
        for candidate in ["internal", "alibaba"]:
            base_url, api_key, model, _ = _endpoint_candidate_from_env(candidate, env)
            if base_url and api_key and model:
                return base_url, api_key, model, candidate
        raise ValueError(
            "Missing required environment variables for an LLM endpoint: "
            f"set {env.label('INTERNAL_LLM_BASE_URL')}/{env.label('INTERNAL_LLM_API_KEY')}/"
            f"{env.label('INTERNAL_LLM_MODEL')} or {env.label('ALIBABA_BASE_URL')}/"
            f"{env.label('ALIBABA_API_KEY')}/{env.label('ALIBABA_MODEL')}"
        )

    base_url, api_key, model, required = _endpoint_candidate_from_env(endpoint_profile, env)
    missing = [name for name, value in required.items() if not value]
    if missing:
        joined = ", ".join(missing)
        raise ValueError(f"Missing required environment variables for {endpoint_profile}: {joined}")
    return base_url, api_key, model, endpoint_profile


def _endpoint_candidate_from_env(
    endpoint_profile: str,
    env: _ScopedEnv,
) -> tuple[str, str, str, dict[str, str]]:
    common_model = env.get("LLM_MODEL", "").strip()
    if endpoint_profile == "alibaba":
        base_url = env.get("ALIBABA_BASE_URL", "").strip()
        api_key = env.get("ALIBABA_API_KEY", "").strip()
        model = env.get("ALIBABA_MODEL", "").strip() or common_model
        return (
            base_url,
            api_key,
            model,
            {
                env.label("ALIBABA_BASE_URL"): base_url,
                env.label("ALIBABA_API_KEY"): api_key,
                f"{env.label('ALIBABA_MODEL')} or {env.label('LLM_MODEL')}": model,
            },
        )

    base_url = env.get("INTERNAL_LLM_BASE_URL", "").strip()
    api_key = env.get("INTERNAL_LLM_API_KEY", "").strip()
    model = env.get("INTERNAL_LLM_MODEL", "").strip() or common_model
    return (
        base_url,
        api_key,
        model,
        {
            env.label("INTERNAL_LLM_BASE_URL"): base_url,
            env.label("INTERNAL_LLM_API_KEY"): api_key,
            f"{env.label('INTERNAL_LLM_MODEL')} or {env.label('LLM_MODEL')}": model,
        },
    )


def _provider_profile_from_env(model: str, env: _ScopedEnv) -> str:
    requested = env.get("LLM_PROVIDER_PROFILE", "auto").strip().casefold()
    if requested not in SUPPORTED_PROVIDER_PROFILES:
        supported = ", ".join(sorted(SUPPORTED_PROVIDER_PROFILES))
        raise ValueError(f"{env.label('LLM_PROVIDER_PROFILE')} must be one of: {supported}")
    if requested != "auto":
        return requested
    if _looks_like_qwen(model):
        return "qwen"
    if _looks_like_glm(model):
        return "glm"
    return "generic"


def _extra_body_from_env(
    model: str,
    provider_profile: str,
    env: _ScopedEnv,
) -> dict[str, Any] | None:
    extra_body: dict[str, Any] = {}

    raw_json = _profile_extra_body_raw(provider_profile, env)
    if raw_json:
        parsed = json.loads(raw_json)
        if not isinstance(parsed, dict):
            raise ValueError(f"{_profile_extra_body_env_name(provider_profile, env)} must be a JSON object")
        extra_body.update(parsed)

    if _qwen_thinking_disabled(model, provider_profile, env):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError(
                f"{_profile_extra_body_env_name(provider_profile, env)}.chat_template_kwargs "
                "must be a JSON object"
            )
        chat_template_kwargs.setdefault("enable_thinking", False)
    elif _qwen_thinking_enabled(model, provider_profile, env):
        chat_template_kwargs = extra_body.setdefault("chat_template_kwargs", {})
        if not isinstance(chat_template_kwargs, dict):
            raise ValueError(
                f"{_profile_extra_body_env_name(provider_profile, env)}.chat_template_kwargs "
                "must be a JSON object"
            )
        chat_template_kwargs.setdefault("enable_thinking", True)

    return extra_body or None


def _max_tokens_from_env(model: str, provider_profile: str, env: _ScopedEnv) -> int:
    thinking_tokens = int(
        env.get("LLM_QWEN_THINKING_MAX_TOKENS", str(DEFAULT_QWEN_THINKING_MAX_TOKENS))
    )
    raw = _profile_max_tokens_raw(provider_profile, env)
    if raw:
        parsed = int(raw)
        if _qwen_thinking_enabled(model, provider_profile, env):
            return max(parsed, thinking_tokens)
        return parsed
    if _qwen_thinking_enabled(model, provider_profile, env):
        return thinking_tokens
    if provider_profile == "glm":
        return DEFAULT_GLM_MAX_TOKENS
    return DEFAULT_LLM_MAX_TOKENS


def _profile_max_tokens_raw(provider_profile: str, env: _ScopedEnv) -> str:
    if provider_profile == "qwen":
        return env.get("LLM_QWEN_MAX_TOKENS", "").strip() or env.get("LLM_MAX_TOKENS", "").strip()
    if provider_profile == "glm":
        return env.get("LLM_GLM_MAX_TOKENS", "").strip() or env.get("LLM_MAX_TOKENS", "").strip()
    return env.get("LLM_MAX_TOKENS", "").strip()


def _profile_extra_body_raw(provider_profile: str, env: _ScopedEnv) -> str:
    if provider_profile == "qwen":
        return env.get("LLM_QWEN_EXTRA_BODY_JSON", "").strip() or env.get("LLM_EXTRA_BODY_JSON", "").strip()
    if provider_profile == "glm":
        return env.get("LLM_GLM_EXTRA_BODY_JSON", "").strip()
    return env.get("LLM_EXTRA_BODY_JSON", "").strip()


def _profile_extra_body_env_name(provider_profile: str, env: _ScopedEnv) -> str:
    if provider_profile == "qwen":
        return f"{env.label('LLM_QWEN_EXTRA_BODY_JSON')} or {env.label('LLM_EXTRA_BODY_JSON')}"
    if provider_profile == "glm":
        return env.label("LLM_GLM_EXTRA_BODY_JSON")
    return env.label("LLM_EXTRA_BODY_JSON")


def _qwen_thinking_disabled(model: str, provider_profile: str, env: _ScopedEnv) -> bool:
    return provider_profile == "qwen" and env.bool("LLM_QWEN_DISABLE_THINKING", False)


def _qwen_thinking_enabled(model: str, provider_profile: str, env: _ScopedEnv) -> bool:
    return provider_profile == "qwen" and not _qwen_thinking_disabled(model, provider_profile, env)


def _looks_like_qwen(model: str) -> bool:
    return "qwen" in model.casefold()


def _looks_like_glm(model: str) -> bool:
    return "glm" in model.casefold()
