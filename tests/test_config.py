from llm_categorizing.config import LLMSettings


def _base_env(monkeypatch, model: str) -> None:
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "test")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", model)
    monkeypatch.setenv("LLM_PROVIDER_PROFILE", "auto")
    monkeypatch.setenv("LLM_MAX_TOKENS", "")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", "")
    monkeypatch.setenv("LLM_QWEN_EXTRA_BODY_JSON", "")
    monkeypatch.setenv("LLM_GLM_EXTRA_BODY_JSON", "")


def test_qwen_model_disables_thinking_by_default(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")

    settings = LLMSettings.from_env()

    assert settings.provider_profile == "qwen"
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_json_is_merged(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"top_k": 20}')

    settings = LLMSettings.from_env()

    assert settings.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }


def test_qwen_thinking_enabled_sets_extra_body_and_larger_tokens(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_QWEN_DISABLE_THINKING", "0")

    settings = LLMSettings.from_env()

    assert settings.max_tokens == 4096
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_explicit_max_tokens_overrides_qwen_thinking_default(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_QWEN_DISABLE_THINKING", "0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")

    settings = LLMSettings.from_env()

    assert settings.max_tokens == 8192


def test_small_explicit_max_tokens_is_bumped_for_qwen_thinking(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_QWEN_DISABLE_THINKING", "0")
    monkeypatch.setenv("LLM_MAX_TOKENS", "1200")

    settings = LLMSettings.from_env()

    assert settings.max_tokens == 4096


def test_glm_profile_does_not_inherit_qwen_extra_body(monkeypatch) -> None:
    _base_env(monkeypatch, "GLM-5")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"chat_template_kwargs":{"enable_thinking":false}}')

    settings = LLMSettings.from_env()

    assert settings.provider_profile == "glm"
    assert settings.max_tokens == 2048
    assert settings.extra_body is None


def test_glm_profile_uses_glm_specific_extra_body(monkeypatch) -> None:
    _base_env(monkeypatch, "GLM-5")
    monkeypatch.setenv("LLM_GLM_EXTRA_BODY_JSON", '{"thinking":false}')
    monkeypatch.setenv("LLM_GLM_MAX_TOKENS", "4096")

    settings = LLMSettings.from_env()

    assert settings.provider_profile == "glm"
    assert settings.max_tokens == 4096
    assert settings.extra_body == {"thinking": False}


def test_profile_can_be_forced(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_PROVIDER_PROFILE", "qwen")

    settings = LLMSettings.from_env()

    assert settings.provider_profile == "qwen"
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}
