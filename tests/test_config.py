from llm_categorizing.config import LLMSettings


def _base_env(monkeypatch, model: str) -> None:
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "internal")
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("INTERNAL_LLM_API_KEYS", "")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "test")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", model)
    monkeypatch.setenv("ALIBABA_BASE_URL", "")
    monkeypatch.setenv("ALIBABA_API_KEYS", "")
    monkeypatch.setenv("ALIBABA_API_KEY", "")
    monkeypatch.setenv("ALIBABA_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "")
    monkeypatch.setenv("LLM_PROVIDER_PROFILE", "auto")
    monkeypatch.setenv("LLM_MAX_TOKENS", "")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", "")
    monkeypatch.setenv("LLM_QWEN_DISABLE_THINKING", "")
    monkeypatch.setenv("LLM_QWEN_THINKING_MAX_TOKENS", "4096")
    monkeypatch.setenv("LLM_QWEN_EXTRA_BODY_JSON", "")
    monkeypatch.setenv("LLM_GLM_EXTRA_BODY_JSON", "")


def test_qwen_model_enables_thinking_by_default(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")

    settings = LLMSettings.from_env()

    assert settings.provider_profile == "qwen"
    assert settings.max_tokens == 4096
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_extra_body_json_is_merged(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"top_k": 20}')

    settings = LLMSettings.from_env()

    assert settings.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": True},
    }


def test_qwen_thinking_can_be_disabled(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_QWEN_DISABLE_THINKING", "1")

    settings = LLMSettings.from_env()

    assert settings.max_tokens == 1200
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_explicit_max_tokens_overrides_qwen_thinking_default(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_MAX_TOKENS", "8192")

    settings = LLMSettings.from_env()

    assert settings.max_tokens == 8192


def test_small_explicit_max_tokens_is_bumped_for_qwen_thinking(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
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
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": True}}


def test_classification_role_can_use_glm_model_while_shared_model_is_qwen(monkeypatch) -> None:
    _base_env(monkeypatch, "Qwen3.6-35B-A3B")
    monkeypatch.setenv("CLASSIFICATION_LLM_ENDPOINT_PROFILE", "internal")
    monkeypatch.setenv("CLASSIFICATION_INTERNAL_LLM_MODEL", "glm-5.1")
    monkeypatch.setenv("CLASSIFICATION_LLM_PROVIDER_PROFILE", "glm")
    monkeypatch.setenv("CLASSIFICATION_LLM_GLM_MAX_TOKENS", "4096")

    settings = LLMSettings.from_env(role="classification")

    assert settings.endpoint_profile == "internal"
    assert settings.model == "glm-5.1"
    assert settings.provider_profile == "glm"
    assert settings.max_tokens == 4096
    assert settings.extra_body is None


def test_knowledge_role_can_use_qwen_thinking_disabled_while_shared_model_is_glm(monkeypatch) -> None:
    _base_env(monkeypatch, "glm-5.1")
    monkeypatch.setenv("LLM_PROVIDER_PROFILE", "glm")
    monkeypatch.setenv("KNOWLEDGE_LLM_ENDPOINT_PROFILE", "internal")
    monkeypatch.setenv("KNOWLEDGE_INTERNAL_LLM_MODEL", "Qwen3.6-35B-A3B")
    monkeypatch.setenv("KNOWLEDGE_LLM_PROVIDER_PROFILE", "qwen")
    monkeypatch.setenv("KNOWLEDGE_LLM_QWEN_DISABLE_THINKING", "1")
    monkeypatch.setenv("KNOWLEDGE_LLM_MAX_TOKENS", "1200")

    settings = LLMSettings.from_env(role="knowledge")

    assert settings.endpoint_profile == "internal"
    assert settings.model == "Qwen3.6-35B-A3B"
    assert settings.provider_profile == "qwen"
    assert settings.max_tokens == 1200
    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_internal_endpoint_profile_accepts_multiple_api_keys(monkeypatch) -> None:
    _base_env(monkeypatch, "glm-5.1")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "internal-key-1, internal-key-2")

    settings = LLMSettings.from_env()

    assert settings.endpoint_profile == "internal"
    assert settings.api_key == "internal-key-1"
    assert settings.api_keys == ("internal-key-1", "internal-key-2")


def test_alibaba_endpoint_profile_uses_alibaba_env(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "alibaba")
    monkeypatch.setenv("ALIBABA_BASE_URL", "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv("ALIBABA_API_KEY", "alibaba-test")
    monkeypatch.setenv("ALIBABA_MODEL", "qwen-plus")

    settings = LLMSettings.from_env()

    assert settings.endpoint_profile == "alibaba"
    assert settings.base_url == "https://dashscope.example.com/compatible-mode/v1"
    assert settings.api_key == "alibaba-test"
    assert settings.api_keys == ("alibaba-test",)
    assert settings.model == "qwen-plus"
    assert settings.provider_profile == "qwen"


def test_alibaba_endpoint_profile_accepts_multiple_api_keys(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "alibaba")
    monkeypatch.setenv("ALIBABA_BASE_URL", "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv("ALIBABA_API_KEY", "")
    monkeypatch.setenv("ALIBABA_API_KEYS", "alibaba-key-1, alibaba-key-2")
    monkeypatch.setenv("ALIBABA_MODEL", "qwen-plus")

    settings = LLMSettings.from_env()

    assert settings.endpoint_profile == "alibaba"
    assert settings.api_key == "alibaba-key-1"
    assert settings.api_keys == ("alibaba-key-1", "alibaba-key-2")


def test_role_scoped_alibaba_api_keys_override_shared_keys(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "alibaba")
    monkeypatch.setenv("ALIBABA_BASE_URL", "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv("ALIBABA_API_KEYS", "shared-1,shared-2")
    monkeypatch.setenv("ALIBABA_MODEL", "qwen-plus")
    monkeypatch.setenv("CLASSIFICATION_ALIBABA_API_KEYS", '["class-1", "class-2"]')

    settings = LLMSettings.from_env(role="classification")

    assert settings.api_key == "class-1"
    assert settings.api_keys == ("class-1", "class-2")


def test_auto_endpoint_profile_falls_back_to_alibaba_when_internal_is_missing(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "auto")
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", "")
    monkeypatch.setenv("ALIBABA_BASE_URL", "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv("ALIBABA_API_KEY", "alibaba-test")
    monkeypatch.setenv("ALIBABA_MODEL", "qwen-turbo")

    settings = LLMSettings.from_env()

    assert settings.endpoint_profile == "alibaba"
    assert settings.model == "qwen-turbo"


def test_explicit_alibaba_endpoint_requires_model(monkeypatch) -> None:
    _base_env(monkeypatch, "internal-prod-model")
    monkeypatch.setenv("LLM_ENDPOINT_PROFILE", "alibaba")
    monkeypatch.setenv("ALIBABA_BASE_URL", "https://dashscope.example.com/compatible-mode/v1")
    monkeypatch.setenv("ALIBABA_API_KEY", "alibaba-test")
    monkeypatch.setenv("ALIBABA_MODEL", "")
    monkeypatch.setenv("LLM_MODEL", "")

    try:
        LLMSettings.from_env()
    except ValueError as exc:
        assert "ALIBABA_MODEL or LLM_MODEL" in str(exc)
    else:
        raise AssertionError("expected missing ALIBABA_MODEL to raise")
