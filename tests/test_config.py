from llm_categorizing.config import LLMSettings


def test_qwen_model_disables_thinking_by_default(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "test")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", "Qwen3.6-35B-A3B")

    settings = LLMSettings.from_env()

    assert settings.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


def test_extra_body_json_is_merged(monkeypatch) -> None:
    monkeypatch.setenv("INTERNAL_LLM_BASE_URL", "http://localhost:1/v1")
    monkeypatch.setenv("INTERNAL_LLM_API_KEY", "test")
    monkeypatch.setenv("INTERNAL_LLM_MODEL", "Qwen3.6-35B-A3B")
    monkeypatch.setenv("LLM_EXTRA_BODY_JSON", '{"top_k": 20}')

    settings = LLMSettings.from_env()

    assert settings.extra_body == {
        "top_k": 20,
        "chat_template_kwargs": {"enable_thinking": False},
    }
