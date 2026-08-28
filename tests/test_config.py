import pytest

from rag.config import DEFAULT_GENERATION_MODELS, PUBLIC_LLM_PROVIDERS, Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "gemini"
    assert s.qdrant_mode == "local"
    assert s.llm_temperature == 0.0
    assert s.top_k_retrieve == 20
    assert s.top_k_final == 5


def test_env_override(monkeypatch):
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    monkeypatch.setenv("QDRANT_MODE", "server")
    monkeypatch.setenv("CHUNK_SIZE", "256")
    s = Settings(_env_file=None)
    assert s.llm_provider == "openai"
    assert s.qdrant_mode == "server"
    assert s.chunk_size == 256


def test_model_resolution():
    s = Settings(_env_file=None)
    assert s.resolved_generation_model == DEFAULT_GENERATION_MODELS["gemini"]
    s2 = Settings(_env_file=None, generation_model="my-custom-model")
    assert s2.resolved_generation_model == "my-custom-model"


def test_public_provider_defaults_are_fixed():
    settings = Settings(_env_file=None)

    assert settings.llm_provider == "gemini"
    assert PUBLIC_LLM_PROVIDERS == ("gemini", "openai")
    assert settings.generation_model_for("gemini") == "gemini-3.5-flash-lite"
    assert settings.generation_model_for("openai") == "gpt-5.6-luna"
    assert settings.llm_fallback_enabled is True


def test_provider_specific_models_do_not_leak_between_providers():
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        gemini_generation_model="gemini-test",
        openai_generation_model="openai-test",
    )

    assert settings.generation_model_for("gemini") == "gemini-test"
    assert settings.generation_model_for("openai") == "openai-test"


def test_legacy_generation_model_only_overrides_environment_default():
    settings = Settings(
        _env_file=None,
        llm_provider="gemini",
        generation_model="legacy-gemini",
    )

    assert settings.generation_model_for("gemini") == "legacy-gemini"
    assert settings.generation_model_for("openai") == DEFAULT_GENERATION_MODELS["openai"]


def test_generation_model_for_rejects_unknown_provider():
    with pytest.raises(ValueError, match="unknown LLM provider"):
        Settings(_env_file=None).generation_model_for("bedrock")
