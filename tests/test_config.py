from rag.config import DEFAULT_GENERATION_MODELS, Settings


def test_defaults():
    s = Settings(_env_file=None)
    assert s.llm_provider == "anthropic"
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
    assert s.resolved_generation_model == DEFAULT_GENERATION_MODELS["anthropic"]
    s2 = Settings(_env_file=None, generation_model="my-custom-model")
    assert s2.resolved_generation_model == "my-custom-model"
