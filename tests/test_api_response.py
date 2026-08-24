from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from rag.api import main as api_main
from rag.api.main import QueryResponse


def _response_payload() -> dict:
    return {
        "answer": "根據檢索到的法規內容無法回答此問題。",
        "refused": True,
        "sources": [],
        "retrieval_hits": [],
        "strategy": "structure",
        "mode": "hybrid",
        "use_reranker": True,
        "provider": "openai",
        "model": "example-model",
    }


def test_query_response_keeps_refusal_stage_optional_for_older_callers():
    response = QueryResponse(**_response_payload())

    assert response.refusal_stage is None


@pytest.mark.parametrize("stage", ["no_hits", "threshold", "llm"])
def test_query_response_accepts_known_refusal_stages(stage):
    response = QueryResponse(**_response_payload(), refusal_stage=stage)

    assert response.refusal_stage == stage


def test_query_response_rejects_unknown_refusal_stage():
    with pytest.raises(ValidationError):
        QueryResponse(**_response_payload(), refusal_stage="other")


def test_query_exposes_answerer_refusal_stage(monkeypatch):
    settings = SimpleNamespace(
        chunking_strategy="structure",
        retrieval_mode="hybrid",
        use_reranker=True,
        llm_provider="openai",
        resolved_generation_model="example-model",
    )
    result = SimpleNamespace(
        text="根據檢索到的法規內容無法回答此問題。",
        refused=True,
        sources=[],
        retrieval=SimpleNamespace(hits=[]),
        refusal_stage="threshold",
    )
    answerer = SimpleNamespace(answer=lambda _question: result)
    monkeypatch.setattr(api_main.state, "settings", settings, raising=False)
    monkeypatch.setattr(api_main.state, "get_answerer", lambda *_args: answerer)

    response = api_main.query(api_main.QueryRequest(question="雇主可以問不在法規庫的問題嗎?"))

    assert response.refused is True
    assert response.refusal_stage == "threshold"
