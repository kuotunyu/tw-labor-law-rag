from types import SimpleNamespace

from rag import factory
from rag.generation.answerer import Answerer
from rag.generation.llm import LLMOutput
from rag.generation.prompts import REFUSAL_PHRASE
from rag.generation.router import RoutedLLM
from rag.models import RetrievedChunk
from rag.retrieval.pipeline import RetrievalPipeline


def make_hit(chunk_id, doc_title, article_label, content, score=0.9):
    return RetrievedChunk(
        score=score,
        payload={
            "chunk_id": chunk_id,
            "doc_title": doc_title,
            "article_label": article_label,
            "content": content,
            "text": f"{doc_title} {article_label}\n{content}",
        },
    )


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k):
        return self.hits[:top_k]


class FakeReranker:
    def __init__(self, hits):
        self.hits = hits

    def rerank(self, query, candidates, top_k):
        return self.hits[:top_k]


class FakeLLM:
    def __init__(
        self,
        response: str,
        *,
        primary_provider: str = "gemini",
        provider: str | None = None,
        model: str | None = None,
        fallback_used: bool = False,
        fallback_from: str | None = None,
    ):
        self.response = response
        self.primary_provider = primary_provider
        self.provider = provider or primary_provider
        self.model = model or f"{self.provider}-test"
        self.fallback_used = fallback_used
        self.fallback_from = fallback_from
        self.calls = []

    def generate(self, system, user, temperature=0.0, max_tokens=1024):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return LLMOutput(
            text=self.response,
            provider=self.provider,
            model=self.model,
            fallback_used=self.fallback_used,
            fallback_from=self.fallback_from,
        )


class FakeConcreteLLM:
    provider = "gemini"
    model = "gemini-test"

    def __init__(self, response: str = "unused"):
        self.response = response
        self.calls = []

    def generate(self, system, user, temperature=0.0, max_tokens=1024):
        self.calls.append({"system": system, "user": user, "temperature": temperature})
        return LLMOutput(text=self.response, provider=self.provider, model=self.model)


def make_pipeline(hits, reranker=None, top_k_final=5):
    return RetrievalPipeline(FakeRetriever(hits), reranker=reranker, top_k_retrieve=20, top_k_final=top_k_final)


def test_answerer_parses_citations():
    hits = [
        make_hit("c1", "勞動基準法", "第 24 條", "加班費規定..."),
        make_hit("c2", "勞動基準法", "第 30 條", "工時規定..."),
    ]
    llm = FakeLLM("依 [1] 規定,加班費要加給。")
    result = Answerer(make_pipeline(hits), llm).answer("加班費怎麼算?")

    assert not result.refused
    assert result.refusal_stage is None
    assert result.sources == [
        {"index": 1, "doc": "勞動基準法", "article": "第 24 條", "content": "加班費規定..."}
    ]
    assert llm.calls[0]["temperature"] == 0.0


def test_answerer_reports_generation_provider_metadata():
    hits = [make_hit("c1", "勞動基準法", "第 24 條", "內容")]
    llm = FakeLLM("依 [1] 回答。", model="gemini-2.5-flash")

    result = Answerer(make_pipeline(hits), llm).answer("問題")

    assert result.generation_called is True
    assert result.requested_provider == "gemini"
    assert result.provider == "gemini"
    assert result.model == "gemini-2.5-flash"
    assert result.fallback_used is False
    assert result.fallback_from is None


def test_answerer_reports_actual_fallback_provider():
    hits = [make_hit("c1", "勞動基準法", "第 24 條", "內容")]
    llm = FakeLLM(
        "依 [1] 回答。",
        primary_provider="gemini",
        provider="openai",
        model="gpt-5.6-luna",
        fallback_used=True,
        fallback_from="gemini",
    )

    result = Answerer(make_pipeline(hits), llm).answer("問題")

    assert result.generation_called is True
    assert result.requested_provider == "gemini"
    assert result.provider == "openai"
    assert result.model == "gpt-5.6-luna"
    assert result.fallback_used is True
    assert result.fallback_from == "gemini"


def test_answerer_parses_fullwidth_bracket_citations():
    """gpt-5.1 was observed emitting ［1］ (full-width) when writing Chinese."""
    hits = [make_hit("c1", "工會法", "第 11 條", "三十人以上連署發起。")]
    llm = FakeLLM("至少需要 30 人連署［1］。")
    result = Answerer(make_pipeline(hits), llm).answer("組工會要幾人?")
    assert [s["index"] for s in result.sources] == [1]


def test_answerer_ignores_out_of_range_citations():
    hits = [make_hit("c1", "勞動基準法", "第 24 條", "內容")]
    llm = FakeLLM("依 [1][5] 規定作答。")
    result = Answerer(make_pipeline(hits), llm).answer("問題")
    assert [s["index"] for s in result.sources] == [1]


def test_answerer_dedupes_repeated_citations():
    hits = [make_hit("c1", "勞動基準法", "第 24 條", "內容")]
    llm = FakeLLM("依 [1] 規定... 再次依 [1] 規定。")
    result = Answerer(make_pipeline(hits), llm).answer("問題")
    assert [s["index"] for s in result.sources] == [1]


def test_answerer_generation_layer_refusal():
    hits = [make_hit("c1", "勞動基準法", "第 24 條", "不相關內容")]
    llm = FakeLLM(f"{REFUSAL_PHRASE},無法回答。")
    result = Answerer(make_pipeline(hits), llm).answer("問題")
    assert result.refused
    assert result.refusal_stage == "llm"
    assert result.sources == []


def test_answerer_no_hits_refuses_without_calling_llm():
    llm = FakeLLM("should not be called")
    result = Answerer(make_pipeline([]), llm).answer("問題")
    assert result.refused
    assert result.refusal_stage == "no_hits"
    assert result.generation_called is False
    assert result.requested_provider == "gemini"
    assert result.provider is None
    assert result.model is None
    assert result.fallback_used is False
    assert result.fallback_from is None
    assert llm.calls == []


def test_answerer_retrieval_layer_refusal_below_threshold():
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "內容", score=0.1)]
    llm = FakeLLM("should not be called")
    pipeline = make_pipeline(hits, reranker=FakeReranker(hits))
    result = Answerer(pipeline, llm, refusal_threshold=0.5).answer("問題")
    assert result.refused
    assert result.refusal_stage == "threshold"
    assert llm.calls == []


def test_threshold_refusal_reports_no_generation_provider():
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "內容", score=0.1)]
    llm = FakeLLM("unused", primary_provider="gemini")
    result = Answerer(
        make_pipeline(hits, reranker=FakeReranker(hits)),
        llm,
        refusal_threshold=0.5,
    ).answer("問題")

    assert result.generation_called is False
    assert result.requested_provider == "gemini"
    assert result.provider is None
    assert result.model is None
    assert result.fallback_used is False
    assert result.fallback_from is None
    assert llm.calls == []


def test_routed_llm_refusal_reports_primary_without_generation():
    primary = FakeConcreteLLM()
    fallback = FakeConcreteLLM()
    fallback.provider = "openai"
    fallback.model = "openai-test"
    llm = RoutedLLM(primary, fallback)

    result = Answerer(make_pipeline([]), llm).answer("問題")

    assert result.generation_called is False
    assert result.requested_provider == "gemini"
    assert result.provider is None
    assert primary.calls == []
    assert fallback.calls == []


def test_factory_uses_injected_routed_llm_for_refusal(monkeypatch):
    pipeline = make_pipeline([])
    llm = RoutedLLM(FakeConcreteLLM())
    settings = SimpleNamespace(rerank_score_threshold=0.5, llm_temperature=0.0)
    monkeypatch.setattr(factory, "build_retrieval_pipeline", lambda *args, **kwargs: pipeline)

    def fail_if_default_llm_is_built(settings):
        raise AssertionError("build_llm must not run for an injected LLM")

    monkeypatch.setattr(factory, "build_llm", fail_if_default_llm_is_built)

    result = factory.build_answerer(settings, object(), object(), llm=llm).answer("問題")

    assert result.generation_called is False
    assert result.requested_provider == "gemini"


def test_factory_default_concrete_llm_refusal_is_compatible(monkeypatch):
    pipeline = make_pipeline([])
    llm = FakeConcreteLLM()
    settings = SimpleNamespace(rerank_score_threshold=0.5, llm_temperature=0.0)
    monkeypatch.setattr(factory, "build_retrieval_pipeline", lambda *args, **kwargs: pipeline)
    monkeypatch.setattr(factory, "build_llm", lambda settings: llm)

    result = factory.build_answerer(settings, object(), object()).answer("問題")

    assert result.generation_called is False
    assert result.requested_provider == "gemini"
    assert result.provider is None
    assert llm.calls == []


def test_answerer_retrieval_layer_passes_threshold_when_high_enough():
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "內容", score=0.9)]
    llm = FakeLLM("依 [1] 回答。")
    pipeline = make_pipeline(hits, reranker=FakeReranker(hits))
    result = Answerer(pipeline, llm, refusal_threshold=0.5).answer("問題")
    assert not result.refused
    assert result.refusal_stage is None
    assert len(llm.calls) == 1


def test_answerer_score_equal_to_threshold_calls_llm():
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "內容", score=0.5)]
    llm = FakeLLM("依 [1] 回答。")
    pipeline = make_pipeline(hits, reranker=FakeReranker(hits))
    result = Answerer(pipeline, llm, refusal_threshold=0.5).answer("問題")
    assert not result.refused
    assert result.refusal_stage is None
    assert len(llm.calls) == 1


def test_answerer_threshold_ignored_without_reranker():
    """Without a reranker, raw retriever scores aren't calibrated — threshold must be a no-op."""
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "內容", score=0.1)]
    llm = FakeLLM("依 [1] 回答。")
    result = Answerer(make_pipeline(hits, reranker=None), llm, refusal_threshold=0.5).answer("問題")
    assert not result.refused
    assert result.refusal_stage is None


def test_answerer_without_reranker_can_still_refuse_at_llm_layer():
    hits = [make_hit("c1", "勞動基準法", "第 1 條", "不充分內容", score=0.1)]
    llm = FakeLLM(f"{REFUSAL_PHRASE},條文不足。")
    result = Answerer(make_pipeline(hits, reranker=None), llm, refusal_threshold=0.5).answer("問題")
    assert result.refused
    assert result.refusal_stage == "llm"
    assert len(llm.calls) == 1
