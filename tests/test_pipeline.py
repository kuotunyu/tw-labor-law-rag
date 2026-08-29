import pytest

from rag.models import RetrievedChunk
from rag.retrieval.pipeline import RetrievalPipeline


def hit(chunk_id, score=1.0):
    return RetrievedChunk(score=score, payload={"chunk_id": chunk_id, "text": chunk_id})


class FakeRetriever:
    def __init__(self, hits):
        self.hits = hits

    def retrieve(self, query, top_k):
        return self.hits[:top_k]


class FakeReranker:
    """Reverses order, deterministically, so tests can tell rerank ran."""

    def rerank(self, query, candidates, top_k):
        return list(reversed(candidates))[:top_k]


def recorded_pipeline(calls, *, rerank=True):
    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query))
            return [hit("labor-law")]

    class RecordingReranker:
        def rerank(self, query, candidates, top_k):
            calls.append(("rerank", query))
            return candidates[:top_k]

    return RetrievalPipeline(
        RecordingRetriever(),
        reranker=RecordingReranker() if rerank else None,
        top_k_retrieve=20,
        top_k_final=5,
    )


def test_pipeline_without_reranker_slices_candidates():
    hits = [hit("a", 0.9), hit("b", 0.8), hit("c", 0.7)]
    pipeline = RetrievalPipeline(FakeRetriever(hits), reranker=None, top_k_retrieve=20, top_k_final=2)
    result = pipeline.run("query")
    assert [h.payload["chunk_id"] for h in result.hits] == ["a", "b"]
    assert result.top_score == 0.9
    assert len(result.candidates) == 3


def test_pipeline_with_reranker_reorders():
    hits = [hit("a"), hit("b"), hit("c")]
    pipeline = RetrievalPipeline(FakeRetriever(hits), reranker=FakeReranker(), top_k_retrieve=20, top_k_final=2)
    result = pipeline.run("query")
    assert [h.payload["chunk_id"] for h in result.hits] == ["c", "b"]


def test_pipeline_empty_candidates():
    pipeline = RetrievalPipeline(FakeRetriever([]), reranker=None, top_k_retrieve=20, top_k_final=5)
    result = pipeline.run("query")
    assert result.hits == []
    assert result.top_score == 0.0


def test_pipeline_passes_top_k_retrieve_to_retriever():
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(top_k)
            return []

    pipeline = RetrievalPipeline(RecordingRetriever(), reranker=None, top_k_retrieve=20, top_k_final=5)
    pipeline.run("query")
    assert calls == [20]


def test_pipeline_expands_off_hours_employer_messages_for_legal_retrieval():
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query))
            return [hit("labor-law")]

    class RecordingReranker:
        def rerank(self, query, candidates, top_k):
            calls.append(("rerank", query))
            return candidates[:top_k]

    question = "主管在休假日用群組傳訊息要求我處理工作，這算加班嗎？"
    expected = f"{question} 雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
    pipeline = RetrievalPipeline(
        RecordingRetriever(),
        reranker=RecordingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    )

    pipeline.run(question)

    assert calls == [("retrieve", expected), ("rerank", expected)]


@pytest.mark.parametrize(
    "question",
    [
        "我同時有勞退新制與勞基法舊制年資，termination package 試算結果差很多；請逐項說明兩種 severance formula 與新制最高上限。",
        "請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。",
    ],
)
def test_pipeline_expands_new_and_old_regime_severance_calculations(question):
    calls = []
    legal_terms = "資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月"
    expected = f"{question} {legal_terms}"

    recorded_pipeline(calls).run(question)

    assert calls == [("retrieve", expected), ("rerank", expected)]


@pytest.mark.parametrize(
    "question",
    [
        "公司終止勞動契約時要注意什麼？",
        "被資遣後可以申請哪些給付？",
        "勞退新制與舊制有什麼差別？",
        "termination package 的新舊軟體版本請試算授權成本。",
    ],
)
def test_pipeline_requires_all_severance_comparison_cue_groups(question):
    calls = []

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", question)]


def test_pipeline_appends_each_matching_rule_once_in_stable_order():
    calls = []
    question = (
        "老闆在休假日用群組傳訊說明我的勞退新制與勞基法舊制資遣費試算，"
        "要我立刻確認。"
    )
    off_hours = "雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
    severance = "資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月"
    expected = f"{question} {off_hours} {severance}"

    recorded_pipeline(calls).run(question)

    assert calls == [("retrieve", expected), ("rerank", expected)]


def test_pipeline_keeps_unrelated_queries_unchanged():
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(query)
            return []

    question = "量子力學和相對論有什麼不同？"
    pipeline = RetrievalPipeline(
        RecordingRetriever(), reranker=None, top_k_retrieve=20, top_k_final=5
    )

    pipeline.run(question)

    assert calls == [question]


@pytest.mark.parametrize(
    "question",
    [
        "老闆假日請大家參加聚餐。",
        "假日群組傳訊息討論朋友聚餐。",
        "老闆在群組傳訊息公布午餐地點。",
    ],
)
def test_pipeline_requires_all_three_off_hours_message_cue_groups(question):
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(query)
            return []

    pipeline = RetrievalPipeline(
        RecordingRetriever(), reranker=None, top_k_retrieve=20, top_k_final=5
    )

    pipeline.run(question)

    assert calls == [question]
