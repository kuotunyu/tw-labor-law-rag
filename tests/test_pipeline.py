import pytest

from rag.models import RetrievedChunk
from rag.retrieval.pipeline import (
    QueryPlan,
    RetrievalPipeline,
    _retrieval_query,
    plan_retrieval_query,
)


def test_plan_retrieval_query_returns_off_hours_route_and_preserves_question():
    question = "主管在休假日用群組傳訊息要求我處理工作，這算加班嗎？"
    plan = plan_retrieval_query(question)

    assert isinstance(plan, QueryPlan)
    assert plan.routes == ("off_hours_employer_message",)
    assert plan.search_query == (
        f"{question} 雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
    )
    assert question in plan.search_query
    assert "off_hours_employer_message" not in question


def test_plan_retrieval_query_returns_severance_route_for_casefolded_english_cues():
    question = "Please compare SEVERANCE and TERMINATION PACKAGE with 勞退新制勞基法舊制試算。"
    plan = plan_retrieval_query(question)

    assert plan.routes == ("severance_comparison",)
    assert plan.search_query.endswith("資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月")
    assert "severance_comparison" not in question


def test_plan_retrieval_query_returns_wage_arrears_route():
    question = "公司 unpaid salary，我想 immediately resign。"
    plan = plan_retrieval_query(question)

    assert plan.routes == ("wage_arrears_termination",)
    assert plan.search_query.endswith(
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 勞工得不經預告終止契約"
    )


def test_plan_retrieval_query_returns_no_routes_without_complete_cue_group():
    plan = plan_retrieval_query("公司欠薪兩個月，我該怎麼追討？")

    assert plan == QueryPlan(search_query="公司欠薪兩個月，我該怎麼追討？", routes=())


def test_plan_retrieval_query_appends_all_routes_in_existing_order():
    question = (
        "老闆在休假日用群組傳訊，說公司欠薪、也要我直接離職，"
        "還附上勞退新制與勞基法舊制資遣費試算。"
    )
    plan = plan_retrieval_query(question)

    assert plan.routes == (
        "off_hours_employer_message",
        "severance_comparison",
        "wage_arrears_termination",
    )
    assert plan.search_query.startswith(question + " ")
    assert plan.search_query.index("雇主 休息日") < plan.search_query.index("資遣費 勞工退休金條例")
    assert plan.search_query.index("資遣費 勞工退休金條例") < plan.search_query.index("勞動基準法 第十四條")


def test_retrieval_query_compatibility_wrapper_returns_plan_search_query():
    question = "公司欠薪，我想直接離職。"

    assert _retrieval_query(question) == plan_retrieval_query(question).search_query


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
        "公司一直拖欠薪水,我可以不經預告直接離職嗎?這樣還能拿到資遣費嗎?",
        (
            "公司已經兩個月沒有付 salary，我想今天直接 resign 又怕拿不到 "
            "severance；雇主欠薪時能否不經預告終止契約，之後還能請求資遣費嗎？"
        ),
        (
            "公司連續欠薪後要求我照常打卡，還說沒有先交 resignation notice 就拿不到"
            "任何錢；我能否立即終止，並依哪個規則請求資遣費？"
        ),
    ],
)
def test_pipeline_expands_wage_nonpayment_worker_termination(question):
    calls = []
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls).run(question)

    expected = f"{question} {legal_terms}"
    assert calls == [("retrieve", expected), ("rerank", expected)]


@pytest.mark.parametrize(
    "wage_cue",
    [
        "欠薪",
        "沒發薪",
        "沒有發薪",
        "未發薪",
        "沒付薪",
        "沒有付薪",
        "未付薪",
        "拖欠工資",
        "積欠工資",
        "沒付工資",
        "沒有付工資",
        "未付工資",
        "未給付工資",
        "未給付工作報酬",
        "沒有付 salary",
        "沒付 salary",
        "unpaid salary",
        "wage arrears",
    ],
)
def test_pipeline_accepts_each_reviewed_wage_nonpayment_cue(wage_cue):
    calls = []
    question = f"公司{wage_cue}，我想直接離職。"
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", f"{question} {legal_terms}")]


@pytest.mark.parametrize(
    "exit_cue",
    [
        "直接離職",
        "立即離職",
        "馬上離職",
        "立刻離職",
        "立即終止",
        "直接終止",
        "直接 resign",
        "immediately resign",
        "resign without notice",
    ],
)
def test_pipeline_accepts_each_reviewed_worker_exit_cue(exit_cue):
    calls = []
    question = f"公司欠薪，我想{exit_cue}。"
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", f"{question} {legal_terms}")]


@pytest.mark.parametrize(
    "question",
    [
        "公司已經欠薪兩個月，該怎麼追討？",
        "我想直接離職，應該怎麼做？",
        "我是雇主，公司欠薪後可以不經預告直接解僱勞工嗎？",
        "公司薪資怎麼算？我將來可能離職。",
    ],
)
def test_pipeline_requires_wage_and_worker_exit_cue_groups(question):
    calls = []

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", question)]


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
        "老闆在休假日用群組傳訊，說公司欠薪、也要我直接離職，"
        "還附上勞退新制與勞基法舊制資遣費試算。"
    )
    off_hours = "雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
    severance = "資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月"
    wage_arrears = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )
    expected = f"{question} {off_hours} {severance} {wage_arrears}"

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
