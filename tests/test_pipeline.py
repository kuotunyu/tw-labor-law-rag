import hashlib
import json
from pathlib import Path

import pytest

from rag.models import RetrievedChunk
from rag.retrieval import reranker as reranker_module
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


def test_plan_retrieval_query_exposes_old_regime_view_only_for_exact_severance_route():
    severance_question = (
        "請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。"
    )
    collision_question = (
        "老闆在休假日用群組傳訊，說公司欠薪、也要我直接離職，"
        "還附上勞退新制與勞基法舊制資遣費試算。"
    )

    exact_plan = plan_retrieval_query(severance_question)
    collision_plan = plan_retrieval_query(collision_question)
    unrelated_plan = plan_retrieval_query("公司欠薪兩個月，我該怎麼追討？")

    assert exact_plan.rerank_only_views == (
        "勞基法舊制 資遣費 每滿一年 一個月平均工資 未滿一年 比例計給",
    )
    assert collision_plan.rerank_only_views == ()
    assert unrelated_plan.rerank_only_views == ()


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


def test_interleave_reranker_rankings_returns_empty_for_three_empty_inputs():
    assert reranker_module.interleave_reranker_rankings([], [], []) == []


def test_interleave_reranker_rankings_is_primary_first_and_keeps_primary_scores():
    candidates = [hit("a"), hit("b"), hit("c"), hit("d")]
    primary = [hit("c", 0.91), hit("b", 0.82), hit("a", 0.74), hit("d", 0.69)]
    secondary = [hit("b", 0.05), hit("d", 0.04), hit("a", 0.03), hit("c", 0.02)]

    merged = reranker_module.interleave_reranker_rankings(candidates, primary, secondary)

    assert [item.payload["chunk_id"] for item in merged] == ["c", "b", "d", "a"]
    assert [item.score for item in merged] == [0.91, 0.82, 0.69, 0.74]


def test_interleave_reranker_rankings_is_deterministic_despite_cross_ranking_repeats():
    candidates = [hit("a"), hit("b"), hit("c")]
    primary = [hit("b", 0.9), hit("a", 0.8), hit("c", 0.7)]
    secondary = [hit("c", 0.3), hit("b", 0.2), hit("a", 0.1)]

    first = reranker_module.interleave_reranker_rankings(candidates, primary, secondary)
    second = reranker_module.interleave_reranker_rankings(candidates, primary, secondary)

    assert [item.payload["chunk_id"] for item in first] == ["b", "c", "a"]
    assert second == first


@pytest.mark.parametrize(
    ("candidates", "primary", "secondary"),
    [
        ([hit("a"), hit("a")], [hit("a"), hit("a")], [hit("a"), hit("a")]),
        ([hit(" ")], [hit(" ")], [hit(" ")]),
        ([hit("a"), hit("b")], [hit("a"), hit("a")], [hit("a"), hit("b")]),
        ([hit("a"), hit("b")], [hit("a"), hit("b")], [hit("a"), hit("a")]),
        ([hit("a"), hit("b")], [hit("a"), hit("b")], [hit("a"), hit("foreign")]),
        ([hit("a"), hit("b")], [hit("a")], [hit("a"), hit("b")]),
        ([hit("a"), hit("b")], [hit("a"), hit("b")], [hit("a")]),
    ],
)
def test_interleave_reranker_rankings_rejects_noncanonical_or_incomplete_permutations(
    candidates, primary, secondary
):
    with pytest.raises(ValueError, match="candidate IDs|ranking"):
        reranker_module.interleave_reranker_rankings(candidates, primary, secondary)


def test_interleave_reranker_rankings_rejects_non_string_chunk_ids_as_unstable():
    unstable = hit(["a"])

    with pytest.raises(ValueError, match="candidate IDs"):
        reranker_module.interleave_reranker_rankings([unstable], [unstable], [unstable])


def test_rerank_all_rejects_a_score_count_mismatch_without_loading_a_model():
    reranker = object.__new__(reranker_module.Reranker)
    reranker._model = type(
        "ScoreMismatchModel",
        (),
        {"compute_score": staticmethod(lambda _pairs, normalize: [0.9])},
    )()

    with pytest.raises(ValueError, match="score count"):
        reranker.rerank_all("query", [hit("a"), hit("b")])


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

        def rerank_all(self, query, candidates):
            calls.append(("rerank", query))
            return candidates

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


def test_pipeline_exact_severance_route_reranks_full_pool_twice_then_merges_top_five():
    question = "請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。"
    candidates = [hit(f"cached-{index}") for index in range(6)]
    primary = [
        hit("cached-0", 0.021),
        hit("cached-1", 0.020),
        hit("cached-2", 0.019),
        hit("cached-3", 0.018),
        hit("cached-4", 0.017),
        hit("cached-5", 0.016),
    ]
    secondary = [
        hit("cached-5", 0.99),
        hit("cached-0", 0.98),
        hit("cached-1", 0.97),
        hit("cached-2", 0.96),
        hit("cached-3", 0.95),
        hit("cached-4", 0.94),
    ]
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return candidates[:top_k]

    class RecordingReranker:
        def rerank_all(self, query, received_candidates):
            calls.append(("rerank", query, len(received_candidates)))
            assert received_candidates == candidates
            return primary if query.endswith("六個月") else secondary

        def rerank(self, query, received_candidates, top_k):
            raise AssertionError("exact severance route must use full-pool reranking")

    result = RetrievalPipeline(
        RecordingRetriever(),
        reranker=RecordingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    ).run(question)

    planned = plan_retrieval_query(question)
    assert calls == [
        ("retrieve", planned.search_query, 20),
        ("rerank", planned.search_query, 6),
        ("rerank", planned.rerank_only_views[0], 6),
    ]
    assert [item.payload["chunk_id"] for item in result.hits] == [
        "cached-0",
        "cached-5",
        "cached-1",
        "cached-2",
        "cached-3",
    ]
    assert result.top_score == 0.021
    assert len(result.candidates) == 6


def test_pipeline_non_exact_route_retains_single_reranker_call():
    question = (
        "老闆在休假日用群組傳訊，說公司欠薪、也要我直接離職，"
        "還附上勞退新制與勞基法舊制資遣費試算。"
    )
    candidates = [hit(f"candidate-{index}") for index in range(3)]
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return candidates

    class RecordingReranker:
        def rerank(self, query, received_candidates, top_k):
            calls.append(("rerank", query, len(received_candidates), top_k))
            return received_candidates[:top_k]

        def rerank_all(self, _query, _received_candidates):
            raise AssertionError("non-exact routes must retain the single-reranker path")

    result = RetrievalPipeline(
        RecordingRetriever(),
        reranker=RecordingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    ).run(question)

    assert len(calls) == 2
    assert calls[0][0] == "retrieve"
    assert calls[1] == ("rerank", calls[0][1], 3, 5)
    assert result.applied_routes == (
        "off_hours_employer_message",
        "severance_comparison",
        "wage_arrears_termination",
    )


def test_pipeline_empty_candidates_makes_no_reranker_call():
    calls = []

    class EmptyRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return []

    class CountingReranker:
        def rerank(self, *_args, **_kwargs):
            calls.append(("rerank",))
            return []

        def rerank_all(self, *_args, **_kwargs):
            calls.append(("rerank_all",))
            return []

    result = RetrievalPipeline(
        EmptyRetriever(),
        reranker=CountingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    ).run("請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。")

    assert result.hits == []
    assert result.top_score == 0.0
    assert [call[0] for call in calls] == ["retrieve"]


def test_pipeline_allows_exactly_twenty_candidates_for_each_exact_route_rerank():
    question = "請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。"
    candidates = [hit(f"candidate-{index}") for index in range(20)]
    calls = []

    class TwentyCandidateRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return candidates

    class CountingReranker:
        def rerank_all(self, query, received_candidates):
            calls.append(("rerank", query, len(received_candidates)))
            return received_candidates

    result = RetrievalPipeline(
        TwentyCandidateRetriever(),
        reranker=CountingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    ).run(question)

    assert len(result.hits) == 5
    assert [call[0] for call in calls] == ["retrieve", "rerank", "rerank"]
    assert [call[2] for call in calls[1:]] == [20, 20]


def test_pipeline_rejects_an_over_twenty_candidate_configuration_before_retrieval():
    calls = []

    class NeverCalledRetriever:
        def retrieve(self, *_args, **_kwargs):
            calls.append("retrieve")
            return []

    with pytest.raises(ValueError, match="top_k_retrieve.*20"):
        RetrievalPipeline(
            NeverCalledRetriever(), reranker=None, top_k_retrieve=21, top_k_final=5
        )

    assert calls == []


def test_pipeline_rejects_a_retriever_pool_over_twenty_before_reranking():
    calls = []
    candidates = [hit(f"candidate-{index}") for index in range(21)]

    class OversizedRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return candidates

    class NeverCalledReranker:
        def rerank_all(self, *_args, **_kwargs):
            calls.append(("rerank_all",))
            return candidates

        def rerank(self, *_args, **_kwargs):
            calls.append(("rerank",))
            return candidates

    pipeline = RetrievalPipeline(
        OversizedRetriever(),
        reranker=NeverCalledReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    )

    with pytest.raises(ValueError, match="candidate pool.*20"):
        pipeline.run("請比較勞退新制與勞基法舊制的資遣費計算公式、工作年資及最高上限。")

    assert [call[0] for call in calls] == ["retrieve"]


_STAGE_REPLAY_PATH = (
    Path(__file__).parent / "fixtures" / "v036_severance_retrieval_stage_replay.json"
)
_STAGE_REPLAY = json.loads(_STAGE_REPLAY_PATH.read_text(encoding="utf-8"))


def _replay_authority(chunk_id, authority_chunk_ids):
    if chunk_id == authority_chunk_ids["new"]:
        law, article = "勞工退休金條例", "第 12 條"
    elif chunk_id == authority_chunk_ids["old"]:
        law, article = "勞動基準法", "第 17 條"
    else:
        law, article = "fixture", ""
    return RetrievedChunk(
        score=1.0,
        payload={
            "chunk_id": chunk_id,
            "text": "offline stage replay chunk",
            "doc_title": law,
            "articles": [article] if article else [],
        },
    )


def _ranks_for_authorities(items, authority_chunk_ids):
    ids = [item.payload["chunk_id"] for item in items]
    return {
        authority: ids.index(chunk_id) + 1
        for authority, chunk_id in authority_chunk_ids.items()
    }


@pytest.mark.parametrize("case", _STAGE_REPLAY["cases"], ids=lambda case: case["qid"])
def test_pipeline_replays_provenance_bound_severance_stage_evidence(case):
    provenance = _STAGE_REPLAY["provenance"]
    dataset = Path(__file__).parents[1] / provenance["current_dataset"]
    source_artifact = Path(__file__).parents[1] / provenance["source_artifact"]
    assert hashlib.sha256(dataset.read_bytes()).hexdigest() == provenance[
        "current_dataset_sha256"
    ]
    assert hashlib.sha256(source_artifact.read_bytes()).hexdigest() == provenance[
        "source_artifact_sha256"
    ]
    assert _STAGE_REPLAY["fixture_kind"] == "deterministic_stage_replay"
    assert _STAGE_REPLAY["purpose"].startswith("Offline pipeline integration")

    questions = {
        json.loads(line)["qid"]: json.loads(line)["question"]
        for line in dataset.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    question = questions[case["qid"]]
    candidates = [
        _replay_authority(chunk_id, case["authority_chunk_ids"])
        for chunk_id in case["candidate_pool"]
    ]
    by_id = {item.payload["chunk_id"]: item for item in candidates}
    primary = [
        RetrievedChunk(score=1.0 - index / 100, payload=by_id[chunk_id].payload)
        for index, chunk_id in enumerate(case["primary_ranking"])
    ]
    secondary = [
        RetrievedChunk(score=0.5 - index / 100, payload=by_id[chunk_id].payload)
        for index, chunk_id in enumerate(case["secondary_ranking"])
    ]
    calls = []

    class ReplayRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query, top_k))
            return candidates[:top_k]

    class ReplayReranker:
        def rerank_all(self, query, received_candidates):
            calls.append(("rerank", query, len(received_candidates)))
            assert received_candidates == candidates
            return primary if query == plan_retrieval_query(question).search_query else secondary

    result = RetrievalPipeline(
        ReplayRetriever(), ReplayReranker(), top_k_retrieve=20, top_k_final=5
    ).run(question)

    stage_ranks = {
        "candidate_pool": _ranks_for_authorities(
            result.candidates, case["authority_chunk_ids"]
        ),
        "primary_ranking": _ranks_for_authorities(
            primary, case["authority_chunk_ids"]
        ),
        "secondary_ranking": _ranks_for_authorities(
            secondary, case["authority_chunk_ids"]
        ),
        "final_top_five": _ranks_for_authorities(
            result.hits, case["authority_chunk_ids"]
        ),
    }

    assert case["qid"] not in " ".join(call[1] for call in calls)
    assert "第 12 條" not in " ".join(call[1] for call in calls)
    assert "第 17 條" not in " ".join(call[1] for call in calls)
    assert stage_ranks == case["expected_stage_ranks"]
    assert all(rank <= 5 for rank in stage_ranks["final_top_five"].values())
    assert calls[0][0] == "retrieve"
    assert [call[0] for call in calls[1:]] == ["rerank", "rerank"]
    assert all(call[2] <= 20 for call in calls[1:])


def test_pipeline_uses_one_planned_query_and_exposes_only_applied_routes():
    question = "公司 unpaid salary，我想 immediately resign。"
    expected_query = (
        f"{question} 勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )
    calls = []

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            calls.append(("retrieve", query))
            return [hit("labor-law")]

    class RecordingReranker:
        def rerank(self, query, candidates, top_k):
            calls.append(("rerank", query))
            return candidates[:top_k]

    result = RetrievalPipeline(
        RecordingRetriever(),
        reranker=RecordingReranker(),
        top_k_retrieve=20,
        top_k_final=5,
    ).run(question)

    assert calls == [("retrieve", expected_query), ("rerank", expected_query)]
    assert result.applied_routes == ("wage_arrears_termination",)
    assert question in expected_query
    assert "wage_arrears_termination" not in expected_query


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

    assert calls == [
        ("retrieve", expected),
        ("rerank", expected),
        ("rerank", "勞基法舊制 資遣費 每滿一年 一個月平均工資 未滿一年 比例計給"),
    ]


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
