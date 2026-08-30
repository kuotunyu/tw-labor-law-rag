"""Composes first-stage retrieval with an optional reranking stage.

Reused by the answerer (Phase 2) and the ablation study (Phase 4): every
setting in the 6-way ablation grid — {vector, hybrid} x rerank on/off, x 2
chunking strategies — is just a different ``RetrievalPipeline`` wiring.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal, TypeAlias

from rag.models import RetrievedChunk
from rag.retrieval.reranker import Reranker, interleave_reranker_rankings
from rag.retrieval.retriever import Retriever

_EMPLOYER_CUES = ("老闆", "主管", "雇主")
_OFF_HOURS_CUES = ("假日", "休假日", "休息日", "例假", "下班後", "非上班時間")
_MESSAGE_CUES = ("群組傳訊", "群組訊息", "傳訊", "回訊", "line群組", "line訊息")
_OFF_HOURS_LEGAL_TERMS = "雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
_SEVERANCE_CUES = ("資遣", "資遣費", "severance", "termination package")
_NEW_REGIME_CUES = ("勞退新制", "新制")
_OLD_REGIME_CUES = ("勞基法舊制", "舊制")
_SEVERANCE_CALC_CUES = ("試算", "計算", "公式", "formula", "上限", "年資")
_SEVERANCE_LEGAL_TERMS = "資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月"
_OLD_REGIME_SEVERANCE_RERANK_VIEW = (
    "勞基法舊制 資遣費 每滿一年 一個月平均工資 未滿一年 比例計給"
)
_MAX_CANDIDATE_POOL = 20
_WAGE_NONPAYMENT_CUES = (
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
)
_WORKER_IMMEDIATE_TERMINATION_CUES = (
    "直接離職",
    "立即離職",
    "馬上離職",
    "立刻離職",
    "立即終止",
    "直接終止",
    "直接 resign",
    "immediately resign",
    "resign without notice",
)
_WAGE_ARREARS_LEGAL_TERMS = (
    "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
    "勞工得不經預告終止契約"
)

RouteName: TypeAlias = Literal[
    "off_hours_employer_message",
    "severance_comparison",
    "wage_arrears_termination",
]


def _matches_all(folded: str, *cue_groups: tuple[str, ...]) -> bool:
    return all(any(cue in folded for cue in cues) for cues in cue_groups)


@dataclass(frozen=True)
class QueryPlan:
    search_query: str
    routes: tuple[RouteName, ...]
    rerank_only_views: tuple[str, ...] = ()


def plan_retrieval_query(query: str) -> QueryPlan:
    """Bridge measured colloquial gaps without spending a provider call.

    The original question remains intact for the answer-generation layer;
    only retrieval and reranking see the appended legal terms.
    """
    folded = query.casefold()
    expansions: list[str] = []
    routes: list[RouteName] = []
    if _matches_all(folded, _EMPLOYER_CUES, _OFF_HOURS_CUES, _MESSAGE_CUES):
        expansions.append(_OFF_HOURS_LEGAL_TERMS)
        routes.append("off_hours_employer_message")
    if _matches_all(
        folded,
        _SEVERANCE_CUES,
        _NEW_REGIME_CUES,
        _OLD_REGIME_CUES,
        _SEVERANCE_CALC_CUES,
    ):
        expansions.append(_SEVERANCE_LEGAL_TERMS)
        routes.append("severance_comparison")
    if _matches_all(folded, _WAGE_NONPAYMENT_CUES, _WORKER_IMMEDIATE_TERMINATION_CUES):
        expansions.append(_WAGE_ARREARS_LEGAL_TERMS)
        routes.append("wage_arrears_termination")
    route_tuple = tuple(routes)
    rerank_only_views = (
        (_OLD_REGIME_SEVERANCE_RERANK_VIEW,)
        if route_tuple == ("severance_comparison",)
        else ()
    )
    return QueryPlan(
        search_query=" ".join((query, *expansions)),
        routes=route_tuple,
        rerank_only_views=rerank_only_views,
    )


def _retrieval_query(query: str) -> str:
    return plan_retrieval_query(query).search_query


@dataclass
class RetrievalResult:
    hits: list[RetrievedChunk]  # final top_k_final, after optional rerank
    candidates: list[RetrievedChunk] = field(default_factory=list)  # pre-rerank top_k_retrieve
    top_score: float = 0.0
    applied_routes: tuple[str, ...] = ()


class RetrievalPipeline:
    def __init__(
        self,
        retriever: Retriever,
        reranker: Reranker | None,
        top_k_retrieve: int,
        top_k_final: int,
    ):
        if top_k_retrieve > _MAX_CANDIDATE_POOL:
            raise ValueError(
                f"top_k_retrieve must not exceed {_MAX_CANDIDATE_POOL} candidates"
            )
        self.retriever = retriever
        self.reranker = reranker
        self.top_k_retrieve = top_k_retrieve
        self.top_k_final = top_k_final

    def run(self, query: str) -> RetrievalResult:
        plan = plan_retrieval_query(query)
        candidates = self.retriever.retrieve(plan.search_query, top_k=self.top_k_retrieve)
        if len(candidates) > _MAX_CANDIDATE_POOL:
            raise ValueError(
                f"candidate pool must not exceed {_MAX_CANDIDATE_POOL} candidates"
            )
        if self.reranker is not None and candidates:
            if plan.rerank_only_views:
                primary_ranking = self.reranker.rerank_all(
                    plan.search_query, candidates
                )
                top_score = primary_ranking[0].score
                secondary_ranking = self.reranker.rerank_all(
                    plan.rerank_only_views[0], candidates
                )
                hits = interleave_reranker_rankings(
                    candidates, primary_ranking, secondary_ranking
                )[: self.top_k_final]
            else:
                hits = self.reranker.rerank(
                    plan.search_query, candidates, top_k=self.top_k_final
                )
                top_score = hits[0].score if hits else 0.0
        elif candidates:
            hits = candidates[: self.top_k_final]
            top_score = hits[0].score if hits else 0.0
        else:
            hits = []
            top_score = 0.0
        return RetrievalResult(
            hits=hits,
            candidates=candidates,
            top_score=top_score,
            applied_routes=plan.routes,
        )
