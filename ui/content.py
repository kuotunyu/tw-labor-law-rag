"""Stable presentation content for the public portfolio interface."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeBaseSummary:
    """Public, non-secret facts about the audited knowledge base."""

    snapshot_date: str
    laws: int
    articles: int


@dataclass(frozen=True)
class ExampleQuestion:
    """One curated question that still uses the production query path."""

    id: str
    category: str
    title: str
    question: str


KNOWLEDGE_BASE = KnowledgeBaseSummary("2026-08-29", 15, 884)

BYOK_PRIVACY_POINTS = (
    "只保留在目前瀏覽器工作階段",
    "不寫入檔案或聊天紀錄",
    "模型費用由 API Key 持有人承擔",
)

EXAMPLE_QUESTIONS = (
    ExampleQuestion(
        "hours",
        "工時",
        "每日與每週工時",
        "勞工每天和每週的正常工作時間上限是多少？",
    ),
    ExampleQuestion(
        "sick-leave",
        "請假",
        "普通傷病假",
        "一年最多可以請幾天病假？請病假薪水怎麼算？",
    ),
    ExampleQuestion(
        "wage-arrears",
        "離職與欠薪",
        "欠薪立即離職",
        "公司一直拖欠薪水，我可以不經預告直接離職嗎？這樣還能拿到資遣費嗎？",
    ),
    ExampleQuestion(
        "severance",
        "資遣費",
        "新舊制比較",
        "適用勞退新制的勞工被資遣時，資遣費怎麼計算？和舊制有什麼不同？",
    ),
    ExampleQuestion(
        "refusal",
        "安全拒答",
        "知識庫外問題",
        "著作權的保護期間是幾年？",
    ),
)


def examples_by_category() -> dict[str, tuple[ExampleQuestion, ...]]:
    """Group examples by category while preserving their display order."""
    grouped: dict[str, list[ExampleQuestion]] = {}
    for item in EXAMPLE_QUESTIONS:
        grouped.setdefault(item.category, []).append(item)
    return {category: tuple(items) for category, items in grouped.items()}
