import json
from pathlib import Path

from ui.content import (
    BYOK_PRIVACY_POINTS,
    EXAMPLE_QUESTIONS,
    KNOWLEDGE_BASE,
    examples_by_category,
)

PROJECT_ROOT = Path(__file__).parents[1]


def test_public_knowledge_summary_tracks_the_release_snapshot():
    snapshot = json.loads(
        (PROJECT_ROOT / "release/corpus_snapshot.json").read_text(encoding="utf-8")
    )

    assert KNOWLEDGE_BASE.snapshot_date == snapshot["snapshot_date"]
    assert KNOWLEDGE_BASE.laws == snapshot["law_count"] == 15
    assert KNOWLEDGE_BASE.articles == snapshot["article_count"] == 884


def test_examples_are_unique_grouped_and_cover_the_demo_journey():
    grouped = examples_by_category()

    assert len(EXAMPLE_QUESTIONS) == 5
    assert set(grouped) == {"工時", "請假", "離職與欠薪", "資遣費", "安全拒答"}
    assert tuple(item for items in grouped.values() for item in items) == EXAMPLE_QUESTIONS
    assert len({item.id for item in EXAMPLE_QUESTIONS}) == len(EXAMPLE_QUESTIONS)
    assert all(item.title.strip() and item.question.strip() for item in EXAMPLE_QUESTIONS)


def test_byok_privacy_points_explain_storage_and_cost_ownership():
    assert BYOK_PRIVACY_POINTS == (
        "只保留在目前瀏覽器工作階段",
        "不寫入檔案或聊天紀錄",
        "模型費用由 API Key 持有人承擔",
    )
