import pytest

from ui.refusal_labels import refusal_stage_label


@pytest.mark.parametrize(
    ("stage", "expected"),
    [
        ("no_hits", "檢索無候選結果(未呼叫 LLM)"),
        ("threshold", "相關度門檻拒答(未呼叫 LLM)"),
        ("llm", "LLM 判定條文依據不足"),
    ],
)
def test_refusal_stage_label_translates_known_stages(stage, expected):
    assert refusal_stage_label(stage, refused=True) == expected


def test_refusal_stage_label_handles_answered_and_legacy_payloads():
    assert refusal_stage_label(None, refused=False) == "未拒答(正常作答)"
    assert refusal_stage_label(None, refused=True) == "舊版 API 未提供"


def test_refusal_stage_label_is_forward_compatible():
    assert refusal_stage_label("future", refused=True) == "未知拒答階段(future)"
