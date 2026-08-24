"""Human-readable labels for refusal-stage values returned by the API."""

from __future__ import annotations

_REFUSAL_STAGE_LABELS = {
    "no_hits": "檢索無候選結果(未呼叫 LLM)",
    "threshold": "相關度門檻拒答(未呼叫 LLM)",
    "llm": "LLM 判定條文依據不足",
}


def refusal_stage_label(stage: str | None, *, refused: bool) -> str:
    """Translate an API stage while remaining tolerant of old/new backends."""

    if stage is None:
        return "舊版 API 未提供" if refused else "未拒答(正常作答)"
    return _REFUSAL_STAGE_LABELS.get(stage, f"未知拒答階段({stage})")
