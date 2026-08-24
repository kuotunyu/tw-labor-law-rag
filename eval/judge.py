"""LLM-as-judge for generation quality, with a Traditional-Chinese rubric.

Hand-rolled instead of RAGAS on purpose: RAGAS's built-in prompts are
English-first and hard to adapt to Traditional-Chinese legal text, and the two
metrics we need (faithfulness / answer relevancy) are simple enough that a
transparent implementation is easier to reason about — and to defend.

One judge call scores both metrics (halves the API-call count, which matters
on free-tier quotas). Retries with exponential backoff on rate limits.
"""

from __future__ import annotations

import json
import re
import time

from lib import is_rate_limit

_JUDGE_SYSTEM = "你是嚴格的評審,負責評估法規問答系統的回答品質。只輸出 JSON,不要輸出任何其他文字。"

_JUDGE_TEMPLATE = """請根據提供的法規條文(context),評估系統回答的兩個指標,各給 1-5 的整數分:

## faithfulness(忠實度)
回答中的事實陳述是否都能被 context 支持?
- 5:所有陳述都有條文依據,沒有捏造或引用條文以外的知識
- 4:幾乎所有陳述有依據,僅有極輕微的措辭延伸
- 3:大致有依據,但有少量無法由條文驗證或過度推論的內容
- 2:有明顯無依據的陳述或數字錯誤
- 1:大量內容缺乏條文依據,或與條文矛盾

## relevancy(切題度)
回答是否直接、完整地回應了問題?
- 5:直接回答問題核心,涵蓋問題的所有子問題,沒有答非所問
- 4:回答了問題核心與大部分子問題
- 3:回答了主要問題,但遺漏子問題或夾雜較多無關內容
- 2:僅部分觸及問題,大半答非所問
- 1:答非所問,幾乎沒有回應問題

## 問題
{question}

## 法規條文(context)
{context}

## 系統回答
{answer}

輸出格式(只輸出這個 JSON,不要加 markdown 標記):
{{"faithfulness": <1-5>, "faithfulness_reason": "<一句話理由>", "relevancy": <1-5>, "relevancy_reason": "<一句話理由>"}}"""

_REQUIRED_KEYS = ("faithfulness", "relevancy")
_JSON_BLOCK = re.compile(r"\{.*\}", re.DOTALL)


def parse_judge_output(raw: str) -> dict:
    """Extract and validate the judge's JSON verdict; raises ValueError if unusable."""
    match = _JSON_BLOCK.search(raw)
    if not match:
        raise ValueError(f"no JSON object in judge output: {raw[:200]!r}")
    try:
        data = json.loads(match.group(0))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in judge output: {exc}") from exc

    for key in _REQUIRED_KEYS:
        if key not in data:
            raise ValueError(f"judge output missing key {key!r}: {data}")
        score = data[key]
        if not isinstance(score, int) or not 1 <= score <= 5:
            raise ValueError(f"judge score {key}={score!r} outside 1-5")
    return data


class Judge:
    def __init__(self, llm, max_retries: int = 4, backoff_base: float = 10.0):
        self.llm = llm
        self.max_retries = max_retries
        self.backoff_base = backoff_base

    def score(self, question: str, context: str, answer: str) -> dict:
        prompt = _JUDGE_TEMPLATE.format(question=question, context=context, answer=answer)
        last_error: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                # Generous budget: reasoning models spend hidden tokens before
                # emitting the (small) JSON verdict.
                raw = self.llm.generate(_JUDGE_SYSTEM, prompt, temperature=0.0, max_tokens=2000)
                return parse_judge_output(raw)
            except ValueError as exc:  # malformed verdict — retry without sleeping
                last_error = exc
            except Exception as exc:
                if not is_rate_limit(exc):
                    raise
                last_error = exc
                time.sleep(self.backoff_base * (2**attempt))
        raise RuntimeError(f"judge failed after {self.max_retries} attempts: {last_error}")
