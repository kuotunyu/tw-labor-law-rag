import sys
from pathlib import Path

import pytest

from rag.generation.llm import LLMOutput

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "eval"))

from judge import Judge, parse_judge_output  # noqa: E402

VALID = '{"faithfulness": 5, "faithfulness_reason": "皆有依據", "relevancy": 4, "relevancy_reason": "涵蓋主要問題"}'


def test_parse_valid_json():
    data = parse_judge_output(VALID)
    assert data["faithfulness"] == 5
    assert data["relevancy"] == 4


def test_parse_json_wrapped_in_markdown_fence():
    raw = f"```json\n{VALID}\n```"
    assert parse_judge_output(raw)["faithfulness"] == 5


def test_parse_json_with_surrounding_prose():
    raw = f"好的,以下是我的評分:\n{VALID}\n希望有幫助。"
    assert parse_judge_output(raw)["relevancy"] == 4


def test_parse_rejects_out_of_range_score():
    with pytest.raises(ValueError):
        parse_judge_output('{"faithfulness": 7, "relevancy": 3}')


def test_parse_rejects_missing_key():
    with pytest.raises(ValueError):
        parse_judge_output('{"faithfulness": 4}')


def test_parse_rejects_non_json():
    with pytest.raises(ValueError):
        parse_judge_output("回答品質很好,給五分。")


class FakeLLM:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = 0

    def generate(self, system, user, temperature=0.0, max_tokens=1024):
        self.calls += 1
        item = self.responses.pop(0)
        if isinstance(item, Exception):
            raise item
        return LLMOutput(text=item, provider="gemini", model="gemini-test")


def test_judge_scores_happy_path():
    judge = Judge(FakeLLM([VALID]))
    verdict = judge.score("問題", "條文", "回答")
    assert verdict["faithfulness"] == 5


def test_judge_retries_rate_limit_then_succeeds():
    llm = FakeLLM([RuntimeError("429 RESOURCE_EXHAUSTED"), VALID])
    judge = Judge(llm, backoff_base=0.0)
    verdict = judge.score("問題", "條文", "回答")
    assert verdict["relevancy"] == 4
    assert llm.calls == 2


def test_judge_retries_malformed_output():
    llm = FakeLLM(["這不是 JSON", VALID])
    judge = Judge(llm, backoff_base=0.0)
    assert judge.score("問題", "條文", "回答")["faithfulness"] == 5


def test_judge_raises_on_non_rate_limit_error():
    llm = FakeLLM([RuntimeError("401 invalid api key")])
    judge = Judge(llm, backoff_base=0.0)
    with pytest.raises(RuntimeError, match="401"):
        judge.score("問題", "條文", "回答")


def test_judge_gives_up_after_max_retries():
    llm = FakeLLM([RuntimeError("429")] * 3)
    judge = Judge(llm, max_retries=3, backoff_base=0.0)
    with pytest.raises(RuntimeError, match="judge failed"):
        judge.score("問題", "條文", "回答")
