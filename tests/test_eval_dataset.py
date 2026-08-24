import json
from pathlib import Path

import pytest

DATASET_DIR = Path(__file__).resolve().parents[1] / "eval" / "dataset"

REQUIRED_FIELDS = {"qid", "question", "answer", "sources", "answerable", "q_type"}


def load_dataset(name: str) -> list[dict]:
    path = DATASET_DIR / name
    if not path.exists():
        pytest.skip(f"{name} not built yet")
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


@pytest.mark.parametrize("name", ["mini_eval.jsonl", "eval_set.jsonl"])
def test_dataset_schema(name):
    rows = load_dataset(name)
    assert rows, f"{name} is empty"

    qids = [r["qid"] for r in rows]
    assert len(qids) == len(set(qids)), "duplicate qids"

    for r in rows:
        assert REQUIRED_FIELDS <= r.keys(), f"{r.get('qid')}: missing fields"
        assert isinstance(r["answerable"], bool)
        if r["answerable"]:
            assert r["sources"], f"{r['qid']}: answerable question needs ground-truth sources"
            for src in r["sources"]:
                assert src["doc"] and src["article"], f"{r['qid']}: incomplete source"
        else:
            assert r["sources"] == [], f"{r['qid']}: unanswerable question must have no sources"


def test_mini_eval_composition():
    rows = load_dataset("mini_eval.jsonl")
    assert len(rows) == 10
    unanswerable = [r for r in rows if not r["answerable"]]
    assert len(unanswerable) == 2


def test_eval_set_composition():
    rows = load_dataset("eval_set.jsonl")
    assert len(rows) == 40
    unanswerable = [r for r in rows if not r["answerable"]]
    assert len(unanswerable) == 10
    related = [r for r in unanswerable if r["q_type"] == "out_of_kb_related"]
    assert len(related) == 5, "need 5 hard (related but out-of-KB) refusal cases"
    # Every law in the corpus should be exercised by at least one answerable question.
    docs = {src["doc"] for r in rows if r["answerable"] for src in r["sources"]}
    assert len(docs) == 15, f"expected all 15 laws covered, got {len(docs)}: {sorted(docs)}"
