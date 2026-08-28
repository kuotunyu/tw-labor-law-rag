import hashlib
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_DIR = PROJECT_ROOT / "eval" / "dataset"
FORMAL_PATH = DATASET_DIR / "eval_set.jsonl"
MINI_PATH = DATASET_DIR / "mini_eval.jsonl"
STRESS_PATH = DATASET_DIR / "reliability_stress_v0.3.1.jsonl"
SNAPSHOT_PATH = PROJECT_ROOT / "release" / "corpus_snapshot.json"
FORMAL_SHA256 = "760e33eaa0821001d37ff974bc037043d019fc670b8f3621b6e713030274ca07"
REQUIRED_FIELDS = {
    "qid",
    "base_qid",
    "question",
    "answer",
    "sources",
    "answerable",
    "q_type",
    "style_tags",
}


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_reliability_stress_shape_and_diversity():
    rows = load_jsonl(STRESS_PATH)

    assert len(rows) == 60
    assert [row["qid"] for row in rows] == [f"stress-{index:03d}" for index in range(1, 61)]
    assert all(REQUIRED_FIELDS <= row.keys() for row in rows)
    assert len({row["qid"] for row in rows}) == 60
    assert len({row["question"] for row in rows}) == 60
    assert sum(row["answerable"] for row in rows) == 40
    assert sum(row["q_type"] == "out_of_kb_related" for row in rows) == 10
    assert sum(row["q_type"] == "out_of_kb_unrelated" for row in rows) == 10
    assert sum(len(row["question"]) >= 40 for row in rows) >= 30
    assert sum("code_switch" in row["style_tags"] for row in rows) >= 15
    assert sum("narrative" in row["style_tags"] for row in rows) >= 15
    assert all(row["style_tags"] for row in rows)


def test_reliability_stress_reuses_audited_ground_truth():
    rows = load_jsonl(STRESS_PATH)
    bases = load_jsonl(FORMAL_PATH) + load_jsonl(MINI_PATH)
    base_by_qid = {row["qid"]: row for row in bases}
    valid_pairs = {
        (source["doc"], source["article"])
        for row in bases
        for source in row["sources"]
    }
    target_laws = {
        law["name"]
        for law in json.loads(SNAPSHOT_PATH.read_text(encoding="utf-8"))["laws"]
    }

    covered_laws = set()
    for row in rows:
        assert row["base_qid"] in base_by_qid
        if row["answerable"]:
            assert row["sources"]
            assert base_by_qid[row["base_qid"]]["answerable"] is True
            for source in row["sources"]:
                assert (source["doc"], source["article"]) in valid_pairs
                covered_laws.add(source["doc"])
        else:
            assert row["sources"] == []
            assert base_by_qid[row["base_qid"]]["answerable"] is False

    assert covered_laws == target_laws


def test_reliability_stress_has_no_secrets_or_direct_identifiers():
    raw = STRESS_PATH.read_text(encoding="utf-8")

    assert not re.search(r"AIza[0-9A-Za-z_-]{20,}", raw)
    assert not re.search(r"\bsk-[0-9A-Za-z_-]{20,}", raw)
    assert not re.search(r"\b\d{10}\b", raw)
    assert not re.search(r"[A-Z][12]\d{8}", raw)
    assert "@gmail.com" not in raw.lower()


def test_formal_dataset_hash_remains_immutable():
    assert hashlib.sha256(FORMAL_PATH.read_bytes()).hexdigest() == FORMAL_SHA256
