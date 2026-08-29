import json
import subprocess
import sys
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from rag.config import Settings
from rag.wage_arrears_regression import (
    build_public_result,
    load_regression_dataset,
    require_cached_models,
    route_expansion_applied,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATASET_PATH = (
    PROJECT_ROOT / "eval" / "dataset" / "wage_arrears_regression_v0.3.4.jsonl"
)
ARTICLE_14_SOURCE = [{"doc": "勞動基準法", "article": "第 14 條"}]


def _write_rows(path: Path, rows: list[object]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def _valid_rows() -> list[dict]:
    return [
        {
            "qid": f"wage-reg-{index:03d}",
            "question": f"測試問題 {index}：公司欠薪，我想直接離職。",
            "expect_expansion": index <= 10,
            "sources": ARTICLE_14_SOURCE if index <= 10 else [],
            "style_tags": ["test"],
        }
        for index in range(1, 21)
    ]


def test_targeted_dataset_has_reviewed_shape_and_routes() -> None:
    rows = load_regression_dataset(DATASET_PATH)

    assert len(rows) == 20
    assert [row["qid"] for row in rows] == [
        f"wage-reg-{index:03d}" for index in range(1, 21)
    ]
    assert sum(row["expect_expansion"] for row in rows) == 10
    assert all(
        route_expansion_applied(row["question"]) is row["expect_expansion"]
        for row in rows
    )
    assert all(row["sources"] == ARTICLE_14_SOURCE for row in rows[:10])
    assert all(row["sources"] == [] for row in rows[10:])


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda rows: rows[1].update(qid=rows[0]["qid"]), "duplicate qid"),
        (lambda rows: rows[0].pop("style_tags"), "fields"),
        (lambda rows: rows[0].update(expect_expansion="yes"), "boolean"),
        (lambda rows: rows[0].update(sources=[]), "sources"),
        (lambda rows: rows[10].update(sources=ARTICLE_14_SOURCE), "sources"),
    ],
)
def test_targeted_dataset_rejects_invalid_contract(
    tmp_path: Path,
    mutate,
    message: str,
) -> None:
    rows = _valid_rows()
    mutate(rows)
    path = tmp_path / "invalid.jsonl"
    _write_rows(path, rows)

    with pytest.raises(ValueError, match=message):
        load_regression_dataset(path)


def test_cached_model_preflight_fails_closed_without_downloading() -> None:
    settings = Settings(_env_file=None)

    def missing_snapshot(**_kwargs):
        raise LocalEntryNotFoundError("not cached")

    with pytest.raises(
        RuntimeError,
        match=(
            "missing pinned local model snapshot: "
            "BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181"
        ),
    ):
        require_cached_models(settings, resolver=missing_snapshot)


def test_cached_model_preflight_returns_only_pinned_local_paths() -> None:
    settings = Settings(_env_file=None)

    def local_snapshot(*, repo_id: str, revision: str, local_files_only: bool) -> str:
        assert local_files_only is True
        return f"local-cache/{repo_id}/{revision}"

    assert require_cached_models(settings, resolver=local_snapshot) == {
        "BAAI/bge-m3": (
            "local-cache/BAAI/bge-m3/5617a9f61b028005a4858fdac845db406aefb181"
        ),
        "BAAI/bge-reranker-v2-m3": (
            "local-cache/BAAI/bge-reranker-v2-m3/"
            "953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e"
        ),
    }


def _raw_cases() -> list[dict]:
    cases = []
    for index in range(1, 21):
        positive = index <= 10
        cases.append(
            {
                "qid": f"wage-reg-{index:03d}",
                "question": f"private question {index}",
                "expected_expansion": positive,
                "expansion_applied": positive,
                "rank": 1 if positive and index % 2 else (2 if positive else None),
                "top_score": 0.123456 + index / 1000,
                "hits": [{"content": "private hit", "endpoint": "https://private"}],
                "worktree": "machine-specific-worktree-marker",
            }
        )
    return cases


def test_public_result_is_deterministic_and_privacy_reduced(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b"fixture\n")

    result = build_public_result(
        dataset_path=dataset,
        code_revision="a" * 40,
        configuration={"chunking": "structure", "retrieval": "hybrid"},
        cases=list(reversed(_raw_cases())),
    )

    assert result["dataset"] == {
        "path": "eval/dataset/wage_arrears_regression_v0.3.4.jsonl",
        "sha256": "e80b71cd14d3cbd65f4173abcbfcf01a545dbca32a72d575108b553a648cc96f",
        "questions": 20,
    }
    assert result["summary"] == {
        "positive_routes": 10,
        "collision_routes_avoided": 10,
        "positive_hit_at_5": 10,
        "positive_hit_at_1": 5,
        "passed": True,
    }
    assert result["cases"][0] == {
        "qid": "wage-reg-001",
        "expected_expansion": True,
        "expansion_applied": True,
        "rank": 1,
        "top_score": 0.1245,
    }
    assert list(result["cases"][-1]) == [
        "qid",
        "expected_expansion",
        "expansion_applied",
        "rank",
        "top_score",
    ]
    serialized = json.dumps(result, ensure_ascii=False)
    assert "private question" not in serialized
    assert "private hit" not in serialized
    assert "https://" not in serialized
    assert "machine-specific-worktree-marker" not in serialized
    assert "AIza" not in serialized
    assert "sk-" not in serialized


def test_public_result_fails_acceptance_when_positive_misses_top_five(
    tmp_path: Path,
) -> None:
    dataset = tmp_path / "dataset.jsonl"
    dataset.write_bytes(b"fixture\n")
    cases = _raw_cases()
    cases[0]["rank"] = 6

    result = build_public_result(
        dataset_path=dataset,
        code_revision="b" * 40,
        configuration={"chunking": "structure", "retrieval": "hybrid"},
        cases=cases,
    )

    assert result["summary"]["positive_hit_at_5"] == 9
    assert result["summary"]["passed"] is False


def test_offline_runner_exposes_help_without_loading_models() -> None:
    completed = subprocess.run(
        [
            sys.executable,
            str(PROJECT_ROOT / "eval" / "run_wage_arrears_regression.py"),
            "--help",
        ],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )
    stdout = completed.stdout.decode("utf-8", errors="replace")
    stderr = completed.stderr.decode("utf-8", errors="replace")

    assert completed.returncode == 0, stderr
    assert "--dataset" in stdout
    assert "--snapshot" in stdout
    assert "--work-dir" in stdout
    assert "--device" in stdout
    assert "--export-official" in stdout
