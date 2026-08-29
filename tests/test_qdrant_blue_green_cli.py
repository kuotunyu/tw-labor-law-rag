from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
from huggingface_hub.errors import LocalEntryNotFoundError

from rag.corpus_audit import build_snapshot
from scripts import rebuild_qdrant_blue_green as cli
from scripts.download_corpus import DUMPS


@pytest.fixture
def local_corpus(tmp_path: Path):
    raw_dir = tmp_path / "raw"
    laws_dir = raw_dir / "laws"
    laws_dir.mkdir(parents=True)
    law = {
        "name": "測試法",
        "nature": "法律",
        "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0000001",
        "last_amended": "20260829",
        "effective_date": "",
        "articles": [
            {"no": "第 1 條", "chapter": "", "content": "測試內容。"},
        ],
    }
    (laws_dir / "test-law.json").write_text(
        json.dumps(law, ensure_ascii=False), encoding="utf-8"
    )

    sources = []
    for source_id, (url, _targets) in DUMPS.items():
        payload = f"{source_id} official bytes".encode()
        (raw_dir / f"chlaw_{source_id}.zip").write_bytes(payload)
        sources.append(
            {"id": source_id, "url": url, "sha256": hashlib.sha256(payload).hexdigest()}
        )

    snapshot = build_snapshot(
        sources=sources,
        laws=[law],
        snapshot_date="2026-08-29",
    )
    snapshot_path = tmp_path / "snapshot.json"
    snapshot_path.write_text(
        json.dumps(snapshot, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    receipt = Path("eval/runs/qdrant-maintenance/test-receipt.json")
    return {
        "raw_dir": raw_dir,
        "laws_dir": laws_dir,
        "snapshot": snapshot_path,
        "receipt": receipt,
    }


def _base_args(local_corpus) -> list[str]:
    return [
        "--candidate-base",
        "labor_laws_20260830_deadbeef",
        "--corpus",
        str(local_corpus["laws_dir"]),
        "--raw-dir",
        str(local_corpus["raw_dir"]),
        "--snapshot",
        str(local_corpus["snapshot"]),
        "--receipt",
        str(local_corpus["receipt"]),
    ]


def test_dry_run_never_reads_writer_credentials_or_loads_models(
    monkeypatch, local_corpus, capsys
):
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_WRITER_API_KEY", raising=False)
    monkeypatch.setattr(
        cli,
        "snapshot_download",
        lambda **_kwargs: pytest.fail("dry-run must not inspect model cache"),
    )

    assert cli.main(_base_args(local_corpus)) == 0

    output = json.loads(capsys.readouterr().out)
    assert set(output) == {
        "status",
        "active_base",
        "candidate_base",
        "collections",
        "snapshot_sha256",
        "execution_required",
    }
    assert output["status"] == "dry_run_ready"
    assert output["execution_required"] is True
    assert not (cli.PROJECT_ROOT / local_corpus["receipt"]).exists()


def test_dry_run_fails_closed_on_snapshot_drift(local_corpus, capsys):
    snapshot = json.loads(local_corpus["snapshot"].read_text(encoding="utf-8"))
    snapshot["laws"][0]["content_sha256"] = "0" * 64
    local_corpus["snapshot"].write_text(json.dumps(snapshot), encoding="utf-8")

    assert cli.main(_base_args(local_corpus)) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {"status": "error", "code": "snapshot_drift"}
    assert "content_sha256" not in captured.err


def test_execute_requires_repeated_candidate_confirmation(local_corpus, capsys):
    args = [
        *_base_args(local_corpus),
        "--execute",
        "--confirm-candidate-base",
        "labor_laws_wrong",
    ]

    assert cli.main(args) == 2
    assert json.loads(capsys.readouterr().err) == {
        "status": "error",
        "code": "candidate_confirmation_mismatch",
    }


@pytest.mark.parametrize("missing", ["QDRANT_URL", "QDRANT_WRITER_API_KEY"])
def test_execute_fails_before_cache_check_when_writer_environment_missing(
    monkeypatch, local_corpus, capsys, missing
):
    monkeypatch.setenv("QDRANT_URL", "https://private.example.test")
    monkeypatch.setenv("QDRANT_WRITER_API_KEY", "private-writer-key")
    monkeypatch.delenv(missing, raising=False)
    monkeypatch.setattr(
        cli,
        "snapshot_download",
        lambda **_kwargs: pytest.fail("missing environment must fail before cache check"),
    )
    candidate = "labor_laws_20260830_deadbeef"
    args = [
        *_base_args(local_corpus),
        "--execute",
        "--confirm-candidate-base",
        candidate,
    ]

    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "status": "error",
        "code": "missing_writer_environment",
    }
    assert "private" not in captured.err


def test_execute_checks_both_pinned_snapshots_without_downloading(
    monkeypatch, local_corpus, capsys
):
    calls = []
    monkeypatch.setenv("QDRANT_URL", "https://private.example.test")
    monkeypatch.setenv("QDRANT_WRITER_API_KEY", "private-writer-key")

    def missing_snapshot(**kwargs):
        calls.append(kwargs)
        raise LocalEntryNotFoundError("private cache path")

    monkeypatch.setattr(cli, "snapshot_download", missing_snapshot)
    candidate = "labor_laws_20260830_deadbeef"
    args = [
        *_base_args(local_corpus),
        "--execute",
        "--confirm-candidate-base",
        candidate,
    ]

    assert cli.main(args) == 2
    captured = capsys.readouterr()
    assert json.loads(captured.err) == {
        "status": "error",
        "code": "missing_model_snapshot",
    }
    assert len(calls) == 2
    assert {call["repo_id"] for call in calls} == {
        "BAAI/bge-m3",
        "BAAI/bge-reranker-v2-m3",
    }
    assert calls[0]["local_files_only"] is True
    assert "private cache path" not in captured.err


def test_parser_exposes_only_approved_maintenance_options():
    parser = cli.build_parser()
    options = {
        option
        for action in parser._actions
        for option in action.option_strings
        if option != "--help"
    }

    assert options == {
        "-h",
        "--candidate-base",
        "--confirm-candidate-base",
        "--active-base",
        "--corpus",
        "--raw-dir",
        "--snapshot",
        "--receipt",
        "--device",
        "--execute",
    }
