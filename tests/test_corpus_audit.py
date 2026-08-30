from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from rag.corpus_audit import (
    build_article_snapshot,
    build_snapshot,
    compare_article_snapshots,
    compare_snapshots,
    summarize_changes,
)
from scripts import audit_corpus, download_corpus

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _law(
    name: str,
    *,
    last_amended: str = "20260121",
    effective_date: str = "20260123",
    content: str = "第一條內容。",
) -> dict:
    return {
        "name": name,
        "nature": "法律",
        "url": f"https://law.moj.gov.tw/{name}",
        "last_amended": last_amended,
        "effective_date": effective_date,
        "articles": [
            {"no": "第 1 條", "chapter": "", "content": content},
            {"no": "第 2 條", "chapter": "", "content": "（刪除）"},
        ],
    }


def _source(source_id: str = "acts", digest: str = "a" * 64) -> dict:
    return {
        "id": source_id,
        "url": f"https://sendlaw.moj.gov.tw/{source_id}",
        "sha256": digest,
    }


def test_build_snapshot_sorts_laws_and_hashes_normalized_content():
    snapshot = build_snapshot(
        sources=[_source()],
        laws=[_law("甲法"), _law("乙法")],
        snapshot_date="2026-08-29",
    )

    assert snapshot["schema_version"] == "1.0"
    assert snapshot["snapshot_date"] == "2026-08-29"
    assert [law["name"] for law in snapshot["laws"]] == ["乙法", "甲法"]
    assert snapshot["law_count"] == 2
    assert snapshot["article_count"] == 2
    assert snapshot["laws"][0]["num_articles"] == 1

    expected_payload = {
        "effective_date": "20260123",
        "last_amended": "20260121",
        "name": "乙法",
        "nature": "法律",
        "url": "https://law.moj.gov.tw/乙法",
        "articles": [{"no": "第 1 條", "chapter": "", "content": "第一條內容。"}],
    }
    expected_bytes = json.dumps(
        expected_payload,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    assert snapshot["laws"][0]["content_sha256"] == hashlib.sha256(expected_bytes).hexdigest()


def test_compare_snapshots_reports_amendment_and_content_changes():
    old = build_snapshot(
        sources=[_source()],
        laws=[_law("測試法")],
        snapshot_date="2026-08-28",
    )
    new = build_snapshot(
        sources=[_source(digest="b" * 64)],
        laws=[_law("測試法", last_amended="20260829", content="修正內容。")],
        snapshot_date="2026-08-29",
    )

    changes = compare_snapshots(old, new)

    assert changes == [
        {"source": "acts", "kind": "sha256", "old": "a" * 64, "new": "b" * 64},
        {"law": "測試法", "kind": "last_amended", "old": "20260121", "new": "20260829"},
        {
            "law": "測試法",
            "kind": "content_sha256",
            "old": old["laws"][0]["content_sha256"],
            "new": new["laws"][0]["content_sha256"],
        },
    ]


def test_compare_snapshots_reports_added_and_removed_laws():
    old = build_snapshot(
        sources=[_source()],
        laws=[_law("舊法")],
        snapshot_date="2026-08-28",
    )
    new = build_snapshot(
        sources=[_source()],
        laws=[_law("新法")],
        snapshot_date="2026-08-29",
    )

    assert compare_snapshots(old, new) == [
        {"law": "新法", "kind": "added"},
        {"law": "舊法", "kind": "removed"},
    ]


def test_change_summary_counts_laws_and_changed_fields():
    changes = [
        {"source": "acts", "kind": "added"},
        {"law": "乙法", "kind": "removed"},
        {"source": "regulations", "kind": "sha256", "old": "a" * 64, "new": "b" * 64},
        {"law": "丙法", "kind": "last_amended", "old": "20260101", "new": "20260830"},
        {"law": "丙法", "kind": "num_articles", "old": 10, "new": 11},
        {"law": "丁法", "kind": "content_sha256", "old": "c" * 64, "new": "d" * 64},
        {"law": "丁法", "kind": "effective_date", "old": "20260101", "new": "20260830"},
    ]

    assert summarize_changes(changes) == {
        "total_changes": 7,
        "subjects_changed": 4,
        "added": 1,
        "removed": 1,
        "sha256": 1,
        "last_amended": 1,
        "effective_date": 1,
        "num_articles": 1,
        "content_sha256": 1,
    }


def test_article_comparison_reports_counts_without_content():
    committed = {
        "甲法": {"第 1 條": "a" * 64, "第 2 條": "b" * 64, "第 3 條": "c" * 64}
    }
    live = {
        "甲法": {"第 1 條": "a" * 64, "第 2 條": "d" * 64, "第 4 條": "e" * 64}
    }

    report = compare_article_snapshots(committed, live)

    assert report["summary"] == {
        "added": 1,
        "removed": 1,
        "changed": 1,
        "unchanged": 1,
    }
    assert report["changes"] == [
        {
            "law": "甲法",
            "article": "第 2 條",
            "kind": "changed",
            "old": "b" * 64,
            "new": "d" * 64,
        },
        {"law": "甲法", "article": "第 3 條", "kind": "removed", "old": "c" * 64},
        {"law": "甲法", "article": "第 4 條", "kind": "added", "new": "e" * 64},
    ]
    assert "content" not in repr(report)


def test_build_article_snapshot_is_content_free_and_rejects_duplicates():
    snapshot = build_article_snapshot([_law("甲法")], snapshot_date="2026-08-29")

    assert snapshot["law_count"] == 1
    assert snapshot["article_count"] == 1
    assert set(snapshot["laws"][0]) == {"name", "articles"}
    assert set(snapshot["laws"][0]["articles"][0]) == {"article", "sha256"}
    assert "第一條內容" not in json.dumps(snapshot, ensure_ascii=False)

    duplicate = _law("甲法")
    duplicate["articles"].append(
        {"no": "第 1 條", "chapter": "", "content": "重複條文。"}
    )
    with pytest.raises(ValueError, match="duplicate article"):
        build_article_snapshot([duplicate], snapshot_date="2026-08-29")


@pytest.mark.parametrize(
    ("sources", "laws", "message"),
    [
        ([_source(digest="bad")], [_law("測試法")], "source sha256"),
        ([_source(), _source()], [_law("測試法")], "duplicate source id"),
        ([_source()], [_law("測試法"), _law("測試法")], "duplicate law name"),
    ],
)
def test_build_snapshot_rejects_ambiguous_or_unverifiable_input(sources, laws, message):
    with pytest.raises(ValueError, match=message):
        build_snapshot(sources=sources, laws=laws, snapshot_date="2026-08-29")


def _archive_bytes(law_names: list[str]) -> bytes:
    laws = "".join(
        f"""
        <法規>
          <法規名稱>{name}</法規名稱><法規性質>法律</法規性質>
          <法規網址>https://law.moj.gov.tw/{name}</法規網址>
          <最新異動日期>20260829</最新異動日期><生效日期>20260830</生效日期>
          <廢止註記></廢止註記>
          <法規內容><條文><條號>第 1 條</條號><條文內容>內容。</條文內容></條文></法規內容>
        </法規>
        """
        for name in law_names
    )
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w") as archive:
        archive.writestr("laws.xml", f"<法規資料庫>{laws}</法規資料庫>")
    return output.getvalue()


def test_build_live_snapshot_requires_and_records_all_15_target_laws():
    archives = {
        url: _archive_bytes(targets)
        for _source_id, (url, targets) in download_corpus.DUMPS.items()
    }

    snapshot = audit_corpus.build_live_snapshot(
        snapshot_date="2026-08-29",
        downloader=lambda url: archives[url],
    )

    assert snapshot["law_count"] == 15
    assert snapshot["article_count"] == 15
    assert {source["id"] for source in snapshot["sources"]} == {"acts", "regulations"}
    assert {law["name"] for law in snapshot["laws"]} == {
        name for _url, names in download_corpus.DUMPS.values() for name in names
    }

    law_snapshot, article_snapshot = audit_corpus.build_live_snapshots(
        snapshot_date="2026-08-29",
        downloader=lambda url: archives[url],
    )
    assert law_snapshot == snapshot
    assert article_snapshot["law_count"] == 15
    assert article_snapshot["article_count"] == 15
    assert "內容。" not in json.dumps(article_snapshot, ensure_ascii=False)


def test_build_live_snapshot_fails_closed_when_target_law_is_missing():
    archives = {}
    for source_id, (url, targets) in download_corpus.DUMPS.items():
        selected = targets[:-1] if source_id == "acts" else targets
        archives[url] = _archive_bytes(selected)

    with pytest.raises(ValueError, match="missing target laws: 大量解僱勞工保護法"):
        audit_corpus.build_live_snapshot(
            snapshot_date="2026-08-29",
            downloader=lambda url: archives[url],
        )


def test_audit_corpus_cli_is_directly_executable():
    process = subprocess.run(
        [sys.executable, "scripts/audit_corpus.py", "--help"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        check=False,
    )

    stderr = process.stderr.decode("utf-8", errors="replace")
    stdout = process.stdout.decode("utf-8", errors="replace")
    assert process.returncode == 0, stderr
    assert "--snapshot-date" in stdout
    assert "--article-check" in stdout


def _write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, ensure_ascii=False), encoding="utf-8")


def _cli_snapshots(content: str = "第一條內容。") -> tuple[dict, dict]:
    laws = [_law("測試法", content=content)]
    return (
        build_snapshot(
            sources=[_source()], laws=laws, snapshot_date="2026-08-29"
        ),
        build_article_snapshot(laws, snapshot_date="2026-08-29"),
    )


def test_audit_corpus_cli_reports_combined_current_summary(
    monkeypatch, tmp_path, capsys
):
    law_snapshot, article_snapshot = _cli_snapshots()
    law_path = tmp_path / "laws.json"
    article_path = tmp_path / "articles.json"
    _write_json(law_path, law_snapshot)
    _write_json(article_path, article_snapshot)
    monkeypatch.setattr(
        audit_corpus,
        "build_live_snapshots",
        lambda **_kwargs: (law_snapshot, article_snapshot),
    )

    assert audit_corpus.main(
        ["--check", str(law_path), "--article-check", str(article_path)]
    ) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload == {
        "status": "current",
        "changes": {"laws": [], "articles": []},
        "summary": {
            "laws": {
                "total_changes": 0,
                "subjects_changed": 0,
                "added": 0,
                "removed": 0,
                "sha256": 0,
                "last_amended": 0,
                "effective_date": 0,
                "num_articles": 0,
                "content_sha256": 0,
            },
            "articles": {"added": 0, "removed": 0, "changed": 0, "unchanged": 1},
        },
    }
    assert "第一條內容" not in repr(payload)


def test_audit_corpus_cli_reports_named_law_and_article_changes(
    monkeypatch, tmp_path, capsys
):
    committed_law, committed_articles = _cli_snapshots()
    live_law, live_articles = _cli_snapshots("已修正條文。")
    law_path = tmp_path / "laws.json"
    article_path = tmp_path / "articles.json"
    _write_json(law_path, committed_law)
    _write_json(article_path, committed_articles)
    monkeypatch.setattr(
        audit_corpus,
        "build_live_snapshots",
        lambda **_kwargs: (live_law, live_articles),
    )

    assert audit_corpus.main(
        ["--check", str(law_path), "--article-check", str(article_path)]
    ) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["status"] == "changed"
    assert payload["changes"]["laws"][0]["law"] == "測試法"
    assert payload["changes"]["articles"][0]["article"] == "第 1 條"
    assert payload["summary"]["articles"]["changed"] == 1
    assert "已修正條文" not in repr(payload)


def test_audit_corpus_bootstrap_requires_current_law_snapshot(
    monkeypatch, tmp_path, capsys
):
    committed_law, _committed_articles = _cli_snapshots()
    live_law, live_articles = _cli_snapshots("已修正條文。")
    law_path = tmp_path / "laws.json"
    target = tmp_path / "article-baseline.json"
    _write_json(law_path, committed_law)
    monkeypatch.setattr(
        audit_corpus,
        "build_live_snapshots",
        lambda **_kwargs: (live_law, live_articles),
    )

    assert audit_corpus.main(
        ["--check", str(law_path), "--bootstrap-article-snapshot", str(target)]
    ) == 1
    assert not target.exists()
    assert json.loads(capsys.readouterr().out)["status"] == "changed"


def test_audit_corpus_bootstrap_writes_content_free_article_snapshot(
    monkeypatch, tmp_path
):
    law_snapshot, article_snapshot = _cli_snapshots()
    law_path = tmp_path / "laws.json"
    target = tmp_path / "article-baseline.json"
    _write_json(law_path, law_snapshot)
    monkeypatch.setattr(
        audit_corpus,
        "build_live_snapshots",
        lambda **_kwargs: (law_snapshot, article_snapshot),
    )

    assert audit_corpus.main(
        ["--check", str(law_path), "--bootstrap-article-snapshot", str(target)]
    ) == 0
    assert json.loads(target.read_text(encoding="utf-8")) == article_snapshot
    assert b"\r\n" not in target.read_bytes()
    assert "第一條內容" not in target.read_text(encoding="utf-8")


def test_audit_corpus_write_requires_and_updates_both_snapshots(
    monkeypatch, tmp_path
):
    law_snapshot, article_snapshot = _cli_snapshots()
    law_path = tmp_path / "laws.json"
    article_path = tmp_path / "articles.json"
    monkeypatch.setattr(
        audit_corpus,
        "build_live_snapshots",
        lambda **_kwargs: (law_snapshot, article_snapshot),
    )

    with pytest.raises(SystemExit) as exc:
        audit_corpus.main(["--write", str(law_path)])
    assert exc.value.code == 2
    assert not law_path.exists()

    assert audit_corpus.main(
        ["--write", str(law_path), "--article-write", str(article_path)]
    ) == 0
    assert json.loads(law_path.read_text(encoding="utf-8")) == law_snapshot
    assert json.loads(article_path.read_text(encoding="utf-8")) == article_snapshot


def test_audit_corpus_cli_returns_two_for_invalid_source_without_leaking_detail(
    monkeypatch,
    capsys,
):
    def fail_audit(**_kwargs):
        raise ValueError("private malformed source detail")

    monkeypatch.setattr(audit_corpus, "build_live_snapshots", fail_audit)

    assert audit_corpus.main(["--check", "release/corpus_snapshot.json"]) == 2
    captured = capsys.readouterr()
    assert "invalid_source" in captured.err
    assert "private malformed source detail" not in captured.err


def test_audit_corpus_cli_sanitizes_unsafe_archive_failure(monkeypatch, capsys):
    def fail_audit(**_kwargs):
        raise download_corpus.CorpusArchiveError("private unsafe XML detail")

    monkeypatch.setattr(audit_corpus, "build_live_snapshots", fail_audit)

    assert audit_corpus.main(["--check", "release/corpus_snapshot.json"]) == 2
    captured = capsys.readouterr()
    assert '"error_type": "CorpusArchiveError"' in captured.err
    assert "private unsafe XML detail" not in captured.err
