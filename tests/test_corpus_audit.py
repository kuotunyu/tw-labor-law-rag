from __future__ import annotations

import hashlib
import io
import json
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest

from rag.corpus_audit import build_snapshot, compare_snapshots
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
