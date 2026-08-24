"""Download the labor-law corpus from Taiwan's National Laws & Regulations Database.

Sources (Open Government Data License v1, updated monthly by the Ministry of Justice):
  - Acts (法律):        https://data.gov.tw/dataset/18289
  - Regulations (命令): https://data.gov.tw/dataset/18290

The script downloads the two official XML dumps (cached under ``data/raw/``),
extracts the 15 target labor laws, and writes one normalized JSON file per law to
``data/raw/laws/`` plus a ``manifest.json``. Two small regulations are copied to
``data/sample/`` so the repo can be smoke-tested without the full download.

Usage:
    python scripts/download_corpus.py [--force-download]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import xml.etree.ElementTree as ET
import zipfile
from pathlib import Path
from typing import Iterator

import httpx

# Windows consoles default to a locale code page (e.g. cp950); force UTF-8 output.
sys.stdout.reconfigure(encoding="utf-8")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"
LAWS_DIR = RAW_DIR / "laws"
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"

DUMPS = {
    # dump_id -> (download URL, target law names)
    "acts": (
        "https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=XML&AuData=CF",
        [
            "勞動基準法",
            "勞工退休金條例",
            "勞工保險條例",
            "性別平等工作法",
            "職業安全衛生法",
            "就業服務法",
            "最低工資法",
            "勞資爭議處理法",
            "勞動事件法",
            "勞工職業災害保險及保護法",
            "工會法",
            "團體協約法",
            "大量解僱勞工保護法",
        ],
    ),
    "regulations": (
        "https://sendlaw.moj.gov.tw/PublicData/GetFile.ashx?DType=XML&AuData=CM",
        [
            "勞動基準法施行細則",
            "勞工請假規則",
        ],
    ),
}

# Small laws shipped with the repo for quick smoke tests / CI.
SAMPLE_LAWS = ["勞工請假規則", "勞動基準法施行細則"]


def normalize_name(name: str) -> str:
    """Drop annotations the database appends to names, e.g. 最低工資法（112.12.27制定）.

    Handles both full-width （） (U+FF08/U+FF09) and ASCII parentheses.
    """
    return re.sub(r"[（(][^（）()]*[）)]\s*$", "", name.strip())


def download_dump(dump_id: str, url: str, force: bool = False) -> Path:
    zip_path = RAW_DIR / f"chlaw_{dump_id}.zip"
    if zip_path.exists() and not force:
        print(f"[skip] {dump_id} dump already downloaded: {zip_path}", flush=True)
        return zip_path

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[download] {dump_id}: {url}", flush=True)
    tmp_path = zip_path.with_suffix(".part")
    with httpx.stream("GET", url, timeout=180.0, follow_redirects=True) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("content-length", 0))
        done = 0
        next_report = 0
        with open(tmp_path, "wb") as f:
            for chunk in resp.iter_bytes(chunk_size=1 << 20):
                f.write(chunk)
                done += len(chunk)
                if done >= next_report:
                    mb = done / (1 << 20)
                    pct = f" ({done / total:.0%})" if total else ""
                    print(f"  ... {mb:.0f} MB{pct}", flush=True)
                    next_report += 20 << 20
    tmp_path.replace(zip_path)
    print(f"[ok] saved {zip_path} ({zip_path.stat().st_size / (1 << 20):.1f} MB)", flush=True)
    return zip_path


def parse_law(elem: ET.Element) -> dict:
    def text(tag: str) -> str:
        return (elem.findtext(tag) or "").strip()

    law = {
        "name": text("法規名稱"),
        "nature": text("法規性質"),
        "category": text("法規類別"),
        "url": text("法規網址"),
        "last_amended": text("最新異動日期"),
        "effective_date": text("生效日期"),
        "abolished": text("廢止註記"),
        "history": text("沿革內容"),
        "preamble": text("前言"),
        "articles": [],
    }
    content = elem.find("法規內容")
    chapter = ""
    if content is not None:
        for child in content:
            tag = child.tag.strip()
            if tag == "編章節":
                chapter = (child.text or "").strip()
            elif tag == "條文":
                law["articles"].append(
                    {
                        "no": (child.findtext("條號") or "").strip(),
                        "chapter": chapter,
                        "content": (child.findtext("條文內容") or "").strip(),
                    }
                )
    return law


def iter_laws(zip_path: Path) -> Iterator[dict]:
    """Stream <法規> elements from the XML dump without loading the whole tree."""
    with zipfile.ZipFile(zip_path) as zf:
        xml_members = [n for n in zf.namelist() if n.lower().endswith(".xml")]
        if not xml_members:
            sys.exit(f"error: no XML member in {zip_path}; members: {zf.namelist()[:10]}")
        member = max(xml_members, key=lambda n: zf.getinfo(n).file_size)
        print(f"[parse] {zip_path.name} -> {member}", flush=True)
        with zf.open(member) as f:
            for _event, elem in ET.iterparse(f, events=("end",)):
                if elem.tag.strip() == "法規":
                    yield parse_law(elem)
                    elem.clear()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force-download", action="store_true", help="re-download the dumps")
    args = parser.parse_args()

    LAWS_DIR.mkdir(parents=True, exist_ok=True)
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)

    manifest = []
    missing = []
    for dump_id, (url, targets) in DUMPS.items():
        zip_path = download_dump(dump_id, url, force=args.force_download)
        wanted = set(targets)
        found = {}
        total = 0
        for law in iter_laws(zip_path):
            total += 1
            norm = normalize_name(law["name"])
            if norm in wanted and not law["abolished"]:
                law["name_raw"] = law["name"]
                law["name"] = norm
                found[norm] = law
        print(f"[ok] scanned {total} laws in {dump_id} dump", flush=True)

        for name in targets:
            law = found.get(name)
            if law is None:
                missing.append(name)
                print(f"  [MISSING] {name}", flush=True)
                continue
            out_path = LAWS_DIR / f"{name}.json"
            payload = json.dumps(law, ensure_ascii=False, indent=1)
            out_path.write_text(payload, encoding="utf-8")
            manifest.append(
                {
                    "name": name,
                    "file": out_path.name,
                    "nature": law["nature"],
                    "url": law["url"],
                    "last_amended": law["last_amended"],
                    "num_articles": len(law["articles"]),
                }
            )
            print(
                f"  [ok] {name}: {len(law['articles'])} articles, amended {law['last_amended']}",
                flush=True,
            )
            if name in SAMPLE_LAWS:
                (SAMPLE_DIR / out_path.name).write_text(payload, encoding="utf-8")

    (LAWS_DIR / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(f"\n[done] {len(manifest)}/{sum(len(t) for _, t in DUMPS.values())} laws -> {LAWS_DIR}", flush=True)
    if missing:
        print(f"[warn] not found (check names): {missing}", flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
