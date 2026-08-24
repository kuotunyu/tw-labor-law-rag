from pathlib import Path

import pytest

from rag.ingestion.loader import load_corpus, load_file, load_law_json, load_markdown

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = PROJECT_ROOT / "data" / "sample"


@pytest.fixture
def sample_law() -> Path:
    path = SAMPLE_DIR / "勞工請假規則.json"
    if not path.exists():
        pytest.skip("sample corpus not present; run scripts/download_corpus.py")
    return path


def test_load_law_json(sample_law):
    units = load_law_json(sample_law)
    assert units, "sample law should yield units"
    first = units[0]
    assert first.doc_title == "勞工請假規則"
    assert first.article_no.startswith("第")
    assert "婚假" in units[1].text or "婚假" in first.text
    # No deleted-article placeholders.
    assert all("刪除" not in u.text or len(u.text) > 10 for u in units)


def test_load_corpus_directory():
    if not SAMPLE_DIR.exists():
        pytest.skip("sample corpus not present")
    units = load_corpus(SAMPLE_DIR)
    titles = {u.doc_title for u in units}
    assert "勞工請假規則" in titles
    assert "勞動基準法施行細則" in titles


def test_load_markdown(tmp_path):
    md = tmp_path / "說明.md"
    md.write_text(
        "# 總則\n前言文字。\n## 定義\n本文件所稱勞工。\n## 適用\n適用全體。\n# 附則\n結尾。",
        encoding="utf-8",
    )
    units = load_markdown(md)
    assert [u.chapter for u in units] == ["總則", "總則 > 定義", "總則 > 適用", "附則"]
    assert units[1].text == "本文件所稱勞工。"


def test_load_file_rejects_unknown_suffix(tmp_path):
    weird = tmp_path / "x.docx"
    weird.write_text("hi", encoding="utf-8")
    with pytest.raises(ValueError):
        load_file(weird)
