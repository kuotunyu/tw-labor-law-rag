import io
import zipfile
from pathlib import Path

import pytest

from rag.ingestion.loader import load_corpus, load_file, load_law_json, load_markdown
from scripts import download_corpus

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


def test_law_loader_preserves_public_provenance(tmp_path):
    law_path = tmp_path / "測試法.json"
    law_path.write_text(
        """{
          "name": "測試法",
          "url": "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0000001",
          "last_amended": "20260121",
          "effective_date": "20260123",
          "articles": [{"no": "第 1 條", "content": "測試內容。"}]
        }""",
        encoding="utf-8",
    )

    unit = load_law_json(law_path)[0]

    assert unit.source_url == "https://law.moj.gov.tw/LawClass/LawAll.aspx?pcode=N0000001"
    assert unit.last_amended == "20260121"
    assert unit.effective_date == "20260123"


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


def make_dump_zip(xml: str | None, *, member: str = "laws.xml") -> bytes:
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        if xml is None:
            zf.writestr("README.txt", "not a corpus")
        else:
            zf.writestr(member, xml)
    return buffer.getvalue()


def make_crc_corrupt_dump_zip() -> bytes:
    xml = "<laws><法規 /></laws>"
    corrupt_member = b"corrupt member payload"
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_STORED) as zf:
        zf.writestr("laws.xml", xml)
        zf.writestr("corrupt.bin", corrupt_member)
    payload = bytearray(buffer.getvalue())
    content_start = payload.index(corrupt_member)
    payload[content_start] ^= 1
    return bytes(payload)


class FakeStreamResponse:
    def __init__(self, payload: bytes):
        self.payload = payload
        self.headers = {"content-length": str(len(payload))}

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False

    def raise_for_status(self) -> None:
        return None

    def iter_bytes(self, chunk_size: int):
        yield self.payload


def test_validate_dump_zip_accepts_law_xml(tmp_path):
    path = tmp_path / "valid.zip"
    path.write_bytes(
        make_dump_zip(
            "<法規資料><法規><法規名稱>勞動基準法</法規名稱></法規></法規資料>"
        )
    )

    assert download_corpus.validate_dump_zip(path) == "laws.xml"


def test_validate_dump_zip_rejects_xml_entities(tmp_path):
    path = tmp_path / "entity.zip"
    path.write_bytes(
        make_dump_zip(
            "<!DOCTYPE 法規資料 [<!ENTITY injected '勞動基準法'>]>"
            "<法規資料><法規><法規名稱>&injected;</法規名稱></法規></法規資料>"
        )
    )

    with pytest.raises(download_corpus.CorpusArchiveError, match="unsafe XML"):
        download_corpus.validate_dump_zip(path)


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (b"not a zip", "ZIP"),
        (make_crc_corrupt_dump_zip(), "CRC"),
        (make_dump_zip(None), "XML"),
        (make_dump_zip("<法規資料></法規資料>"), "法規"),
        (make_dump_zip("<法規資料>"), "XML"),
        (make_dump_zip("<法規資料><法規 /></法規資料"), "XML"),
    ],
)
def test_validate_dump_zip_rejects_invalid_archives(tmp_path, payload, message):
    path = tmp_path / "invalid.zip"
    path.write_bytes(payload)

    with pytest.raises(download_corpus.CorpusArchiveError, match=message):
        download_corpus.validate_dump_zip(path)


def test_download_dump_skips_http_only_for_valid_cache(tmp_path, monkeypatch):
    payload = make_dump_zip("<法規資料><法規 /></法規資料>")
    path = tmp_path / "chlaw_acts.zip"
    path.write_bytes(payload)
    monkeypatch.setattr(download_corpus, "RAW_DIR", tmp_path)

    def unexpected_stream(*args, **kwargs):
        raise AssertionError("HTTP must not run for a valid cache")

    monkeypatch.setattr(download_corpus.httpx, "stream", unexpected_stream)

    assert download_corpus.download_dump("acts", "https://example.test") == path


def test_download_dump_replaces_invalid_cache_with_valid_download(tmp_path, monkeypatch):
    replacement = make_dump_zip("<法規資料><法規 /></法規資料>")
    path = tmp_path / "chlaw_acts.zip"
    path.write_bytes(b"invalid cache")
    monkeypatch.setattr(download_corpus, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        download_corpus.httpx,
        "stream",
        lambda *args, **kwargs: FakeStreamResponse(replacement),
    )

    assert download_corpus.download_dump("acts", "https://example.test") == path
    assert path.read_bytes() == replacement


def test_download_dump_never_overwrites_valid_cache_with_invalid_force_download(
    tmp_path, monkeypatch
):
    original = make_dump_zip("<法規資料><法規 /></法規資料>")
    path = tmp_path / "chlaw_acts.zip"
    path.write_bytes(original)
    monkeypatch.setattr(download_corpus, "RAW_DIR", tmp_path)
    monkeypatch.setattr(
        download_corpus.httpx,
        "stream",
        lambda *args, **kwargs: FakeStreamResponse(b"not a zip"),
    )

    with pytest.raises(download_corpus.CorpusArchiveError):
        download_corpus.download_dump("acts", "https://example.test", force=True)

    assert path.read_bytes() == original
    assert not (tmp_path / "chlaw_acts.part").exists()
