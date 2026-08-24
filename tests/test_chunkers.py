import pytest

from rag.ingestion.chunkers import (
    FixedSizeChunker,
    StructureAwareChunker,
    get_chunker,
    split_sentences,
)
from rag.models import SourceUnit


def make_unit(text: str, article_no: str = "", doc_id: str = "測試法", chapter: str = "") -> SourceUnit:
    return SourceUnit(
        text=text, doc_id=doc_id, doc_title=doc_id, article_no=article_no, chapter=chapter
    )


# ── split_sentences ─────────────────────────────────────


def test_split_sentences_on_chinese_punctuation():
    sents = split_sentences("第一句。第二句！第三句？", hard_limit=100)
    assert len(sents) == 3
    assert sents[0] == "第一句。"


def test_split_sentences_hard_limit():
    sents = split_sentences("甲" * 250, hard_limit=100)
    assert [len(s) for s in sents] == [100, 100, 50]


def test_split_sentences_preserves_all_text():
    text = "一、勞工:指受僱之人。\n二、雇主:指僱用勞工之人。"
    assert "".join(split_sentences(text, hard_limit=100)) == text


# ── StructureAwareChunker ───────────────────────────────


def test_structure_one_chunk_per_article():
    units = [
        make_unit("勞工結婚者給予婚假八日。", "第 2 條", chapter="第 一 章"),
        make_unit("工資照給。", "第 3 條", chapter="第 一 章"),
    ]
    chunks = StructureAwareChunker().chunk(units)
    assert len(chunks) == 2
    assert chunks[0].articles == ["第 2 條"]
    assert chunks[0].text.startswith("測試法 第 一 章 第 2 條\n")
    assert chunks[0].content == "勞工結婚者給予婚假八日。"
    assert chunks[0].chunk_id != chunks[1].chunk_id


def test_structure_long_article_split_shares_metadata():
    long_text = "。".join(f"第{i}項規定內容" for i in range(1, 200)) + "。"
    chunks = StructureAwareChunker(max_chars=300).chunk([make_unit(long_text, "第 9 條")])
    assert len(chunks) > 1
    assert all(c.articles == ["第 9 條"] for c in chunks)
    assert all(len(c.content) <= 300 for c in chunks)
    # No text lost.
    assert "".join(c.content for c in chunks).replace("\n", "") == long_text.replace("\n", "")


def test_structure_article_label():
    chunks = StructureAwareChunker().chunk([make_unit("內容。", "第 5 條")])
    assert chunks[0].article_label == "第 5 條"


# ── FixedSizeChunker ────────────────────────────────────


def make_articles(n: int, sent_len: int = 30, sents_per_article: int = 5) -> list[SourceUnit]:
    units = []
    for i in range(1, n + 1):
        body = "".join(f"第{i}條第{j}項" + "規" * (sent_len - 7) + "。" for j in range(1, sents_per_article + 1))
        units.append(make_unit(body, f"第 {i} 條"))
    return units


def test_fixed_respects_chunk_size():
    chunks = FixedSizeChunker(chunk_size=200, overlap=40).chunk(make_articles(6))
    assert len(chunks) > 1
    # Sentence packing may slightly overshoot only via the forced first-fresh sentence.
    assert all(len(c.content) <= 200 + 40 for c in chunks)


def test_fixed_has_overlap():
    chunks = FixedSizeChunker(chunk_size=200, overlap=60).chunk(make_articles(6))
    for prev, cur in zip(chunks, chunks[1:]):
        tail = prev.content[-20:]
        assert tail in cur.content, "consecutive chunks should share overlapping text"


def test_fixed_tracks_spanned_articles():
    chunks = FixedSizeChunker(chunk_size=200, overlap=40).chunk(make_articles(6, sent_len=30))
    spanning = [c for c in chunks if len(c.articles) > 1]
    assert spanning, "windows crossing article boundaries should record every article"
    for c in spanning:
        assert "～" in c.article_label


def test_fixed_no_text_lost():
    units = make_articles(4)
    chunks = FixedSizeChunker(chunk_size=200, overlap=40).chunk(units)
    joined = "".join(c.content for c in chunks)
    for unit in units:
        for sent in split_sentences(unit.text, hard_limit=200):
            assert sent in joined


def test_fixed_rejects_bad_overlap():
    with pytest.raises(ValueError):
        FixedSizeChunker(chunk_size=100, overlap=100)


# ── factory ─────────────────────────────────────────────


def test_get_chunker():
    assert isinstance(get_chunker("structure"), StructureAwareChunker)
    assert isinstance(get_chunker("fixed", chunk_size=300, overlap=50), FixedSizeChunker)
    with pytest.raises(ValueError):
        get_chunker("semantic")
