import json

from rag.indexing.bm25_index import BM25Index


def write_chunks(path, rows):
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def make_row(chunk_id, text):
    return {"chunk_id": chunk_id, "text": text, "content": text, "doc_title": "測試法", "articles": []}


def test_build_and_search(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [
            make_row("c1", "勞工結婚者給予婚假八日，工資照給。"),
            make_row("c2", "雇主延長工作時間之工資，依標準加給。"),
            make_row("c3", "女工分娩前後應停止工作，給予產假八星期。"),
        ],
    )
    index = BM25Index.build(chunks_path)
    assert len(index) == 3

    hits = index.search("結婚 婚假", top_k=3)
    assert hits, "should find at least one match"
    assert hits[0].payload["chunk_id"] == "c1"


def test_search_returns_empty_for_no_match(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(chunks_path, [make_row("c1", "勞工結婚者給予婚假八日。")])
    index = BM25Index.build(chunks_path)
    hits = index.search("量子力學相對論", top_k=5)
    assert hits == []


def test_save_and_load_roundtrip(tmp_path):
    chunks_path = tmp_path / "chunks.jsonl"
    write_chunks(
        chunks_path,
        [make_row("c1", "勞工結婚者給予婚假八日。"), make_row("c2", "雇主延長工作時間之工資加給。")],
    )
    index = BM25Index.build(chunks_path)
    save_path = tmp_path / "bm25.pkl"
    index.save(save_path)

    loaded = BM25Index.load(save_path)
    assert len(loaded) == 2
    hits_before = index.search("婚假", top_k=2)
    hits_after = loaded.search("婚假", top_k=2)
    assert [h.payload["chunk_id"] for h in hits_before] == [h.payload["chunk_id"] for h in hits_after]
