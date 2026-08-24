from rag.models import RetrievedChunk
from rag.retrieval.fusion import reciprocal_rank_fusion


def hit(chunk_id, score=1.0):
    return RetrievedChunk(score=score, payload={"chunk_id": chunk_id})


def test_single_list_passthrough_order():
    hits = [hit("a"), hit("b"), hit("c")]
    fused = reciprocal_rank_fusion([hits], k=60)
    assert [h.payload["chunk_id"] for h in fused] == ["a", "b", "c"]


def test_disjoint_lists_union_all():
    list1 = [hit("a"), hit("b")]
    list2 = [hit("c"), hit("d")]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert {h.payload["chunk_id"] for h in fused} == {"a", "b", "c", "d"}


def test_overlap_boosts_rank_to_top():
    # "b" is #2 in list1 but #1 in list2 -> should outrank "a" (which only appears once, at #1).
    list1 = [hit("a"), hit("b")]
    list2 = [hit("b"), hit("z")]
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    assert fused[0].payload["chunk_id"] == "b"


def test_rrf_score_formula():
    list1 = [hit("a")]  # rank 1
    list2 = [hit("x"), hit("a")]  # rank 2
    fused = reciprocal_rank_fusion([list1, list2], k=60)
    a = next(h for h in fused if h.payload["chunk_id"] == "a")
    expected = 1 / (60 + 1) + 1 / (60 + 2)
    assert abs(a.score - expected) < 1e-9


def test_empty_lists():
    assert reciprocal_rank_fusion([], k=60) == []
    assert reciprocal_rank_fusion([[], []], k=60) == []


def test_payload_preserved_from_first_occurrence():
    rich_hit = RetrievedChunk(score=0.9, payload={"chunk_id": "a", "content": "詳細內容"})
    fused = reciprocal_rank_fusion([[rich_hit], [hit("a")]], k=60)
    assert fused[0].payload["content"] == "詳細內容"
