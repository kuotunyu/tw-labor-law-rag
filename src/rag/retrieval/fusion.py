"""Reciprocal Rank Fusion — combines multiple ranked result lists into one.

RRF score for a document d: sum, over every list containing d, of
``1 / (k + rank_L(d))`` (rank is 1-based). Documents absent from a list simply
don't contribute from it. The point of RRF is exactly this: it needs no score
normalization across heterogeneous retrievers (cosine similarity vs. BM25
score), only rank position.
"""

from __future__ import annotations

from rag.models import RetrievedChunk


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievedChunk]], k: int = 60
) -> list[RetrievedChunk]:
    scores: dict[str, float] = {}
    payloads: dict[str, dict] = {}

    for results in result_lists:
        for rank, hit in enumerate(results, start=1):
            key = hit.payload["chunk_id"]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            payloads.setdefault(key, hit.payload)

    fused = [RetrievedChunk(score=score, payload=payloads[key]) for key, score in scores.items()]
    fused.sort(key=lambda h: h.score, reverse=True)
    return fused
