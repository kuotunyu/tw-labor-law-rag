"""Cross-encoder reranking with bge-reranker-v2-m3.

Scores are sigmoid-normalized to roughly [0, 1] (``normalize=True``), which is
what makes ``rerank_score_threshold`` in :mod:`rag.config` meaningful — raw
cosine/BM25/RRF scores from the first-stage retrievers are not comparable
across queries (Phase 1 finding: unanswerable questions scored just as high as
answerable ones), but the cross-encoder score is.
"""

from __future__ import annotations

from numbers import Real
from types import MethodType
from typing import Any, Sequence

from rag.config import DEFAULT_RERANKER_MODEL_REVISION
from rag.indexing.embedder import resolve_device, resolve_model_snapshot
from rag.models import RetrievedChunk


def _canonical_chunk_id(hit: RetrievedChunk) -> str:
    """Return a stable chunk ID, rejecting malformed retrieval evidence."""
    chunk_id = hit.payload.get("chunk_id")
    if not isinstance(chunk_id, str) or not chunk_id or chunk_id != chunk_id.strip():
        raise ValueError("candidate IDs must be non-blank canonical strings")
    return chunk_id


def _canonical_chunk_ids(hits: Sequence[RetrievedChunk], *, label: str) -> tuple[str, ...]:
    chunk_ids = tuple(_canonical_chunk_id(hit) for hit in hits)
    if len(set(chunk_ids)) != len(chunk_ids):
        raise ValueError(f"{label} candidate IDs must be unique")
    return chunk_ids


def _require_exact_permutation(
    expected_ids: tuple[str, ...],
    ranking: Sequence[RetrievedChunk],
    *,
    label: str,
) -> tuple[str, ...]:
    ranking_ids = _canonical_chunk_ids(ranking, label=label)
    if len(ranking_ids) != len(expected_ids) or set(ranking_ids) != set(expected_ids):
        raise ValueError(f"{label} ranking must be an exact candidate-ID permutation")
    return ranking_ids


def interleave_reranker_rankings(
    candidates: Sequence[RetrievedChunk],
    primary_ranking: Sequence[RetrievedChunk],
    secondary_ranking: Sequence[RetrievedChunk],
) -> list[RetrievedChunk]:
    """Merge two full reranker permutations by deterministic primary-first depth.

    The secondary ranking contributes order only.  Each emitted hit comes from
    the primary ranking, retaining its primary-query score even when that makes
    the merged public scores non-monotonic.
    """
    candidate_ids = _canonical_chunk_ids(candidates, label="input")
    primary_ids = _require_exact_permutation(candidate_ids, primary_ranking, label="primary")
    secondary_ids = _require_exact_permutation(
        candidate_ids, secondary_ranking, label="secondary"
    )
    primary_by_id = dict(zip(primary_ids, primary_ranking, strict=True))
    merged: list[RetrievedChunk] = []
    emitted_ids: set[str] = set()
    for primary_id, secondary_id in zip(primary_ids, secondary_ids, strict=True):
        for chunk_id in (primary_id, secondary_id):
            if chunk_id not in emitted_ids:
                merged.append(primary_by_id[chunk_id])
                emitted_ids.add(chunk_id)
    return merged


def ensure_prepare_for_model(tokenizer: Any) -> None:
    """Restore the narrow tokenizer API FlagEmbedding needs on Transformers 5.x.

    FlagEmbedding 1.4 calls ``prepare_for_model`` with already-tokenized query
    and passage IDs. Transformers 5 removed that public helper, while retaining
    the lower-level special-token builders. This adapter implements only the
    fail-closed ``only_second``/no-padding path used by the pinned reranker.
    """

    if hasattr(tokenizer, "prepare_for_model"):
        return

    def prepare_for_model(
        self: Any,
        ids: list[int],
        pair_ids: list[int] | None = None,
        *,
        truncation: str,
        max_length: int,
        padding: bool,
        **_kwargs: Any,
    ) -> dict[str, list[int]]:
        if pair_ids is None or truncation != "only_second" or padding is not False:
            raise ValueError("unsupported FlagEmbedding tokenizer compatibility request")
        tokenizer_type = type(self).__name__
        if tokenizer_type not in {"XLMRobertaTokenizer", "XLMRobertaTokenizerFast"}:
            raise ValueError(f"unsupported reranker tokenizer type: {tokenizer_type}")
        special_tokens = 4
        pair_limit = max_length - len(ids) - special_tokens
        if pair_limit < 0:
            raise ValueError("reranker query exceeds the configured maximum length")
        truncated_pair = pair_ids[:pair_limit]
        if self.bos_token_id is None or self.eos_token_id is None:
            raise ValueError("reranker tokenizer has no BOS/EOS token IDs")
        input_ids = [
            self.bos_token_id,
            *ids,
            self.eos_token_id,
            self.eos_token_id,
            *truncated_pair,
            self.eos_token_id,
        ]
        token_type_ids = [0] * len(input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": [1] * len(input_ids),
            "token_type_ids": token_type_ids,
        }

    tokenizer.prepare_for_model = MethodType(prepare_for_model, tokenizer)


class Reranker:
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        model_revision: str = DEFAULT_RERANKER_MODEL_REVISION,
        device: str = "auto",
        local_files_only: bool = False,
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = resolve_device(device)
        self.local_files_only = local_files_only
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: avoid loading the cross-encoder until first use
            from FlagEmbedding import FlagReranker

            model_path = resolve_model_snapshot(
                self.model_name,
                self.model_revision,
                local_files_only=self.local_files_only,
            )
            self._model = FlagReranker(
                model_path,
                use_fp16=self.device.startswith("cuda"),
                devices=[self.device],
                trust_remote_code=False,
            )
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer is not None:
                ensure_prepare_for_model(tokenizer)
        return self._model

    def rerank_all(self, query: str, candidates: list[RetrievedChunk]) -> list[RetrievedChunk]:
        """Score every candidate and return one complete stable ID permutation."""
        if not candidates:
            return []
        _canonical_chunk_ids(candidates, label="input")
        pairs = [[query, c.payload["text"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, Real):
            scores = [scores]
        else:
            try:
                scores = list(scores)
            except TypeError as exc:
                raise ValueError("reranker score output must be a scalar or sequence") from exc
        if len(scores) != len(candidates):
            raise ValueError("reranker score count must match candidate count")
        reranked = [
            RetrievedChunk(score=float(s), payload=c.payload) for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        _require_exact_permutation(
            _canonical_chunk_ids(candidates, label="input"), reranked, label="reranker"
        )
        return reranked

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        return self.rerank_all(query, candidates)[:top_k]
