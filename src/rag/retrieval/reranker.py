"""Cross-encoder reranking with bge-reranker-v2-m3.

Scores are sigmoid-normalized to roughly [0, 1] (``normalize=True``), which is
what makes ``rerank_score_threshold`` in :mod:`rag.config` meaningful — raw
cosine/BM25/RRF scores from the first-stage retrievers are not comparable
across queries (Phase 1 finding: unanswerable questions scored just as high as
answerable ones), but the cross-encoder score is.
"""

from __future__ import annotations

from types import MethodType
from typing import Any

from rag.config import DEFAULT_RERANKER_MODEL_REVISION
from rag.indexing.embedder import resolve_device
from rag.models import RetrievedChunk


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
    ):
        self.model_name = model_name
        self.model_revision = model_revision
        self.device = resolve_device(device)
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: avoid loading the cross-encoder until first use
            from FlagEmbedding import FlagReranker

            self._model = FlagReranker(
                self.model_name,
                use_fp16=self.device.startswith("cuda"),
                devices=[self.device],
                revision=self.model_revision,
                trust_remote_code=False,
            )
            tokenizer = getattr(self._model, "tokenizer", None)
            if tokenizer is not None:
                ensure_prepare_for_model(tokenizer)
        return self._model

    def rerank(self, query: str, candidates: list[RetrievedChunk], top_k: int) -> list[RetrievedChunk]:
        if not candidates:
            return []
        pairs = [[query, c.payload["text"]] for c in candidates]
        scores = self.model.compute_score(pairs, normalize=True)
        if isinstance(scores, float):
            scores = [scores]
        reranked = [
            RetrievedChunk(score=float(s), payload=c.payload) for c, s in zip(candidates, scores)
        ]
        reranked.sort(key=lambda h: h.score, reverse=True)
        return reranked[:top_k]
