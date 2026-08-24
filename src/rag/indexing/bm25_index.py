"""BM25 keyword index over chunk payloads, tokenized with jieba.

Persisted as one pickle per chunking strategy (the BM25Okapi model + the
chunk payloads it was built from), rebuilt whenever ``build_index.py`` runs.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.indexing.tokenizer import tokenize
from rag.models import RetrievedChunk


class BM25Index:
    def __init__(self, payloads: list[dict], bm25: BM25Okapi):
        self.payloads = payloads
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks_path: Path) -> "BM25Index":
        with open(chunks_path, encoding="utf-8") as f:
            payloads = [json.loads(line) for line in f if line.strip()]
        tokenized_corpus = [tokenize(p["text"]) for p in payloads]
        return cls(payloads, BM25Okapi(tokenized_corpus))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump({"payloads": self.payloads, "bm25": self.bm25}, f)

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        with open(path, "rb") as f:
            data = pickle.load(f)
        return cls(data["payloads"], data["bm25"])

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievedChunk(score=float(scores[i]), payload=self.payloads[i]) for i in ranked if scores[i] > 0]

    def __len__(self) -> int:
        return len(self.payloads)
