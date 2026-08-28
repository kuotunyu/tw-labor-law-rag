"""BM25 keyword index over chunk payloads, tokenized with jieba.

Persisted as versioned JSON containing only chunk payloads. Loading rebuilds
the small BM25 model instead of deserializing executable Python objects.
"""

from __future__ import annotations

import json
from pathlib import Path

from rank_bm25 import BM25Okapi

from rag.indexing.tokenizer import tokenize
from rag.models import RetrievedChunk


class BM25Index:
    FORMAT = "bm25-payloads-v1"

    def __init__(self, payloads: list[dict], bm25: BM25Okapi):
        self.payloads = payloads
        self.bm25 = bm25

    @classmethod
    def build(cls, chunks_path: Path) -> "BM25Index":
        with open(chunks_path, encoding="utf-8") as f:
            payloads = [json.loads(line) for line in f if line.strip()]
        return cls.from_payloads(payloads)

    @classmethod
    def from_payloads(cls, payloads: list[dict]) -> "BM25Index":
        if not payloads or any(
            not isinstance(payload, dict)
            or not isinstance(payload.get("text"), str)
            or not payload["text"].strip()
            for payload in payloads
        ):
            raise ValueError("BM25 requires valid non-empty text payloads")
        tokenized_corpus = [tokenize(payload["text"]) for payload in payloads]
        return cls(payloads, BM25Okapi(tokenized_corpus))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {"format": self.FORMAT, "payloads": self.payloads},
                ensure_ascii=False,
            ),
            encoding="utf-8",
        )

    @classmethod
    def load(cls, path: Path) -> "BM25Index":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("invalid BM25 JSON artifact") from exc
        if (
            not isinstance(data, dict)
            or data.get("format") != cls.FORMAT
            or not isinstance(data.get("payloads"), list)
        ):
            raise ValueError("invalid BM25 JSON artifact")
        return cls.from_payloads(data["payloads"])

    def search(self, query: str, top_k: int) -> list[RetrievedChunk]:
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        return [RetrievedChunk(score=float(scores[i]), payload=self.payloads[i]) for i in ranked if scores[i] > 0]

    def __len__(self) -> int:
        return len(self.payloads)
