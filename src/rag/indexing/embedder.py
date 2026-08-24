"""BGE-M3 dense embedder with a content-hash cache.

The cache makes ablation runs cheap: vectors are keyed by
``sha256(model_name + text)`` and stored in a single SQLite file, so
re-indexing with a different chunking strategy only embeds genuinely new text.
"""

from __future__ import annotations

import hashlib
import sqlite3
import threading
from pathlib import Path

import numpy as np

_SQLITE_VAR_LIMIT = 500  # stay under SQLite's ~999 bound-variable cap


class EmbeddingCache:
    """Safe to share across threads (e.g. FastAPI's request threadpool):
    the connection is opened with ``check_same_thread=False`` and every
    access is serialized through a lock, since SQLite connections still
    aren't safe for concurrent use from multiple threads simultaneously.
    """

    def __init__(self, path: Path):
        path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(path), check_same_thread=False)
        self._lock = threading.Lock()
        with self._lock:
            self.conn.execute(
                "CREATE TABLE IF NOT EXISTS embeddings ("
                "key TEXT PRIMARY KEY, dim INTEGER NOT NULL, vec BLOB NOT NULL)"
            )
            self.conn.commit()

    def get_many(self, keys: list[str]) -> dict[str, np.ndarray]:
        found: dict[str, np.ndarray] = {}
        unique = list(dict.fromkeys(keys))
        with self._lock:
            for i in range(0, len(unique), _SQLITE_VAR_LIMIT):
                batch = unique[i : i + _SQLITE_VAR_LIMIT]
                placeholders = ",".join("?" * len(batch))
                rows = self.conn.execute(
                    f"SELECT key, dim, vec FROM embeddings WHERE key IN ({placeholders})", batch
                )
                for key, dim, blob in rows:
                    found[key] = np.frombuffer(blob, dtype=np.float32).reshape(dim)
        return found

    def put_many(self, items: dict[str, np.ndarray]) -> None:
        with self._lock:
            self.conn.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dim, vec) VALUES (?, ?, ?)",
                [(k, v.shape[0], v.astype(np.float32).tobytes()) for k, v in items.items()],
            )
            self.conn.commit()

    def __len__(self) -> int:
        with self._lock:
            return self.conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]


def resolve_device(device: str) -> str:
    if device != "auto":
        return device
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


class BGEM3Embedder:
    DIM = 1024

    def __init__(
        self,
        model_name: str = "BAAI/bge-m3",
        device: str = "auto",
        cache_path: Path | None = None,
        batch_size: int = 64,
        max_length: int = 1024,
    ):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.batch_size = batch_size
        self.max_length = max_length
        self.cache = EmbeddingCache(cache_path) if cache_path else None
        self._model = None

    @property
    def model(self):
        if self._model is None:  # lazy: loading BGE-M3 takes seconds + VRAM
            from FlagEmbedding import BGEM3FlagModel

            self._model = BGEM3FlagModel(
                self.model_name,
                use_fp16=self.device.startswith("cuda"),
                devices=[self.device],
            )
        return self._model

    def _key(self, text: str) -> str:
        return hashlib.sha256(f"{self.model_name}\n{text}".encode("utf-8")).hexdigest()

    def encode(self, texts: list[str]) -> np.ndarray:
        """Embed ``texts`` (cache-aware), preserving input order."""
        keys = [self._key(t) for t in texts]
        vectors: dict[str, np.ndarray] = self.cache.get_many(keys) if self.cache else {}

        missing = [(i, k) for i, k in enumerate(keys) if k not in vectors]
        if missing:
            new_texts = [texts[i] for i, _ in missing]
            output = self.model.encode(
                new_texts, batch_size=self.batch_size, max_length=self.max_length
            )
            dense = np.asarray(output["dense_vecs"], dtype=np.float32)
            fresh = {}
            for (_, key), vec in zip(missing, dense):
                fresh[key] = vec
                vectors[key] = vec
            if self.cache:
                self.cache.put_many(fresh)

        return np.stack([vectors[k] for k in keys])

    def encode_query(self, query: str) -> np.ndarray:
        return self.encode([query])[0]
