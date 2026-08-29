"""Build the vector index (and chunk store) from the downloaded corpus.

For each chunking strategy this produces:
  - a Qdrant collection ``<collection_name>_<strategy>``
  - ``storage/chunks_<strategy>.jsonl`` — the chunk payloads (reused by the
    BM25 index in Phase 2, and handy for debugging)

Usage:
    python scripts/build_index.py [--strategy structure|fixed|all] [--corpus data/raw/laws]
"""

import argparse
import json
import time
from pathlib import Path

import _bootstrap  # noqa: F401  (sys.path + stdout encoding)

from rag.config import PROJECT_ROOT, get_settings
from rag.indexing.bm25_index import BM25Index
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore
from rag.ingestion.chunkers import get_chunker
from rag.ingestion.loader import load_corpus
from rag.retrieval.retriever import bm25_path_for, collection_for


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--strategy", choices=["structure", "fixed", "all"], default="all")
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data" / "raw" / "laws")
    args = parser.parse_args()

    settings = get_settings()
    strategies = ["structure", "fixed"] if args.strategy == "all" else [args.strategy]

    units = load_corpus(args.corpus)
    docs = {u.doc_id for u in units}
    print(f"[load] {len(units)} units from {len(docs)} documents ({args.corpus})")

    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    store = VectorStore(settings)

    for strategy in strategies:
        chunker = get_chunker(strategy, settings.chunk_size, settings.chunk_overlap)
        chunks = chunker.chunk(units)
        lengths = [len(c.content) for c in chunks]
        print(
            f"[chunk] {strategy}: {len(chunks)} chunks "
            f"(len avg {sum(lengths) / len(lengths):.0f}, max {max(lengths)})"
        )

        chunks_path = settings.storage_dir / f"chunks_{strategy}.jsonl"
        chunks_path.parent.mkdir(parents=True, exist_ok=True)
        with open(chunks_path, "w", encoding="utf-8") as f:
            for chunk in chunks:
                f.write(json.dumps(chunk.payload(), ensure_ascii=False) + "\n")
        print(f"[write] {chunks_path}")

        t0 = time.perf_counter()
        vectors = embedder.encode([c.text for c in chunks])
        print(f"[embed] {len(vectors)} vectors in {time.perf_counter() - t0:.1f}s "
              f"(device={embedder.device})")

        collection = collection_for(settings, strategy)
        store.recreate_collection(collection, dim=vectors.shape[1])
        store.upsert_chunks(collection, chunks, vectors)
        print(f"[index] collection '{collection}': {store.count(collection)} points")

        t0 = time.perf_counter()
        bm25 = BM25Index.build(chunks_path)
        bm25_path = bm25_path_for(settings, strategy)
        bm25.save(bm25_path)
        print(f"[bm25] {len(bm25)} docs indexed in {time.perf_counter() - t0:.1f}s -> {bm25_path}\n")

    store.close()


if __name__ == "__main__":
    main()
