"""Ad-hoc CLI query against the full RAG pipeline (development smoke test).

Usage:
    python scripts/ask.py "加班費怎麼算?"
    python scripts/ask.py "被資遣之後可以領多少失業給付?" --mode hybrid --no-rerank
    python scripts/ask.py "加班費怎麼算?" --retrieve-only   # skip the LLM call
"""

import argparse

import _bootstrap  # noqa: F401

from rag.config import get_settings
from rag.factory import build_answerer, build_retrieval_pipeline
from rag.indexing.embedder import BGEM3Embedder
from rag.indexing.vector_store import VectorStore


def print_hits(hits):
    for rank, hit in enumerate(hits, start=1):
        preview = hit.payload["content"].replace("\n", " ")
        if len(preview) > 80:
            preview = preview[:80] + "…"
        print(f"{rank}. [{hit.score:.4f}] {hit.citation}")
        print(f"   {preview}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query")
    parser.add_argument("--strategy", choices=["structure", "fixed"], default=None)
    parser.add_argument("--mode", choices=["vector", "bm25", "hybrid"], default=None)
    parser.add_argument("--no-rerank", action="store_true")
    parser.add_argument("--provider", choices=["anthropic", "openai", "gemini", "ollama"], default=None)
    parser.add_argument("--retrieve-only", action="store_true", help="skip the LLM call")
    args = parser.parse_args()

    settings = get_settings()
    embedder = BGEM3Embedder(
        model_name=settings.embedding_model,
        model_revision=settings.embedding_model_revision,
        device=settings.device,
        cache_path=settings.storage_dir / "emb_cache.sqlite",
    )
    store = VectorStore(settings)
    use_reranker = not args.no_rerank

    if args.retrieve_only:
        pipeline = build_retrieval_pipeline(
            settings, embedder, store, strategy=args.strategy, mode=args.mode, use_reranker=use_reranker
        )
        result = pipeline.run(args.query)
        print(f"query: {args.query}\n")
        print_hits(result.hits)
        store.close()
        return

    from rag.generation.llm import build_llm

    llm = build_llm(settings, provider=args.provider)
    answerer = build_answerer(
        settings,
        embedder,
        store,
        strategy=args.strategy,
        mode=args.mode,
        use_reranker=use_reranker,
        llm=llm,
    )
    result = answerer.answer(args.query)

    print(f"query: {args.query}")
    print(f"provider: {args.provider or settings.llm_provider} ({llm.model})\n")
    print("=== 檢索結果 ===")
    print_hits(result.retrieval.hits)
    print(f"\n=== 答案 (refused={result.refused}) ===")
    print(result.text)
    if result.sources:
        print("\n=== 引用來源 ===")
        for src in result.sources:
            print(f"[{src['index']}] {src['doc']} {src['article']}")

    store.close()


if __name__ == "__main__":
    main()
