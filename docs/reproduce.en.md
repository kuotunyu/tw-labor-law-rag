# Reproduce, test, and verify

[繁體中文](reproduce.md) | [English](reproduce.en.md)

[README.en.md](../README.en.md) keeps only the shortest path; the full reviewer path and scope notes moved here unchanged.

## Clean reviewer path

Requirements: Python 3.11 and [uv](https://docs.astral.sh/uv/). Dependency installation may access the configured Python package indexes. After dependencies are installed, these checks do not need a model download, provider, API key, Qdrant, Docker, GPU, or runtime network service.

```bash
uv sync --locked
uv lock --check
uv run python scripts/verify_release.py
uv run pytest -q
uv build
uv run python -W error::UserWarning -c "import sys; sys.path.insert(0, 'src'); import rag.api.main; print('FastAPI import: ok')"
uv run python -W error::UserWarning scripts/ask.py --help
```

The package test builds both sdist and wheel and verifies that the runtime legal-term dictionary is included. The release verifier checks the canonical dataset identity, 8×40 trace grid, metrics, the 0.03 score/stage contract, configuration agreement, two source-data snapshots, strict official-trace schemas, the complete publication inventory, privacy/secret patterns, manually reviewed binary hashes, and immutable GitHub Action pins. Its Git-history audit covers every publishable commit reachable from heads, tags, and remotes; GitHub Actions' ephemeral, non-publishable `refs/remotes/pull/*` merge refs and local `refs/archive/*` recovery evidence remain outside the publication graph.

For the full procedure and expected results, see [REVIEWER_GUIDE.md](release/REVIEWER_GUIDE.md). The mapping from each material claim to config, trace, result, and test is in [CLAIM_MATRIX.md](release/CLAIM_MATRIX.md).

## What can and cannot be recomputed

The retrieval, answerability, refusal, citation, configuration, and ablation arithmetic is recomputed by `scripts/verify_release.py`. Faithfulness and relevancy are different: their committed numeric verdicts can be re-aggregated, but the public evidence intentionally excludes complete generated answers, judge reasons, and provider responses. The underlying provider judgments therefore cannot be regenerated or independently re-judged from this repository.

## Running the application

Application use requires the full corpus, indexes, embedding/reranker models, and either a configured provider or local Ollama. These are intentionally outside the offline reviewer path.

```bash
uv sync
cp .env.example .env
uv run python scripts/download_corpus.py
uv run python scripts/build_index.py
uv run python scripts/ask.py "加班費怎麼算?"
uv run python scripts/run_api.py
```

## Scope

This is the `v0.3.5` source-only runtime and deployment release. Its formal model-quality metrics retain the unchanged `v0.1.0` evidence baseline. This release adds a reviewer-first private BYOK interface, a ten-case offline portfolio regression, and a content-free 15-instrument/884-article freshness baseline without presenting the compact demonstration as a new model-quality benchmark. The v0.3.2 Gemini/OpenAI safety cross-check remains archived provider evidence: Gemini observed refusal accuracy `0.8`, citation success `1.0`, and estimated cost `US$0.0022620`; OpenAI observed refusal accuracy `1.0`, citation success `1.0`, and estimated cost `US$0.0026414`. Its strict public trace excludes question/answer text, provider payloads, and credentials, and the cross-check does not replace the formal model-quality baseline. This is an evidence-backed software portfolio artifact, not legal advice or a production legal service. The complete corpus, model weights, private indexes, and raw provider artifacts remain outside this source release.
