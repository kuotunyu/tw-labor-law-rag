# Traditional Chinese Hybrid RAG for Taiwan Labor Law

[繁體中文](README.md) | [English](README.en.md)

[![CI](https://github.com/kuotunyu/tw-labor-law-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kuotunyu/tw-labor-law-rag/actions/workflows/ci.yml)

An evidence-oriented retrieval-augmented generation system targeting 15 Taiwan labor-law instruments (13 acts and 2 regulations). It combines BM25 and BGE-M3 dense retrieval with Reciprocal Rank Fusion, reranks candidates with `bge-reranker-v2-m3`, and generates answers with article-level citations. A two-stage refusal policy rejects low-scoring retrievals before generation and asks the generator to refuse when the retrieved law is insufficient.

## Verified portfolio results

The primary `structure-aware / hybrid + reranker` configuration was evaluated on a committed 40-question set: 30 answerable questions covering all 15 target instruments and 10 unanswerable questions.

| Claim | Result | Public evidence status |
|---|---:|---|
| Retrieval Hit@5 | 0.967 (29/30) | Fully recomputable offline from committed trace |
| Retrieval MRR@10 | 0.906 | Fully recomputable offline from committed trace |
| Final unanswerable refusal accuracy | 10/10 | Fully recomputable offline from committed trace |
| Direct threshold refusals | 9/10 unanswerable; 0/30 answerable | Fully recomputable offline from committed trace |
| Generator-layer refusals | 2 (`eval-32` correct, `eval-10` false refusal) | Fully recomputable as recorded outcomes |
| Faithfulness | 4.90/5 across 29 judged answers | Re-aggregated archived provider evidence |
| Answer relevancy | 5.00/5 across 29 judged answers | Re-aggregated archived provider evidence |

The retrieval, answerability, refusal, citation, configuration, and ablation arithmetic is recomputed by `scripts/verify_release.py`. Faithfulness and relevancy are different: their committed numeric verdicts can be re-aggregated, but the public evidence intentionally excludes complete generated answers, judge reasons, and provider responses. The underlying provider judgments therefore cannot be regenerated or independently re-judged from this repository.

The 0.03 reranker gate is calibrated only against this formal 30-answerable/10-unanswerable set. It is not a universal answerability classifier. A real-use question outside the formal set, written as a long colloquial narrative with the English word “deadline,” scored 0.0146 and was directly false-refused even though the correct article remained in the candidates. This demonstrates a query-style boundary; the available evidence does not estimate its prevalence.

## Architecture

```text
law JSON / Markdown / text / PDF
  -> loader and cleaner
  -> structure-aware articles or 400-character / 80-overlap windows
  -> BGE-M3 vector retrieval (top 20) + jieba/BM25 retrieval (top 20)
  -> RRF (k=60)
  -> bge-reranker-v2-m3
  -> top 5
  -> threshold refusal (<0.03) or generator
  -> answer with numbered citations or generator-layer refusal
  -> FastAPI / Streamlit
```

The measured eight-way ablation covers both chunking strategies and BM25, vector, hybrid, and hybrid-plus-reranker retrieval. See [DESIGN.md](DESIGN.md) for trade-offs and [EVAL_REPORT.md](EVAL_REPORT.md) for the complete table and failure analysis.

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

The package test builds both sdist and wheel and verifies that the runtime legal-term dictionary is included. The release verifier checks the canonical dataset identity, 8×40 trace grid, metrics, the 0.03 score/stage contract, configuration agreement, two source-data snapshots, strict official-trace schemas, the complete publication inventory, privacy/secret patterns, manually reviewed binary hashes, and immutable GitHub Action pins. Its Git-history audit covers every publishable commit reachable from heads, tags, and remotes; local `refs/archive/*` recovery evidence is preserved outside the publication graph.

For the full procedure and expected results, see [docs/release/REVIEWER_GUIDE.md](docs/release/REVIEWER_GUIDE.md). The mapping from each material claim to config, trace, result, and test is in [docs/release/CLAIM_MATRIX.md](docs/release/CLAIM_MATRIX.md).

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

## Data, license, and publication boundary

The full 15-instrument corpus is downloaded at runtime and is not distributed in the repository. Two small regulation samples are distributed for loader/chunking smoke tests:

- `data/sample/勞工請假規則.json`
- `data/sample/勞動基準法施行細則.json`

They are normalized extracts from the Ministry of Justice Department of Information Management dataset [中文法規_命令資料檔下載](https://data.gov.tw/dataset/18290), published under Taiwan's [Open Government Data License 1.0](https://data.gov.tw/license). OGDL permits reproduction, distribution, adaptation, and sublicensing when its attribution requirement is retained. The samples remain under OGDL; the repository's original code is under the [MIT License](LICENSE). See [OGDL_ATTRIBUTION.md](docs/release/OGDL_ATTRIBUTION.md) for the retained attribution and snapshot hashes.

Private raw runs are preserved locally and excluded from the public allowlist. Public official traces do not contain prompts, complete generated answers, judge reasons, provider responses, request identifiers, token usage, API metadata, credentials, private paths, or personal identifiers. See [PUBLICATION_BOUNDARY.md](docs/release/PUBLICATION_BOUNDARY.md).

## Scope

This is the `v0.2.0` source-only reliability release. Its formal model-quality metrics retain the unchanged `v0.1.0` formal evidence baseline; this release hardens publishable-history auditing, corpus-download integrity, and the Ollama thinking boundary without claiming a newly executed provider benchmark. It is an evidence-backed software portfolio artifact, not legal advice and not a production legal service. The complete corpus, model weights, indexes, provider artifacts, and a hosted deployment are outside the publication scope.
