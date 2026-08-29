---
title: Taiwan Labor Law RAG
sdk: docker
app_port: 7860
---
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

The separate `v0.3.1 reliability stress evidence` uses 40 answerable and 20 unanswerable long-form/code-switched questions against an isolated rebuild of the audited 2026-08-29 **15-instrument / 884-article** snapshot. It measured Hit@5 **0.950**, MRR@10 **0.908**, one direct false refusal among 40 answerable questions, and direct rejection of 17/20 unanswerable questions. The unchanged formal-set guard simultaneously reproduced Hit@5 **0.967**, MRR@10 **0.906**, zero direct false refusals among 30 answerable questions, and direct rejection of 9/10 unanswerable questions. No threshold candidate was Pareto-better across both sets, so 0.03 was retained.

A fail-closed US$5-per-provider cross-check is complete for Gemini `gemini-3.5-flash-lite` and OpenAI `gpt-5.6-luna`. The public evidence contains only ten de-identified trace rows, recomputable metrics, and each provider's US$5 budget ledger; it excludes provider payloads, credentials, and raw run artifacts.

The 0.03 reranker gate is calibrated only against this formal 30-answerable/10-unanswerable set. It is not a universal answerability classifier. A real-use question outside the formal set, written as a long colloquial narrative with the English word “deadline,” scored 0.0146 and was directly false-refused even though the correct article remained in the candidates. This demonstrates a query-style boundary; the available evidence does not estimate its prevalence.

## Public BYOK Docker Space (live)

**Live demo:** [steven0226-tw-labor-law-rag-demo.hf.space](https://steven0226-tw-labor-law-rag-demo.hf.space)

The portfolio deployment uses BYOK (Bring Your Own Key). A visitor selects Gemini `gemini-3.5-flash-lite` or OpenAI `gpt-5.6-luna` and enters a dedicated key in a masked field. The key exists only in the current Streamlit session, one loopback request header, and one request-scoped provider client. It is never written to files, chat history, shared settings, or cross-request caches. The public Space has no owner `GEMINI_API_KEY` or `OPENAI_API_KEY` and performs no cross-provider fallback, so visitors cannot spend the owner's model-token balance.

The Space receives a collection-scoped read-only Qdrant key. A temporary write/manage key is revoked immediately after the two collections are built locally. Startup scrolls payloads read-only and rebuilds the structure/fixed BM25 indexes in memory; private `data/raw/` and `storage/bm25_*.json` artifacts are not shipped. Defaults are 20 queries per demo session, two concurrent queries globally, a 60-second provider timeout, and at most 1,000 unexpired anonymous sessions. Key isolation, read-only access, and free `cpu-basic` acceptance passed before the Space was made public. See the [BYOK Hugging Face runbook](docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md).

## v0.3.2 reliability, provenance, and dual-model runtime

This is the `v0.3.2` source-only runtime and deployment release. The public API/UI defaults to Gemini `gemini-3.5-flash-lite`. When OpenAI is also configured on the server, a user may select `gpt-5.6-luna` per request. The model names can be overridden independently with server-side `GEMINI_GENERATION_MODEL` and `OPENAI_GENERATION_MODEL`. When its key is configured, `LLM_PROVIDER=gemini` controls the default for a request that omits a provider; otherwise the API uses the other configured public provider. `LLM_FALLBACK_ENABLED=true` permits fallback. `GEMINI_API_KEY` and `OPENAI_API_KEY` remain only in the API server environment: the UI neither accepts, stores, nor displays them.

The fallback boundary is fixed: only an operational failure of the primary provider—such as transport failure, rate limiting, a 5xx service response, or an empty response—may trigger at most one attempt through the other configured public provider. Retrieval-layer refusal does not call a generator. A model refusal based on the retrieved law, a provider safety block, or a policy rejection never falls back. The formal evaluation path continues to bind directly to one generator and one judge provider with runtime fallback off, so routing changes cannot silently change the evaluated configuration.

The Streamlit sidebar's **Answer model** selector shows only configured Gemini/OpenAI entries returned by API `/models`; the selected provider is sent with each `/query`. In a query response, `requested_provider` records the requested route, `provider` and `model` are metadata for the model that actually generated the answer, `fallback_used`/`fallback_from` describe rerouting, and `generation_called=false` means retrieval refused before generation. The UI displays requested and actual models separately and warns when fallback occurred. Live provider smoke tests require local server-side secrets and are outside public offline CI.

The `v0.1.0` formal model-quality metrics remain historical results produced by the generator and judge models recorded in `release/manifest.json`; this runtime release has not replaced or independently re-judged those values. It did rerun retrieval and threshold behavior against both the 60-question stress suite and the 40-question formal set as a regression guard, without calling a provider.

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

The package test builds both sdist and wheel and verifies that the runtime legal-term dictionary is included. The release verifier checks the canonical dataset identity, 8×40 trace grid, metrics, the 0.03 score/stage contract, configuration agreement, two source-data snapshots, strict official-trace schemas, the complete publication inventory, privacy/secret patterns, manually reviewed binary hashes, and immutable GitHub Action pins. Its Git-history audit covers every publishable commit reachable from heads, tags, and remotes; GitHub Actions' ephemeral, non-publishable `refs/remotes/pull/*` merge refs and local `refs/archive/*` recovery evidence remain outside the publication graph.

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

Private raw runs are preserved locally and excluded from the public allowlist. Public official traces other than the provider cross-check do not contain prompts, complete generated answers, judge reasons, provider responses, request identifiers, token usage, API metadata, credentials, private paths, or personal identifiers. Provider cross-check traces publish only strict allowlisted metadata: provider, model, answerability/refusal and citation outcomes, token counts, estimated cost, and elapsed time; they exclude prompts, questions, answers, provider payloads, credentials, private paths, and personal identifiers. See [PUBLICATION_BOUNDARY.md](docs/release/PUBLICATION_BOUNDARY.md).

## Scope

This is the `v0.3.2` source-only runtime and deployment release. Its formal model-quality metrics retain the unchanged `v0.1.0` formal evidence baseline. This release adds a full-corpus snapshot, per-citation legal-source provenance, a 60-question reliability stress benchmark, and hard-capped dual-provider evidence. The Gemini/OpenAI cross-check completed against the fixed models with five requests each under the US$5 cap; the public evidence excludes complete prompts, answers, provider payloads, credentials, and raw run artifacts. It is an evidence-backed software portfolio artifact, not legal advice and not a production legal service. The complete corpus, model weights, private indexes, and raw provider artifacts remain outside this source release.
