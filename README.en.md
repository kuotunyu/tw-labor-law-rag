---
title: Taiwan Labor Law RAG
sdk: docker
app_port: 7860
---
# Traditional Chinese Hybrid RAG for Taiwan Labor Law

[繁體中文](README.md) | [English](README.en.md)

[![CI](https://github.com/kuotunyu/tw-labor-law-rag/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/kuotunyu/tw-labor-law-rag/actions/workflows/ci.yml)

Ask a plain-language question about Taiwanese labour law (how overtime pay is calculated, how many days of annual leave, what severance is owed) and the system finds the relevant articles among 15 labour statutes (884 articles), answers with the statute name, article number, and official source link, and refuses when the law does not support an answer. It is aimed at workers, HR staff, and job seekers who need the legal basis quickly.

> **TL;DR** — Hybrid RAG: BM25 + BGE-M3 dense retrieval fused with RRF, reranked by `bge-reranker-v2-m3`, article-level citations, two-stage refusal. FastAPI + Streamlit, Docker.

![Streamlit UI demo: a marriage-leave question answered with cited sources and the retrieval debug panel](docs/screenshot-demo.png)

## Results

| Metric | Result | How to read it |
|---|---:|---|
| Retrieval Hit@5 | **0.967** (29/30) | Share of the 30 answerable questions whose correct article is in the top 5 |
| Retrieval MRR@10 | **0.906** | How high the correct article ranks (1.0 = always first) |
| Refusal of unanswerable questions | **10/10** | 9 stopped at retrieval, without calling the LLM |
| False refusals of answerable questions | **1/30** | At the LLM stage; 0/30 at the retrieval threshold |
| Faithfulness / answer relevancy | **4.90 / 5.00** (out of 5) | 29 answered questions, scored by an LLM judge |
| Knowledge base | **15 instruments / 884 articles** | 13 acts and 2 regulations; snapshot audited 2026-08-29 |

These numbers come from a 40-question evaluation set written for this project: 30 answerable questions covering all 15 instruments plus 10 deliberately unanswerable ones, each checked by hand against the statute text. The primary configuration is `structure-aware / hybrid + reranker`.

**Links:** [evaluation report](EVAL_REPORT.md) | [design trade-offs](DESIGN.md) | [per-question evaluation records](eval/official/README.md) | [繁體中文](README.md). The hosted demo is a private Space on Hugging Face (invitation only, URL not listed); it can also be run locally:

```bash
uv sync                                    # Python 3.11 + uv
cp .env.example .env                       # add a Gemini or OpenAI API key
uv run python scripts/download_corpus.py   # official open data
uv run python scripts/build_index.py       # vector + BM25 indexes
uv run python scripts/ask.py "加班費怎麼算?"
```

See [docs/reproduce.en.md](docs/reproduce.en.md) for the API and the offline reviewer path.

## How it works

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

- **Two-stage refusal:** a top reranker score below 0.03 refuses before generation; above it, the generator is asked to refuse when the retrieved law is insufficient.
- **Three hand-written domain query-expansion rules** ([`src/rag/retrieval/pipeline.py`](src/rag/retrieval/pipeline.py)): when a question contains specific colloquial cues together, fixed statute terms are appended to the retrieval query only — (1) employer + off-hours + messaging cues add rest-day and working-time terms; (2) severance + new-regime + old-regime + calculation cues add Labor Pension Act, Labor Standards Act, service-years, average-wage, and six-month terms; (3) wage-nonpayment + immediate-resignation cues add Labor Standards Act Article 14 terms. BM25, dense retrieval, and the reranker see the expanded string; the generator still receives the original question. A string match of the current code against the evaluation questions shows the rules fire on 2 of the 40 formal questions, 4 of the 60 stress questions, and 2 of the 10 demonstration-regression questions. The formal-set numbers above come from result files produced at v0.1.0 (2026-08-24), before these rules existed (two were added on 08-29 and 08-30), and were not recomputed afterwards, so they do not include any effect of the rules. Rule 3 was written after the fact for the formal set's single failure (eval-10); the scores were deliberately left unrecomputed to avoid tuning on the test questions (see [EVAL_REPORT.md](EVAL_REPORT.md)).

## Results in detail

**Contribution of each retrieval stage** (eight-way ablation × 40 questions; structure-aware chunking shown):

| Retrieval | Hit@5 | MRR@10 |
|---|---:|---:|
| BM25 only | 0.833 | 0.672 |
| Vector only | 0.900 | 0.850 |
| Hybrid (RRF) | 0.933 | 0.822 |
| **Hybrid + reranker (primary)** | **0.967** | **0.906** |

Fusion raises recall but hurts ranking; the reranker restores it. Fixed-size chunking with hybrid + reranker scores a higher Hit@5 (1.000) and a lower MRR@10 (0.847); structure-aware chunking is the primary configuration because the correct article ranks higher and citations resolve to a single article. See [EVAL_REPORT.md](EVAL_REPORT.md) for all eight configurations and the failure analysis.

**Refusal:** all 10 unanswerable questions were refused — 9 directly by the 0.03 threshold and 1 (`eval-32`) by the generator. The single false refusal (`eval-10`, 1/30) also happened at the generator: retrieval missed the correct article and the generator refused rather than guess.

**Stress set:** a separate 60-question set (40 answerable, 20 unanswerable) of long-form, code-switched questions, run against an isolated rebuild of the 2026-08-29 **15-instrument / 884-article** snapshot, measured Hit@5 **0.950**, MRR@10 **0.908**, one direct false refusal among 40 answerable questions, and direct rejection of 17/20 unanswerable questions. The same run reproduced the formal-set 0.967 / 0.906, 0/30, and 9/10. No threshold candidate was better on both sets, so 0.03 was retained.

**Two-provider safety check** (`v0.3.2 provider safety cross-check`): five requests per provider under a US$5-per-provider cap that aborts when exceeded, for Gemini `gemini-3.5-flash-lite` and OpenAI `gpt-5.6-luna`. Gemini observed refusal accuracy `0.8`, citation success `1.0`, and estimated cost `US$0.0022620`; OpenAI observed refusal accuracy `1.0`, citation success `1.0`, and estimated cost `US$0.0026414`. With five requests each this is a safety cross-check, not a model-quality evaluation, and it does not replace the `v0.1.0` formal results. The public trace is strictly content-free: it excludes question/answer text, provider payloads, and credentials.

**Offline demonstration regression and freshness:** a ten-case, fully offline regression (no LLM calls): all six answerable cases retrieve the required articles and all ten retrieval-stage decisions match expectations. A per-article SHA-256 fingerprint of the 15 instruments / 884 articles supports a manual audit for added, removed, or changed articles; nothing is scheduled automatically.

<a id="scope"></a>

## Scope and limitations

- **The evaluation sets are small and written for this project:** 40 formal questions (30 answerable) and 60 stress questions. The numbers describe behaviour on these questions and do not estimate real-world prevalence.
- **The formal numbers are the `v0.1.0` results** and later releases did not rewrite them. This is the `v0.3.5` source-only runtime and deployment release: source code and deployment configuration only; the complete corpus, model weights, private indexes, and raw provider artifacts are outside the repository.
- **Faithfulness and relevancy are archived LLM-judge scores:** the committed numeric verdicts can be re-aggregated, but complete generated answers and judge reasons are not published, so they cannot be re-judged from this repository. Retrieval and refusal numbers can be recomputed offline.
- **The 0.03 reranker threshold is not a universal answerability classifier.** The stress set measured one direct false refusal among 40 answerable questions. A real-use question outside the formal set, written as a long colloquial narrative with the English word “deadline,” scored 0.0146 and was directly false-refused even though the correct article remained in the candidates; the available evidence does not estimate how often this happens.
- **The knowledge base covers only these 15 instruments (snapshot audited 2026-08-29)**; it is not a general legal database, and statute amendments require a manual re-audit and index rebuild.
- This is a software portfolio artifact, not legal advice or a production legal service.

## Reproduce and test

```bash
uv run python scripts/verify_release.py   # recompute the committed evaluation numbers offline
uv run ruff check .
uv run pytest -q
```

`verify_release.py` needs no model, API key, Qdrant, or Docker: it recomputes the retrieval and refusal numbers above from the committed per-question records and checks the public file list and privacy/secret scan. The full list of checks and the clean reviewer path are in [docs/reproduce.en.md](docs/reproduce.en.md).

## Data, license, and publication boundary

The full 15-instrument corpus is downloaded at runtime and is not distributed in the repository. Two small regulation samples are distributed for loader/chunking smoke tests:

- `data/sample/勞工請假規則.json`
- `data/sample/勞動基準法施行細則.json`

They are normalized extracts from the Ministry of Justice Department of Information Management dataset [中文法規_命令資料檔下載](https://data.gov.tw/dataset/18290), published under Taiwan's [Open Government Data License 1.0](https://data.gov.tw/license). OGDL permits reproduction, distribution, adaptation, and sublicensing when its attribution requirement is retained. The samples remain under OGDL; the repository's original code is under the [MIT License](LICENSE).

Private raw runs are preserved locally and excluded from the public allowlist. Public official traces other than the provider cross-check do not contain prompts, complete generated answers, judge reasons, provider responses, request identifiers, token usage, API metadata, credentials, private paths, or personal identifiers. Provider cross-check traces publish only strict allowlisted metadata: provider, model, answerability/refusal and citation outcomes, token counts, estimated cost, and elapsed time; they exclude prompts, questions, answers, provider payloads, credentials, private paths, and personal identifiers.

## Further reading

- [DESIGN.md](DESIGN.md) — design decisions and trade-offs
- [EVAL_REPORT.md](EVAL_REPORT.md) — full evaluation tables, ablation, and failure analysis
- [eval/official/README.md](eval/official/README.md) — published evaluation artifacts
- [docs/changelog.en.md](docs/changelog.en.md) — v0.3.2–v0.3.5 release notes
- [docs/reproduce.en.md](docs/reproduce.en.md) — clean reviewer path, running the application, and what `verify_release.py` checks
- [docs/private-demo.en.md](docs/private-demo.en.md) — how the private demo handles keys
- [docs/release/](docs/release/REVIEWER_GUIDE.md) — release and audit documents: [three-minute tour](docs/release/V035_REVIEWER_TOUR.md), [interview demo](docs/release/V035_INTERVIEW_DEMO.md), [claim-to-evidence table](docs/release/CLAIM_MATRIX.md), [OGDL attribution and hashes](docs/release/OGDL_ATTRIBUTION.md), [publication boundary](docs/release/PUBLICATION_BOUNDARY.md)
