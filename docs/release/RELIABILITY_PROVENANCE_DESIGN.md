# v0.3.1 Reliability and Corpus Provenance Design

## Status and decision authority

The owner approved autonomous continuation while away from the computer and set a hard
external-model budget of **US$5 for Gemini and US$5 for OpenAI**. This design keeps those
limits as release invariants. Work that does not need a provider runs first; provider calls
are optional until credentials are available and must stop before either cap can be exceeded.

The existing `v0.1.0` 40-question formal evidence remains immutable. New measurements are
reported as a separate `v0.3.1` reliability stress benchmark and must never be blended into,
or presented as a replacement for, the historical formal metrics.

## Evidence motivating this work

- The formal set has 40 questions, only four `scenario` questions, no question of 40 or more
  characters, and no Chinese/ASCII code-switching question.
- A real-use narrative question containing `deadline` was false-refused at the retrieval
  threshold even though the correct article remained in the candidate set.
- A live read-only audit on 2026-08-29 successfully extracted all 15 target laws from the
  Ministry of Justice acts and regulations XML dumps. The public release manifest currently
  records hashes for only two redistributed samples, not the complete private corpus snapshot.
- `load_law_json` currently discards `url`, `last_amended`, and `effective_date`, so indexed
  chunks and API citations cannot prove which legal snapshot they came from.
- The workstation has an RTX 4090 and immutable cached snapshots of BGE-M3 and
  bge-reranker-v2-m3, so the retrieval benchmark can run locally without provider cost.

## Considered approaches

### A. Separate stress benchmark plus provenance (selected)

Preserve the formal baseline, add a deliberately difficult question set, publish a
privacy-safe corpus snapshot manifest, propagate legal-source metadata through retrieval,
and measure the current gate before changing production behavior.

This provides the strongest evidence with the lowest risk. It is reproducible, mostly
offline, and does not manufacture a better historical score by changing the old dataset.

### B. Lower the global 0.03 threshold immediately

This would fix the one observed false refusal but would also send more unanswerable questions
to the LLM. The current 30/10 formal set is too small and stylistically uniform to justify the
trade-off. This approach is rejected until the stress benchmark supplies stronger evidence.

### C. Replace the formal benchmark with a provider-generated benchmark

This creates larger sample volume quickly, but it weakens provenance, spends provider budget,
and makes the new numbers incomparable with `v0.1.0`. Provider generation may assist with
paraphrases, but every committed question must retain human-auditable ground-truth sources and
the resulting set stays separate from formal evidence.

## Architecture

### 1. Corpus snapshot and live freshness audit

Add a focused corpus-audit module and CLI that reuse the official Ministry of Justice source
URLs and the exact 15-law allowlist already used by `download_corpus.py`.

The committed snapshot records:

- schema version and snapshot date;
- both official source URLs and SHA-256 hashes of the downloaded ZIP bytes;
- for each sorted law: normalized name, nature, official URL, latest amendment date,
  effective date, non-deleted article count, and SHA-256 of deterministic normalized JSON;
- aggregate law and article counts.

`--write` deliberately refreshes the committed snapshot. The default/check mode downloads the
current official dumps and reports additions, removals, amendment-date changes, article-count
changes, and content-hash changes without writing repository files. CI validates schema and
internal hashes offline; CI does not depend on government network availability.

### 2. Provenance propagation

Extend `SourceUnit` and `Chunk` with optional `source_url`, `last_amended`, and
`effective_date` fields. Law JSON loading populates them; Markdown, text, and PDF loaders keep
empty defaults. Both chunking strategies preserve them in payloads.

The answer source object and API `SourceOut` expose these optional fields. The Streamlit source
panel shows the latest amendment date and an official-law link when present. Old Qdrant points
without the fields remain valid, so code deployment can precede collection rebuilding.

No absolute local source path, credential, raw provider response, or private corpus text is
added to the public release.

### 3. Reliability stress dataset

Add a versioned JSONL dataset separate from `eval_set.jsonl`:

- 40 answerable questions, including one narrative or code-switching variant for every one of
  the 30 formal answerable questions plus 10 high-risk variants;
- 20 unanswerable questions: 10 domain-related and 10 unrelated;
- coverage of all 15 target laws;
- explicit `base_qid`, `style_tags`, `answerable`, `q_type`, and existing ground-truth source
  references;
- at least 30 questions of 40 or more characters, at least 15 code-switching questions, and
  at least 15 narrative/scenario questions.

Questions may be drafted with a provider, but deterministic tests enforce identifiers,
counts, source shape, diversity constraints, absence of secrets/PII, and separation from the
formal evidence hash. Ground truth is inherited from already verified formal questions or is
checked against the audited official snapshot.

### 4. Offline retrieval and threshold analysis

Add a runner for the production configuration (`structure + hybrid + reranker`) that writes
raw runs only under ignored `eval/runs/`. A privacy-reduced committed trace may contain qid,
rank, top score, threshold decision, latency, and expected answerability; it may not contain
full generated answers, provider payloads, secrets, or host paths.

The report includes Hit@5, MRR@10, direct false-refusal rate, direct unanswerable coverage, and
a threshold sweep over fixed candidate values. It reports evidence; it does not automatically
change the production threshold.

The production threshold may change only if a candidate is Pareto-better on the stress set,
does not regress the immutable formal retrieval evidence, and passes all existing refusal
contracts. Otherwise `0.03` remains unchanged and the report states why.

### 5. Bounded provider cross-check

An optional provider-evaluation wrapper uses the existing request-scoped adapters. It accepts
explicit per-provider budgets with defaults of zero and refuses to run without a positive cap.
For this authorized run, caps are `gemini=5.00` and `openai=5.00` US dollars.

The wrapper:

- validates fixed maxima of at most 20,000 input and 1,024 output tokens before the first request;
- bounds the actual system + user prompt conservatively from UTF-8 bytes plus a 1,024-token message envelope before every request;
- records provider/model, request count, input/output token usage when returned, estimated
  cost, and remaining cap;
- stops before the next request when its conservative maximum could exceed the cap;
- never prints or persists API keys;
- keeps raw answers and judge reasons strictly beneath ignored `eval/runs/`; an arbitrary output path outside it is rejected before provider I/O;
- exports only aggregate and privacy-reduced verdict evidence.

If credentials are not available, all offline work still completes and the provider phase is
reported as pending credentials rather than silently substituting another model.

## Failure handling and rollback

- Official-download failure leaves the committed snapshot untouched and produces a non-zero
  diagnostic with the source that failed.
- Missing target laws, abolished-law matches, unsafe XML, duplicate normalized names, and hash
  mismatch are hard failures.
- Model-loading or local-index failure does not trigger paid-provider calls.
- Provider 401/403/429/5xx errors are recorded without cross-provider fallback unless the
  selected evaluation explicitly requests both providers.
- Every cloud mutation remains out of this milestone. Updating Qdrant requires a new temporary
  writer key and a separate reviewed execution step; the current runtime key remains read-only.
- Code rollback is the previous Git commit. Snapshot rollback is the previous committed JSON.

## Verification and release gates

1. Baseline and final full test suites pass.
2. Corpus snapshot unit tests cover deterministic hashing, all 15 laws, schema validation,
   change detection, and fail-closed download/XML behavior.
3. Loader/chunker/answer/API/UI tests cover optional provenance and old-payload compatibility.
4. Stress dataset tests prove size, style diversity, law coverage, source integrity, and formal
   evidence immutability.
5. Retrieval runner tests use fakes for deterministic threshold metrics; the real local run is
   captured separately.
6. Budget tests prove no request can begin when its conservative maximum crosses either cap.
7. Ruff, Bandit, pip-audit policy, release verifier, privacy scan, build, and package smoke pass.
8. Changes integrate only through a protected GitHub PR and required CI.
9. Hugging Face remains public, healthy, and on free `cpu-basic`; no owner provider key is added.

## Explicitly deferred

- Rebuilding the two Qdrant collections and revoking the new writer key, because no writer
  credential is currently available.
- Automatic recurring monitoring, because the owner explicitly deleted that schedule.
- A production threshold change unless the new evidence meets the gate above.
