# v0.3.6 Route-Aware Refusal Policy Design

**Status:** Superseded by the v0.3.6 Retrieval Coverage Pivot on 2026-08-30.

> **Supersession note (2026-08-30):** This approved route-aware lower-threshold
> design is retained unchanged as the audit trail for its NO-GO evidence. The
> pivot design in `V036_RETRIEVAL_COVERAGE_PIVOT_DESIGN.md` replaces its active
> decision: production remains at the global `0.03` threshold, and the
> severance coverage change is retrieval-only. Do not use this document's
> `0.015` threshold or generation expectation for case `027` as an active
> implementation contract.

**Date:** 2026-08-30

**Release target:** v0.3.6

## 1. Context

The v0.3.1 reliability evidence measures one direct false refusal among forty
answerable stress questions at the production reranker threshold of `0.03`.
The affected case, `stress-037`, is an old-versus-new severance comparison. The
retrieval result places the expected source at rank 2, but its top normalized
reranker score is `0.0175`, so the answerer stops before generation.

Three of twenty unanswerable stress cases score above `0.03`. They ask for
benefit calculations or annually published values that are outside the audited
fifteen-instrument corpus. These are not evidence that the global threshold
should be raised: the existing two-layer design intentionally lets
high-scoring, semantically related questions reach the generation-layer
citation-completeness refusal.

The defect is therefore narrow: one already-reviewed deterministic query route
retrieves the correct authorities but is governed by a global threshold that is
too strict for that route.

## 2. Goals

1. Reduce direct false refusals in the forty-answerable stress set from one to
   zero.
2. Preserve direct unanswerable coverage at or above the existing `0.85`.
3. Preserve the formal forty-question retrieval metrics and zero direct false
   refusals on its thirty answerable cases.
4. Keep the default global threshold at `0.03` for every non-target query.
5. Produce deterministic, content-free, provider-free evidence for the policy.
6. Keep the private demo on free `cpu-basic`, Qdrant Free Tier, and visitor BYOK.

## 3. Non-goals

- Expanding the audited fifteen-instrument corpus.
- Treating one reranker score as a universal answerability classifier.
- Forcing all unanswerable questions to stop at the retrieval layer.
- Adding a classifier, LLM router, owner-funded provider call, or paid hardware.
- Rebuilding, renaming, or writing to the existing Qdrant collections.
- Changing the public UI, provider selection, citation contract, or fallback
  policy.

## 4. Options considered

### 4.1 Route-aware threshold policy — selected

Use an evidence-calibrated threshold only when the applied route set is exactly
`severance_comparison`. All other route combinations use the global `0.03`.
This directly targets the measured defect and limits the blast radius.

### 4.2 Lower the global threshold — rejected

A lower global threshold would admit more semantically related out-of-corpus
questions and would invalidate the existing cross-dataset Pareto decision. It
solves the positive case by weakening unrelated decisions.

### 4.3 Add a learned answerability classifier — rejected

A classifier or LLM gate requires new training or provider evidence, adds
latency and cost, and creates a second model-calibration problem. The observed
defect does not justify that complexity.

## 5. Architecture

### 5.1 Query planning

Add a pure `plan_retrieval_query(question)` function. It returns an ephemeral
`QueryPlan` containing:

- `search_query`: the original question plus any deterministic legal terms;
- `routes`: a normalized tuple of route identifiers.

The initial route identifiers are:

- `off_hours_employer_message`;
- `severance_comparison`;
- `wage_arrears_termination`.

Route matching continues to require every cue group already defined for that
expansion. `_retrieval_query(question)` remains as a compatibility wrapper that
returns only `QueryPlan.search_query`.

`QueryPlan` exists only during one request. Neither the original question nor
the expanded search query is added to logs, public artifacts, API responses, or
stored retrieval results.

### 5.2 Retrieval result

`RetrievalPipeline.run()` uses the plan's search query for both retrieval and
reranking. `RetrievalResult` adds only `applied_routes`, a normalized tuple that
contains no user text or legal content. Existing `hits`, `candidates`, and
`top_score` contracts remain compatible.

### 5.3 Refusal policy

Add a pure refusal-policy component that accepts:

- whether a reranker is active;
- whether any hits exist;
- normalized applied routes;
- the unrounded top reranker score;
- global threshold `0.03`;
- the calibrated severance-comparison threshold.

The decision order is:

1. No hits: refuse at `no_hits`.
2. No reranker: do not apply a score threshold, preserving existing behavior.
3. Applied routes exactly equal `("severance_comparison",)`: use the calibrated
   route threshold.
4. Empty routes, unknown routes, duplicate routes, or any multi-route
   combination: use the global `0.03`.
5. Refuse at `threshold` only when the unrounded top score is strictly below the
   effective threshold. Equality continues to pass to generation.

The answerer and every offline evaluation runner call this same policy. No
runner may reconstruct the decision with its own comparison.

## 6. Calibration evidence

### 6.1 Dataset

Create a thirty-case reviewed dataset:

- fifteen answerable positives covering statutory Chinese, colloquial Chinese,
  code switching, punctuation, long narratives, reversed old/new ordering,
  formula wording, caps, and mixed tenure;
- fifteen collision negatives covering a single regime, ordinary termination,
  notice only, wage arrears, generic retirement, unrelated old/new wording, and
  near-keyword queries that must not receive the route-specific policy.

The dataset may contain the reviewed questions and expected public source
identities. The official result remains content-free.

### 6.2 Candidate selection

Evaluate route thresholds `0`, `0.005`, `0.01`, `0.015`, `0.02`, `0.025`, and
`0.03`. Select the highest candidate that satisfies every gate:

1. All fifteen positives activate exactly `severance_comparison`.
2. Both expected authorities appear within Top 5 for every positive.
3. No positive is directly refused.
4. All fifteen negatives satisfy their route and refusal contracts.
5. The forty-answerable stress set has zero direct false refusals.
6. The twenty-unanswerable stress set retains direct coverage of at least
   `0.85`.
7. The formal set retains Hit@5 and MRR@10 at or above its committed baseline,
   with zero direct false refusals.

If no candidate passes every gate, v0.3.6 is a NO-GO and the production policy
remains unchanged. The algorithm never selects a weaker candidate merely to
ship the release.

### 6.3 Official artifact

The official result records only:

- dataset and corpus snapshot hashes;
- pinned model revisions and retrieval configuration;
- code revision;
- candidate thresholds and aggregate gates;
- per-case qid, answerability, source ranks, applied routes, top score, effective
  threshold, refusal decision, and pass/fail outcome.

It must not contain question text, legal text, endpoints, URLs, credentials,
provider payloads, local paths, account identity, or generated answers. The run
is offline and constructs no LLM adapter.

## 7. Configuration and compatibility

The selected threshold becomes a typed setting and is bound in the release
manifest only after the official artifact passes. Invalid, non-finite, or
out-of-range configuration fails Settings validation at startup. An unknown or
ambiguous runtime route set never receives a lower threshold; it falls back to
the stricter global policy.

Existing callers of `_retrieval_query`, `RetrievalPipeline.run`, and
`RetrievalResult` remain source compatible. The API and Streamlit response
schemas do not gain user-facing controls. Debug evidence may report route names
and the effective threshold, but never the planned search query.

## 8. Testing

Required automated coverage:

1. Query-plan unit tests for each route, no-route questions, multiple routes,
   code switching, and collision negatives.
2. Refusal-policy table tests covering no hits, no reranker, equality, unknown
   routes, duplicate routes, multi-route fallback, and the exact severance-only
   override.
3. Answerer tests proving whether generation is called at each boundary.
4. Runner tests proving production and evaluation use the same policy function.
5. Thirty-case artifact validation and deterministic rebuild tests.
6. Release-verifier positive and tamper tests for dataset hash, model revisions,
   candidate grid, selected threshold, metrics, privacy fields, and code
   revision.
7. Full existing formal, reliability, portfolio, wage-arrears, API, UI,
   security, package, and publication-boundary gates.

## 9. Deployment and rollback

The release reuses the current read-only Qdrant candidate collections. No
writer or transition key is created. The exact candidate tree is uploaded to
the existing private Space without changing visibility, replica count,
hardware, variables, or secrets.

Acceptance uses the free private Space preflight, API/session policy checks,
inventory comparison, and no-provider smoke checks. It sends no Gemini or
OpenAI request. A deployment receipt binds the source commit, Space revision,
free hardware, BYOK policy, and unchanged collection base before the immutable
tag is created.

Rollback restores the recorded v0.3.5 Space revision. Because v0.3.6 changes no
index data or credential scope, Qdrant requires no rollback.

## 10. Acceptance criteria

v0.3.6 may ship only when:

- a candidate threshold is selected by the documented algorithm;
- targeted positives and negatives pass 30/30;
- stress direct false refusals are 0/40;
- stress direct unanswerable coverage is at least 17/20;
- formal retrieval metrics do not regress and formal direct false refusals stay
  at zero;
- all provider request counters for the new evidence are zero;
- existing release, privacy, security, packaging, and CI gates pass;
- the private Space remains private, free, read-only, and BYOK.

There are no unresolved design questions. Implementation must stop rather than
relax any acceptance gate without a new reviewed design.
