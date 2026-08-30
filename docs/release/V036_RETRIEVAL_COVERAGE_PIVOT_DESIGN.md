# v0.3.6 Retrieval Coverage Pivot Design

Status: evidence-driven pivot approved by the project controller on 2026-08-30.

This document supersedes the proposed `0.015` production threshold in
`V036_ROUTE_AWARE_REFUSAL_POLICY_DESIGN.md`. It does not erase that design or
its NO-GO evidence; those files remain the audit trail explaining why the
pivot was necessary.

## Decision summary

v0.3.6 will improve severance-comparison retrieval coverage without lowering
the production refusal threshold.

- Production keeps one global reranker threshold: `0.03`.
- Route metadata remains deterministic and is used for retrieval behavior,
  not for a lower refusal threshold.
- An exact singleton `severance_comparison` route receives one additional
  old-regime semantic rerank view over the existing fused Top-20 pool.
- The primary and secondary reranker orders are merged by deterministic,
  primary-first, deduplicating interleave.
- The primary query's original top score remains the only score used by the
  refusal gate. Secondary-view scores never change admission.
- Collision case `severance-policy-027` is expected to stop at the retrieval
  threshold. It remains unanswerable, route-negative, and a strict collision
  canary; only its over-specified generation-layer expectation changes.
- The release must be calibrated again from a clean committed revision on the
  deployment-equivalent CPU/FP32 path. Production remains fixed at `0.03`;
  the seven historical candidates are an evaluation-only ablation. If its
  highest passing candidate is not `0.03`, release remains NO-GO.

## Why the former design is no longer valid

The original `0.015` proposal was motivated by a v0.3.1 public trace where
`stress-037` scored `0.0175`. The severance query expansion shipped later, in
v0.3.3. With the current pinned pipeline the fresh score is
`0.3114268601959007`, and every candidate from `0` through `0.03` produces the
same passing stress and formal metrics.

The current Task 6 evidence is therefore a correct NO-GO for its checked-in
contract, but it does not support a lower production threshold. Shipping
`0.015` would increase generation admission without a measured benefit.

The three target failures have different causes:

- `010` and `014`: `勞動基準法|第 17 條` is present in the fused candidate
  pool at rank 5, then the cross-encoder moves it to ranks 6 and 9 before the
  final Top-5 cut.
- `027`: no special route applies and its score `0.02350945240753301` is below
  the unchanged global `0.03`. Direct refusal is the intended safe behavior.

## Retrieval behavior

### Query planning

`plan_retrieval_query()` remains the single deterministic planner. A plan
contains:

- the primary expanded search query;
- the ordered route tuple;
- zero or one rerank-only semantic view in v0.3.6.

Only exact routes `("severance_comparison",)` receive the old-regime view:

```text
勞基法舊制 資遣費 每滿一年 一個月平均工資 未滿一年 比例計給
```

The view contains no qid, source identifier, article number, payload-derived
term, or user-specific value. Multi-route and collision shapes receive no
secondary view.

### Candidate retrieval

Dense retrieval, BM25 retrieval, RRF `k=60`, and the fused Top-20 candidate
pool remain unchanged. The second view reranks that same pool; it does not
perform another embedding search or modify Qdrant.

### Deterministic merge

For an exact severance-comparison route:

1. Rerank all fused candidates with the primary expanded query.
2. Rerank the same candidates with the old-regime view.
3. Walk both stable rankings by depth.
4. At each depth, emit the primary item first, then the secondary item.
5. Deduplicate the expected cross-ranking repetition by canonical `chunk_id`.
6. Materialize every emitted item with its primary-query score.
7. Cut the merged order to `top_k_final`.

Candidate IDs must be non-blank and unique. Each reranker result must contain
every candidate ID exactly once and no other ID; score-count mismatches,
truncated output, duplicates within one ranking, or unequal ID sets fail
closed. Repetition between the primary and secondary permutations is expected
and is the only repetition removed by interleave. Stable behavior must not
depend on dictionary or set iteration order.

The primary ranking's first score is stored separately as `top_score` before
the merge. The secondary score is ranking-only evidence and must never be used
by the refusal policy. The merged order is authoritative and must not be
re-sorted by the retained primary scores; public hit scores may therefore be
non-monotonic.

For all other route shapes, the existing one-pass reranker behavior is
unchanged.

## Refusal behavior

Production answer generation uses a single pure refusal decision:

- no hits -> direct `no_hits` refusal;
- reranker disabled -> no score-threshold refusal;
- reranker enabled and primary `top_score < 0.03` -> direct `threshold`
  refusal;
- otherwise -> generation layer.

The production Settings, Answerer, factory, public traces, and runners must not
carry an active `severance_comparison_score_threshold` override. Route tuples
may still be published as content-free retrieval evidence.

Calibration code may sweep hypothetical route candidates to demonstrate that
no lower threshold is needed, but the production runtime must remain global.
Artifacts distinguish `production_threshold=0.03` from
`route_ablation.highest_passing_candidate`; the latter is not a production
selection.

## Dataset contract correction

`severance-policy-027` remains:

- `case_type=collision_negative`;
- `answerable=false`;
- no required source or route;
- `severance_comparison` prohibited.

Its dataset contract changes from a generation boolean to an exact outcome
enum. `027` requires `expected_outcome="threshold"`, empty routes, positive hit
count, primary score below global `0.03`, refusal stage `threshold`, and no
generation. A `no_hits` decision does not pass. The related `024` case requires
`expected_outcome="generation"` and remains the high-scoring admission
collision canary. Every other row is migrated from its existing boolean to an
equivalent exact outcome without changing meaning.

## Acceptance gates

All gates are fail-closed:

1. Fifteen severance-comparison positives have routes exactly equal to
   `("severance_comparison",)` and retrieve both canonical authorities within
   Top 5.
2. All thirty target contracts pass, including exact refusal-stage behavior.
3. Forty answerable stress cases have zero direct false refusals.
4. Twenty unanswerable stress cases retain at least `17/20` direct refusals.
5. Formal Hit@5 is at least `0.9666666666666667`.
6. Formal MRR@10 is at least `0.9055555555555554`.
7. Formal answerable direct false refusals remain `0/30`.
8. Production threshold is fixed at `0.03`; the evaluation-only route
   ablation's highest passing candidate is also `0.03`.
9. Provider adapters and provider requests are both zero.
10. Every target and guard route equals a separately computed plan.
11. The authoritative acceptance run uses `--device cpu` and binds
    `execution_device=cpu` plus `precision_mode=fp32` to match free
    `cpu-basic` deployment.
12. The official artifact uses schema `1.3` and is replayable,
    privacy-allowlisted, deterministic, and bound to a clean committed source
    revision, `rrf_k`, corpus/dataset hashes, pinned model revisions, semantic
    view SHA-256, merge-policy version, and unrounded primary-score semantics.
13. Exactly one first-stage retrieval occurs per query. When reranking is
    enabled and the candidate pool is non-empty, exact singleton routes make
    two reranker function calls scoring at most twenty pairs each; every other
    route shape makes one. With reranking disabled or no candidates, model
    reranker calls and scored pairs are both zero.

If any gate fails, v0.3.6 remains NO-GO and no accepted artifact is exported.

## Cost and security constraints

- No paid Hugging Face hardware.
- No owner-funded Gemini/OpenAI calls.
- No Qdrant upgrade or writer key.
- The extra computation is one additional local reranker call over at most
  twenty pairs only for exact severance-comparison queries.
- BYOK behavior, session limits, private Space visibility, and the read-only
  Qdrant runtime key remain unchanged.

## Rejected alternatives

- Lowering the route threshold to `0.015`: unsupported by fresh evidence.
- Enlarging final Top K: evades the existing Top-5 quality contract.
- Hard-coding Article 17 or required source IDs: case-specific and masks
  ranking quality.
- A generic keyword append: the offline ablation still left `014` outside Top
  5.
- Replacing/fine-tuning the reranker: disproportionate for this release.
- Forcing `027` into generation with a new route or lower global threshold:
  weakens policy without improving answer correctness.

## Release claim

If accepted, v0.3.6 may claim deterministic multi-view retrieval coverage for
mixed old/new severance questions while retaining the established global
refusal threshold. It must not claim that a route-specific lower refusal
threshold improved reliability.

## Evidence lifecycle

The existing
`eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json` is immutable audit
evidence for the superseded design and must never be overwritten by the pivot
runner. Its bytes remain permanently available at bound commit `9890c785` and
are replayed with that commit's evaluator. On pivot acceptance it is removed
from the final public worktree, not published; Git history preserves the audit.
A failed pivot uses the distinct path
`eval/diagnostics/severance_retrieval_pivot_v0.3.6_no_go.json`.

Accepted schema `1.3` retains full input precision and
records `production_threshold` separately from the evaluation-only route
ablation. It also binds the semantic-view hash, merge-policy version, exact
CPU/FP32 execution mode, and primary-query score semantics.

Before calibration, all decision-relevant pivot documents, dataset,
implementation, and tests are committed in a clean worktree. Any later change
to retrieval, refusal, dataset, model/device/precision, evidence construction,
or replay logic invalidates the artifact and returns the process to acceptance
calibration. Documentation-only commits after the bound source revision may
not alter those inputs. Deterministic regeneration runs from a clean checkout
of the recorded source commit.

### Revision-binding amendment: tracked Python closure

The first Task 5 implementation attempted to infer the decision dependency
closure from Python imports. Five review-fix rounds demonstrated that this is
not a reliable release boundary: Python import and execution semantics include
aliases, namespace access, decorators, class bodies, lambdas, and point-order
effects that a small static analyzer cannot prove complete.

Schema `1.3` therefore uses a simpler conservative contract:

- bind the SHA-256 and Git blob identity of **every Git-tracked file whose
  suffix case-folds to `.py` in the repository at the recorded source
  revision**, including tests;
- additionally bind `pyproject.toml`, `uv.lock`, `.python-version`, the
  deployment `Dockerfile`, `legal_terms.txt`, and the separately declared
  corpus, dataset, model, and replay source artifacts;
- require the artifact's tracked-code path set to equal the set obtained from
  NUL-delimited full `git ls-tree -r -z <recorded revision>` records under
  those rules, preserving normalized repo-relative POSIX path, Git mode,
  object type, blob OID, and SHA-256;
- require the current `HEAD`/index tracked-code set and each actual checkout
  byte sequence to equal the recorded set and blobs. Add, remove, rename, mode
  change, sparse/missing file, duplicate/case-fold collision, symlink, gitlink,
  submodule, path escape, or extra binding fails closed even if a blob is
  unchanged;
- scan the verified repository code roots (`src`, `eval`, and `scripts`) before
  project imports and reject ignored or untracked
  importable artifacts (`.py`, case variants such as `.PY`, `.pyc`, `.pyo`,
  platform extension modules, `.pth`, and importable archives) plus
  `__pycache__` and cache trees. No repository virtual-environment or broad
  repository root is placed on `sys.path`;
- reject missing, extra, renamed, untracked, or changed bound code before any
  model construction.

Task 6 creates a dedicated authoritative environment outside the repository
from the bound lock, offline, frozen, and without development dependency
groups; it must pass the equivalent of `uv sync --check --frozen --no-dev`.
Acceptance and replay then start through a committed stdlib-only bootstrap
invoked by that environment's interpreter with isolated Python (`-I -S`) and
an explicit `--environment-root`. The bootstrap derives platform-specific
site-package directories from that root and its `pyvenv.cfg`; it must not use
`sys.prefix` or `sysconfig` under Python 3.11 `-S` to discover the environment.
Before adding verified project or third-party paths to `sys.path`, it must:

1. clear/reject `PYTHONPATH`, disable user-site and site customization, and
   reject unapproved external local import roots. `.pth` files may exist in the
   approved environment but are never processed and must not contribute
   `sys.path` entries; the interpreter's own stdlib archive is trusted, while
   other zip import roots are rejected;
2. validate the recorded/current Git sets, blobs, checkout bytes, declared
   input artifacts, ignored/untracked importable scan, and clean tree;
3. bind and verify Python implementation, full version, ABI, OS/platform, the
   selected no-development lock groups and markers, and a PEP 503-normalized,
   duplicate-free exact installed `{distribution: version}` inventory obtained
   with `importlib.metadata.distributions(path=[approved sites])`. Every
   inventory entry must be selected by the frozen lock for that environment;
   validation imports no third-party package;
4. add only the exact verified `src`, `eval`, and `scripts` roots plus approved
   environment sites directly, without importing `site` or processing `.pth`,
   then import evaluator/model code.

Tests monkeypatch every cache/model/index/provider constructor and prove zero
calls for each bootstrap failure. The trusted enforcement boundary is the OS,
interpreter binary, bootstrap/launcher, and approved third-party installation;
the artifact detects honest repository and environment-resolution drift. The
distribution inventory proves selected versions, not installed wheel bytes,
and the claim does not defend a malicious verifier, compromised interpreter,
or modified package installation that retains the same metadata.

The bound Dockerfile records deployment setup semantics, but its floating base
tag is not a bit-for-bit image receipt. CPU/FP32 acceptance plus the later
private Space smoke establish behavioral compatibility, not base-image digest
equivalence.

This conservative superset deliberately invalidates acceptance after any
tracked Python change, even when that file might not execute in one run. It
replaces the incomplete import-closure and dynamic-execution analyzer; release
correctness no longer depends on proving Python control flow. Documentation and
non-code public packaging may still change after acceptance only where the
existing replay verifier proves that no bound input changed.

The preliminary multi-view ablation is design evidence, not a release result.
Only the committed CPU/FP32 acceptance run may establish the release claim.
