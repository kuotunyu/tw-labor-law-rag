# v0.3.6 Retrieval Coverage Pivot Implementation Plan

This plan executes `V036_RETRIEVAL_COVERAGE_PIVOT_DESIGN.md` after the original
route-threshold calibration correctly returned NO-GO. Every task uses strict
TDD, a fresh implementer, task-scoped review, and fail-closed evidence.

## Task 1: Lock the pivot contract

Files:

- `eval/dataset/severance_refusal_policy_v0.3.6.jsonl`
- `eval/dataset/README.md`
- `src/rag/severance_refusal_policy.py`
- `tests/test_severance_refusal_policy.py`
- original v0.3.6 design/plan status notes

Work:

1. Replace the generation boolean with an exact `expected_outcome` enum for all
   thirty rows, preserving the meaning of twenty-eight rows.
2. Add failing contract/scorer tests proving `023` and `027` are route-negative,
   unanswerable, have positive hits, and stop specifically at `threshold`;
   `no_hits` must fail for either case.
3. Require all fifteen positives to have routes exactly equal to the singleton
   severance route. Preserve non-exact behavior for every collision row.
4. Correct `027` from generation to threshold and refine `023` from the former
   no-generation meaning to its measured threshold stage; keep `024` as
   explicit generation.
5. Add explicit supersession notes to the original design and plan.
6. Prove all other 28 dataset rows and coverage counts are unchanged.

## Task 2: Add deterministic multi-view planning and merge primitives

Files:

- `src/rag/retrieval/pipeline.py`
- `src/rag/retrieval/reranker.py`
- `tests/test_pipeline.py`

Work:

1. Extend `QueryPlan` with immutable rerank-only semantic views.
2. Emit the old-regime view only for exact singleton
   `severance_comparison` plans.
3. Implement a pure primary-first deduplicating interleave over two exact
   permutations of the same candidate-ID set.
4. Require canonical unique candidate IDs; reject score-count mismatch,
   truncated/duplicate/foreign reranker output; preserve primary-query scores.
5. Cover empty lists, unequal lengths, within-ranking duplicates, expected
   cross-ranking repetition, foreign/missing IDs, unstable IDs, route
   isolation, authoritative non-monotonic merged order, and determinism.

## Task 3: Integrate exact-route two-view reranking

Files:

- `src/rag/retrieval/pipeline.py`
- `tests/test_pipeline.py`
- relevant cached retrieval integration tests

Work:

1. Keep first-stage retrieval and candidate pool unchanged.
2. For exact severance route, rerank the full pool with primary and secondary
   views, merge, then cut Top 5.
3. Store primary top score before merging; secondary scores never control
   refusal.
4. Keep every other route shape on the existing single reranker path.
5. Add cached/offline integration coverage for `010`, `014`, and `015` that
   proves both authorities are Top 5 and records stage ranks without source-ID
   steering.
6. Assert call counts: one retrieval for every query; when reranking is enabled
   and candidates exist, two at-most-20-pair reranker calls only for exact
   singleton routes and one otherwise; reranker-disabled or empty-candidate
   paths make zero model calls and score zero pairs.

## Task 4: Remove the unsupported production threshold override

Files:

- `src/rag/config.py`
- `src/rag/factory.py`
- `src/rag/generation/answerer.py`
- `src/rag/retrieval/refusal_policy.py`
- evaluation runners and corresponding tests

Work:

1. RED-test that production uses only `rerank_score_threshold`.
2. Remove `severance_comparison_score_threshold` from Settings, Answerer, and
   factory wiring.
3. Simplify the shared production refusal policy to one global threshold.
4. Preserve route evidence without using it to choose an admission threshold.
5. Update reliability/portfolio/provider runners and prove provider isolation.

## Task 5: Pivot the calibration contract

Files:

- `src/rag/severance_refusal_policy.py`
- `eval/run_severance_refusal_policy.py`
- `scripts/verify_release.py`
- `tests/test_severance_refusal_policy.py`
- `tests/test_release_verification.py`
- diagnostic/official README files

Work:

1. Keep fresh, unrounded, replayable target/stress/formal evidence.
2. Recompute `027` against its exact direct-refusal contract.
3. Sweep the same seven hypothetical route candidates only as a named
   `route_ablation`; do not expose them through the production policy.
4. Record `production_threshold=0.03` separately and require
   `route_ablation.highest_passing_candidate=0.03`.
5. Keep clean-revision, resolved-device, `rrf_k`, local-only model, privacy,
   no-LLM, and no-artifact-on-failure guarantees.
6. Bump the artifact schema to `1.3`; bind semantic-view hash, merge-policy version,
   full-precision primary-score semantics, CPU/FP32, exact route checks, and
   retrieval/reranker call counts.
7. Complete all decision-relevant public replay and release-verifier logic
   before the clean acceptance run; tests must reject schema/provenance/gate
   mismatches without model execution.
8. Refactor the large evaluator only where needed to keep production and
   calibration responsibilities visibly separated.
9. Bind the exact Git-revision set and hashes of every tracked `*.py` file,
   plus `pyproject.toml`, `uv.lock`, `.python-version`, the deployment
   `Dockerfile`, and `legal_terms.txt`; Python suffix matching is
   case-insensitive. Delete the attempted import-closure and
   dynamic-execution analyzer. Use a committed stdlib-only `python -I -S`
   bootstrap plus an explicit dedicated environment outside the repository,
   synchronized offline/frozen/no-dev from the bound lock. The bootstrap takes
   that validated environment root explicitly and validates recorded/current
   Git sets, modes/blobs, checkout bytes, ignored/untracked importable artifacts
   under exact code roots, clean state, isolated `sys.path`,
   interpreter/platform/ABI, selected lock groups/markers, and the exact
   normalized installed distribution inventory before importing project,
   cache, index, or model code. Tests must prove that a
   changed, added, removed, renamed, untracked, ignored, aliased, or extra
   importable file invalidates replay with zero constructors called.

## Task 6: Run fresh offline acceptance and export the official artifact

Files:

- `eval/official/severance_refusal_policy_v0.3.6.json`
- `eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json` (immutable old audit)
- `eval/diagnostics/severance_retrieval_pivot_v0.3.6_no_go.json` (pivot failure only)
- `eval/official/README.md`
- task report

Work:

1. Commit the implementation candidate and require a clean tree.
2. Run authoritatively with `--device cpu`, `TRANSFORMERS_OFFLINE=1`,
   `HF_HUB_OFFLINE=1`, and no providers. Bind `cpu` and `fp32`; CUDA evidence
   may be diagnostic only and cannot pass release.
3. Rebuild the isolated index from the committed corpus snapshot.
4. Evaluate all 30 target, 60 stress, and 40 formal questions once.
5. Require every acceptance gate from the pivot design.
6. On acceptance, export the official artifact and remove the old NO-GO JSON
   from the final worktree without overwriting it; commit `9890c785` remains
   its immutable audit location.
7. On failure, write the new versioned pivot diagnostic and stop.
8. Invalidate and rerun if any decision-relevant file changes after the bound
   candidate commit.

## Task 7: Resume release verification and documentation

Files:

- `release/public-files.txt`
- release manifest/provenance files
- `README.md`, architecture/release documentation, tests

Work:

1. Bind the already-verified accepted artifact and final public file set.
2. Run the already-committed replay gates without changing their logic. Any
   needed replay/verifier code change invalidates the artifact and returns to
   Task 5 then Task 6.
   Every tracked Python file, including tests, is read-only in Task 7; any
   Python edit likewise returns to Task 5 then Task 6. Task 7 may add or edit
   only unbound documentation/public packaging after acceptance.
3. Verify no private paths, URLs, identities, credentials, question text, or
   legal excerpts leak into release artifacts.
4. Document the multi-view ranking behavior and retained global threshold.
5. Remove every claim about an active `0.015` route-specific threshold.
6. Confirm the old NO-GO JSON is absent from the public worktree and
   `release/public-files.txt`, while its immutable audit commit remains
   documented.

## Task 8: Final validation and private deployment

Work:

1. Run the full repository suite, Ruff, release verifier, privacy scan, and
   deterministic artifact regeneration.
2. Obtain final whole-branch review.
3. Merge only after the release verifier is green.
4. Deploy to the existing private Hugging Face Space on free `cpu-basic`.
5. Run zero-provider BYOK fixture/simulation smoke tests. Any live Gemini or
   OpenAI request requires a visitor to enter and fund their own key; the
   project owner does not fund deployment smoke calls.
6. Confirm Qdrant uses the single read-only v0.3.4+ runtime key and Free Tier.
7. Remove temporary diagnostics and local experiment material after evidence is
   safely committed.

## Stop conditions

Stop immediately and report NO-GO if:

- either canonical severance authority is absent from Top 5 for any positive;
- any collision receives a prohibited route;
- stress/formal metrics regress;
- production threshold is not `0.03` or the route ablation's highest passing
  candidate is not `0.03`;
- the authoritative run is not CPU/FP32 or differs from deployed inference;
- any provider is constructed or called during calibration;
- route, score, device, revision, hash, or privacy evidence cannot be replayed;
- the accepted artifact and verifier disagree;
- either reranker output is not a complete unique permutation of its input
  candidate IDs, or merged results are re-sorted by score.
