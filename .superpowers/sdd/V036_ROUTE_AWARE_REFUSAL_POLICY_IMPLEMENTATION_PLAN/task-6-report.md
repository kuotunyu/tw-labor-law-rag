# Task 6 report — v0.3.6 route-aware refusal calibration

## Outcome

**NO-GO.** The configured candidate sweep selected no threshold. The runner
therefore exited non-zero and did not create
`eval/official/severance_refusal_policy_v0.3.6.json`.

- Branch: `codex/v036-route-aware-refusal`
- Required base/candidate source revision: `f3577e84e9bf6dee00866c4e409511a0ddcf3f08`
- Worktree was clean before Task 6.
- Official run origin: fresh offline retrieval.
- Pinned models were resolved from local caches after forcing
  `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`.
- One isolated retrieval-only pipeline processed 30 target, 60 stress, and 40
  formal questions. It constructed no LLM adapter and sent no provider request.
- The current strict Task 5 replay schema is `1.2`; Task 6 uses that checked-in
  schema rather than the obsolete `1.0` wording in the original plan prose.

## Strict TDD evidence

Each runner behavior was introduced with a focused failing test before the
smallest implementation change. The observed RED failures were:

1. Missing runner CLI: subprocess exit `2` because
   `eval/run_severance_refusal_policy.py` did not exist.
2. Missing offline preflight: `AttributeError` for `Settings`.
3. Missing work-directory guard: `AttributeError` for `_prepare_work_dir`.
4. Missing one-pass target observation builder: `AttributeError` for
   `_run_target_cases`.
5. Missing fresh guard evidence builder: `AttributeError` for
   `_run_guard_cases`.
6. Missing seven-point sweep: `AttributeError` for `evaluate_candidate`.
7. Missing deterministic accepted-artifact composition: `AttributeError` for
   `_build_accepted_artifact`.
8. NO-GO message was initially absent: expected `NO-GO`, received
   `no candidate threshold satisfies the complete gate set`.
9. Missing deterministic JSON writer: `AttributeError` for
   `_write_public_json`.
10. Missing exact provenance binding: `AttributeError` for
    `_build_provenance`.
11. Missing main orchestration/export: expected run-local result file was not
    created.

Focused GREEN command and result:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
```

```text
........................................................................ [ 88%]
.........                                                                [100%]
81 passed in 2.90s
```

Focused lint command and result:

```powershell
uv run ruff check eval/run_severance_refusal_policy.py tests/test_severance_refusal_policy.py
```

```text
All checks passed!
```

## Official calibration command and result

Exact command:

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
uv run python eval/run_severance_refusal_policy.py --offline --device auto --export-official
```

Observed execution evidence:

- Corpus audit parsed `acts.zip` and `regulations.zip` and matched the committed
  snapshot.
- Structure index: `884 chunks, 884 points`.
- Fixed index: `481 chunks, 481 points`.
- Fresh retrieval completed for target `30/30`, stress `60/60`, and formal
  `40/40`.
- Exit code: `1`.
- Final error:

```text
RuntimeError: NO-GO: no candidate threshold satisfies the complete gate set
```

The export occurs only after complete acceptance, so the official artifact was
not written.

## Candidate and gate evidence

The retained local index was replayed with the same pinned offline pipeline to
print only content-free candidate aggregates. Every configured candidate
produced the same result:

| Candidate | Target | Stress false refusals | Stress direct coverage | Formal Hit@5 | Formal MRR@10 | Formal false refusals | Complete |
|---:|---:|---:|---:|---:|---:|---:|:---:|
| 0 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.005 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.01 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.015 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.02 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.025 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |
| 0.03 | 27/30 | 0/40 | 17/20 (`0.85`) | `1.0` | `0.9388888888888888` | 0/30 | fail |

The two formal columns are fresh achieved metrics. Their committed pass
baselines are Hit@5 at least `0.9666666666666667` and MRR@10 at least
`0.9055555555555554`; the achieved values exceed both baselines.

Target aggregates for every candidate:

```text
positive_routes=15
positive_sources_at_5=13
positive_generation_allowed=15
collision_contracts=14
passed_cases=27
```

Content-free failing inputs from the retained index:

```text
severance-policy-010 routes=[severance_comparison]
  required_source_1_rank=1 required_source_2_rank=absent
  top_score=0.5001251697514135
severance-policy-014 routes=[severance_comparison]
  required_source_1_rank=1 required_source_2_rank=absent
  top_score=0.9620363047907355
severance-policy-027 routes=[] source_ranks={}
  top_score=0.02350945240753301
```

The first two rows miss the second required authority within Top 5. The third
row correctly avoids the target route, but its score is below the unchanged
global `0.03`, so it is directly refused instead of admitted to generation.
These failures are invariant across the severance-route candidate grid.

## Root-cause classification

This is not a threshold-selection tradeoff. Stress and formal guards pass at
all seven candidates, while all three target failures are independent of the
candidate threshold:

- two failures are Top-5 retrieval-quality misses for the second required
  authority;
- one failure is a collision target contract versus the existing global
  threshold on a no-route question.

The next controller decision therefore requires separate retrieval-quality
and/or reviewed target-contract root-cause work. This Task 6 run provides no
authority to change the dataset, gates, global threshold, expected `0.015`, or
production setting.

## Artifact/privacy state

- `eval/official/severance_refusal_policy_v0.3.6.json`: intentionally absent.
- No question, answer, legal text, endpoint, URL, credential, secret, API key,
  account identity, or local path was exported to an official artifact.
- The existing v0.3.1 official artifacts were not edited or used as decision
  inputs; the guard decisions use fresh unrounded scores.
- Provider adapters constructed: `0`.
- Provider requests sent: `0`.

## Final proportional verification

Exact broader test command:

```powershell
uv run pytest tests/test_severance_refusal_policy.py tests/test_reliability.py tests/test_reliability_dataset.py tests/test_pipeline.py tests/test_refusal_policy.py tests/test_config.py tests/test_answerer.py tests/test_factory.py -q -p no:cacheprovider
```

Result:

```text
246 passed in 3.79s
```

Exact repository-wide Ruff command and result:

```powershell
uv run ruff check .
```

```text
All checks passed!
```

Exact diff command:

```powershell
git diff --check
```

Result: exit `0`, no output.

No-artifact and no-LLM-construction checks:

```powershell
if (Test-Path -LiteralPath 'eval\official\severance_refusal_policy_v0.3.6.json') { throw 'unexpected official artifact exists' }
rg -n 'build_answerer|build_llm|LLMAdapter|RoutedLLM|Gemini|OpenAI' eval/run_severance_refusal_policy.py
```

Result: official artifact absent; the runner scan returned no matches.

---

## Review fix round 1/5 — authoritative rerun evidence

This section supersedes the initial run above for review and provenance. The
controller approved a narrow contract amendment: Task 5 schema `1.2` and
`eval/dataset/README.md` supersede the stale Task 6 instructions to reaggregate
rounded v0.3.1 trace scores and publish schema `1.0`. All 130 rows below came
from one fresh offline pipeline with unrounded scores. The v0.3.1 values were
used only as metric baselines, never as decision inputs.

### Additional RED evidence

Each review fix began with a focused failing test:

1. The Hugging Face import-order subprocess test detected model-library import
   before the offline environment existed.
2. The reranker cache-only test observed that snapshot resolution omitted
   `local_files_only=True`.
3. The local pipeline test observed that index/reranker construction did not
   explicitly force local-only resolution and did not bind the resolved device.
4. The guard-route disagreement test showed that planned routes replaced the
   pipeline's returned routes rather than failing closed.
5. The zero-hit target test exposed the hard-coded `has_hits=True` decision.
6. The provenance test lacked decision-relevant `rrf_k` and an exact resolved
   execution device; the dirty-tree test lacked a clean-revision precondition.
7. The NO-GO envelope test failed because no durable diagnostic builder or
   offline replay function existed.
8. The real-main persistence test failed with argparse exit `2` because
   `--diagnostics-output` did not exist:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_runner_main_persists_replayable_no_go_without_official_export -q -p no:cacheprovider
```

```text
1 failed in 1.05s
```

### GREEN and mutation evidence

The fixed focused policy suite:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
```

```text
89 passed in 4.12s
```

The cache-only/import-order, independent no-LLM construction, real-main NO-GO,
and replay-mutation checks were then rerun together:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_offline_flag_precedes_every_hugging_face_import_snapshot tests/test_embedding_cache.py::test_offline_reranker_resolves_snapshot_local_only tests/test_severance_refusal_policy.py::test_local_pipeline_forces_both_model_loaders_local_only_without_llm tests/test_severance_refusal_policy.py::test_runner_main_persists_replayable_no_go_without_official_export tests/test_severance_refusal_policy.py::test_no_go_replay_rejects_mutated_candidate_aggregate -q -p no:cacheprovider
```

```text
5 passed in 3.50s
```

The mutation test changes a retained candidate aggregate and proves replay
rejects it with `NO-GO evidence replay mismatch`.

Broader pre-run verification:

```powershell
uv run pytest tests/test_severance_refusal_policy.py tests/test_embedding_cache.py tests/test_reliability.py tests/test_reliability_dataset.py tests/test_pipeline.py tests/test_refusal_policy.py tests/test_config.py tests/test_answerer.py tests/test_factory.py -q -p no:cacheprovider
```

```text
267 passed in 9.66s
```

```powershell
uv run ruff check .
```

```text
All checks passed!
```

### Clean implementation revision

The implementation and amended contract were committed before calibration:

```text
192fb7081132d1de5eed52d5b29ad84737f951f2
fix: preserve replayable refusal no-go evidence
```

Immediately before the official calibration command,
`git status --porcelain --untracked-files=all` returned no output, and both the
official artifact and diagnostic output were absent. The diagnostic provenance
now binds that clean implementation SHA, resolved `execution_device: cuda`,
`rrf_k: 60`, pinned model revisions, input hashes, and zero provider counters.

### Fresh 130-query official calibration rerun

Exact command:

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
uv run python eval/run_severance_refusal_policy.py --offline --device auto --export-official
```

Observed result:

- exit code `1`, the deliberate NO-GO status;
- corpus snapshot audit passed;
- structure index: `884 chunks, 884 points, 191.5s`;
- fixed index: `481 chunks, 481 points, 50.6s`;
- fresh target `30/30`, stress `60/60`, and formal `40/40` completed;
- every returned guard route matched its separately planned route;
- `selected_threshold: null`;
- all seven candidates list only `target` in `failed_gates`;
- `eval/official/severance_refusal_policy_v0.3.6.json` remained absent;
- `eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json` was written.

The diagnostic is `40,358` bytes with SHA-256
`508b1e8e0d176153f08fc6ba80a89bc2ee0c5a3d54fb3ad9081de4ff6462d46f`.
It contains exactly 30 raw target observations, 60 fresh stress rows, 40 fresh
formal rows, and seven candidate aggregates. Provider adapters and requests are
both exactly zero.

### Reproduced gates and diagnosis

The review rerun reproduced the original diagnosis exactly for all candidates
`0`, `0.005`, `0.01`, `0.015`, `0.02`, `0.025`, and `0.03`:

| Gate | Fresh achieved | Committed requirement | Status |
|---|---:|---:|:---:|
| Target contracts | 27/30 | 30/30 | fail |
| Stress false refusals | 0/40 | 0/40 | pass |
| Stress direct unanswerable | 17/20 (`0.85`) | at least 17/20 | pass |
| Formal Hit@5 | `1.0` | at least `0.9666666666666667` | pass |
| Formal MRR@10 | `0.9388888888888888` | at least `0.9055555555555554` | pass |
| Formal false refusals | 0/30 | 0/30 | pass |

The exact retained target failures also reproduced:

```text
severance-policy-010 hit_count=5 routes=[severance_comparison]
  勞工退休金條例|第 12 條 rank=1; second required source absent
  top_score=0.5001251697514135
severance-policy-014 hit_count=5 routes=[severance_comparison]
  勞工退休金條例|第 12 條 rank=1; second required source absent
  top_score=0.9620363047907355
severance-policy-027 hit_count=5 routes=[] source_ranks={}
  top_score=0.02350945240753301
```

The first two failures remain retrieval-quality/root-cause work: the second
required authority misses Top 5 despite five returned hits. The third is a
reviewed target-contract/global-policy interaction: no special route applies,
so the unchanged global `0.03` refuses its `0.02350945240753301` score. None is
repairable by calibrating the severance-route threshold grid. The conservative
NO-GO therefore remains correct; no dataset, gate, score, global/route
threshold, expected `0.015`, or production setting was changed.

### Replay, privacy, and publication boundary

Exact replay command and result:

```powershell
uv run python -c "import json; from pathlib import Path; from eval import _bootstrap; from rag.severance_refusal_policy import replay_no_go_evidence; p=Path('eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json'); e=json.loads(p.read_text(encoding='utf-8')); assert replay_no_go_evidence(e)==e; print('replay_ok=True rows=130 candidates=7')"
```

```text
replay_ok=True rows=130 candidates=7
```

JSON parsing and the exact-key privacy scan both passed:

```powershell
uv run python -m json.tool eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json > $null
rg -n '"(question|content|answer|endpoint|url|credential|secret|api_key)"|Users[/\\]|AI-Portfolio' eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json
```

The scan returned no matches. The diagnostic is explicitly absent from
`release/public-files.txt`, while the accepted official path remains absent.
It is durable non-release evidence, not authority to publish `0.015`.

### Post-rerun broader verification

The broader suite including the current release verifier was run after writing
the diagnostic:

```powershell
uv run pytest tests/test_severance_refusal_policy.py tests/test_embedding_cache.py tests/test_reliability.py tests/test_reliability_dataset.py tests/test_pipeline.py tests/test_refusal_policy.py tests/test_config.py tests/test_answerer.py tests/test_factory.py tests/test_official_artifacts.py tests/test_release_verification.py -q -p no:cacheprovider
```

```text
389 passed, 2 failed in 15.39s
```

Both failures are the same expected current-branch publication-boundary check:
the v0.3.6 Task 1–6 tracked files are not in the still-v0.3.5
`release/public-files.txt`. That set already included the Task 2/Task 6 reports,
dataset, runner, source, tests, and plan before the new diagnostic README was
considered. Task 6 deliberately does not mutate Task 7's release allowlist,
and this NO-GO diagnostic is intentionally non-release. The failures do not
indicate a policy, replay, privacy, retrieval, or official-artifact regression;
they accurately prevent this NO-GO branch from being treated as a releasable
public tree.

The proportional Task 6/retrieval/official suite excluding that known Task 7
publication binding passed cleanly:

```powershell
uv run pytest tests/test_severance_refusal_policy.py tests/test_embedding_cache.py tests/test_reliability.py tests/test_reliability_dataset.py tests/test_pipeline.py tests/test_refusal_policy.py tests/test_config.py tests/test_answerer.py tests/test_factory.py tests/test_official_artifacts.py -q -p no:cacheprovider
```

```text
285 passed in 8.04s
```

Final Ruff and diff checks:

```powershell
uv run ruff check .
git diff --check
```

```text
All checks passed!
git diff --check: exit 0, no output
```

---

## Review fix round 2/5 — final target-route provenance rerun

### Strict TDD and mutation evidence

The target-route disagreement test was added first. It independently planned
the first target question, returned an observed empty route tuple, and required
the runner to fail closed:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_runner_rejects_target_route_disagreement_instead_of_recording_it -q -p no:cacheprovider
```

RED result before the production change:

```text
Failed: DID NOT RAISE RuntimeError
1 failed in 1.07s
```

The minimal fix separately calls `plan_retrieval_query(case.question)` and
requires exact tuple equality before recording the authoritative observed
`retrieval.applied_routes`. Focused GREEN:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_runner_rejects_target_route_disagreement_instead_of_recording_it tests/test_severance_refusal_policy.py::test_runner_retrieves_each_target_once_and_records_unrounded_observation -q -p no:cacheprovider
```

```text
2 passed in 0.86s
```

A separate main-level NO-GO integration test now leaves the real
`_build_local_pipeline` and real `factory.build_retrieval_pipeline` boundary in
place. Only slow offline index/model internals and collected evidence are
controlled doubles. Every LLM/provider adapter constructor, `build_llm`,
`build_answerer`, and `RoutedLLM` is replaced by a fail-fast sentinel; the test
also requires the one real `RetrievalPipeline` instance to cross target and both
guard collection boundaries.

The current retrieval-only implementation passed in `0.94s`. To prove the test
detects the prohibited regression, the runner import was temporarily mutated
with `apply_patch` from `build_retrieval_pipeline` to `build_answerer`. Exact
mutation command/result:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_runner_main_no_go_keeps_real_retrieval_factory_provider_free -q -p no:cacheprovider
```

```text
Failed: LLM/provider construction is forbidden in calibration main
1 failed in 1.24s
```

The mutation was then reverted with `apply_patch`. Combined GREEN:

```powershell
uv run pytest tests/test_severance_refusal_policy.py::test_runner_main_no_go_keeps_real_retrieval_factory_provider_free tests/test_severance_refusal_policy.py::test_runner_rejects_target_route_disagreement_instead_of_recording_it -q -p no:cacheprovider
```

```text
2 passed in 1.06s
```

The report tables above now distinguish the committed formal gates—Hit@5 at
least `0.9666666666666667` and MRR@10 at least
`0.9055555555555554`—from the higher fresh achieved metrics `1.0` and
`0.9388888888888888`.

### Clean candidate and pre-run verification

The implementation, tests, and baseline wording were committed before the
fresh calibration:

```text
356fe82c6ce0066422e77fa8f291d2aba1244dee
fix: validate target calibration routes
```

`git status --porcelain --untracked-files=all` returned no output immediately
before the run. Pre-run commands/results:

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
uv run pytest tests/test_embedding_cache.py tests/test_reliability.py tests/test_reliability_dataset.py tests/test_pipeline.py tests/test_refusal_policy.py tests/test_config.py tests/test_answerer.py tests/test_factory.py tests/test_official_artifacts.py -q -p no:cacheprovider
uv run ruff check .
git diff --check
```

```text
91 passed in 3.92s
196 passed in 3.50s
All checks passed!
git diff --check: exit 0, no output
```

### Fresh 130-query rerun

Exact command:

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
uv run python eval/run_severance_refusal_policy.py --offline --device auto --export-official
```

Observed result:

- exit code `1`, the deliberate conservative NO-GO;
- structure index: `884 chunks, 884 points, 180.5s`;
- fixed index: `481 chunks, 481 points, 107.6s`;
- target `30/30`, stress `60/60`, and formal `40/40` completed fresh;
- all target and guard observed route tuples exactly equaled their independently
  planned tuples;
- all seven candidates failed only `target`, with no selected threshold;
- official artifact remained absent;
- provider adapters/requests remained exactly `0/0`.

The current diagnostic contains 30 target, 60 stress, and 40 formal rows plus
seven candidates. Its provenance binds candidate SHA
`356fe82c6ce0066422e77fa8f291d2aba1244dee`, exact device `cuda`, and `rrf_k=60`.
The 40,358-byte file has SHA-256
`ebed566b3778f6674ef55641564861ef3c17bb899197bb40e3ba7f81bb6e010c`.

An exact JSON comparison against the prior diagnostic, excluding only
provenance, proved that target observations, both guard sets, all candidate
aggregates, and failed gates reproduced identically. Every candidate again
achieved target `27/30`, stress false refusals `0/40`, stress direct coverage
`17/20`, formal Hit@5 `1.0`, formal MRR@10 `0.9388888888888888`, and formal
false refusals `0/30`. Cases 010, 014, and 027 and their exact scores are
unchanged, so the retrieval-quality/target-contract diagnosis is unchanged.

Replay and privacy verification:

```powershell
uv run python -c "import json; from pathlib import Path; from eval import _bootstrap; from rag.severance_refusal_policy import replay_no_go_evidence; p=Path('eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json'); e=json.loads(p.read_text(encoding='utf-8')); assert replay_no_go_evidence(e)==e; print('replay_ok=True rows=130 candidates=7')"
uv run python -m json.tool eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json > $null
rg -n '"(question|content|answer|endpoint|url|credential|secret|api_key)"|Users[/\\]|AI-Portfolio' eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json
```

```text
replay_ok=True rows=130 candidates=7
privacy scan: 0 matches
diagnostic in release/public-files.txt: false
official artifact present: false
```

Post-rerun verification repeated the focused and broader commands above:

```text
91 passed in 17.22s
196 passed in 2.89s
All checks passed!
git diff --check: exit 0, no output
```

The two known Task 7 publication-set failures documented in round 1 are
unchanged and outside this conservative Task 6 NO-GO fix; no release allowlist,
dataset, gate, candidate, threshold, score, or accepted artifact was changed.
