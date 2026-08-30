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
