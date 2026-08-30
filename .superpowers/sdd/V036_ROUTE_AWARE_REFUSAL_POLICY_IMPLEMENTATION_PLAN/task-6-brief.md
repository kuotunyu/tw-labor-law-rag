### Task 6: Build fresh offline calibration evidence and select the threshold

> **Approved Task 6 contract amendment (2026-08-30):** Task 5 schema `1.2`
> and `eval/dataset/README.md` supersede the stale guard-reaggregation and
> schema `1.0` instructions below. Target, stress, and formal rows must all come
> from the same fresh offline retrieval pipeline with unrounded scores. Rounded
> v0.3.1 public trace values are metric baselines only, never decision inputs.
> An accepted official artifact uses schema `1.2`; a NO-GO instead writes a
> durable replayable content-free non-release diagnostic while the official
> artifact remains absent.

**Files:**

- Create: `eval/run_severance_refusal_policy.py`
- Create: `eval/official/severance_refusal_policy_v0.3.6.json`
- Create on NO-GO: `eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json`
- Create: `eval/diagnostics/README.md`
- Modify: `eval/official/README.md`
- Modify: `tests/test_severance_refusal_policy.py`

- [ ] **Step 1: Add failing runner and deterministic rebuild tests**

Test argument parsing, forced `TRANSFORMERS_OFFLINE=1` and `HF_HUB_OFFLINE=1`
before any Hugging Face import, cache-only embedder/reranker model resolution,
cached-model preflight, no LLM construction, strict empty work directory,
content-free export, deterministic result rebuilding after removing run time,
and NO-GO when any gate fails.

- [ ] **Step 2: Implement one offline retrieval pass**

The runner must reuse `_materialize_audited_corpus`, `_build_indexes`, the
committed corpus snapshot, and one pinned local pipeline. For each new case it
records source ranks, authoritative `retrieval.applied_routes`, `hit_count`, and
unrounded `top_score`. Assert that the returned routes equal the separately
planned routes; disagreement fails closed.
Evaluate every candidate using `decide_retrieval_refusal` without rerunning
retrieval.

For guards, run all committed stress/formal questions through that same fresh
offline pipeline and preserve unrounded ranks, hit counts, applied routes, and
top scores. Use v0.3.1 aggregates only as metric baselines. Do not edit those
artifacts or use their rounded trace scores as decision inputs.

An accepted official artifact uses schema `1.2` and contains:

- dataset/corpus/source-artifact canonical SHA-256 values;
- exact pinned model names/revisions, retrieval settings, resolved execution
  device, and `rrf_k`;
- candidate grid and one selected threshold;
- per-candidate target, stress, and formal gates;
- thirty content-free case observations;
- zero provider adapters and zero provider requests;
- the candidate source Git SHA.

The candidate source SHA must identify a clean committed implementation. If no
candidate passes all gates or selection differs from `0.015`, write raw
content-free target observations, fresh guard rows, seven candidate aggregates,
failed gates, hashes, configuration, and provenance to the non-release
diagnostic path. It must replay without retrieval or model loading and remains
outside `release/public-files.txt`.

- [ ] **Step 3: Run the fresh offline calibration**

```powershell
$env:TRANSFORMERS_OFFLINE = '1'
$env:HF_HUB_OFFLINE = '1'
uv run python eval/run_severance_refusal_policy.py --offline --device auto --export-official
```

Expected: selected threshold is exactly `0.015`; target contracts are 30/30;
stress answerable direct false refusals are 0/40; stress direct unanswerable
coverage is at least 17/20; formal Hit@5/MRR@10 meet the committed values and
formal direct false refusals remain zero; provider adapters/requests are zero.

If the selected value is not `0.015` or any gate fails, stop with NO-GO. Do not
edit the artifact, setting, dataset, or gate to force acceptance. Preserve the
diagnostic envelope and leave the official artifact absent.

- [ ] **Step 4: Verify deterministic and privacy-safe evidence**

```powershell
uv run pytest tests/test_severance_refusal_policy.py -q -p no:cacheprovider
uv run python -m json.tool eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json | Out-Null
rg -n '"(question|content|answer|endpoint|url|credential|secret|api_key)"|Users[/\\]|AI-Portfolio' eval/diagnostics/severance_refusal_policy_v0.3.6_no_go.json
```

Expected: tests and JSON parse pass; privacy scan returns no matches. Replay the
diagnostic and document the outcome without unverified claims. For acceptance,
apply the equivalent checks to the official artifact instead.

- [ ] **Step 5: Commit accepted calibration evidence**

```powershell
git add eval/run_severance_refusal_policy.py eval/official/severance_refusal_policy_v0.3.6.json eval/official/README.md tests/test_severance_refusal_policy.py
git diff --cached --check
git commit -m "eval: bind v0.3.6 refusal calibration"
```

---
