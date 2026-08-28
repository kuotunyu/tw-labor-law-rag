# v0.3.1 Reliability and Corpus Provenance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a reproducible 15-law corpus snapshot, citation provenance, a separate
60-question narrative stress benchmark, and strictly budgeted Gemini/OpenAI cross-checks
without changing the immutable `v0.1.0` formal evidence.

**Architecture:** Pure modules under `src/rag/` own deterministic snapshot, stress-metric,
and budget logic. Thin scripts handle network/model I/O and keep raw runs in ignored paths.
Existing loader/chunker/answer/API/UI boundaries carry optional provenance end to end, while
old payloads remain valid.

**Tech Stack:** Python 3.12, defusedxml, httpx, Qdrant client, FlagEmbedding, FastAPI,
Streamlit, pytest, Ruff, Bandit, pip-audit.

## Global Constraints

- Preserve the canonical SHA-256 of `eval/dataset/eval_set.jsonl`:
  `760e33eaa0821001d37ff974bc037043d019fc670b8f3621b6e713030274ca07`.
- Keep provider caps at or below `gemini=5.00` and `openai=5.00` US dollars.
- Never persist or print API keys, raw provider payloads, judge reasons, or private host paths.
- Do not change the production rerank threshold unless the completed stress evidence satisfies
  the design gate.
- Do not mutate Qdrant Cloud in this plan; the runtime key remains read-only.
- Hugging Face must remain public and on free `cpu-basic` hardware.
- Every new public file must be added to the sorted exact allowlist.

---

### Task 1: Deterministic corpus snapshot and live comparison

**Files:**
- Create: `src/rag/corpus_audit.py`
- Create: `scripts/audit_corpus.py`
- Create: `tests/test_corpus_audit.py`
- Create: `release/corpus_snapshot.json`
- Modify: `release/public-files.txt`
- Modify: `release/manifest.json`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_release_verification.py`

**Interfaces:**
- Consumes: official archive bytes and the existing `DUMPS`, `iter_laws`, and
  `normalize_name` definitions from `scripts.download_corpus`.
- Produces: `build_snapshot(sources, laws, snapshot_date) -> dict`,
  `compare_snapshots(committed, live) -> list[dict]`, and CLI exit status 0 for no change,
  1 for detected freshness change, 2 for invalid source data.

- [ ] **Step 1: Write failing deterministic snapshot tests**

```python
def test_build_snapshot_sorts_laws_and_hashes_normalized_content():
    snapshot = build_snapshot(
        sources=[{"id": "acts", "url": "https://example/acts", "sha256": "a" * 64}],
        laws=[law_fixture("乙法"), law_fixture("甲法")],
        snapshot_date="2026-08-29",
    )
    assert [law["name"] for law in snapshot["laws"]] == ["乙法", "甲法"]
    assert len(snapshot["laws"][0]["content_sha256"]) == 64


def test_compare_snapshots_reports_amendment_and_content_changes():
    changes = compare_snapshots(old_snapshot(), new_snapshot())
    assert {change["kind"] for change in changes} == {"last_amended", "content_sha256"}
```

- [ ] **Step 2: Run the tests and confirm missing-module failure**

Run: `python -m pytest tests/test_corpus_audit.py -q`

Expected: import failure for `rag.corpus_audit`.

- [ ] **Step 3: Implement canonical JSON hashing and snapshot comparison**

```python
def canonical_sha256(value: object) -> str:
    payload = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def compare_snapshots(committed: Mapping[str, Any], live: Mapping[str, Any]) -> list[dict]:
    old = {law["name"]: law for law in committed["laws"]}
    new = {law["name"]: law for law in live["laws"]}
    changes = []
    for name in sorted(old.keys() | new.keys()):
        if name not in old:
            changes.append({"law": name, "kind": "added"})
            continue
        if name not in new:
            changes.append({"law": name, "kind": "removed"})
            continue
        for field in ("last_amended", "effective_date", "num_articles", "content_sha256"):
            if old[name][field] != new[name][field]:
                changes.append({"law": name, "kind": field,
                                "old": old[name][field], "new": new[name][field]})
    return changes
```

- [ ] **Step 4: Implement the thin CLI and write the audited snapshot**

Run: `python scripts/audit_corpus.py --write release/corpus_snapshot.json`

Expected: `15 laws`, both official source hashes, and no raw corpus files in the repository.

- [ ] **Step 5: Extend release verification**

Add offline checks for schema `1.0`, exactly 15 unique target laws, aggregate article-count
arithmetic, 64-character lowercase hashes, and source URLs matching the two official URLs.

- [ ] **Step 6: Run focused and release tests**

Run: `python -m pytest tests/test_corpus_audit.py tests/test_release_verification.py -q`

Run: `python scripts/verify_release.py`

Expected: all pass and `source_data.full_snapshot_laws == 15`.

- [ ] **Step 7: Commit**

```bash
git add src/rag/corpus_audit.py scripts/audit_corpus.py tests/test_corpus_audit.py \
  release/corpus_snapshot.json release/manifest.json release/public-files.txt \
  src/rag/release_verification.py tests/test_release_verification.py
git commit -m "feat: add auditable full-corpus snapshot"
```

### Task 2: Carry legal provenance into citations

**Files:**
- Modify: `src/rag/models.py`
- Modify: `src/rag/ingestion/loader.py`
- Modify: `src/rag/ingestion/chunkers.py`
- Modify: `src/rag/generation/answerer.py`
- Modify: `src/rag/api/main.py`
- Modify: `ui/app.py`
- Modify: `tests/test_loader.py`
- Modify: `tests/test_chunkers.py`
- Modify: `tests/test_answerer.py`
- Modify: `tests/test_api_response.py`
- Modify: `tests/test_ui_byok_app.py`

**Interfaces:**
- Consumes: `url`, `last_amended`, and `effective_date` from normalized law JSON.
- Produces: optional same-named fields in `SourceUnit`, `Chunk.payload()`, answer sources,
  FastAPI `SourceOut`, and Streamlit source rendering.

- [ ] **Step 1: Add failing loader and chunk propagation tests**

```python
def test_law_loader_preserves_public_provenance(tmp_path):
    path = write_law(tmp_path, url="https://law.moj.gov.tw/example",
                     last_amended="20260121", effective_date="20260123")
    unit = load_law_json(path)[0]
    assert unit.source_url == "https://law.moj.gov.tw/example"
    assert unit.last_amended == "20260121"
    assert unit.effective_date == "20260123"
```

- [ ] **Step 2: Run focused tests and verify attribute failure**

Run: `python -m pytest tests/test_loader.py tests/test_chunkers.py -q`

- [ ] **Step 3: Add optional fields and propagate through both chunkers**

```python
@dataclass
class SourceUnit:
    text: str
    doc_id: str
    doc_title: str
    article_no: str = ""
    chapter: str = ""
    source_path: str = ""
    source_url: str = ""
    last_amended: str = ""
    effective_date: str = ""
```

Add matching defaults to `Chunk`, copy them at every chunk construction site, and retain
empty strings for non-law formats.

- [ ] **Step 4: Add failing answer/API/UI compatibility tests**

Prove new payloads render a date/link and legacy payloads with no provenance still serialize
and render without error.

- [ ] **Step 5: Extend answer sources and `SourceOut`**

```python
class SourceOut(BaseModel):
    index: int
    doc: str
    article: str
    content: str
    source_url: str = ""
    last_amended: str = ""
    effective_date: str = ""
```

Use `payload.get(field, "")` in `Answerer._parse_sources` for backward compatibility.

- [ ] **Step 6: Render the official link and date**

In `render_sources`, show `最新異動：YYYY-MM-DD` and a Markdown link labelled
`全國法規資料庫` only when those values are present.

- [ ] **Step 7: Run focused and full tests, then commit**

Run: `python -m pytest tests/test_loader.py tests/test_chunkers.py tests/test_answerer.py tests/test_api_response.py tests/test_ui_byok_app.py -q`

Run: `python -m pytest -q`

Commit: `feat: expose legal-source provenance in citations`.

### Task 3: Add the separate 60-question reliability stress set

**Files:**
- Create: `eval/dataset/reliability_stress_v0.3.1.jsonl`
- Create: `tests/test_reliability_dataset.py`
- Modify: `eval/dataset/README.md`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: ground-truth source references from `eval_set.jsonl`, `mini_eval.jsonl`, and the
  audited corpus snapshot.
- Produces: 60 rows with existing evaluation fields plus `base_qid` and `style_tags`.

- [ ] **Step 1: Write failing dataset-contract tests**

```python
def test_reliability_stress_shape_and_diversity():
    rows = load_jsonl(STRESS_PATH)
    assert len(rows) == 60
    assert sum(row["answerable"] for row in rows) == 40
    assert sum(len(row["question"]) >= 40 for row in rows) >= 30
    assert sum("code_switch" in row["style_tags"] for row in rows) >= 15
    assert sum("narrative" in row["style_tags"] for row in rows) >= 15
```

Also assert 10 related-unanswerable, 10 unrelated-unanswerable, all 15 law names covered,
unique qids/questions, valid base qids, no key-like strings, and the unchanged formal hash.

- [ ] **Step 2: Run the contract test and verify missing-file failure**

Run: `python -m pytest tests/test_reliability_dataset.py -q`

- [ ] **Step 3: Draft and validate all 60 rows**

Use qids `stress-001` through `stress-060`. Every answerable row reuses an audited source
reference; no row invents a legal article. Style tags come from:
`narrative`, `code_switch`, `typo`, `indirect`, `multi_intent`, and `short_colloquial`.

- [ ] **Step 4: Document evidence separation and run dataset tests**

Run: `python -m pytest tests/test_eval_dataset.py tests/test_reliability_dataset.py -q`

- [ ] **Step 5: Commit**

Commit: `test: add narrative reliability stress benchmark`.

### Task 4: Offline production retrieval run and threshold sweep

**Files:**
- Create: `src/rag/reliability.py`
- Create: `eval/run_reliability_eval.py`
- Create: `tests/test_reliability.py`
- Create after real run: `eval/official/reliability_results.json`
- Create after real run: `eval/official/reliability_trace.jsonl`
- Modify: `eval/official/README.md`
- Modify: `release/public-files.txt`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_official_artifacts.py`

**Interfaces:**
- Consumes: stress rows and production `RetrievalPipeline` results.
- Produces: `compute_reliability_metrics(rows, thresholds) -> dict` and privacy-reduced
  official results/trace.

- [ ] **Step 1: Write failing pure-metric tests**

```python
def test_threshold_sweep_counts_direct_false_refusals():
    metrics = compute_reliability_metrics(
        rows=[trace("a", True, rank=1, score=0.02),
              trace("u", False, rank=None, score=0.01)],
        thresholds=[0.0, 0.03],
    )
    assert metrics["threshold_sweep"]["0.03"]["direct_false_refusals"] == 1
    assert metrics["threshold_sweep"]["0.03"]["direct_unanswerable_coverage"] == 1.0
```

- [ ] **Step 2: Implement deterministic metrics and privacy-reduced export**

Trace fields are limited to `qid`, `answerable`, `rank`, `top_score`, `threshold_refused`, and
`elapsed_ms`.

- [ ] **Step 3: Build local indexes from the audited official dump**

Use a task-specific temporary storage directory and `QDRANT_MODE=local`; do not reuse or
delete any existing storage. Run both index strategies once with the cached immutable model.

- [ ] **Step 4: Run the real production retrieval configuration on all 60 questions**

Run: `python eval/run_reliability_eval.py --dataset eval/dataset/reliability_stress_v0.3.1.jsonl --export-official`

- [ ] **Step 5: Verify the threshold decision gate**

If no candidate threshold is Pareto-better under the design criteria, retain `0.03` and record
that outcome. Never edit `rag.config` merely to improve one metric.

- [ ] **Step 6: Add release-verifier coverage, run tests, and commit**

Run: `python -m pytest tests/test_reliability.py tests/test_official_artifacts.py tests/test_release_verification.py -q`

Run: `python scripts/verify_release.py`

Commit: `eval: publish offline reliability stress evidence`.

### Task 5: Strictly budgeted Gemini/OpenAI cross-check

**Files:**
- Create: `src/rag/provider_budget.py`
- Create: `eval/run_provider_crosscheck.py`
- Create: `tests/test_provider_budget.py`
- Create after authorized run: `eval/official/provider_crosscheck_results.json`
- Create after authorized run: `eval/official/provider_crosscheck_trace.jsonl`
- Modify: `eval/official/README.md`
- Modify: `release/public-files.txt`
- Modify: `src/rag/release_verification.py`

**Interfaces:**
- Consumes: provider, model, per-token pricing, usage, conservative maximum input/output
  tokens, and hard dollar cap.
- Produces: `BudgetLedger.can_start(max_input_tokens, max_output_tokens) -> bool`,
  `record(usage)`, and privacy-reduced provider evidence.

The reviewed safety envelope caps request maxima at 20,000 input and 1,024 output tokens,
bounds each real system + user prompt from UTF-8 bytes plus message overhead before I/O, and
allows raw output only below ignored `eval/runs/`.

- [ ] **Step 1: Write failing hard-cap tests**

```python
def test_ledger_refuses_request_whose_maximum_crosses_cap():
    ledger = BudgetLedger(cap_usd=5, input_per_million=10, output_per_million=30)
    ledger.record(input_tokens=100_000, output_tokens=100_000)
    assert not ledger.can_start(max_input_tokens=200_000, max_output_tokens=100_000)


def test_key_never_appears_in_serialized_ledger():
    assert "api_key" not in BudgetLedger(cap_usd=5, input_per_million=1,
                                           output_per_million=1).to_dict()
```

- [ ] **Step 2: Implement Decimal-based cost arithmetic and fail-closed validation**

Reject non-positive prices, caps above the CLI authorization, negative token counts, unknown
usage, and requests without conservative maximums.

- [ ] **Step 3: Resolve credentials without printing them**

Check process environment and explicitly selected ignored `.env` files. Print only provider
availability booleans. Do not fall back to owner Space secrets or unrelated account keys.

- [ ] **Step 4: Run a small deterministic cross-check under both US$5 caps**

Begin with five questions per provider. Expand only while the ledger proves the next request
cannot exceed the cap. Store raw responses only under ignored `eval/runs/`.

- [ ] **Step 5: Export and verify privacy-reduced evidence**

The committed trace contains qid, requested/actual provider and model, refused flag, citation
count, token counts, estimated cost, and bounded numeric verdicts; it excludes question text,
answer text, keys, provider payloads, and judge reasons.

- [ ] **Step 6: Run tests and commit**

Run: `python -m pytest tests/test_provider_budget.py tests/test_release_verification.py -q`

Commit: `eval: add capped dual-provider cross-check`.

### Task 6: Final verification, protected integration, and free deployment

**Files:**
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `EVAL_REPORT.md`
- Modify: `DESIGN.md`
- Modify: `docs/release/CLAIM_MATRIX.md`
- Modify: `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: verified snapshot, stress evidence, threshold outcome, and provider evidence if run.
- Produces: honest public claims, a protected GitHub PR, and a healthy free Space.

- [ ] **Step 1: Update claims without blending evidence versions**

Label old metrics `v0.1.0 formal evidence`, new retrieval metrics `v0.3.1 reliability stress
evidence`, and provider results `bounded provider cross-check`.

- [ ] **Step 2: Run all local gates**

```bash
python -m pytest -q
python -m ruff check .
python scripts/verify_release.py
python -m bandit -r src scripts eval ui -ll
python -m pip_audit --ignore-vuln GHSA-9356-575X-2W9M \
  --ignore-vuln GHSA-R73W-4P7P-8V6P --ignore-vuln GHSA-M9XQ-6H2J-65R3 \
  --ignore-vuln GHSA-GPCC-6H7G-4XJ9
python -m build
```

- [ ] **Step 3: Push a feature branch and create a protected PR**

Required GitHub CI must pass. Integrate with rebase merge; do not bypass branch protection.

- [ ] **Step 4: Reconcile canonical `main` and deploy changed public files**

Use a parent-commit-guarded Hugging Face commit. Confirm repository SHA, runtime SHA,
`RUNNING`, public visibility, and both current/requested hardware equal `cpu-basic`.

- [ ] **Step 5: Verify the public application**

Health endpoint and root return 200; both provider choices render; no owner key exists; source
panels remain backward compatible until Qdrant is rebuilt.

- [ ] **Step 6: Clean the completed worktree and branches**

Resolve and verify the exact worktree path before removal. Preserve all unrelated historical
worktrees. Confirm canonical HEAD equals `origin/main` and status is clean.
