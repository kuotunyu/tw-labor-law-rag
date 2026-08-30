# v0.3.5 Evidence and Freshness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a compact, deterministic portfolio regression that proves representative citation and refusal behavior, and make the existing manual corpus audit easier to interpret without introducing a scheduler.

**Architecture:** Reuse the pinned corpus, retriever, reranker, threshold, official-artifact, and release-verification boundaries already in the repository. Add a ten-question JSONL contract and a pure result builder modelled on the v0.3.4 wage-arrears regression. Extend the existing read-only live corpus audit with a structured summary instead of creating a second downloader.

**Tech Stack:** Python 3.11, Pydantic, Qdrant in-memory vector store, FlagEmbedding/SentenceTransformers caches, pytest, JSONL/JSON, uv, Ruff.

## Global Constraints

- Do not call Gemini, OpenAI, Hugging Face paid hardware, or any other paid provider.
- Do not write to Qdrant; all evaluation is local and uses the audited corpus snapshot.
- Keep the existing embedding, reranker, RRF, threshold `0.03`, and 15-law corpus unchanged.
- The portfolio regression proves retrieval/source-rank and deterministic refusal policy, not exact generated prose.
- Official artifacts contain no law text, prompts, API keys, endpoints, local paths, timestamps that vary per run, or secret-like fields.
- Corpus checking remains manual and read-only by default; preserve the existing explicit operator-only `--write` mode, but do not create a schedule, heartbeat, automation, or automatic update.
- The audit continues to return `0` for current, `1` for changed, and `2` for invalid/unavailable source data.
- Every new tracked path must be added to Python-sorted `release/public-files.txt` and bound into `release/manifest.json` where it is release evidence.

---

## File map

- `eval/dataset/portfolio_demo_v0.3.5.jsonl`: ten representative answerability, source, route, and refusal contracts.
- `src/rag/portfolio_demo_regression.py`: strict dataset parser and deterministic result/summary builder.
- `tests/test_portfolio_demo_regression.py`: dataset validation, rank/refusal scoring, and runner integration tests.
- `eval/run_portfolio_demo_regression.py`: offline retrieval runner using the pinned corpus and model caches.
- `eval/official/portfolio_demo_v0.3.5.json`: de-identified, recomputable official result.
- `src/rag/release_verification.py`: recompute and verify the portfolio evidence contract.
- `tests/test_release_verification.py`: positive and tamper-detection verifier coverage.
- `tests/test_official_artifacts.py`: exact dataset/result completeness and content-free checks.
- `src/rag/corpus_audit.py`: pure change-summary helper.
- `scripts/audit_corpus.py`: include structured summary in manual audit output.
- `tests/test_corpus_audit.py`: summary and exit-code regression tests.
- `release/corpus_article_snapshot.json`: public article-number/hash inventory with no legal text.
- `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`: operator-only manual audit/update flow.
- `docs/release/CLAIM_MATRIX.md`: claim-to-evidence mapping for the new regression and manual freshness gate.
- `docs/release/EVAL_REPORT.md`: calibrated v0.3.5 evidence interpretation.
- `eval/official/README.md`: reproduction command and evidence boundary.
- `release/manifest.json`: dataset/result hashes and expected portfolio summary.
- `release/public-files.txt`: exact public inventory additions.

---

### Task 1: Ten-question portfolio dataset contract

**Files:**
- Create: `eval/dataset/portfolio_demo_v0.3.5.jsonl`
- Create: `src/rag/portfolio_demo_regression.py`
- Create: `tests/test_portfolio_demo_regression.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: one JSON object per line.
- Produces: immutable `PortfolioCase` records with exact fields `qid`, `question`, `category`, `answerable`, `sources`, `prohibited_sources`, `expect_threshold_refusal`, `expected_refusal_stage`, `required_routes`, `prohibited_routes`, `rationale`, and `style_tags`.

- [ ] **Step 1: Write the failing parser and dataset-contract tests**

```python
from pathlib import Path

import pytest

from rag.portfolio_demo_regression import load_cases


DATASET = Path(__file__).parents[1] / "eval/dataset/portfolio_demo_v0.3.5.jsonl"


def test_portfolio_dataset_has_exact_representative_contract():
    cases = load_cases(DATASET)
    assert [case.qid for case in cases] == [f"portfolio-{number:03d}" for number in range(1, 11)]
    assert sum(case.answerable for case in cases) == 6
    assert sum(case.expect_threshold_refusal for case in cases) == 4
    assert {source["law"] for case in cases for source in case.sources} >= {
        "勞動基準法",
        "勞工請假規則",
        "勞工退休金條例",
        "勞動基準法施行細則",
    }
    assert next(case for case in cases if case.qid == "portfolio-006").required_routes == ("wage_arrears",)
    assert next(case for case in cases if case.qid == "portfolio-002").prohibited_routes == ("wage_arrears",)
    assert next(case for case in cases if case.qid == "portfolio-002").prohibited_sources == ({"law": "勞動基準法", "article": "第 14 條"},)


def test_portfolio_parser_rejects_unknown_or_inconsistent_fields(tmp_path):
    bad = tmp_path / "bad.jsonl"
    bad.write_text('{"qid":"portfolio-001","question":"x","category":"x","answerable":false,"sources":[{"law":"勞動基準法","article":"第 30 條"}],"prohibited_sources":[],"expect_threshold_refusal":true,"expected_refusal_stage":"threshold","required_routes":[],"prohibited_routes":[],"rationale":"x","style_tags":[],"extra":1}\n', encoding="utf-8")
    with pytest.raises(ValueError, match="portfolio case"):
        load_cases(bad)
```

- [ ] **Step 2: Run the focused test and confirm the missing module**

Run: `uv run pytest tests/test_portfolio_demo_regression.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'rag.portfolio_demo_regression'`.

- [ ] **Step 3: Implement the strict parser**

Use frozen dataclasses or strict Pydantic models. Reject unknown fields, blank strings, duplicate/non-sequential qids, answerable cases without sources, unanswerable cases with sources, overlap between expected/prohibited sources, overlap between required/prohibited routes, and any case where `answerable == expect_threshold_refusal`. Require `expected_refusal_stage == "threshold"` exactly when `expect_threshold_refusal` is true and `None` otherwise.

- [ ] **Step 4: Add the exact ten cases**

| qid | category | question intent | expected sources | threshold refusal | required/prohibited routes |
|---|---|---|---|---:|---|
| `portfolio-001` | 工時 | 每日與每週正常工時 | 勞動基準法第 30 條 | false | `[]` |
| `portfolio-002` | 請假 | 婚假天數與薪水 | 勞工請假規則第 2 條; prohibit 勞動基準法第 14 條 | false | required `[]`; prohibited `["wage_arrears"]` |
| `portfolio-003` | 請假 | 普通傷病假與工資 | 勞工請假規則第 4 條 | false | `[]` |
| `portfolio-004` | 特別休假 | 年度終結未休工資 | 勞動基準法第 38 條、施行細則第 24-1 條 | false | `[]` |
| `portfolio-005` | 資遣費 | 新舊制比較 | 勞工退休金條例第 12 條、勞動基準法第 17 條 | false | `[]` |
| `portfolio-006` | 欠薪 | 欠薪立即終止與資遣費 | 勞動基準法第 14 條 | false | `["wage_arrears"]` |
| `portfolio-007` | 時效性 | 現行最低工資金額 | none | true | `[]` |
| `portfolio-008` | 知識庫邊界 | 失業給付 | none | true | `[]` |
| `portfolio-009` | 知識庫邊界 | 著作權保護期間 | none | true | `[]` |
| `portfolio-010` | 知識庫邊界 | 公司最低資本額 | none | true | `[]` |

Use `rationale` for a short human-review explanation and `style_tags` to label only the intended demonstration property, such as `citation`, `multi_source`, `targeted_route`, `time_sensitive_refusal`, `out_of_scope_refusal`, or `collision`; do not encode expected generated prose.

- [ ] **Step 5: Run parser, dataset, and lint checks**

```powershell
uv run pytest tests/test_portfolio_demo_regression.py -q -p no:cacheprovider
uv run ruff check src/rag/portfolio_demo_regression.py tests/test_portfolio_demo_regression.py
uv run python -c "from rag.portfolio_demo_regression import load_cases; from pathlib import Path; assert len(load_cases(Path('eval/dataset/portfolio_demo_v0.3.5.jsonl'))) == 10"
```

Expected: all commands exit 0.

- [ ] **Step 6: Commit Task 1**

```powershell
git add eval/dataset/portfolio_demo_v0.3.5.jsonl src/rag/portfolio_demo_regression.py tests/test_portfolio_demo_regression.py release/public-files.txt
git commit -m "test: define portfolio evidence contract"
```

---

### Task 2: Deterministic result builder and offline runner

**Files:**
- Modify: `src/rag/portfolio_demo_regression.py`
- Create: `eval/run_portfolio_demo_regression.py`
- Modify: `tests/test_portfolio_demo_regression.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: `PortfolioCase`, retrieved law/article identities and ranks, applied route names, threshold-refusal decision, candidate score, audited snapshot hash, and code revision.
- Produces: content-free per-case rows and one summary with `total`, `answerable`, `unanswerable`, `source_recall_at_5`, `answerable_pass_rate`, `threshold_refusal_accuracy`, `route_accuracy`, and `passed`.

- [ ] **Step 1: Write failing pure result-builder tests**

```python
from rag.portfolio_demo_regression import build_result, summarize_results


def test_result_builder_scores_sources_and_refusal_without_answer_text(cases):
    result = build_result(
        cases[0],
        retrieved=[("勞動基準法", "第 30 條", 1)],
        applied_routes=[],
        threshold_refused=False,
        top_score=0.61,
    )
    assert result["source_ranks"] == {"勞動基準法|第 30 條": 1}
    assert result["passed"] is True
    assert "answer" not in result


def test_summary_requires_all_expected_sources_at_five_and_exact_refusal(cases):
    results = [
        build_result(case, retrieved=[], applied_routes=[], threshold_refused=case.expect_threshold_refusal, top_score=0.0)
        for case in cases
    ]
    summary = summarize_results(results)
    assert summary["threshold_refusal_accuracy"] == 1.0
    assert summary["passed"] is False
```

- [ ] **Step 2: Run focused tests and confirm missing interfaces**

Run: `uv run pytest tests/test_portfolio_demo_regression.py -q -p no:cacheprovider`

Expected: FAIL because `build_result` and `summarize_results` do not exist.

- [ ] **Step 3: Implement exact pass arithmetic**

For answerable cases, `passed` requires every expected source at rank `<= 5`, no prohibited source at rank `<= 5`, and `threshold_refused is False`. For unanswerable cases, `passed` requires `threshold_refused is True`. Every case also requires all `required_routes` and no `prohibited_routes`. Record `refusal_stage`, `generation_allowed`, and `generation_called=False`; the latter proves the runner did not invoke a provider, while a threshold refusal must also have `generation_allowed=False`. Round reported rates to six decimal places. Reject duplicate retrieved law/article identities and non-positive ranks.

- [ ] **Step 4: Write a runner contract test with fakes**

Monkeypatch the loader/retriever boundary so the test proves: all ten cases run; no LLM/router/provider module is imported or called; output is stable under a fixed snapshot; and `--output` writes UTF-8 JSON with sorted keys plus a trailing newline.

- [ ] **Step 5: Implement the offline runner**

Follow `eval/run_wage_arrears_regression.py` for repository bootstrap, pinned model-cache discovery, collection construction, candidate retrieval, and deterministic JSON writing. Reuse the production query augmentation/route decision, hybrid retriever, reranker, and threshold decision. Bind the artifact to the dataset hash, audited snapshot path/hash, pinned retrieval configuration, and `git rev-parse HEAD` code revision. Accept `--dataset`, `--snapshot`, `--data-dir`, `--output`, and `--offline`; default to the v0.3.5 paths. Never construct an LLM or send a provider request.

- [ ] **Step 6: Run unit and runner checks**

```powershell
uv run pytest tests/test_portfolio_demo_regression.py -q -p no:cacheprovider
uv run ruff check src/rag/portfolio_demo_regression.py eval/run_portfolio_demo_regression.py tests/test_portfolio_demo_regression.py
uv run python eval/run_portfolio_demo_regression.py --help
```

Expected: all commands exit 0; help identifies the runner as offline and content-free.

- [ ] **Step 7: Commit Task 2**

```powershell
git add src/rag/portfolio_demo_regression.py eval/run_portfolio_demo_regression.py tests/test_portfolio_demo_regression.py release/public-files.txt
git commit -m "feat: add offline portfolio regression runner"
```

---

### Task 3: Official portfolio evidence and release binding

**Files:**
- Create: `eval/official/portfolio_demo_v0.3.5.json`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_release_verification.py`
- Modify: `tests/test_official_artifacts.py`
- Modify: `eval/official/README.md`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: Task 2 deterministic JSON and its exact dataset/snapshot hashes.
- Produces: verifier output section `portfolio_demo_regression` and tamper-evident manifest binding.

- [ ] **Step 1: Write failing verifier and official-artifact tests**

Add a fixture contract and assert that `_verify_portfolio_demo_evidence(root, contract)` recomputes all rates from per-case rows; rejects a changed source rank, refusal flag, qid, dataset hash, snapshot hash, or summary; and returns only the calibrated summary. Assert the official file has exactly ten qids and contains none of `answer`, `content`, `prompt`, `api_key`, `endpoint`, or absolute-path values.

- [ ] **Step 2: Run focused tests and confirm the verifier is missing**

Run: `uv run pytest tests/test_release_verification.py tests/test_official_artifacts.py -q -p no:cacheprovider`

Expected: FAIL because the portfolio evidence verifier and official result do not exist.

- [ ] **Step 3: Run the actual offline regression**

```powershell
$env:TRANSFORMERS_OFFLINE='1'
$env:HF_HUB_OFFLINE='1'
uv run python eval/run_portfolio_demo_regression.py --offline --output eval/official/portfolio_demo_v0.3.5.json
```

Expected: ten cases complete without network/provider access. Inspect every failed case before continuing; change code only for a reproducible contract bug, never to hide a genuine result.

- [ ] **Step 4: Bind and verify the artifact**

Add `evidence.portfolio_demo_regression` to `release/manifest.json` with dataset path/hash, results path/hash, corpus snapshot path/hash, code revision, expected total/answerable/unanswerable counts, and the exact recomputed summary. Implement `_verify_portfolio_demo_evidence`, call it from `verify_release`, and include the returned section in the final report. The verifier must also prove required/prohibited route behavior, threshold-stage alignment, and `generation_called=False` for every case.

- [ ] **Step 5: Document calibrated interpretation**

In `eval/official/README.md`, `docs/release/EVAL_REPORT.md`, and `docs/release/CLAIM_MATRIX.md`, distinguish this ten-case demonstration regression from the 40-question formal benchmark, 60-case reliability suite, and archived provider judgments. Claim only the measured source/refusal behavior present in the artifact.

- [ ] **Step 6: Run evidence verification**

```powershell
uv run pytest tests/test_portfolio_demo_regression.py tests/test_official_artifacts.py tests/test_release_verification.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
uv run ruff check src/rag/portfolio_demo_regression.py src/rag/release_verification.py eval/run_portfolio_demo_regression.py tests/test_portfolio_demo_regression.py
git diff --check
```

Expected: all commands exit 0 and the verifier reports ten bound portfolio cases.

- [ ] **Step 7: Commit Task 3**

```powershell
git add eval/official/portfolio_demo_v0.3.5.json src/rag/release_verification.py tests/test_release_verification.py tests/test_official_artifacts.py eval/official/README.md docs/release/EVAL_REPORT.md docs/release/CLAIM_MATRIX.md release/manifest.json release/public-files.txt
git commit -m "test: bind v0.3.5 portfolio evidence"
```

---

### Task 4: Actionable law- and article-level manual freshness report

**Files:**
- Modify: `src/rag/corpus_audit.py`
- Modify: `scripts/audit_corpus.py`
- Modify: `tests/test_corpus_audit.py`
- Create: `release/corpus_article_snapshot.json`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_release_verification.py`
- Modify: `tests/test_official_artifacts.py`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md`

**Interfaces:**
- Consumes: the existing law/source snapshot plus a new content-free article-number/hash snapshot built during the same official-source download.
- Produces: `summarize_changes(changes)` for source/law fields and `compare_article_snapshots(committed, live)` for added, removed, changed, and unchanged articles; CLI JSON gains one deterministic combined `summary` object.

- [ ] **Step 1: Write failing summary tests**

```python
from rag.corpus_audit import compare_article_snapshots, summarize_changes


def test_change_summary_counts_laws_and_changed_fields():
    changes = [
        {"source": "acts", "kind": "added"},
        {"law": "乙法", "kind": "removed"},
        {"source": "regulations", "kind": "sha256", "old": "a" * 64, "new": "b" * 64},
        {"law": "丙法", "kind": "last_amended", "old": "20260101", "new": "20260830"},
        {"law": "丙法", "kind": "num_articles", "old": 10, "new": 11},
        {"law": "丁法", "kind": "content_sha256", "old": "c" * 64, "new": "d" * 64},
        {"law": "丁法", "kind": "effective_date", "old": "20260101", "new": "20260830"},
    ]
    assert summarize_changes(changes) == {
        "total_changes": 7,
        "subjects_changed": 4,
        "added": 1,
        "removed": 1,
        "sha256": 1,
        "last_amended": 1,
        "effective_date": 1,
        "num_articles": 1,
        "content_sha256": 1,
    }


def test_article_comparison_reports_counts_without_content():
    committed = {
        "甲法": {"第 1 條": "a" * 64, "第 2 條": "b" * 64, "第 3 條": "c" * 64}
    }
    live = {
        "甲法": {"第 1 條": "a" * 64, "第 2 條": "d" * 64, "第 4 條": "e" * 64}
    }
    report = compare_article_snapshots(committed, live)
    assert report["summary"] == {"added": 1, "removed": 1, "changed": 1, "unchanged": 1}
    assert report["changes"] == [
        {"law": "甲法", "article": "第 2 條", "kind": "changed", "old": "b" * 64, "new": "d" * 64},
        {"law": "甲法", "article": "第 3 條", "kind": "removed", "old": "c" * 64},
        {"law": "甲法", "article": "第 4 條", "kind": "added", "new": "e" * 64},
    ]
    assert "content" not in repr(report)
```

- [ ] **Step 2: Run the focused test and confirm the helper is missing**

Run: `uv run pytest tests/test_corpus_audit.py -q -p no:cacheprovider`

Expected: collection fails because the new summary/article interfaces do not exist.

- [ ] **Step 3: Implement content-free article fingerprints and comparisons**

Add `build_article_snapshot(laws, snapshot_date)` with schema `1.0`, law names, article labels, and SHA-256 of canonical `{no, chapter, content}` records; it must not retain article text. Add `compare_article_snapshots` with the exact sorted change schema in Step 1. Reject duplicate law/article identities and invalid hashes. Keep `release/corpus_snapshot.json` unchanged so the active v0.3.4 Qdrant receipt retains its original provenance hash.

- [ ] **Step 4: Implement the combined manual audit output**

Download/parse official archives once, derive both snapshots, and preserve `build_live_snapshot()` as a backward-compatible wrapper. Reject an unknown law-change `kind` rather than silently dropping it. In check mode emit:

```json
{
  "status": "current|changed",
  "changes": {"laws": [], "articles": []},
  "summary": {
    "laws": {"total_changes": 0, "subjects_changed": 0, "added": 0, "removed": 0, "sha256": 0, "last_amended": 0, "effective_date": 0, "num_articles": 0, "content_sha256": 0},
    "articles": {"added": 0, "removed": 0, "changed": 0, "unchanged": 884}
  }
}
```

Add `--article-check` defaulting to `release/corpus_article_snapshot.json`. Preserve explicit operator-only `--write`, but require its paired `--article-write` so both baselines advance together. Add one bootstrap-only `--bootstrap-article-snapshot PATH` mode that writes the article snapshot only after the freshly downloaded law/source snapshot compares current with the committed `--check` file; otherwise return `1` without writing. Check/bootstrap modes never invoke Qdrant.

- [ ] **Step 5: Test current, changed, bootstrap, write, and invalid-source exits**

Extend CLI tests to assert `0`, `1`, and `2` remain unchanged; bootstrap refuses a changed law/source snapshot; paired write is required; output never contains downloaded law content; and a temporary file is replaced only after validation. Run:

```powershell
uv run pytest tests/test_corpus_audit.py -q -p no:cacheprovider
uv run ruff check src/rag/corpus_audit.py scripts/audit_corpus.py tests/test_corpus_audit.py
```

Expected: both commands exit 0.

- [ ] **Step 6: Bootstrap and bind the v0.3.5 article snapshot**

Run the bootstrap mode, which must compare the freshly downloaded law/source snapshot with the committed baseline before it writes anything:

```powershell
uv run python scripts/audit_corpus.py --check release/corpus_snapshot.json --bootstrap-article-snapshot release/corpus_article_snapshot.json
```

Expected: exit `0` and one new article snapshot only when the law/source baseline is current. Then run the default check once against both baselines. Verify the artifact has 15 unique laws, 884 unique non-deleted articles, valid hashes, and no content/chapter/URL fields. Bind its path/hash/counts in `release/manifest.json`; extend the release verifier and official-artifact privacy tests to reject content, duplicates, wrong totals, invalid hashes, and manifest tampering.

- [ ] **Step 7: Document the attended operator flow**

Update the runbook with this exact sequence: run `uv run python scripts/audit_corpus.py`; stop if exit `2`; review the named laws/fields when exit `1`; create a time-limited writer key only after review; rebuild candidate collections; verify receipt and read-only runtime; switch candidate base; revoke writer/transition keys; delete downloaded key files. State explicitly that no schedule exists.

- [ ] **Step 8: Run the bound freshness checks**

```powershell
uv run pytest tests/test_corpus_audit.py tests/test_official_artifacts.py tests/test_release_verification.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: all commands exit 0 and the verifier reports 15 laws/884 article fingerprints with no legal text.

- [ ] **Step 9: Commit Task 4**

```powershell
git add src/rag/corpus_audit.py scripts/audit_corpus.py tests/test_corpus_audit.py release/corpus_article_snapshot.json src/rag/release_verification.py tests/test_release_verification.py tests/test_official_artifacts.py release/manifest.json release/public-files.txt docs/deployment/BYOK_HUGGINGFACE_RUNBOOK.md
git commit -m "feat: prove manual article freshness"
```

---

### Task 5: Evidence/freshness checkpoint

**Files:**
- No product files expected; fix only failures attributable to Tasks 1–4.

**Interfaces:**
- Consumes: all prior tasks in this plan.
- Produces: one independently reviewable evidence/freshness checkpoint.

- [ ] **Step 1: Run the checkpoint**

```powershell
uv lock --check
uv run ruff check .
uv run pytest tests/test_portfolio_demo_regression.py tests/test_corpus_audit.py tests/test_answerer.py tests/test_api_response.py tests/test_qdrant_maintenance.py tests/test_official_artifacts.py tests/test_release_verification.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Review the checkpoint diff**

Run: `git log --oneline e3940b9..HEAD; git diff --stat e3940b9..HEAD; git status --short`

Expected: only planned dataset, evaluation, audit, documentation, test, manifest, and inventory paths appear; the worktree is clean.
