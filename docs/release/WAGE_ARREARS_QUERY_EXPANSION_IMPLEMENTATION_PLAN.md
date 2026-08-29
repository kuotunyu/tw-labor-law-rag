# Wage-Arrears Query Expansion Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a narrowly gated, deterministic Article 14 retrieval expansion for wage-nonpayment questions in which a worker asks about leaving immediately.

**Architecture:** Extend the existing pure `_retrieval_query()` helper with one additional all-cue rule. Retrieval and reranking receive the expanded search string, while `Answerer` continues to send the original user question to the generator. No provider, Qdrant, API, UI, prompt, model, or threshold interface changes.

**Tech Stack:** Python 3.11+, pytest, Ruff, Qdrant client test doubles, uv, existing release verifier

## Global Constraints

- Require at least one reviewed wage-nonpayment cue and one reviewed worker immediate-termination cue.
- Append exactly `勞動基準法 第十四條 不依勞動契約給付工作報酬 勞工得不經預告終止契約`.
- Preserve expansion order: off-hours, severance, wage arrears.
- Preserve the original question for generation.
- Do not change the reranker threshold, retrieval depth, model revisions, prompts, BYOK policy, UI, Qdrant collections, or Space settings.
- Do not call Gemini or OpenAI, mutate Qdrant, use a paid service, or expose any API key.
- Do not rewrite the committed formal Hit@5, MRR, or refusal evidence.
- Keep public release inventory exact and privacy verification passing.

## File Structure

- `tests/test_pipeline.py`: deterministic positive, collision, and stable-order retrieval-query contracts.
- `tests/test_answerer.py`: integration contract proving expansion does not reach the generation question.
- `src/rag/retrieval/pipeline.py`: finite cue tables and the single new all-cue expansion rule.
- `README.md` and `README.en.md`: v0.3.4 behavior and evidence-boundary release notes.
- `DESIGN.md`: architecture decision and trade-off record.
- `EVAL_REPORT.md`: historical `eval-10` result plus the precisely limited v0.3.4 mitigation status.
- `pyproject.toml`, `uv.lock`, `release/manifest.json`, and `src/rag/release_verification.py`: v0.3.4 version contract.
- `tests/test_release_verification.py`: expected v0.3.4 release contract and exact public-file count.
- `docs/release/WAGE_ARREARS_QUERY_EXPANSION_DESIGN.md`: approved specification status.

---

### Task 1: Define the failing retrieval and generation contracts

**Files:**
- Modify: `tests/test_pipeline.py`
- Modify: `tests/test_answerer.py`

**Interfaces:**
- Consumes: `RetrievalPipeline.run(query: str) -> RetrievalResult`
- Consumes: `Answerer.answer(question: str) -> Answer`
- Produces: an executable contract for the exact Article 14 term block, collision safety, rule order, and original generation question

- [ ] **Step 1: Add the positive pipeline contract**

Add this test after the existing severance expansion tests in `tests/test_pipeline.py`:

```python
@pytest.mark.parametrize(
    "question",
    [
        "公司一直拖欠薪水,我可以不經預告直接離職嗎?這樣還能拿到資遣費嗎?",
        "公司已經兩個月沒有付 salary，我想今天直接 resign 又怕拿不到 severance；雇主欠薪時能否不經預告終止契約，之後還能請求資遣費嗎？",
        "公司連續欠薪後要求我照常打卡，還說沒有先交 resignation notice 就拿不到任何錢；我能否立即終止，並依哪個規則請求資遣費？",
    ],
)
def test_pipeline_expands_wage_nonpayment_worker_termination(question):
    calls = []
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls).run(question)

    expected = f"{question} {legal_terms}"
    assert calls == [("retrieve", expected), ("rerank", expected)]
```

In the same file, add exhaustive finite-vocabulary contracts so every reviewed
cue is executable evidence rather than an untested table entry:

```python
@pytest.mark.parametrize(
    "wage_cue",
    [
        "欠薪",
        "沒發薪",
        "沒有發薪",
        "未發薪",
        "沒付薪",
        "沒有付薪",
        "未付薪",
        "拖欠工資",
        "積欠工資",
        "沒付工資",
        "沒有付工資",
        "未付工資",
        "未給付工資",
        "未給付工作報酬",
        "沒有付 salary",
        "沒付 salary",
        "unpaid salary",
        "wage arrears",
    ],
)
def test_pipeline_accepts_each_reviewed_wage_nonpayment_cue(wage_cue):
    calls = []
    question = f"公司{wage_cue}，我想直接離職。"
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", f"{question} {legal_terms}")]


@pytest.mark.parametrize(
    "exit_cue",
    [
        "直接離職",
        "立即離職",
        "馬上離職",
        "立刻離職",
        "立即終止",
        "直接終止",
        "直接 resign",
        "immediately resign",
        "resign without notice",
    ],
)
def test_pipeline_accepts_each_reviewed_worker_exit_cue(exit_cue):
    calls = []
    question = f"公司欠薪，我想{exit_cue}。"
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", f"{question} {legal_terms}")]
```

- [ ] **Step 2: Add collision tests**

Add this parameterized test in `tests/test_pipeline.py`:

```python
@pytest.mark.parametrize(
    "question",
    [
        "公司已經欠薪兩個月，該怎麼追討？",
        "我想直接離職，應該怎麼做？",
        "我是雇主，公司欠薪後可以不經預告直接解僱勞工嗎？",
        "公司薪資怎麼算？我將來可能離職。",
    ],
)
def test_pipeline_requires_wage_and_worker_exit_cue_groups(question):
    calls = []

    recorded_pipeline(calls, rerank=False).run(question)

    assert calls == [("retrieve", question)]
```

- [ ] **Step 3: Extend the stable-order contract**

Replace the body of `test_pipeline_appends_each_matching_rule_once_in_stable_order` with:

```python
calls = []
question = (
    "老闆在休假日用群組傳訊，說公司欠薪、也要我直接離職，"
    "還附上勞退新制與勞基法舊制資遣費試算。"
)
off_hours = "雇主 休息日 例假 工作時間 延長工作時間 出勤 加班"
severance = "資遣費 勞工退休金條例 勞動基準法 工作年資 平均工資 六個月"
wage_arrears = (
    "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
    "勞工得不經預告終止契約"
)
expected = f"{question} {off_hours} {severance} {wage_arrears}"

recorded_pipeline(calls).run(question)

assert calls == [("retrieve", expected), ("rerank", expected)]
```

- [ ] **Step 4: Add the answer-generation boundary test**

Add this test to `tests/test_answerer.py`:

```python
def test_answerer_expands_retrieval_but_keeps_original_generation_question():
    captured = {}

    class RecordingRetriever:
        def retrieve(self, query, top_k):
            captured["query"] = query
            return [make_hit("c1", "勞動基準法", "第 14 條", "證據內容")]

    question = "公司一直拖欠薪水，我可以直接離職嗎？"
    legal_terms = (
        "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
        "勞工得不經預告終止契約"
    )
    llm = FakeLLM("依 [1] 回答。")
    pipeline = RetrievalPipeline(
        RecordingRetriever(), reranker=None, top_k_retrieve=20, top_k_final=5
    )

    Answerer(pipeline, llm).answer(question)

    assert captured["query"] == f"{question} {legal_terms}"
    assert question in llm.calls[0]["user"]
    assert legal_terms not in llm.calls[0]["user"]
```

- [ ] **Step 5: Run the new contract and verify the red state**

Run:

```powershell
uv run pytest tests/test_pipeline.py tests/test_answerer.py -q
```

Expected: the three positive cases, stable-order case, and answerer retrieval assertion fail because the Article 14 expansion does not exist; collision tests and unrelated existing tests pass.

- [ ] **Step 6: Commit the failing contract**

```powershell
git add tests/test_pipeline.py tests/test_answerer.py
git commit -m "test: define wage arrears query expansion contract"
```

---

### Task 2: Implement the minimal deterministic expansion

**Files:**
- Modify: `src/rag/retrieval/pipeline.py`
- Test: `tests/test_pipeline.py`
- Test: `tests/test_answerer.py`

**Interfaces:**
- Consumes: `_matches_all(folded: str, *cue_groups: tuple[str, ...]) -> bool`
- Produces: `_WAGE_NONPAYMENT_CUES`, `_WORKER_IMMEDIATE_TERMINATION_CUES`, and `_WAGE_ARREARS_LEGAL_TERMS`
- Preserves: `_retrieval_query(query: str) -> str`

- [ ] **Step 1: Add the reviewed cue tables and legal terms**

Add after `_SEVERANCE_LEGAL_TERMS` in `src/rag/retrieval/pipeline.py`:

```python
_WAGE_NONPAYMENT_CUES = (
    "欠薪",
    "沒發薪",
    "沒有發薪",
    "未發薪",
    "沒付薪",
    "沒有付薪",
    "未付薪",
    "拖欠工資",
    "積欠工資",
    "沒付工資",
    "沒有付工資",
    "未付工資",
    "未給付工資",
    "未給付工作報酬",
    "沒有付 salary",
    "沒付 salary",
    "unpaid salary",
    "wage arrears",
)
_WORKER_IMMEDIATE_TERMINATION_CUES = (
    "直接離職",
    "立即離職",
    "馬上離職",
    "立刻離職",
    "立即終止",
    "直接終止",
    "直接 resign",
    "immediately resign",
    "resign without notice",
)
_WAGE_ARREARS_LEGAL_TERMS = (
    "勞動基準法 第十四條 不依勞動契約給付工作報酬 "
    "勞工得不經預告終止契約"
)
```

- [ ] **Step 2: Add the all-cue rule after the severance rule**

Append this branch immediately before `_retrieval_query()` returns:

```python
if _matches_all(folded, _WAGE_NONPAYMENT_CUES, _WORKER_IMMEDIATE_TERMINATION_CUES):
    expansions.append(_WAGE_ARREARS_LEGAL_TERMS)
```

- [ ] **Step 3: Run the targeted tests and verify green**

Run:

```powershell
uv run pytest tests/test_pipeline.py tests/test_answerer.py -q
```

Expected: all targeted tests pass.

- [ ] **Step 4: Run the retrieval-adjacent regression tests**

Run:

```powershell
uv run pytest tests/test_retriever.py tests/test_fusion.py tests/test_pipeline.py tests/test_answerer.py -q
uv run ruff check src/rag/retrieval/pipeline.py tests/test_pipeline.py tests/test_answerer.py
```

Expected: both commands pass with no new warning or lint finding.

- [ ] **Step 5: Commit the implementation**

```powershell
git add src/rag/retrieval/pipeline.py
git commit -m "feat: expand wage arrears termination queries"
```

---

### Task 3: Prepare the evidence-limited v0.3.4 release metadata

**Files:**
- Modify: `tests/test_release_verification.py`
- Modify: `pyproject.toml`
- Modify: `uv.lock`
- Modify: `release/manifest.json`
- Modify: `src/rag/release_verification.py`
- Modify: `README.md`
- Modify: `README.en.md`
- Modify: `DESIGN.md`
- Modify: `EVAL_REPORT.md`

**Interfaces:**
- Consumes: implemented deterministic query-expansion contract from Task 2
- Produces: a consistent `v0.3.4` release/package contract
- Preserves: formal evidence version `v0.1.0` and all committed quality metrics

- [ ] **Step 1: Move version expectations to v0.3.4**

In `tests/test_release_verification.py`, change the default package/release fixture and both report assertions from `0.3.3`/`v0.3.3` to `0.3.4`/`v0.3.4`. Keep `formal_evidence_version` equal to `v0.1.0` and keep the exact public-file assertion equal to `133`.

- [ ] **Step 2: Run the version contract and verify the red state**

Run:

```powershell
uv run pytest tests/test_release_verification.py::test_release_version_contract_is_explicit_and_consistent -q
```

Expected: FAIL because `pyproject.toml` and `release/manifest.json` still report v0.3.3.

- [ ] **Step 3: Update the release version contract**

Make these exact changes:

```text
pyproject.toml: project.version = "0.3.4"
release/manifest.json: release_version = "v0.3.4"
src/rag/release_verification.py: expected release version = "v0.3.4"
```

Then run:

```powershell
uv lock
uv lock --check
```

Expected: the root package entry in `uv.lock` becomes `0.3.4`, and the lock check passes.

- [ ] **Step 4: Update evidence-limited documentation**

Add this section near the top of `README.md`:

```markdown
## v0.3.4 欠薪／立即離職檢索強化

只有同時命中「欠薪」與「勞工立即離職」兩組已審閱 cue 的問題，檢索管線才會補上《勞動基準法》第 14 條的固定法規詞。BM25、向量檢索與 reranker 看到擴充查詢；生成模型仍收到使用者原始問題。

本版沒有新增 provider 呼叫、調整 0.03 門檻、重建 Qdrant 或改寫歷史指標。`v0.1.0` formal baseline 與 `v0.3.1` reliability evidence 保持原證據版本；v0.3.4 的公開主張只涵蓋可由單元測試驗證的決定論式路由契約。
```

Add the English counterpart near the top of `README.en.md`:

```markdown
## v0.3.4 wage-arrears/immediate-exit retrieval hardening

Only questions matching both reviewed wage-nonpayment and worker immediate-exit cue groups receive fixed Labor Standards Act Article 14 retrieval terms. BM25, dense retrieval, and the reranker see the expanded query; generation still receives the visitor's original question.

This release adds no provider call, 0.03 threshold change, Qdrant rebuild, or historical metric rewrite. The `v0.1.0` formal baseline and `v0.3.1` reliability evidence keep their original evidence versions; the v0.3.4 public claim is limited to the unit-testable deterministic routing contract.
```

Add this decision record before the numbered sections in `DESIGN.md`:

```markdown
## v0.3.4 為什麼只針對欠薪／立即離職做 query expansion？

**選擇**：只有欠薪與勞工立即終止兩組 cue 同時命中時，才為檢索及 reranker 補入《勞動基準法》第 14 條用語；生成階段保留原始問題。

**理由**：正式評估唯一誤拒 `eval-10` 已留下具體詞彙鴻溝證據。兩組 cue 的 conjunction 能修補這個已量測案例，又避免把一般薪資、一般離職或雇主解僱問題廣泛導向第 14 條，且不增加 provider 成本。

**Tradeoff**：這是有限詞彙的決定論式規則，不是通用法律 query rewriting；未重跑完整固定模型評估前，不宣稱歷史 Hit@5、MRR 或誤拒率已提升。
```

In `EVAL_REPORT.md` under Case 1, retain every historical number and add:

```markdown
> **v0.3.4 mitigation boundary:** The runtime now applies a deterministic Article 14 retrieval expansion only when reviewed wage-nonpayment and worker immediate-exit cue groups both match. This is a tested routing contract, not a recomputation of the historical `eval-10` result or aggregate metrics.
```

- [ ] **Step 5: Run release tests and verifier**

Run:

```powershell
uv run pytest tests/test_release_verification.py tests/test_official_artifacts.py -q
uv run python scripts/verify_release.py
```

Expected: both commands pass; the verifier reports release `v0.3.4`, package `0.3.4`, formal evidence `v0.1.0`, 133 public files, and unchanged formal/reliability metrics.

- [ ] **Step 6: Commit the release preparation**

```powershell
git add pyproject.toml uv.lock release/manifest.json src/rag/release_verification.py tests/test_release_verification.py README.md README.en.md DESIGN.md EVAL_REPORT.md
git commit -m "release: prepare v0.3.4 wage arrears retrieval"
```

---

### Task 4: Run the complete offline verification chain

**Files:**
- Verify: entire tracked public source tree
- Modify only if a verification failure identifies a concrete defect

**Interfaces:**
- Consumes: all Task 1-3 commits
- Produces: completion evidence with no network provider call or Qdrant mutation

- [ ] **Step 1: Run style, lock, and security verification**

Run:

```powershell
uv lock --check
uv run ruff check .
uv run bandit -r src scripts -ll
$env:PYTHONUTF8 = '1'
uv run pip-audit --local
Remove-Item Env:PYTHONUTF8
```

Expected: all commands exit zero. Only explicitly committed dependency-audit policy applies; no new advisory may be ignored.

- [ ] **Step 2: Run the full tests and release verifier**

Run:

```powershell
uv run pytest -q
uv run python scripts/verify_release.py
```

Expected: all tests pass; the only accepted warnings are the pre-existing third-party jieba deprecation warnings. Release verifier status is `pass`.

- [ ] **Step 3: Build and inspect the source artifact**

Run:

```powershell
uv build
uv run pytest tests/test_packaging.py -q
```

Expected: wheel and source distribution build successfully, packaging tests pass, and no private corpus, `.env`, API key, local index, or worktree path enters the artifacts.

- [ ] **Step 4: Inspect the final branch boundary**

Run:

```powershell
git diff --check b500e30971b5082a15ebe77587705856ca2a1d95..HEAD
git diff --stat b500e30971b5082a15ebe77587705856ca2a1d95..HEAD
git status --short --branch
```

Expected: no whitespace error, only approved v0.3.4 files differ, and the worktree is clean.

- [ ] **Step 5: Record the verified status**

After every Step 1-4 command passes, change the first status line in `docs/release/WAGE_ARREARS_QUERY_EXPANSION_DESIGN.md` to:

```markdown
**Status:** Implemented and verified on 2026-08-30
```

Run:

```powershell
git add docs/release/WAGE_ARREARS_QUERY_EXPANSION_DESIGN.md
git commit -m "docs: record v0.3.4 verification"
uv run python scripts/verify_release.py
git status --short --branch
```

Expected: the final verifier passes and the worktree is clean. Do not merge, tag, push, or deploy until the branch review is complete.

---

# v0.3.4 Wage-Arrears Targeted Regression Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Commit a privacy-reduced, reproducible 20-case offline regression that proves the v0.3.4 Article 14 expansion boundary and positive-case Hit@5 without rewriting historical formal metrics.

**Architecture:** A small `rag.wage_arrears_regression` module owns dataset validation and normalized result construction. A dedicated eval entry point reuses the audited-corpus and isolated-index helpers from the reliability runner, rejects missing pinned model snapshots before any download, executes only structure/hybrid/rerank retrieval, and exports one deterministic public JSON result after the acceptance gates pass.

**Tech Stack:** Python 3.11, pytest, Pydantic settings, Hugging Face local cache, FlagEmbedding, BM25, Qdrant local mode, Ruff, Bandit, pip-audit.

## Global Constraints

- Use exactly 20 reviewed rows: 10 positive and 10 collision cases.
- Positive rows require `勞動基準法` Article `第 14 條` in the final top five.
- Top-one rate is reported but is not a hard gate.
- Do not call Gemini, OpenAI, Anthropic, Ollama, or any generation/judge provider.
- Do not connect to or mutate Qdrant Cloud; any local index must live only under ignored `eval/runs/` state.
- Require both pinned model snapshots to exist in the local Hugging Face cache before corpus or index work begins.
- Never commit question text, raw hit text, secrets, endpoints, absolute paths, or raw local traces in the official result.
- Keep `eval/dataset/eval_set.jsonl`, `eval/dataset/reliability_stress_v0.3.1.jsonl`, and all existing official result artifacts byte-for-byte unchanged.
- Keep release/package version `v0.3.4` / `0.3.4`; this adds evidence, not runtime behavior.

---

### Task 1: Freeze and validate the targeted dataset

**Files:**
- Create: `eval/dataset/wage_arrears_regression_v0.3.4.jsonl`
- Create: `src/rag/wage_arrears_regression.py`
- Create: `tests/test_wage_arrears_regression.py`

**Interfaces:**
- Consumes: `rag.retrieval.pipeline._retrieval_query` as the shipped routing behavior.
- Produces: `load_regression_dataset(path: Path) -> list[dict]`, `route_expansion_applied(question: str) -> bool`, and a frozen 20-row dataset.

- [x] **Step 1: Write failing schema and routing tests**

Add tests that import the missing module, load the missing dataset, and assert literal expectations:

```python
def test_targeted_dataset_has_reviewed_shape_and_routes():
    rows = load_regression_dataset(DATASET_PATH)
    assert len(rows) == 20
    assert [row["qid"] for row in rows] == [f"wage-reg-{i:03d}" for i in range(1, 21)]
    assert sum(row["expect_expansion"] for row in rows) == 10
    assert all(route_expansion_applied(row["question"]) is row["expect_expansion"] for row in rows)
    assert all(
        row["sources"] == [{"doc": "勞動基準法", "article": "第 14 條"}]
        for row in rows[:10]
    )
    assert all(row["sources"] == [] for row in rows[10:])
```

Add a malformed temporary JSONL case and assert `ValueError` names its qid and invalid field. The production mutation these tests catch is accepting a duplicate id, a non-boolean decision, a positive row without Article 14, or a collision row with a gold source.

- [x] **Step 2: Run the tests and verify RED**

Run:

```powershell
uv run pytest tests/test_wage_arrears_regression.py -q
```

Expected: collection fails because `rag.wage_arrears_regression` does not exist.

- [x] **Step 3: Add the 20 literal dataset rows and minimal validator**

Implement strict JSONL parsing with these required keys:

```python
REQUIRED_FIELDS = {"qid", "question", "expect_expansion", "sources", "style_tags"}
ARTICLE_14_SOURCE = [{"doc": "勞動基準法", "article": "第 14 條"}]

def load_regression_dataset(path: Path) -> list[dict]:
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    seen: set[str] = set()
    for index, row in enumerate(rows, start=1):
        if not isinstance(row, dict):
            raise ValueError(f"row {index}: expected object")
        qid = row.get("qid", f"row {index}")
        if set(row) != REQUIRED_FIELDS:
            raise ValueError(f"{qid}: fields must equal {sorted(REQUIRED_FIELDS)}")
        if not isinstance(row["qid"], str) or not row["qid"]:
            raise ValueError(f"row {index}: qid must be a non-empty string")
        if row["qid"] in seen:
            raise ValueError(f"{row['qid']}: duplicate qid")
        seen.add(row["qid"])
        if not isinstance(row["question"], str) or not row["question"].strip():
            raise ValueError(f"{row['qid']}: question must be a non-empty string")
        if type(row["expect_expansion"]) is not bool:
            raise ValueError(f"{row['qid']}: expect_expansion must be boolean")
        expected_sources = ARTICLE_14_SOURCE if row["expect_expansion"] else []
        if row["sources"] != expected_sources:
            raise ValueError(f"{row['qid']}: sources do not match expansion decision")
        if not isinstance(row["style_tags"], list) or not row["style_tags"]:
            raise ValueError(f"{row['qid']}: style_tags must be a non-empty list")
    expected_ids = [f"wage-reg-{index:03d}" for index in range(1, 21)]
    if [row["qid"] for row in rows] != expected_ids:
        raise ValueError("dataset qids must be wage-reg-001 through wage-reg-020")
    return rows

def route_expansion_applied(question: str) -> bool:
    return _WAGE_ARREARS_LEGAL_TERMS in _retrieval_query(question)
```

Use qids `wage-reg-001` through `wage-reg-020`. Rows 1-10 cover colloquial Chinese, statutory Chinese, code switching, punctuation, and narratives. Rows 11-20 cover wage recovery only, ordinary resignation only, employer dismissal, notice-only, generic salary, severance-only, and unrelated Article 14 wording.

- [x] **Step 4: Run GREEN and the existing pipeline contract tests**

Run:

```powershell
uv run pytest tests/test_wage_arrears_regression.py tests/test_pipeline.py -q
```

Expected: all tests pass and no provider is initialized.

- [x] **Step 5: Commit Task 1**

```powershell
git add eval/dataset/wage_arrears_regression_v0.3.4.jsonl src/rag/wage_arrears_regression.py tests/test_wage_arrears_regression.py
git commit -m "test: add v0.3.4 wage arrears regression set"
```

---

### Task 2: Build a fail-closed offline runner and normalized result contract

**Files:**
- Create: `eval/run_wage_arrears_regression.py`
- Modify: `src/rag/wage_arrears_regression.py`
- Modify: `tests/test_wage_arrears_regression.py`

**Interfaces:**
- Consumes: Task 1 dataset helpers, reliability runner corpus/index helpers, `lib.match_rank`, pinned `Settings` model revisions.
- Produces: `require_cached_models(settings: Settings) -> dict[str, str]`, `build_public_result(...) -> dict`, and CLI exit code 0 only when all acceptance gates pass.

- [x] **Step 1: Write failing cache-preflight and privacy-result tests**

Test a resolver that raises `LocalEntryNotFoundError` and assert the helper raises:

```text
missing pinned local model snapshot: BAAI/bge-m3@5617a9f61b028005a4858fdac845db406aefb181
```

Build 20 literal raw case summaries and assert `build_public_result` returns only stable qid, expected/applied decision, rank, and rounded score. Recursively serialize the result and assert the original questions, `https://`, `AIza`, `sk-`, and the worktree path are absent. The production mutations caught are allowing an implicit model download or leaking raw questions/hits into public evidence.

- [x] **Step 2: Run the focused tests and verify RED**

```powershell
uv run pytest tests/test_wage_arrears_regression.py -q
```

Expected: failures name the missing cache-preflight and result-builder functions.

- [x] **Step 3: Implement minimal helpers and runner**

Use `huggingface_hub.snapshot_download(..., local_files_only=True)` for both pinned repositories before `_materialize_audited_corpus`. The runner must:

```python
def require_cached_models(settings: Settings, resolver=None) -> dict[str, str]:
    resolve = resolver or snapshot_download
    models = {
        settings.embedding_model: settings.embedding_model_revision,
        settings.reranker_model: settings.reranker_model_revision,
    }
    resolved = {}
    for repo_id, revision in models.items():
        try:
            resolved[repo_id] = str(
                resolve(repo_id=repo_id, revision=revision, local_files_only=True)
            )
        except LocalEntryNotFoundError as exc:
            raise RuntimeError(
                f"missing pinned local model snapshot: {repo_id}@{revision}"
            ) from exc
    return resolved

def build_public_result(*, dataset_path, code_revision, configuration, cases):
    public_cases = [
        {
            "qid": case["qid"],
            "expected_expansion": case["expected_expansion"],
            "expansion_applied": case["expansion_applied"],
            "rank": case["rank"],
            "top_score": round(case["top_score"], 4),
        }
        for case in sorted(cases, key=lambda item: item["qid"])
    ]
    positives = [case for case in public_cases if case["expected_expansion"]]
    collisions = [case for case in public_cases if not case["expected_expansion"]]
    summary = {
        "positive_routes": sum(case["expansion_applied"] for case in positives),
        "collision_routes_avoided": sum(
            not case["expansion_applied"] for case in collisions
        ),
        "positive_hit_at_5": sum(
            case["rank"] is not None and case["rank"] <= 5 for case in positives
        ),
        "positive_hit_at_1": sum(case["rank"] == 1 for case in positives),
    }
    summary["passed"] = (
        summary["positive_routes"] == 10
        and summary["collision_routes_avoided"] == 10
        and summary["positive_hit_at_5"] == 10
    )
    return {
        "schema_version": "1.0",
        "dataset": {
            "path": "eval/dataset/wage_arrears_regression_v0.3.4.jsonl",
            "sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
            "questions": len(public_cases),
        },
        "code_revision": code_revision,
        "configuration": configuration,
        "summary": summary,
        "cases": public_cases,
    }
```

The runner must then follow this exact sequence:

```text
parse args -> validate dataset -> require cached models -> create empty ignored run dir
-> materialize audited corpus -> build isolated local indexes
-> build structure/hybrid/rerank pipeline -> execute 20 questions
-> write raw trace only inside run dir -> build normalized public result
-> optionally export official JSON -> return 0 only when 10/10 routes, 10/10 collisions,
   and 10/10 positive Hit@5 pass
```

Do not import or construct an LLM, answerer, judge, or provider router. Write JSON with UTF-8, `sort_keys=True`, stable qid ordering, two-space indentation, and one trailing newline.

- [x] **Step 4: Run GREEN and CLI help smoke**

```powershell
uv run pytest tests/test_wage_arrears_regression.py -q
uv run python eval/run_wage_arrears_regression.py --help
```

Expected: tests pass; help lists `--dataset`, `--snapshot`, `--work-dir`, `--device`, and `--export-official` without loading a model.

- [x] **Step 5: Commit Task 2**

```powershell
git add eval/run_wage_arrears_regression.py src/rag/wage_arrears_regression.py tests/test_wage_arrears_regression.py
git commit -m "feat: add offline wage arrears regression runner"
```

---

### Task 3: Produce and verify official targeted evidence

**Files:**
- Create: `eval/official/wage_arrears_regression_v0.3.4.json`
- Modify: `release/manifest.json`
- Modify: `release/public-files.txt`
- Modify: `src/rag/release_verification.py`
- Modify: `tests/test_release_verification.py`
- Modify: `tests/test_official_artifacts.py`

**Interfaces:**
- Consumes: committed Task 1-2 code, local pinned model snapshots, audited corpus snapshot.
- Produces: public manifest section `evidence.wage_arrears_regression`, verified 138-file publication boundary, deterministic official result.

- [x] **Step 1: Write failing release-contract tests before generating evidence**

Add a test requiring the official JSON and literal invariants:

```python
assert result["schema_version"] == "1.0"
assert result["dataset"]["questions"] == 20
assert result["summary"] == {
    "positive_routes": 10,
    "collision_routes_avoided": 10,
    "positive_hit_at_5": 10,
    "positive_hit_at_1": result["summary"]["positive_hit_at_1"],
    "passed": True,
}
assert 0 <= result["summary"]["positive_hit_at_1"] <= 10
assert result["summary"]["positive_hit_at_1"] == sum(
    case["rank"] == 1 for case in result["cases"] if case["expected_expansion"]
)
assert all("question" not in case and "hits" not in case for case in result["cases"])
```

Extend the release-verifier fixture so a wrong dataset hash, case count, qid order, route count, rank, or `passed` flag raises `VerificationError`. Change the expected public file count from 133 to 138 only in the same RED commit.

- [x] **Step 2: Run the release tests and verify RED**

```powershell
uv run pytest tests/test_wage_arrears_regression.py tests/test_official_artifacts.py tests/test_release_verification.py -q
```

Expected: tests fail because the official artifact and manifest contract are absent.

- [x] **Step 3: Commit executable code state and run the real local evaluation**

First commit any verifier code needed to validate the forthcoming artifact, then run:

```powershell
uv run python eval/run_wage_arrears_regression.py --device auto --export-official
```

Expected: both pinned models resolve locally before network corpus access; all route/collision gates are 10/10; every positive Article 14 rank is 1-5; the process exits zero. The isolated local Qdrant and raw traces remain only under ignored `eval/runs/`.

- [x] **Step 4: Register the evidence and make the tests GREEN**

Add dataset path/hash, code revision, configuration, counts, and official result path to `release/manifest.json`. Add all five newly tracked paths in sorted order to `release/public-files.txt`. Implement semantic verifier checks against the dataset and official JSON rather than trusting the summary fields.

Run:

```powershell
uv run pytest tests/test_wage_arrears_regression.py tests/test_official_artifacts.py tests/test_release_verification.py -q
uv run python scripts/verify_release.py
```

Expected: all focused tests and release verification pass with 138 public files.

- [x] **Step 5: Commit Task 3**

```powershell
git add eval/official/wage_arrears_regression_v0.3.4.json release/manifest.json release/public-files.txt src/rag/release_verification.py tests/test_release_verification.py tests/test_official_artifacts.py
git commit -m "eval: record v0.3.4 wage arrears regression"
```

---

### Task 4: Document the evidence boundary and run complete verification

**Files:**
- Modify: `docs/release/WAGE_ARREARS_QUERY_EXPANSION_DESIGN.md`
- Modify: `docs/release/WAGE_ARREARS_QUERY_EXPANSION_IMPLEMENTATION_PLAN.md`
- Modify: `EVAL_REPORT.md`
- Verify: entire tracked source tree

**Interfaces:**
- Consumes: verified Task 3 official result and manifest.
- Produces: auditable report language and a clean review-ready branch.

- [x] **Step 1: Update only the post-release evidence paragraph**

Record the exact route, collision, Hit@5, and observed Hit@1 counts from the official artifact. State explicitly that the formal 40-question and v0.3.1 reliability metrics were not recomputed or replaced, no generation provider was called, and Qdrant Cloud was not accessed.

- [x] **Step 2: Run immutable-history checks**

Use `git show 3ec5ade:<path>` and SHA-256 to assert these paths are unchanged from v0.3.4:

```text
eval/dataset/eval_set.jsonl
eval/dataset/reliability_stress_v0.3.1.jsonl
eval/official/ablation_results.json
eval/official/e2e_results.json
eval/official/reliability_results.json
eval/official/reliability_trace.jsonl
eval/official/reliability_formal_trace.jsonl
```

- [x] **Step 3: Run the complete offline verification chain**

```powershell
uv run pytest -q
uv run ruff check .
uv lock --check
uv run bandit -r src scripts eval -ll
$env:PYTHONUTF8='1'
uv run pip-audit --local
Remove-Item Env:PYTHONUTF8
uv run python scripts/verify_release.py
uv build
uv run pytest tests/test_packaging.py -q
```

Expected: every command exits zero; no new security ignore is added; source/wheel artifacts exclude `.env`, local indexes, raw traces, and worktree paths.

- [x] **Step 4: Commit documentation and inspect branch**

```powershell
git add docs/release/WAGE_ARREARS_QUERY_EXPANSION_DESIGN.md docs/release/WAGE_ARREARS_QUERY_EXPANSION_IMPLEMENTATION_PLAN.md EVAL_REPORT.md
git commit -m "docs: report v0.3.4 targeted regression"
git diff --check main...HEAD
git status --short --branch
```

Expected: no whitespace errors, the branch is clean, and no merge, tag, push, deployment, or remote mutation has occurred.
