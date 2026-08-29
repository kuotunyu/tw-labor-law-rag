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
