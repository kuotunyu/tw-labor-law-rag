# v0.3.5 Portfolio and UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give hiring reviewers a three-minute evidence-first project tour and give invited users a clear, secure three-step BYOK question flow.

**Architecture:** Keep the Streamlit/FastAPI boundary and existing request path. Add a small pure presentation module for curated examples and snapshot copy, then make `ui/app.py` consume it through progressive disclosure. README and reviewer documents point from concise portfolio claims to the existing evidence artifacts.

**Tech Stack:** Python 3.11, Streamlit 1.58, FastAPI, httpx, pytest, Streamlit AppTest, Markdown, Mermaid, Ruff, uv.

## Global Constraints

- Keep the Hugging Face Space private and on `cpu-basic` with one replica and no paid persistent storage.
- Do not add Gemini, OpenAI, Qdrant, or session secret values to source, tests, screenshots, logs, or history.
- Public BYOK keeps provider fallback disabled and never uses an owner LLM key.
- Do not change embedding, reranker, RRF, threshold `0.03`, Qdrant, or the 15-law target corpus.
- The UI must say that an entered key is not validated until a provider request succeeds.
- Expert retrieval controls remain available but are collapsed by default.
- Example questions must use the normal `/query` request path and must not have hard-coded answers.
- No provider call, paid service, schedule, or public-visibility change occurs in automated tests.
- Every new tracked path must be added to Python-sorted `release/public-files.txt`.
- Use Traditional Chinese for the primary UI and README, with an evidence-calibrated English README.

---

## File map

- `ui/content.py`: immutable public snapshot facts, BYOK copy, visual tokens, and curated example-question records.
- `ui/app.py`: Streamlit layout, progressive disclosure, example selection, answer/source rendering, and responsive styling.
- `tests/test_ui_content.py`: pure content, snapshot-alignment, and grouping tests.
- `tests/test_ui_byok_app.py`: end-to-end Streamlit state, privacy, examples, controls, and responsive-copy tests.
- `README.md`: concise Traditional-Chinese reviewer path and truthful private-demo status.
- `README.en.md`: equivalent calibrated English reviewer path.
- `docs/release/V035_REVIEWER_TOUR.md`: three-minute technical-review path.
- `docs/release/V035_INTERVIEW_DEMO.md`: deterministic three-to-five-minute interview demonstration script.
- `scripts/run_ui_fixture_api.py`: local-only deterministic model/session/query fixture used for screenshots and browser acceptance.
- `tests/test_ui_fixture_api.py`: fixture response contract and secret-boundary tests.
- `docs/screenshot-demo.png`: current v0.3.5 UI screenshot without a key or secret-bearing browser state.
- `release/manifest.json`: updated reviewed-binary hash for the screenshot.
- `release/public-files.txt`: exact public inventory additions.

---

### Task 1: Pure portfolio content and example contracts

**Files:**
- Create: `ui/content.py`
- Create: `tests/test_ui_content.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: committed `release/corpus_snapshot.json` in tests only.
- Produces: `KnowledgeBaseSummary`, `ExampleQuestion`, `KNOWLEDGE_BASE`, `EXAMPLE_QUESTIONS`, `BYOK_PRIVACY_POINTS`, and `examples_by_category()`.

- [ ] **Step 1: Write the failing content-contract test**

```python
import json
from pathlib import Path

from ui.content import (
    BYOK_PRIVACY_POINTS,
    EXAMPLE_QUESTIONS,
    KNOWLEDGE_BASE,
    examples_by_category,
)


def test_public_snapshot_and_examples_match_the_release_contract():
    root = Path(__file__).parents[1]
    snapshot = json.loads((root / "release/corpus_snapshot.json").read_text(encoding="utf-8"))
    assert KNOWLEDGE_BASE.snapshot_date == snapshot["snapshot_date"]
    assert KNOWLEDGE_BASE.laws == snapshot["law_count"] == 15
    assert KNOWLEDGE_BASE.articles == snapshot["article_count"] == 884
    assert len(EXAMPLE_QUESTIONS) == 5
    assert set(examples_by_category()) == {"工時", "請假", "離職與欠薪", "資遣費", "安全拒答"}
    assert len({item.id for item in EXAMPLE_QUESTIONS}) == len(EXAMPLE_QUESTIONS)
    assert all(item.question.strip() for item in EXAMPLE_QUESTIONS)
    assert BYOK_PRIVACY_POINTS == (
        "只保留在目前瀏覽器工作階段",
        "不寫入檔案或聊天紀錄",
        "模型費用由 API Key 持有人承擔",
    )
```

- [ ] **Step 2: Run the focused test and confirm the missing module**

Run: `uv run pytest tests/test_ui_content.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'ui.content'`.

- [ ] **Step 3: Implement the immutable content module**

```python
from dataclasses import dataclass


@dataclass(frozen=True)
class KnowledgeBaseSummary:
    snapshot_date: str
    laws: int
    articles: int


@dataclass(frozen=True)
class ExampleQuestion:
    id: str
    category: str
    title: str
    question: str


KNOWLEDGE_BASE = KnowledgeBaseSummary("2026-08-29", 15, 884)
BYOK_PRIVACY_POINTS = (
    "只保留在目前瀏覽器工作階段",
    "不寫入檔案或聊天紀錄",
    "模型費用由 API Key 持有人承擔",
)
EXAMPLE_QUESTIONS = (
    ExampleQuestion("hours", "工時", "每日與每週工時", "勞工每天和每週的正常工作時間上限是多少？"),
    ExampleQuestion("sick-leave", "請假", "普通傷病假", "一年最多可以請幾天病假？請病假薪水怎麼算？"),
    ExampleQuestion("wage-arrears", "離職與欠薪", "欠薪立即離職", "公司一直拖欠薪水，我可以不經預告直接離職嗎？這樣還能拿到資遣費嗎？"),
    ExampleQuestion("severance", "資遣費", "新舊制比較", "適用勞退新制的勞工被資遣時，資遣費怎麼計算？和舊制有什麼不同？"),
    ExampleQuestion("refusal", "安全拒答", "知識庫外問題", "著作權的保護期間是幾年？"),
)


def examples_by_category() -> dict[str, tuple[ExampleQuestion, ...]]:
    return {item.category: (item,) for item in EXAMPLE_QUESTIONS}
```

- [ ] **Step 4: Run content and public-inventory checks**

Run:

```powershell
uv run pytest tests/test_ui_content.py -q -p no:cacheprovider
uv run ruff check ui/content.py tests/test_ui_content.py
uv run python -c "from pathlib import Path; p=[x for x in Path('release/public-files.txt').read_text(encoding='utf-8').splitlines() if x]; assert p==sorted(p); assert len(p)==len(set(p))"
```

Expected: all commands exit 0.

- [ ] **Step 5: Commit Task 1**

```powershell
git add ui/content.py tests/test_ui_content.py release/public-files.txt
git commit -m "feat: add portfolio presentation contracts"
```

---

### Task 2: Progressive-disclosure BYOK interface

**Files:**
- Modify: `ui/app.py:1-313`
- Modify: `tests/test_ui_byok_app.py:1-151`

**Interfaces:**
- Consumes: Task 1 `KNOWLEDGE_BASE`, `EXAMPLE_QUESTIONS`, `BYOK_PRIVACY_POINTS`, and `examples_by_category()`.
- Produces: the same `submit_query(api_url, payload, api_key=..., session_token=...)` call; a selected example becomes the `question` field without bypassing the API.

- [ ] **Step 1: Extend AppTest with the desired hierarchy and copy**

```python
assert any("知識庫快照：2026-08-29" in item.value for item in app.caption)
assert any(expander.label == "進階比較設定" for expander in app.sidebar.expander)
assert any(element.value == "三步開始問答" for element in app.subheader)
assert any(button.label == "每日與每週工時" for button in app.button)

app.text_input[0].set_value("gemini-visitor-secret-key").run()
rendered = "\n".join(str(item.value) for item in app.info + app.success + app.caption)
assert "已填入，但尚未向模型供應商驗證" in rendered
```

- [ ] **Step 2: Add an example-path AppTest assertion**

```python
example_button = next(button for button in app.button if button.label == "每日與每週工時")
example_button.click().run()
query_request = next(item for item in requests if item["path"] == "/query")
assert query_request["payload"]["question"] == "勞工每天和每週的正常工作時間上限是多少？"
assert query_request["provider_key"] == "gemini-visitor-secret-key"
```

- [ ] **Step 3: Run AppTest and confirm hierarchy assertions fail**

Run: `uv run pytest tests/test_ui_byok_app.py -q -p no:cacheprovider`

Expected: FAIL because the current app has no advanced expander, example buttons, or “尚未驗證” copy.

- [ ] **Step 4: Implement the restrained legal-dossier presentation tokens**

Add one `st.markdown(..., unsafe_allow_html=True)` stylesheet immediately after `st.set_page_config` with these exact tokens:

```css
:root {
  --law-ink: #20242c;
  --law-muted: #667085;
  --law-paper: #fbfaf7;
  --law-line: #d8d2c5;
  --law-accent: #a43b32;
  --law-accent-soft: #f5e9e6;
}
.stApp { background: var(--law-paper); color: var(--law-ink); }
[data-testid="stHeader"] { background: rgba(251, 250, 247, .92); }
[data-testid="stSidebar"] { border-right: 1px solid var(--law-line); }
.block-container { max-width: 980px; padding-top: 2.25rem; }
@media (max-width: 700px) {
  .block-container { padding: 1rem .85rem 5rem; }
}
```

Do not hide Streamlit controls, focus outlines, labels, or status messages.

- [ ] **Step 5: Move expert controls into the collapsed sidebar expander**

```python
with st.sidebar:
    st.subheader("檢索設定")
    with st.expander("進階比較設定", expanded=False):
        strategy = st.selectbox("Chunking 策略", list(STRATEGY_LABELS), format_func=STRATEGY_LABELS.get)
        mode = st.selectbox("檢索模式", list(MODE_LABELS), format_func=MODE_LABELS.get)
        use_reranker = st.checkbox("啟用 Reranker (bge-reranker-v2-m3)", value=True)
        st.caption("這些選項用於比較消融設定；一般問答可維持預設值。")
```

- [ ] **Step 6: Implement three-step onboarding and truthful key state**

Use `st.subheader("三步開始問答")`, numbered captions for provider/key/question, and this exact filled state:

```python
if visitor_key.strip():
    st.info("API Key 已填入，但尚未向模型供應商驗證；第一次成功送出後才代表可用。")
else:
    st.info(f"請輸入 {provider_name} API Key，再選擇範例或提出問題。")
st.caption("　•　".join(BYOK_PRIVACY_POINTS))
```

- [ ] **Step 7: Add example buttons that feed the production request path**

```python
pending_question = None
st.subheader("試一個代表性問題")
for item in EXAMPLE_QUESTIONS:
    if st.button(item.title, key=f"example-{item.id}", disabled=not byok_ready):
        pending_question = item.question

typed_question = st.chat_input(
    "輸入你的勞動法規問題...",
    disabled=selected_provider is None or not byok_ready,
)
question = typed_question or pending_question
```

Keep the existing single `if question:` block and `submit_query` call unchanged below this assignment.

- [ ] **Step 8: Run UI, API-client, and privacy tests**

Run:

```powershell
uv run pytest tests/test_ui_content.py tests/test_ui_byok_app.py tests/test_ui_api_client.py tests/test_byok_policy.py -q -p no:cacheprovider
uv run ruff check ui tests/test_ui_content.py tests/test_ui_byok_app.py
```

Expected: all commands exit 0; AppTest confirms the visitor key is absent from payload, rendered copy, and retained history.

- [ ] **Step 9: Commit Task 2**

```powershell
git add ui/app.py tests/test_ui_byok_app.py
git commit -m "feat: clarify secure demo onboarding"
```

---

### Task 3: Reviewer tour and truthful repository opening

**Files:**
- Create: `docs/release/V035_REVIEWER_TOUR.md`
- Create: `docs/release/V035_INTERVIEW_DEMO.md`
- Modify: `README.md:1-80`
- Modify: `README.en.md:1-80`
- Modify: `tests/test_release_verification.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: existing formal/reliability/provider evidence and Task 1 example identifiers.
- Produces: stable anchors for “3-minute tour,” “reproduce evidence,” “architecture,” “limitations,” and “private demo.”

- [ ] **Step 1: Write failing README state-alignment assertions**

```python
def test_readmes_present_a_truthful_reviewer_path():
    zh = (PROJECT_ROOT / "README.md").read_text(encoding="utf-8")
    en = (PROJECT_ROOT / "README.en.md").read_text(encoding="utf-8")
    for text in (zh, en):
        assert "V035_REVIEWER_TOUR.md" in text
        assert "V035_INTERVIEW_DEMO.md" in text
        assert "2026-08-29" in text
    assert "private Space" in zh
    assert "private Space" in en
    assert "公開 BYOK Docker Space（已上線）" not in zh
    assert "Public BYOK Docker Space (Live)" not in en
```

- [ ] **Step 2: Run the assertion and confirm current public-demo wording fails**

Run: `uv run pytest tests/test_release_verification.py -q -p no:cacheprovider`

Expected: FAIL on missing reviewer-tour links and stale public-demo wording.

- [ ] **Step 3: Write the three-minute reviewer tour**

The document must contain these sections with direct links to existing evidence:

```markdown
# v0.3.5 Three-Minute Reviewer Tour
## 0:00–0:30 — Problem and user boundary
## 0:30–1:15 — Why Hybrid Search
## 1:15–2:00 — What the measurements prove
## 2:00–2:30 — Refusal, citations, and known limits
## 2:30–3:00 — BYOK security and free infrastructure
## Reproduce without provider keys
```

The measurement section must distinguish offline-recomputable retrieval/refusal evidence from archived provider judgments.

- [ ] **Step 4: Write the interview demonstration script**

Use these exact four demonstrations in order:

1. `hours` — daily/weekly normal hours and Article 30 citation.
2. `severance` — Retirement Pension Act Article 12 plus Labor Standards Act Article 17.
3. `wage-arrears` — the v0.3.4 Article 14 targeted route.
4. `refusal` — copyright duration; threshold refusal and no LLM call when the deterministic gate rejects it.

For each demonstration include expected sources, the UI element to open, the engineering point to explain, and the statement that generated wording may vary while retrieval/policy expectations are fixed.

- [ ] **Step 5: Rewrite the README openings**

Place this information before release chronology in both languages:

- one-sentence problem/solution;
- badges or a compact table for `15 laws`, `884 articles`, `Hit@5 0.967`, and `MRR@10 0.906`;
- current snapshot date `2026-08-29`;
- links to reviewer tour, evidence reproduction, architecture, limitations, and interview demo;
- private-demo wording: invited/owner-controlled access only;
- BYOK wording: visitors supply their own provider key and the owner does not fund their model tokens.

- [ ] **Step 6: Run documentation and publication tests**

Run:

```powershell
uv run pytest tests/test_release_verification.py tests/test_official_artifacts.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: all commands exit 0 after public-file count and new-path assertions are updated to the exact tracked set.

- [ ] **Step 7: Commit Task 3**

```powershell
git add README.md README.en.md docs/release/V035_REVIEWER_TOUR.md docs/release/V035_INTERVIEW_DEMO.md tests/test_release_verification.py release/public-files.txt
git commit -m "docs: add v0.3.5 reviewer journey"
```

---

### Task 4: Current UI screenshot and visual acceptance

**Files:**
- Create: `scripts/run_ui_fixture_api.py`
- Create: `tests/test_ui_fixture_api.py`
- Modify: `docs/screenshot-demo.png`
- Modify: `release/manifest.json`
- Modify: `tests/test_release_verification.py`
- Modify: `release/public-files.txt`

**Interfaces:**
- Consumes: Task 2 public API-client schema and Task 3 screenshot placement.
- Produces: one reviewed PNG and its exact SHA-256 in `publication.reviewed_binaries` and `publication.history.reviewed_binary_sha256`.

- [ ] **Step 1: Write the failing fixture-response contract test**

```python
from scripts.run_ui_fixture_api import response_for


def test_fixture_returns_public_byok_contract_without_echoing_secrets():
    models = response_for("GET", "/models", {}, {})
    assert models["requires_api_key"] is True
    assert models["default_provider"] == "gemini"
    assert {item["provider"] for item in models["providers"]} == {"gemini", "openai"}

    headers = {"X-Provider-Api-Key": "test-only-secret"}
    query = response_for(
        "POST",
        "/query",
        {"provider": "gemini", "strategy": "structure", "mode": "hybrid", "use_reranker": True},
        headers,
    )
    assert query["answer"] == "一般情況下，勞工每日正常工作時間不得超過 8 小時。[1]"
    assert query["sources"][0]["article"] == "第 30 條"
    assert "test-only-secret" not in repr(query)
```

- [ ] **Step 2: Run the fixture test and confirm the missing module**

Run: `uv run pytest tests/test_ui_fixture_api.py -q -p no:cacheprovider`

Expected: collection fails with `ModuleNotFoundError: No module named 'scripts.run_ui_fixture_api'`.

- [ ] **Step 3: Implement the local-only fixture server**

Create a `ThreadingHTTPServer` executable that exposes only `GET /models`, `POST /session`, and `POST /query`. Put the pure `response_for(method, path, payload, headers)` dispatch ahead of the handler so the test can call it without opening a port. Require any non-empty `X-Provider-Api-Key` for `/query`, return status `401` when absent, never echo request headers, and return one Article 30 source from `law.moj.gov.tw`. Bind to `127.0.0.1` and accept `--port` with default `8765`.

- [ ] **Step 4: Run the fixture contract and static checks**

```powershell
uv run pytest tests/test_ui_fixture_api.py -q -p no:cacheprovider
uv run ruff check scripts/run_ui_fixture_api.py tests/test_ui_fixture_api.py
```

Expected: both commands exit 0.

- [ ] **Step 5: Start the deterministic local fixture API and Streamlit app**

In terminal 1, run:

```powershell
uv run python scripts/run_ui_fixture_api.py --port 8765
```

In terminal 2, run:

```powershell
$env:API_URL='http://127.0.0.1:8765'
uv run streamlit run ui/app.py --server.headless true --server.port 8501
```

Expected: Streamlit is reachable at `http://127.0.0.1:8501`, model discovery lists Gemini/OpenAI, and no real provider or Qdrant request occurs.

- [ ] **Step 6: Capture desktop and narrow-viewport evidence**

Use the browser testing tool at widths `1440×1000` and `390×844`. Verify keyboard focus, no horizontal scrolling, visible private/BYOK copy, collapsed advanced settings, disabled examples before key entry, and enabled examples after entering the fixture key `test-only-not-a-provider-key`.

Save the desktop image as `docs/screenshot-demo.png`. Keep the narrow image in the ignored local run-artifact directory, not the repository.

- [ ] **Step 7: Record and test the reviewed binary hash**

Run:

```powershell
$hash=(Get-FileHash docs/screenshot-demo.png -Algorithm SHA256).Hash.ToLower()
Write-Output $hash
```

Update both reviewed-binary locations in `release/manifest.json` and the exact expected hash in `tests/test_release_verification.py`. Do not add any other binary.

- [ ] **Step 8: Run visual-boundary and UI tests**

Run:

```powershell
uv run pytest tests/test_ui_content.py tests/test_ui_byok_app.py tests/test_ui_fixture_api.py tests/test_release_verification.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: all commands exit 0 and the release verifier reports one reviewed binary with the new hash.

- [ ] **Step 9: Commit Task 4**

```powershell
git add scripts/run_ui_fixture_api.py tests/test_ui_fixture_api.py docs/screenshot-demo.png release/manifest.json tests/test_release_verification.py release/public-files.txt
git commit -m "docs: refresh v0.3.5 interface evidence"
```

---

### Task 5: Portfolio/UI checkpoint

**Files:**
- No product files expected; fix only failures attributable to Tasks 1–4.

**Interfaces:**
- Consumes: all prior tasks in this plan.
- Produces: one independently reviewable portfolio/UI checkpoint.

- [ ] **Step 1: Run the focused checkpoint**

```powershell
uv lock --check
uv run ruff check .
uv run pytest tests/test_ui_content.py tests/test_ui_byok_app.py tests/test_ui_api_client.py tests/test_ui_fixture_api.py tests/test_release_verification.py tests/test_official_artifacts.py -q -p no:cacheprovider
uv run python scripts/verify_release.py
git diff --check
```

Expected: every command exits 0.

- [ ] **Step 2: Review the checkpoint diff**

Run: `git log --oneline e3940b9..HEAD; git diff --stat e3940b9..HEAD; git status --short`

Expected: only planned UI, documentation, tests, screenshot, manifest, and inventory paths appear; the worktree is clean.
