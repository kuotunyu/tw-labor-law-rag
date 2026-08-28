# BYOK Hugging Face Docker Space Runbook

本文件只適用於 `DEPLOYMENT_MODE=public_byok`。公開 Demo 的訪客必須使用自己的 Gemini 或 OpenAI API Key；部署環境不得持有擁有者的 LLM Key。

## 1. 安全與費用邊界

- 一個 Hugging Face Docker Space 對外只開 Streamlit `0.0.0.0:7860`；FastAPI 只綁 `127.0.0.1:8000`。
- Space Secrets 只允許 `QDRANT_API_KEY` 與 `SESSION_SIGNING_SECRET`。
- `QDRANT_API_KEY` 必須只讀且只可存取 `labor_laws_structure`、`labor_laws_fixed`。
- 禁止在 Space 設定 `GEMINI_API_KEY`、`OPENAI_API_KEY` 或 Qdrant write/manage Key。
- 訪客 Key 不得貼入 issue、commit、終端輸出、聊天或 `.env`；只可輸入 UI 密碼欄位。
- 本部署只允許 Qdrant Free Tier 與 Hugging Face `cpu-basic`；不得申請付費硬體、持久 storage 或額外 replica。
- LLM 費用由訪客自己的 API Key 承擔，部署端不持有或代付模型額度。
- 正式 Gemini／OpenAI cross-check 的每家 US$5 額度只適用於本機、專案專用金鑰的評估 runner；不得把擁有者的評估金鑰放進公開 Space。
- BGE-M3 與 reranker 固定在已審閱的官方 immutable revision，且 `trust_remote_code=False`。Transformers 4.x 因 FlagEmbedding 相容性暫留四組已知 advisory 例外；專案不使用其 Trainer、LightGlue、X-CLIP 或遠端自訂程式碼路徑，CI 只忽略這四組明確編號，任何新增 advisory 仍會失敗。

## 2. 本機驗證

在部署 worktree 執行：

```powershell
uv sync --locked
uv run pytest -q
uv run ruff check .
uv run bandit -r src scripts -ll
$env:PYTHONUTF8='1'
uv run pip-audit --local
Remove-Item Env:PYTHONUTF8
uv run python scripts/verify_release.py
```

Windows 專案路徑若含中文，`PYTHONUTF8=1` 可避免 `pip-audit` 的 `pip-api` 把子程序輸出以錯誤編碼解碼；它不會變更 audit 範圍或忽略項目。
`transformers>=5.5,<6` 排除 4.x 模型設定載入 RCE；本專案的窄相容層只補回 FlagEmbedding 1.4 對 XLM-R pair encoding 所需的舊 API，固定 model revision 與 `trust_remote_code=False` 仍維持不變。

任何一項失敗都停止部署；不得用真實 API 呼叫代替離線測試。

## 3. 建立 Qdrant collections

1. 建立 Standard 單節點 cluster `tw-labor-law-rag-demo`。
2. 建立短期 `tw-labor-index-writer`，只給建立兩個 collections 所需的最低 write/manage 權限。
3. 在新的 PowerShell 視窗以遮罩提示輸入 writer Key：

```powershell
$env:DATA_DIR = Read-Host 'Paste the private corpus laws directory'
$env:QDRANT_MODE = 'server'
$env:QDRANT_URL = Read-Host 'Paste the Qdrant cluster endpoint'
$env:QDRANT_API_KEY = Read-Host -MaskInput 'Paste the temporary Qdrant writer key'
uv run python scripts/build_index.py --strategy all --corpus $env:DATA_DIR
Remove-Item Env:QDRANT_API_KEY
Remove-Item Env:DATA_DIR
Remove-Item Env:QDRANT_URL
Remove-Item Env:QDRANT_MODE
```

4. 在 Qdrant 控制台確認兩個 point count 都大於 0。
5. 立即撤銷／刪除 `tw-labor-index-writer`。
6. 建立 `tw-labor-runtime-reader`，限定兩個 collections 且 read-only。

writer Key 不可保存。runtime reader 值只輸入 Hugging Face Secret。

`v0.3.1` 的程式可讀取新舊兩種 Qdrant payload。既有雲端 collections 沒有
`source_url`、`last_amended`、`effective_date` 時仍可正常問答，只是不顯示新增的法源日期／連結。
要補齊 provenance 必須另外建立短期 writer Key、用通過 audit 的相同 snapshot 重建兩個 collections、驗證後立刻撤銷；不得用 runtime reader 嘗試寫入，也不得在無人值守時擴權。

## 4. 建立 private Hugging Face Space

1. 先登入：

```powershell
uvx --from huggingface_hub hf auth login
```

2. 在 new-Space 畫面選 Docker、Private 與 `cpu-basic`（2 vCPU、16 GB RAM、50 GB 非持久磁碟，硬體時薪 US$0），只保留 1 個 replica，不購買持久 storage，也不設定付費硬體才支援的自訂休眠時間。免費層閒置後可自動睡眠並由訪客喚醒。CPU 驗收失敗時保持 private/paused，禁止自動改用任何付費硬體。
3. Space Variables：

```text
DEPLOYMENT_MODE=public_byok
QDRANT_MODE=server
QDRANT_URL=<cluster endpoint; non-secret>
QDRANT_TIMEOUT_SECONDS=60
COLLECTION_NAME=labor_laws
API_URL=http://127.0.0.1:8000
GEMINI_GENERATION_MODEL=gemini-3.5-flash-lite
OPENAI_GENERATION_MODEL=gpt-5.6-luna
BYOK_SESSION_QUERY_LIMIT=20
BYOK_SESSION_TTL_SECONDS=86400
BYOK_MAX_TRACKED_SESSIONS=1000
BYOK_MAX_CONCURRENCY=2
BYOK_REQUEST_TIMEOUT_SECONDS=60
BYOK_MAX_QUESTION_CHARS=2000
```

4. Space Secrets：

```text
QDRANT_API_KEY=<collection-scoped read-only key>
SESSION_SIGNING_SECRET=<cryptographically random value>
```

可在本機產生 signing secret 並直接上傳；不得列印或貼入聊天：

```powershell
$env:SESSION_SIGNING_SECRET = uv run python -c "import secrets; print(secrets.token_urlsafe(48))"
```

上傳後立即從本機環境移除：

```powershell
Remove-Item Env:SESSION_SIGNING_SECRET
Remove-Item Env:QDRANT_API_KEY
Remove-Item Env:QDRANT_URL
```

最後再次確認 Secrets 清單沒有 `GEMINI_API_KEY`、`OPENAI_API_KEY`。

## 5. 推送到 private Space

從已驗證的 deployment branch 推送，不新增或修改 GitHub remote：

```powershell
$hfUser = uv run --with huggingface_hub python -c "from huggingface_hub import HfApi; print(HfApi().whoami()['name'])"
git push "https://huggingface.co/spaces/$hfUser/tw-labor-law-rag-demo" HEAD:main
```

GitHub `main`、release tags 與其他 worktrees 在此步保持不變。

## 6. Private acceptance

依序留下不含秘密值的證據：

1. Space 啟動完成，Gemini/OpenAI 兩個選項正確。
2. 未輸入 Key 時不能送出問題。
3. 使用專用、低額度 Gemini Key 完成一題，清除 Key。
4. 使用專用、低額度 OpenAI Key 完成一題，清除 Key。
5. requested/actual provider 與 model 相同，沒有 fallback。
6. 第 21 次請求被 session quota 拒絕；同時超過 2 題時立即回 429。
7. Space logs 找不到測試 Key 全值或任一事先記錄的 8 字元片段，也沒有問題、答案、provider body 或 Qdrant URL。
8. runtime Qdrant Key 可讀，create/upsert/delete 操作遭拒。
9. image/repository 不含 private `data/raw/` 或 `storage/bm25_*.json`。
10. 若 collections 尚未重建，答案仍可顯示法規／條號且應明確記錄為 legacy payload；重建後才驗收法源 URL 與修正／生效日期。

記錄 branch commit、完整測試數、release verifier 結果、Space revision、硬體 tier 與兩個 collection point counts。Key、endpoint、session token 不列入證據。

## 7. GO / NO-GO 與 rollback

- 任一驗收失敗：Space 保持 private，回到對應 TDD 任務修復後重跑完整驗證。
- 全部通過：再取得擁有者明確同意，才把 Space visibility 改為 public。
- 若公開後出現費用、濫用或隱私異常：立即將 Space 改為 private 或 pause，旋轉 Qdrant read-only Key，檢查 logs；因 Space 沒有 owner LLM Key，不需要為網站事件旋轉擁有者的 Gemini/OpenAI Key。
- 回復 deployment branch 或 Space revision 時，不改寫 GitHub `main`、正式 tags 或 private archive history。
