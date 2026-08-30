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
`transformers>=5.5,<6` 排除 4.x 模型設定載入 RCE；模型 revision 必須是完整 40 位 commit SHA，並先解析為不可變的 Hub snapshot 本機路徑。本專案的窄相容層只補回 FlagEmbedding 1.4 對 XLM-R pair encoding 所需的舊 API，`trust_remote_code=False` 仍維持不變。

任何一項失敗都停止部署；不得用真實 API 呼叫代替離線測試。

## 3. 人工建立或更新 Qdrant collections

本專案維持 **Qdrant Free** 與 Hugging Face **CPU Basic**。索引更新只能在操作人員在場時手動執行，不建立排程、cron、監控自動修復或無人值守的 writer 工作。更新工具採 blue-green：建立新的 candidate pair，正式 pair 保持可讀；不得對正式 collection 執行 build_index.py。

### 3.1 本機 audit 與 dry-run

在已驗證的 worktree 執行。`YYYYMMDD_HASH` 是操作人員依日期與已提交 snapshot 短雜湊選定的非秘密名稱；兩次出現必須完全相同，例如 `labor_laws_20260830_deadbeef`。

```powershell
uv run python scripts/audit_corpus.py
$auditExit = $LASTEXITCODE
if ($auditExit -eq 2) { throw '官方來源無法取得或資料無效；停止更新。' }
if ($auditExit -eq 1) { throw '偵測到法規或條文差異；先審閱輸出的法規、欄位與條號，另開 release 任務。' }
if ($auditExit -ne 0) { throw "未知 audit exit code: $auditExit" }

uv run python scripts/download_corpus.py --force-download
$candidateBase = 'labor_laws_YYYYMMDD_HASH'
uv run python scripts/rebuild_qdrant_blue_green.py --candidate-base $candidateBase
```

`audit_corpus.py` 直接從法務部官方來源比對已提交的 law/source 與逐條文 content-free baseline。exit `0` 才能繼續；exit `1` 必須先人工審閱輸出中具名的法規、變動欄位與條號；exit `2` 代表來源無法取得或資料無效，立即停止。不得在差異尚未審閱前建立 writer key，也不得用 `--write` 自動接受差異。專案沒有 audit 排程、cron、heartbeat 或自動更新。

後續 dry-run 只讀本機 official archives、15 部 normalized laws、`release/corpus_snapshot.json` 與逐條文 baseline；語料資料夾只允許第一層 law JSON 與選用的 `manifest.json`，任何子目錄、額外 JSON、Markdown、文字或 PDF 都會停止。它不讀 writer Key、不檢查或載入模型、不建立 Qdrant client、不寫 receipt。任何 snapshot、來源雜湊、法規 metadata、條文數或內容雜湊差異都必須停止，另開 release 任務審閱；不得用參數略過或自動改寫 committed snapshot。

### 3.2 有人值守的 candidate build

1. 只有在 3.1 audit 為 current 且 dry-run 通過後，才在 Qdrant Cloud 為這一次維護建立 temporary writer key；它只在本 PowerShell 工作階段存在，不放進 `.env`、Hugging Face、聊天、issue 或 commit。
2. 確認 BGE-M3 與 reranker 的固定 revision 已在本機 cache。維護工具只接受既有 cache，不會自動改用付費 GPU 或下載未審閱 revision。
3. 使用遮罩輸入 endpoint 與 key，重複 candidate 名稱後才允許執行：

```powershell
$candidateBase = 'labor_laws_YYYYMMDD_HASH'
$env:QDRANT_URL = Read-Host 'Paste the Qdrant cluster endpoint'
$env:QDRANT_WRITER_API_KEY = Read-Host -MaskInput 'Paste the temporary writer key'
try {
  uv run python scripts/rebuild_qdrant_blue_green.py --execute `
    --candidate-base $candidateBase `
    --confirm-candidate-base $candidateBase
} finally {
  Remove-Item Env:QDRANT_WRITER_API_KEY -ErrorAction SilentlyContinue
  Remove-Item Env:QDRANT_URL -ErrorAction SilentlyContinue
}
```

工具會以已稽核且只讀取一次的語料，在第一次 Qdrant write 前完成固定 400/80 設定的兩種 chunking 與 embedding，並要求 `fixed=481`、`structure=884`、向量維度 `1024`，然後只建立 `$candidateBase` 對應的 collections。實際模型解析強制 `local_files_only`。若同名 candidate 已存在、point count 不符、payload provenance 不完整或任何步驟失敗，正式 collections 不受影響，且工具不自動刪除或覆寫 partial candidate。

成功時才會在 ignored 的 `eval/runs/qdrant-maintenance/$candidateBase.json` 寫入不含 endpoint、key、法規全文或本機絕對路徑的 receipt；若同名 receipt 已存在會停止，不會覆寫歷史。無論成功或失敗，立即回 Qdrant Cloud **撤銷 temporary writer key**；partial candidate 的檢查或刪除屬另一個具破壞性的人工任務，本命令未獲授權執行。

### 3.3 Private cutover、驗收與 rollback

1. 先記錄 **舊 COLLECTION_NAME** 與舊 runtime reader key 的識別名稱，不記錄 key 值。
2. 建立一把只讀 transition key，暫時只允許讀取舊 pair 與 candidate pair；先把 Hugging Face Secret `QDRANT_API_KEY` 換成 transition key，保持舊 `COLLECTION_NAME` 重啟並確認健康。
3. 將 Space 保持 private，把 Variable `COLLECTION_NAME` 改成 `$candidateBase` 後 Restart。
4. 必須依序看到 Space `RUNNING`、domain `READY`、`/health` HTTP 200、啟動 logs 無敏感資料，再用訪客自己的低額度 API Key 驗收 Gemini 與 OpenAI 各一題；引用來源、修正日期與可用時的生效日期都要正確。
5. 驗收成功後建立只可讀 candidate pair 的新 `tw-labor-runtime-reader`，更新 Space Secret 並再次重啟驗收；最後撤銷舊 reader、temporary writer 與 transition key，並刪除本機 Downloads 中這次下載的所有 key 檔。
6. 若任一驗收失敗，立即 rollback：把 `COLLECTION_NAME` 恢復成舊值，使用仍可讀舊 pair 的 transition key Restart，確認健康後再診斷 candidate。不得刪除舊 collections。

舊 collections 的刪除是獨立、具破壞性的容量管理操作；不在 build 或 cutover 命令範圍內。Free Tier 空間不足時先停止更新並人工評估，不自動升級付費方案，也不以刪除正式 pair 換取空間。

`v0.3.1` 之後的程式可讀取新舊兩種 Qdrant payload。舊 collections 沒有 `source_url`、`last_amended`、`effective_date` 時仍可正常問答，只是不顯示新增的法源日期／連結。

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
