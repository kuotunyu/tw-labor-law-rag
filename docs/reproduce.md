# 重現、測試與發佈檢查

[English](reproduce.en.md) ｜ [繁體中文](reproduce.md)

[README](../README.md) 只留最短的啟動方式；完整的執行、測試與檢查說明原樣移到這裡。

## 在本機跑起來

需求:Python 3.11、[uv](https://docs.astral.sh/uv/)。有 NVIDIA GPU 可大幅加速 embedding/rerank,純 CPU 也能跑(較慢)。

```bash
# 1. 安裝依賴
uv sync

# 2. 設定環境變數（公開 API/UI 至少在伺服器端填 Gemini / OpenAI 一組 key）
cp .env.example .env
# 編輯 .env，填入對應 API key；不要將 .env 提交到 Git

# 3. 下載語料(全國法規資料庫官方開放資料,約 30MB,首次執行)
uv run python scripts/download_corpus.py

# 4. 建索引(向量 + BM25,兩種 chunking 策略各一份;有 GPU 約 1 分鐘)
uv run python scripts/build_index.py

# 5. 命令列問答(開發用,免啟動伺服器)
uv run python scripts/ask.py "加班費怎麼算?"

# 6. 或啟動 API + 前端
uv run python scripts/run_api.py &          # http://localhost:8000/docs
uv run streamlit run ui/app.py              # http://localhost:8501
```

### 用 Docker(Qdrant server mode)

```bash
docker compose up -d qdrant
# 將 .env 的 QDRANT_MODE 改為 server,QDRANT_URL 保持 http://localhost:6333
uv run python scripts/build_index.py --strategy all      # 對 Qdrant 服務建兩種索引
docker compose up --build api ui
```

## 跑測試與評估

測試與 release verifier 不依賴 GPU、模型權重、Qdrant 或真實 LLM API;heavy components 皆延遲載入,unit tests 使用純邏輯、fixture、cache 或明確的 test double。[GitHub Actions](../.github/workflows/ci.yml) 會在 `main` push、`v*` tag push 與所有 pull request 執行。

```bash
uv run python scripts/verify_release.py          # committed evidence 離線重算與公開邊界稽核
uv run ruff check .                             # locked lint gate
uv run pytest                                    # 單元、正式產物、privacy 與 package 測試
uv build                                         # sdist + wheel;驗證 runtime dictionary 有打包
```

重新執行 `eval/ablation.py` 需要既有索引與本機模型;`eval/run_e2e_eval.py` 還需要 provider,不屬於公開離線 reviewer path。可公開、去識別化的正式指標與逐題 trace 已收錄在 [`eval/official/`](../eval/official/README.md);`eval/runs/` 保留原始本機執行結果,不進版控。完整 clean reviewer 步驟見 [REVIEWER_GUIDE.md](release/REVIEWER_GUIDE.md)。

## `verify_release.py` 檢查什麼

`uv run python scripts/verify_release.py` 不載入模型、不呼叫 provider、不啟動 Qdrant/Docker，會核對 40 題正式集、60 題壓力集、10 題 portfolio regression、8×40 ablation grid、Hit@5/MRR、0.03 threshold sweep、15 部／884 條 law/source 與逐條文 content-free snapshots、設定一致性、OGDL samples、official trace schema、provider complete contract、完整 publication inventory、secret/privacy scan、人工審閱 binary hashes 與 GitHub Action pins。Git 歷史稽核涵蓋 heads、tags、remotes 的所有可公開 commits；GitHub Actions 暫時產生、不可發布的 `refs/remotes/pull/*` 合成 merge refs 除外，本機 `refs/archive/*` recovery evidence 也會保留在 publication graph 之外。0.03 reranker threshold 不是通用 answerability classifier；壓力集已量測到 1/40 直接誤拒，因此只保留現值而不宣稱問題已消失。

哪些數字可以從公開檔案重算、哪些不行：retrieval、answerability 與 refusal 算術可從 committed privacy-reduced traces 完整離線重算。實際作答的 29 題平均 faithfulness **4.90/5**、relevancy **5.00/5** 則屬 **archived provider evidence**:repository 可離線重新聚合已提交的 judge 數字,但不含完整生成答案、judge 理由或 provider response,因此不能從公開 evidence 重新產生或獨立複判這些評分。完整方法與限制見 [EVAL_REPORT.md](../EVAL_REPORT.md),去識別化逐題 trace 見 [`eval/official/`](../eval/official/README.md),claim 到 evidence 的映射見 [claim matrix](release/CLAIM_MATRIX.md)。

## 公開範圍

這是 `v0.3.5` source-only runtime and deployment release。正式模型品質指標沿用未變更的 `v0.1.0` formal evidence baseline；本版新增 reviewer-first 私有 BYOK 介面、10 題離線 portfolio regression 與 15 部／884 條 content-free 逐條文 freshness baseline，但不把示範回歸寫成新的模型品質基準。v0.3.2 Gemini／OpenAI safety cross-check 仍是 archived provider evidence，兩家各五筆請求均在 US$5 硬上限內：Gemini refusal accuracy `0.8`、citation success `1.0`、estimated cost `US$0.0022620`；OpenAI refusal accuracy `1.0`、citation success `1.0`、estimated cost `US$0.0026414`。公開 trace 嚴格不含 question/answer text、provider payload 或憑證；此 cross-check 不取代正式模型品質基準。它是 evidence-backed software portfolio artifact，不是法律意見，也不是 production legal service。完整 corpus、模型權重、私有索引與 provider raw artifacts 仍不在本次 source release 範圍。
