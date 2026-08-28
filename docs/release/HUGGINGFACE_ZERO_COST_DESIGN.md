# Hugging Face 零新增費用部署設計

日期：2026-08-28

## 目標與成功條件

將繁體中文勞動法規 RAG 作品部署至既有的 Hugging Face PRO 個人帳號，但不產生 PRO 月費之外的任何 Hugging Face 硬體費用。公開訪客必須使用自己的 Gemini 或 OpenAI API Key；部署不得持有或消耗站長的 LLM Key。

成功條件如下：

- Space 的 requested 與 current hardware 均為 `cpu-basic`，不得要求 T4、CPU Upgrade、持久磁碟或額外 replicas。
- Hugging Face 硬體時薪為 US$0；不設定只能在付費硬體使用的自訂休眠功能。
- Space 只保存 collection-scoped、read-only 的 Qdrant runtime key 與 session signing secret。
- Space 不保存 `GEMINI_API_KEY` 或 `OPENAI_API_KEY`；每位訪客自行提供 provider key。
- Private 驗收通過後才可考慮公開；若 CPU 資源不足，保持 private/paused，不自動升級付費硬體。

## 方案選擇

採用 Hugging Face Docker Space 的免費 `CPU Basic`（2 vCPU、16 GB RAM、50 GB 非持久磁碟）。沿用既有 Streamlit、loopback FastAPI、Qdrant Cloud 與 BYOK 架構，不改成 GitHub Pages，也不為 ZeroGPU 重寫 Gradio 應用。

未採用的方案：

- T4 small：會按分鐘產生額外費用，不符合零新增費用限制。
- CPU Upgrade：雖較便宜但仍按小時計費。
- GitHub Pages：只能承載靜態前端，不能執行現有 Python RAG 後端，也不能安全保存 Qdrant runtime secret。
- ZeroGPU／Community GPU Grant：需要重構或等待審核，資源可用性不確定；可作未來改善，不是本次上線依賴。

## 架構與資料流

1. 訪客開啟 Space；免費 CPU Space 若在睡眠狀態，由 Hugging Face 自動喚醒。
2. Streamlit 收到訪客問題及其 Gemini/OpenAI API Key；Key 僅保存在目前工作階段。
3. 同容器 FastAPI 以 CPU 執行 BGE-M3 query embedding、Qdrant read-only 查詢、BM25 hybrid retrieval 與 reranking。
4. FastAPI 使用該次請求的訪客 Key 呼叫選定 provider，不使用站長 Key，也不跨 provider fallback。
5. 回應只包含答案、引用與非敏感診斷欄位；不得記錄訪客 Key、問題全文、provider body 或 Qdrant secret。

## 免費層資源策略

- 保留現有模型與 Qdrant 向量相容性，先量測 CPU 啟動、記憶體與單題延遲，不先修改模型或索引。
- 使用 lazy model loading；接受首次查詢較慢，換取 US$0 硬體成本。
- 維持每工作階段 20 題、全域同時 2 題、單題 60 秒 timeout，避免免費 CPU 被單一訪客占滿。
- 接受免費 CPU 閒置後自動休眠；不要求常駐、不購買持久磁碟。
- 若啟動 OOM、持續 timeout 或無法完成查詢，判定 CPU 驗收失敗並保持 private/paused；不得自動選用任何付費 tier。

## 錯誤處理與安全邊界

- 缺少訪客 Key、無效 Key、provider 限流或逾時時，回傳不含上游敏感內容的明確錯誤。
- Qdrant 連線失敗時停止回答，不降級至本機私有資料，也不建立新 collection。
- runtime key 的寫入、刪除與管理操作必須持續得到 403；若權限異常，立即停止公開流程並旋轉 key。
- logs 或 repository 若出現秘密值、私人 corpus、絕對路徑或本機 BM25 檔案，驗收立即失敗。
- 所有硬體操作前後都讀回 requested/current hardware；任何非 `cpu-basic` 狀態都立即 pause，且不繼續測試。

## 驗收與發布

Private Space 依序驗證：

1. requested/current hardware 都是 `cpu-basic`，沒有額外 storage 或 replica。
2. Space 能完成 build、startup 與 health check，並記錄啟動時間及峰值記憶體跡象。
3. 未輸入 Key 時不能送出問題。
4. 使用專用低額度 Gemini 與 OpenAI 測試 Key 各完成一題；provider/model 必須與選擇一致且無 fallback。
5. 檢查 session quota、global concurrency、timeout 與安全 logs。
6. 重驗 Qdrant 兩個 collections 可讀且寫入遭拒，point counts 不變。
7. 重跑專案測試、lint 與 release verifier。

只有全部通過才將 pull request 從 draft 轉為 ready，並另外取得公開 Space／合併 `main` 的確認。驗收後可將 Space 保持免費 CPU 自動睡眠；遇到濫用、資源不足或隱私異常則改回 private/paused。

## 費用保證

本設計不申請 T4、CPU Upgrade、額外 replica 或持久 storage。Hugging Face 硬體新增費用目標為 US$0。既有 Hugging Face PRO 訂閱、Gemini 預付額度與 Qdrant 免費叢集不在本次新增消費範圍；訪客的 LLM token 由訪客自己的 API Key 支付。
