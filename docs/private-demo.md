# 私有展示與索引維護

[English](private-demo.en.md) ｜ [繁體中文](private-demo.md)

線上展示的運作方式與雲端索引的更新方式，原本放在 [README](../README.md)，內容原樣移到這裡。完整操作步驟見 [BYOK Hugging Face runbook](deployment/BYOK_HUGGINGFACE_RUNBOOK.md)。

## 私有 BYOK Docker Space（邀請制）

**Demo 狀態：** private Space 正常運行；僅限擁有者與受邀審閱者，不公開列出入口。

私有展示模式採 BYOK（Bring Your Own Key）：受邀者選擇 Gemini `gemini-3.5-flash-lite` 或 OpenAI `gpt-5.6-luna`，並在遮罩欄位輸入自己的專用 API Key。Key 只存在目前 Streamlit 工作階段、送往同容器 loopback FastAPI 的單次內部 header，以及該次請求建立的 provider client；不寫入檔案、聊天紀錄、共用設定或跨請求快取。Space 不設定站長的 `GEMINI_API_KEY`／`OPENAI_API_KEY`，也不做跨 provider fallback，因此受邀者不會消耗站長的模型 token 額度。

Space 只持有 Qdrant 兩個法規 collections 的唯讀 Key；建索引使用的短期 write/manage Key 於本機完成後立即撤銷。啟動時只讀 scroll payload，在記憶體重建 structure/fixed 兩份 BM25，不把私有 `data/raw/` 或 `storage/bm25_*.json` 放入 image。預設每個展示工作階段 20 題、全域同時 2 題、單題 timeout 60 秒，最多保留 1,000 個未過期的匿名工作階段。Key 隔離、唯讀權限與免費 `cpu-basic` 已完成驗收；完整操作與 rollback 見 [BYOK Hugging Face runbook](deployment/BYOK_HUGGINGFACE_RUNBOOK.md)。

### 人工更新 Qdrant 法規索引

雲端法規索引只接受有人值守的 blue-green 更新。先用 `scripts/rebuild_qdrant_blue_green.py` dry-run 驗證本機 official archives、normalized corpus 與 committed snapshot 完全一致；execute mode 另要求 temporary writer key 與重複 candidate 名稱，且只建立新 pair，不覆寫、重建或刪除正式 collections。完整指令、private cutover 與 rollback 見 [BYOK Hugging Face runbook](deployment/BYOK_HUGGINGFACE_RUNBOOK.md)；安全邊界與失敗模型見 [blue-green Qdrant maintenance design](release/BLUE_GREEN_QDRANT_MAINTENANCE_DESIGN.md)。
