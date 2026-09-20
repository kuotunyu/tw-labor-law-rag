# 版本紀錄

[English](changelog.en.md) ｜ [繁體中文](changelog.md)

各版本的變更說明，原本放在 [README](../README.md)，內容原樣移到這裡。正式評估數字沿用 `v0.1.0` 那次評估的結果（見 [EVAL_REPORT.md](../EVAL_REPORT.md)）；以下各版本都沒有改寫那些數字。v0.3.5 的英文版發佈說明另見 [V035_RELEASE_NOTES.md](release/V035_RELEASE_NOTES.md)。

## v0.3.5 Portfolio readiness

本版把私有 BYOK 展示整理成 reviewer-first 體驗：首頁先說明可驗證能力與費用邊界，再引導受邀者選擇 Gemini／OpenAI、於遮罩欄位輸入自己的專用 Key，並以逐步狀態、引用來源與可展開 debug 證據呈現結果。Space 保持 private、免費 `cpu-basic`，不持有站長的 LLM Key，也不做跨 provider fallback。

新增 10 題完全離線、content-free 的示範回歸：6/6 可答題來源契約通過，10/10 路由與檢索階段決策契約通過，provider calls 為 0。另以法務部官方來源建立 15 部／884 條逐條文 SHA-256 baseline；人工 audit 會同時報告 law/source 欄位與新增、移除、變更條號，不建立排程或自動 writer。這些證據不取代既有 40 題 formal baseline、60 題 reliability suite 或 archived provider judgments。

## v0.3.4 欠薪／立即離職檢索強化

只有同時命中「欠薪」與「勞工立即離職」兩組已審閱 cue 的問題，檢索管線才會補上《勞動基準法》第 14 條的固定法規詞。BM25、向量檢索與 reranker 看到擴充查詢；生成模型仍收到使用者原始問題。

本版沒有新增 provider 呼叫、調整 0.03 門檻、重建 Qdrant 或改寫歷史指標。`v0.1.0` formal baseline 與 `v0.3.1` reliability evidence 保持原證據版本；v0.3.4 的公開主張只涵蓋可由單元測試驗證的決定論式路由契約。

## v0.3.3 新舊制資遣費檢索強化

這是 `v0.3.3` source-only runtime and deployment release。當問題同時包含資遣、新制、舊制與計算／比較語意時，檢索管線會以決定論式 query expansion 補上「勞工退休金條例、勞動基準法、工作年資、平均工資、六個月」等法規檢索詞。擴充內容只送往 BM25、向量檢索與 reranker；生成模型仍收到使用者的原始問題，避免檢索輔助詞改寫使用者意圖。

這項擴充必須同時命中四組 cue 才會啟用，因此一般資遣、退休或單純制度差異問題不會被廣泛改寫。`v0.1.0` 正式模型品質基準、`v0.3.1` reliability evidence 與 `v0.3.2` provider safety cross-check 仍維持原來的證據版本；本版沒有用新的 provider 呼叫改寫歷史指標。

## v0.3.2 provider safety cross-check：可靠性、來源與雙模型 runtime

這是 `v0.3.2` source-only runtime and deployment release。公開 API/UI 預設使用 Gemini `gemini-3.5-flash-lite`，若伺服器同時設定 OpenAI，使用者可逐次請求選擇 `gpt-5.6-luna`。這些型號可分別由 server-side `GEMINI_GENERATION_MODEL` 與 `OPENAI_GENERATION_MODEL` 覆寫；對應 key 已設定時，`LLM_PROVIDER=gemini` 決定省略請求選擇時的預設 provider，否則 API 會改用另一個已設定的公開 provider；`LLM_FALLBACK_ENABLED=true` 才允許備援。`GEMINI_API_KEY` 與 `OPENAI_API_KEY` 只存在 API 伺服器環境，前端不接收、保存或顯示 key。

備援邊界是固定的：只有主 provider 發生連線、限流、5xx 服務或空回應等 operational failure 時，才會最多嘗試另一個已設定的公開 provider 一次。檢索階段拒答不會呼叫生成模型；模型依據條文拒答、provider 安全擋下或政策拒絕也不會 fallback。正式評估路徑仍直接固定單一 generator/judge provider，不使用 runtime fallback，避免路由變動改寫評估設定。

Streamlit 側邊欄的「回答模型」只顯示 API `/models` 回傳的已設定 Gemini/OpenAI；送出問題時會將選擇的 provider 一併傳給 `/query`。回應中 `requested_provider` 保留指定 provider，`provider` 與 `model` 是實際生成結果的 metadata，`fallback_used`/`fallback_from` 說明是否改走備援，`generation_called=false` 表示在檢索層已拒答。UI 會分開顯示指定與實際作答模型，並在改走備援時警示。Live provider smoke test 需要伺服器端本機 secrets，不屬公開 offline CI。

`v0.1.0` 的正式模型品質指標仍是歷史結果，由 `release/manifest.json` 所列 generator 與 judge 模型產生；本版沒有取代或重新審計這些數值。本版已在不呼叫 provider 的情況下，以 60 題壓力集與既有 40 題正式集 guard 重跑 retrieval 與 threshold 行為。

Gemini `gemini-3.5-flash-lite`／OpenAI `gpt-5.6-luna` 的 US$5 硬上限 safety cross-check 已完成並 fail closed：兩家各五筆請求；Gemini refusal accuracy `0.8`、citation success `1.0`、estimated cost `US$0.0022620`；OpenAI refusal accuracy `1.0`、citation success `1.0`、estimated cost `US$0.0026414`。公開 evidence 僅含去識別化、嚴格 content-free 的十筆 trace、可重算的 metrics 與每家 US$5 預算 ledger；trace 不含 question/answer text、provider payload、憑證或原始 run artifacts。這是 safety cross-check，不取代 `v0.1.0` formal evidence baseline 的正式模型品質指標。

私有 BYOK Space 與人工更新 Qdrant 索引的說明移到 [private-demo.md](private-demo.md)。
