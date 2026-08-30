# v0.3.5 Interview Demonstration Script

目標時間 3–5 分鐘。Live demo 僅限已獲邀請的 private Space 使用者，並使用展示者自己的 Gemini/OpenAI API Key。生成文字可能因 provider 而異；以下固定驗收的是檢索、路由、拒答與引用來源，不是逐字答案。

## 開場（20 秒）

1. 指出知識庫快照 `2026-08-29`、15 部法規、884 條非刪除條文。
2. 說明 BYOK：Key 不進聊天紀錄、不寫檔，費用由持有人承擔。
3. 保持預設 `structure-aware / hybrid / reranker`；進階比較設定維持收合。

## Demo 1 — `hours`：單一法條與精確數字（40 秒）

- 點擊：`每日與每週工時`
- 問題：`勞工每天和每週的正常工作時間上限是多少？`
- 預期來源：《勞動基準法》第 30 條
- 打開：`引用來源`，檢查法規、條號、官方連結、異動／生效日期
- 工程重點：BM25 與向量檢索互補，引用 metadata 與當次 context 對齊

## Demo 2 — `severance`：跨法規檢索（50 秒）

- 點擊：`新舊制比較`
- 問題：`適用勞退新制的勞工被資遣時，資遣費怎麼計算？和舊制有什麼不同？`
- 預期來源：《勞工退休金條例》第 12 條，以及《勞動基準法》第 17 條
- 打開：`引用來源` 與 `檢索細節（debug）`
- 工程重點：窄範圍 query expansion 只影響 retrieval，原問題原封不動交給 generator；跨法規案例必須保留兩個 authority

## Demo 3 — `wage-arrears`：已知失敗的回歸修正（50 秒）

- 點擊：`欠薪立即離職`
- 問題：`公司一直拖欠薪水，我可以不經預告直接離職嗎？這樣還能拿到資遣費嗎？`
- 預期來源：《勞動基準法》第 14 條（並由 context 支援相關資遣費說明）
- 打開：`檢索細節（debug）`
- 工程重點：v0.3.4 只在「欠薪」與「勞工立即離職」兩組 cue 同時出現時啟動 Article 14 route；20 題 targeted regression 同時測正例與 collision negatives

## Demo 4 — `refusal`：知識庫邊界與零生成（40 秒）

- 點擊：`知識庫外問題`
- 問題：`著作權的保護期間是幾年？`
- 預期：threshold 拒答、沒有引用、`generation_called=false`
- 打開：`檢索細節（debug）`
- 工程重點：低分問題在生成前停止，不讓模型用常識補答案，也不產生 provider token 費用

## 收尾（30 秒）

- 正式結果是 40 題 formal、60 題 reliability、8 組 ablation；這四題只是可快速觀察系統邊界的展示路徑。
- faithfulness/relevancy 是 archived provider evidence，公開 repository 可重聚合數字，但刻意不發布完整回答與 judge 理由，因此不能聲稱可獨立重判。
- 系統不是法律意見，且 snapshot 後的新修法或法規外問題必須先人工更新／擴充知識庫。

若 private Space 睡眠、尚在啟動或 provider 暫時不可用，改走 [`V035_REVIEWER_TOUR.md`](V035_REVIEWER_TOUR.md) 與 [`eval/official/`](../../eval/official/README.md) 的完全離線證據，不為了 demo 改用付費硬體。
