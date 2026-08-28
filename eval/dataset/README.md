# 評估資料集

## 檔案

- `mini_eval.jsonl` — 15 題開發 smoke set（13 可答 + 2 不可答）。其中 `mini-11`～`mini-15` 補足正式 v0.1.0 題集未直接覆蓋的五個條文；它們尚未納入正式指標。
- `eval_set.jsonl` — v0.1.0 的 40 題正式評估集（30 可答 + 10 不可答），其 committed evidence 與 canonical hash 維持不變。
- `reliability_stress_v0.3.1.jsonl` — 與正式證據分離的 60 題可靠性壓力集（40 可答 + 10 領域相關不可答 + 10 無關不可答）。它覆蓋 15 部法規，刻意加入長敘事、中英混合、間接問法與錯字；答案及引用繼承已人工查證的正式／mini 題，不用於回寫或取代 v0.1.0 指標。

`mini-11`～`mini-15` 於 2026-08-28 對照全國法規資料庫的《勞動基準法》第 32、36、54 條與《性別平等工作法》第 13、32-1 條；內容亦與本專案保留的 20240731／20230816 corpus snapshot 一致。

## Schema(每行一題)

| 欄位 | 說明 |
|---|---|
| `qid` | 題目編號 |
| `question` | 使用者問題(自然口語,不一定用法條術語) |
| `answer` | 標準答案(對照條文原文人工查證) |
| `sources` | ground truth 出處:`[{doc, article}]`,retrieval 指標以此計算 |
| `answerable` | 知識庫中是否有答案;`false` 者期望系統誠實拒答 |
| `q_type` | 題型:`single_article_numeric` / `single_article_list` / `multi_article` / `scenario` / `negation` / `out_of_kb_related` / `out_of_kb_unrelated` |
| `base_qid` | 僅壓力集使用；指向繼承答案與 ground truth 的正式／mini 題 |
| `style_tags` | 僅壓力集使用；標示 `narrative` / `code_switch` / `typo` / `indirect` / `multi_intent` 等問法壓力 |

## 出題原則

- 問題用求職者/勞工的自然口語,刻意避免直接複述條文用語(測試語意檢索,而非字面匹配)
- 標準答案必須可完全由 `sources` 列出的條文推得(faithfulness 的 ground truth)
- 不可答題分兩種:領域相關但不在庫(如就業保險法)、完全無關(如稅法)——前者是拒答機制最難的案例
- 條文以下載當時的版本為準;執行 `download_corpus.py` 後可在本機產生、不進版控的 `data/raw/laws/manifest.json` 查看 `last_amended`
