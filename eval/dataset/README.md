# 評估資料集

## 檔案

- `mini_eval.jsonl` — 10 題迷你評估集(8 可答 + 2 不可答)。開發期每個階段結束都跑,用來快速量化改動效果。
- `eval_set.jsonl` — 40 題完整評估集(Phase 4 建立):30 可答 + 10 不可答。

## Schema(每行一題)

| 欄位 | 說明 |
|---|---|
| `qid` | 題目編號 |
| `question` | 使用者問題(自然口語,不一定用法條術語) |
| `answer` | 標準答案(對照條文原文人工查證) |
| `sources` | ground truth 出處:`[{doc, article}]`,retrieval 指標以此計算 |
| `answerable` | 知識庫中是否有答案;`false` 者期望系統誠實拒答 |
| `q_type` | 題型:`single_article_numeric` / `single_article_list` / `multi_article` / `scenario` / `negation` / `out_of_kb_related` / `out_of_kb_unrelated` |

## 出題原則

- 問題用求職者/勞工的自然口語,刻意避免直接複述條文用語(測試語意檢索,而非字面匹配)
- 標準答案必須可完全由 `sources` 列出的條文推得(faithfulness 的 ground truth)
- 不可答題分兩種:領域相關但不在庫(如就業保險法)、完全無關(如稅法)——前者是拒答機制最難的案例
- 條文以下載當時的版本為準;執行 `download_corpus.py` 後可在本機產生、不進版控的 `data/raw/laws/manifest.json` 查看 `last_amended`
