# 評估資料集

## 檔案

- `mini_eval.jsonl` — 15 題開發 smoke set（13 可答 + 2 不可答）。其中 `mini-11`～`mini-15` 補足正式 v0.1.0 題集未直接覆蓋的五個條文；它們尚未納入正式指標。
- `eval_set.jsonl` — v0.1.0 的 40 題正式評估集（30 可答 + 10 不可答），其 committed evidence 與 canonical hash 維持不變。
- `reliability_stress_v0.3.1.jsonl` — 與正式證據分離的 60 題可靠性壓力集（40 可答 + 10 領域相關不可答 + 10 無關不可答）。它覆蓋 15 部法規，刻意加入長敘事、中英混合、間接問法與錯字；答案及引用繼承已人工查證的正式／mini 題，不用於回寫或取代 v0.1.0 指標。
- `severance_refusal_policy_v0.3.6.jsonl` — v0.3.6 路由感知拒答門檻的 30 題人工複核校準契約（15 題新舊制資遣費比較正例 + 15 題語意碰撞負例）。此檔是含題文的私有輸入；公開證據只可保留 qid、來源名次、路由名稱、數值決策輸入與布林契約結果。

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

## v0.3.6 資遣費拒答校準契約

`severance_refusal_policy_v0.3.6.jsonl` 每行只有 `qid`、`question`、`case_type`、`answerable`、`sources`、`required_routes`、`prohibited_routes`、`expect_generation` 與 `style_tags`。資料集固定為 `severance-policy-001`～`030`，前 15 題必須是可答正例，並同時以《勞工退休金條例》第 12 條與《勞動基準法》第 17 條為來源；後 15 題是單一制度、一般終止、預告、欠薪、退休、無關新舊用語、部分 cue 與多路由碰撞。

正例必須正確地只啟用 `severance_comparison`，並覆蓋法律中文、口語中文、中英混合、標點、長敘事、新舊制反轉次序、公式、上限與混合年資。碰撞負例不得落入單一 `severance_comparison` 特殊門檻；只有欠薪立即終止題可啟用 `wage_arrears_termination`，而同時命中兩條路由的題必須保留多路由身分以驗證全域門檻 fallback。載入器對欄位、順序、布林值、來源、路由、樣式覆蓋與重複值全部 fail closed。

校準觀測列只接受 `qid`、正規來源名次、allowlist 路由與未捨入 `top_score`；來源、路由、可答性與生成預期都由每個 qid 的內建正規契約重新計算，不接受呼叫者預先計算的 pass/fail 布林值。Stress 與 formal guard 必須提供新鮮、未捨入的 raw score，每列標示 `score_precision: raw_unrounded`，並在公開產物中以兩組 raw evidence canonical SHA-256 綁定；既有只保留小數四位的 public trace 不符合此輸入契約。選擇器會透過共用拒答政策重播每個候選的全部決策，而公開產物會移除 raw guard 列，只保留經重算的匯總與內容無關的 30 筆 case 結果。
