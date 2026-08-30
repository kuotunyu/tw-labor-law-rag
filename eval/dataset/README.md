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

`severance_refusal_policy_v0.3.6.jsonl` 每行只有 `qid`、`question`、`case_type`、`answerable`、`sources`、`required_routes`、`prohibited_routes`、`expected_outcome` 與 `style_tags`。`expected_outcome` 是精確列舉：`generation`、`no_hits` 或 `threshold`，不再以布林值近似決策階段。資料集固定為 `severance-policy-001`～`030`，前 15 題必須是可答正例，並同時以《勞工退休金條例》第 12 條與《勞動基準法》第 17 條為來源；後 15 題是單一制度、一般終止、預告、欠薪、退休、無關新舊用語、部分 cue 與多路由碰撞。

正例的路由必須精確等於單一 `severance_comparison`，並覆蓋法律中文、口語中文、中英混合、標點、長敘事、新舊制反轉次序、公式、上限與混合年資。碰撞負例仍採「所有 `required_routes` 都存在，且所有 `prohibited_routes` 都不存在」；額外的非禁止路由可以保留。共用拒答政策另外限定：只有路由組正確等於單一 `severance_comparison` 才使用特殊門檻，加上任何其他路由都回退全域門檻。`severance-policy-023` 明確要求 `no_hits`；`024` 保持 `generation`；`027` 明確要求空路由、正命中、低於全域 `0.03` 的分數與 `threshold`，因此 `no_hits` 不通過其契約。載入器對欄位、順序、列舉值、來源、路由、樣式覆蓋與重複值全部 fail closed。

校準觀測列只接受 `qid`、正規來源名次、allowlist 路由、`hit_count` 與未捨入 `top_score`；來源、路由、可答性與精確 outcome 都由每個 qid 的內建正規契約重新計算，不接受呼叫者預先計算的 pass/fail 布林值。Stress 與 formal guard 的 fresh-run 輸入列只接受正規 `qid`、可答性、名次、`hit_count`、正規路由與保留輸入精度的 `top_score`；不接受呼叫者提供 `has_hits` 或 `reranker_enabled`。`hit_count` 必須介於 0 與固定 `top_k_final=5` 之間；零命中必須同時是零分與 null 名次，正命中必須有大於零且位於 reranker 分數範圍的分數，正名次不得大於命中數。評分器由 `hit_count > 0` 推導 `has_hits`，並由固定 retrieval configuration 推導 `reranker_enabled=true`；兩個推導值連同 `hit_count` 都會公開，供 Task 7 重播。任一 guard 組的全部分數若都與小數四位相容即 fail closed，因此不能直接使用 v0.3.1 的捨入 public trace；Task 6 必須從新鮮的離線檢索 pipeline 產生這些列。正式產物 schema 版本為 `1.2`。

正式產物以 `run_origin: fresh_offline_retrieval` 作為釋出來源列舉，並記錄乾淨 committed candidate revision、exact resolved execution device 與 decision-relevant `rrf_k`；公開 guard 列則與 dataset、corpus snapshot、code revision、固定 model revisions 及 retrieval configuration 一起計算 `guard_evidence_binding_sha256`。這個列舉是等待 release verifier 跨任務檢查的來源聲明，綁定 hash 只確保公開列與公開 metadata 一致，兩者都不獨立證明實際執行來源。Task 7 可以從公開列、全域門檻與七個候選門檻重播所有 stress/formal 決策與匯總。
