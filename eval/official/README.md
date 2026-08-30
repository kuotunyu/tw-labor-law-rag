# 正式評估產物

這個目錄保存可隨程式碼公開、由 CI 驗證的精簡評估證據。原始執行紀錄仍留在本機
`eval/runs/`：其中有本機絕對路徑、完整模型輸出與除錯 log，因此刻意不進版控。

## 內容

- `ablation_results.json`：8 組 retrieval/chunking 消融設定的彙總指標。
- `ablation_trace.jsonl`：8 × 40 筆逐題 rank、top score 與 latency；不重複存題目文字。
- `e2e_results.json`：生成品質、引用解析覆蓋率、最終拒答率，以及按拒答層拆分的指標。
- `e2e_trace.jsonl`：40 筆逐題的拒答階段、分數、引用與 judge 數字；不含完整生成答案或 judge 理由。
- `reliability_results.json`：60 題可靠性壓力集的 Hit@5、MRR、延遲、拒答門檻掃描，以及既有 40 題正式集 guard 結果。
- `reliability_trace.jsonl`：60 筆只含 qid、answerable、rank、top score、threshold decision 與 latency 的隱私精簡 trace；不含問題文字、檢索內容或模型輸出。
- `reliability_formal_trace.jsonl`：40 筆正式 guard 的同格式隱私精簡 trace，供 verifier 獨立重算 guard 指標與 Pareto 決策。
- `portfolio_demo_v0.3.5.json`：10 題履歷／面試展示路徑的 content-free 離線檢索邊界證據；不含問題、答案、法條內文或 provider 請求。

## v0.3.5 compact portfolio regression

10 題包含 6 題可回答案例與 4 題知識邊界案例。在不建立 LLM、不呼叫 provider、不連線 Qdrant Cloud 的情況下，可回答題必要法源 Hit@5 `1.0`、可回答邊界通過率 `1.0`、定向路由契約 `1.0`、檢索階段決策契約 `1.0`。兩題明確跨領域問題在 `0.03` threshold 下停止；「現行最低工資金額」與「失業給付」和庫內法規語意相近，因此正確通過檢索層、留給後段引用完整性規則判斷。

此 artifact 對後兩題只證明「未被檢索層誤停」，並不宣稱已執行或驗證 LLM 拒答；每題均固定 `generation_called=false`。它是快速展示用 regression，不取代 40 題 formal baseline、60 題 reliability stress suite 或 archived provider evidence。`release/manifest.json` 綁定 dataset、snapshot、result、model revisions、runtime config 與精確彙總；release verifier 會重算每題排名、路由與邊界算術。

## v0.3.2 provider safety cross-check

Gemini `gemini-3.5-flash-lite`／OpenAI `gpt-5.6-luna` 的正式 safety cross-check 已完成，`release/manifest.json` 記錄完整 contract，且本目錄收錄 `provider_crosscheck_results.json` 與 `provider_crosscheck_trace.jsonl`。兩家各五筆請求：Gemini refusal accuracy `0.8`、citation success `1.0`、estimated cost `US$0.0022620`；OpenAI refusal accuracy `1.0`、citation success `1.0`、estimated cost `US$0.0026414`。這是 safety cross-check，不取代 `v0.1.0` formal evidence baseline 的正式模型品質評估。provider trace 嚴格 content-free，不含 question/answer text、provider payload 或 credentials。

其他 official traces 不含 token usage 或 API metadata。provider trace 僅含 strict allowlisted provider/model/verdict/token count/cost/latency，仍排除 prompts/questions/answers/provider payload/credentials/private paths/PII。
每份結果內含公開評估集的 SHA-256，可確認題目版本一致。

Retrieval、answerability、refusal、citation、ablation、reliability stress 與 formal guard summaries，以及門檻 Pareto 決策，都可由 committed traces 完整離線重算。Faithfulness/relevancy 只可從 trace 中留下的 numeric verdicts 再聚合;缺少的 provider output 與 judge reasoning 是刻意的 privacy/publication boundary,所以這兩項應標為 **archived provider evidence**,不得描述為只靠公開資料即可重生的評分。

可靠性壓力集使用 2026-08-29 經稽核的 15 部法規／884 條非刪除條文，以及固定 revision 的 BGE-M3 與 bge-reranker-v2-m3，在隔離的 local Qdrant 執行。現行 `0.03` 門檻於壓力集直接誤拒 1/40 可答題，並攔下 17/20 不可答題；既有正式集仍重現 0/30 直接誤拒與 9/10 攔截。門檻掃描沒有同時在壓力集與正式 guard 全面不劣、且至少一項更好的候選，因此保留 `0.03`，不自動修改 production config。

正式 run 當時 eval-26 的答案使用全形 `［1］`,舊 parser 因此留下空的 `cited_sources`;
目前程式已同時支援全形與半形括號並有回歸測試。產物保留當時的 28/29 解析結果,
以免把歷史結果悄悄改寫成修正後的數字;完整分析見 [`../../EVAL_REPORT.md`](../../EVAL_REPORT.md) 案例 3。

`refusal_stage` 的值：

- `null`：系統正常回答。
- `no_hits`：retrieval 沒有任何候選，未呼叫生成模型即拒答。
- `threshold`：rerank top score 低於門檻，未呼叫生成模型即拒答。
- `llm`：通過 retrieval gate，但生成模型判定 context 不足而拒答。

## 離線驗證 committed artifacts

```powershell
uv run python scripts/verify_release.py
uv run pytest tests/test_official_artifacts.py tests/test_release_verification.py -q
```

這條路徑不載入模型、不呼叫 provider、不啟動 Qdrant/Docker,並會核對 canonical dataset hash、40/30/10 組成、60 題壓力集、10 題 compact portfolio regression、8×40 grid、全部彙總算術、15 部／884 條 snapshot、provider complete contract、strict trace fields 與 privacy/secret patterns。

## 從 retained private raw runs 重新匯出

明確指定要公開的完整 run；exporter 不會自動挑「最新」目錄，避免誤選中斷的實驗：

```powershell
uv run python eval/export_official.py `
  --ablation-run eval/runs/<timestamp>-ablation `
  --e2e-run eval/runs/<timestamp>-e2e
```

exporter 會先驗證每組 QID、answerable 標記與 raw summary，然後才以固定順序、UTF-8
輸出；全程不呼叫模型或網路。加上 `--check` 可確認現有檔案是否與指定 raw run 完全一致。

這個 export 指令需要本機 retained `eval/runs/`;該目錄不在 publication allowlist,因此 public reviewer 應使用上一節的 committed-artifact 驗證,不能假設 raw runs 隨 repository 提供。

本次正式資料使用 2026-07-06 的 40 題評估集；完整出題原則與標準答案在
[`../dataset/`](../dataset/)。
