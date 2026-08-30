# v0.3.5 Three-Minute Reviewer Tour

這是一條給招募者與技術面試官的最短審閱路徑。專案是技術作品，不提供法律意見；完整限制見 [`README.md`](../../README.md#scope)。

## 0:00–0:30 — Problem and user boundary

一般語言模型容易在法規問題上混合記憶、推測與過期資訊。本專案將回答限制在 2026-08-29 稽核的台灣 15 部勞動法規／884 條非刪除條文，先檢索、重排，再把最多五筆法源交給生成模型。沒有足夠依據時，系統必須拒答。

展示環境是邀請制 private Space。訪客自行提供 Gemini 或 OpenAI API Key；站長不提供模型額度。Key 只進入單次 request header 與 request-scoped provider client，不寫入檔案、回答歷史或共用設定。

## 0:30–1:15 — Why Hybrid Search

BM25 擅長條號、法規專有名詞與精確詞彙；BGE-M3 向量檢索補足口語、同義與長句；RRF 在不混合不可比原始分數的情況下融合兩個排名；`bge-reranker-v2-m3` 再把候選縮到五筆。完整資料流與取捨見 [`DESIGN.md`](../../DESIGN.md#2-系統架構)。

兩個特別容易失敗的多法源問題使用窄範圍、可測試的 query expansion：新舊制資遣費，以及欠薪後勞工立即終止契約。擴充詞只進檢索，不會改寫交給模型的原始問題。

## 1:15–2:00 — What the measurements prove

| 證據 | 結果 | 能否離線重算 |
|---|---:|---|
| 40 題正式集、30 題可答 | Hit@5 `0.967`、MRR@10 `0.906` | 是，從 privacy-reduced trace 重算 |
| 10 題不可答 | 最終拒答 `10/10`；threshold 先擋 `9/10` | 是 |
| 60 題可靠性壓力集 | Hit@5 `0.950`、MRR@10 `0.908` | 是 |
| 8 組消融 | structure/fixed × BM25/vector/hybrid/reranker | 是 |
| 29 題 provider judge | faithfulness `4.90/5`、relevancy `5.00/5` | 只能重聚合已封存分數，不能重新評判 |

正式方法、失敗案例與校準限制見 [`EVAL_REPORT.md`](../../EVAL_REPORT.md)；每一項主張如何連到設定、trace、結果與測試，見 [`CLAIM_MATRIX.md`](CLAIM_MATRIX.md)。

## 2:00–2:30 — Refusal, citations, and known limits

- reranker top score `< 0.03` 時直接拒答，`generation_called=false`，不產生模型費用；
- 通過門檻後，生成模型仍可因法源不足而拒答；
- 每個引用只可指向當次 context 中的來源，並保留法規、條號、官方連結、最新異動日與生效日；
- 引用格式正確不等於法律結論必然正確；
- 0.03 是依固定資料集校準的 gate，不是通用 answerability classifier；壓力集已留下 `1/40` 可答題被直接誤拒的反例；
- 當年度最低工資金額、失業給付與著作權不在目前 15 部法規可可靠回答的範圍。

## 2:30–3:00 — BYOK security and free infrastructure

private Hugging Face Space 固定使用免費 `cpu-basic`、單一 replica、無付費持久 storage。Qdrant 使用 Free Tier；runtime key 只可讀目前 candidate 的 fixed/structure collections。建索引使用的 writer/transition keys 已撤銷。公開原始碼與私有執行資料的分界由 [`release/public-files.txt`](../../release/public-files.txt) 及 release verifier 強制執行。

## Reproduce without provider keys

```bash
uv sync --locked
uv lock --check
uv run python scripts/verify_release.py
uv run pytest -q
uv build
```

這條路徑不載入模型、不呼叫 Gemini/OpenAI、不需要 Qdrant、Docker 或 GPU。去識別化結果與重算說明位於 [`eval/official/`](../../eval/official/README.md)。
