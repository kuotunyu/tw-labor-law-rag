# Public Claim Matrix

本表把 README、README.en、DESIGN 與 EVAL_REPORT 的 material claims 映射到 committed config、trace、result 與 test。`scripts/verify_release.py` 是整合 verifier,但不取代下列原始 evidence。

## Evidence classes

- **offline-recomputed**:只讀 committed dataset/trace 即可重新計算並與 result 比較。
- **archived-provider/re-aggregated**:可重新聚合 committed numeric verdicts,但公開 evidence 不足以重新產生 provider output 或 judge 決定。
- **historical-observation**:來自保留在本機、不公開的 raw run 或實際使用觀察;公開文件保留明確邊界,不宣稱可重生。
- **configuration**:可由 committed defaults、script target list、lock、workflow 或 package artifact 核對。

## Matrix

| Public claim / location | Class | Config / source | Trace | Result | Test / verifier | Boundary |
|---|---|---|---|---|---|---|
| 目標 corpus 為 15 部（13 法律、2 命令）、2026-08-29 snapshot 共 884 條非刪除條文;README、README.en、EVAL_REPORT | configuration + offline-verified snapshot | `scripts/download_corpus.py::DUMPS`;`release/corpus_snapshot.json` | — | snapshot 記錄兩份官方 ZIP hash、15 部逐法條數與 metadata | `scripts/audit_corpus.py`;`tests/test_corpus_audit.py`;release verifier | 完整 raw/normalized corpus 不公開；committed snapshot 可驗證當次來源與完整性，但不保證法規今日未變 |
| 正式集 40 題、30 可答、10 不可答、15 部皆涵蓋;README、EVAL_REPORT | offline-recomputed | `eval/dataset/eval_set.jsonl` | dataset 本身 | official results 的 dataset metadata | `tests/test_eval_dataset.py`;`tests/test_release_verification.py` | Canonical UTF-8/LF SHA=`760e33…ca07` |
| 可靠性壓力集 60 題（40 可答、20 不可答）、15 部皆涵蓋、40 題中英夾雜、55 題長句;README、EVAL_REPORT | offline-recomputed | `eval/dataset/reliability_stress_v0.3.1.jsonl` | dataset 本身 | `eval/official/reliability_results.json` | `tests/test_reliability_dataset.py`;release verifier | 策展壓力集，不代表自然流量分布 |
| 題型 16 numeric、5 list、4 multi、4 scenario、1 negation、5 related OOKB、5 unrelated OOKB;EVAL_REPORT | offline-recomputed | `eval/dataset/eval_set.jsonl:q_type` | — | `release/manifest.json` | `scripts/verify_release.py` | 題型是策展 taxonomy,不是自然流量分布 |
| 8 組 ablation × 40 題;README、README.en、EVAL_REPORT | offline-recomputed | `release/manifest.json` | `eval/official/ablation_trace.jsonl` 320 rows | `eval/official/ablation_results.json` | `tests/test_official_artifacts.py`;release verifier | 每組 QID set 必須與 dataset 完全相同 |
| 主設定 Hit@5=0.967 (29/30)、MRR@10=0.906;全部四份主文件 | offline-recomputed | `src/rag/config.py` primary defaults | ablation trace 的 `structure/hybrid/reranker=true` | ablation results | `tests/test_official_artifacts.py`;release verifier | 只在 30 題可答子集計算 |
| 八組 Hit@5/MRR/latency table;DESIGN、EVAL_REPORT | offline-recomputed | official settings | ablation trace `rank`,`elapsed_ms` | ablation results 8 rows | `test_official_ablation_metrics_recompute_from_trace`;release verifier | latency trace 四捨五入到 0.1ms,比較容差 0.1ms |
| RRF k=60、retrieve top-20、final top-5;README 架構、DESIGN | configuration | `src/rag/config.py`;official result settings | — | both official result files | release verifier compares all three | 不是從結果反推的參數 |
| Structure chunking、400/80 fixed fallback;README、README.en、DESIGN | configuration + offline-recomputed comparison | `src/rag/config.py`;`src/rag/ingestion/chunkers.py` | ablation trace | ablation results | chunker tests;release verifier | 884 條與平均 147 字是 historical corpus observation |
| threshold=0.03;30/30 可答未被直接擋、9/10 不可答直接擋;README、DESIGN、EVAL_REPORT | configuration + offline-recomputed | `src/rag/config.py` | `eval/official/e2e_trace.jsonl` | e2e results | release verifier逐列核對 top score/stage;regression tests reject mismatch | 30/10 calibration range,不是 universal classifier |
| 壓力集 Hit@5=0.950、MRR@10=0.908、直接誤拒 1/40、攔截 17/20；formal guard 仍為 0.967/0.906、0/30、9/10;README、EVAL_REPORT | offline-recomputed | fixed model revisions、snapshot contract、`src/rag/config.py` | `eval/official/reliability_trace.jsonl` | `eval/official/reliability_results.json` | `tests/test_reliability.py`;release verifier | 0.03 sweep 無 Pareto-better 候選，故保留設定；不代表誤拒已消失 |
| 最終拒答 10/10、誤拒 1/30、threshold 9、LLM 2;README、README.en、EVAL_REPORT | offline-recomputed | refusal-stage contract in `src/rag/evaluation.py` | e2e trace | e2e results | `test_official_e2e_metrics_recompute_from_trace`;release verifier | LLM stage 是 recorded outcome;不需重呼 provider 即可計數 |
| 作答 29、generation calls 31、citation parse 28/29;EVAL_REPORT | offline-recomputed | citation/refusal aggregation | e2e trace | e2e results | official artifact tests;release verifier | eval-26 空引用保留歷史 parser 結果 |
| Faithfulness 4.90/5、relevancy 5.00/5;README、README.en、DESIGN、EVAL_REPORT | archived-provider/re-aggregated | generator `openai/gpt-5.1`;judge `openai/gpt-5-mini` recorded in result | e2e trace 29 numeric judge objects | e2e results | `compute_e2e_metrics`;official tests;release verifier | 無完整答案、judge reason/provider response,不可公開重生或獨立複判 |
| eval-10 正解未進主設定 top-5,LLM 誤拒;EVAL_REPORT case 1 | offline-recomputed outcome + historical interpretation | dataset ground truth | ablation/e2e traces | official results | release verifier validates rank/refusal grids | 原因是文件中的 interpretation,不當作因果證明 |
| 逐引用帶法規來源 URL、最後修正日與生效日，legacy payload 仍可讀;README、API/UI | configuration + executable verification | corpus metadata、chunk payload、API schema、UI renderer | — | — | loader/chunker/vector-store/API/UI tests | UI 只把 `https://law.moj.gov.tw/` 來源渲染成連結；既有雲端 payload 未重建前只顯示原有欄位 |
| Gemini 3.5 Flash-Lite／GPT-5.6 Luna cross-check 各 US$5 硬帽且目前 `pending_credentials`;README、EVAL_REPORT、DESIGN | configuration + fail-closed contract | `src/rag/provider_budget.py`;`eval/run_provider_crosscheck.py`;manifest | 正式 trace 不應存在 | 正式 result 不應存在 | provider budget/cross-check tests；release verifier 拒絕 pending 狀態下出現 artifact | 未使用其他專案金鑰、替代模型或 placeholder；不能宣稱已完成真實 provider 比較 |
| Official traces 無 prompt/provider response/API metadata/private path/PII;README、official README、PUBLICATION_BOUNDARY | offline-verified schema/privacy | strict field allowlists | ablation、e2e、reliability traces | — | `tests/test_release_verification.py`;`tests/test_official_artifacts.py`;release verifier | Scanner 回報只含 path/category/location,不回傳命中值 |
| Publishable Git history 只涵蓋 heads/tags/remotes，並逐 commit 掃描 path/content/binary/identity；PUBLICATION_BOUNDARY、REVIEWER_GUIDE | offline-verified publication boundary | `release/manifest.json:publication.history` | — | verifier 的 `publication.history_commits` 與 `history_ref_namespaces` | `tests/test_release_verification.py`;`scripts/verify_release.py` | 本機 `refs/archive/*` recovery evidence 不屬 publication graph；一般 `archive/*` branch 仍須稽核 |
| 兩份 sample 可隨 source archive 再散布;README、README.en、OGDL_ATTRIBUTION | configuration + official license | dataset 18290、OGDL 1.0、sample source URLs | — | `release/manifest.json` snapshot hashes | release verifier核對 hash/last-amended/provider URL | 必須保留 attribution;sample 不改授權為 MIT |
| Locked clean install、package build、CLI/FastAPI import;README、README.en、REVIEWER_GUIDE | configuration + executable verification | `pyproject.toml`;`uv.lock`;CI | — | built sdist/wheel | package test covers distributions/API import;CI and reviewer guide explicitly smoke CLI/API | 不執行 API lifespan,不載入模型或 provider |
| GitHub Actions 全 SHA pin、read-only permissions;REVIEWER_GUIDE | configuration | `.github/workflows/ci.yml` | — | — | action-pin scanner + release verifier | `actions/checkout` pin v6.0.3 commit;`setup-uv` pin v8.1.0 commit |

## Resume-safe wording

適合履歷或 portfolio 的一句話:

> Built a Traditional-Chinese hybrid RAG system and shipped a source-only public portfolio release with an auditable 15-law/884-article snapshot, per-citation legal-source provenance, a 40-question formal baseline, a 60-question reliability stress suite, eight retrieval ablations, offline-recomputed Hit@5/MRR/refusal evidence, privacy-reduced traces, and explicit archived/pending provider-evidence boundaries.

不應寫成「fully reproducible LLM evaluation」或「100% answerability classifier」;前者缺少可公開重生的 provider judgments,後者被 eval-32 的 0.9797 與正式集外 0.0146 誤拒共同否定。
