# 設計決策(DESIGN)

> 本文件記錄每個非顯而易見的技術選型:為什麼這樣做、考慮過什麼替代方案、實際量測結果如何。數據來源見 [EVAL_REPORT.md](EVAL_REPORT.md)。

## Evidence interpretation

40 題資料集、8 組 ablation、Hit@5、MRR、latency 算術、answer/refusal counts 與設定一致性,可由 `scripts/verify_release.py` 從 committed privacy-reduced traces 離線重算。Faithfulness/relevancy 則是 archived provider evidence:公開 repository 只保留 numeric verdicts 供再聚合,沒有完整生成答案、judge reason 或 provider response,因此不能宣稱可從公開 evidence 重新產生 judge 決定。逐項映射見 [docs/release/CLAIM_MATRIX.md](docs/release/CLAIM_MATRIX.md)。

## v0.3.2 provider safety cross-check

Gemini `gemini-3.5-flash-lite` 與 OpenAI `gpt-5.6-luna` 都完成五筆 safety cross-check：Gemini observed refusal accuracy `0.8`、citation success `1.0`、estimated cost `US$0.0022620`；OpenAI observed refusal accuracy `1.0`、citation success `1.0`、estimated cost `US$0.0026414`。公開 trace 維持嚴格 content-free：不含 question/answer text、provider payload 或 credentials。這是執行器安全邊界的 cross-check，不取代 `v0.1.0` formal evidence baseline 的正式模型品質評估。

## 1. 為什麼是 Hybrid Search,不是純向量?

**選擇**:BM25(關鍵字)+ BGE-M3(向量),用 RRF 融合。

**理由**:向量檢索對「語意相近但字面不同」的查詢很強,但對法律文件有個結構性弱點——條文常用生僻或精確的法律術語(如「普通傷病假」),使用者卻用口語提問(如「病假」)。這種情況下關鍵字比對反而更可靠,只要斷詞正確就能精準命中。反過來,BM25 完全無法處理「拖欠薪水」對應「不依勞動契約給付工作報酬」這種零共同字的詞彙鴻溝,向量檢索才是解方。兩者的失敗模式是互補的,不是其中一個更好而已。

**實測佐證**:40 題評估集上,純 BM25 hit@5 只有 0.833–0.867(structure/fixed),純向量 0.900–0.933,hybrid 進一步提升到 0.933–0.967。個別案例更直接:eval-17(病假)向量命中、BM25 完全 miss;eval-01(加班費,fixed chunking)BM25 靠關鍵字救回向量漏掉的條文。詳見 EVAL_REPORT.md「發現 1」。

**Tradeoff**:多維護一份 BM25 索引(pickle,含 jieba 斷詞結果)、多一次檢索呼叫。延遲增加很小(BM25 是 CPU 上的稀疏檢索,毫秒級),換來的召回率提升值得。

## 2. 為什麼用 RRF 融合,不是加權平均分數?

**選擇**:Reciprocal Rank Fusion,`score = Σ 1/(k + rank)`,k=60。

**理由**:BM25 分數(未正規化的 tf-idf 變體,範圍隨語料而變)和向量 cosine 相似度(0–1)量綱完全不同,直接加權平均需要為每個資料集手調正規化參數,而且脆弱——換一批文件、換一個 embedding model,權重就要重調。RRF 只看排名不看分數,對兩種檢索系統的分數分布完全不敏感,是業界(如 Elasticsearch、Azure AI Search)hybrid search 的標準做法。k=60 是 Cormack et al. (2009) 原始論文的建議值,也是多數實作的預設——分母加 60 讓排名前幾名的分數差距不會過度放大,避免任一路檢索的「隨機第一名」壓過另一路的「穩定前三名」。

**Tradeoff**:RRF 犧牲了排序精度換取召回率——實測 hybrid 的 MRR@10 比純向量略低(structure:0.822 vs 0.850),因為 RRF 把兩路檢索的候選都混進來,原本向量檢索排第一的正解不一定还排第一。這正是還需要 reranker 的原因(見下一節)。

## 3. 為什麼 Hybrid 之後還需要 Reranker?

**選擇**:bge-reranker-v2-m3(cross-encoder),對 RRF 融合後的 top-20 重排序,取 top-5。

**理由**:第一階段檢索(BM25、向量)都是「bi-encoder」架構——查詢和文件分別編碼成向量再比對,速度快但無法讓查詢和文件的每個詞彼此關注(attention)。Cross-encoder 把查詢和候選文件一起餵進模型,精度高很多,但因為每個候選都要跑一次完整前向傳播,無法用於全庫檢索(20 個候選跑 rerank 還行,本次 observed structure corpus 的 884 個候選全部 rerank 就太慢)。因此標準做法是「先用便宜的方法縮小候選集,再用貴的方法精排」。

**實測佐證**:reranker 不只讓排序更好,也提供一個適合做第一層快速拒答的訊號,但它不是完整的 answerability classifier。40 題正式評估中,可答題 top-1 為 0.0668–0.9981,不可答題為 0.0005–0.9797,分佈有明顯重疊;不過 0.03 門檻仍讓正式集的 30/30 可答題通過,並直接擋下 9/10 不可答題。剩下的 eval-32 雖與庫內主題高度相關、分數達 0.9797,條文卻不足以回答,最後由 LLM 正確拒答。這正好支持「reranker 快速篩選 + LLM 語意判斷」的兩層設計。此外 hybrid+rerank 把 MRR 從 0.822 拉回 0.906,補回 RRF 犧牲掉的排序精度。這個 30/10 範圍不能外推為所有口語問法的可靠邊界;正式集外已有 0.0146 的敘事式可答問題被直接誤拒,詳見 EVAL_REPORT 案例 7。

**Tradeoff**:延遲增加約 230ms(見 EVAL_REPORT.md 消融表的 hybrid 58ms → hybrid+rerank 292ms)。對互動式問答系統這個延遲可接受;若要做高吞吐量批次處理,可以考慮只在分數接近門檻的邊界案例才觸發 rerank。

## 4. 為什麼是兩層拒答,不是一層?

**選擇**:
1. **檢索層**:reranker top-1 分數 < 0.03 → 直接拒答,不呼叫 LLM
2. **生成層**:prompt 明確指示「條文不足以回答時,輸出固定短語」,LLM 自行判斷

**理由**:只靠 prompt 指示 LLM 拒答,風險是模型仍可能用參數化知識(pretraining 學到的東西)硬湊答案,尤其問題聽起來眼熟時(如「失業給付」跟勞動法規高度相關但實際不在庫裡)。只靠檢索層分數門檻,又會誤判「檢索到相關但不完整」的情況——例如檢索到的條文只回答了問題的一半,這種細膩判斷 cross-encoder 分數做不到,只有 LLM 讀懂上下文才能判斷。兩層各司其職:檢索層擋掉「明顯不在庫裡」的問題(省一次 LLM 呼叫),生成層擋掉「檢索到東西但答不了」的邊緣情況。

**實測佐證**:40 題評估集,10 題不可答全數正確拒答(10/10):9 題由檢索層直接擋下,eval-32 則通過門檻後由生成層拒答。30 題可答中沒有任何一題被第一層門檻直接擋下;唯一誤判是 eval-10 在 context 不足時被生成層拒答(1/30)。也就是說,觀察到的失敗方向是「該回答卻誠實說不知道」,而不是「不該回答卻瞎掰」——這是法律問答中刻意偏好的安全方向。

**Tradeoff**:門檻(0.03)是用 10 題迷你集初步校準的經驗值,不是理論推導出的絕對值。40 題正式評估顯示它能讓 30/30 可答題進入生成層,同時直接擋下 9/10 不可答題;但 eval-32 的高分也證明不能把它解讀成完整的可答性邊界。換語料、reranker 模型或問題分佈時,都必須重新校準——因此門檻放在 config,而不是散落寫死在流程裡。

## 5. Chunking:為什麼兩種策略都做,分別怎麼決定參數?

**選擇**:
- **structure-aware**:法規按「條」為單位切,每個 chunk 前綴「法規名 + 章節 + 條號」作為 context header;超長條文(如勞基法§28)才依句子邊界二次切分,上限 1000 字
- **fixed-size**:400 字滑動視窗、overlap 80 字,尊重句子邊界(不會從句子中間硬切)

**理由**:法規文件本身有天然的結構單位(條文),structure-aware 是「利用文件結構」這個常見 RAG 優化手法的具體實作;fixed-size 則是不假設任何文件結構、可以套用在任何純文字上的通用方法(README 的 PDF/Markdown/txt 支援也是靠這個)。兩者都做,一方面是規格要求的消融對照組,另一方面是誠實面對「structure-aware 需要文件有清楚結構才能用」——通用性與精確度的 tradeoff 需要用數據說話,不能只憑直覺選一種。

400 字的選擇考量:BGE-M3 支援到 8192 token 的長上下文,但法規條文本身多數在 100–300 字之間(884 條文平均長度 147 字,structure-aware chunk 平均長度也是 147),400 字大約能容納 1–2 條完整條文,overlap 80 字(20%)則是常見的經驗法則,避免關鍵資訊剛好落在切點兩側被拆散。

**實測佐證**:fixed-size 的 hit@5 略高(1.000 vs structure 的 0.967),但 MRR 明顯較差(0.847 vs 0.906)——固定視窗偶爾能「意外」把多條相關條文包進同一個 chunk 而提高召回,但整體語意被稀釋,正解的排序位置較不穩定。最直接的例子是 eval-01(加班費):fixed/vector 下勞基法§24 被切成語意不集中的片段而未進 top-5;structure/vector 完整保留該條文,清楚命中第一。生成任務需要的是「最相關的條文排在最前面」(因為只取 top-5 給 LLM),所以正式環境預設用 structure-aware,fixed-size 保留作為沒有結構化文件時的 fallback。

## 6. 為什麼自建 LLM-as-judge,不用 RAGAS?

**選擇**:手刻的 judge(`eval/judge.py`),中文 rubric,一次呼叫同時評 faithfulness 與 relevancy。正式 run 的 numeric verdicts 是可再聚合的歷史 evidence,不是可由公開 repository 完整重生的 deterministic benchmark。

**理由**:RAGAS 的內建 prompt 是英文優先設計,套在繁體中文法律文本上,經驗上常出現評分標準與語言習慣對不齊的問題(例如對「精簡但正確」的中文法律用語誤判為資訊不足)。這兩個指標的定義其實不複雜,自己刻一個中文 rubric、附上 1–5 分的具體錨點描述(而不是讓模型自由發揮),反而更容易掌控評分標準、也更容易在 EVAL_REPORT.md 裡向讀者交代「這個分數是怎麼打出來的」。一次呼叫評兩個指標也把 judge 的 API 呼叫量減半,在免費/低額度方案上很有感。

**Tradeoff**:自建 judge 需要自己驗證評分穩定性(RAGAS 有社群驗證過的 prompt),且既有 `v0.1.0` 正式評分只用單一 provider(gpt-5-mini)。`v0.3.2` 已完成 Gemini／OpenAI 各 US$5 硬上限的五筆 safety cross-check；這些 observed safety metrics 不以其他專案金鑰、替代模型或 placeholder 補數字，也不改寫 formal baseline。

## 7. Qdrant:為什麼 local/server 雙模式?

**選擇**:`QDRANT_MODE=local`(embedded 檔案模式,免 Docker)或 `server`(docker-compose 服務),同一份程式碼透過 config 切換。

**理由**:開發機沒有 Docker Desktop 時,local mode 讓開發與跑分完全不受阻——這也是為什麼整個 Phase 0–4 都能在沒有 Docker 的環境下走完。交付物仍然提供 docker-compose,滿足「可用 server 模式跑」的部署需求。程式碼層面透過 `VectorStore` 類別包一層,呼叫端(retriever、build_index、eval 腳本)完全不知道背後是哪個模式,只有 `config.py` 的一個環境變數差異。

**Tradeoff**:local mode 是單行程鎖定(portalocker),同一個 storage 目錄不能被兩個行程同時開啟——這在 Phase 4 跑評估時實際踩過:FastAPI server 開著時執行評估腳本會直接報 `AlreadyLocked`,必須先停掉 API server。Server mode 沒有這個限制,支援多行程並發存取,是正式部署時該用 server mode 的直接理由。

## 8. LLM Provider 抽象:為什麼四選一,thin adapter 設計?

**選擇**:`LLMAdapter` 只定義一個 `generate(system, user, temperature, max_tokens) -> str` 方法,Anthropic / OpenAI / Gemini / Ollama 各自實作。

**理由**:最初規格只要求 Anthropic/OpenAI/Ollama 三選一,後來因為使用者手邊 key 的可用性(Anthropic 額度不明、OpenAI 需要儲值、Gemini 有免費額度但限制嚴格)實際加了 Gemini 當第四個,新增過程只動了 `llm.py` 一個檔案加一個類別、`config.py` 加一個 Literal 選項——這個抽象層的價值在這次擴充中直接被驗證:接口設計得夠薄,加供應商不需要碰檢索、生成組裝、拒答判斷的任何邏輯。

**Tradeoff**:代價是每個供應商的模型特性差異(temperature 支援、token 參數命名、額度限制)全部要在各自的 adapter 裡個別處理,不能假設行為一致。這次實際踩到三個坑,現在都在 `llm.py` 的 adapter 裡有明確處理與註解:
- **Gemini 2.5「思考」token**:預設開啟的 thinking 會佔用 `max_output_tokens` 預算,實測 980/1024 token 花在看不見的推理上,只剩 40 token 輸出可見答案,長答案被腰斬。修法:flash 系列模型明確關閉 thinking(`thinking_budget=0`),同時把預設 token 上限從 1024 提高到 2048。
- **GPT-5 系列參數相容性**:`max_tokens` 被拒絕(需改用 `max_completion_tokens`),部分模型(如 gpt-5-mini)拒絕非預設 `temperature`。Adapter 在收到 400 錯誤時偵測是哪個參數不支援,自動移除後重試一次,之後同一個 adapter instance 記住這個限制不再重試。
- **額度不透明**:provider 帳戶實際配額可能低於公開頁面的一般說明；一般評估腳本保留 429 退避，而新的預算 cross-check 不跨 provider fallback，並在任何未知 token usage 時 fail closed。

## 9. 為什麼引用格式是 `[數字]` 而不是直接附條文全文?

**選擇**:LLM 在答案中用 `[1][2]` 標註引用編號,對應 prompt 裡條文的呈現順序;API 另外回傳結構化的 `sources` 陣列(法規名、條號、原文段落),由前端渲染成可展開的引用面板。

**理由**:把「回答的行內引用標記」和「引用內容的展示」分離,LLM 只需要輸出簡短的數字標記(降低生成錯誤的機會),實際的法規名稱、條號、原文由檢索結果直接帶出(不經過 LLM 转述,不會被生成過程扭曲)。這也讓引用驗證變成一個簡單的規則檢查:解析出的編號如果超出提供的條文數量範圍,直接捨棄該筆引用,不會因為 LLM 引用錯誤編號而顯示錯誤的法條。

**Tradeoff**:繁體中文生成偶爾會用全形括號「［1］」而非半形「[1]」(gpt-5.1 實測觀察到),引用解析的正規表示式因此需要同時支援兩種括號——這是 Phase 4 端到端評估才抓到的真實案例(單元測試原本只覆蓋半形),已修正並補上回歸測試。

## 10. 為什麼要有完整 corpus snapshot 與逐引用 provenance?

**選擇**:每次可靠性 run 先下載法務部法律／命令 ZIP，核對來源 URL、ZIP SHA-256、15 部法規清單、各法規代碼／修正／生效日期、逐條 canonical hash 與 884 條總數；只有整份 snapshot 完全相符才建立索引。`SourceUnit → Chunk → Qdrant payload → Answer.sources → FastAPI → Streamlit` 全鏈路帶 `source_url`、`last_amended`、`effective_date`，舊 payload 缺欄位時則安全顯示既有引用。

**理由**:只保留兩份 sample 無法證明真正建索引的 15 部語料是哪一版；只在 UI 顯示條號也無法讓使用者回查官方法規。snapshot 把「何時、從哪裡、哪些條文」變成可機器驗證的輸入契約，逐引用 provenance 則把這份契約延伸到答案展示。UI 只把 `https://law.moj.gov.tw` 連結渲染成可點擊網址，避免任意 payload URL 變成釣魚連結。

**Tradeoff**:法規一修訂，稽核會故意失敗，必須人工審閱差異、更新 snapshot、重建兩份索引並重跑評估。現有 Qdrant cloud collections 是舊 payload；在取得新的臨時 writer key 前仍相容但不會憑空出現新增的日期欄位。

## 11. 為什麼新增 60 題壓力集卻不取代 40 題正式集?

**選擇**:保留 `v0.1.0` 的 40 題／8 組消融／provider judge 結果作為 immutable formal baseline；另以 `v0.3.1 reliability stress evidence` 發布 40 可答、20 不可答、偏長句與中英夾雜的 retrieval/refusal 壓力結果，並用正式集當 regression guard。

**理由**:新資料可以針對已觀察到的口語問法盲點，但若直接把新舊題混成一個分數，讀者無法分辨提升來自系統變更還是題目組成變更。分版後可同時看到壓力集確實抓到 1/40 直接誤拒，以及正式集仍重現 0/30。門檻 sweep 必須同時在兩組資料不劣才可自動變更；本次沒有 Pareto-better 候選，因此保留 0.03。

**Tradeoff**:60 題仍是策展樣本，不能推估真實流量發生率；它只把「口語風格可能失敗」從單一 anecdote 提升為可重跑的明確測量。

## 12. 為什麼 provider cross-check 要先做硬預算，而不是跑完再算錢?

**選擇**:`BudgetLedger` 使用 `Decimal`，CLI cap 預設為 0 且不得高於每家 US$5 的本次授權；每次呼叫前對實際 system + user prompt 以 UTF-8 byte 數加 1,024-token message envelope 做保守上界，再連同最大 1,024 output tokens 計算最壞成本。請求 maxima 固定不得高於 20,000 input／1,024 output tokens，額度不足或 prompt 超界就不送出。Gemini 成本把 `candidatesTokenCount + thoughtsTokenCount` 都算輸出，OpenAI 使用回傳的 prompt／completion usage；缺欄位、負值或超出 maxima 都停止。

**理由**:事後統計只能描述已經花掉的錢，不能限制下一次呼叫。先各跑 5 題、確認模型與 usage 完整後才擴張，可同時驗證指定模型沒有 fallback，並把故障半徑限制在很小的初始批次。

**Tradeoff**:保守 maxima 會提早停止而留下未使用額度；這是刻意的安全偏向。公開 trace 只留 qid、provider/model、拒答、引用數、token、成本與 0/1 verdict，完整問題／答案只能進 ignored `eval/runs/`，任意 `--work-dir` 若逃出該目錄會在呼叫前被拒絕。`v0.3.2` 已完成兩家各五筆的 safety batch；公開 trace 仍嚴格不留 question/answer text、provider payload 或 credentials。

## 13. 為什麼升級 Transformers 5.x 但保留 FlagEmbedding？

**選擇**:`transformers>=5.5,<6` 避開已公告的 4.x 模型設定載入 RCE；BGE-M3 在 5.x 原生可用。模型名稱與 40 位 commit SHA 先由 `snapshot_download` 解析成不可變的本機 snapshot 路徑，再交給 FlagEmbedding，避免其 1.4 版吞掉 `revision` 後悄悄載入 mutable `main`。FlagEmbedding reranker 仍呼叫已移除的 `prepare_for_model`，因此本專案只對固定的 XLM-R tokenizer 補回等價的 pair special-token 組裝，遇到非完整 SHA、其他 tokenizer 或不支援的 truncation/padding 參數就 fail closed。

**驗證**:實際 snapshot 路徑末段分別等於 `5617a9f…b181` 與 `953dc6f…d41e`；固定 reranker snapshot 在 4.57.6 與 5.16.1 對同一組樣本產生完全相同的 normalized scores (`0.966139`, `0.000103`)；BGE-M3 embedding shape 仍為 1,024。完整測試與 `pip-audit --local` 均通過，沒有用 vulnerability ignore 取得綠燈。
