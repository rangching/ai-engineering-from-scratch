# Capstone 02 — RAG over Codebase (跨 Repo 語意搜尋)

> 2026 年，每個認真的工程組織都跑著一套懂語意、不只比對字串的內部程式碼搜尋。Sourcegraph Amp、Cursor 的 codebase answers、Augment 的企業級 graph、Aider 的 repomap、Pinterest 的內部 MCP，形狀都一樣：匯入許多 repo、用 tree-sitter 解析、對函式與類別層級的 chunk 做 embedding、混合搜尋、re-rank，最後附上引用來回答。這個 capstone 要你做出一套能處理 10 個 repo、共 2M 行程式碼，並且在每次 git push 時都能增量重建索引的系統。

**Type:** Capstone
**Languages:** Python (ingestion), TypeScript (API + UI)
**Prerequisites:** Phase 5 (NLP foundations), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P17
**Time:** 30 小時

## 問題 (Problem)

到了 2026 年，每個前沿 coding agent 都內建 codebase 檢索層，因為光靠 context window 解決不了跨 repo 的問題。Claude 的 1M-token context 有幫助，但不能取代排序過的檢索。對原始 chunk 做單純的 cosine 搜尋，會在產生的程式碼、monorepo 的重複內容，以及很少被 import 的長尾 symbol 上污染結果。正式環境的做法，是對具備 AST 概念的 chunk 做混合 (dense + BM25) 搜尋，加上 re-ranker，並由 symbol 參照 graph 支撐。

你要靠索引一整批真實的 repo 來學這件事，不是只有一個教學用 repo，然後衡量 MRR@10、引用忠實度和增量更新的新鮮度。失敗模式都在基礎建設上：10 萬個檔案的 monorepo、一次 push 動到一半的檔案、一個得跨四個 repo 才能正確回答的查詢。

## 核心概念 (Concept)

具備 AST 概念的匯入管線，用 tree-sitter 解析每個檔案，擷取函式與類別節點，並依節點邊界切 chunk，而不是用固定的 token 視窗。每個 chunk 有三種表示：dense embedding (Voyage-code-3 或 nomic-embed-code)、sparse BM25 詞項，以及一段簡短的自然語言摘要。摘要提供第三種可檢索的模態：使用者問「X 是怎麼授權的」，摘要會提到「authz」，即使程式碼裡只有 `check_permission`。

檢索是混合式的。一個查詢同時觸發 dense 與 BM25 搜尋，合併 top-k，把聯集交給 cross-encoder re-ranker (Cohere rerank-3 或 bge-reranker-v2-gemma-2b)。重新排序後的清單送到長 context 的合成器 (啟用 prompt caching 的 Claude Sonnet 4.7，或自行架設的 Llama 3.3 70B)，並要求每個主張都以檔案與行號範圍引用。沒有引用的答案會被後置過濾器拒絕。

增量更新的新鮮度是基礎建設問題。Git push 會觸發 diff：哪些檔案變了、哪些 symbol 變了。只有受影響的 chunk 會重新 embedding。受影響的跨檔案 symbol 邊 (imports、method calls) 會重新計算。索引保持一致，不必在每次 commit 都重處理 2M 行。

## 架構 (Architecture)

```
git push --> webhook --> ingest worker (LlamaIndex Workflow)
                           |
                           v
             tree-sitter parse + AST chunk
                           |
            +--------------+----------------+
            v              v                v
          dense        BM25 index       summary (LLM)
        (Voyage / bge)  (Tantivy)        (Haiku 4.5)
            |              |                |
            +------> Qdrant / pgvector <----+
                            |
                            v
                      symbol graph (Neo4j / kuzu)
                            |
  query --> LangGraph agent (retrieve -> rerank -> synth)
                            |
                            v
                 Claude Sonnet 4.7 1M context
                            |
                            v
                 answer + file:line citations
```

## 技術堆疊 (Stack)

- 解析：tree-sitter，搭配 17 種語言 grammar (Python、TS、Rust、Go、Java、C++ 等)
- Dense embeddings：Voyage-code-3 (託管) 或 nomic-embed-code-v1.5 (自行架設)，bge-code-v1 作為備援
- Sparse 索引：Tantivy (Rust) 搭配 BM25F，對 symbol 名稱與內文做欄位加權
- 向量資料庫：Qdrant 1.12 的混合搜尋，或給 5000 萬以下向量團隊用的 pgvector + pgvectorscale
- Chunk 摘要模型：Claude Haiku 4.5 或 Gemini 2.5 Flash，啟用 prompt cache
- Re-ranker：Cohere rerank-3 或自行架設的 bge-reranker-v2-gemma-2b
- 編排：匯入用 LlamaIndex Workflows，查詢 agent 用 LangGraph
- 合成器：啟用 prompt caching 的 Claude Sonnet 4.7 (1M context)
- Symbol graph：Neo4j (託管) 或 kuzu (內嵌)，存放 import 與 call 邊
- 可觀測性：每個檢索與合成步驟都有 Langfuse spans

```figure
ce-hybrid-retrieval
```

## 動手做 (Build It)

1. **匯入 walker。** 在每次 push hook 時走訪 git 歷史，收集有變動的檔案。對每個檔案用 tree-sitter 解析，擷取函式與類別節點及其完整原始碼範圍。輸出 chunk 紀錄 `{repo, path, start_line, end_line, symbol, body}`。

2. **Chunk 摘要器。** 把 chunk 批次送進 Haiku 4.5 呼叫，system preamble 啟用 prompt caching。Prompt：「用一句話摘要這個函式，點出它的公開契約與副作用。」把摘要和 chunk 一起存。

3. **Embedding 池。** 兩條平行佇列：dense (Voyage-code-3，batch 128) 與摘要 (同一個模型，但對摘要字串做)。把向量寫入 Qdrant，payload 為 `{repo, path, start_line, end_line, symbol, kind}`。

4. **BM25 索引。** 欄位加權的 Tantivy 索引：symbol 名稱權重 4、symbol 內文權重 1、摘要權重 2。除了「找出做 X 的函式」，也支援「找出名為 X 的函式」這類查詢。

5. **Symbol graph。** 對每個 chunk 記錄邊：imports (這個檔案使用 repo Z 的 symbol Y)、calls (這個函式呼叫類別 C 的方法 M)、繼承。存進 kuzu。查詢時用來跨 repo 邊界擴展檢索。

6. **查詢 agent。** 三個節點的 LangGraph。`retrieve` 平行觸發 dense + BM25，依 (repo, path, symbol) 去重。`rerank` 對 top-50 跑 cross-encoder，保留 top-10。`synth` 呼叫 Claude Sonnet 4.7，把重新排序的 chunk 放進 context，快取 system prompt，並要求 file:line 引用。

7. **強制引用。** 解析模型輸出；任何沒有 `(repo/path:start-end)` 錨點的主張，都會被標記並重新詢問，或直接丟棄。只把有引用的答案回傳給使用者。

8. **增量重建索引。** 每次 webhook 時計算 symbol 層級的 diff。只有文字變動的 chunk 會重新 embedding。import 變動的 chunk 會重新計算 symbol 邊。衡量目標：對 2M-LOC 的 repo 群，50 個檔案的 push 在 60 秒內完成重建索引。

9. **評測。** 標註 100 個跨 repo 問題與標準 file:line 答案。衡量 MRR@10、nDCG@10、引用忠實度 (有可驗證錨點的主張比例)，以及 p50/p99 延遲。

## 實際使用 (Use It)

```
$ code-rag ask "how is S3 multipart abort wired into our retry budget?"
[retrieve]  12 chunks dense + 7 chunks bm25, 16 unique after dedup
[rerank]    top-5 kept (cohere rerank-3)
[synth]     claude-sonnet-4.7, cache hit rate 68%, 2.1s
answer:
  Multipart aborts are triggered by `AbortMultipartOnFail` in
  services/uploader/retry.go:122-148, which decrements the per-bucket
  retry budget defined in config/budgets.yaml:34-51 ...
  citations: [services/uploader/retry.go:122-148, config/budgets.yaml:34-51,
              libs/s3client/multipart.ts:44-61]
```

## 交付 (Ship It)

交付的 skill 是 `outputs/skill-codebase-rag.md`。給定一批 repo，它會架起匯入管線、混合索引與查詢 agent，並對任何跨 repo 問題回傳有引用的答案。評分標準：

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 檢索品質 | 在 100 題 held-out 集合上的 MRR@10 與 nDCG@10 |
| 20 | 引用忠實度 | 答案主張中有可驗證 file:line 錨點的比例 |
| 20 | 延遲與規模 | 在已索引資料量下，10k QPS 的 p95 查詢延遲 |
| 20 | 增量索引正確性 | 50 個檔案的 commit，從 git push 到可搜尋的時間 |
| 15 | UX 與答案格式 | 引用可點擊、片段預覽、後續追問的提示 |
| **100** | | |

## 練習 (Exercises)

1. 把 Voyage-code-3 換成自行架設的 nomic-embed-code。衡量 MRR@10 的差距，回報啟用 re-ranking 後差距是否縮小。

2. 在資料集中注入 20% 的產生程式碼 (LLM 產生的 boilerplate)，重新評測。觀察檢索污染。在 payload 加上 "generated" 旗標，並降低這些命中的權重。

3. 在你的資料量下，對 Qdrant 混合搜尋與 pgvector + pgvectorscale 做 benchmark。回報 batch size 1 的 p99。

4. 加入以抽樣為基礎的漂移檢查：每週重跑一次 100 題評測。MRR@10 下降超過 5% 就發出警示。

5. 擴充到跨語言的 symbol 解析：一個 Python 函式透過 gRPC 呼叫 Go 服務。用 symbol graph 把兩者連起來。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| AST-aware chunking | 「函式層級切分」 | 依 tree-sitter 節點邊界切程式碼，而不是用固定的 token 視窗 |
| Hybrid search | 「Dense + sparse」 | 平行執行 BM25 與向量搜尋，合併 top-k，再 rerank |
| Cross-encoder rerank | 「第二階段排序」 | 把每個 (query, candidate) 配對一起評分的模型，比 cosine 更準 |
| Prompt caching | 「快取的 system prompt」 | 2026 年 Claude / OpenAI 的功能，重複前綴 token 最多可折扣 90% |
| Symbol graph | 「程式碼 graph」 | 跨檔案與跨 repo 的 imports、calls、繼承邊 |
| Citation faithfulness | 「有依據的答案比例」 | 使用者點擊錨點並閱讀所指範圍後，能驗證的主張比例 |
| Incremental re-index | 「Push 到可搜尋的時間」 | 從 git push 到變動的 symbol 可被查詢的實際經過時間 |

## 延伸閱讀 (Further Reading)

- [Sourcegraph Amp](https://ampcode.com) — 正式環境的跨 repo 程式碼智慧
- [Sourcegraph Cody RAG architecture](https://sourcegraph.com/blog/how-cody-understands-your-codebase) — 這個 capstone 的參考深入解析
- [Aider repo-map](https://aider.chat/docs/repomap.html) — tree-sitter 排序的 repo 檢視
- [Augment Code enterprise graph](https://www.augmentcode.com) — 商用的 symbol-graph RAG
- [Qdrant hybrid search docs](https://qdrant.tech/documentation/concepts/hybrid-queries/) — 參考實作
- [Voyage AI code embeddings](https://docs.voyageai.com/docs/embeddings) — Voyage-code-3 細節
- [Cohere rerank-3](https://docs.cohere.com/reference/rerank) — cross-encoder 參考
- [Pinterest MCP internal search](https://medium.com/pinterest-engineering) — 內部平台參考
