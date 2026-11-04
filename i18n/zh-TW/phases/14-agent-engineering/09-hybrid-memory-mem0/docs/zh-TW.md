# 混合 Memory：Vector + Graph + KV (Hybrid Memory: Vector + Graph + KV)

> 混合 memory 會平行執行三種 store：用 vector 做語意相似度、用 KV 做快速事實查詢、用 graph 做實體關係推理，再用一個計分層在 retrieval 時把它們融合。這是外部 memory 常見的正式環境模式；Mem0（Chhikara et al., 2025）是其中一個參考實作。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼單一 store（只有 vector、只有 graph、只有 KV）不足以支撐 agent memory。
- 說出 Mem0 的三個平行 store，以及各自最佳化的目標。
- 描述 Mem0 的融合計分（relevance、importance、recency），以及為什麼它是加權總和，而不是階層式。
- 用 stdlib 實作 toy 三 store memory，`add()` 寫入三者，`search()` 融合結果。

## 問題 (The Problem)

每種 store 只適合三類查詢中的一類：

- **語意相似度**：「上週我們討論 agent drift 時談了什麼？」Vector 勝出；KV 和 graph 會漏掉。
- **事實查詢**：「使用者的電話號碼是什麼？」KV 勝出；vector 太浪費，graph 太大材小用。
- **關係推理**：「哪些客戶共用同一個帳單實體？」Graph 勝出；vector 和 KV 都答不出來。

正式環境的 agent 會在同一個 session 裡發出這三種查詢。單一 store 的 memory 對其中兩種永遠是錯的。Mem0 的貢獻，是把三者接在同一個 `add`/`search` 介面後面，再加上一個融合它們的計分函式。

## 核心概念 (The Concept)

### 三個平行 store (Three stores in parallel)

Mem0（arXiv:2504.19413, April 2025）在 `add(text, user_id, metadata)` 時：

1. 從文字中擷取候選事實（由 LLM 驅動的步驟）。
2. 把每個事實寫進 vector store（embedding），供語意搜尋。
3. 把每個事實寫進 KV store，以 (user_id, fact_type, entity) 為 key，做 $O(1)$ 查詢。
4. 把每個事實以有型別的邊寫進 graph store（Mem0g），供關係查詢。

在 `search(query, user_id)` 時：

1. Vector store 依 embedding cosine 回傳 top-k。
2. KV store 回傳以查詢推導出的 (user_id, type, entity) 為 key 的直接命中。
3. Graph store 回傳從查詢實體出發可到達的子圖。
4. 計分層把三者融合。

### 融合計分 (Fusion scoring)

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance**：vector cosine、KV 精確比對、graph 路徑權重。
- **Importance**：寫入時標記或學習而得（有些事實更重要：姓名、ID、政策）。
- **Recency**：自上次寫入或讀取以來，隨時間指數衰減。

權重依產品調整。聊天 agent 的 `w_recency` 較高；合規 agent 的 `w_importance` 較高；retrieval agent 的 `w_relevance` 較高。

### Mem0g 與時間推理 (Mem0g and temporal reasoning)

Mem0g 加了一個衝突偵測器。當新事實與既有的邊矛盾時，既有的邊會被標記為無效但不刪除。時間性查詢（「三月時使用者住在哪個城市？」）會走訪在該時間點有效的子圖。

這就是合規等級的行為，Letta 的失效處理模式把它一般化了。

### Benchmark 數字 (Benchmark numbers)

Mem0 論文回報（2025）：

- **LoCoMo**（長篇對話 memory）：91.6
- **LongMemEval**（長時程 episodic memory）：93.4
- **BEAM 1M**（1M-token memory benchmark）：64.1

比較用的 baseline（full-context 128k LLM、扁平 vector store、扁平 KV）都落後 10 分以上。單看 benchmark 不足以決定選擇，營運形態才是，但這些數字顯示融合設計並不是四捨五入的誤差。

### Scope 分類 (Scope taxonomy)

Mem0 依 scope 切分 memory：

- **User memory**：跨 session 持久，以 `user_id` 為 key。
- **Session memory**：在單一 thread 內持久。
- **Agent memory**：每個 agent 實例的狀態。

每次寫入都要選一個 scope。Retrieval 可以跨 scope 查詢，並為各 scope 設定權重。不經思考就混用 scope，就會發生「助理跟 Alice 講了 Bob 的專案」這類事件。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Embedding drift。** 前一百次查詢看起來正確的 vector 結果，會隨語料成長而退化。要定期對最常使用的前 N 筆記錄重新 embedding。
- **KV schema 蔓延 (KV schema creep)。** `(user_id, type, entity)` 看起來很簡單，直到每個團隊都加上自己的 `type`。每季稽核一次 type 集合。
- **Graph 爆炸 (Graph explosion)。** 一個雜訊很大的擷取器，每則訊息會加 50 條邊。限制每次 `add` 呼叫的 graph 寫入數量；丟掉低信心的邊。

```figure
ae-memory-fusion
```

## 動手做 (Build It)

`code/main.py` 用 stdlib 實作三 store 模式：

- `VectorStore`：用簡單的 token 重疊相似度代替 embedding。
- `KVStore`：以 `(user_id, fact_type, entity)` 為 key 的 dict。
- `GraphStore`：有型別的邊 (subject, relation, object, valid)。
- `Mem0`：頂層 facade，提供 `add()`、`search()`、融合計分，以及感知 scope 的 retrieval。
- 一份在多使用者、多 session 對話上的實際 trace。

執行：

```
python3 code/main.py
```

輸出顯示三條獨立的召回路徑，加上融合後的 top-k。改動 `main()` 開頭的計分權重，觀察排名如何變化。

## 實際使用 (Use It)

- **Mem0 (Apache 2.0)**：可用於正式環境。用 Postgres + Qdrant + Neo4j 自架，或使用託管雲端。
- **Letta**：三層 core/recall/archival；vector 和 graph 後端自己帶。
- **Zep**：商用替代方案，具時間性 KG 與事實擷取。
- **自行打造**：當你需要精確控制擷取器（合規）或融合權重（recency 為主的語音 agent）時。

## 交付 (Ship It)

`outputs/skill-hybrid-memory.md` 產生三 store 的 memory 骨架，內含融合計分器、scope 分類和時間性失效處理。

## 練習 (Exercises)

1. 把 toy vector 相似度換成真正的 embedding 模型（sentence-transformers、Ollama、OpenAI embeddings）。在合成的長對話上量測 recall@10。排名在 1000 次寫入後會漂移嗎？
2. 加一個時間性查詢：`search(query, as_of=timestamp)`。只回傳在該時間點或之前有效的記錄。哪個 store 需要最多改動？
3. 實作衝突偵測器：如果進來的事實與某條 graph 邊矛盾，就讓舊邊失效並記錄兩者。用「使用者住在 Berlin」-> 「使用者住在 Lisbon」測試。
4. 把融合計分器加上 `user_feedback` 維度（對取回的記錄按讚）。如何防止被操弄（agent 只回傳它已經喜歡過的記錄）？
5. 讀 Mem0 文件（`docs.mem0.ai`）。把 toy 移植成 `mem0` client 呼叫。在同樣的 20 個測試查詢上比較 retrieval 品質。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Hybrid memory | 「Vector 加 graph 加 KV」 | 三個平行寫入的 store，在 retrieval 時融合 |
| Fact extraction | 「Memory 匯入」 | 把文字拆成 (entity, relation, fact) 元組的 LLM 步驟 |
| Fusion scoring | 「Relevance 排名」 | relevance、importance、recency 的加權總和 |
| Scope | 「Memory namespace」 | user / session / agent，決定誰看得到什麼 |
| Mem0g | 「Memory graph」 | 帶時間有效性、供關係查詢用的有型別邊 |
| Temporal invalidation | 「軟刪除」 | 把被推翻的邊標為無效；絕不刪除 |
| Embedding drift | 「Retrieval 腐化」 | 語料成長使 vector 品質下降；定期重新 embedding |

## 延伸閱讀 (Further Reading)

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413)：原始論文
- [Mem0 docs](https://docs.mem0.ai/platform/overview)：正式環境 API、SDK、託管雲端
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560)：虛擬 context 的前身
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks)：三層的姊妹設計
