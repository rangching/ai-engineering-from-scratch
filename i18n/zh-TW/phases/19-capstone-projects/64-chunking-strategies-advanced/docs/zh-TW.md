# Chunking 策略比較 (Chunking Strategies, Compared)

> Chunking 決定了你的 retriever 能夠撈出什麼。邊界切錯了，下游的 embedding 模型、reranker、LLM 都無法修復這個損害。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 lessons 04 (embeddings), 06 (RAG), 07 (advanced RAG); Phase 19 Track B foundations (lessons 20-29)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)
- 從零實作五種 chunking 策略：fixed-window、sentence、recursive-split、semantic clustering，以及結構化的 markdown 標題。
- 在帶有黃金標註答案區段的 fixture 語料上量測 recall@k，並解釋為何某個策略在散文上勝出，另一個策略在技術文件上勝出。
- 讀懂 chunk 長度分佈，並辨認每個策略帶入的失敗模式：孤兒句子、切在符號中間、只有標題的 chunk、語意漂移。
- 不跑 benchmark，只檢查三個性質就能為新語料選出預設值：文件類型、平均段落長度，以及格式是否帶有明確結構。

## 問題 (The Problem)

每個 RAG 管線都從把來源文件切成小塊開始，小到 embedding 模型放得下，大到每一塊都帶有自成一體的想法。要在哪裡下刀，不是超參數。它是 retriever 所能回傳內容的上限。

一個問「預算中止門檻長什麼樣子」的查詢，只有在存放中止門檻的 chunk 找得到時才會成功。如果 fixed-window 切分器把門檻值從周圍的脈絡切開了，embedding 會移到另一個群集，BM25 分數下降，reranker 看到的是雜訊，LLM 生成的答案就是錯的。2024 年的論文 "LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs" 量到，單單 chunking 的選擇就造成 35 個百分點的檢索 recall 絕對落差。2025 年關於 contextual chunk header 的後續研究縮小了差距，但沒有消除。

本課並排建立五種策略，在帶有黃金標註答案區段的 fixture 語料上執行，讓你自己讀 recall 數字。

## 核心概念 (The Concept)

```mermaid
flowchart LR
  Doc[Source Document] --> S1[Fixed Window]
  Doc --> S2[Sentence]
  Doc --> S3[Recursive Split]
  Doc --> S4[Semantic Cluster]
  Doc --> S5[Structural Markdown]
  S1 --> Chunks1[Chunks]
  S2 --> Chunks2[Chunks]
  S3 --> Chunks3[Chunks]
  S4 --> Chunks4[Chunks]
  S5 --> Chunks5[Chunks]
  Chunks1 --> Index[Embedding Index]
  Chunks2 --> Index
  Chunks3 --> Index
  Chunks4 --> Index
  Chunks5 --> Index
  Index --> Eval[Recall@k vs Gold Spans]
```

### Fixed-window

暴力的 baseline。每 N 個字元切一刀。可選擇性地重疊，讓在位置 N 被切斷的句子，完整出現在從位置 N - overlap 開始的 chunk 裡。快速、確定性，但在邊界上很糟。把它當作對照組，而不是預設值。

### Sentence

用 regex 或簡單的狀態機依句子邊界切分。把一個或多個句子打包進 chunk，直到目標字元預算。不會再切在字詞中間。仍然會切在段落中間和章節中間。這是許多早期 RAG 管線的預設值，對沒有其他結構的散文是合理的選擇。

### Recursive split

這是 2023 年代函式庫所普及的階層式策略。先試著用最強的分隔符號切（雙換行、段落），退而求其次用下一個（單換行），再到句子，最後到字元。當 chunk 符合預算時，遞迴終止。對結構不一致的文件很強，因為它會依區域調整。

### Semantic clustering

對每個句子做 embedding。把共享同一個主題 centroid 的連續句子聚成一群。每當與 centroid 的累計相似度掉到門檻以下，就切一刀。邊界反映的是意義，而不是字元。建置較慢，且依賴 embedding 模型，但對於在段落內切換主題的文件很有韌性。

### 結構化的 markdown 標題 (Structural markdown headers)

對於帶有明確結構的文件（markdown、reStructuredText、RFC 風格的編號章節），在標題邊界切分。每個 chunk 成為標題，加上其下一直到同級或更高級的下一個標題為止的所有內容。每個主題的 chunk 最小，但只有在語料格式良好時才能使用。

### Recall@k 如何衡量邊界的選擇 (How recall@k measures the boundary choice)

黃金標註的查詢帶有答案區段在來源文件內的確切字元位移。Chunking 之後，你問：retriever 回傳的前 k 個 chunk 中，有沒有任何一個與黃金區段重疊？如果有，該查詢的 recall@k 為 1。沒有則為 0。對整個查詢集取平均。對每個策略執行相同的評估，差距就會顯示哪個邊界政策在你手上的語料上存活下來。

```figure
ci-chunk-boundaries
```

## 動手做 (Build It)

`code/main.py` 實作了：

- `fixed_window(text, size, overlap)` - baseline。
- `sentence_chunks(text, target)` - 簡單的句子打包器。
- `recursive_split(text, separators, target)` - 階層式遞迴。
- `semantic_chunks(text, similarity_threshold)` - 建立在確定性 mock embedding 之上、以 centroid 為基礎的聚類。
- `structural_markdown(text)` - 感知標題的切分器。
- `mock_embed(text, dim)` - 基於雜湊的 embedding，讓迴圈能離線執行。
- `DenseIndex` - 與 Phase 19 Track B 混合檢索課程中使用的相同形狀。
- `eval_recall(strategy, corpus, queries, k)` - 比較迴圈。
- 一個 `main()`，在 fixture 語料上執行每個策略，並印出 recall@k 表格。

執行它：

```bash
python3 code/main.py
```

輸出是一張小表格，每個策略一列，每個 k 一欄。Sentence 在結構化的 fixture 上輸了。Structural-markdown 在 markdown fixture 上勝出。Recursive 在混合的 fixture 上表現穩健，因為遞迴會適應。Semantic clustering 在沒有有用結構線索的散文 fixture 上勝出。

## 這張表藏不住的失敗模式 (Failure modes the table will not hide)

**孤兒句子。** Sentence 打包產生的 chunk 缺少主題句。Embedding 於是指向錯誤的群集。

**切在符號中間。** 在程式碼或 YAML 內使用 fixed-window，會把識別字切成兩半。兩半都 embed 成雜訊。

**只有標題的 chunk。** Structural markdown 會輸出只含 `## Title` 的 chunk。把它們過濾掉，或附上下一個 chunk 的第一段。

**語意漂移。** 當語料整體都在主題上時，semantic clustering 會切得不夠。5000 字元的 chunk 把許多具體答案塞進一個擴散的 embedding。把 semantic 與硬性字元上限結合。

**過時的 embedding。** Semantic clustering 使用 embedding 模型。如果你更換模型，chunk 也會跟著改變。把 chunk 用的模型與檢索用的模型分開固定，或一起重建 index。

## 不跑 benchmark 就選預設值 (Choosing a default without running the benchmark)

三個性質決定新語料的預設 chunker。

| 性質 | 值 | 預設 |
|------|----|------|
| 文件類型 | 沒有結構的散文 | Recursive split，目標 800 |
| 文件類型 | Markdown / RFC / API 文件 | Structural markdown |
| 文件類型 | 程式碼 | 感知 AST（超出範圍；見 Phase 19 lesson 02） |
| 段落長度 | 長、單一主題 | Sentence，目標 500 |
| 段落長度 | 短、混合主題 | Semantic，門檻 0.6 |

有疑問的時候，選 recursive split。它是最強的單一策略 baseline。

## 實際使用 (Use It)

正式環境的做法：

- 在發佈新管線之前先跑評估；不要相信你的函式庫預設的策略。
- 每當你更換 embedding 模型或語料組合時，重新跑評估；贏家取決於語料。
- 把策略名稱存在每個 chunk 的 metadata 裡，之後才能歸因退化。

## 交付 (Ship It)

第 69 課的 Track F 端到端 RAG 系統，使用這裡選出的 chunker 作為第一階段。第 68 課的評估 harness 從本課 `eval_recall` 回傳的同樣形狀讀取 recall@k。挑選在你的語料上勝出的策略，並往後傳遞。

## 練習 (Exercises)

1. 新增第六種策略：使用 `tiktoken` 而不是字元計數的 token-window。在同一個 fixture 上與 fixed-window 比較。
2. 在散文 fixture 中注入 30% 的程式碼區塊。重新執行表格。解釋為什麼除了 structural markdown 之外，每個策略的 recall 都下降。
3. 把確定性 embedding 換成你專案真正的 provider 所提供的。量測 semantic-clustering 的 recall 差異。回報各策略之間的差距是變寬還是變窄。
4. 為每個 chunk 新增 `summary` 欄位：一句話的 centroid 描述。把摘要附加到 chunk 本文後重新執行評估。量測 recall 的提升。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Recall@k | 「我們有拿到正確的 chunk 嗎？」 | 前 k 個 chunk 中任何一個與黃金答案區段重疊的查詢所占的比例 |
| Chunk overlap | 「滑動視窗」 | 把前一個 chunk 的最後 N 個字元重新放進下一個 chunk |
| Structural splitter | 「感知標題的 chunk」 | 在 H1/H2/H3 邊界切分；標題文字是 chunk 的一部分 |
| Semantic chunker | 「感知主題的 chunk」 | 對句子做 embedding，依 centroid 相似度聚類，在漂移時切分 |
| Centroid drift | 「主題轉移」 | 累計平均與下一個句子之間的 cosine 相似度掉到門檻以下 |

## 延伸閱讀 (Further Reading)

- [LongRAG: Enhancing Retrieval-Augmented Generation with Long-context LLMs (arXiv 2406.15319)](https://arxiv.org/abs/2406.15319)
- [Anthropic, Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval)
- [LlamaIndex, Chunking strategies for production RAG](https://docs.llamaindex.ai/en/stable/optimizing/production_rag/)
- Phase 11 lesson 06 - RAG 基礎
- Phase 11 lesson 07 - 進階 RAG
- Phase 19 lesson 65 - 對這裡產生的 chunk 排名的混合檢索
- Phase 19 lesson 68 - 在正式環境中為策略選擇評分的評估 harness
