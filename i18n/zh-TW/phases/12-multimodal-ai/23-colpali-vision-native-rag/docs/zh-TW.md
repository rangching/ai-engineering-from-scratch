# ColPali 與 Vision-Native 文件 RAG (ColPali and Vision-Native Document RAG)

> 傳統 RAG 把 PDF 解析成文字、切成 chunk、對 chunk 做 embedding、儲存向量。每一步都會損失訊號：OCR 丟掉圖表資料、切塊打斷表格列、文字 embedding 忽略圖片。ColPali (Faysse et al., 2024 年 7 月) 問了更簡單的問題：為什麼要擷取文字？直接透過 PaliGemma 對頁面圖片做 embedding，使用 ColBERT 風格的 late interaction 做檢索，並保留文件攜帶的所有版面、圖片、字型與格式訊號。已發表的 benchmark：在視覺豐富的文件上，端到端準確率比文字 RAG 好 20-40%。ColQwen2、ColSmol 與 VisRAG 延伸了這個模式。這堂課讀 vision-native RAG 的論點，並建立一個迷你的類 ColPali 索引器。

**Type:** Build
**Languages:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Prerequisites:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 解釋 bi-encoder 檢索（每份文件一個向量）與 late-interaction 檢索（每份文件多個向量）的差別。
- 描述 ColBERT 的 MaxSim 運算，以及 ColPali 如何把它從文字 token 推廣到圖片 patch。
- 建立迷你的類 ColPali 索引器：頁面 → patch embedding → 對 query-term embedding 做 MaxSim → top-k 頁面。
- 在發票 / 財報的使用情境上，比較 ColPali + Qwen2.5-VL 生成器與文字 RAG + GPT-4。

## 問題 (The Problem)

PDF 上的文字 RAG 丟掉了文件的大部分內容。財報的第三季營收成長通常在圖表裡；醫療報告的發現在有註記的圖片中；法律合約的簽名區塊是版面上的事實，不是文字上的事實。

文字 RAG 管線：

1. PDF → 透過 OCR / pdftotext 得到文字。
2. 文字 → 300-500 個 token 的 chunk。
3. Chunk → bi-encoder embedding（一個向量）。
4. 使用者 query → embedding → cosine similarity → top-k chunk。
5. Chunk + query → LLM。

五個有損的步驟。圖表沒被捕捉。表格被切到不同 chunk。多欄版面被壓平。圖片註記消失。

ColPali 的解法：跳過 OCR，直接對頁面圖片做 embedding。使用 ColBERT 風格的 late interaction 做檢索，讓模型在 query 時能關注細粒度的 patch。

## 核心概念 (The Concept)

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) 是一種文字檢索方法。它不是每份文件一個向量，而是每個 token 產生一個向量。在 query 時：

- Query token 得到自己的 embedding（$N_q$ 個向量）。
- 文件 token 得到 embedding（$N_d$ 個向量，通常會快取）。
- 分數 = 對所有 query token，各取其在文件 token 上 cosine similarity 的最大值，再加總：$\sum_i \max_j \cos(q_i, d_j)$。

這就是 MaxSim 運算。每個 query token「挑出」與它最匹配的文件 token。最後的分數是加總。

優點：召回強，能處理詞彙層級的語意。缺點：每份文件有 $N_d$ 個向量，儲存昂貴。

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) 把 ColBERT 的模式套用到圖片上。

- 每一頁由 PaliGemma (ViT + 語言) 編碼成 patch embedding：每頁 $N_p$ 個向量。
- 每個使用者 query（文字）被編碼成 query-token embedding：$N_q$ 個向量。
- 分數 = $\sum_i \max_j \cos(q_i, p_j)$，也就是 query 文字 token 與頁面圖片 patch 之間的 MaxSim。
- 依總分取出 top-k 頁面。

文件輸入時：用 PaliGemma 對每頁做 embedding，儲存所有 patch embedding。Query 時：對 query token 做 embedding，對所有儲存的頁面 embedding 計算 MaxSim，回傳 top-k 頁面。

優點：在視覺豐富的文件上，端到端勝過文字 RAG 20-40%。每個 patch 向量都捕捉了局部版面與內容。

缺點：每頁 $N_p$ 個 patch × 4 byte 浮點數 × $D$ 維向量，儲存成長很快。可用 PQ / OPQ 量化減緩。

### ColQwen2 與 ColSmol

ColQwen2 (illuin-tech, 2024-2025) 把 PaliGemma 換成 Qwen2-VL。更好的基礎 encoder，更好的檢索。

ColSmol 是用於本機 / 邊緣的較小規模變體。約 10 億參數的 ColSmol 檢索器可以在消費級 GPU 上執行。

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) 是不同的變體：不對 patch 做 MaxSim，而是用 VLM 把每一頁 pool 成單一向量，再用 bi-encoder 檢索。索引更快、儲存更小，但召回較弱。

品質對成本的取捨：要品質用 ColPali，要規模用 VisRAG。

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) 把 multi-modal 檢索擴展到多頁、多文件推理。跨文件檢索頁面，為 VLM 組合出多頁 context。

### ViDoRe：benchmark

ColPali 的配套 benchmark。Visual Document Retrieval Evaluation。任務包括財報、科學論文、行政文件、病歷、手冊。指標：nDCG@5。

ColPali-v1 在 ViDoRe 上的 nDCG@5 約 80%；同樣文件上的文字 RAG 約 50-60%。

### 端到端的 RAG 管線 (The end-to-end RAG pipeline)

對 vision-native RAG：

1. 輸入：PDF → 頁面圖片 → PaliGemma 編碼 → 儲存所有 patch embedding。
2. Query：使用者文字 → query-token embedding → 對所有已索引頁面做 MaxSim → top-k 頁面。
3. 生成：top-k 頁面圖片 + query → VLM（Qwen2.5-VL 或 Claude）→ 答案。

全程沒有 OCR。圖片、圖表、字型、版面全都流進答案。

### 儲存量計算 (Storage math)

一份 50 頁的財報，每頁 729 個 patch、128 維 embedding：

- ColPali：50 * 729 * 128 * 4 bytes = 原始約 18 MB，8 倍 PQ 後約 2.3 MB。
- 文字 RAG：50 個 chunk * 768 維 * 4 bytes = 約 150 kB。

每份文件的原始儲存量，ColPali 約是 120 倍。大規模時，8 倍 OPQ / PQ 把它降到約 15 倍，通常可以接受。

### 文字 RAG 仍然勝出的情況 (When text-RAG still wins)

- 沒有版面訊號的純文字文件（wiki 文章、聊天紀錄）。文字 RAG 更簡單，儲存也更便宜。
- 儲存成本主導的數百萬頁檔案庫。
- 要求在檢索之外還必須提供可擷取 OCR 文字的嚴格法規。

2026 年的其他一切，包括財報、科學論文、法律合約、病歷、UX 文件，vision-native RAG 都勝出。

```figure
mm-maxsim
```

## 實際使用 (Use It)

`code/main.py`：

- 玩具 patch encoder：把「頁面」（小型的特徵向量網格）對應到一個 patch embedding 陣列。
- MaxSim 評分器：計算 query token embedding 集合與頁面 patch 集合之間的 ColBERT 風格分數。
- 索引 5 個玩具頁面，執行 3 個 query，回傳附分數的 top-k。

## 交付 (Ship It)

這堂課產出 `outputs/skill-vision-rag-designer.md`。給定文件 RAG 專案，它會挑選 ColPali / ColQwen2 / VisRAG / 文字 RAG，並估算儲存量。

## 練習 (Exercises)

1. 一份 200 頁的年報，每頁 729 個 patch、128 維 embedding、4 byte 浮點數。計算原始儲存量與 PQ 壓縮（8 倍）後的儲存量。

2. MaxSim 是 $\sum_i \max_j \cos(q_i, p_j)$。這個加總捕捉到什麼，是單純的平均相似度捕捉不到的？

3. ColPali 以 patch 集合為單位索引頁面。如果改成在詞的層級索引（像 ColBERT 那樣）會有什麼改變？有哪些取捨？

4. 為 100 萬頁語料庫設計端到端管線，每個 query 的延遲預算為 500ms。在 ColQwen2 / VisRAG 之間做選擇並說明理由。

5. 閱讀 M3DocRAG (arXiv:2411.04952)。描述其多頁 attention 模式，以及它與單頁 ColPali 檢索有何不同。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | 使用每個 token 或每個 patch 的 embedding + MaxSim 做檢索，而不是單一文件向量 |
| MaxSim | "Max-over-patches" | 對每個 query token，挑出相似度最高的文件 token；再對 query 加總 |
| Bi-encoder | "Single-vector" | 每份文件一個向量；較快但損失細粒度 |
| Multi-vector | "Many-vectors-per-doc" | 每份文件 / 頁面儲存 $N_p$ 個向量；儲存成本增加但召回提升 |
| Patch embedding | "Page feature" | 來自 VLM encoder、每個圖片 patch 一個向量，每頁都會快取 |
| ViDoRe | "Vision doc bench" | ColPali 用於視覺文件檢索的 benchmark 套件 |
| PQ quantization | "Product quantization" | 在維持向量相似度的同時，把儲存量縮小約 8 倍的壓縮方法 |

## 延伸閱讀 (Further Reading)

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
