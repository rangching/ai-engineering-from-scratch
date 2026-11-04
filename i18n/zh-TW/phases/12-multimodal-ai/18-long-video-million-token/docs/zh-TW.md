# 百萬 Token Context 下的長影片理解 (Long-Video Understanding at Million-Token Context)

> 一支 1 小時、4K、24 FPS 的影片，經過 patch 化與 embedding，會產生約 6000 萬個 token。一集 2 小時的 podcast 轉成逐字稿是 3 萬個 token。一部完整的藍光電影，即使用積極的 pooling 壓縮，也有數十萬個 token。Google 的 Gemini 1.5 (2024 年 3 月) 以 1000 萬 token 的 context 開啟了這個時代，在長達一小時的影片上做到可靠的 needle-in-a-haystack 召回。LWM (Liu et al., 2024 年 2 月) 展示了 ring attention 的擴展路徑。LongVILA 和 Video-XL 進一步擴大了輸入量。VideoAgent 則用 agentic retrieval 取代原始 context。每種做法在運算、召回與工程複雜度上都有不同的取捨。這堂課將它們並排來看。

**Type:** Build
**Languages:** Python (stdlib, needle-in-haystack simulator + agentic-retrieval router)
**Prerequisites:** Phase 12 · 17 (video temporal tokens)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 計算在不同 FPS 與 pooling 下，長影片的 visual token 總數。
- 解釋四條擴展路徑：暴力 context (Gemini 1.5)、ring attention (LWM)、token 壓縮 (LongVILA / Video-XL)、agentic retrieval (VideoAgent)。
- 在準確率與延遲上，比較原始 context 的 video VLM 與 agentic-retrieval 的 video VLM (VideoAgent)。
- 為 30 分鐘的影片設計 needle-in-a-haystack 測試，並量測在特定分鐘的召回率。

## 問題 (The Problem)

在 384 原生解析度下，Qwen2.5-VL 規模的單一影格 patch 約為 729 個 token。做 3x3 pooling 後是每影格 81 個 token。一段 30 分鐘、1 FPS 的片段 = 1800 個影格 = 145,800 個 token。2025 年的開源 VLM 做得到，但很吃緊。在 2 FPS 下是 291,600 個 token，只有最大的 context 才放得下。

一部 2 小時、1 FPS 的電影是 58.3 萬個 token。超出大多數 2026 年開源模型的能力；需要 Gemini 2.5 Pro，或更積極的 pooling。

出現了四條擴展路徑。

## 核心概念 (The Concept)

### 路徑 1：暴力 context (Gemini 1.5、Claude Opus)

用硬體硬幹。把 context 擴展到數百萬 token，在一次 forward pass 中處理所有東西。

Gemini 1.5 Pro 推出時是 100 萬 token；Gemini 1.5 Ultra 到 1000 萬；2026 年的 Gemini 2.5 Pro 可以可靠地處理數小時的影片。論文 (arXiv:2403.05530) 記錄了在最多約 950 萬 token 內，needle-in-a-haystack 召回率為 99.7%。

工程：自訂的 attention 實作，搭配記憶體階層（局部 + 全域 + 稀疏），再加上 MoE 專家路由以提升長 context 的效率。沒有完整公開細節。非開源。

### 路徑 2：Ring attention (LWM、LongVILA)

Ring attention 把長序列分散到多個裝置上，排成「環」狀，每個裝置持有一個區塊。整條序列的 attention 是這樣進行的：每個裝置依環狀模式把自己的區塊送給下一個，計算部分 attention，再彙總。

LWM (Liu et al., 2024) 就是這樣訓練出 100 萬 token context 的模型。每個裝置的記憶體隨其持有的區塊大小而變，所以增加裝置大致能線性延伸 context。整體 attention 運算量仍然是 context 的二次方；ring 把這些工作分散到各裝置，並讓 key/value 的傳遞與運算重疊。

LongVILA (arXiv:2408.10188) 把這個模式調整給 VLM 使用。1400 個影格、每影格 192 個 token = 268k context，以 8 路平行的 ring attention 訓練。

### 路徑 3：Token 壓縮 (Video-XL、LongVA)

比暴力 context 便宜：在 LLM 看到序列之前先積極壓縮。

Video-XL (arXiv:2409.14485) 使用 visual summary token：每個 N 影格的片段產生一個「摘要」token，它對這 N 個影格做 attention。推論時，LLM 每個片段只看到一個摘要 token，大幅縮小 context。

LongVA 以「long context transfer」技術把 LLM context 從 20 萬擴展到 200 萬。在長 context 文字上訓練，再透過共用表示遷移到長 context 影片。

Token 壓縮用特定時間戳的召回率換取可擴展性。模型大致知道發生了什麼，但有時會漏掉確切的影格。

### 路徑 4：Agentic retrieval (VideoAgent)

不要把整段影片餵給 LLM。反而把影片當作資料庫，用 LLM 來查詢它。

VideoAgent (arXiv:2403.10517)：

1. LLM 讀取問題。
2. LLM 向檢索工具詢問相關片段（「給我有貓的片段」）。
3. 工具回傳符合的片段時間戳。
4. LLM 透過 VLM 讀取這些片段。
5. LLM 組合出答案，或提出後續查詢。

這是把 LLM-as-agent 模式套用到長影片。推論更便宜（只有相關片段被編碼），工程更困難（檢索品質成為瓶頸）。

### Needle-in-a-haystack benchmark

標準的長 context 測試：在影片中的隨機位置插入獨特的視覺或文字標記，然後問一個需要回憶它的查詢。

指標：跨影片長度與標記位置的 Recall@k。

Gemini 2.5 Pro 在最長 90 分鐘的影片上召回率 >99%。開源 72B 模型（Qwen2.5-VL-72B、InternVL3-78B）在 30 分鐘時約 85-90%，超過 60 分鐘就退化。

VideoAgent 在 2 小時以上可以追平或勝過原始 context 的模型，因為只要工具夠好，檢索就能找到那根針。

### 該選哪條路 (Which path to pick)

15 分鐘的片段要達到前沿準確率：開源 72B + 原生 context 通常行得通。選 Qwen2.5-VL-72B。

30 分鐘到 1 小時的內容：開源用 LongVILA 或 Video-XL；封閉用 Gemini 2.5 Pro。品質門檻很重要，前沿就選封閉。

2 小時以上的內容：VideoAgent 或類似的檢索模式。或者，摘要成較小的區塊，再餵入階層式摘要。

### 2026 年的 production 模式 (2026 production pattern)

實務上，production 的長影片管線是混合式的：

1. 對整段影片做 dynamic-FPS 取樣 + 積極的 pooling（得到 10 萬 token 的全域表示）。
2. 交給 72B VLM 產生全域摘要。
3. 如果使用者問細節問題，以該摘要作為索引執行 agentic retrieval。

這結合了用於全域理解的暴力 context，以及用於局部細節的檢索。

```figure
mm-video-token-budget
```

## 實際使用 (Use It)

`code/main.py`：

- 計算從 1 分鐘到 3 小時的影片，在不同 FPS + pooling 下的 token 預算。
- 模擬 needle-in-a-haystack 執行：在隨機時間戳注入標記，提出問題，為召回評分。
- 包含 agentic-retrieval 路由器模擬器，挑出特定片段餵給下游 VLM。

執行預算表，感受規模的差距。

## 交付 (Ship It)

這堂課產出 `outputs/skill-long-video-strategy-planner.md`。給定影片長度與查詢複雜度，它會在暴力 context、壓縮與 agentic retrieval 之間做選擇，並計算延遲與品質的預期。

## 練習 (Exercises)

1. 一段 45 分鐘的講座，1 FPS，每影格 81 個 token。總 token 數是多少？放得進哪些模型的 context？

2. 設計一個 needle-in-a-haystack 測試：你在第幾分鐘注入標記，確切的查詢格式是什麼？

3. 在 1 小時的影片上，比較暴力 context 的 Qwen2.5-VL-72B（80k context）與 VideoAgent（Claude 3.5 + 檢索）。哪個在召回上勝出？哪個在延遲上勝出？

4. Ring attention 的每裝置記憶體隨序列長度除以裝置數而變，而整體 attention 運算量仍是二次方。解釋原因，以及如果拿掉 ring 輪轉階段會壞掉什麼。

5. 閱讀 Gemini 1.5 第 5 節關於 needle-in-a-haystack 的內容。論文在 100 萬與 1000 萬 token 的交界處，對召回率有什麼發現？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Brute context | "Just more tokens" | 把 LLM context 擴展到數百萬 token；一次 pass 處理所有東西 |
| Ring attention | "LWM-style parallel" | 分散式 attention 模式，每個裝置持有一個區塊並輪轉 |
| Token compression | "Summary tokens" | 在進入 LLM 前，透過學習到的壓縮器減少每個片段的 token |
| Needle-in-haystack | "NIH test" | 在隨機位置插入獨特標記，測試時要求模型回憶它 |
| Agentic retrieval | "LLM as query planner" | LLM 向檢索工具要求相關片段，透過 VLM 讀取，再組合答案 |
| VideoAgent | "Retrieval pattern for video" | 標準的 agentic-retrieval 設計：問題 -> 工具 -> 片段 -> 答案 |

## 延伸閱讀 (Further Reading)

- [Gemini Team — Gemini 1.5 (arXiv:2403.05530)](https://arxiv.org/abs/2403.05530)
- [Liu et al. — LWM / RingAttention (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Xue et al. — LongVILA (arXiv:2408.10188)](https://arxiv.org/abs/2408.10188)
- [Shu et al. — Video-XL (arXiv:2409.14485)](https://arxiv.org/abs/2409.14485)
- [Wang et al. — VideoAgent (arXiv:2403.10517)](https://arxiv.org/abs/2403.10517)
