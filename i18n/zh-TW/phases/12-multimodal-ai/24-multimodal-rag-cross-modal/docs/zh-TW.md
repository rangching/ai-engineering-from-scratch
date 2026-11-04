# Multimodal RAG 與跨 Modality 檢索 (Multimodal RAG and Cross-Modal Retrieval)

> Vision-native 文件 RAG 只是其中一塊。Production 等級的 multimodal RAG 範圍更廣，要跨文字、圖片、音訊與影片檢索，用在像是旅遊規劃（「幫我找一間安靜、有自然光的純素早午餐」）、醫療分流（「什麼傷勢符合這張照片加這些筆記」）、電商（「跟這張自拍相似、我的尺寸的穿搭」）以及現場服務（「診斷這個引擎聲音加上零件照片」）等工作流程。2025 年的三篇綜述（Abootorabi et al.、Mei et al.、Zhao et al.）整理出子問題：跨 modality 檢索、檢索融合、生成 grounding、multimodal 評估。這堂課讀這些綜述，並設計一條 production 管線。

**Type:** Build
**Languages:** Python (stdlib, cross-modal retriever with fusion + grounded generator)
**Prerequisites:** Phase 12 · 23 (ColPali), Phase 11 (RAG basics)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 設計跨 modality 檢索：文字 → 圖片、圖片 → 文字、音訊 → 影片等。
- 比較三種融合策略：分數融合、基於 attention 的融合、MoE 融合。
- 解釋生成 grounding：當來源是多種 modality 的混合時，「引用你的來源」長什麼樣子。
- 說出 2025 年三篇經典的 multimodal RAG 綜述及其子問題分類。

## 問題 (The Problem)

單一 modality 的 RAG 是已解決的模式：對 query 做 embedding，對 chunk 做 embedding，檢索，塞進 LLM。Multimodal RAG 則需要：

1. 多個檢索 head（每個 modality 都需要落在相容空間中的 embedding）。
2. 跨 modality 的檢索結果融合。
3. 能跨 modality 引用來源的生成 grounding。
4. 涵蓋跨 modality 訊號的評估指標。

2025 年的綜述都得到相同的分類。

## 核心概念 (The Concept)

### 跨 modality 檢索 (Cross-modal retrieval)

給定 modality A 的 query，檢索 modality B 的文件。三種模式：

1. 共用 embedding 空間。CLIP 和 CLAP 在共用空間中產生文字 + 圖片 / 文字 + 音訊的 embedding。跨 modality 的 cosine similarity 可直接使用。限於 CLIP 訓練過的配對。

2. 各 modality 的 encoder + 轉換。文字 encoder + 圖片 encoder + 一個在各空間之間對應的小型轉換模組。Gupta et al. 的 Sen2Sen 與其他 2024 年的設計。彈性高，但增加複雜度。

3. 用 VLM 當 encoder。使用 VLM 的 hidden state 作為檢索表示。VLM 支援的任何 modality 都能用。品質較高，成本也較高。

選擇：文字+圖片用 CLIP / SigLIP 2；文字+音訊用 CLAP；要前沿品質的跨 modality 用 VLM-hidden-states。

### 融合策略 (Fusion strategies)

你檢索到 10 筆結果：5 張圖片、3 段文字、2 段音訊。如何合併？

分數融合（最便宜）。每個 modality 有自己的檢索器，各自回傳分數。先在 modality 內正規化分數，再加總。簡單，通常有效。

基於 attention 的融合。串接所有檢索到的項目，讓一個小型 attention 網路為它們加權。需要訓練。

MoE 融合。Gating 網路路由到各 modality 專屬的專家。不同類型的 query 路由方式不同，視覺問題會給圖片更高的權重。

Production 預設：分數融合，並稍微偏向 query 的主導 modality。如果 A/B 測試顯示在你的領域有明顯的好處，再升級到 MoE。

### 生成 grounding (Generation grounding)

LLM 應該引用是哪個檢索到的項目驅動了每項主張。對 multimodal：

- 文字來源：標準引用 `[1]`。
- 圖片來源：`[img 3]` 加上簡短 caption。
- 音訊：`[audio 2 at 0:34]`。

用感知 grounding 的資料訓練生成器：訓練目標中的每項主張都標有來源索引。推論時，模型自然就會輸出引用。

### 2025 年的綜述 (The 2025 surveys)

Abootorabi et al. (arXiv:2502.08826，"Ask in Any Modality")：multimodal RAG 的分類法。涵蓋檢索、融合、生成。涵蓋範圍最廣。

Mei et al. (arXiv:2504.08748，"A Survey of Multimodal RAG")：聚焦於子任務 benchmark 與失敗模式。對評估設計有用。

Zhao et al. (arXiv:2503.18016)：以視覺為主的綜述。在 ColPali 家族的工作上很強。

三篇都讀完，你就掌握了截至 2025 年春季的最新技術。大部分子問題仍未解決。

### MuRAG：奠基論文

MuRAG (Chen et al., 2022) 是第一個 multimodal RAG。從 multimodal 知識庫檢索圖片 + 文字，生成答案。在 VLM 浪潮之前就展示了可行性。現代系統 (REACT、VisRAG、M3DocRAG) 都建立在它之上。

### Production 旅遊規劃範例 (A production trip-planner example)

Query：「find me a quiet vegan brunch with natural light.」

管線：

1. 分解 query。「quiet」→ 音訊/評論關鍵字；「vegan brunch」→ 菜單品項；「natural light」→ 圖片特徵。
2. 依 modality 檢索：
   - 對評論做文字檢索：「vegan brunch, quiet ambiance.」
   - 對餐廳照片做圖片檢索：「natural light, airy.」
   - 對環境音片段做音訊檢索：「low decibel, no music.」
3. 融合分數。每間餐廳有一個綜合分數。
4. Top-k 餐廳 → 帶著所有證據的 VLM 生成器 → 附引用的答案。

這遠遠超出文字 RAG。每個 modality 都補上文字單獨會漏掉的訊號。

### Agentic multimodal RAG

多跳：如果第一次檢索沒有回傳高信心的答案，LLM 會重新表述並再次檢索。Phase 14 的 agentic RAG 模式適用於此。範例：

- 檢索初始 top-10 → LLM 說「太雜了，過濾 <40 dB」→ 重新檢索。
- 檢索圖片 → LLM 看到其中一張有菜單 → 檢索菜單文字 → 回答。

增加複雜度，但能處理單次檢索處理不了的 query。

### 評估 (Evaluation)

跨 modality 評估仍不成熟。常見的替代指標：

- 每個 modality 的 Recall@k。
- 融合後的 top-k 準確率。
- 人工判定的端到端滿意度。
- 任務專屬（完成的訂位、達成的購買）。

沒有橫跨所有 modality 的標準 benchmark。大多數論文在特定領域的任務上評估。

```figure
contrastive-matrix
```

## 實際使用 (Use It)

`code/main.py`：

- 三個模擬檢索器（文字、圖片、音訊），在共用的餐廳語料庫上運作。
- 以可設定權重結合各 modality 分數的分數融合。
- 一個輸出附引用最終答案的生成器 stub。
- 一個簡單的 agentic 迴圈，在信心低時重新表述 query。

## 交付 (Ship It)

這堂課產出 `outputs/skill-multimodal-rag-designer.md`。給定帶有 multimodal query 流程的產品規格，它會設計檢索器、融合、生成器與評估。

## 練習 (Exercises)

1. 提出一個醫療分流的 multimodal RAG：query = 傷勢照片 + 文字症狀。哪些 modality 從哪個知識庫檢索？

2. 分數融合是簡單的加權總和。它有哪種失敗模式是 MoE 融合避免掉的？

3. 閱讀 Abootorabi et al. 的分類法（第 3 節）。三個經典子問題是什麼，它們如何對應到你選擇的產品？

4. 為旅遊規劃的 multimodal RAG 設計評估規格。哪些指標涵蓋圖片召回、音訊召回與綜合正確性？

5. Agentic 多跳 RAG 每次來回都有延遲稅。query 難度到什麼程度時，準確率的增益才值得這個延遲？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Cross-modal retrieval | "Query one modality, retrieve another" | 文字 query 檢索圖片；圖片 query 檢索文字；需要共用空間或轉換器 |
| Score fusion | "Combine scores" | 各 modality 檢索分數的加權總和；最簡單的融合 |
| MoE fusion | "Modality-routed experts" | Gating 網路為每個 query 挑選要信任哪個 modality 的分數 |
| Grounded generation | "Cite your sources" | 答案中的每項主張都標上來源索引 |
| MuRAG | "First multimodal RAG" | 2022 年確立 multimodal RAG 模式的論文 |
| Agentic multi-hop | "Reformulate and retry" | 第一輪信心低時，LLM 重新查詢檢索器 |

## 延伸閱讀 (Further Reading)

- [Abootorabi et al. — Ask in Any Modality (arXiv:2502.08826)](https://arxiv.org/abs/2502.08826)
- [Mei et al. — A Survey of Multimodal RAG (arXiv:2504.08748)](https://arxiv.org/abs/2504.08748)
- [Zhao et al. — Vision RAG Survey (arXiv:2503.18016)](https://arxiv.org/abs/2503.18016)
- [Chen et al. — MuRAG (arXiv:2210.02928)](https://arxiv.org/abs/2210.02928)
- [Liu et al. — REACT (arXiv:2301.10382)](https://arxiv.org/abs/2301.10382)
