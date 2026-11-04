# Janus-Pro：統一 Multimodal 模型的 Decoupled Encoder (Janus-Pro: Decoupled Encoders for Unified Multimodal Models)

> 統一的 multimodal 模型有一個無法避免的矛盾。理解需要語意特徵，也就是 SigLIP 或 DINOv2 輸出、富含概念層級資訊的向量。生成需要利於重建的 code，也就是能組回清晰像素的 VQ token。這兩個目標在單一 encoder 中無法相容。Janus (DeepSeek, 2024 年 10 月) 與 Janus-Pro (DeepSeek, 2025 年 1 月) 主張解法是不要再硬湊：把兩個 encoder 解耦。任務之間共用 transformer 主體，但理解走 SigLIP，生成走 VQ tokenizer。在 70 億參數下，Janus-Pro 於 GenEval 上勝過 DALL-E 3，同時在 MMMU 上追平 LLaVA。這堂課讀為什麼兩個 encoder 能成功，而一個 encoder 會失敗。

**Type:** Build
**Languages:** Python (stdlib, dual-encoder routing + shared-body signal)
**Prerequisites:** Phase 12 · 13 (Transfusion), Phase 12 · 14 (Show-o)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼單一共用 encoder 會折損理解或生成的品質。
- 描述 Janus-Pro 的路由：輸入端為了理解使用 SigLIP 特徵，為了生成則在輸入與輸出兩端都使用 VQ token。
- 追蹤讓 Janus-Pro 成功、而 Janus 沒有成功的資料混合 scaling。
- 比較解耦式 (Janus-Pro)、耦合連續式 (Transfusion) 與耦合離散式 (Show-o) 架構。

## 問題 (The Problem)

統一模型在理解與生成之間共用 transformer 主體。先前的嘗試（Chameleon、Show-o、Transfusion）在兩個方向都使用同一個 visual tokenizer。這個 tokenizer 是一種妥協：

- 為重建（生成）最佳化：VQ-VAE 捕捉細緻的像素細節，但產生的 token 語意連貫性較弱。
- 為語意（理解）最佳化：SigLIP embedding 把「貓」的圖片聚在「貓」的 token 附近，但不容許良好的重建。

Show-o 和 Transfusion 為此付出代價，在其中一個方向有明顯的品質稅。Janus-Pro 問：既然兩種任務需求不同，為什麼要求只用一個 tokenizer？

## 核心概念 (The Concept)

### 解耦的視覺編碼 (Decoupled visual encoding)

Janus-Pro 的架構把兩個 encoder 分開：

- 理解路徑。輸入圖片 → SigLIP-SO400m → 2 層 MLP → transformer 主體。
- 生成路徑。輸入圖片（若以既有圖片為條件）→ VQ tokenizer → token ID → transformer 主體。
- 輸出生成。由 transformer 預測的圖片 token → VQ decoder → 像素。

transformer 主體是共用的。主體上游與下游的一切都是任務專屬的。

輸入以 prompt 格式區分：`<understand>` 標籤走 SigLIP；`<generate>` 走 VQ。或者由任務隱含決定路由。

### 為什麼有效 (Why this works)

理解 loss 得到 SigLIP 特徵，而 CLIP 風格的 pretraining 已經為語意相似度調校過。模型的感知 benchmark 比 Show-o / Transfusion 進步，因為輸入特徵更適合這個任務。

生成 loss 得到 VQ token，而 tokenizer 已經為重建調校過。圖片品質比 Show-o 進步，因為 VQ code 能乾淨地組回像素。

共用的 transformer 主體看到兩種輸入分布（SigLIP 與 VQ），並學會同時處理它們。主張是：資料夠多、參數夠多時，主體會吸收這種切換。

### 資料 scaling：Janus 對 Janus-Pro (Data scaling — Janus vs Janus-Pro)

Janus（原版，arXiv 2410.13848）引入了解耦，但規模小（13 億參數、資料有限）。Janus-Pro (arXiv 2501.17811) 擴大了規模：

- 70 億參數（原本 13 億）。
- 階段 1（alignment）有 9000 萬組圖文配對，原本是 7200 萬。
- 階段 2（統一）有 7200 萬，原本是 2600 萬。
- 階段 3 新增 20 萬筆圖片生成 instruction 樣本。

結論：Janus-Pro-7B 在 MMMU 上追平 LLaVA（60.3 對約 58），並在 GenEval 上勝過 DALL-E 3（0.80 對 0.67）。一個開源模型，在統一光譜的兩端都有競爭力。

### JanusFlow：rectified flow 變體

JanusFlow (arXiv 2411.07975) 把 VQ 生成路徑換成 rectified-flow 生成路徑（連續）。分工變成理解用 SigLIP、生成用 rectified-flow。品質天花板進一步提高。架構仍是共用主體的解耦 encoder。

### 共用主體的工作 (The shared body's job)

transformer 主體處理統一序列，但有兩種輸入分布。它的工作是：

- 理解：吃進 SigLIP 特徵 + 文字 token → 以 autoregressive 方式輸出文字。
- 生成：吃進文字 token +（可選的圖片 VQ token）→ 以 autoregressive 方式輸出圖片 VQ token。

主體每個 block 沒有 modality 專屬的權重。它就是你會在 Qwen 或 Llama 裡面找到的文字風格 transformer，再加上兩個輸入 adapter。

有趣的是，這代表 Janus-Pro 的主體可以從預訓練 LLM 初始化。Janus-Pro 確實從 DeepSeek-MoE-7B 初始化。這個選擇很重要：LLM 貢獻了純從頭訓練的統一模型很難達到的推理能力。

### 與 InternVL-U 的比較 (Compared to InternVL-U)

InternVL-U（課程 12.10）是 2026 年的後續作品。它結合了：

- 原生 multimodal pretraining（InternVL3 backbone）。
- 解耦 encoder 路由（輸入用 SigLIP，輸出用 VQ + diffusion head）。
- 統一的理解 + 生成 + 編輯。

InternVL-U 把 Janus-Pro 的架構選擇納入更大的框架。解耦 encoder 的想法，如今是大規模統一模型的預設。

### 限制 (Limitations)

解耦 encoder 增加了架構複雜度。要訓練兩個 tokenizer，維護兩條輸入路徑，有兩套失敗模式。對不需要生成的產品，Janus-Pro 是過度設計，挑 LLaVA 家族的理解模型就好。

對不需要理解的產品，Janus-Pro 是大材小用，挑 Stable Diffusion 3 / Flux 模型。

對兩者都需要的產品，Janus-Pro 現在是參考用的開源架構。

```figure
l5-janus-decouple
```

## 實際使用 (Use It)

`code/main.py` 模擬 Janus-Pro 的路由：

- 兩個模擬 encoder：類 SigLIP 的（產生 256 維語意向量）與類 VQ 的（產生整數 code）。
- 一個 prompt 路由器，依任務標籤挑選 encoder。
- 一個共用主體（替身），不管是哪個 encoder 產生的，都處理 token 序列。
- 從階段 1 (alignment) 切換到階段 3 (instruction tune) 的加權取樣排程。

為 3 個範例印出路由路徑：圖片 QA、T2I、圖片編輯。

## 交付 (Ship It)

這堂課產出 `outputs/skill-decoupled-encoder-picker.md`。給定一個想要接近前沿品質的統一生成 + 理解的產品，它會在 Janus-Pro、JanusFlow 或 InternVL-U 之間挑選，並給出具體的資料規模建議。

## 練習 (Exercises)

1. Janus-Pro-7B 在 GenEval 上勝過 DALL-E 3。解釋為什麼 70 億參數的開源模型能在生成上追平前沿專有模型，卻無法在理解上做到。

2. 實作一個路由函式：給定 prompt 文字，分類為 `understand` 或 `generate`。你如何處理像「描述然後畫草圖」這樣有歧義的 prompt？

3. JanusFlow 用 rectified flow 取代 VQ 路徑。transformer 主體現在輸出什麼？loss 有什麼改變？

4. 提出 Janus-Pro 架構再加一個解耦 encoder 就能處理的第四種任務。例如：圖片分割（DINO 風格）、深度（MiDaS 風格）。

5. 閱讀 Janus-Pro 第 4.2 節關於資料 scaling 的內容。相較於 Janus，哪個資料階段對 T2I 品質提升貢獻最大？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Decoupled encoding | "Two visual encoders" | 每個方向各用獨立的 tokenizer 或 encoder：理解用語意式，生成用重建式 |
| Shared body | "One transformer" | 單一 transformer 處理任一 encoder 的輸出；沒有 modality 專屬權重 |
| SigLIP for understanding | "Semantic features" | CLIP 家族的 vision tower，提供豐富的概念特徵，但重建能力差 |
| VQ for generation | "Reconstruction codes" | 能乾淨解碼回像素的 vector-quantized token |
| JanusFlow | "Rectified-flow variant" | 以連續 flow-matching 生成 head 取代 VQ 的 Janus-Pro |
| Routing tag | "Task tag" | 用來挑選輸入 encoder 的 prompt 標記（`<understand>` / `<generate>`） |

## 延伸閱讀 (Further Reading)

- [Wu et al. — Janus (arXiv:2410.13848)](https://arxiv.org/abs/2410.13848)
- [Chen et al. — Janus-Pro (arXiv:2501.17811)](https://arxiv.org/abs/2501.17811)
- [Ma et al. — JanusFlow (arXiv:2411.07975)](https://arxiv.org/abs/2411.07975)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Dong et al. — DreamLLM (arXiv:2309.11499)](https://arxiv.org/abs/2309.11499)
