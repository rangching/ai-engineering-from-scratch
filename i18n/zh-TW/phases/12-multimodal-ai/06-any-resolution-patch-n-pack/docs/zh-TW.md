# Any-Resolution Vision：Patch-n'-Pack 與 NaFlex (Any-Resolution Vision: Patch-n'-Pack and NaFlex)

> 真實的圖片不是 224x224 的正方形。收據是 1:3，圖表是 16:9，醫療掃描可能是 4096x4096，手機截圖是 9:19.5。2024 年以前 VLM 的做法（把所有東西縮放成固定正方形）丟掉了讓 OCR、文件理解與高解析度場景解析得以運作的訊號。NaViT (Google, 2023) 證明你可以用 block-diagonal masking，把不同解析度的 patch 打包進單一 transformer batch。Qwen2-VL 的 M-RoPE (2024) 完全拋棄了絕對位置表。LLaVA-NeXT 的 AnyRes 則把高解析度圖片切成基底圖加子圖。SigLIP 2 的 NaFlex 變體 (2025) 現在是開源 VLM 的預設 encoder，適合想用單一 checkpoint 服務所有長寬比的人。這堂課會從頭到尾實作 patch-n'-pack。

**Type:** Build
**Languages:** Python (stdlib, patch packer + block-diagonal mask)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 12 · 05 (LLaVA)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 把一個 batch 中不同解析度圖片的 patch 打包成單一序列，並建立 block-diagonal attention mask。
- 針對特定任務，在 AnyRes tiling (LLaVA-NeXT)、NaFlex (SigLIP 2) 與 M-RoPE (Qwen2-VL) 之間做選擇。
- 在不縮放的情況下，計算 OCR、圖表與攝影的 token 預算。
- 說出正方形縮放的三種失敗模式：文字被壓扁、內容被裁掉、token 浪費在 padding 上。

## 問題 (The Problem)

Transformer 預期的是序列。一個 batch 是由相同長度的序列堆疊而成。如果你的圖片是 224x224，每次都得到 196 個 patch token，不需要 padding，搞定。用 224 訓練，用 224 推論，再也不必想解析度的事。

現實世界不配合。文件是直式的（8.5x11 英寸，大約 2:3）。圖表截圖是橫式的（16:9）。收據又高又窄（1:3）。醫療影像的尺寸是 2048x2048 或更大。手機截圖是 1170x2532（0.46:1）。

2024 年以前的三個選項，以及各自為何失敗：

1. 縮放成固定正方形（224x224 或 336x336）。壓扁會扭曲文字和人臉。縮小會毀掉圖表標籤和 OCR 內容。這是 LLaVA-1.5 之前的標準做法。
2. 裁切成固定長寬比。你丟掉了圖片的大部分，而且挑選裁切位置本身就是個視覺問題。
3. 以最長邊做 padding。解決了扭曲，但對直式圖片會把 50% 以上的 token 浪費在 padding 上。所有這些 pad token 還要付出二次方的 attention 成本。

2024-2025 年的答案：讓 transformer 以圖片的原生解析度吃 patch，然後想辦法把異質的 batch 打包成單一序列，不浪費運算。

## 核心概念 (The Concept)

### NaViT 與 patch-n'-pack

NaViT (Dehghani et al., 2023) 是證明這個做法能大規模運作的論文。概念很機械化：

1. 對 batch 中的每張圖片，以選定的 patch size（例如 14）計算它的原生 patch 網格。
2. 把每張圖片的 patch 攤平成它自己的可變長度序列。
3. 把所有圖片的 patch 串接成 batch 的一條長序列。
4. 建立 block-diagonal attention mask，讓圖片 A 的 patch 只在圖片 A 內部做 attention。
5. 攜帶每個 patch 的位置資訊（2D RoPE 或分數位置 embedding）。

一個 batch 有三張圖片，336x336（576 個 token）、224x224（256 個 token）和 448x336（768 個 token），會變成一條 1600 個 token 的序列，搭配 1600x1600 的 block-diagonal mask。沒有 padding，沒有浪費的運算。transformer 能處理任意長寬比。

NaViT 還引入了訓練期間的分數式 patch 丟棄：在整個 batch 中隨機丟掉 50% 的 patch，既有正則化效果，也加快訓練。SigLIP 2 繼承了這個做法。

### AnyRes (LLaVA-NeXT)

LLaVA-NeXT 的 AnyRes 是務實的替代方案。給定一張高解析度圖片與固定的 encoder（336 的 CLIP 或 SigLIP），把圖片切塊：

1. 從預先定義的集合中挑一種網格配置，例如 (1x1)、(1x2)、(2x1)、(1x3)、(3x1)、(2x2) 等，選最符合圖片長寬比的。
2. 把整張圖片切進網格；每個 tile 成為一個 336x336 的裁切。
3. 同時產生縮圖：整張圖片縮放成 336x336，作為全域 context 的 token。
4. 把每個 tile 都通過凍結的 336 encoder。串接 tile token 與縮圖 token。

對 672x672 的圖片，2x2 網格加縮圖：4 * 576 + 576 = 2880 個 visual token。昂貴但有效，LLM 同時看到局部細節與全域 context。

當你的 encoder 是凍結的、只支援單一解析度時，AnyRes 是首選。對大型圖片，它會讓 token 數爆炸（1344x1344 的圖片用 4x4 網格是 9216 + 576 ≈ 9800 個 token，在任何文字進來之前就溢出 8k 的 LLM context）。

### M-RoPE (Qwen2-VL)

Qwen2-VL 引入了 Multimodal Rotary Position Embedding。它不用 NaViT 的分數位置，也不用 AnyRes 的 tile 加縮圖，而是讓每個 patch 帶有 3D 位置（時間、高度、寬度）。query/key 的旋轉能處理任意的 H、W 與時間長度。

M-RoPE 不需重訓就原生支援動態解析度。推論時你餵入任何 HxW 的圖片，patch embedder 產生 H/14 x W/14 個 token，每個 token 得到它的 (t=0, r=row, c=col) 位置，RoPE 以正確的頻率旋轉 attention，完成。Qwen2.5-VL 與 Qwen3-VL 延續這個做法。InternVL3 的 V2PE 是相同的想法，只是各 modality 使用可變的編碼。

與 AnyRes 不同，M-RoPE 在原生解析度下是 $O(H \times W / P^2)$ 個 token，沒有 tile 的乘法式額外負擔。與 NaViT 不同，它仍預期每次 forward 只有一張圖片。跨解析度的 batching 仍需要在上面疊加 patch-n'-pack。

### NaFlex (SigLIP 2)

NaFlex 是 SigLIP 2 checkpoint 的 native-flex 模式。單一模型在推論時能服務多種序列長度（256、729、1024 個 token）。內部在訓練時使用 NaViT 風格的 patch-n'-pack，以及每個 patch 的絕對分數位置。賣點是：一個 checkpoint，在推論時依任務挑選你的 token 預算。

語意任務（分類、檢索）用 256 個 token。OCR 或圖表理解用 1024 個 token。不需重訓。

### 打包 mask (The packing mask)

block-diagonal mask 是大多數實作容易出錯的地方。對於長度 `N_total`、涵蓋圖片 `i=0..B-1`（長度為 `n_i`）的打包序列，形狀為 `(N_total, N_total)` 的 mask `M`，當兩個索引落在同一張圖片的區塊內時為 1，否則為 0。你可以用累積長度清單建出來：

```
offsets = [0, n_0, n_0+n_1, ..., N_total]
M[i, j] = 1 iff there exists b where offsets[b] <= i < offsets[b+1] and offsets[b] <= j < offsets[b+1]
```

在 PyTorch 中，用 `torch.block_diag` 或明確的 gather，這只要一行。FlashAttention 的可變長度路徑 (`cu_seqlens`) 完全略過 mask，直接用累積長度 tensor 在各序列內做 attention，對一般 batch 比 dense mask 快約 10 倍。

### Token 預算 (Token budgets)

依任務挑選策略：

- OCR / 文件：1024-4096 個 token。1024 的 SigLIP 2 NaFlex，或 AnyRes 3x3 + 縮圖。
- 圖表與 UI：384-448 原生解析度下 729-1024 個 token。搭配最大像素上限的 Qwen2.5-VL 動態解析度。
- 自然照片：256-576 個 token 就夠了。下游 LLM 看到的已經足夠。把 token 花在內容密度高的地方。
- 影片：空間 pooling 之後每影格 64-128 個 token，2-8 FPS。課程 12.17 會講到。

2026 年的 production 規則：為每個任務挑一個最大像素上限，以原生長寬比編碼直到該上限，打包 batch，並跳過 padding。Qwen2.5-VL 為了這個旋鈕，正好提供了 `min_pixels` 和 `max_pixels`。

```figure
mm-patch-n-pack
```

## 實際使用 (Use It)

`code/main.py` 為一個由整數像素座標組成的異質圖片 batch 實作 patch-n'-pack。它會：

- 接收一串 (H, W) 圖片尺寸。
- 在 patch size 14 下計算每張圖片的 patch 序列長度。
- 把它們打包成總長度為 `sum(n_i)` 的單一序列。
- 建立 block-diagonal attention mask（為了清楚，用 dense 形式）。
- 比較打包後的成本與正方形縮放、AnyRes tiling。
- 印出混合 batch（收據、圖表、截圖、照片）的 token 預算表。

執行它。算出來的數字，就是 2026 年每個開源 VLM 都使用 patch-n'-pack 的原因。

## 交付 (Ship It)

這堂課產出 `outputs/skill-resolution-budget-planner.md`。給定一個混合長寬比的工作負載（OCR、圖表、照片、影片影格）與總 token 預算，它會挑出正確的策略（NaFlex、AnyRes、M-RoPE 或固定正方形），並輸出每個請求的設定。當你為產品評估 VLM 規模時就用這個 skill，它能避免默默發生、毀掉延遲預算的 10 倍 token 暴增。

## 練習 (Exercises)

1. 一張收據是 600x1500（1:2.5）。在 patch size 14 下，原生解析度有多少 token？縮放成 336 正方形後有多少？實務上哪一種損失較多 OCR 準確率？

2. 為長度 256、576、729、1024 的四張圖片的 batch 建立 block-diagonal mask。驗證 attention 矩陣是 2585x2585，且恰好有 `256^2 + 576^2 + 729^2 + 1024^2` 個非零元素。

3. 對 patch 14 下的 1792x896 圖片，比較：(a) 縮放成 336 正方形再編碼，(b) AnyRes 2x1 + 縮圖，(c) 原生解析度的 M-RoPE。哪一個用的 token 最少？哪一個保留的細節最多？

4. 實作分數式 patch 丟棄：給定一個打包序列，均勻隨機丟掉 50% 的 token，並相應更新 block-diagonal mask。量測 mask 稀疏度的變化。

5. 閱讀 Qwen2-VL 論文 (arXiv:2409.12191) 的第 3.2 節。用兩句話描述 `min_pixels` 和 `max_pixels` 控制什麼，以及為什麼兩個界限都重要。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Patch-n'-pack | "NaViT-style packing" | 把來自不同圖片的可變長度 patch 序列串接進同一個 batch 維度 |
| Block-diagonal mask | "Packing mask" | 限制每張圖片的 patch 只能關注自己、不能關注打包中鄰居的 attention mask |
| AnyRes | "LLaVA-NeXT tiling" | 把高解析度圖片切成固定大小的 tile 網格加一張全域縮圖；用固定 encoder 編碼每個 tile |
| NaFlex | "SigLIP 2 native-flex" | 單一 SigLIP 2 checkpoint，推論時不需重訓就能服務 256/729/1024 個 token 的預算 |
| M-RoPE | "Multimodal RoPE" | 3D 旋轉位置編碼（時間、row、column），不需位置表就能處理任意 H、W、T |
| cu_seqlens | "FlashAttention packing" | FlashAttention varlen 路徑使用的累積長度 tensor，取代 dense block-diagonal mask |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的每請求旋鈕，對過小或過大的輸入限制 token 數 |
| Visual token budget | "How many tokens per image" | 每張圖片輸出的 patch token 大致數量；決定 LLM 的 prompt 預算與 attention 成本 |

## 延伸閱讀 (Further Reading)

- [Dehghani et al. — Patch n' Pack: NaViT (arXiv:2307.06304)](https://arxiv.org/abs/2307.06304)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Laurençon et al. — What matters when building vision-language models? (Idefics2, arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
