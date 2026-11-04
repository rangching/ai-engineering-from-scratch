# Flamingo 與 Few-Shot VLM 的 Gated Cross-Attention (Flamingo and Gated Cross-Attention for Few-Shot VLMs)

> DeepMind 的 Flamingo (2022) 有兩件事領先所有人。它證明單一模型可以處理任意交錯的圖片、影片與文字序列；也證明 VLM 可以做 in-context learning：給一個含三組 (image, caption) 範例的 few-shot prompt，模型不需要任何梯度更新就能替新圖片產生 caption。機制是 gated cross-attention 層，插在凍結 LLM 既有的層與層之間，帶有一個從零開始的 learned tanh gate，因此初始化時 LLM 的文字能力完全保留。這堂課會走一遍 Flamingo 的 Perceiver resampler 與 gated cross-attention 架構，也就是 Gemini 交錯輸入和 Idefics2 visual token 的祖先。

**Type:** Learn
**Languages:** Python (stdlib, gated cross-attention + Perceiver resampler demo)
**Prerequisites:** Phase 12 · 03 (BLIP-2 Q-Former)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 解釋 gated cross-attention 如何在初始化時透過 tanh(gate) = 0 保留凍結 LLM 的文字能力。
- 走一遍 Perceiver resampler：N 個圖片 patch → 透過 cross-attention 變成 K 個固定的「latent」query。
- 描述 Flamingo 如何用尊重圖片位置的 causal masking 處理圖文交錯序列。
- 重現 few-shot 的 multimodal prompt 結構（3 個圖片加 caption 的範例，後面接一張 query 圖片）。

## 問題 (The Problem)

BLIP-2 把 32 個 visual token 餵進凍結 LLM 的輸入層。每個 prompt 一張圖片時行得通。但如果你想餵*很多*張與文字交錯的圖片呢，例如「這是圖片 A，幫它寫 caption；這是圖片 B，幫它寫 caption；現在這是圖片 C，幫它寫 caption」？LLM 的 self-attention 得在單一串流中處理圖片 token 與文字 token，而哪些位置可以關注哪些圖片，這個問題會變得很麻煩。

Flamingo 的答案：完全不動 LLM 的輸入串流。在既有的 LLM block 之間插入額外的 cross-attention 層。文字 token 照常通過 LLM 的 causal self-attention。每隔幾個 LLM block，文字 token 還會透過一個新的 gated 層，對圖片特徵做 cross-attention。這個 gate（初始化為零）代表在第零步時，新層是 no-op，模型的行為與預訓練的 LLM 完全一樣。隨著訓練進行，gate 逐漸打開，視覺資訊才開始流入。

Flamingo 回答的第二個問題是：每個 prompt 的圖片數量不固定（0 張、1 張或很多張），該怎麼處理？答案是 Perceiver resampler，一個小型的 cross-attention 模組，不管你有多少 patch，都會替每張圖片產生固定數量的 visual latent token。總數會隨圖片數量增加，但每個文字 token 在 cross-attention 中只會讀取一張圖片的 latent，所以不論 prompt 裡有多少張圖片，單一文字 token 看到的形狀都一樣。

## 核心概念 (The Concept)

### 凍結的 LLM (The frozen LLM)

Flamingo 從一個凍結的 Chinchilla 70B LLM 開始。700 億個權重完全不動。既有的文字 self-attention 與 FFN 照常運作。

### Perceiver resampler

對 prompt 中的每張圖片，ViT 產生 $N$ 個 patch token。Perceiver resampler 有 $K$ 個固定的 learnable latent（Flamingo 用 $K=64$）。每個 resampler block 分兩個子步驟：

1. Cross-attention：$K$ 個 latent 關注 $N$ 個 patch token（Q 來自 latent，K/V 來自 patch）。
2. latent 內部的 self-attention + FFN。

經過 6 個 resampler block 後，不管 ViT 產生多少 patch，輸出都是 $K=64$ 個維度 1024 的 visual token。224x224 的圖片（196 個 patch）和 480x480 的圖片（900 個 patch），出來都是 64 個 resampler token。

對於影片，resampler 在時間軸上套用：每個影格的 patch 產生 64 個 latent，再用時間位置編碼讓模型區分 $t=0$ 與 $t=N$。整段影片會變成 $T \cdot 64$ 個 visual token。

### Gated cross-attention

在凍結 LLM 每 $M$ 層之間（Flamingo 用 $M=4$），插入一個新的 gated cross-attention block：

```
x_after_llm_block = llm_block(x_before)
cross = cross_attn(x_after, resampler_output)
gated = tanh(alpha) * cross + x_after
x_before_next_block = gated
```

- `alpha` 是初始化為零的可學習純量。
- `tanh(0) = 0`，所以初始化時 gated 分支的貢獻為零。
- 隨著 `alpha` 離開零，cross-attention 的貢獻平順地增加。
- residual connection 代表就算 gate 完全打開，也不會覆寫 LLM 的文字表示；只是把視覺資訊加在上面。

這是 Flamingo 最重要的單一設計選擇：視覺條件是加法式的、受 gate 控制的，而且初始化時為零。第 0 步的 Flamingo，在純文字輸入下就是一個完整的 Chinchilla 70B。

### 針對交錯輸入的 masked cross-attention (Masked cross-attention for interleaved inputs)

在像「<image A> caption A <image B> caption B <image C> ?」這樣的 prompt 中，每個文字 token 應該只看得到序列中排在它前面的圖片。cross-attention mask 強制規定：位置 `t` 的文字 token 只關注圖片 `i_t` 的 resampler token，其中 `i_t` 是位置 `t` 之前最近的一張圖片；其他圖片的 token 都被遮住。「只看最近的前一張圖片」或「看所有前面的圖片」都是有效的選擇；Flamingo 選了前者。

### In-context few-shot learning

Flamingo 的 prompt 長這樣：

```
<image1> A photo of a cat. <image2> A photo of a dog. <image3> A photo of a
```

模型看到補全的模式，就輸出「bird」（或 image3 顯示的任何東西）。沒有梯度更新。凍結 LLM 的 in-context learning 能力透過 gated cross-attention 一路帶過來，這是論文的重點，也是它重要的原因。

### 訓練資料 (Training data)

Flamingo 用三個資料集訓練：

1. MultiModal MassiveWeb (M3W)：4300 萬個網頁，圖片與文字交錯，並重建閱讀順序。
2. Image-Text Pairs：ALIGN（18 億組配對）加 LTIP（3.12 億組配對），總共約 21 億組配對。
3. Video-Text Pairs (VTP)：2700 萬段短影片。

OBELICS (2023) 是這個交錯網頁語料庫的開源重現，Idefics、Idefics2 和大多數開源的「類 Flamingo」模型都用它訓練。

### OpenFlamingo 與 Otter

OpenFlamingo (2023) 是開源重現。架構完全相同（Perceiver resampler + 在凍結 LLaMA 或 MPT 上的 gated cross-attention）。有 3B、4B、9B 的 checkpoint。由於基礎 LLM 較小、資料較少，品質落後 Flamingo。

Otter (2023) 建立在 OpenFlamingo 之上，用 MIMIC-IT（一個 multimodal 指令資料集）做 instruction tuning，顯示 gated cross-attention 也適用於遵循指令。

### 後代 (The descendants)

- Idefics / Idefics2 / Idefics3：Hugging Face 的 gated cross-attention 系譜，逐步簡化（Idefics2 捨棄 resampler，改用直接的 patch token 加 adaptive pooling）。
- Flamingo 到 Chameleon 的轉變：到 2024 年許多團隊轉向 early-fusion（課程 12.11）；Flamingo 式 gated cross-attention 仍用在需要凍結 backbone 的 production 場景。
- Gemini 的交錯輸入：概念上繼承了 Flamingo 的交錯格式彈性，不過確切機制是專有的。

### 與 BLIP-2 的比較 (Comparison to BLIP-2)

| | BLIP-2 | Flamingo |
|---|---|---|
| 視覺橋樑 | 輸入處的一個 Q-Former | 每 M 層一個 gated cross-attention |
| Visual token | 每張圖片 32 個 | 每張圖片每個 cross-attn 層 64 個 |
| 凍結 LLM | 是 | 是 |
| Few-shot in-context | 弱 | 強，是論文的核心 |
| 交錯輸入 | 沒有原生支援 | 有，是設計目標 |
| 訓練資料 | 1.3 億組配對 | 約 21 億組配對 + 2700 萬段影片 + 4300 萬個交錯網頁 |
| 參數量 | 訓練 1.88 億 | 訓練約 100 億（cross-attn 層） |
| 運算量 | 8 張 A100 跑幾天 | 數千顆 TPUv4 跑幾週 |

預算有限的單圖 VQA 選 BLIP-2。交錯、few-shot 或多圖推理選 Flamingo/Idefics2。

```figure
cross-attention-fusion
```

## 實際使用 (Use It)

`code/main.py` 示範：

1. 在 36 個假的 patch token 上，用 8 個 learnable latent 跑 Perceiver resampler（純 Python cross-attention）。
2. 一個 gated cross-attention 步驟：`alpha = 0` → 輸出等於輸入（LLM 不變），然後 `alpha = 2.0` → 混入視覺貢獻。
3. 一個交錯 mask 產生器，為「(image 1) (text 1) (image 2) (text 2)」序列產生 2D attention mask。

## 交付 (Ship It)

這堂課產出 `outputs/skill-gated-bridge-diagnostic.md`。給定一個 open VLM 的設定（有沒有 resampler、cross-attn 頻率、gate 機制），它會辨識出 Flamingo 系譜的元素，並解釋凍結策略。對於除錯為什麼 fine-tune 後文字表現退化很有用（答案：gate 太快開得太大）。

## 練習 (Exercises)

1. 計算 Flamingo-9B 的視覺參數量：9B LLM + 14 億 gated cross-attention 層 + 6400 萬 resampler。被訓練的參數占總數的多少比例？

2. 在 PyTorch 中實作 gated residual `y = tanh(alpha) * cross + x`。用實驗證明在 `alpha=0` 時，初始化時 `y==x` 完全成立。

3. 閱讀 OpenFlamingo 第 3.2 節 (arXiv:2308.01390)，了解當每個 prompt 的圖片數量不同時，他們如何處理 batch 中的多張圖片。描述其 padding 策略。

4. 為什麼 Flamingo 的 cross-attention mask 讓文字 token *只*關注最近的前一張圖片，而不是所有前面的圖片？閱讀 Flamingo 論文第 2.4 節並解釋這個取捨。

5. In-context few-shot：為一個新的 Flamingo 變體，建構含 4 個「圖片 → 主要物體顏色」範例的 prompt。範例數量從 0 變化到 8 時，描述預期的準確率變化模式。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|----------------|------------------------|
| Perceiver resampler | "Fixed-latent cross-attention" | 從數量不定的輸入 patch 產生 K 個固定 token 的模組 |
| Gated cross-attention | "Tanh-gated bridge" | Residual 層 `y = tanh(alpha)*cross + x`，alpha 可學習，初始為 0 |
| Interleaved input | "Mixed sequence" | 圖片與文字依閱讀順序自由混合的 prompt 格式 |
| Frozen LLM | "No LLM gradients" | 文字 LLM 的權重不更新；只有 resampler 與 cross-attn 層在訓練 |
| Few-shot | "In-context examples" | 在 prompt 中給幾組 (image, answer)；模型不需 finetuning 就能泛化 |
| OBELICS | "Interleaved web corpus" | 包含 1.41 億個網頁、圖文依閱讀順序排列的開源資料集 |
| Chinchilla | "70B frozen base" | Flamingo 的凍結文字 LLM，來自 DeepMind 的 Chinchilla 論文 |
| Gate schedule | "How alpha moves" | 訓練期間 cross-attention gate 打開的速率 |
| Cross-attn frequency | "Every M layers" | 多久插入一個 gated cross-attention block；Flamingo 用 $M=4$ |
| OpenFlamingo | "Open reproduction" | MosaicML/LAION 的 3-9B 開源 checkpoint；架構與 Flamingo 相同 |

## 延伸閱讀 (Further Reading)

- [Alayrac et al. — Flamingo (arXiv:2204.14198)](https://arxiv.org/abs/2204.14198) — 原始論文。
- [Awadalla et al. — OpenFlamingo (arXiv:2308.01390)](https://arxiv.org/abs/2308.01390) — 開源重現。
- [Laurençon et al. — OBELICS (arXiv:2306.16527)](https://arxiv.org/abs/2306.16527) — 交錯網頁語料庫。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — 通用的 Perceiver 架構。
- [Li et al. — Otter (arXiv:2305.03726)](https://arxiv.org/abs/2305.03726) — 經 instruction tuning 的 Flamingo 後代。
- [Laurençon et al. — Idefics2 (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246) — Flamingo 做法的現代簡化版。
