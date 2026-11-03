# Visual Autoregressive Modeling (VAR)：Next-Scale Prediction

> Diffusion model 沿著時間迭代 sampling（去雜訊步驟）。VAR 則沿著尺度迭代 sampling：它先預測 1x1 的 token，再預測 2x2，然後 4x4，一路到最終解析度，每個尺度都以前一個尺度為條件。這篇 2024 年的論文顯示，VAR 在圖像生成上符合 GPT 風格的 scaling law，且在相同運算預算下勝過 DiT。這堂課會建出核心機制。

**Type:** Build
**Languages:** Python (with PyTorch)
**Prerequisites:** Phase 7 Lesson 03 (Multi-Head Attention), Phase 8 Lesson 06 (DDPM)
**Time:** ~90 分鐘

## 問題 (The Problem)

Autoregressive 生成之所以主導了語言建模，是因為它的擴展性可預測：更多運算、更多參數，perplexity 更低、輸出更好。2024 年之前，圖像生成有兩條主要的 AR 嘗試：PixelRNN/PixelCNN（逐像素），以及 DALL-E 1 / Parti / MuseGAN（在 VQ-VAE 編碼上逐 token）。

兩者都有生成順序的問題。像素和 token 排列在 2D 網格中，但 AR 模型卻必須以 1D 的 raster 順序拜訪它們。靠前的角落像素完全不知道圖像最後會變成什麼。生成品質的擴展性比 GPT 用在文字上差，在相同運算量下也從未達到 diffusion model 的品質。

VAR 透過改變被生成的東西，解決了生成順序的問題。VAR 不是在空間上逐個預測圖像 token，而是預測解析度逐步提高的整張圖像。步驟 1：預測一個 1x1 的 token（整張圖像的「摘要」）。步驟 2：預測 2x2 的 token 網格（較粗的特徵）。步驟 3：預測 4x4 的網格。步驟 K：預測最終的 (H/8)x(W/8) 網格。

每個尺度會 attend 到所有先前的尺度（在「尺度順序」上是因果的），並在自己的尺度內平行進行。順序問題消失了：尺度 k 的整張圖像在一次 transformer pass 中就產生。

## 核心概念 (The Concept)

### VQ-VAE 多尺度 Tokenizer (VQ-VAE Multi-Scale Tokenizer)

VAR 需要一個 **多尺度離散 tokenizer**。對於圖像 $x$，它會產生一連串解析度逐步提高的 token 網格：

```
x -> encoder -> latent f
f -> tokenize at 1x1: token grid z_1 of shape (1, 1)
f -> tokenize at 2x2: token grid z_2 of shape (2, 2)
...
f -> tokenize at (H/p)x(W/p): token grid z_K of shape (H/p, W/p)
```

每個 $z_k$ 都使用同一個 codebook（典型大小 4096-16384）。各個尺度的 tokenization 並非彼此獨立：它的訓練方式，是使各尺度殘差的總和能重建出 $f$：

```math
f \approx \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_1), \text{target\_size}) + \cdots + \mathop{\mathrm{upsample}}(\mathop{\mathrm{embed}}(z_K), \text{target\_size})
```

這是 **residual VQ** 的一種變體。尺度 $k$ 捕捉尺度 $1 \ldots k-1$ 遺漏的部分。Decoder 取所有尺度 embedding 的總和，產生圖像。

多尺度 VQ tokenizer 只訓練一次（像 VQGAN 一樣），然後凍結。所有生成的工作，都由其上的 autoregressive 模型完成。

### Next-Scale Prediction

生成模型是一個 transformer，它看到所有先前尺度的 token，並預測下一個尺度的 token。

輸入序列的結構：
```
[START, z_1 tokens, z_2 tokens, z_3 tokens, ..., z_K tokens]
```

位置 embedding 同時編碼尺度索引和尺度內的空間位置。Attention 在尺度順序上是因果的，在尺度內則是整塊的：尺度 $k$、位置 $(i, j)$ 的 token，可以 attend 到尺度 $1 \ldots k$ 的所有 token，包括尺度 $k$ 本身的每個位置。尺度內沒有順序、也沒有因果性，所以同一尺度內的所有位置都是平行預測的。

訓練 loss：在每個尺度 $k$，給定所有先前尺度的 token，預測 token $z_k$。對離散 VQ 編碼使用 cross-entropy loss。結構與 GPT 相同，只是「序列」現在是以尺度來組織的。

### 生成 (Generation)

推論時：
```
generate z_1 = sample from p(z_1)                    # 1 token
generate z_2 = sample from p(z_2 | z_1)              # 4 tokens in parallel
generate z_3 = sample from p(z_3 | z_1, z_2)         # 16 tokens in parallel
...
decode: f = sum of embed-and-upsample scales 1..K
image = VAE_decoder(f)
```

當 $K = 10$ 個尺度時，生成只需 10 次 transformer forward pass。每次 pass 會平行產生整個尺度，在尺度內不做逐 token 的 autoregression。對 256x256 的圖像，這大約是 10 次 pass，而 DiT 需要 28-50 次。

### 為什麼 Next-Scale 勝過 Next-Token (Why Next-Scale Wins Over Next-Token)

三個結構上的優勢：
1. **由粗到細符合自然圖像的統計特性。** 人類視覺感知和圖像資料集都呈現與尺度相關的規律：低頻結構穩定且可預測；高頻細節取決於低頻內容。Next-scale prediction 利用了這一點。
2. **尺度內平行生成。** 與 GPT 風格的 token AR 不同，VAR 在一步中產生一個尺度的所有 token。有效的生成長度是對數尺度，而不是線性。
3. **沒有生成順序的偏誤。** 尺度 k 的 token 看得到整個尺度 k-1；沒有「在左邊」或「在上面」這種偏誤，迫使靠前的 token 在後面的脈絡可用之前就得做出決定。

### Scaling Law

Tian et al. 證明，VAR 在 ImageNet 上的 FID 遵循冪律 (power-law) 的擴展曲線，就像 GPT 在 perplexity 上那樣。把參數或運算量加倍，可靠地讓誤差減半。這是第一個能像語言模型一樣乾淨地展現這種擴展行為的圖像生成模型。結果是，VAR 的規模預測可以由運算量推得，而不是每個架構各自靠經驗猜測。

### 與 Diffusion 的關係 (Relationship to Diffusion)

VAR 和 diffusion 共享同一個資料壓縮的故事：兩者都把生成問題拆成一連串較簡單的子問題。

- Diffusion：逐步加入雜訊，學會還原一步。
- VAR：逐步增加解析度，學會預測下一個尺度。

它們是穿過這個問題的不同軸線。兩者都得到可處理的條件分佈。經驗上，VAR 的推論更快（pass 次數更少，尺度內全部平行），且在 class-conditional ImageNet 上達到或超越 DiT。Text-conditional 的 VAR（VARclip、HART）是活躍的研究方向。

```figure
gx-var-next-scale
```

## 動手做 (Build It)

在 `code/main.py` 中你會：
1. 在合成的「圖像」資料（2D Gaussian 環）上建立一個小型 **多尺度 VQ tokenizer**。
2. 訓練一個 **VAR 風格的 transformer** 來做 next-scale 預測 token。
3. 呼叫 transformer 4 次（4 個尺度）來 sampling，然後解碼。
4. 驗證以尺度順序訓練，確實讓尺度內的生成變成平行。

這是一個玩具實作。重點是看到以尺度為結構的 attention mask，以及尺度內平行生成確實能運作。

## 交付 (Ship It)

這堂課會產生 `outputs/skill-var-tokenizer-designer.md`：一個用來設計多尺度 tokenizer 的 skill，涵蓋尺度數量、尺度比例、codebook 大小、殘差共享，以及 decoder 架構。

## 練習 (Exercises)

1. **尺度數量的 ablation。** 用 4、6、8、10 個尺度訓練 VAR。量測重建品質對 autoregressive pass 次數的關係。尺度越多 = 殘差越細 = 品質越好，但 pass 次數越多。

2. **Codebook 大小。** 用 512、4096、16384 的 codebook 大小訓練 tokenizer。較大的 codebook 重建更好，但預測更困難。找出轉折點。

3. **尺度內平行的檢查。** 對一個訓練好的 VAR，明確地量測 attention 模式。在尺度 k 內，模型是否 attend 到跨尺度的位置，但不 attend 尺度內的位置？驗證 mask 的實作。

4. **VAR 與 DiT 的擴展性比較。** 針對相同的 ImageNet class-conditional 任務，在相同的參數預算下訓練 VAR 和 DiT（例如 33M、130M、458M）。畫出 FID 對運算量的圖。VAR 在每個大小上都應該領先 DiT，在小規模上重現論文的結果。

5. **文字條件。** 擴充 VAR，透過 adaLN 接受文字 embedding（CLIP pooled）作為額外的條件輸入。這就是 HART 的配方。在與文字對齊的 sampling 上，FID 改善多少？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| VAR | 「Visual AutoRegressive」 | 透過對 VQ token 網格金字塔做 next-scale prediction 來生成圖像 |
| Next-scale prediction | 「先預測粗的，再預測細的」 | 模型預測解析度逐漸提高的各尺度 token，並以所有先前的尺度為條件 |
| Multi-scale VQ tokenizer | 「Residual VQ」 | 產生 K 個解析度遞增的 token 網格的 VQ-VAE，由 decoder 加總所有尺度 |
| Scale k | 「金字塔第 k 層」 | K 個解析度層級之一，從 k=1 的 1x1 到 k=K 的 (H/p)x(W/p) |
| Parallel-within-scale | 「每個尺度一次 forward」 | 尺度 k 的所有 token 在一次 transformer pass 中預測，而不是 autoregressive 地逐一預測 |
| Causal-across-scales | 「依尺度順序的 attention」 | 尺度 $k$ 的 token 可以 attend 到尺度 $1 \ldots k$，但不能 attend 到尺度 $k+1 \ldots K$ |
| Residual VQ | 「累加式 tokenization」 | 每個尺度的 token 編碼較低尺度留下的殘差；decoder 加總所有尺度的 embedding |
| VAR scaling law | 「圖像版 GPT scaling」 | FID 對運算量遵循可預測的冪律，就像語言模型的 perplexity |
| HART | 「混合 VAR + 文字」 | 結合 MaskGIT 風格的迭代 decode 與 VAR 尺度結構的 text-conditional VAR 變體 |
| Scale position embedding | 「(scale, row, col) 三元組」 | 位置編碼同時帶有尺度索引和尺度內的空間座標 |

## 延伸閱讀 (Further Reading)

- [Tian et al., 2024 — "Visual Autoregressive Modeling: Scalable Image Generation via Next-Scale Prediction"](https://arxiv.org/abs/2404.02905) — VAR 論文，標準參考文獻
- [Peebles and Xie, 2022 — "Scalable Diffusion Models with Transformers"](https://arxiv.org/abs/2212.09748) — DiT，作為 diffusion 的比較基準
- [Esser et al., 2021 — "Taming Transformers for High-Resolution Image Synthesis"](https://arxiv.org/abs/2012.09841) — VQGAN，VAR 的多尺度 tokenizer 所延伸的 tokenizer 家族
- [van den Oord et al., 2017 — "Neural Discrete Representation Learning"](https://arxiv.org/abs/1711.00937) — VQ-VAE，離散圖像 tokenization 的基礎
- [Tang et al., 2024 — "HART: Efficient Visual Generation with Hybrid Autoregressive Transformer"](https://arxiv.org/abs/2410.10812) — text-conditional VAR
