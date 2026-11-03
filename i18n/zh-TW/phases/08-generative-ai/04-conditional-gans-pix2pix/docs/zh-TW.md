# Conditional GAN 與 Pix2Pix (Conditional GANs & Pix2Pix)

> 2014 到 2017 年第一個大突破，是控制 GAN 生成什麼。附上一個標籤、一張圖像，或一句話。Pix2Pix 做的是圖像版本，而且在窄領域的圖像到圖像任務上，至今仍勝過所有通用的 text-to-image 模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 03 (GANs), Phase 4 · 06 (U-Net), Phase 3 · 07 (CNNs)
**Time:** ~75 分鐘

## 問題 (The Problem)

無條件的 GAN 會抽出任意的人臉。拿來做 demo 可以，在正式環境毫無用處。你想要的是：*把草圖轉成照片*、*把地圖轉成空照圖*、*把白天場景轉成夜晚*、*替灰階圖像上色*。這些任務都是給你一張輸入圖像 $x$，要你輸出有語意對應的 $y$。每個 $x$ 都有許多合理的 $y$。均方誤差會把它們攤平成一團糊。Adversarial loss 不會，因為「看起來真實」是銳利的。

Conditional GAN (Mirza & Osindero, 2014) 把條件 $c$ 同時當作 $G$ 和 $D$ 的輸入。Pix2Pix (Isola et al., 2017) 把它特化：條件是一整張輸入圖像，generator 是 U-Net，discriminator 是 *以 patch 為單位* 的分類器 (PatchGAN)，loss 是 adversarial + L1。這個配方即使在 2026 年，在窄領域的圖像到圖像任務上仍勝過從零訓練的 text-to-image 模型，因為它是用 *成對資料* 訓練的，你擁有的正是所需的訊號。

## 核心概念 (The Concept)

![Pix2Pix：U-Net generator、PatchGAN discriminator](../assets/pix2pix.svg)

**條件式 G。** $G(x, z) \to y$。在 Pix2Pix 中，$z$ 是 G 內部的 dropout（沒有輸入雜訊，Isola 發現明確的雜訊會被忽略）。

**條件式 D。** $D(x, y) \to [0, 1]$。輸入是（條件, 輸出）這個 *配對*。這是關鍵差異：D 必須判斷 $y$ 是否與 $x$ 一致，而不只是 $y$ 看起來是否真實。

**U-Net generator。** 在 bottleneck 兩側有 skip connection 的 encoder-decoder。對於輸入和輸出共享低階結構（邊緣、輪廓）的任務至關重要。沒有 skip，高頻細節就會消失。

**PatchGAN discriminator。** D 不輸出單一的真/假分數，而是輸出 $N \times N$ 的網格，每個格子判斷約 70×70 像素的 receptive field，最後取平均。這是 Markov random field 的假設：真實感是局部的。訓練快得多、參數更少、輸出更銳利。

**Loss。**

```math
\begin{aligned}
\text{loss}_G &= -\log D(x, G(x)) + \lambda \cdot \lVert y - G(x) \rVert_1 \\
\text{loss}_D &= -\log D(x, y) - \log(1 - D(x, G(x)))
\end{aligned}
```

L1 項穩定訓練，並把 G 推向已知的目標。L1 比 L2 給出更銳利的邊緣（中位數而不是平均數）。$\lambda = 100$ 是 Pix2Pix 的預設值。

## CycleGAN：沒有成對資料時 (CycleGAN — when you don't have pairs)

Pix2Pix 需要成對的 $(x, y)$ 資料。CycleGAN (Zhu et al., 2017) 以多一個 loss 為代價，拿掉了這個需求：*cycle consistency* loss。兩個 generator：$G: X \to Y$ 和 $F: Y \to X$。訓練它們使 $F(G(x)) \approx x$ 且 $G(F(y)) \approx y$。這讓你不需要成對範例，就能把馬變成斑馬、夏天變成冬天。

2026 年，非成對的圖像到圖像大多改用 diffusion（ControlNet、IP-Adapter）而不是 CycleGAN，但 cycle-consistency 的想法幾乎在每一篇非成對領域適應的論文裡都還活著。

```figure
gx-patchgan
```

## 動手做 (Build It)

`code/main.py` 在 1-D 資料上實作一個小型 conditional GAN。條件 $c$ 是類別標籤（0 或 1）。任務：針對給定的類別，從條件分佈中產生樣本。

### 步驟 1：把條件接到 G 和 D 的輸入上

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot 編碼是最簡單的做法。較大的模型會用學出來的 embedding、FiLM modulation 或 cross-attention。

### 步驟 2：條件式訓練

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator 必須吻合 *給定條件下* 的真實分佈，而不是邊際分佈。

### 步驟 3：驗證每個類別的輸出

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## 常見陷阱 (Pitfalls)

- **條件被忽略。** G 學會把條件邊際化，而 D 因為條件訊號太弱而從不懲罰。解法：更積極地對 D 施加條件（放在早期的層，而不只是後面），使用 projection discriminator (Miyato & Koyama 2018)。
- **L1 權重太低。** G 會漂向任意看起來真實但不忠實的輸出。對 Pix2Pix 類型的任務，從 $\lambda \approx 100$ 開始。
- **L1 權重太高。** G 產生模糊的輸出，因為 L1 仍然是一種 $L_p$ 範數。訓練穩定後再逐步調降。
- **D 中的 ground-truth 洩漏。** 把 $(x, y)$ 串接起來當作 D 的輸入，而不是只給 $y$。沒有這樣做，D 就無法檢查一致性。
- **每個類別各自 mode collapse。** 每個類別可能獨立塌縮。要執行類別條件式的多樣性檢查。

## 實際使用 (Use It)

2026 年圖像到圖像任務的現況：

| 任務 | 最佳做法 |
|------|---------|
| 草圖 → 照片，同一領域，成對資料 | Pix2Pix / Pix2PixHD（仍然快、仍然銳利） |
| 草圖 → 照片，非成對 | ControlNet 搭配 Scribble 條件模型 |
| 語意分割 → 照片 | SPADE / GauGAN2 或 SD + ControlNet-Seg |
| 風格轉換 | 搭配 IP-Adapter 或 LoRA 的 diffusion；GAN 方法已屬舊式 |
| 深度圖 → 照片 | 在 Stable Diffusion 上使用 ControlNet-Depth |
| 超解析度 | Real-ESRGAN (GAN)、ESRGAN-Plus，或 SD-Upscale (diffusion) |
| 上色 | ColTran、基於 diffusion 的上色器，或 Pix2Pix-color |
| 白天 → 夜晚、季節、天氣 | CycleGAN 或基於 ControlNet 的方法 |

當 (a) 你有數千組成對範例、(b) 任務窄而且可重複、(c) 你需要快速推論時，Pix2Pix 仍然是正確的工具。在通用的開放領域任務上，diffusion 勝出。

## 交付 (Ship It)

存成 `outputs/skill-img2img-chooser.md`。這個 skill 接收任務描述、資料可得性（成對 vs 非成對、N 個樣本）以及延遲/品質預算，輸出：做法（Pix2Pix、CycleGAN、ControlNet 變體、SDXL + IP-Adapter）、訓練資料需求、推論成本，以及評估流程（LPIPS、FID、任務專屬指標）。

## 練習 (Exercises)

1. **簡單。** 修改 `code/main.py`，加入第三個類別。確認 G 仍然能把每個類別的雜訊映射到正確的 mode。
2. **中等。** 在 1-D 設定中，把 L1 換成 perceptual 風格的 loss（例如用一個凍結的小型 D 當作特徵擷取器）。它會改變條件分佈的銳利度嗎？
3. **困難。** 在 1-D 設定中草擬一個 CycleGAN：兩個分佈、兩個 generator、cycle loss。證明它在沒有成對資料的情況下，學得會在兩者之間映射。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Conditional GAN | 「有標籤的 GAN」 | G(z, c)、D(x, c)。兩個網路都看得到條件。 |
| Pix2Pix | 「圖像到圖像 GAN」 | 成對的 cGAN，搭配 U-Net G 和 PatchGAN D + L1 loss。 |
| U-Net | 「有 skip 的 encoder-decoder」 | 對稱的 conv 網路；skip 保留高頻資訊。 |
| PatchGAN | 「局部真實感分類器」 | D 輸出每個 patch 的分數，而不是全域分數。 |
| CycleGAN | 「非成對圖像轉換」 | 兩個 G + cycle-consistency loss；不需要成對資料。 |
| SPADE | 「GauGAN」 | 用語意圖對中間 activation 做正規化；分割圖到圖像。 |
| FiLM | 「Feature-wise linear modulation」 | 由條件產生每個特徵的 affine 轉換；便宜的條件化方式。 |

## 實務筆記 (Production note)：Pix2Pix 作為延遲受限的基準線

當你有成對資料和窄任務（草圖 → 渲染圖、語意圖 → 照片、白天 → 夜晚）時，Pix2Pix 的一次推論在延遲上勝過 diffusion 一個數量級。實務上的比較通常是：

| 路徑 | 步數 | 單張 L4 上 512² 的典型延遲 |
|------|------|--------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint 或 SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

在靜態 batch（每個請求的 FLOPs 都相同）下，Pix2Pix 的吞吐量勝出。Diffusion 則在品質與泛化上勝出。現代的做法通常是：針對窄任務上線一個 Pix2Pix 風格的 distilled 模型，並為長尾輸入準備一個 diffusion 備援。

## 延伸閱讀 (Further Reading)

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — cGAN 論文。
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix。
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN。
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD。
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN。
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D。
