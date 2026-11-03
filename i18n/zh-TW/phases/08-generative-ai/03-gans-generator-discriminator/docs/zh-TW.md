# GAN：Generator 對 Discriminator (GANs — Generator vs Discriminator)

> Goodfellow 在 2014 年的技巧，是完全跳過密度。兩個網路：一個做假貨，一個抓假貨。它們互相對打，直到假貨和真貨分不出來。這照理說不該行得通，而且常常真的行不通。但一旦成功，在窄領域裡，樣本仍然是文獻中最銳利的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 08 (Optimizers), Phase 8 · 02 (VAE)
**Time:** ~75 分鐘

## 問題 (The Problem)

VAE 產生模糊的樣本，因為它的 MSE decoder loss 對 *平均* 圖像是 Bayes 最佳的，而許多合理數字的平均就是一個模糊的數字。你想要的是獎勵 *合理性 (plausibility)* 的 loss，而不是逐像素逼近任何單一目標。合理性沒有封閉形式，你得把它學出來。

Goodfellow 的想法：訓練一個分類器 $D(x)$ 去分辨真圖和假圖。訓練一個 generator $G(z)$ 去騙過 $D$。$G$ 的 loss 訊號，就是 $D$ 當下認為什麼東西看起來真實。隨著 $G$ 進步，這個訊號也跟著更新，等於追著一個移動的目標跑。如果兩個網路都收斂，$G$ 就學到了資料分佈，而且從頭到尾沒寫過 `log p(x)`。

這就是 adversarial training。數學上是一個 minimax 賽局：

```math
\min_G \max_D \; \mathbb{E}_{\text{real}}[\log D(x)] + \mathbb{E}_{\text{fake}}[\log(1 - D(G(z)))]
```

2026 年的 GAN 已經不是 SOTA 的生成器（diffusion 和 flow matching 拿走了這頂皇冠）。但 StyleGAN 2/3 仍是史上最銳利的人臉模型，GAN discriminator 被當作 diffusion 訓練中的 *perceptual loss*，而 adversarial training 也是讓即時 diffusion 得以上線的快速 1-step distillation（SDXL-Turbo、SD3-Turbo、LCM）背後的動力。

## 核心概念 (The Concept)

![GAN 訓練：generator 與 discriminator 的 minimax](../assets/gan.svg)

**Generator $G(z)$。** 把雜訊向量 $z \sim \mathcal{N}(0, I)$ 映射成樣本 $\hat{x}$。形狀像 decoder 的網路（dense 或 transposed conv）。

**Discriminator $D(x)$。** 把樣本映射成一個純量機率（或分數）。真 → 1，假 → 0。

**Loss。** 兩個交替的更新：

- **訓練 $D$：** $\text{loss}_D = -[ \log D(x) + \log(1 - D(G(z))) ]$。真=1、假=0 的 binary cross-entropy。
- **訓練 $G$：** $\text{loss}_G = -\log D(G(z))$。這是 Goodfellow 用的 *non-saturating* 形式（原始的 $\log(1 - D(G(z)))$ 在 $D$ 很有把握時會飽和，把梯度殺掉）。

**訓練迴圈。** $D$ 一步，$G$ 一步，重複。

**為什麼有效。** 如果 $G$ 完美吻合 $p_{\text{data}}$，$D$ 就不可能比隨機猜測做得更好，到處都輸出 0.5；$G$ 也不再得到梯度。這就是均衡。

**為什麼會壞掉。** Mode collapse（$G$ 找到一個 $D$ 分辨不出來的 mode，就永遠只造這一種）、梯度消失（$D$ 學得太快，$\log D$ 飽和）、訓練不穩定（learning rate、batch size，什麼都會影響）。

## 讓 GAN 真正能用的變體 (Variants that made GANs work)

| 年份 | 創新 | 修正 |
|------|------|------|
| 2015 | DCGAN | Conv/deconv、batch norm、LeakyReLU，第一個穩定的架構。 |
| 2017 | WGAN, WGAN-GP | 用 Wasserstein distance + gradient penalty 取代 BCE。解決梯度消失。 |
| 2017 | Spectral normalization | 用 Lipschitz 限制 discriminator。2026 年的 discriminator 仍在使用。 |
| 2018 | Progressive GAN | 先訓練低解析度，再逐步加層。第一批百萬畫素等級的成果。 |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm。固定領域擬真度的 state of the art。 |
| 2021 | StyleGAN3 | 無 aliasing、translation-equivariant，2026 年仍是人臉的黃金標準。 |
| 2022 | StyleGAN-XL | 條件式、感知類別、規模更大。 |
| 2024 | R3GAN | 以更強的正則化重新包裝；不需要額外技巧就能跑 1024²。 |

```figure
gan-minimax
```

## 動手做 (Build It)

`code/main.py` 在 1-D 資料（兩個 Gaussian 的混合）上訓練一個小型 GAN。Generator 和 discriminator 都是單一隱藏層的 MLP。我們手寫 forward、backward 和 minimax 迴圈。目的是親眼看到兩種關鍵的失效模式（mode collapse + 梯度消失）如何發生。

### 步驟 1：non-saturating loss

標準的 Goodfellow loss $\log(1 - D(G(z)))$，在 D 非常有把握地把 G 的假貨判成假時會趨近 0。這時 G 的梯度基本上是零，G 無法進步。Non-saturating 形式 $-\log D(G(z))$ 的漸近行為相反：D 越有把握它就越大，給 G 很強的訊號。

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### 步驟 2：每個 generator 步驟配一個 discriminator 步驟

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

給 G 用全新的假貨，否則梯度是過時的。

### 步驟 3：留意 mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

典型症狀：兩個真實 mode 其中一個不再被生成。Discriminator 也不會去糾正它，因為它從來沒被當成假貨看到。

## 常見陷阱 (Pitfalls)

- **Discriminator 太強。** 把 D 的 learning rate 砍到 1/2 到 1/5，或加入 instance/layer 雜訊。如果 D 的準確率超過 95%，G 就死了。
- **Generator 記住單一 mode。** 對 D 的輸入加雜訊、使用 minibatch-discriminator 層，或改用 WGAN-GP。
- **Batch norm 洩漏統計量。** 真實 batch 和假 batch 流過同一個 BN 層，會混在一起。改用 instance norm 或 spectral norm。
- **玩弄 Inception score。** FID 和 IS 在樣本數少時很吵。評估時至少用 ≥10k 個樣本。
- **「一次 sampling」對條件式任務是個謊言。** 你還是需要 CFG scale、truncation trick 和重新 sampling，才能得到可用的輸出。

## 實際使用 (Use It)

2026 年的 GAN 選擇：

| 情境 | 選擇 |
|------|------|
| 擬真人臉，固定姿態 | StyleGAN3（最銳利、最小） |
| 動漫 / 風格化人臉 | StyleGAN-XL 或 Stable Diffusion LoRA |
| 圖像到圖像轉換 | Pix2Pix / CycleGAN (Phase 8 · 04) 或 ControlNet (Phase 8 · 08) |
| 快速 1-step text-to-image | Diffusion 的 adversarial distillation（SDXL-Turbo、SD3-Turbo） |
| Diffusion 訓練器內的 perceptual loss | 作用於圖像裁切的小型 GAN discriminator |
| 任何多模態、開放式的任務 | 別用，改用 diffusion 或 flow matching |

GAN 銳利但狹窄。一旦你的領域開放起來，像是照片、任意文字提示、影片，就換成 diffusion。Adversarial 的技巧以元件的形式存活下來（perceptual loss、distillation），而不是獨立的生成器。

## 交付 (Ship It)

存成 `outputs/skill-gan-debugger.md`。這個 skill 接收一次失敗的 GAN 訓練（loss 曲線、樣本網格、資料集大小），輸出依可能性排序的原因清單、一行式的修正，以及重新執行的流程。

## 練習 (Exercises)

1. **簡單。** 用預設設定執行 `code/main.py`。接著設定 `D_LR = 5 * G_LR` 再執行一次。G 的 loss 多快會塌縮成常數？
2. **中等。** 把 Goodfellow 的 BCE loss 換成 WGAN loss：$\text{loss}_D = \mathbb{E}[D(\text{fake})] - \mathbb{E}[D(\text{real})]$，$\text{loss}_G = -\mathbb{E}[D(\text{fake})]$，並把 D 的權重 clip 到 `[-0.01, 0.01]`。訓練是否更穩定？比較實際時間（wall-clock）的收斂速度。
3. **困難。** 把 1-D 範例擴充到 2-D 資料（環狀排列的 8 個 Gaussian 混合）。追蹤 generator 在第 1k、5k、10k 步時抓到 8 個 mode 中的幾個。實作 minibatch discrimination 並重新量測。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Generator | 「G」 | 雜訊到樣本的網路，`G: z → x̂`。 |
| Discriminator | 「D」 | 分類器 `D: x → [0, 1]`，分辨真假。 |
| Minimax | 「這場賽局」 | 聯合目標函數的 $\min_G \max_D$。 |
| Non-saturating loss | 「那個修正」 | 對 G 使用 $-\log D(G(z))$，而不是 $\log(1 - D(G(z)))$。 |
| Mode collapse | 「G 只記住一種東西」 | 即使資料多樣，generator 卻只產生很少種不同的輸出。 |
| WGAN | 「Wasserstein」 | 用 Earth-Mover distance + gradient penalty 取代 BCE；梯度更平滑。 |
| Spectral norm | 「Lipschitz 技巧」 | 限制 D 的權重範數以約束其斜率；讓訓練穩定。 |
| StyleGAN | 「真正有用的那個」 | Mapping network + AdaIN；人臉領域同類最佳，2026 年依然如此。 |

## 實務筆記 (Production note)：一次推論是 GAN 長久的優勢

在開放領域的生成上，GAN 的樣本品質已不再領先，但推論成本仍然勝出。用 production-inference 文獻的語彙來說，GAN 具有：

- **沒有 prefill、沒有 decode 階段。** 只有一次 $G(z)$ forward pass。TTFT ≈ 總延遲。
- **沒有 KV-cache 壓力。** 唯一的狀態是權重。Batch size 受限於 activation 記憶體，而不是 cache。
- **Continuous batching 很簡單。** 因為每個請求都花同樣固定的 FLOPs，在伺服器目標占用率下用靜態 batch 通常就是最佳的，不需要 in-flight scheduler。

這就是為什麼 GAN distillation（SDXL-Turbo、SD3-Turbo、ADD、LCM）是 2026 年快速 text-to-image 的主流技術：它把 20-50 步的 diffusion 管線，壓成 1-4 次 GAN 式的 forward pass，同時保留 diffusion 基礎模型的分佈。Adversarial loss 以訓練期旋鈕的形式留下來，用來把慢的 generator 變成快的。

## 延伸閱讀 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — 最初的 GAN 論文。
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — 第一個穩定的架構。
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN。
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN。
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2。
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3。
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo。
