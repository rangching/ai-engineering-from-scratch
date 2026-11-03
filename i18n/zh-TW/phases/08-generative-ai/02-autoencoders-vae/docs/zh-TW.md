# Autoencoder 與 Variational Autoencoder (VAE)

> 一般的 autoencoder 先壓縮再重建。它只是在記憶，不會生成。加上一個技巧：強迫 code 看起來像 Gaussian，就得到一個 sampler。這個技巧，也就是 $z = \mu + \sigma \cdot \epsilon$ 的 reparameterization，正是你在 2026 年用的每一個 latent-diffusion 和 flow-matching 圖像模型，輸入端都有一個 VAE 的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 3 · 07 (CNNs), Phase 8 · 01 (Taxonomy)
**Time:** ~75 分鐘

## 問題 (The Problem)

把一張 784 像素的 MNIST 數字壓縮成 16 個數字的 code，再重建回來。一般的 autoencoder 可以把重建 MSE 做得很漂亮，但 code 空間是一團坑坑疤疤的混亂。在 code 空間隨便挑一個點去 decode，你只會得到雜訊。它沒有 sampler，只是一個包裝成別的東西的壓縮模型。

你真正想要的是：(a) code 空間是一個乾淨、平滑、可以 sampling 的分佈，例如各向同性的 Gaussian $\mathcal{N}(0, I)$；(b) decode 任何一個樣本都能得到像樣的數字；(c) encoder 和 decoder 仍然壓縮得好。三個目標，一個架構，一個 loss。

Kingma 2013 年的 VAE 這樣解決：訓練 encoder 輸出一個 *分佈* $q(z \mid x) = \mathcal{N}(\mu(x), \sigma(x)^2)$，透過 KL penalty 把這個分佈拉向 prior $\mathcal{N}(0, I)$，然後在 decode 之前從 $q(z \mid x)$ 抽 $z$。推論時，丟掉 encoder，抽 $z \sim \mathcal{N}(0, I)$，decode。讓 code 空間有結構的，就是 KL penalty。

2026 年的 VAE 很少單獨上線，就原始圖像品質而言已經被 diffusion 超越，但它們是每個 latent-diffusion 模型（SD 1/2/XL/3、Flux、AudioCraft）慣用的 encoder。學會 VAE，你就學會了你用的每條圖像管線裡看不見的第一層。

## 核心概念 (The Concept)

![Autoencoder vs VAE：reparameterization trick](../assets/vae.svg)

**Autoencoder。** $z = \text{encoder}(x)$，$\hat{x} = \text{decoder}(z)$，loss = $\lVert x - \hat{x} \rVert^2$。Code 空間沒有結構。

**VAE encoder。** 輸出兩個向量：$\mu(x)$ 和 $\log \sigma^2(x)$。它們定義 $q(z \mid x) = \mathcal{N}(\mu, \mathop{\mathrm{diag}}(\sigma^2))$。

**Reparameterization trick。** 從 $q(z \mid x)$ 抽樣是不可微的。把樣本改寫成 $z = \mu + \sigma \cdot \epsilon$，其中 $\epsilon \sim \mathcal{N}(0, I)$。這樣 $z$ 就是 $(\mu, \sigma)$ 的確定性函數加上一個與參數無關的雜訊，梯度可以流過 `μ` 和 $\sigma$。

**Loss。** Evidence Lower BOund (ELBO)，包含兩項：

```math
\begin{aligned}
\text{loss} &= \text{reconstruction} + \beta \cdot \mathrm{KL}[q(z \mid x) \,\Vert\, \mathcal{N}(0, I)] \\
&= \lVert x - \hat{x} \rVert^2 + \beta \cdot \sum_i \frac{\sigma_i^2 + \mu_i^2 - \log \sigma_i^2 - 1}{2}
\end{aligned}
```

Reconstruction 把 $\hat{x}$ 推向 $x$。KL 把 $q(z \mid x)$ 推向 prior。兩者互相拉扯。$\beta$ 小（$<1$）= 樣本較銳利，code 空間不那麼像 Gaussian。$\beta$ 大（$>1$）= code 空間較乾淨，樣本較模糊。β-VAE（Higgins 2017）讓這個旋鈕出名，也開啟了 disentanglement 的研究。

**Sampling。** 推論時：抽 $z \sim \mathcal{N}(0, I)$，通過 decoder 做 forward。一次 forward pass，不像 diffusion 要迭代 sampling。

```figure
vae-latent-grid
```

## 動手做 (Build It)

`code/main.py` 實作了一個不依賴 numpy 或 torch 的小型 VAE。輸入是 8 維的合成資料，取自 8-D 的 2 成分 Gaussian 混合。Encoder 和 decoder 都是單一隱藏層的 MLP。我們實作 tanh activation、forward pass、loss，以及手寫的 backward pass。這不是正式環境用的程式，是教學用的。

### 步驟 1：encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

用 $\log \sigma^2$ 而不是 $\sigma$，這樣網路輸出不受限制（對 σ 做 softplus 是個陷阱：在 $\sigma \approx 0$ 時梯度會消失）。

### 步驟 2：reparameterize 與 decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### 步驟 3：ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

因為兩個分佈都是 Gaussian，KL 有精確的封閉形式。不要用數值積分。2026 年還是有人上線用 Monte Carlo 估計 KL 的程式，慢 3 倍卻毫無理由。

### 步驟 4：生成

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

這就是生成模型。五行。

## 常見陷阱 (Pitfalls)

- **Posterior collapse。** KL 項把 $q(z \mid x) \to \mathcal{N}(0, I)$ 推得太用力，導致 $z$ 不帶任何關於 $x$ 的資訊。解法：β-annealing（從 $\beta=0$ 開始，逐步升到 1）、free bits，或對不活躍的維度略過 KL。
- **樣本模糊。** Gaussian decoder likelihood 意味著 MSE 重建，而 MSE 對 L2 來說是 Bayes 最佳（取平均），但一組合理數字的平均就是一個模糊的數字。解法：離散 decoder（VQ-VAE、NVAE），或只把 VAE 當 encoder，再在 latent 上疊 diffusion（Stable Diffusion 就是這樣做的）。
- **β 太大、又太早。** 見 posterior collapse。從 $\beta \approx 0.01$ 開始，再逐步拉高。
- **Latent 維度太小。** MNIST 用 16-D 就夠，ImageNet 256² 用 256-D，ImageNet 1024² 用 2048-D。Stable Diffusion 的 VAE 把 512×512×3 壓成 64×64×4（每邊縮小 8 倍，空間面積縮小 64 倍；channel 從 3 增加到 4，所以整體數值量約少 48 倍）。

## 實際使用 (Use It)

2026 年的 VAE 選擇：

| 情境 | 選擇 |
|------|------|
| Diffusion 用的圖像 latent encoder | Stable Diffusion VAE (`sd-vae-ft-ema`) 或 Flux VAE |
| 音訊 latent encoder | Encodec (Meta)、SoundStream 或 DAC (Descript) |
| 影片 latent | Sora 的時空 patch、Latte VAE、WAN VAE |
| Disentangled 表示學習 | β-VAE、FactorVAE、TCVAE |
| 離散 latent（供 transformer 建模） | VQ-VAE、RVQ (ResidualVQ) |
| 用於生成的連續 latent | 一般 VAE，再在該 latent 空間上條件化 flow/diffusion 模型 |

Latent-diffusion 模型就是在 encoder 與 decoder 之間放了一個 diffusion 模型的 VAE。VAE 負責粗略壓縮，diffusion 模型負責吃重的工作。影片（VAE + video-diffusion DiT）和音訊（Encodec + MusicGen transformer）也是同樣的模式。

## 交付 (Ship It)

存成 `outputs/skill-vae-trainer.md`。

這個 skill 接收：資料集概況 + 目標 latent 維度 + 下游用途（重建、sampling，或作為 latent-diffusion 輸入），輸出：架構選擇（plain/β/VQ/RVQ）、β 排程、latent 維度、decoder likelihood（Gaussian vs categorical），以及評估計畫（recon MSE、每個維度的 KL、$q(z \mid x)$ 與 $\mathcal{N}(0, I)$ 之間的 Fréchet distance）。

## 練習 (Exercises)

1. **簡單。** 把 `code/main.py` 裡的 $\beta$ 改成 `0.01`、`0.1`、`1.0`、`5.0`。記錄最後的重建 MSE 與 KL。對你的合成資料來說，哪個 $\beta$ 是 Pareto 最佳？
2. **中等。** 把 Gaussian decoder likelihood 換成 Bernoulli likelihood（cross-entropy loss）。在同一份合成資料的二值化版本上比較樣本品質。
3. **困難。** 把 `code/main.py` 擴充成迷你 VQ-VAE：用 K=32 個項目的 codebook 做最近鄰查詢，取代連續的 $z$。比較重建 MSE，並回報有多少 codebook 項目被用到（codebook collapse 是真的會發生）。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Autoencoder | 編碼再解碼的網路 | $x \to z \to \hat{x}$，學 MSE。不具生成能力。 |
| VAE | 附帶 sampler 的 AE | Encoder 輸出分佈，KL penalty 塑造 code 空間。 |
| ELBO | Evidence lower bound | $\log p(x) \ge \text{recon} - \mathrm{KL}[q(z \mid x) \Vert p(z)]$；當 $q = p(z \mid x)$ 時達到緊緻。 |
| Reparameterization | $z = \mu + \sigma \cdot \epsilon$ | 把隨機節點改寫成確定性部分加純雜訊。讓梯度可以穿過 sampling 反向傳播。 |
| Prior | $p(z)$ | latent 的目標分佈，通常是 $\mathcal{N}(0, I)$。 |
| Posterior collapse | 「KL 項贏了」 | Encoder 忽略 $x$、輸出 prior；decoder 只能自己幻想。 |
| β-VAE | 可調的 KL 權重 | $\text{loss} = \text{recon} + \beta \cdot \mathrm{KL}$。$\beta$ 越高 = 越 disentangled 但越模糊。 |
| VQ-VAE | 離散 latent | 把連續 $z$ 換成最近的 codebook 向量；讓 transformer 建模成為可能。 |

## 實務筆記 (Production note)：VAE 是 diffusion 伺服器上最熱的路徑

在 Stable Diffusion / Flux / SD3 管線中，每個請求會呼叫 VAE 兩次：一次 encode（做 img2img / inpainting 時），一次 decode。在 1024² 下，decoder 這一趟常常是整條管線中最大的 activation 記憶體尖峰，因為它要把 `128×128×16` 的 latent 放大回 `1024×1024×3`。兩個實務上的結論：

- **對 decode 做 slice 或 tile。** `diffusers` 提供 `pipe.vae.enable_slicing()` 和 `pipe.vae.enable_tiling()`。Tiling 用一點接縫瑕疵，換來 $O(\text{tile}^2)$ 的記憶體，而不是 $O(H \cdot W)$。在消費級 GPU 上跑 1024² 以上時是必要的。
- **Decoder 用 bf16，最後縮放的數值用 fp32。** SD 1.x 的 VAE 以 fp32 釋出，在 1024² 以上轉成 fp16 時會 *悄悄產生 NaN*。SDXL 提供 `madebyollin/sdxl-vae-fp16-fix`，請一律優先使用 fp16-fix 版本，或改用 bf16。

## 延伸閱讀 (Further Reading)

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 論文。
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE。
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE。
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art 圖像 VAE。
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion；VAE 作為 encoder。
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec，音訊 VAE 的標準。
