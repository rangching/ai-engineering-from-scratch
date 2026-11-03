# Diffusion Model：從零實作 DDPM (Diffusion Models — DDPM from Scratch)

> Ho、Jain、Abbeel (2020) 給了這個領域一套讓人停不下來的配方。用一千個小步驟，把資料用雜訊毀掉。訓練一個神經網路來預測雜訊。推論時把過程倒轉。今天每個主流的圖像、影片、3D 和音樂模型都跑在這個迴圈上，上面可能再加上 flow matching 或 consistency 的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 02 (Backprop), Phase 8 · 02 (VAE)
**Time:** ~75 分鐘

## 問題 (The Problem)

你想要 $p_{\text{data}}(x)$ 的 sampler。GAN 玩的是一場常常發散的 minimax 賽局。VAE 用 Gaussian decoder 產生模糊樣本。你真正想要的訓練目標是：(a) 一個穩定的單一 loss（沒有鞍點、沒有 minimax），(b) 是 $\log p(x)$ 的下界（所以你有 likelihood），(c) 樣本品質能追上 SOTA。

Sohl-Dickstein et al. (2015) 給了理論上的答案：定義一個逐步加入 Gaussian 雜訊的 Markov chain $q(x_t \mid x_{t-1})$，並訓練一條反向的 chain $p_\theta(x_{t-1} \mid x_t)$ 來去雜訊。Ho、Jain、Abbeel (2020) 證明 loss 可以簡化成一行：預測雜訊，並把數學整理乾淨。2020 年這只是個好奇的玩意。2021 年它產生了 state-of-the-art 的樣本。2022 年它變成了 Stable Diffusion。2026 年它是基礎底座。

## 核心概念 (The Concept)

![DDPM：正向加雜訊，反向去雜訊](../assets/ddpm.svg)

**正向過程 $q$。** 用 $T$ 個小步驟加入 Gaussian 雜訊。讓數學變得可處理的原因在於封閉形式：累積後的步驟也是 Gaussian：

```math
q(x_t \mid x_0) = \mathcal{N}\left( \sqrt{\bar{\alpha}_t} \cdot x_0, (1 - \bar{\alpha}_t) \cdot I \right)
```

其中 $\bar\alpha_t = \prod_{s=1}^{t} (1 - \beta_s)$，對應一個 $\beta_t$ 的排程。讓 $\beta_t$ 在 $T=1000$ 步內從 1e-4 線性增加到 0.02，$x_T$ 就近似 $\mathcal{N}(0, I)$。

**反向過程 $p_\theta$。** 學一個神經網路 $\epsilon_\theta(x_t, t)$ 來預測被加入的雜訊。給定 $x_t$，這樣去雜訊：

```math
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \cdot \left( x_t - \frac{\beta_t}{\sqrt{1 - \bar{\alpha}_t}} \cdot \epsilon_\theta(x_t, t) \right) + \sigma_t \cdot z
```

其中 $\sigma_t$ 是 $\sqrt{\beta_t}$ 或學出來的變異數。這個式子很醜，但只是代數：給定 posterior $q(x_{t-1} \mid x_t, x_0)$ 解出 $x_{t-1}$，再把 $x_0$ 換成由雜訊預測得到的估計值。

**訓練 loss。**

```math
L_{\text{simple}} = \mathbb{E}_{x_0, t, \epsilon} \left[ \left\lVert \epsilon - \epsilon_\theta\left( \sqrt{\bar{\alpha}_t} \cdot x_0 + \sqrt{1 - \bar{\alpha}_t} \cdot \epsilon, t \right) \right\rVert^2 \right]
```

從資料抽 $x_0$，挑一個隨機的 $t$，抽 $\epsilon \sim \mathcal{N}(0, I)$，用封閉形式一步算出有雜訊的 $x_t$，然後對雜訊做回歸。一個 loss，沒有 minimax、沒有 KL、沒有 reparameterization 的技巧。

**Sampling。** 從 $x_T \sim \mathcal{N}(0, I)$ 開始。從 $t = T$ 到 1 迭代反向步驟。完成。

## 為什麼有效 (Why it works)

三個直覺：

1. **去雜訊容易；生成很難。** 在 $t=T$ 時，資料是純雜訊，網路要解的是一個簡單的問題。在 $t=0$ 時，網路只需要清理幾個像素。在中間的 $t$，問題很難，但網路的同一組權重會從每個雜訊層級收到許多梯度。

2. **偽裝的 score matching。** Vincent (2011) 證明，預測雜訊等同於估計 $\nabla_x \log q(x_t \mid x_0)$，也就是 *score*。反向 SDE 用這個 score 沿著密度梯度往上走，是一種朝高機率區域前進的引導式隨機漫步。

3. **ELBO 化簡成單純的 MSE。** 完整的 variational lower bound 在每個 timestep 都有一個 KL 項。在 DDPM 的參數化下，這些 KL 項簡化成帶有特定係數的雜訊預測 MSE；Ho 把係數丟掉（稱為「simple」loss），品質反而 *變好*。

```figure
diffusion-denoise
```

## 動手做 (Build It)

`code/main.py` 實作了 1-D 的 DDPM。資料是雙峰混合。「網路」是一個小型 MLP，輸入 $(x_t, t)$，輸出預測的雜訊。訓練就是那一行 loss。Sampling 則迭代反向 chain。

### 步驟 1：正向排程（封閉形式）

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### 步驟 2：一步取得 `x_t`

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### 步驟 3：一個訓練步驟

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### 步驟 4：反向 sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

對於一個 40 個 timestep、24 單元 MLP 的 1-D 問題，大約 200 個 epoch 就能學會雙峰混合。

## 時間條件 (Time conditioning)

網路需要知道它正在對哪個 timestep 去雜訊。兩種標準做法：

- **Sinusoidal embedding。** 類似 Transformer 的位置編碼。$\text{embed}(t) = [\sin(t/\omega_0), \cos(t/\omega_0), \sin(t/\omega_1), \ldots]$。通過一個 MLP，再 broadcast 進網路。
- **FiLM / group-norm 條件化。** 在每個區塊把 embedding 投影成每個通道的 scale/bias (FiLM)。

我們的玩具程式用 sinusoidal → concat。正式的 U-Net 則使用 FiLM。

## 常見陷阱 (Pitfalls)

- **排程影響很大。** 線性 $\beta$ 是 DDPM 的預設，但 cosine 排程（Nichol & Dhariwal, 2021）在相同運算量下能給出更好的 FID。品質停滯時就換排程。
- **Timestep embedding 很脆弱。** 直接把原始的 $t$ 當 float 傳入，在玩具 1-D 問題上可行，對圖像就不行；一定要用正規的 embedding。
- **V-prediction vs ε-prediction。** 在狹窄的區段（t 非常小或非常大），$\epsilon$ 的訊雜比很差。V-prediction（$v = \alpha \cdot \epsilon - \sigma \cdot x$）更穩定；SDXL、SD3 和 Flux 都使用它。
- **Classifier-free guidance。** 推論時，同時計算條件式與無條件的 $\epsilon$，然後 $\epsilon_{\text{cfg}} = (1 + w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$，其中 $w \approx 3\text{-}7$。第 08 課會介紹。
- **1000 步太多了。** 正式環境使用 DDIM（20-50 步）、DPM-Solver（10-20 步），或 distillation（1-4 步）。見第 12 課。

## 實際使用 (Use It)

| 角色 | 2026 年的典型技術組合 |
|------|---------------------|
| 圖像像素空間 diffusion（小型、玩具） | DDPM + U-Net |
| 圖像 latent diffusion | VAE encoder + U-Net 或 DiT（第 07 課） |
| 影片 latent diffusion | 時空 DiT (Sora、Veo、WAN) |
| 音訊 latent diffusion | Encodec + diffusion transformer |
| 科學（分子、蛋白質、物理） | Equivariant diffusion (EDM、RFdiffusion、AlphaFold3) |

Diffusion 是通用的生成骨幹。Flow matching（第 13 課）是 2024-2026 年的競爭者，在相同品質下通常以推論速度勝出。

## 交付 (Ship It)

存成 `outputs/skill-diffusion-trainer.md`。這個 skill 接收資料集 + 運算預算，輸出：排程（linear/cosine/sigmoid）、預測目標（ε/v/x）、步數、guidance scale、sampler 家族，以及評估流程。

## 練習 (Exercises)

1. **簡單。** 把 `code/main.py` 裡的 T 從 40 改成 10。樣本品質（輸出的視覺化 histogram）如何退化？T 到多少時，雙峰結構會塌縮？
2. **中等。** 從 ε-prediction 改成 v-prediction。重新推導反向步驟。比較最終樣本的品質。
3. **困難。** 加入 classifier-free guidance。以類別標籤 $c \in \lbrace 0, 1 \rbrace$ 作為條件，訓練時 10% 的機率丟掉它，sampling 時使用 $\epsilon = (1+w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$。量測 $w = 0, 1, 3, 7$ 時命中條件 mode 的比率。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Forward process | 「加雜訊」 | 固定的 Markov chain $q(x_t \mid x_{t-1})$，用來毀掉資料。 |
| Reverse process | 「去雜訊」 | 學出來的 chain $p_\theta(x_{t-1} \mid x_t)$，用來重建資料。 |
| β schedule | 「雜訊階梯」 | 每一步的變異數；linear、cosine 或 sigmoid。 |
| α̅ | 「Alpha bar」 | 累積乘積 $\prod (1 - \beta)$；讓你能從 $x_0$ 以封閉形式得到 $x_t$。 |
| Simple loss | 「對雜訊的 MSE」 | $\lVert \epsilon - \epsilon_\theta(x_t, t) \rVert^2$；所有 variational 推導最後都收斂到這個。 |
| ε-prediction | 「預測雜訊」 | 輸出是被加入的雜訊；標準 DDPM。 |
| V-prediction | 「預測 velocity」 | 輸出是 $\alpha \cdot \epsilon - \sigma \cdot x$；在各個 t 上條件更好。 |
| DDPM | 「那篇論文」 | Ho et al. 2020；線性 β、1000 步、U-Net。 |
| DDIM | 「確定性 sampler」 | 非 Markov 的 sampler，20-50 步，訓練目標相同。 |
| Classifier-free guidance | 「CFG」 | 混合條件式與無條件的雜訊預測，以放大條件的影響。 |

## 實務筆記 (Production note)：diffusion 推論是步數問題

DDPM 論文跑 T=1000 個反向步驟。沒有人在正式環境這樣上線。每個真實的推論技術組合都會選下列三種策略之一，而每一種都能清楚對應到正式環境中「延遲從哪裡來」的框架：

1. **更快的 sampler，同一個模型。** DDIM（20-50 步）、DPM-Solver++（10-20）、UniPC（8-16）。直接取代反向迴圈；訓練好的 $\epsilon_\theta$ 權重完全不動。延遲降低 20-50 倍。
2. **Distillation。** 訓練一個 student，用更少的步數去匹配 teacher：Progressive Distillation（2 → 1）、Consistency Models（任意 → 1-4）、LCM、SDXL-Turbo、SD3-Turbo。延遲再降低 5-10 倍，但需要重新訓練。
3. **快取與編譯。** `torch.compile(unet, mode="reduce-overhead")`、TensorRT-LLM 的 diffusion 後端、`xformers`/SDPA attention、bf16 權重。單步延遲降低約 2 倍。可以與 (1) 和 (2) 疊加。

對於正式環境的 diffusion 伺服器，預算的討論和文獻對 LLM 的描述相同：延遲是 `num_steps × step_cost + VAE_decode`，吞吐量是 `batch_size × (num_steps × step_cost)^-1`。TTFT 很小（一步）；相當於 TPOT 的就是整個回應時間，因為從使用者的角度，圖像生成是「一次到位」的。

## 延伸閱讀 (Further Reading)

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — diffusion 的起源論文，領先它的時代。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM。
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM，更少的步數。
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine 排程、學出來的變異數。
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — 統一的記號，最乾淨的配方。
