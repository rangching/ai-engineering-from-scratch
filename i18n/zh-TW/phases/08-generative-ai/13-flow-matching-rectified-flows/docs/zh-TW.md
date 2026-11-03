# Flow Matching 與 Rectified Flow (Flow Matching & Rectified Flows)

> Diffusion model 需要 20-50 個 sampling 步驟，因為它們從雜訊走到資料的路徑是彎的。Flow matching (Lipman et al., 2023) 與 rectified flow (Liu et al., 2022) 訓練的是直線路徑。路徑越直，需要的步數越少，推論就越快。Stable Diffusion 3、Flux.1 和 AudioCraft 2 在 2024 年都轉向了 flow matching。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 06 (DDPM), Phase 1 · Calculus
**Time:** ~45 分鐘

## 問題 (The Problem)

DDPM 的反向過程，是從 $\mathcal{N}(0, I)$ 走回資料分佈的 1000 步隨機漫步。DDIM 把它壓縮成 20-50 個確定性步驟。你想要更少的步數，最好是一步。阻礙在於求解反向過程的 ODE 是 stiff 的，路徑是彎的。

如果你能訓練模型，讓從雜訊到資料的路徑是一條 *直線*，那麼從 $t=1$ 到 $t=0$ 的單一 Euler 步驟就夠了。Flow matching 直接建立這件事：定義從 $x_1 \sim \mathcal{N}(0, I)$ 到 $x_0 \sim \text{data}$ 的直線內插，訓練一個向量場 $v_\theta(x, t)$ 去吻合它對時間的導數，推論時再做積分。

Rectified flow (Liu 2022) 更進一步：用 reflow 程序反覆拉直路徑，產生越來越接近線性的 ODE。經過兩次 reflow 迭代後，2 步的 sampler 就能達到 50 步 DDPM 的品質。

## 核心概念 (The Concept)

![Flow matching：雜訊與資料之間的直線內插](../assets/flow-matching.svg)

### 直線 flow (Straight-line flow)

定義：

```math
x_t = t \cdot x_1 + (1 - t) \cdot x_0, \quad t \in [0, 1]
```

其中 $x_0 \sim \text{data}$ 且 $x_1 \sim \mathcal{N}(0, I)$。沿著這條直線，對時間的導數是常數：

```math
\frac{dx_t}{dt} = x_1 - x_0
```

定義一個神經向量場 $v_\theta(x_t, t)$，並訓練它去吻合這個導數：

```math
L = \mathbb{E}_{x_0, x_1, t} \left\lVert v_\theta(x_t, t) - (x_1 - x_0) \right\rVert^2
```

這就是 **conditional flow matching** loss (Lipman 2023)。訓練不需要模擬：你從不展開 ODE，只要抽樣 $(x_0, x_1, t)$ 然後做回歸。

### Sampling

推論時，沿著時間 *往回* 對學出來的向量場做積分：

```math
x_{t-\Delta t} = x_t - \Delta t \cdot v_\theta(x_t, t)
```

從 $x_1 \sim \mathcal{N}(0, I)$ 開始，用 Euler 步驟一路走到 $t=0$。

### Rectified flow (Liu 2022)

直線 flow 行得通，但學出來的路徑 *其實並不直*，因為許多 $x_0$ 可能對應到同一個 $x_1$，所以路徑會彎曲。Rectified flow 的 reflow 步驟：

1. 用隨機配對訓練 flow 模型 $v_1$。
2. 抽 N 組 $(x_1, x_0)$，做法是把 $v_1$ 從 $x_1$ 積分到它落地的 $x_0$。
3. 在這些配對範例上訓練 $v_2$。由於這些配對現在是「ODE 匹配」的，它們之間的直線內插確實會更平。
4. 重複。

實務上，2 次 reflow 迭代就能讓路徑接近線性，使 2-4 步的推論成為可能。SDXL-Turbo、SD3-Turbo、LCM 都是從 flow matching distill 出來的模型。

### 為什麼它在 2024 年的圖像領域勝出

三個原因：

1. **訓練不需要模擬**：訓練期間不用展開 ODE，實作很簡單。
2. **更好的 loss 幾何**：直線路徑有一致的訊雜比，而 DDPM 的 ε-loss 在排程兩端的 SNR 很差。
3. **推論更快**：4-8 步就有 SDXL-Turbo 的品質；搭配 consistency distillation 則 1 步。

## Flow matching 與 DDPM：確切的關聯 (Flow matching vs DDPM — the exact connection)

使用 Gaussian 條件路徑的 flow matching，就是 *帶有特定雜訊排程的* diffusion。選定 $x_t = \alpha(t) x_0 + \sigma(t) x_1$ 的排程，flow matching 就還原成以 Stratonovich 重新表述的 diffusion，其中 $v = \alpha' \cdot x_0 - \sigma' \cdot x_1$。對 Gaussian 路徑而言，兩者在代數上是等價的。

Flow matching 新增的是：目標的 *清晰度*（單純的速度）、更乾淨的 loss，以及嘗試非 Gaussian 內插的許可。

```figure
normalizing-flow
```

## 動手做 (Build It)

`code/main.py` 在雙峰 Gaussian 混合上實作 1-D flow matching。向量場 $v_\theta(x, t)$ 是一個以直線目標訓練的小型 MLP。推論時，積分 1、2、4 和 20 個 Euler 步驟，並比較樣本品質。

### 步驟 1：訓練 loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### 步驟 2：多步推論

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### 步驟 3：比較步數

預期 4 步的 sampler 就已經能追上 20 步的品質，這對延遲來說意義重大。

## 常見陷阱 (Pitfalls)

- **時間的參數化。** Flow matching 使用 $t \in [0, 1]$，$t=0$ 是資料，$t=1$ 是雜訊。DDPM 使用 $t \in [0, T]$，$t=0$ 是資料，$t=T$ 是雜訊。方向相同，尺度不同。論文常常搞錯這點。
- **排程的選擇。** Rectified flow 的直線就是「那個」flow-matching 排程，但你也可以使用 cosine 或 logit-normal 的 t 取樣（SD3 這樣做），以涵蓋更好的尺度範圍。
- **Reflow 的成本。** 為 reflow 產生配對資料集，每個樣本都需要完整的一次推論。只有在你真的需要 1-2 步推論時才做 reflow。
- **Classifier-free guidance 仍然適用。** 只要在線性組合中把 $\epsilon$ 換成 $v$：$v_{\text{cfg}} = (1+w) v_{\text{cond}} - w v_{\text{uncond}}$。

## 實際使用 (Use It)

| 使用情境 | 2026 年的技術組合 |
|---------|-----------------|
| Text-to-image，最佳品質 | Flow matching：SD3、Flux.1-dev |
| Text-to-image，1-4 步 | Distilled flow matching：Flux.1-schnell、SD3-Turbo、SDXL-Turbo |
| 即時推論 | 從 flow-matched 基礎模型做 consistency distillation（LCM、PCM） |
| 音訊生成 | Flow matching：Stable Audio 2.5、AudioCraft 2 |
| 影片生成 | Flow matching 混合 diffusion（Sora、Veo、Stable Video） |
| 科學 / 物理（粒子軌跡、分子） | Flow matching + equivariant 向量場 |

2025-2026 年只要論文說「比 diffusion 更快」，幾乎都是 flow matching + distillation。

## 交付 (Ship It)

存成 `outputs/skill-fm-tuner.md`。這個 skill 接收 diffusion 風格的模型規格，並將其轉換成 flow-matching 訓練設定：排程選擇、時間取樣分佈（uniform / logit-normal）、optimizer、reflow 計畫、目標步數、評估流程。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`，比較 1 步與 20 步相對於真實資料分佈的 MSE。
2. **中等。** 把 $t$ 的取樣從 uniform 改成 logit-normal（讓取樣集中在中間的 t）。模型品質有改善嗎？
3. **困難。** 實作一次 reflow 迭代：透過積分第一個模型產生配對的 $(x_0, x_1)$，在這些配對上訓練第二個模型，並比較 1 步的樣本品質。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Flow matching | 「直線版 diffusion」 | 訓練 $v_\theta(x, t)$ 去吻合沿著內插路徑的 $x_1 - x_0$。 |
| Rectified flow | 「Reflow」 | 把學出來的 flow 拉直的反覆程序。 |
| Velocity field | 「v_θ」 | 模型的輸出，也就是 $x_t$ 該移動的方向。 |
| Straight-line interpolant | 「那條路徑」 | $x_t = (1-t) \cdot x_0 + t \cdot x_1$；目標導數很簡單。 |
| Euler sampler | 「一階 ODE solver」 | 最簡單的積分器；路徑是直的時候效果很好。 |
| Logit-normal t | 「SD3 的取樣」 | 讓 $t$ 的取樣集中在梯度最強的中間值。 |
| Consistency distillation | 「1 步 sampler」 | 訓練一個 student，把任何 $x_t$ 直接映射到 $x_0$。 |
| CFG with velocity | 「v-CFG」 | $v_{\text{cfg}} = (1+w) v_{\text{cond}} - w v_{\text{uncond}}$；同樣的技巧，新的變數。 |

## 實務筆記 (Production note)：Flux.1-schnell 是最快的 flow matching

Flow matching 在正式環境的勝利就是 Flux.1-schnell：一個 flow-matched 的 DiT，被 distill 成 1-4 個推論步驟，同時保有 Flux-dev 等級的品質。Niels 的「Run Flux on an 8GB machine」notebook 是參考的部署配方：T5 + CLIP 編碼、量化的 MMDiT 去雜訊（schnell 4 步，dev 則 50 步）、VAE decode。成本計算如下：

| 變體 | 步數 | L4 上 1024² 的延遲 | 總 FLOPs（相對值） |
|------|------|-------------------|------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08×（快 12 倍） |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

正式環境的規則：**flow-matched 基礎模型 + distillation = 2026 年快速 text-to-image 的預設做法。** 每家主要廠商都上線這個組合：SD3-Turbo（SD3 + flow + distillation）、Flux-schnell（Flux-dev + rectified-flow 拉直）、CogView-4-Flash。純 diffusion 的基礎模型只存在於舊的 checkpoint。

## 延伸閱讀 (Further Reading)

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3，大規模的 rectified flow。
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — 涵蓋 FM + diffusion 的通用框架。
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — diffusion / flow 的 1 步 distillation。
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo 變體。
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — 正式環境中的 flow matching。
