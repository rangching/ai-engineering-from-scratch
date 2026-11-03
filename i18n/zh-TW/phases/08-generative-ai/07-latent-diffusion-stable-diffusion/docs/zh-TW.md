# Latent Diffusion 與 Stable Diffusion (Latent Diffusion & Stable Diffusion)

> 在 512×512 圖像上做像素空間的 diffusion，簡直是運算上的戰爭罪行。Rombach et al. (2022) 注意到，生成圖像並不需要全部 786k 個維度，只要足夠捕捉語意結構就好，其餘交給另一個 decoder。把 diffusion 放進 VAE 的 latent 空間裡跑。就這一個想法，就是 Stable Diffusion。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 02 (VAE), Phase 8 · 06 (DDPM), Phase 7 · 09 (ViT)
**Time:** ~75 分鐘

## 問題 (The Problem)

512² 的像素空間 diffusion，代表 U-Net 要處理形狀為 `[B, 3, 512, 512]` 的 tensor。對一個 500M 參數的 U-Net，每個 sampling 步驟約 100 GFLOPS。五十步就是每張圖像 5 TFLOPS。拿十億張圖像去訓練，運算帳單會荒謬到不行。

這些 FLOPs 大多花在讓感知上不重要的細節穿過網路，也就是有損 VAE 可以壓縮掉的高頻紋理。Rombach 的想法是：訓練一次 VAE（*第一階段*），凍結它，然後完全在 4 通道 64×64 的 latent 空間裡跑 diffusion（*第二階段*）。同樣的 U-Net，只有 1/16 的像素，在相當的品質下 FLOPs 大約少 64 倍。

這就是 Stable Diffusion 的配方。SD 1.x / 2.x 在 `64×64×4` 的 latent 上使用 860M 的 U-Net，SDXL 在 `128×128×4` 上使用 2.6B 的 U-Net，SD3 則把 U-Net 換成搭配 flow matching 的 Diffusion Transformer (DiT)。Flux.1-dev（Black Forest Labs, 2024）釋出了 12B 參數的 DiT-MMDiT。全都跑在同一個兩階段的底座上。

## 核心概念 (The Concept)

![Latent diffusion：VAE 壓縮 + 在 latent 空間做 diffusion](../assets/latent-diffusion.svg)

**兩個階段，分開訓練。**

1. **第一階段：VAE。** Encoder $E(x) \to z$，decoder $D(z) \to x$。目標壓縮：每個空間軸縮小 8 倍，並調整通道數，使 latent 總大小約為像素數的 1/16。Loss = 重建（L1 + LPIPS perceptual）+ KL（權重很小，這樣 $z$ 不會被強迫變得太像 Gaussian，因為我們不需要從 $z$ 做精確的 sampling）。通常會再加上 adversarial loss，讓解碼後的圖像銳利。

2. **第二階段：在 $z$ 上做 diffusion。** 把 $z = E(x_{\text{real}})$ 當作資料。訓練一個 U-Net（或 DiT）對 $z_t$ 去雜訊。推論時：透過 diffusion 抽出 $z_0$，然後 $x = D(z_0)$。

**文字條件。** 多了兩個元件。一個凍結的文字 encoder（SD 1.x 用 CLIP-L，SD 2/XL 用 CLIP-L+OpenCLIP-G，SD3 和 Flux 用 T5-XXL）。以及 cross-attention 注入：每個 U-Net 區塊取 `[Q = 圖像特徵, K = V = 文字 token]` 並把它們混合。這些 token 是文字影響圖像的唯一途徑。

**Loss function 與第 06 課完全相同。** 同樣是對雜訊的 DDPM / flow matching MSE。你只是換了資料的領域。

## 架構變體 (Architecture variants)

| 模型 | 年份 | 骨幹 | Latent 形狀 | 文字 encoder | 參數量 |
|------|------|------|------------|-------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | 相同 | 1-4 步 sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B，1-4 步 |

趨勢是：把 U-Net 換成 DiT（對 latent patch 做 transformer）、擴大文字 encoder（在 prompt 遵循度上 T5 勝過 CLIP）、增加 latent 通道數（4 → 16 提供更多細節空間）。

```figure
noise-schedule
```

## 動手做 (Build It)

`code/main.py` 在第 06 課的 DDPM 之上，疊了一個玩具 1-D「VAE」（為了示範，encoder 固定為乘以 0.5，decoder 則是它的精確反函數；真正的 VAE 會是 conv 網路），並加入搭配 classifier-free guidance 的類別條件。它顯示出，不管你是在原始 1-D 數值上跑，還是在編碼後的數值上跑，同一個 diffusion loss 都有效。這就是關鍵洞見。

### 步驟 1：encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

真正的 VAE 有訓練出來的權重。就教學而言，這個線性映射已足以說明，diffusion 作用在 $z$ 上，完全不在乎原本的資料空間。

### 步驟 2：在 `z` 空間做 diffusion

與第 06 課相同的 DDPM。網路看到的資料是 $z = E(x)$。抽出 $z_0$ 之後，用 $D(z_0)$ 解碼。

### 步驟 3：classifier-free guidance

訓練時，有 10% 的機率丟掉類別標籤（換成 null token）。推論時，同時計算 $\epsilon_{\text{cond}}$ 和 $\epsilon_{\text{uncond}}$，然後：

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

$w = 0$ = 沒有 guidance（完整多樣性），$w = 3$ = 預設，$w = 7+$ = 飽和 / 過度銳利。

### 步驟 4：文字條件（概念，不是程式）

把類別標籤換成凍結的文字 encoder 輸出。透過 cross-attention 把文字 embedding 餵給 U-Net：

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

這就是類別條件式 diffusion 模型和 Stable Diffusion 之間唯一實質的差異。

## 常見陷阱 (Pitfalls)

- **VAE scale 不一致。** SD 1.x 的 VAE 在 encode 之後會乘上一個縮放常數（`scaling_factor ≈ 0.18215`）。忘了這個，U-Net 就會在變異數嚴重錯誤的 latent 上訓練。每個 checkpoint 都附有這個值。
- **文字 encoder 默默出錯。** SD3 需要 T5-XXL 且 >=128 個 token，退回只用 CLIP 會有損失。務必檢查 `use_t5=True`，否則 prompt 的忠實度會暴跌。
- **混用 latent 空間。** SDXL、SD3、Flux 都使用不同的 VAE。在 SDXL latent 上訓練的 LoRA 無法用在 SD3 上。Hugging Face diffusers 0.30+ 會拒絕載入不相符的 checkpoint。
- **CFG 太高。** `w > 10` 會產生飽和、油膩的圖像，並以犧牲多樣性為代價過度擬合 prompt。最佳區間是 `w = 3-7`。
- **Negative prompt 洩漏。** 空的 negative prompt 會變成 null token；有內容的 negative prompt 則會變成 $\epsilon_{\text{uncond}}$。這兩者並不相同；有些管線會默默地預設使用 null。

## 實際使用 (Use It)

2026 年的正式環境技術組合：

| 目標 | 建議骨幹 |
|------|---------|
| 窄領域、成對資料、從零訓練模型 | SDXL fine-tune (LoRA / 完整)，最快上線 |
| 開放領域 text-to-image，開放權重 | Flux.1-dev (12B, Apache / 非商用) 或 SD3.5-Large |
| 最快推論，開放權重 | Flux.1-schnell (1-4 步, Apache) 或 SDXL-Lightning |
| 最佳 prompt 遵循度，託管服務 | GPT-Image / DALL-E 3（仍然是）、Midjourney v7、Imagen 4 |
| 編輯工作流程 | Flux.1-Kontext (Dec 2024)，原生接受圖像 + 文字 |
| 研究、基準線 | SD 1.5，古老但研究得很透徹 |

## 交付 (Ship It)

存成 `outputs/skill-sd-prompter.md`。這個 skill 接收文字 prompt + 目標風格，輸出：模型 + checkpoint、CFG scale、sampler、negative prompt、解析度、選用的 ControlNet/IP-Adapter 組合，以及逐步的 QA 檢查清單。

## 練習 (Exercises)

1. **簡單。** 用 guidance $w \in \lbrace 0, 1, 3, 7, 15 \rbrace$ 執行 `code/main.py`。記錄各類別的樣本平均值。$w$ 到多少時，類別平均值會偏離真實資料的平均值？
2. **中等。** 把玩具線性 encoder 換成帶有重建 loss 的 tanh-MLP encoder/decoder 配對。在新的 latent 上重新訓練 diffusion。樣本品質有改變嗎？
3. **困難。** 用 diffusers 建立真正的 Stable Diffusion 推論：載入 `sdxl-base`，用 CFG=7 跑 30 個 Euler 步驟，並計時。再換成 `sdxl-turbo`，4 步、CFG=0。相同主題、不同品質，描述改變了什麼、以及為什麼。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| First stage | 「VAE」 | 訓練好的 encoder/decoder 配對；把 512² 壓縮成 64²。 |
| Second stage | 「U-Net」 | 作用在 latent 空間上的 diffusion 模型。 |
| CFG | 「Guidance scale」 | $(1+w) \cdot \epsilon_{\text{cond}} - w \cdot \epsilon_{\text{uncond}}$；調整條件的強度。 |
| Null token | 「空的 prompt embedding」 | 用於 $\epsilon_{\text{uncond}}$ 的無條件 embedding。 |
| Cross-attention | 「文字如何進來」 | 每個 U-Net 區塊把文字 token 當作 K 和 V 來做 attention。 |
| DiT | 「Diffusion Transformer」 | 用對 latent patch 的 transformer 取代 U-Net；擴展性更好。 |
| MMDiT | 「Multi-modal DiT」 | SD3 的架構：文字與圖像各有一條流，並做 joint attention。 |
| VAE scaling factor | 「魔術數字」 | 把 latent 除以約 5.4，讓 diffusion 在單位變異數的空間中運作。 |

## 實務筆記 (Production note)：在 8GB 消費級 GPU 上跑 Flux-12B

Flux 的參考整合是經典的「我有一張消費級 GPU，這能上線嗎？」配方。訣竅是把 production inference 文獻列出的同一套三個旋鈕，套用到 diffusion DiT 上：

1. **分階段載入。** Flux 有四個網路，永遠不需要同時待在 VRAM：T5-XXL 文字 encoder（fp32 約 10 GB）、CLIP-L（小）、12B 的 MMDiT，以及 VAE。先 encode prompt，*刪掉* encoder，載入 DiT，去雜訊，*刪掉* DiT，載入 VAE，decode。消費級 8GB GPU 一次只放得下一個階段。
2. **透過 bitsandbytes 做 4-bit 量化。** 在 T5 encoder 和 DiT 上都使用 `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)`。記憶體降低 8 倍，根據 Aritra 的 benchmark（連結在 notebook 中），text-to-image 的品質下降幾乎察覺不到。
3. **CPU offload。** `pipe.enable_model_cpu_offload()` 會隨著每次 forward pass 的進行，自動在 CPU 與 GPU 之間交換模組。多增加 10-20% 的延遲，但至少讓管線跑得起來。

記憶體計算如下：`10 GB T5 / 8 = 1.25 GB`（量化後），`12 B params × 0.5 bytes = ~6 GB`（量化後的 DiT），再加上 activation。以 stas00 的說法，這是 TP=1 推論的極端：沒有 model parallelism，量化到最大。正式環境你會在 H100 上跑 TP=2 或 TP=4；如果是單一台開發用筆電，這就是配方。

## 延伸閱讀 (Further Reading)

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion。
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL。
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3、MMDiT。
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG。
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — Flux.1 系列。
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — 上述每個 checkpoint 的參考實作。
