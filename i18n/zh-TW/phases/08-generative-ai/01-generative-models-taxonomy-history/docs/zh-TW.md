# 生成模型：分類與歷史 (Generative Models — Taxonomy & History)

> 每一個圖像模型、文字模型、影片模型和 3D 模型，都落在五個類別其中之一。選錯類別，你會跟數學纏鬥好幾週。選對類別，過去十二年這個領域的進展就會在你腦中整齊疊起來。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 2 (ML Fundamentals), Phase 3 (Deep Learning Core), Phase 7 · 14 (Transformers)
**Time:** ~45 分鐘

## 問題 (The Problem)

生成模型只做一件事：給定從某個未知分佈 $p_{\text{data}}(x)$ 抽出的訓練樣本，輸出看起來像來自同一個分佈的新樣本。人臉、句子、MIDI 檔、蛋白質結構，瞇起眼睛看都是同一個問題。

麻煩在於，$p_{\text{data}}$ 所在的空間有數百萬個維度（一張 512x512 的 RGB 圖像約有 786k 個維度），樣本只落在這個空間裡一個很薄的 manifold 上，而你手上大概只有 10M 筆範例。想硬算出密度是沒希望的。每一種生成模型都是一種妥協：把一個難題換成一個稍微沒那麼難的題目。

過去十二年留下來五個家族。知道每個家族做了哪種妥協，你就知道它為什麼在某些任務上贏、在另一些任務上崩潰。

## 核心概念 (The Concept)

![生成模型的五個家族：依建模對象分類](../assets/taxonomy.svg)

**1. 顯式密度，可精確計算 (Explicit density, tractable)。** 把 $\log p(x)$ 寫成可以真的算出來的總和。Autoregressive 模型（PixelCNN、WaveNet、GPT）分解成 $p(x) = \prod_i p(x_i \mid x_{<i})$。Normalizing flow（RealNVP、Glow）則把 $p(x)$ 建成一個簡單基底分佈的可逆轉換。優點：likelihood 精確，訓練 loss 乾淨。缺點：autoregressive 的推論是循序進行的（序列長時很慢），flow 需要可逆的架構（對架構限制很大）。

**2. 顯式密度，近似 (Explicit density, approximate)。** 從下方界定 $\log p(x)$（ELBO），然後最佳化這個下界。VAE（Kingma 2013）用 encoder-decoder 搭配 variational posterior。Diffusion model（DDPM，Ho 2020）訓練一個 denoiser，隱含地最佳化加權的 ELBO。Diffusion 是 2026 年圖像、影片和 3D 的主流骨幹。

**3. 隱式密度 (Implicit density)。** 完全跳過密度：學一個產生樣本的 generator $G(z)$，和一個分辨真假的 discriminator $D(x)$。這就是 GAN（Goodfellow 2014）。推論很快（一次 forward pass），但訓練出了名地不穩定。即使到了 2026 年，StyleGAN 1/2/3 在固定領域的擬真度（人臉、臥室）上仍是 state of the art。

**4. Score-based / 連續時間 (Score-based / continuous-time)。** 直接學 log-density 的梯度 $\nabla_x \log p(x)$（也就是 score）。Song & Ermon (2019) 證明 score matching 可以把 diffusion 推廣成 SDE。Flow matching（Lipman 2023）是 2024-2026 年的熱門：訓練不需要模擬、路徑更直、sampling 比 DDPM 快 4 到 10 倍。Stable Diffusion 3、Flux、AudioCraft 2 都使用 flow matching。

**5. 離散編碼上的 token-based autoregressive (Token-based autoregressive over discrete codes)。** 用 VQ-VAE 或 residual quantizer 把高維資料壓縮成一小段離散 token 序列，再用 Transformer 對 token 序列建模。Parti、MuseNet、AudioLM、VALL-E、Sora 的 patch tokenizer 都是這樣做的。這就是第 1 類加上一個學出來的 tokenizer。

## 簡史 (A brief history)

| 年份 | 模型 | 為什麼重要 |
|------|------|-----------|
| 2013 | VAE (Kingma) | 第一個訓練 loss 可用的深度生成模型。 |
| 2014 | GAN (Goodfellow) | 隱式密度，沒有 likelihood，卻產生驚人銳利的樣本。 |
| 2015 | DRAW, PixelCNN | 循序式圖像生成。 |
| 2017 | Glow, RealNVP | 可逆 flow；網路加深後仍有精確 likelihood。 |
| 2017 | Progressive GAN | 第一批百萬畫素等級的人臉。 |
| 2019 | StyleGAN / StyleGAN2 | 擬真人臉，在這個單一領域至今仍難以超越。 |
| 2020 | DDPM (Ho) | Diffusion 變得實用。 |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image 進入主流。 |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion 加上文字條件，成為日常用品。 |
| 2022 | ControlNet, LoRA | 對預訓練的 diffusion model 做精細控制。 |
| 2023 | SDXL, Midjourney v5, Flow matching | 規模加上更好的訓練動態。 |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | 影片 diffusion；flow matching 勝出。 |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | 達到正式產品等級的影片。 |
| 2026 | Consistency + Rectified Flow | 從 diffusion 骨幹做一步 sampling。 |

## 五個問題的快速分診 (The five-question triage)

新的生成模型論文出來時，先回答這五個問題，再去讀方法章節。

1. **建模的對象是什麼？** 像素、latent、離散 token、3D Gaussian、mesh，還是波形？
2. **密度是顯式還是隱式？** 他們有寫出 $\log p(x)$ 嗎？
3. **Sampling：一次到位還是迭代？** 迭代代表推論較慢；一次到位通常代表 adversarial 或經過 distillation。
4. **條件：無條件、類別、文字、圖像、姿態？** 這會決定 loss 和架構的骨架。
5. **評估：FID、CLIP score、IS、人類偏好，還是任務準確率？** 每一種都有已知的失效模式（見第 14 課）。

這個階段的每一堂課你都會重新回答這五個問題。到最後，它們會變成反射動作。

```figure
autoencoder-bottleneck
```

## 動手做 (Build It)

這堂課的程式是一個輕量的視覺化：用三種玩具方法（kernel density、離散 histogram，以及一個「類 GAN」的最近樣本 generator），從樣本擬合一個 1-D 的混合高斯，讓你在一個畫面就印得完的問題上，看出顯式密度與隱式密度的差別。

執行 `code/main.py`。它會從雙峰高斯混合抽出 2000 個樣本，然後印出：

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

注意：前兩種方法可以讓你問「這個點有多可能？」。第三種不行。這就是 *顯式 vs 隱式* 的差別，之後每一堂課都會用到。

## 實際使用 (Use It)

2026 年，哪個任務該選哪個家族？

| 任務 | 最佳家族 | 原因 |
|------|---------|------|
| 擬真人臉，窄領域 | StyleGAN 2/3 | 仍是最銳利、推論最快的。 |
| 通用 text-to-image | Latent diffusion + flow matching | SD3、Flux.1、DALL-E 3。 |
| 快速 text-to-image | Rectified flow + distillation | SDXL-Turbo、SD3-Turbo、LCM。 |
| Text-to-video | Diffusion Transformer + flow matching | Sora、Veo 2、Kling。 |
| 語音 + 音樂 | Token-based AR (AudioLM, VALL-E, MusicGen) 或 flow matching (AudioCraft 2) | 離散 token 的擴展成本低。 |
| 3D 場景 | Gaussian Splatting 擬合、diffusion prior | 3D-GS 用於重建，diffusion 用於新視角。 |
| 密度估計（不需要 sampling） | Flows | 唯一有精確 $\log p(x)$ 的家族。 |
| 模擬 / 物理 | Flow matching、score SDE | 直線路徑、平滑的向量場。 |

## 交付 (Ship It)

存成 `outputs/skill-model-chooser.md`。

這個 skill 接收一段任務描述，輸出：(1) 該用哪個家族、(2) 三個開源與三個託管選項的排名清單、(3) 你該留意的可能失效模式，以及 (4) 運算與時間預算。

## 練習 (Exercises)

1. **簡單。** 針對這五個產品，找出它們的家族與骨幹：ChatGPT image、Midjourney v7、Sora、Runway Gen-3、ElevenLabs。證據要來自公開的技術報告。
2. **中等。** 你明天要讀的論文宣稱 sampling 比 diffusion 快 100 倍。寫下三個問題，檢查這個加速在加入條件和高解析度之後是否還成立。
3. **困難。** 選一個你在意的領域（例如蛋白質結構、CAD、分子、軌跡）。針對該領域目前的 SOTA 模型回答這五個分診問題，並草擬更好的模型會改變什麼。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Generative model | 「它會做出新東西」 | 學出 $p_{\text{data}}(x)$ 的 sampler，也可以選擇性提供 $\log p(x)$。 |
| Explicit density | 「你可以算出來」 | 模型提供封閉形式或可計算的 $\log p(x)$。 |
| Implicit density | 「GAN 那一類」 | 只有 sampler，沒辦法對給定的點計算 $p(x)$。 |
| ELBO | 「Evidence lower bound」 | $\log p(x)$ 的一個可計算下界；VAE 和 diffusion 都在最佳化它。 |
| Score | 「Log-density 的梯度」 | $\nabla_x \log p(x)$；diffusion 和 SDE 模型學的就是這個場。 |
| Manifold hypothesis | 「資料住在一個曲面上」 | 高維資料集中在低維 manifold 上；這也是降維有效的原因。 |
| Autoregressive | 「預測下一塊」 | 把聯合分佈分解成條件分佈的乘積。 |
| Latent | 「壓縮後的編碼」 | 低維表示，decoder 可以從它重建輸入。 |

## 實務筆記 (Production note)：五個家族，五種推論形狀

每個家族對應不同的推論伺服器成本曲線。production-inference 的文獻把 LLM 推論拆成 prefill + decode；同樣的拆法也適用於這裡：

- **Autoregressive（第 1 和第 5 類）。** 循序 decode 主宰延遲；KV-cache、continuous batching 和 speculative decoding 都可以直接套用。
- **VAE / diffusion / flow-matching（第 2 和第 4 類）。** 沒有 LLM 意義上的 decode。成本 = `num_steps × step_cost`，而 `step_cost` 是在完整 latent 解析度上做一次 transformer 或 U-Net forward。實務上能調的旋鈕是步數（DDIM / DPM-Solver / distillation）、batch size 和精度（bf16 / fp8 / int4）。
- **GAN（第 3 類）。** 一次 forward pass。沒有排程，沒有 KV-cache。TTFT ≈ 總延遲。這就是 StyleGAN 在窄領域的使用體驗上仍然勝出的原因。

當你在論文摘要看到「比 diffusion 更快」，把它翻譯成「更少的步數 × 相同的單步成本」，或「相同的步數 × 更便宜的單步成本」。其他都是行銷話術。

## 延伸閱讀 (Further Reading)

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — GAN 論文。
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — VAE 論文。
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM 論文。
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — 把 diffusion 視為 SDE。
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching 論文。
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3。
