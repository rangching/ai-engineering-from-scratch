# 影片生成 (Video Generation)

> 圖像是 2-D tensor，影片是 3-D 的。理論相同；運算量卻難 10 到 100 倍。OpenAI 的 Sora（2024 年 2 月）證明了這是可能的。到了 2026 年，Veo 2、Kling 1.5、Runway Gen-3、Pika 2.0 和 WAN 2.2 都能從文字生成 1080p 的正式影片，而開放權重的陣營（CogVideoX、HunyuanVideo、Mochi-1、WAN 2.2）落後約 12 個月。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 7 · 09 (ViT), Phase 8 · 06 (DDPM)
**Time:** ~45 分鐘

## 問題 (The Problem)

一段 10 秒、24fps 的 1080p 影片，是 240 張 1920×1080×3 像素的畫格。每個片段約 1.5 GB 的原始資料。像素空間的 diffusion 不可行。你需要：

1. **時空壓縮。** 一個能對影片（而不是單張畫格）編碼的 VAE，把影片編碼成時空 patch 的序列。
2. **時間上的連貫性。** 畫格之間要在數秒內共享內容、光線和物體身分。網路必須對動作建模。
3. **運算預算。** 同樣的模型大小，影片訓練比圖像貴 10 到 100 倍。
4. **條件。** 文字、圖像（第一格）、音訊，或另一段影片。大多數正式的模型四種都接受。

解決這個問題的架構，是套用在時空 patch 上的 **Diffusion Transformer (DiT)**，並在龐大的（prompt, caption, video）資料集上訓練。Diffusion loss 與第 06 課相同。

## 核心概念 (The Concept)

![影片 diffusion：patchify、DiT、decode](../assets/video-generation.svg)

### Patchify

用 3D VAE（學出來的時空壓縮）對影片編碼。Latent 的形狀是 `[T_latent, H_latent, W_latent, C_latent]`。切成大小為 `[t_p, h_p, w_p]` 的 patch。對 Sora 風格的模型，`t_p = 1`（逐格 patch）或 `t_p = 2`（每兩格）。一段 10 秒的 1080p 影片會壓縮成約 20,000-100,000 個 patch。

### 時空 DiT (Spatiotemporal DiT)

Transformer 處理攤平的 patch 序列。每個 patch 都有 3D 位置 embedding（時間 + y + x）。Attention 通常會被分解：

- **空間 attention**：在每一格的 patch 之內。
- **時間 attention**：跨畫格、在相同空間位置。
- **完整 3D attention** 貴 16 到 100 倍；只用在低解析度或研究中。

### 文字條件

使用大型文字 encoder 的 cross-attention（Sora 用 T5-XXL，CogVideoX-5B 也用 T5-XXL）。長 prompt 很重要：Sora 的訓練集有 GPT 生成的密集重新標註，平均每個片段 200 個 token。

### 訓練

在時空 latent 上使用標準 diffusion loss（ε 或 v prediction）。資料：網路影片 + 約 1 億個精選片段 + 合成的文字標註。運算量：即使是小型研究也需要 10,000+ GPU 小時；Sora 等級則是 100,000+。

## 2026 年的正式產品版圖 (The 2026 production landscape)

| 模型 | 日期 | 最長時間 | 最高解析度 | 開放權重？ | 特色 |
|------|------|---------|-----------|-----------|------|
| Sora (OpenAI) | 2024-02 | 60s | 1080p | 否 | 第一個在大規模下展現世界模擬器特性的模型 |
| Sora Turbo | 2024-12 | 20s | 1080p | 否 | 正式版 Sora，推論快 5 倍 |
| Veo 2 (Google) | 2024-12 | 8s | 4K | 否 | 2025 年品質與物理表現最高 |
| Veo 3 | 2025 Q3 | 15s | 4K | 否 | 原生音訊與更強的鏡頭控制 |
| Kling 1.5 / 2.1 (Kuaishou) | 2024-2025 | 10s | 1080p | 否 | 2025 年 Q1 最佳的人體動作 |
| Runway Gen-3 Alpha | 2024-06 | 10s | 768p | 否 | 上層有專業影片工具 |
| Pika 2.0 | 2024-10 | 5s | 1080p | 否 | 角色一致性最強 |
| CogVideoX (THUDM) | 2024 | 10s | 720p | 是 (2B, 5B) | 第一個開放的 5B 規模影片模型 |
| HunyuanVideo (Tencent) | 2024-12 | 5s | 720p | 是 (13B) | 2024 年底開放權重的 SOTA |
| Mochi-1 (Genmo) | 2024-10 | 5.4s | 480p | 是 (10B) | 授權最寬鬆 |
| WAN 2.2 (Alibaba) | 2025-07 | 5s | 720p | 是 | 2025 年中最強的開放模型 |

開放權重縮小差距的速度比圖像領域更快：到 2026 年中，HunyuanVideo + WAN 2.2 的 LoRA 已經驅動了大部分開源工作流程。

```figure
video-diffusion-denoise
```

## 動手做 (Build It)

`code/main.py` 模擬時空 DiT 的核心概念：把一小段合成影片 patchify，加上每個 patch 的位置 embedding，並用 transformer 風格、跨 patch 的 attention 對整個序列去雜訊。不用 numpy，純 Python。我們展示，當相鄰畫格的 patch 共用 denoiser 和位置 embedding 時，即使在 1-D 也會浮現時間上的連貫性。

### 步驟 1：把合成的 1-D「影片」patchify

```python
def make_video(T_frames=8, rng=None):
    # a "video" is a sequence of 1-D values following a smooth trajectory
    base = rng.gauss(0, 1)
    return [base + 0.3 * t + rng.gauss(0, 0.1) for t in range(T_frames)]
```

### 步驟 2：每一格的位置 embedding

```python
def pos_embed(t, dim):
    return sinusoidal(t, dim)
```

### 步驟 3：denoiser 看到整個序列

我們的小型網路不是獨立地對每一格去雜訊，而是把所有畫格的數值連同它們的位置 embedding 串接起來，一起預測所有畫格的雜訊。

### 步驟 4：時間連貫性測試

訓練後，sample 一段影片。量測相鄰畫格之間的差異。如果模型學到了時間結構，這些差異會比獨立 sampling 每一格時來得小。

## 常見陷阱 (Pitfalls)

- **獨立逐格 sampling = 閃爍。** 如果你對每一格分別跑圖像 diffusion，輸出會閃爍，因為每一格的雜訊是獨立的。影片 diffusion 透過 attention 或共享雜訊把畫格耦合起來，解決了這件事。
- **天真的 3D attention = OOM。** 對 10 秒 1080p latent 做完整 3D attention，是數千億次運算。要分解成空間 + 時間。
- **資料標註比資料量更重要。** Sora 相較先前工作最主要的升級，是用細節多出約 10 倍的標註來訓練（GPT-4 重新標註的片段）。OpenAI 的技術報告對此說得很明確。
- **第一格條件。** 大多數正式的模型也接受一張圖像作為第一格。這就是「image-to-video」模式；訓練中包含這個變體。
- **物理漂移。** 長片段（>10s）會累積細微的不一致。滑動視窗生成 + 關鍵格錨定會有幫助。

## 實際使用 (Use It)

| 使用情境 | 2026 年的選擇 |
|---------|-------------|
| 最高品質的 text-to-video，託管 | Veo 3 或 Sora |
| 鏡頭可控的電影感 | 搭配 motion brush 的 Runway Gen-3 |
| 跨片段的角色一致性 | Pika 2.0 或 Kling 2.1 |
| 開放權重、快速 fine-tune | WAN 2.2 + LoRA |
| Image-to-video | WAN 2.2-I2V、Kling 2.1 I2V 或 Runway |
| 音訊到影片的對嘴 | Veo 3（原生音訊）或專用的對嘴模型 |
| 影片編輯 | Runway Act-Two、Kling Motion Brush、Flux-Kontext（靜態畫格） |

在品質相當的情況下，每秒影片的成本在 2024 到 2026 年間下降了 20 倍。

## 交付 (Ship It)

存成 `outputs/skill-video-brief.md`。這個 skill 接收影片簡報（長度、長寬比、風格、鏡頭規劃、主體一致性、音訊），輸出：模型 + 託管方式、prompt 骨架（鏡頭語言、主體描述、動作描述詞）、seed + 可重現流程，以及逐格的 QA 檢查清單。

## 練習 (Exercises)

1. **簡單。** 在 `code/main.py` 中，比較 (a) 獨立逐格 sampling 與 (b) 整個序列聯合 sampling 的相鄰畫格差異。回報差異的平均值與變異數。
2. **中等。** 加入第一格條件：把第 0 格固定為給定的值，再 sample 其餘畫格。量測這個固定值如何傳播。
3. **困難。** 使用 HuggingFace diffusers 在本機 GPU 上跑 CogVideoX-2B。對 720p、6 秒的片段計時 20 個推論步驟。分析時空 attention 的效能，找出瓶頸。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Video VAE | 「3-D VAE」 | 把 `(T, H, W, C)` 壓縮成時空 latent 的 encoder。 |
| Patches | 「那些 token」 | latent 中固定大小的 3-D 區塊；DiT 的輸入。 |
| Factorized attention | 「空間 + 時間」 | 先在空間上做 attention，再在時間上做；略過完整的 3-D attention。 |
| Image-to-video (I2V) | 「讓這張照片動起來」 | 模型接收一張圖像 + 文字，輸出以它為開頭的影片。 |
| Keyframe conditioning | 「錨定畫格」 | 固定特定畫格，以控制影片的走向。 |
| Motion brush | 「方向提示」 | 使用者在圖像上畫出動作向量的 UI 輸入。 |
| Re-captioning | 「密集標註」 | 用 LLM 以詳細的 prompt 重新標註訓練片段。 |
| Flicker | 「時間上的瑕疵」 | 畫格之間的不一致；用耦合的去雜訊來修正。 |

## 實務筆記 (Production note)：影片 latent 是記憶體頻寬問題

一段 10 秒、24 fps 的 1080p 片段，是 240 格 × 1920 × 1080 × 3 ≈ 1.5 GB 的原始像素。經過 4× 影片 VAE 壓縮（`2 × spatial × 2 × temporal`）後，每個請求的 latent 約 100 MB。把它通過時空 DiT，batch 1 跑 30 步，你每步要透過 HBM 搬運約 3 GB：瓶頸是記憶體頻寬，而不是 FLOPs。

三個正式環境的旋鈕，都直接來自 production-inference 文獻的推論章節：

- **跨 DiT 的 TP。** Text-to-video 模型動輒 ≥10B 參數。在 4 張 H100 上用 TP=4 是標準做法；405B 等級的模型用 PP=2 × TP=2。每步延遲大致隨 TP 線性下降，直到碰到 all-reduce 的牆。
- **畫格 batching = continuous batching。** 在生成時，影片在概念上是一批由 attention 連結的畫格。Continuous batching（in-flight scheduling）同樣適用：如果模型架構允許滑動視窗生成，就可以在第 `t-1` 格回傳的同時，開始渲染第 `t+1` 格。
- **片段層級的 prefill cache。** 對 image-to-video 來說，第一格條件類似 LLM 的 prompt prefill：算一次，在時間 decoder 的各次 pass 中重複使用。這實際上就是影片的 KV-cache。

## 延伸閱讀 (Further Reading)

- [Brooks et al. (2024). Video generation models as world simulators](https://openai.com/index/video-generation-models-as-world-simulators/) — Sora 技術報告。
- [Yang et al. (2024). CogVideoX: Text-to-Video Diffusion Models with An Expert Transformer](https://arxiv.org/abs/2408.06072) — CogVideoX。
- [Kong et al. (2024). HunyuanVideo: A Systematic Framework for Large Video Generative Models](https://arxiv.org/abs/2412.03603) — HunyuanVideo。
- [Genmo (2024). Mochi 1 preview model card](https://huggingface.co/genmo/mochi-1-preview) — Mochi-1。
- [Alibaba (2025). WAN 2.2](https://wanvideo.io/) — 2025 年中的開放 SOTA。
- [Ho, Salimans, Gritsenko et al. (2022). Video Diffusion Models](https://arxiv.org/abs/2204.03458) — 影片 diffusion 的開創性論文。
- [Blattmann et al. (2023). Align your Latents (Video LDM)](https://arxiv.org/abs/2304.08818) — Stable Video Diffusion 的前身。
