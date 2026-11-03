# Inpainting、Outpainting 與圖像編輯 (Inpainting, Outpainting & Image Editing)

> Text-to-image 做新東西。Inpainting 修舊東西。在正式環境中，70% 可計費的圖像工作都是編輯：換背景、移除 logo、延伸畫布、重新生成一隻手。Inpainting 就是 diffusion 賺回成本的地方。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 8 · 08 (ControlNet & LoRA)
**Time:** ~75 分鐘

## 問題 (The Problem)

客戶傳來一張完美的商品照，背景卻有個分散注意力的招牌。你想把招牌擦掉，其他部分則要逐像素保持不變。你不能從零重跑 text-to-image，因為結果的顏色、光線、商品角度都會不同。你想 *只* 重新生成被遮罩的區域，而且要求重新生成的內容尊重周圍的脈絡。

這就是 inpainting。變體包括：

- **Inpainting。** 在遮罩內重新生成，保留遮罩外的像素。
- **Outpainting。** 在遮罩外（或畫布之外）重新生成，保留內部。
- **圖像編輯 (Image editing)。** 重新生成整張圖像，但保持與原圖在語意或結構上的忠實度（SDEdit、InstructPix2Pix）。

2026 年每個 diffusion 管線都附有 inpainting 模式。Flux.1-Fill、Stable Diffusion Inpaint、SDXL-Inpaint、DALL-E 3 Edit。它們依據相同的原理運作。

## 核心概念 (The Concept)

![Inpainting：感知遮罩的去雜訊，並重新注入脈絡](../assets/inpainting.svg)

### 天真的做法（以及為什麼是錯的）

用遮罩跑標準的 text-to-image。在每個 sampling 步驟，把有雜訊 latent 中未遮罩的區域，換成經過正向 diffusion 的乾淨圖像。這樣可以動……但效果很差。邊界瑕疵會滲出來，因為模型對遮罩區域內有什麼毫無資訊。

### 正規的 inpainting 模型

訓練一個修改過的 U-Net，輸入是 9 個通道而不是 4 個：

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

額外的通道是 VAE 編碼後的來源圖像副本，加上一個單通道的遮罩。訓練時，你隨機遮住圖像的區域，並訓練模型只對被遮罩的區域去雜訊，未遮罩的區域則作為乾淨的條件訊號給定。推論時，模型可以「看到」遮罩區域周圍有什麼，並產生連貫的補全。

SD-Inpaint、SDXL-Inpaint、Flux-Fill 都使用這種 9 通道（或類似）的輸入。Diffusers 的 `StableDiffusionInpaintPipeline`、`FluxFillPipeline`。

### SDEdit (Meng et al., 2022)：免費的編輯

把來源圖像加上雜訊，加到某個中間的 `t`，然後用新的 prompt 從 `t` 到 0 跑反向 chain。不需要重新訓練。起始 `t` 的選擇，是在忠實度與創作自由度之間取捨：

- `t/T = 0.3` → 幾乎與來源相同，小幅度的風格變化
- `t/T = 0.6` → 中等程度的編輯，保留粗略結構
- `t/T = 0.9` → 從接近純雜訊生成，幾乎不保留來源

### InstructPix2Pix (Brooks et al., 2023)

在 `(input_image, instruction, output_image)` 三元組上 fine-tune 一個 diffusion model。推論時，同時以輸入圖像和文字指令（「讓它變成日落」、「加一隻龍」）作為條件。有兩個 CFG scale：圖像 scale 和文字 scale。

### RePaint (Lugmayr et al., 2022)

保留標準的無條件 diffusion model。在每個反向步驟中重新取樣：偶爾跳回到雜訊更多的狀態，再重新生成。這避免了邊界瑕疵。在你沒有訓練好的 inpainting 模型時使用。

```figure
inpaint-mask-reinject
```

## 動手做 (Build It)

`code/main.py` 在 5 維資料上實作一個玩具版 1-D inpainting 方案。我們在 5-D 混合資料上訓練 DDPM，每個樣本是來自兩個群集之一的 5 個 float。推論時，我們「遮住」5 個維度中的 2 個，在每一步注入未遮罩那 3 個維度經過 noisy-forward 的版本，並只重新生成被遮罩的維度。

### 步驟 1：5-D DDPM 資料

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### 步驟 2：在全部 5 個維度上訓練 denoiser

標準 DDPM。對於 5-D 的有雜訊輸入，網路輸出 5-D 的雜訊預測。

### 步驟 3：推論時使用感知遮罩的反向過程

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

這是天真的做法，在玩具 1-D 資料上有效。真正的圖像 inpainting 使用 9 通道輸入，因為紋理的連貫性更重要。

### 步驟 4：outpainting

Outpainting 就是遮罩反過來的 inpainting：遮住新的（先前不存在的）畫布，其餘部分用原圖填入。訓練目標完全相同。

## 常見陷阱 (Pitfalls)

- **接縫 (Seams)。** 天真的做法會留下可見的邊界，因為梯度資訊不會跨過遮罩流動。解法：把遮罩膨脹 8-16 個像素，或使用正規的 inpainting 模型。
- **遮罩洩漏。** 如果條件圖像未遮罩區域的品質低或有雜訊，會污染遮罩內的生成結果。要先去雜訊或稍微模糊。
- **CFG 與遮罩大小互相影響。** 小遮罩搭配高 CFG = 飽和的色塊。小幅度的編輯要降低 CFG。
- **SDEdit 的忠實度斷崖。** 從 `t/T = 0.5` 到 `t/T = 0.6`，可能就會失去主體的身分。要掃描並存 checkpoint。
- **Prompt 不一致。** Prompt 應該描述 *整張* 圖像，而不只是新內容。要寫「一隻貓坐在椅子上」，而不是「一隻貓」。

## 實際使用 (Use It)

| 任務 | 管線 |
|------|------|
| 移除物件，小遮罩 | SD-Inpaint 或 Flux-Fill，標準 prompt |
| 替換天空 | SD-Inpaint + "blue sky at sunset" |
| 延伸畫布 | SDXL outpaint 模式（8px 羽化）或搭配 outpaint 遮罩的 Flux-Fill |
| 重新生成手 / 臉 | 搭配重新描述主體的 prompt 的 SD-Inpaint + ControlNet-Openpose |
| 改變單一區域的風格 | 在遮罩區域上以 `t/T=0.5` 做 SDEdit |
| 「讓它變成日落」 | InstructPix2Pix 或 Flux-Kontext |
| 替換背景 | SAM 遮罩 → SD-Inpaint |
| 超高擬真度 | 最困難的情況用 Flux-Fill 或 GPT-Image（託管） |

SAM（Meta 的 Segment Anything，2023）+ diffusion inpaint 是 2026 年的去背管線。SAM 2（2024）可以處理影片。

## 交付 (Ship It)

存成 `outputs/skill-editing-pipeline.md`。這個 skill 接收原始圖像 + 編輯描述 + 選用的遮罩（或 SAM prompt），輸出：遮罩生成方式、基礎模型、CFG scale（圖像 + 文字）、SDEdit 的 t 或 inpainting 模式，以及 QA 檢查清單。

## 練習 (Exercises)

1. **簡單。** 在 `code/main.py` 中，把被遮罩的維度比例從 0.2 調到 0.8。比例到多少時，inpaint 的品質（被遮罩維度的殘差）會等同於無條件生成？
2. **中等。** 實作 RePaint：每 10 個反向步驟，往回跳 5 步（加雜訊）並重新去雜訊。量測它是否降低了遮罩邊緣的邊界殘差。
3. **困難。** 使用 Hugging Face diffusers，在 20 個人臉重新生成任務上比較：SD 1.5 Inpaint + ControlNet-Openpose 對 Flux.1-Fill。分別評分姿態遵循度與身分保留度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Inpainting | 「把洞補起來」 | 在遮罩內重新生成；保留遮罩外的像素。 |
| Outpainting | 「延伸畫布」 | 在畫布外重新生成；保留內部。 |
| 9-channel U-Net | 「正規的 inpainting 模型」 | 以 `noisy \| encoded-source \| mask` 作為輸入的 U-Net。 |
| SDEdit | 「帶雜訊層級的 img2img」 | 加雜訊到時間 `t`，再用新的 prompt 去雜訊。 |
| InstructPix2Pix | 「純文字編輯」 | 在（圖像, 指令, 輸出）三元組上 fine-tune 的 diffusion。 |
| RePaint | 「不用重新訓練」 | 在反向過程中定期重新加雜訊，以減少接縫。 |
| SAM | 「Segment Anything」 | 透過點擊或框選產生遮罩；搭配 inpaint 使用。 |
| Flux-Kontext | 「帶脈絡的編輯」 | 接受參考圖像 + 指令來編輯的 Flux 變體。 |

## 實務筆記 (Production note)：編輯管線對延遲很敏感

使用者編輯圖像時，期望來回在 5 秒以內。在 L4 上，1024² 的 30 步 SDXL-Inpaint 需要 3-4 秒，再加上 SAM 遮罩生成（~200 ms）以及 VAE encode/decode（合計 ~500 ms）。以正式環境的框架來看，這是受 TTFT 限制，而不是受吞吐量限制：batch 1、低並行度，要把每個階段都壓到最小：

- **SAM-H 是慢的那個。** 1024² 下的 SAM-H 約 200 ms；SAM-ViT-B 約 40 ms，品質略有損失。SAM 2（影片）會增加時間上的開銷；單張圖像編輯不要用它。
- **能省略 encode 就省略。** `pipe.image_processor.preprocess(img)` 會編碼成 latent。如果你有上一次生成的 latent（迭代式編輯 UI 很常見），直接透過 `latents=...` 傳入，就能省下一次 VAE encode。
- **遮罩膨脹對吞吐量也有影響。** 小遮罩意味著 U-Net forward pass 大半是浪費的（未遮罩的像素反正都被固定住了）。`diffusers` 的 `StableDiffusionInpaintPipeline` 不論如何都會跑完整個 U-Net；只有 9 通道的正規 inpaint 變體才會利用遮罩運算。
- **Flux-Kontext 是 2025 年的答案。** 對 `(source_image, instruction)` 做一次 forward pass：沒有獨立的遮罩，沒有 SDEdit 的雜訊掃描。在 H100 上約 1.5 秒就能完成一次編輯。架構上的教訓：把階段合併起來。

## 延伸閱讀 (Further Reading)

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — 免訓練的 inpainting。
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit。
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — 文字指令編輯。
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM，遮罩的來源。
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — 影片版 SAM。
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention 層級的編輯。
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — 2024 年的工具。
