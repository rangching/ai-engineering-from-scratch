# ControlNet、LoRA 與條件控制 (ControlNet, LoRA & Conditioning)

> 單靠文字是個笨拙的控制訊號。ControlNet 讓你複製一個預訓練的 diffusion model，並用深度圖、姿態骨架、塗鴉或邊緣圖來操控它。LoRA 讓你只訓練 1000 萬個參數，就能 fine-tune 一個 20 億參數的模型。兩者結合，把 Stable Diffusion 從玩具變成 2026 年每間代理商都在用的圖像管線。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 07 (Latent Diffusion), Phase 10 (LLMs from Scratch — for LoRA foundation)
**Time:** ~75 分鐘

## 問題 (The Problem)

像「一個穿紅洋裝的女人在熱鬧的街上遛狗」這樣的 prompt，並沒有告訴模型狗在 *哪裡*、女人是 *什麼姿態*，或街道的 *透視*。文字大約只能釘住你要描述一張圖像所需資訊的 10%。其餘是視覺的，無法用文字有效描述。

為每一種訊號（姿態、深度、canny、分割）從零訓練一個新的條件式模型，成本高得離譜。你想要的是：保持 2.6B 參數的 SDXL 骨幹凍結，接上一個小型的旁路網路來讀取條件，並讓它輕推骨幹的中間特徵。這就是 ControlNet。

你也想教模型新概念（你的臉、你的產品、你的風格），而不必重新訓練整個模型。你想要小 100 倍的 delta。這就是 LoRA：插入現有 attention 權重中的 low-rank adapter。

ControlNet + LoRA + 文字 = 2026 年實務工作者的工具箱。大多數正式環境的圖像管線，都會在 SDXL / SD3 / Flux 基礎模型上疊加 2-5 個 LoRA、1-3 個 ControlNet，再加上一個 IP-Adapter。

## 核心概念 (The Concept)

![ControlNet 複製 encoder；LoRA 加上 low-rank delta](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

取一個預訓練的 SD。*複製* U-Net 的 encoder 那一半。凍結原本的。訓練複製出來的那份去接受額外的條件輸入（邊緣、深度、姿態）。用 *zero-convolution* skip connection（初始化為零的 1×1 conv，一開始等於什麼都不做，再學出一個 delta）把複製的那份接回原始 U-Net 的 decoder 那一半。

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv 初始化意味著 ControlNet 一開始就是 identity，連訓練前都不會造成傷害。用標準的 diffusion loss，在 100 萬組（prompt, 條件, 圖像）三元組上訓練。

每種模態的 ControlNet 都以小型旁路模型的形式發佈（SDXL 約 360M，SD 1.5 約 70M）。推論時你可以組合它們：

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

對模型中任何線性層 $W \in \mathbb{R}^{d \times d}$，凍結 $W$ 並加上一個 low-rank delta：

```math
W' = W + \Delta W, \quad \Delta W = B A, \quad A \in \mathbb{R}^{r \times d}, \quad B \in \mathbb{R}^{d \times r}
```

其中 $r \ll d$。Attention 的標準 rank 是 4-16，重度 fine-tune 則是 64-128。新增的參數量：$2 \cdot d \cdot r$，而不是 $d^2$。對於 $d=640$、$r=16$ 的 SDXL attention：每個 adapter 20k 個參數，而不是 410k，減少 20 倍。整個模型下來：一個 LoRA 通常是 20-200MB，而基礎模型是 5GB。

推論時你可以縮放 LoRA：$W' = W + \alpha \cdot B A$。$\alpha = 0.5\text{-}1.5$ 是正常範圍。多個 LoRA 可以相加疊加（但一般要注意，它們之間會以非線性的方式互相影響）。

### IP-Adapter (Ye et al., 2023)

一個小型 adapter，接受 *圖像* 作為條件（與文字並用）。使用 CLIP image encoder 產生圖像 token，並與文字 token 一起注入 cross-attention。每個基礎模型約 20MB。讓你不需要 LoRA，就能做「依照這張參考圖的風格生成圖像」。

## 組合性矩陣 (Composability matrix)

| 工具 | 控制什麼 | 大小 | 何時使用 |
|------|---------|------|---------|
| ControlNet | 空間結構（姿態、深度、邊緣） | 70-360MB | 精確的版面、構圖 |
| LoRA | 風格、主體、概念 | 20-200MB | 個人化、風格 |
| IP-Adapter | 來自參考圖像的風格或主體 | 20MB | 文字無法描述外觀時 |
| Textual Inversion | 以新 token 表示的單一概念 | 10KB | 舊式做法，大多已被 LoRA 取代 |
| DreamBooth | 針對某個主體的完整 fine-tune | 2-5GB | 強烈的身分一致性，運算量高 |
| T2I-Adapter | 較輕量的 ControlNet 替代方案 | 70MB | 邊緣裝置、推論預算有限 |

ControlNet ≈ 空間。LoRA ≈ 語意。兩者並用。

```figure
v4-controlnet-zero
```

## 動手做 (Build It)

`code/main.py` 在 1-D 上模擬這兩種機制：

1. **LoRA。** 一個預訓練的線性層 $W$。凍結它。訓練一個 low-rank 的 $BA$，使 $W + BA$ 吻合目標線性層。證明 $r = 1$ 就足以完美學會 rank-1 的修正。

2. **ControlNet-lite。** 一個「凍結的基礎」預測器，加上一個讀取額外訊號的「旁路網路」。旁路網路的輸出由一個初始化為零的可學習純量來控制（我們版本的 zero-conv）。訓練並觀察這個 gate 逐漸升高。

### 步驟 1：LoRA 的數學

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### 步驟 2：零初始化的旁路網路

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

在步驟 0，輸出與基礎模型完全相同。早期訓練只會緩慢更新 `gate`，不會發生災難性的漂移。

## 常見陷阱 (Pitfalls)

- **過度縮放 LoRA。** $\alpha = 2$ 或 $\alpha = 3$ 是常見的「讓它更強」的 hack，會產生過度風格化 / 壞掉的輸出。請維持 $\alpha \le 1.5$。
- **ControlNet 權重衝突。** 同時用權重 1.0 的 Pose ControlNet 和權重 1.0 的 Depth ControlNet，通常會過頭。權重總和 ≈ 1.0 是安全的預設值。
- **LoRA 用在錯的基礎模型。** SDXL 的 LoRA 在 SD 1.5 上會默默地什麼都不做，因為 attention 的維度對不上。Diffusers 0.30+ 會發出警告。
- **Textual Inversion 漂移。** 在某個 checkpoint 上訓練的 token，換到另一個 checkpoint 上會嚴重漂移。LoRA 的可攜性更好。
- **LoRA 權重合併與儲存。** 你可以把 LoRA 烘進基礎模型權重以加快推論（執行時不用再做加法），但就失去了在執行時縮放 $\alpha$ 的能力。兩個版本都要留著。

## 實際使用 (Use It)

| 目標 | 2026 年的管線 |
|------|-------------|
| 重現品牌的藝術風格 | 用約 30 張精選圖像、rank 32 訓練的 LoRA |
| 把我的臉放進生成的圖像 | DreamBooth 或 LoRA + IP-Adapter-FaceID |
| 特定姿態 + prompt | ControlNet-Openpose + SDXL + 文字 |
| 具深度感的構圖 | ControlNet-Depth + SD3 |
| 參考圖 + prompt | IP-Adapter + 文字 |
| 精確的版面 | ControlNet-Scribble 或 ControlNet-Canny |
| 替換背景 | ControlNet-Seg + Inpainting（第 09 課） |
| 快速 1-step 風格 | SDXL-Turbo 上的 LCM-LoRA |

## 交付 (Ship It)

存成 `outputs/skill-sd-toolkit-composer.md`。這個 skill 接收一項任務（輸入素材：prompt、選用的參考圖像、選用的姿態、選用的深度圖、選用的塗鴉），輸出工具組合、權重，以及可重現的 seed 流程。

## 練習 (Exercises)

1. **簡單。** 在 `code/main.py` 中，把 LoRA 的 rank `r` 從 1 調到 4。在哪個 rank，LoRA 會剛好吻合 rank-2 的目標 delta？
2. **中等。** 在兩個目標轉換上分別訓練兩個 LoRA。同時載入它們，並展示它們的加法式交互作用。這個交互作用何時會破壞線性？
3. **困難。** 用 diffusers 疊加：SDXL-base + Canny-ControlNet（權重 0.8）+ 風格 LoRA（$\alpha$ 0.8）+ IP-Adapter（權重 0.6）。當各層疊加的權重改變時，量測 FID 與 prompt 遵循度之間的取捨。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| ControlNet | 「空間控制」 | 複製的 encoder + zero-conv skip；讀取條件圖像。 |
| Zero convolution | 「一開始是 identity」 | 初始化為零的 1×1 conv；ControlNet 一開始什麼都不做。 |
| LoRA | 「Low-rank adapter」 | $W + BA$，$r \ll d$；參數比完整 fine-tune 少 100 倍。 |
| rank r | 「那個旋鈕」 | LoRA 的壓縮程度；通常 4-16，重度個人化用 64+。 |
| $\alpha$ | 「LoRA 強度」 | 執行時對 LoRA delta 的縮放。 |
| IP-Adapter | 「參考圖像」 | 透過 CLIP-image token 達成的小型圖像條件 adapter。 |
| DreamBooth | 「完整的主體 fine-tune」 | 用某個主體的約 30 張圖像訓練整個模型。 |
| Textual Inversion | 「新 token」 | 只學一個新的 word embedding；舊式做法，大多已被取代。 |

## 實務筆記 (Production note)：LoRA 切換、ControlNet 通道、多租戶服務

真正的 text-to-image SaaS 會在同一個基礎 checkpoint 上服務數百個 LoRA 和十幾個 ControlNet。這個服務問題很像 LLM 的多租戶（文獻在 continuous batching 和 LoRAX / S-LoRA 之下討論了 LLM 的情況）：

- **熱切換 LoRA，不要合併。** 把 $W' = W + \alpha \cdot B \cdot A$ 合併進基礎模型，每步推論大約快 3-5%，但會把 $\alpha$ 和基礎模型都固定死。讓 LoRA 以 rank-r delta 的形式常駐 VRAM；diffusers 提供 `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])`，可針對每個請求啟用。切換成本是 `2 · d · r · num_layers` 個權重，MB 等級，不到一秒。
- **ControlNet 作為第二條 attention 通道。** 複製的 encoder 與基礎模型平行執行。兩個權重各為 1.0 的 ControlNet = 每步多兩次 forward pass，而不是合併成一次。Batch-size 的餘裕會以平方速度下降。每個啟用的 ControlNet 要預留約 1.5 倍的單步成本。
- **LoRA 也要量化。** 如果你已經量化了基礎模型（見第 07 課，在 8GB 上跑 Flux），LoRA delta 也可以乾淨地量化成 8-bit 或 4-bit。QLoRA 風格的載入讓你能在 4-bit 的 Flux 基礎模型上疊加 5-10 個 LoRA，而不會爆記憶體。

Flux 專屬：Niels 的 Flux-on-8GB notebook 把基礎模型量化成 4-bit；在這個量化後的基礎模型上，使用 `weight_name="pytorch_lora_weights.safetensors"` 疊加風格 LoRA（`pipe.load_lora_weights("user/style-lora")`）依然可行。這就是 2026 年大多數 SaaS 代理商採用的配方。

## 延伸閱讀 (Further Reading)

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet。
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA（最初為 LLM 設計；可移植到 diffusion）。
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter。
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — 比 ControlNet 更輕量的替代方案。
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth。
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — 參考管線。
