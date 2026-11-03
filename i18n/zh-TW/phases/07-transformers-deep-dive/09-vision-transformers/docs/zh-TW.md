# Vision Transformer (ViT) (Vision Transformers (ViT))

> 影像是一個 patch 網格。句子是一個 token 網格。同一個 transformer 兩個都吃得下。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 4 · 03 (CNNs), Phase 4 · 14 (Vision Transformers intro)
**Time:** ~45 分鐘

## 問題 (The Problem)

2020 年以前，電腦視覺就等於 convolution。ImageNet、COCO 和偵測基準上的每個 SOTA 都用 CNN backbone。Transformer 是給語言用的。

Dosovitskiy et al. (2020) 的 "An Image is Worth 16x16 Words" 證明了你可以完全拿掉 convolution。把影像切成固定大小的 patch，把每個 patch 線性投影成 embedding，把這串序列餵給普通的 transformer encoder。在足夠的規模下（ImageNet-21k 預訓練或更大），ViT 與基於 ResNet 的模型不相上下，甚至更好。

ViT 是 2026 年一個更廣泛模式的起點：一種架構，多種模態。Whisper 把音訊 tokenize。ViT 把影像 tokenize。機器人用 action token。影片用 pixel token。Transformer 不在乎，你餵它一個序列，它就會學。

到了 2026 年，ViT 與它的後代（DeiT、Swin、DINOv2、ViT-22B、SAM 3）擁有大部分的視覺領域。CNN 仍然在邊緣裝置和對延遲敏感的任務上勝出。其他地方的堆疊裡，多半都有某個 ViT。

## 核心概念 (The Concept)

![Image → patches → tokens → transformer](../assets/vit.svg)

### 步驟 1：Patchify (Step 1 — patchify)

把 `H × W × C` 的影像切成 `N × (P·P·C)` 的扁平 patch 序列。典型設定：`224 × 224` 影像、`16 × 16` patch → 196 個 patch，每個 768 個值。

```
image (224, 224, 3) → 14 × 14 grid of 16x16x3 patches → 196 vectors of length 768
```

Patch 大小是調整的槓桿。Patch 越小 = token 越多、解析度越好、attention 成本是二次方。Patch 越大 = 越粗略、越便宜。

### 步驟 2：線性 embedding (Step 2 — linear embedding)

單一學習得來的矩陣把每個扁平 patch 投影到 `d_model`。等同於 kernel 大小為 `P`、stride 為 `P` 的 convolution。在 PyTorch 中這就是字面上的 `nn.Conv2d(C, d_model, kernel_size=P, stride=P)`，兩行實作。

### 步驟 3：在前面加上 `[CLS]` token，並加入 positional embedding (Step 3 — prepend `[CLS]` token, add positional embeddings)

- 在前面加一個可學習的 `[CLS]` token。它最後的 hidden state 就是用於分類的影像表示。
- 加上可學習的 positional embedding（原版 ViT）或 sinusoidal 2D（後來的變體）。
- 2024 年之後，RoPE 被擴充成 2D 位置編碼，有時不需要明確的 embedding。

### 步驟 4：標準 transformer encoder (Step 4 — standard transformer encoder)

堆疊 L 個 `LayerNorm → Self-Attention → + → LayerNorm → MLP → +` 的 block。與 BERT 完全相同。沒有任何視覺專用的層。這就是這篇論文在教學上的重點。

### 步驟 5：Head (Step 5 — head)

分類：取 `[CLS]` hidden state → linear → softmax。對 DINOv2 或 SAM，則丟掉 `[CLS]`，直接使用 patch embedding。

### 重要的變體 (Variants that mattered)

| 模型 | 年份 | 改變 |
|-------|------|--------|
| ViT | 2020 | 原版。固定 patch 大小，完整的 global attention。 |
| DeiT | 2021 | Distillation；只用 ImageNet-1k 就能訓練。 |
| Swin | 2021 | 階層式，使用 shifted window。成本固定為次二次方。 |
| DINOv2 | 2023 | 自監督（沒有標籤）。最好的通用視覺特徵。 |
| ViT-22B | 2023 | 22B 參數；scaling law 適用。 |
| SigLIP | 2023 | ViT + 語言配對，sigmoid contrastive loss。 |
| SAM 3 | 2025 | Segment anything；ViT-Large + 可提示的 mask decoder。 |

### 為什麼花了一段時間 (Why it took a while)

ViT 需要*大量*資料才能追上 CNN，因為它完全沒有 CNN 的 inductive bias（平移不變性、局部性）。如果沒有超過 1 億張標註影像，或強大的自監督預訓練，在相同算力下 CNN 仍然勝出。DeiT 在 2021 年用 distillation 技巧修正了這點；DINOv2 在 2023 年用自監督永久解決了這個問題。

```figure
n5-patch-stream
```

## 動手做 (Build It)

請看 `code/main.py`。純標準函式庫的 patchify + 線性 embedding + 健全性檢查。不做訓練，任何有實際規模的 ViT 都需要 PyTorch 和數小時的 GPU 時間。

### 步驟 1：假影像 (fake image)

一張 24 × 24 的 RGB 影像，表示成 `(R, G, B)` tuple 的列所組成的 list。我們使用 6×6 的 patch → 16 個 patch，每個是 108 維的 embedding 向量。

### 步驟 2：Patchify (patchify)

```python
def patchify(image, P):
    H = len(image)
    W = len(image[0])
    patches = []
    for i in range(0, H, P):
        for j in range(0, W, P):
            patch = []
            for di in range(P):
                for dj in range(P):
                    patch.extend(image[i + di][j + dj])
            patches.append(patch)
    return patches
```

Raster 順序：沿著網格以 row-major 排列。每個 ViT 都使用這個順序。

### 步驟 3：線性 embed (linear embed)

把每個扁平 patch 乘上一個隨機的 `(patch_flat_size, d_model)` 矩陣。驗證在加上 `[CLS]` 之後，輸出形狀是 `(N_patches + 1, d_model)`。

### 步驟 4：計算實際規模 ViT 的參數量 (count parameters for a realistic ViT)

印出 ViT-Base 的參數量：12 層、12 個 head、d=768、patch=16。與 ResNet-50（約 25M）比較。ViT-Base 約 86M。ViT-Large 約 307M。ViT-Huge 約 632M。

## 實際使用 (Use It)

```python
from transformers import ViTImageProcessor, ViTModel
import torch
from PIL import Image

processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224-in21k")
model = ViTModel.from_pretrained("google/vit-base-patch16-224-in21k")

img = Image.open("cat.jpg")
inputs = processor(img, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, 197, 768): [CLS] + 196 patches
cls_emb = out[:, 0]                       # image representation
```

**DINOv2 embedding 是 2026 年影像特徵的預設選擇。** 凍結 backbone，訓練一個很小的 head。適用於分類、檢索、偵測、captioning。Meta 的 DINOv2 checkpoint 在每一項非文字的視覺任務上都勝過 CLIP。

**挑選 patch 大小。** 小模型使用 16×16（ViT-B/16）。密集預測（分割）使用 8×8 或 14×14（SAM、DINOv2）。非常大的模型使用 14×14。

## 交付 (Ship It)

請看 `outputs/skill-vit-configurator.md`。這個 skill 會根據資料集大小、解析度與算力預算，為新的視覺任務挑選 ViT 變體與 patch 大小。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。驗證 patch 數量等於 `(H/P) * (W/P)`，而且扁平 patch 的維度等於 `P*P*C`。
2. **中等。** 實作 2D sinusoidal positional embedding：對每個 patch 的 `row` 和 `col` 各用一個獨立的 sinusoidal 編碼，再串接起來。把它們餵進一個小型 PyTorch ViT，並在 CIFAR-10 上和可學習的 positional embedding 比較準確率。
3. **困難。** 建立一個 3 層 ViT（PyTorch），用 4×4 的 patch 在 1,000 張 MNIST 影像上訓練。測量測試準確率。接著在同樣 1,000 張影像上加入 DINOv2 預訓練（簡化版：只訓練 encoder 根據被遮住的 patch 預測 patch embedding）。準確率有改善嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Patch | 「Vision transformer 的 token」 | 影像中 `P × P × C` 區域的像素值組成的扁平向量。 |
| Patchify | 「切開 + 攤平」 | 把影像切成不重疊的 patch，把每個攤平成向量。 |
| `[CLS]` token | 「影像摘要」 | 加在前面的可學習 token；它最後的 embedding 就是影像表示。 |
| Inductive bias | 「模型的假設」 | ViT 的先驗比 CNN 少；需要更多資料來彌補差距。 |
| DINOv2 | 「自監督 ViT」 | 使用影像增強 + momentum teacher，在沒有標籤的情況下訓練。2026 年最好的通用影像特徵。 |
| SigLIP | 「CLIP 的後繼者」 | 用 sigmoid contrastive loss 訓練的 ViT + 文字 encoder；在相同算力下比 CLIP 更好。 |
| Swin | 「有窗口的 ViT」 | 階層式 ViT，使用 local attention + shifted window；次二次方。 |
| Register tokens | 「2023 年的招數」 | 幾個額外的可學習 token，用來吸收 attention sink；能改善 DINOv2 特徵。 |

## 延伸閱讀 (Further Reading)

- [Dosovitskiy et al. (2020). An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929) — ViT 論文。
- [Touvron et al. (2021). Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877) — DeiT。
- [Liu et al. (2021). Swin Transformer: Hierarchical Vision Transformer using Shifted Windows](https://arxiv.org/abs/2103.14030) — Swin。
- [Oquab et al. (2023). DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193) — DINOv2。
- [Darcet et al. (2023). Vision Transformers Need Registers](https://arxiv.org/abs/2309.16588) — 針對 DINOv2 的 register-token 修正。
