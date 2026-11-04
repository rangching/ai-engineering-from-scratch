# Vision Transformers 與 Patch Token 基本單元 (Vision Transformers and the Patch-Token Primitive)

> 在談任何 multimodal 之前，圖片必須先變成 transformer 能吃的 token 序列。2020 年的 ViT 論文用 16x16 像素的 patch、一個線性投影，加上 position embedding 解決了這件事。五年後，每一個 2026 年的前沿模型（原生 2576px 的 Claude Opus 4.7、Gemini 3.1 Pro、Qwen3.5-Omni）依然從這裡開始。encoder 從 ViT 換成 DINOv2 再換成 SigLIP 2，加入了 register token，位置編碼變成 2D-RoPE，但基本單元始終沒變。這堂課從頭到尾讀完 patch-token 流程，並用標準函式庫的 Python 實作出來，讓 Phase 12 其餘內容對「visual token」有具體的心智模型。

**Type:** Learn
**Languages:** Python (stdlib, patch tokenizer + geometry calculator)
**Prerequisites:** Phase 7 (Transformers), Phase 4 (Computer Vision)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 把 HxWx3 的圖片轉成 patch token 序列，並加上正確的位置編碼。
- 計算指定 (patch size、解析度、hidden dim、深度) 的 ViT 的序列長度、參數量與 FLOPs。
- 說出讓 ViT 從 2020 年研究走向 2026 年正式上線的三項升級：self-supervised pretraining (DINO / MAE)、register token、native-resolution packing。
- 針對下游任務，在 CLS pooling、mean pooling 與 register token 之間做選擇。

## 問題 (The Problem)

Transformer 處理的是向量序列。文字本來就是序列（byte 或 token）。圖片則是帶有三個顏色通道的 2D 像素網格，不是序列。如果把每個像素都攤平，一張 224x224 的 RGB 圖片會變成 150,528 個 token，在這種長度下做 self-attention 根本不可行（成本與序列長度成二次方）。

2020 年之前的做法，是在前面接一個 CNN 特徵擷取器：ResNet 產生 7x7、每格 2048 維的 feature map，再把這 49 個 token 餵給 transformer。這樣可行，但會繼承 CNN 的偏差（平移等變性、局部感受野），也失去 transformer 吃規模的胃口。

Dosovitskiy 等人 (2020) 問了一個直白的問題：如果直接跳過 CNN 呢？把圖片切成固定大小的 patch（例如 16x16 像素），把每個 patch 線性投影成一個向量，加上 position embedding，再丟給一般的 transformer。當時這是異端：沒有 convolution 的視覺模型。資料夠多時 (JFT-300M，之後是 LAION)，它在 ImageNet 上打敗了 ResNet，而且還在持續進步。

到了 2026 年，ViT 這個基本單元已是毫無爭議的基礎。每個 open-weights VLM 的 vision tower 都是它的後代 (DINOv2、SigLIP 2、CLIP、EVA、InternViT)。問題不再是「要不要用 patch？」，而是「patch 多大、解析度怎麼排程、用什麼 pretraining 目標、用什麼位置編碼。」

## 核心概念 (The Concept)

### 把 patch 當成 token (Patches as tokens)

給定形狀為 `(H, W, 3)` 的圖片 `x` 與 patch 大小 `P`，把圖片切成 `(H/P) x (W/P)` 個互不重疊的 patch。每個 patch 是一個 `P x P x 3` 的像素立方體。把每個立方體攤平成 `3 P^2` 的向量，再套用形狀為 `(3 P^2, D)` 的共用線性投影 `W_E`，把每個 patch 對應到模型的 hidden dimension `D`。

以標準的 ViT-B/16 設定為例：
- 解析度 224、patch size 16 → 網格 14x14 → 196 個 patch token。
- 每個 patch 有 `16 x 16 x 3 = 768` 個像素值，投影到 `D = 768`。
- 加上一個可學習的 `[CLS]` token → 序列長度 197。

patch 投影在數學上等同於 kernel size 為 `P`、stride 為 `P`、輸出 `D` 個 channel 的 2D convolution。實際的 production 程式碼就是這樣實作：`nn.Conv2d(3, D, kernel_size=P, stride=P)`。「線性投影」是概念上的說法，kernel 的寫法則比較有效率。

### 位置編碼 (Positional embeddings)

patch 本身沒有順序，transformer 看到的是一袋 patch。早期 ViT 加的是可學習的 1D position embedding（每個位置一個 768 維向量，共 197 個）。這樣可行，但會把模型綁在訓練解析度上：推論時只要網格改變，就得對位置表做插值。

現代 vision backbone 使用 2D-RoPE (Qwen2-VL 的 M-RoPE、SigLIP 2 的預設) 或分解式 2D 位置。2D-RoPE 依 patch 的 (row, column) 索引來旋轉 query 與 key 向量，模型就能從旋轉角度推得相對的 2D 位置。不需要位置表，推論時可處理任意大小的網格。

### CLS token、pooled output 與 register token

圖片層級的表示法是什麼？目前有三種選擇並存：

1. `[CLS]` token。在 patch 序列前面加一個可學習向量。經過所有 transformer block 後，CLS token 的 hidden state 就是圖片表示。沿用自 BERT，原始 ViT 與 CLIP 都用它。
2. Mean pool。把 patch token 輸出的 hidden state 取平均。SigLIP、DINOv2 與大多數現代 VLM 都用這種。
3. Register token。Darcet 等人 (2023) 觀察到，沒有明確 sink token 的 ViT 在訓練後，會長出高 norm 的「artifact」patch，劫持 self-attention。加入 4 到 16 個可學習的 register token 可以吸收這個負擔，並提升 dense prediction（分割、深度）的品質。DINOv2 與 SigLIP 2 都附帶 register。

這個選擇對下游任務很重要。分類用 CLS 就夠了。對於把 patch token 餵進 LLM 的 VLM，則完全跳過 pooling，每個 patch 都成為 LLM 的輸入 token。register 會在交給 LLM 之前丟掉（它們是鷹架，不是內容）。

### Pretraining：supervised、contrastive、masked、self-distilled

2020 年的 ViT 是在 JFT-300M 上以 supervised classification 做 pretraining，很快就被取代：

- CLIP (2021)：在 4 億組配對上做 contrastive 圖文訓練。見課程 12.02。
- MAE (2021, He et al.)：遮住 75% 的 patch，重建像素。屬於 self-supervised，只需要圖片。
- DINO (2021) / DINOv2 (2023)：用 student-teacher 做 self-distillation，不需要標籤，也不需要 caption。2023 年的 DINOv2 ViT-g/14 是最強的純視覺 backbone，也是「dense feature」用途的預設選擇。
- SigLIP / SigLIP 2 (2023, 2025)：把 CLIP 換成 sigmoid loss，並用 NaFlex 支援原生長寬比。是 2026 年 open VLM (Qwen、Idefics2、LLaVA-OneVision) 中最主流的 vision tower。

你選的 pretraining 方式，決定 backbone 擅長什麼：CLIP/SigLIP 擅長與文字做語意比對，DINOv2 擅長 dense visual feature，MAE 則適合當下游 finetuning 的起點。

### Scaling laws

ViT scaling (Zhai et al. 2022) 確立了 ViT 的品質會依模型大小、資料量與運算量，遵循可預測的規律。在固定運算量下：
- 更大的模型加上更多資料 → 品質更好。
- Patch size 是在序列長度與細節之間取捨的槓桿。Patch 14（DINOv2/SigLIP SO400m 常用）每張圖產生的 token 比 patch 16 多，對 OCR 與 dense 任務較好，但速度較慢。
- 解析度是另一個大槓桿。從 224 提升到 384 再到 512，幾乎都有幫助，代價是 FLOPs 呈二次方成長。

ViT-g/14（10 億參數、patch 14、解析度 224 → 256 個 token）與 SigLIP SO400m/14（4 億參數、patch 14）是 2026 年 open VLM 的兩個主力 encoder。

### ViT 的參數量 (Parameter count for a ViT)

完整計算在 `code/main.py`。以 224 解析度的 ViT-B/16 為例：

```
patch_embed = 3 * 16 * 16 * 768 + 768  =  591k
cls + pos    = 768 + 197 * 768          =  152k
block        = 4 * 768^2 (QKVO) + 2 * 4 * 768^2 (MLP) + 2 * 2*768 (LN)
             = 12 * 768^2 + 3k          =  7.1M
12 blocks    = 85M
final LN    = 1.5k
total       ≈ 86M
```

載入 checkpoint 之前，先這樣粗估每個 ViT。backbone 的大小決定了任何下游 VLM 的 VRAM 下限。

### 2026 年的 production 設定 (2026 production config)

2026 年大多數 open VLM 搭配的 encoder，是原生解析度 (NaFlex) 的 SigLIP 2 SO400m/14。它具備：
- 4 億參數。
- Patch size 14、預設解析度 384 → 每張圖 729 個 patch token。
- 圖片層級任務用 mean pool；做 VQA 時，全部 729 個 patch 都流進 LLM。
- 4 個 register token，交給 LLM 之前丟掉。
- 搭配圖片層級縮放的 2D-RoPE，支援原生長寬比。

這份設定中的每個決定，都能追溯到一篇你可以讀的論文。

```figure
image-patch-tokens
```

## 實際使用 (Use It)

`code/main.py` 是一個 patch tokenizer 加幾何計算器。它接收 (圖片 H, W、patch P、hidden D、深度 L)，並回報：

- patch 化之後的網格形狀與序列長度。
- 一張合成的 8x8 像素玩具圖片的 token 序列（走一遍攤平加投影的路徑）。
- 依 patch embed、position embed、transformer block 與 head 拆分的參數量。
- 目標解析度下每次 forward pass 的 FLOPs。
- ViT-B/16 @ 224、ViT-L/14 @ 336、DINOv2 ViT-g/14 @ 224、SigLIP SO400m/14 @ 384 的比較表。

執行它。把參數量對照已發表的數字。調整 patch size 與解析度，感受 token 數量的成本。

## 交付 (Ship It)

這堂課產出 `outputs/skill-patch-geometry-reader.md`。給定一個 ViT 設定（patch size、解析度、hidden dim、深度），它會產生 token 數、參數量與 VRAM 的估算，並附上理由。每當你要為 VLM 挑 vision backbone 時就用這個 skill，它能避免「token 爆量、LLM context 被塞滿」這種意外。

## 練習 (Exercises)

1. 計算 Qwen2.5-VL 在原生 1280x720 輸入、patch size 14 時的 patch-token 序列長度。這跟只用 CLS 的表示法相比如何？

2. 一張 1080p 影格 (1920x1080) 在 patch 14 下會產生多少 token？以 30 FPS 處理 5 分鐘的影片，總共有多少 visual token？pooling、影格取樣與 token merging，哪一個省最多？

3. 用純 Python 實作 patch token 的 mean pooling。驗證對 DINOv2 輸出的 196 個 token 做 mean pool，結果與模型在要求 pooled embedding 時 `forward` 回傳的一致。

4. 閱讀 "Vision Transformers Need Registers" (arXiv:2309.16588) 的第 3 節。用兩句話描述 register 吸收了什麼 artifact，以及它為何對下游 dense prediction 重要。

5. 修改 `code/main.py` 以支援 patch-n'-pack：給定一串不同解析度的圖片，產生單一的 packed 序列與 block-diagonal attention mask。等你學到課程 12.06 時再對照驗證。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|----------------|------------------------|
| Patch | "16x16 pixel square" | 輸入圖片中固定大小、互不重疊的區域，會變成一個 token |
| Patch embedding | "Linear projection" | 共用的可學習矩陣（或 stride=P 的 Conv2d），把攤平的 patch 像素對應成 D 維向量 |
| CLS token | "Class token" | 加在前面的可學習向量，其最終 hidden state 代表整張圖片；2026 年已是選用 |
| Register token | "Sink token" | 額外的可學習 token，吸收 ViT 在 pretraining 期間產生的高 norm attention artifact |
| Position embedding | "Positional info" | 讓序列具備順序感的逐位置向量或旋轉；2D-RoPE 是現代預設 |
| Grid | "Patch grid" | 在指定解析度與 patch size 下，由 (H/P) x (W/P) 個 patch 組成的 2D 陣列 |
| NaFlex | "Native flexible resolution" | SigLIP 2 的功能：單一模型不必重新訓練，就能支援多種長寬比與解析度 |
| Backbone | "Vision tower" | 預訓練好的 image encoder，其 patch-token 輸出會餵給 VLM 中的 LLM |
| Pooling | "Image-level summary" | 把 patch token 變成單一向量的策略：CLS、mean、attention pool 或基於 register |
| Patch 14 vs 16 | "Finer vs coarser grid" | Patch 14 每張圖產生更多 token，OCR 的細節更好但較慢；patch 16 是經典預設 |

## 延伸閱讀 (Further Reading)

- [Dosovitskiy et al. — An Image is Worth 16x16 Words (arXiv:2010.11929)](https://arxiv.org/abs/2010.11929) — 原始的 ViT。
- [He et al. — Masked Autoencoders Are Scalable Vision Learners (arXiv:2111.06377)](https://arxiv.org/abs/2111.06377) — MAE，self-supervised pretraining。
- [Oquab et al. — DINOv2 (arXiv:2304.07193)](https://arxiv.org/abs/2304.07193) — 大規模的 self-distillation，不需標籤。
- [Darcet et al. — Vision Transformers Need Registers (arXiv:2309.16588)](https://arxiv.org/abs/2309.16588) — register token 與 artifact 分析。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 2026 年的預設 vision tower。
- [Zhai et al. — Scaling Vision Transformers (arXiv:2106.04560)](https://arxiv.org/abs/2106.04560) — 實證的 scaling laws。
