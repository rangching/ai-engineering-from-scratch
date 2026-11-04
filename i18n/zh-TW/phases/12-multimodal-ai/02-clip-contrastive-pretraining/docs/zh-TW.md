# CLIP 與 Contrastive Vision-Language Pretraining (CLIP and Contrastive Vision-Language Pretraining)

> OpenAI 的 CLIP (2021) 證明了一個足以驅動接下來五年的想法：只用雜亂的網路圖文配對與 contrastive loss，就能把 image encoder 和 text encoder 對齊到同一個向量空間。零監督標籤，4 億組配對。得到的 embedding 空間可以做 zero-shot 分類、圖文檢索，也成為每個 2026 年 VLM 的 vision tower。SigLIP 2 (2025) 用 sigmoid 取代 softmax，以更低成本超越 CLIP 的規模。這堂課從 InfoNCE 一路走到 sigmoid pairwise loss 的數學，並用標準函式庫的 Python 實作訓練步驟。

**Type:** Build
**Languages:** Python (stdlib, InfoNCE + sigmoid loss implementations)
**Prerequisites:** Phase 12 · 01 (ViT patches), Phase 7 (Transformers)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 從 mutual information 推導 InfoNCE loss，並實作數值穩定的向量化版本。
- 解釋為什麼 sigmoid pairwise loss (SigLIP) 能擴展到 batch 32768 以上，而不需要 softmax 所要求的 all-gather 開銷。
- 建立文字模板 (`a photo of a {class}`)，對 cosine similarity 取 argmax，執行 zero-shot ImageNet 分類。
- 說出 CLIP / SigLIP pretraining 給你的四個槓桿：batch size、temperature、prompt template、資料品質。

## 問題 (The Problem)

CLIP 之前的視覺模型都是 supervised。收集有標籤的資料集 (ImageNet：120 萬張圖片、1000 個類別)，訓練 CNN，上線。標籤很貴，標籤會偏向標註者能達成共識的內容，而且不 finetuning 就無法轉移到新任務。

網路上的圖片加 caption，免費就有十億組以上、標得很鬆散的配對。一張黃金獵犬的照片配上 alt text「我的狗 Max 在公園」，就帶有監督訊號：文字描述了圖片。問題是：能把它變成有用的訓練嗎？

CLIP 的答案：把圖文配對當成配對任務。給定一個 batch 的 N 張圖片和 N 段 caption，學著把每張圖片配到自己的 caption，並對抗另外 N-1 個干擾項。監督訊號是「這兩個東西屬於一起；這 N-1 個不是。」沒有類別標籤，沒有人工標註，只有 contrastive loss。

得到的 embedding 空間，做到的比 CLIP 當初訓練的目標更多。ImageNet zero-shot 之所以可行，是因為「a photo of a cat」會 embed 到貓的圖片附近，即使那些圖片從未被明確標為貓。這個賭注催生了每一個 2026 年的 VLM。

## 核心概念 (The Concept)

### Dual encoder

CLIP 有兩座塔：

- Image encoder `f`：ViT 或 ResNet，每張圖片輸出一個 D 維向量。
- Text encoder `g`：小型 transformer，每段 caption 輸出一個 D 維向量。

兩座塔都把輸出正規化成單位長度。因為兩者都是單位 norm，所以相似度為 `cos(f(x), g(y)) = f(x)^T g(y)`。

對一個 batch 的 $N$ 組 (image, caption) 配對，建立形狀為 `(N, N)` 的相似度矩陣 `S`：

```math
S[i, j] = \cos(f(x_i), g(y_j)) / \tau
```

其中 `tau` 是可學習的 temperature（CLIP 初始化為 0.07，在 log 空間中學習）。

### InfoNCE loss

CLIP 對 row 與 column 使用對稱的 cross-entropy：

```
loss_i2t = CE(S, labels=identity)     # each image's positive is its own caption
loss_t2i = CE(S^T, labels=identity)   # each caption's positive is its own image
loss = (loss_i2t + loss_t2i) / 2
```

這就是 InfoNCE。CE 裡的 softmax 迫使每張圖片與自己的 caption 的匹配程度，高於 batch 中其他所有 caption。「負樣本」就是 batch 中的其他項目。batch 越大 = 負樣本越多 = 訊號越強。CLIP 以 batch 32k 訓練，規模很重要。

### Temperature

`tau` 控制 softmax 的銳利程度。低 $\tau$ → 分布尖銳，有 hard negative mining 的效果。高 $\tau$ → 分布平滑，所有樣本都有貢獻。CLIP 學的是 $\log(1/\tau)$，並做 clip 以防止崩潰。SigLIP 2 固定初始的 $\tau$，改用可學習的 bias。

### 為什麼 sigmoid 擴展性更好 (SigLIP)

Softmax 需要整個相似度矩陣保持同步。在分散式訓練中，你必須把每個 embedding all-gather 到每個 replica，再做 softmax。通訊量與 world size 成二次方。

SigLIP 把 softmax 換成逐元素的 sigmoid：對每一組 `(i, j)`，loss 是「這是不是配對正確的一組？」的二元分類。正類標籤是對角線，其餘都是負類。loss 為：

```math
L = -\frac{1}{N} \sum_{(i, j)} \Big[ y_{ij} \log \mathop{\mathrm{sigmoid}}(S[i,j]) + (1-y_{ij}) \log \mathop{\mathrm{sigmoid}}(-S[i,j]) \Big]
```

`y_ij = 1` 若 `i == j`，否則為 0。每一對的 loss 彼此獨立，不需要 all-gather。每張 GPU 計算自己的區塊再加總。SigLIP 2 能便宜地擴展到 batch 32k-512k，而 CLIP 則需要成比例更多的通訊。

### Zero-shot 分類 (Zero-shot classification)

給定 N 個類別名稱，為每個類別建立一個文字模板：

```
"a photo of a {class}"
```

用 text encoder 對每個模板做 embedding。用 image encoder 對你的圖片做 embedding。對 cosine similarity 取 argmax 就是預測的類別。不需要在目標類別上訓練。

Prompt 模板很重要。CLIP 原始論文每個類別用了 80 個模板（一般、藝術、照片、繪畫等），再把 embedding 取平均，ImageNet 多拿 3 分。現代做法通常只挑一兩個模板。

### Linear probe 與 finetuning

Zero-shot 只是 baseline。Linear probe（在凍結的 CLIP 特徵上，針對你的目標類別訓練一個線性層）在領域內任務上勝過 zero-shot。完整 finetuning 在領域內又勝過 linear probe，但可能損害 zero-shot 的遷移能力。三種模式，三種取捨。

### SigLIP 2：NaFlex 與 dense feature

SigLIP 2 (2025) 新增：
- NaFlex：單一模型處理可變的長寬比與解析度。
- 針對分割與深度估計提供更好的 dense feature，目標是在 VLM 中作為凍結的 backbone。
- 多語言：以 100 多種語言訓練，而 CLIP 只有英文。
- 10 億參數的規模，CLIP 最大只到 4 億。

在 2026 年的 open VLM 中，SigLIP 2 SO400m/14 是預設的 vision tower。純圖文檢索時，若 LAION-2B 的特定訓練分布符合你的查詢模式，CLIP 仍是預設選擇。

### ALIGN、BASIC、OpenCLIP、EVA-CLIP

ALIGN (Google, 2021)：想法與 CLIP 相同，規模 18 億組配對，90% 是雜訊，證明了雜訊資料也能擴展。OpenCLIP (LAION)：在 LAION-400M / 2B 上開源重現 CLIP，有多種規模，是最常用的開源 checkpoint。EVA-CLIP：從 masked image modeling 初始化，是強力的 VLM backbone。BASIC：Google 的 CLIP+ALIGN 混合版。全都屬於同一個家族，只是資料與調校不同。

### Zero-shot 的天花板 (The zero-shot ceiling)

CLIP 等級的模型在 ImageNet zero-shot 上大約卡在 76%（CLIP-G、OpenCLIP-G）。要超越，不是需要大得多的資料（SigLIP 2 達到 80% 以上），就是需要改變架構（supervised head、更多參數）。這個 benchmark 已趨於飽和；真正的價值在於下游 VLM 所使用的 embedding 空間。

```figure
multimodal-fusion
```

## 實際使用 (Use It)

`code/main.py` 實作了：

1. 一個玩具 dual encoder（以 hash 為基礎的圖片特徵、文字字元特徵），讓你不用 numpy 就能看到 InfoNCE 的形狀。
2. 純 Python 的 InfoNCE loss（以 log-sum-exp 確保數值穩定）。
3. 用於比較的 sigmoid pairwise loss。
4. 一個 zero-shot 分類流程：對一組文字 prompt 計算 cosine similarity，取 argmax 作為預測。

執行它，觀察 loss 曲線。絕對數值只是玩具，但曲線的形狀與真正的 CLIP trainer 輸出一致。

## 交付 (Ship It)

這堂課產出 `outputs/skill-clip-zero-shot.md`。給定一組圖片（以路徑提供）與目標類別清單，它會用 CLIP 模板建立文字 prompt，以指定的 checkpoint（例如 `openai/clip-vit-large-patch14`）對兩邊做 embedding，並回傳附相似度分數的 top-1 / top-5 預測。這個 skill 拒絕對不在 prompt 清單中的類別下任何結論。

## 練習 (Exercises)

1. 手算一個含 4 組配對的 batch 的 InfoNCE。建立 4x4 相似度矩陣，做 softmax，取出對角線，計算 cross-entropy。用這個手算結果驗證你的 Python 實作。

2. SigLIP 除了 temperature 之外還用了 bias 參數 `b`：`S'[i,j] = S[i,j]/tau + b`。當 batch 的類別嚴重不平衡（每個 row 的負樣本遠多於正樣本）時，`b` 扮演什麼角色？請閱讀 SigLIP 第 3 節 (arXiv:2303.15343)。

3. 建立一個貓 vs 狗的 zero-shot 分類器。試兩個 prompt 模板：`a photo of a {class}` 與 `a picture of a {class}`。在 100 張測試圖片上量測準確率。模板的 ensemble 有勝過單一模板嗎？

4. 計算 512 張 GPU、batch 32k 的訓練中，softmax InfoNCE 與 sigmoid pairwise 的通訊成本。哪個是 $O(N)$，哪個是 $O(N^2)$？請引用 SigLIP 第 4 節。

5. 閱讀 OpenCLIP scaling-laws 論文 (arXiv:2212.07143, Cherti et al.)。從圖表重現他們關於資料 scaling 的結論：在固定模型大小下，ImageNet zero-shot 準確率與訓練資料量之間的 log-linear 關係是什麼？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|----------------|------------------------|
| InfoNCE | "Contrastive loss" | 對 batch 的相似度矩陣做 cross-entropy；每個項目的正樣本是它配對的項目，負樣本是其他所有項目 |
| Sigmoid loss | "SigLIP loss" | 逐對的 binary cross-entropy；沒有 softmax、沒有 all-gather，在分散式訓練中擴展成本低 |
| Temperature | "tau" | 在 softmax/sigmoid 之前縮放 logits 的純量；控制分布的銳利程度 |
| Zero-shot | "no-finetune classification" | 用文字 prompt 建構類別 embedding，再以 cosine similarity 分類；不在目標類別上訓練 |
| Prompt template | "a photo of a ..." | 包在類別名稱外面的文字框架；會讓 zero-shot 準確率差 1-5 分 |
| Dual encoder | "Two-tower" | 一個 image encoder 加一個 text encoder，輸出在共用的 D 維空間 |
| Hard negative | "Tough distractor" | 與正樣本夠相似、模型得費力才能分開的負樣本 |
| Linear probe | "Frozen + one layer" | 只在凍結特徵上訓練一個線性分類器；用來衡量特徵品質 |
| NaFlex | "Native flexible resolution" | SigLIP 2 的能力：不用縮放，就能接收任何長寬比與解析度的圖片 |
| Temperature scaling | "log-parametrized tau" | CLIP 參數化的是 `log(1/tau)`，讓梯度表現正常；並做 clip，避免 tau 崩潰到接近零 |

## 延伸閱讀 (Further Reading)

- [Radford et al. — Learning Transferable Visual Models From Natural Language Supervision (arXiv:2103.00020)](https://arxiv.org/abs/2103.00020) — CLIP 論文。
- [Zhai et al. — Sigmoid Loss for Language Image Pre-Training (arXiv:2303.15343)](https://arxiv.org/abs/2303.15343) — SigLIP。
- [Tschannen et al. — SigLIP 2 (arXiv:2502.14786)](https://arxiv.org/abs/2502.14786) — 多語言加 NaFlex。
- [Jia et al. — ALIGN (arXiv:2102.05918)](https://arxiv.org/abs/2102.05918) — 用雜訊網路資料做規模化。
- [Cherti et al. — Reproducible scaling laws for contrastive language-image learning (arXiv:2212.07143)](https://arxiv.org/abs/2212.07143) — OpenCLIP scaling laws。
