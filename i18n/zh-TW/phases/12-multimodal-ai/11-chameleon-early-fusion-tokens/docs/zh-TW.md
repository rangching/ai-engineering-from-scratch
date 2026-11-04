# Chameleon 與 Early-Fusion 純 Token Multimodal 模型 (Chameleon and Early-Fusion Token-Only Multimodal Models)

> 我們目前看過的每個 VLM 都把圖片和文字分開。visual token 來自 vision encoder，流過 projector，然後在 LLM 內與文字相遇。視覺與文字的詞彙表從不重疊。Chameleon (Meta, 2024 年 5 月) 問：如果重疊呢？訓練一個 VQ-VAE，把圖片轉成來自共用詞彙表的離散 token 序列。這樣每份 multimodal 文件都只是一條序列：文字 token 和圖片 token 交錯，只有一個 autoregressive loss。副作用是：模型可以生成混合 modality 的輸出，在一次推論呼叫中交替輸出文字和圖片 token。這堂課讀 early-fusion 的論點，並從頭到尾做出一個玩具版本。

**Type:** Build
**Languages:** Python (stdlib, VQ-VAE tokenizer + interleaved decoder)
**Prerequisites:** Phase 12 · 05, Phase 8 (Generative AI)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼共用詞彙表加單一 loss，會改變模型能做的事。
- 描述 VQ-VAE 如何把圖片 token 化成與 transformer 的 next-token 目標相容的離散序列。
- 說出 Chameleon 的訓練穩定技巧：QK-Norm、dropout 位置、LayerNorm 順序。
- 比較 Chameleon 與 BLIP-2 的 Q-Former 做法，並描述各自適用的時機。

## 問題 (The Problem)

基於 adapter 的 VLM (LLaVA、BLIP-2、Qwen-VL) 把文字和圖片視為兩種不同的東西。文字 token 走 `embed(text_token)`；圖片走 `visual_encoder(image) → projector → ... pseudo_tokens`。模型有兩條輸入路徑，在中途合併。

這帶來三個後果：

1. LLM 只能消化圖片，不能輸出圖片。輸出只有文字。
2. 混合 modality 的文件（段落與圖片交替，像一篇文章）很麻煩，你得在模型外解析 multimodal 輸入，或把多次生成串接起來。
3. 分布不匹配。visual token 與文字 token 位於 hidden 空間的不同區域，造成細微的對齊問題。

Chameleon 否定了這個前提：圖片只是來自共用詞彙表的離散 token 序列。在交錯文件上訓練模型，一個 loss、一個 autoregressive decoder，就能免費解鎖混合 modality 生成。

## 核心概念 (The Concept)

### 把 VQ-VAE 當作圖片 tokenizer (VQ-VAE as image tokenizer)

tokenizer 是 vector-quantized variational autoencoder。架構：

- Encoder：CNN + ViT，把圖片對應到空間特徵圖，比如 32x32 個維度 256 的特徵。
- Codebook：K 個向量組成的可學習詞彙表（Chameleon 用 8192），同樣是維度 256。
- Quantization：對每個空間特徵，以 L2 距離找出最近的 codebook 項目，把連續特徵換成整數索引。
- Decoder：CNN，把量化後的特徵還原成像素。

訓練：VAE 重建 loss + commitment loss + codebook loss。codebook 索引構成圖片的離散字母表。

在 Chameleon 中：一張圖片變成 32*32 = 1024 個 token，取自 8192 的詞彙表。把這些圖片 ID 放進與文字 BPE token 相同的 ID 空間。Chameleon 的單一 tokenizer 有 65,536 個項目，這個數字已經包含 8192 個圖片 code 和圖片分隔符等特殊 token。transformer 看到一條序列、一個 loss。

### 共用詞彙表 (The shared vocabulary)

Chameleon 的詞彙表結合了文字 token、圖片 token 與 modality 分隔符。每個 token 只有一個 ID。輸入 embedding 層把每個 ID 對應到 D 維 hidden 向量。輸出投影把 hidden 對應回詞彙表的 logits。softmax 挑出下一個 token，不管是哪種 modality。

分隔符很重要：`<image>` 和 `</image>` 標籤包住圖片 token 序列。生成時，如果模型輸出 `<image>`，下游軟體就知道接下來的 1024 個 token 是要送給 decoder 渲染像素的 VQ 索引。

### 混合 modality 生成 (Mixed-modality generation)

推論就是在共用詞彙表上做 next-token 預測。範例 prompt：「Draw a cat and describe it.」Chameleon 輸出：

```
<image> 4821 1029 2891 ... (1024 image tokens) </image>
The cat is orange, sitting on a windowsill...
```

模型自行決定順序，可能先圖後文、先文後圖，或交錯。同一個 decoder，同一個 loss。

相較之下，adapter VLM 的生成只有文字。Chameleon 重新打開了模型輸出 modality 的問題。

### 訓練穩定性：QK-Norm、dropout、LayerNorm 順序 (Training stability)

early-fusion 訓練在大規模時並不穩定。Chameleon 論文記錄了三個技巧：

- QK-Norm。在 attention 內，點積之前，對 query 和 key 投影套用 LayerNorm。防止深層網路的 logit 大小爆炸。2024 年後的多個大型模型都有使用。
- Dropout 位置。在每個 residual-add 之後都放 dropout，而不只是 attention 和 MLP 之後。當圖片 token 的梯度可能主導時，需要更多正則化。
- LayerNorm 順序。residual 分支上用 Pre-LN（標準做法），再在最後一個 block 的 skip connection 上多加一個 LN。穩定最後一層的梯度流動。

沒有這些技巧，340 億參數的 Chameleon 訓練在多個 checkpoint 發散。有了它們才收斂。訓練配方的貢獻不亞於架構。

### Tokenizer 的重建天花板 (The tokenizer's reconstruction ceiling)

VQ-VAE 是有損的。在 8192 個 codebook 項目、每張 512x512 圖片 1024 個 token 的情況下，重建 PSNR 上限約 26-28 dB。這足以生成可辨識的圖片，但明顯比連續空間的 diffusion 差（Stable Diffusion 3 達到 32+ dB）。

tokenizer 是瓶頸。更好的 tokenizer (MAGVIT-v2、IBQ、SBER-MoVQGAN) 提高了天花板。Emu3（課程 12.12）光靠更好的 tokenizer，就達到 SDXL 品質的生成。

### Chameleon 對比 BLIP-2 / LLaVA

Chameleon（early fusion，共用詞彙表）：
- 一個 loss，一個 decoder。
- 生成混合 modality 的輸出。
- tokenizer 是品質天花板。
- 昂貴：推論路徑上每張生成的圖片都要跑 VQ-VAE decoder。

BLIP-2 / LLaVA（late fusion，分開的塔）：
- 視覺輸入、只有文字輸出。
- 重用預訓練 LLM。
- 理解任務沒有 tokenizer 瓶頸。
- 便宜：單次 forward pass。

依任務挑選。如果需要圖片生成，用 Chameleon 家族。如果只需要理解，adapter-VLM 更簡單，也重用更多預訓練運算。

### Fuyu 與 AnyGPT

Fuyu (Adept, 2023) 是相關做法：完全跳過獨立的 vision encoder，把原始圖片 patch 當成 token 直接通過 LLM 的輸入投影餵進去，不需要 tokenizer。比 Chameleon 簡單，但失去共用詞彙表的輸出生成能力。

AnyGPT (Zhan et al., 2024) 把 Chameleon 擴展到四種 modality：文字、圖片、語音、音樂。每種都用同樣的 VQ-VAE 技巧，共用 transformer。any-to-any 生成。課程 12.16 會更深入介紹。

```figure
vq-codebook
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具版的端到端 early-fusion 模型：

- 一個小型 VQ-VAE 風格的 quantizer，把 8x8 patch 對應到 codebook 索引 (K=16)。
- 由 (文字 id 0..31) + (圖片 id 32..47) + (分隔符 48, 49) 組成的共用詞彙表。
- 一個玩具 autoregressive decoder（bigram 表），在合成 caption + 圖片 token 序列上訓練。
- 取樣迴圈：給定 prompt，輸出交替的文字 + 圖片 token。

程式刻意讓 transformer 保持很小（bigram），讓你能從頭到尾追蹤訊號流動。

## 交付 (Ship It)

這堂課產出 `outputs/skill-tokenizer-vs-adapter-picker.md`。給定產品規格（只理解 vs 理解 + 生成、要求的圖片品質、成本預算），它會在 Chameleon 家族（early fusion）與 LLaVA 家族（late fusion）之間做選擇，並用量化的經驗法則說明理由。

## 練習 (Exercises)

1. Chameleon 使用 K=8192 個 codebook 項目，每張 512x512 圖片 1024 個 token。估算相對於 24-bit RGB 圖片的壓縮比。它是有損的嗎？損失多少？

2. 4K 圖片 (3840x2160) 在相同 VQ-VAE 密度下會產生多少圖片 token？Chameleon 式的模型能在一次推論呼叫中生成 4K 圖片嗎？什麼會先壞掉：context、tokenizer 品質，還是 KV cache？

3. 用純 Python 實作 QK-Norm。給定 64 維的 query 和 key，顯示 LayerNorm 前後的點積。為什麼在深層網路中控制大小很重要？

4. 閱讀 Chameleon 第 2.3 節關於訓練穩定性的內容。描述論文在 340 億參數、沒有 QK-Norm 時觀察到的確切失敗模式。「norm explosion」的特徵是什麼？

5. 擴充玩具 decoder，讓它在只給文字 prompt 時輸出混合 modality 的回應。在訓練資料分布為 60% 先文字 / 40% 先圖片時，量測模型選擇先圖片 vs 先文字的頻率。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Early fusion | "Unified tokens" | 圖片從第一步起就被轉成與 transformer 詞彙表共用的離散 token |
| VQ-VAE | "Image tokenizer" | CNN + ViT + codebook，把圖片對應到 transformer 可預測的整數索引 |
| Shared vocabulary | "One dictionary" | 涵蓋文字 + 圖片 + modality 分隔符的單一 token ID 空間 |
| QK-Norm | "Attention stabilizer" | 在 query 和 key 做點積前對其套用 LayerNorm，防止 norm 爆炸 |
| Mixed-modality generation | "Text + image output" | 在一次推論中自主產生交錯的文字與圖片 token |
| Codebook size | "K entries" | VQ-VAE 可量化到的離散向量數量；在壓縮與保真度之間取捨 |
| Tokenizer ceiling | "Reconstruction limit" | 解碼 VQ token 所能達到的最佳 PSNR；限制模型的圖片品質 |

## 延伸閱讀 (Further Reading)

- [Chameleon Team — Chameleon: Mixed-Modal Early-Fusion Foundation Models (arXiv:2405.09818)](https://arxiv.org/abs/2405.09818)
- [Aghajanyan et al. — CM3 (arXiv:2201.07520)](https://arxiv.org/abs/2201.07520)
- [Yu et al. — CM3Leon (arXiv:2309.02591)](https://arxiv.org/abs/2309.02591)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Adept — Fuyu-8B blog (adept.ai)](https://www.adept.ai/blog/fuyu-8b)
