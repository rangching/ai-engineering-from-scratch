# Transfusion：在同一個 Transformer 中結合 Autoregressive 文字與 Diffusion 圖片 (Transfusion: Autoregressive Text + Diffusion Image in One Transformer)

> Chameleon 和 Emu3 把所有賭注都押在離散 token 上。它們行得通，但量化瓶頸很明顯：圖片品質停在連續空間 diffusion 模型之下。Transfusion (Meta, Zhou et al., 2024 年 8 月) 下了相反的賭注：讓圖片保持連續，完全拋棄 VQ-VAE，用兩個 loss 訓練一個 transformer。文字 token 用 next-token 預測，圖片 patch 用 flow-matching / diffusion loss。兩個目標最佳化的是同一組權重。Stable Diffusion 3 底層的架構 (MMDiT) 是它的近親。這堂課讀 Transfusion 的論點，做出一個玩具版的雙 loss 訓練器，並追蹤讓同一個 transformer 兼顧兩項工作的 attention mask。

**Type:** Build
**Languages:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 接線一個在同一個 backbone 上跑兩個 loss 的 transformer（文字 token 用 NTP，圖片 patch 用 diffusion MSE）。
- 解釋為什麼圖片 patch 之間用雙向 attention、文字 token 之間用 causal attention，是正確的 mask 選擇。
- 在運算、品質與程式碼複雜度上，比較 Transfusion 風格（連續圖片、diffusion loss）與 Chameleon 風格（離散圖片、NTP）。
- 說出 MMDiT 的貢獻：每個 block 有各 modality 專屬的權重，並在 residual stream 做 joint attention。

## 問題 (The Problem)

離散對連續圖片 token 的爭論，比 LLM 還要久。連續表示（原始像素、VAE latent）保留細節。離散 token（VQ 索引）符合 transformer 的原生詞彙，但在量化步驟損失細節。

Chameleon / Emu3 走離散路線：一個 loss、一個架構，但圖片保真度被 tokenizer 品質封頂。

Diffusion 模型走連續路線：圖片品質出色，但是與 LLM 分開的模型，有複雜的 noise-schedule 工程，也無法與文字生成乾淨地整合。

Transfusion 問：我們能兩者兼得嗎？讓圖片保持連續，仍然只訓練一個模型，把兩個 loss 縫進同一個梯度步驟。

## 核心概念 (The Concept)

### 雙 loss 架構 (The two-loss architecture)

單一 decoder-only transformer 處理的序列包含：

- 文字 token（離散，來自 BPE 詞彙表）。
- 圖片 patch（連續，16x16 像素的區塊透過線性 embedding 投影到 hidden 維度，與 ViT encoder 的輸入相同）。
- `<image>` 與 `</image>` 標籤，標出連續 patch 所在位置。

forward pass 只跑一次。loss 針對每個 token 選擇兩個 head 之一：

- 文字 token：在詞彙表 logits head 上做標準 cross-entropy。
- 圖片 patch：在連續 patch 上做 diffusion loss，預測加到每個 patch 上的 noise。

梯度流過共用的 transformer 主體。兩個 loss 同時改善共用權重。

### Attention mask：causal 文字 + 雙向圖片

文字 token 必須是 causal，你不能讓文字 token 關注未來的文字，否則 teacher forcing 會壞掉。但圖片 patch 代表同一個快照；它們在同一個圖片區塊內應該彼此雙向 attention。

mask 如下：

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i)   # text attends to images that closed before it
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

在訓練與推論時以 block-triangular mask 實作。

### Transformer 內的 diffusion loss (Diffusion loss inside the transformer)

diffusion loss 是標準的：對圖片 patch 加 noise，要求模型預測 noise（或等價地預測乾淨的 patch）。Transfusion 的版本使用 flow matching：預測從有雜訊到乾淨的速度場。

訓練時：
1. 對每個圖片 patch $x_0$，抽樣隨機的 timestep $t$。
2. 抽樣 noise $\epsilon$，計算 $x_t = (1-t) \cdot x_0 + t \cdot \epsilon$（flow matching 的線性插值）。
3. transformer 預測 $v_\theta(x_t, t)$；$\text{loss} = \mathop{\mathrm{MSE}}(v_\theta(x_t, t), \epsilon - x_0)$。
4. 與同一序列的文字 NTP loss 一起 backprop。

推論時，生成方式是：
- 文字 token：標準 autoregressive 取樣。
- 圖片 patch：以先前的文字 token 為條件的 diffusion 取樣迴圈（通常 10-30 步）。

### MMDiT：Stable Diffusion 3 的變體

Stable Diffusion 3 (Esser et al., 2024 年 3 月) 在與 Transfusion 差不多的時間推出了 MMDiT (Multimodal Diffusion Transformer)。兩者架構是兄弟。

MMDiT 的主要差異：

- 每個 block 有各 modality 專屬的權重。每個 transformer block 對文字 token 與圖片 patch 有獨立的 Q、K、V 與 MLP 權重。Attention 是 joint（跨 modality）的；其他一切都是 modality 專屬的。
- Rectified flow 訓練。一種特定的 flow-matching 變體，取樣方式已知，數學也比 DDPM 簡單。
- 規模。MMDiT 是 SD3 的 backbone（20 億與 80 億參數的版本）。Transfusion 論文擴展到 70 億。

兩者都收斂到同一個核心想法：一個 transformer 對文字跑 NTP，對連續圖片表示跑 diffusion。

### 為什麼這勝過 Chameleon 風格 (Why this beats Chameleon-style)

在圖片生成上，連續 diffusion 與離散 NTP 的品質差距是可量測的。Transfusion 論文回報：

- 在 70 億參數下，FID 比同尺寸的 Chameleon 風格模型好 3-5 分。
- 不需要訓練 tokenizer，圖片 encoder 更簡單（線性投影到 hidden，與 ViT 的輸入層相同）。
- 推論時可以平行去噪圖片 patch，這點不像 autoregressive 的圖片 token。

缺點：Transfusion 是雙 loss 模型，讓訓練動態更棘手。需要調 loss 權重。NTP 與 diffusion 之間的排程不匹配，可能導致其中一個 head 主導。

### 下游發展 (What sits downstream)

Janus-Pro（課程 12.15）改良了 Transfusion 的想法，為理解與生成解耦 vision encoder（一邊用 SigLIP，另一邊用 VQ），同時共用 transformer 主體。Show-o（課程 12.14）把 diffusion 換成離散 diffusion（masked 預測）。統一生成家族在 Transfusion 之後迅速分支。

原生輸出圖片的封閉 production 模型，可能使用這個家族的某種後代，但它們的架構未公開，所以任何此類說法都只能當作猜測。

```figure
cfg-guidance-scale
```

## 實際使用 (Use It)

`code/main.py` 在一個類似小型 MNIST 的問題上建立玩具版 Transfusion：

- 文字 caption 是描述數字 (0-9) 的短整數序列。
- 圖片是 4x4 的位元組網格。
- 一對共用權重的線性投影充當 transformer 的替身；文字用 NTP loss，有雜訊的 patch 用 MSE loss。
- 訓練迴圈交替兩個 loss，attention mask 是明確的。
- 一個 mask 示範，印出文字、圖片、文字序列的 block-triangular attention 模式。不執行生成：真正的取樣器會逐 token 解碼文字，然後在多個步驟中去噪圖片 patch，如下面的練習所計算的。

transformer 是玩具。雙 loss 的接線、attention mask 的建構與推論迴圈，才是真正的成果。

## 交付 (Ship It)

這堂課產出 `outputs/skill-two-loss-trainer-designer.md`。給定新的 multimodal 訓練任務（文字 + 圖片、文字 + 音訊、文字 + 影片），它會設計雙 loss 排程（loss 權重、mask 形狀、共用與 modality 專屬的 block），並標出實作風險。

## 練習 (Exercises)

1. 一個 Transfusion 風格的模型以 70% 文字 token 和 30% 圖片 patch 訓練。圖片 diffusion loss 的量級約為文字 NTP loss 的 10 倍。什麼樣的 loss 權重能平衡它們？

2. 為序列 `[T, T, <image>, P, P, P, P, </image>, T]` 實作 block-triangular mask。把每個項目標成 0 或 1。

3. MMDiT 有各 modality 專屬的 QKV 權重。相較於 Transfusion 完全共用的 transformer，這增加了多少參數量開銷？在 70 億參數下值得嗎？

4. 生成：給定文字 prompt，模型先對 50 個 token 跑 NTP，然後遇到 `<image>`，再對 256 個 patch 跑 20 個去噪步驟的 diffusion。總共有多少次 forward pass？

5. 閱讀 SD3 論文第 3 節。描述 rectified flow，以及它為什麼比 DDPM 在更少的推論步驟內收斂。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | 單一 transformer 在同一個梯度步驟中，同時最佳化文字 token 的 cross-entropy 與連續圖片 patch 的 MSE |
| Flow matching | "Rectified flow" | 預測從 noise 到乾淨資料的速度場的 diffusion 變體；數學比 DDPM 簡單 |
| MMDiT | "Multimodal DiT" | Stable Diffusion 3 的架構：joint attention，各 modality 專屬的 MLP 與 norm |
| Block-triangular mask | "Causal text + bidirectional image" | 在文字間為 causal、在圖片區域內為雙向的 attention mask |
| Continuous image representation | "No VQ" | 以實數向量表示的圖片 patch，而不是整數 codebook 索引 |
| Velocity prediction | "v-parameterization" | 網路輸出是 noise 與資料之間的速度場，而不是 noise 本身 |

## 延伸閱讀 (Further Reading)

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
