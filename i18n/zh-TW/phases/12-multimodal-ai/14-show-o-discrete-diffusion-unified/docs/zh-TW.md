# Show-o 與 Discrete-Diffusion 統一模型 (Show-o and Discrete-Diffusion Unified Models)

> Transfusion 混合了連續與離散表示。Show-o (Xie et al., 2024 年 8 月) 走另一條路：文字 token 使用 causal next-token 預測，圖片 token 則承襲 MaskGIT 精神，使用 masked discrete diffusion。兩者都在同一個 transformer 裡，搭配混合 attention mask。結果是把 VQA、text-to-image、inpainting 與混合 modality 生成統一在同一個 backbone 上：每個 modality 一個 tokenizer，一種 loss 形式（把 next-token 擴展成 masked 預測）。這堂課走過 Show-o 的設計，說明為什麼 masked discrete diffusion 是平行、少步驟的圖片生成器，並與 Transfusion 和 Emu3 對照。

**Type:** Learn
**Languages:** Python (stdlib, masked-discrete-diffusion sampler)
**Prerequisites:** Phase 12 · 13 (Transfusion)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 解釋 masked discrete diffusion：先均勻遮住 token，再要求 transformer 還原它們的排程。
- 在速度與品質上，比較平行圖片解碼 (Show-o、MaskGIT) 與 autoregressive 圖片解碼 (Chameleon、Emu3)。
- 說出 Show-o 在單一 checkpoint 中處理的四種任務：文字生成、VQA、T2I、圖片 inpainting。
- 挑選 masking 排程（cosine、linear、truncated），並推論它對樣本品質的影響。

## 問題 (The Problem)

Transfusion 的雙 loss 訓練可行，但動態較棘手：連續的 diffusion loss 與離散的 NTP loss 處於不同的數值尺度。平衡 loss 權重是超參數搜尋。這個架構有效，但很複雜。

Show-o 的答案：讓兩種 modality 都保持離散（像 Chameleon），但以 masked discrete diffusion 平行生成圖片，而不是循序生成。訓練目標變成單一的 masked-token 預測，自然地泛化了 next-token 預測。

## 核心概念 (The Concept)

### Masked discrete diffusion (MaskGIT)

最初的 Chang et al. (2022) MaskGIT 技巧很優雅。從完全被遮住的圖片開始（每個 token 都是特殊的 `<MASK>` id）。每一步平行預測所有被遮住的 token，然後保留信心最高的 top-K 個預測，並重新遮住其餘的。經過約 8-16 次迭代，所有 token 都被填滿。每一步要解除多少個 token 的遮罩排程是調過的，cosine 排程效果不錯。

訓練很簡單：從 [0, 1] 均勻抽樣一個遮罩比例，套用到圖片的 VQ token 上，訓練 transformer 還原被遮住的 token。這正是 BERT 對文字所做的事，只是放大到圖片生成。

### Show-o：一個 transformer，混合 mask

Show-o 把 MaskGIT 放進 causal-language-model 的 transformer 裡。attention mask 是：

- 文字 token：causal（標準 LLM）。
- 圖片 token：在圖片區塊內完全雙向（讓被遮住的 token 在預測時能看到其他每個圖片 token）。
- Text-to-image：文字關注先前的圖片，圖片關注先前的文字。

訓練在以下之間交替：
1. 對文字序列做標準 NTP。
2. T2I 樣本：文字 → 帶有被遮住圖片 token 的圖片，使用 masked-token 預測 loss。
3. VQA 樣本：圖片 → 帶有被遮住文字 token 的文字（其實就是 NTP）。

統一的 loss 是對 `<MASK>` token 的 cross-entropy，涵蓋了文字 NTP（只有最後一個 token 被「遮住」）與圖片 masked-diffusion（隨機子集被遮住）。

### 平行取樣 (Parallel sampling)

Show-o 以約 16 步生成一張圖片，而不是約 1000 步（逐 token 的 autoregressive）或約 20 步（diffusion）。每一步平行預測所有被遮住的 token；確定 top-K 個有信心的；重複。

比較：
- Chameleon / Emu3（對 token 做 autoregressive）：N_tokens 次 forward pass，通常每張圖片 1024-4096 次。
- Transfusion（連續 diffusion）：約 20 步，每步一次完整的 transformer pass。
- Show-o（masked discrete diffusion）：約 16 步，每步一次完整的 transformer pass。

在相近規模的模型下，Show-o 比 Chameleon 快；步數大致與 Transfusion 相當，但每步成本較低（離散詞彙表 logits 對上連續 MSE loss）。

### 單一 checkpoint 中的任務 (Tasks in one checkpoint)

Show-o 在推論時支援四種任務，由 prompt 格式選擇：

- 文字生成：標準 autoregressive 文字輸出。
- VQA：圖片輸入，文字輸出。
- T2I：文字輸入，透過 masked discrete diffusion 輸出圖片。
- Inpainting：圖片的部分 token 被遮住，填補它們。

inpainting 能力是 masked 預測訓練免費帶來的。遮住 VQ-token 網格的某個區域，餵入其餘部分加上文字 prompt，預測被遮住的 token。

### Masking 排程 (Masking schedule)

每一步解除多少個 token 的排程會左右品質。Show-o 建議 cosine：

```math
\text{mask\_ratio}(t) = \cos\left(\frac{\pi t}{2T}\right) \qquad t = 0, \ldots, T
```

在第 0 步，所有 token 都被遮住（比例 1.0）。在第 $T$ 步，沒有任何被遮住。Cosine 把重心放在預測資訊量最大的中段比例。Linear 排程也行，但更快進入平台期。

### Show-o2

Show-o2（2025 年後續作品，arXiv 2506.15564）擴展了 Show-o：更大的 LLM 基底、更好的 tokenizer、改良的 mask 排程。相同的架構模式。

### Show-o 的定位 (Where Show-o sits)

在 2026 年的分類中：

- 離散 token + NTP：Chameleon、Emu3。簡單但推論慢。
- 離散 token + masked diffusion：Show-o、MaskGIT、LlamaGen、Muse。平行取樣，仍受 tokenizer 損失影響。
- 連續 + diffusion：Transfusion、MMDiT、DiT。品質最高，訓練較複雜。
- VLM 中的連續 + flow matching：JanusFlow、InternVL-U。最新。

依任務挑選：想在單一開源模型中同時要 T2I + inpainting + VQA、速度又合理時，用 Show-o；品質至上、又負擔得起雙 loss 接線時，用 Transfusion。

```figure
masked-diffusion-unmask
```

## 實際使用 (Use It)

`code/main.py` 模擬 Show-o 的取樣：

- 一個 16 個 VQ token 的玩具網格。
- 一個模擬的「transformer」，根據 prompt 與目前已解除遮罩的 token 預測 logits。
- 使用 cosine 排程、共 8 步的平行 masked 取樣。
- 印出中間狀態（mask 模式的演變）與最終 token。

執行它，看著 mask 一步一步消解。

## 交付 (Ship It)

這堂課產出 `outputs/skill-unified-gen-model-picker.md`。給定同時需要理解（VQA、captioning）與生成（T2I、inpainting），並受限於 open-weights 的產品，它會在 Show-o 家族、Transfusion/MMDiT 家族與 Emu3 / Chameleon 家族之間挑選，並說明具體的取捨。

## 練習 (Exercises)

1. Masked discrete diffusion 約以 16 步取樣。為什麼不是 1 步？如果在第 0 步就解除所有遮罩，會壞掉什麼？

2. Inpainting 在 masked diffusion 中是免費的。提出一個 Show-o 的 inpainting 勝過專門模型的產品使用情境（真實或假設的皆可）。

3. Cosine 排程對 linear 排程：追蹤 T=8 時每一步被解除遮罩的 token 數。哪一個更平衡？

4. 一張 512x512 的 Show-o 圖片是 1024 個 token。在詞彙表 $K=16384$ 下，模型輸出 $1024 \cdot \log_2(16384) = 14{,}336$ 位元（約 1.75 KiB）的資料。Stable Diffusion 輸出 512*512*24 位元 = 6,291,456 位元（約 768 KiB）的原始像素。壓縮比是多少？它換來什麼品質？

5. 閱讀 LlamaGen (arXiv:2406.06525)。LlamaGen 的 class-conditional autoregressive 圖片模型與 Show-o 的 masked 做法有何不同？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | 訓練時預測被遮住的 token；推論時迭代地解除信心最高的預測的遮罩 |
| Cosine schedule | "Unmask schedule" | 遮罩比例隨推論步驟衰減；把信心的增長集中在中段 |
| Parallel decoding | "All tokens at once" | 每一步在一次 forward pass 中預測所有被遮住的 token，然後確定 top-K |
| Hybrid attention | "Causal + bidirectional" | 在文字 token 間為 causal、在圖片區塊內為雙向的 mask |
| Inpainting | "Fill-in generation" | 以部分 token 被遮住的圖片為條件，預測缺失的部分；訓練目標免費帶來 |
| Commitment rate | "Top-K per step" | 每次迭代宣告「完成」的 token 數量；控制推論速度與品質的取捨 |

## 延伸閱讀 (Further Reading)

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
