# 完整的 Transformer：Encoder + Decoder (The Full Transformer — Encoder + Decoder)

> Attention 是主角。其他一切，包含 residual、normalization、feed-forward、cross-attention，都是讓你能把它疊得很深的鷹架。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 分鐘

## 問題 (The Problem)

單一 attention 層是特徵擷取器，不是模型。每層只有一次 matmul，對語言來說容量不夠。你需要深度，而沒有正確的管線，深度就會壞掉。

2017 年 Vaswani 的論文打包了六個設計決策，把一個 attention 層變成可以堆疊的 block。之後的每個 transformer，不管是 encoder-only（BERT）、decoder-only（GPT）還是 encoder-decoder（T5），都繼承同一副骨架。2026 年這些 block 已經經過改良（RMSNorm、SwiGLU、pre-norm、RoPE），但骨架完全相同。

這堂課就是那副骨架。後面的課程會把它特化：06 做 encoder，07 做 decoder，08 做 encoder-decoder。

## 核心概念 (The Concept)

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### 六個零件 (The six pieces)

1. **Embedding + 位置訊號。** Token → 向量。位置透過 RoPE（現代）或 sinusoidal（經典）注入。
2. **Self-attention。** 每個位置 attend 到其他所有位置。Decoder 中會加 mask。
3. **Feed-forward network (FFN)。** 逐位置的兩層 MLP：`W_2 · activation(W_1 · x)`。預設擴張比為 4 倍。
4. **Residual connection。** `x + sublayer(x)`。沒有它，gradient 在大約 6 層之後就會消失。
5. **Layer normalization。** `LayerNorm` 或 `RMSNorm`（現代）。穩定 residual stream。
6. **Cross-attention（只有 decoder）。** Query 來自 decoder，key 和 value 來自 encoder 輸出。

看一個向量流過一個 block：attention 在各位置間混合，residual 把它往前帶，FFN 轉換它，norm 讓整條 stream 保持穩定。

```figure
transformer-block
```

### Encoder block（BERT、T5 encoder 使用） (Encoder block)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder 是雙向的。沒有 mask。所有位置都看得到所有位置。

### Decoder block（GPT、T5 decoder 使用） (Decoder block)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder 每個 block 有三個 sublayer。中間那個 cross-attention，是資訊從 encoder 流向 decoder 的唯一地方。在純 decoder-only 架構（GPT）中，cross-attention 會被省略，只剩 masked self-attention + FFN。

### Pre-norm 對 post-norm (Pre-norm vs post-norm)

原始論文：`x + sublayer(LN(x))` 對 `LN(x + sublayer(x))`。Post-norm 在 2019 年前後失寵，因為不仔細做 warmup 就很難訓練得深。Pre-norm（`LN` 放在 sublayer *之前*）是 2026 年的預設：Llama、Qwen、GPT-3+、Mistral 全都用它。

### 2026 年現代化的 block (The 2026 modernized block)

Vaswani 2017 用的是 LayerNorm + ReLU。現代的堆疊把兩者都換掉了。正式上線的 block 實際長這樣：

| 元件 | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN 擴張 | 4× | 2.6×（SwiGLU 用三個矩陣，總參數量相符） |
| 位置 | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA（或 MLA） |
| Bias 項 | 有 | 無 |

RMSNorm 拿掉了 LayerNorm 的 mean-centering（少一次減法），省下運算，而且經驗上至少一樣穩定。SwiGLU（`Swish(W1 x) ⊙ W3 x`）在 Llama、PaLM 和 Qwen 的論文中，一致比 ReLU/GELU 的 FFN 好約 0.5 個 ppl 點。

### 參數量 (Parameter count)

對於一個 `d_model = d`、FFN 擴張比為 `r` 的 block：

- MHA：`4 · d²`（Q、K、V、O projection）
- FFN (SwiGLU)：`3 · d · (r · d)` ≈ `3rd²`
- Norm：可忽略

在 `d = 4096, r = 2.6, layers = 32`（大約是 Llama 3 8B）時，總計：`32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (67M + 131M) ≈ 32 · 198M ≈ 6.3B` 個參數在 block 裡，再加上 embedding 與輸出 head 約 1B（128K vocab × 4096，不共用權重）。這樣就接近公開的 8B。真正的模型用 GQA（較小的 K 和 V projection）和較寬的 FFN（`r = 3.5`），把參數從 attention 移到 FFN，但總量仍接近 8B。

## 動手做 (Build It)

### 步驟 1：基本建構零件 (the building blocks)

使用第 03 課那個小型 `Matrix` 類別（為了獨立性，複製到這個檔案裡）：

- `layer_norm(x, eps=1e-5)` — 減去平均值，除以標準差。
- `rms_norm(x, eps=1e-6)` — 除以 RMS。不減平均值。
- `gelu(x)` 以及 `silu(x) * W3 x`（SwiGLU）。
- `ffn_swiglu(x, W1, W2, W3)`。
- `encoder_block(x, params)` 與 `decoder_block(x, enc_out, params)`。

完整的接線請看 `code/main.py`。

### 步驟 2：接出 2 層 encoder 和 2 層 decoder (wire a 2-layer encoder and a 2-layer decoder)

把它們疊起來。把 encoder 輸出傳給每個 decoder 的 cross-attention。在輸出 projection 之前加最後一個 LN。

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### 步驟 3：在玩具範例上跑 forward (run forward on a toy example)

餵進一個 6 個 token 的來源和一個 5 個 token 的目標。驗證輸出形狀是 `(5, vocab)`。不做訓練，這堂課談的是架構，不是 loss。

### 步驟 4：換成 RMSNorm + SwiGLU (swap in RMSNorm + SwiGLU)

把 LayerNorm 和 ReLU-FFN 換成 RMSNorm 和 SwiGLU。確認形狀仍然相符。這就是 2026 年的現代化，只要替換一個函式。

## 實際使用 (Use It)

PyTorch/TF 的參考實作：`nn.TransformerEncoderLayer`、`nn.TransformerDecoderLayer`。但 2026 年多數正式上線的程式碼都自己寫 block，因為：

- Flash Attention 是在 attention 內部呼叫，不是透過 `nn.MultiheadAttention`。
- GQA / MLA 不在標準函式庫的參考實作裡。
- RoPE、RMSNorm、SwiGLU 不是 PyTorch 的預設。

HF `transformers` 有乾淨的參考 block 值得一讀：`modeling_llama.py` 是 2026 年標準的 decoder-only block。它大約 500 行，值得走過一遍。

**Encoder、decoder、encoder-decoder，何時選哪個：**

| 需求 | 選擇 | 範例 |
|------|------|---------|
| 分類、embedding、針對文字的 QA | Encoder-only | BERT, DeBERTa, ModernBERT |
| 文字生成、聊天、程式碼、推理 | Decoder-only | GPT, Llama, Claude, Qwen |
| 結構化輸入 → 結構化輸出（翻譯、摘要） | Encoder-decoder | T5, BART, Whisper |

Decoder-only 在語言領域勝出，因為它 scale 得最乾淨，而且理解與生成都能處理。當輸入有明確的「來源序列」身分時（翻譯、語音辨識、結構化任務），encoder-decoder 仍然是最好的選擇。

## 交付 (Ship It)

請看 `outputs/skill-transformer-block-reviewer.md`。這個 skill 會對照 2026 年的預設，審查新的 transformer block 實作，並標出缺少的部分（pre-norm、RoPE、RMSNorm、GQA、FFN 擴張比）。

## 練習 (Exercises)

1. **簡單。** 在 `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True` 下，計算你的 encoder_block 的參數量。實作這個 block 並用 `sum(p.numel() for p in block.parameters())` 驗證。
2. **中等。** 從 post-norm 切換到 pre-norm。兩者都初始化，並在隨機輸入上，量測堆疊 12 層之後的 activation norm。Post-norm 的 activation 應該會爆炸；pre-norm 的應該維持在有界範圍。
3. **困難。** 在玩具複製任務上（把 `x` 反向複製）實作 4 層 encoder-decoder。訓練 100 步。回報 loss。換成 RMSNorm + SwiGLU + RoPE，loss 有下降嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Block | 「一個 transformer 層」 | norm + attention + norm + FFN 的堆疊，外面包著 residual connection。 |
| Residual | 「Skip connection」 | 輸出為 `x + f(x)`；讓 gradient 能流過很深的堆疊。 |
| Pre-norm | 「在前面 normalize，不是在後面」 | 現代做法：`x + sublayer(LN(x))`。不需要 warmup 的雜技就能訓練得更深。 |
| RMSNorm | 「沒有平均值的 LayerNorm」 | 除以 RMS；少一個運算，經驗上穩定性相同。 |
| SwiGLU | 「大家都換過去的 FFN」 | `Swish(W1 x) ⊙ W3 x → W2`。在 LM ppl 上勝過 ReLU/GELU。 |
| Cross-attention | 「Decoder 如何看到 encoder」 | Q 來自 decoder、K/V 來自 encoder 輸出的 MHA。 |
| FFN expansion | 「中間那層 MLP 有多寬」 | hidden size 對 d_model 的比例，通常是 4（LayerNorm）或 2.6（SwiGLU）。 |
| Bias-free | 「拿掉 +b 項」 | 現代堆疊在線性層中省略 bias；ppl 略有改善，模型也更小。 |

## 延伸閱讀 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 最初的 block 規格。
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — 為什麼 pre-norm 在深層網路上勝過 post-norm。
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm。
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — SwiGLU 論文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 2026 年標準的 decoder-only block。
