# Positional Encoding：Sinusoidal、RoPE、ALiBi (Positional Encoding — Sinusoidal, RoPE, ALiBi)

> Attention 具有排列不變性。沒有位置訊號的話，"The cat sat on the mat" 和 "mat the on sat cat the" 會產生相同的輸出。三種演算法修正了這點，各自對「位置」的意義下了不同的賭注。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention)
**Time:** ~45 分鐘

## 問題 (The Problem)

Scaled dot-product attention 對順序是盲的。Attention 矩陣 `softmax(Q K^T / √d) V` 是由成對的相似度算出來的。把 `X` 的列打亂，輸出的列也會以同樣方式被打亂。Attention 內部完全不在乎位置。

對 bag-of-words 模型來說，這不是 bug。但對語言、程式碼、音訊、影片，也就是任何順序帶有意義的東西來說，這是致命的。

解法是想辦法把位置注入 embedding。答案分成三個時代：

1. **Absolute sinusoidal**（Vaswani 2017）。把位置的 `sin/cos` 加到 embedding 上。簡單、不需要學習，但超出訓練長度後的外推效果很差。
2. **RoPE — Rotary Position Embeddings**（Su 2021）。依位置按比例的角度旋轉 Q 和 K 向量。直接在 dot product 中編碼*相對*位置。2026 年的主流。
3. **ALiBi — Attention with Linear Biases**（Press 2022）。完全跳過 embedding；依距離對 attention score 加上每個 head 專屬的線性懲罰。長度外推能力極佳。

到 2026 年，幾乎所有前沿開放模型都用 RoPE：Llama 2/3/4、Qwen 2/3、Mistral、Mixtral、DeepSeek-V3、Kimi。少數長 context 模型使用 ALiBi 或它的現代變體。Absolute sinusoidal 已成歷史。

## 核心概念 (The Concept)

![Sinusoidal absolute vs RoPE rotations vs ALiBi distance bias](../assets/positional-encoding.svg)

### Absolute sinusoidal

預先計算一個固定矩陣 `PE`，形狀為 `(max_len, d_model)`：

```math
\begin{aligned}
PE_{(pos, 2i)} &= \sin\left(pos / 10000^{2i / d_{\text{model}}}\right) \\
PE_{(pos, 2i+1)} &= \cos\left(pos / 10000^{2i / d_{\text{model}}}\right)
\end{aligned}
```

然後在 attention 之前做 `X' = X + PE[:N]`。每個維度都是不同頻率的正弦波。模型學會從相位模式讀出位置。超過 `max_len` 就失效：模型只看過位置 0 到 2047，沒人告訴它位置 2048 會發生什麼事。

### RoPE

旋轉 Q 和 K 向量（不是 embedding）。對一對維度 $(2i, 2i+1)$：

```math
\begin{bmatrix} q'_{2i} \\ q'_{2i+1} \end{bmatrix} = \begin{bmatrix} \cos(pos \cdot \theta_i) & -\sin(pos \cdot \theta_i) \\ \sin(pos \cdot \theta_i) & \cos(pos \cdot \theta_i) \end{bmatrix} \begin{bmatrix} q_{2i} \\ q_{2i+1} \end{bmatrix}
\qquad \theta_i = \text{base}^{-2i / d_{\text{head}}}, \quad \text{base} = 10000 \text{ by default}
```

對 key 以位置 `pos_k` 套用同樣的旋轉。Dot product $q'_m \cdot k'_n$ 就只成為 $(m - n)$ 的函式。也就是說：**attention score 只取決於相對距離**，即使旋轉是依據絕對位置計算的。漂亮的把戲。

擴充 RoPE：可以縮放 `base`（NTK-aware、YaRN、LongRoPE），不用重新訓練就能外推到更長的 context。Llama 3 就是這樣把 context 從 8K 擴充到 128K。

### ALiBi

跳過 embedding 的把戲，直接對 attention score 加 bias：

```math
\text{attn\_score}[i, j] = \frac{q_i \cdot k_j}{\sqrt{d}} - m_h \cdot |i - j|
```

其中 $m_h$ 是每個 head 專屬的斜率（例如 $1 / 2^{8h/H}$）。較近的 token 被加分；較遠的 token 被扣分。沒有訓練時的成本。論文顯示它的長度外推勝過 sinusoidal，在原始訓練長度上則與 RoPE 相當。

### 2026 年該選哪個 (What to pick in 2026)

| 變體 | 外推能力 | 訓練成本 | 使用者 |
|---------|---------------|---------------|---------|
| Absolute sinusoidal | 差 | 免費 | 最初的 transformer、早期 BERT |
| Learned absolute | 無 | 極小 | GPT-2, GPT-3 |
| RoPE | 搭配 scaling 後良好 | 免費 | Llama 2/3/4, Qwen 2/3, Mistral, DeepSeek-V3, Kimi |
| RoPE + YaRN | 極佳 | fine-tune 階段 | Qwen2-1M, Llama 3.1 128K |
| ALiBi | 極佳 | 免費 | BLOOM, MPT, Baichuan |

RoPE 勝出，是因為它不必改變架構就能放進 attention、編碼相對位置，而且它的 `base` 超參數為長 context 的 fine-tuning 提供了乾淨的調整旋鈕。

```figure
rope-explorer
```

## 動手做 (Build It)

### 步驟 1：Sinusoidal 編碼 (sinusoidal encoding)

請看 `code/main.py`。只要 4 行的計算：

```python
def sinusoidal(N, d):
    pe = [[0.0] * d for _ in range(N)]
    for pos in range(N):
        for i in range(d // 2):
            theta = pos / (10000 ** (2 * i / d))
            pe[pos][2 * i]     = math.sin(theta)
            pe[pos][2 * i + 1] = math.cos(theta)
    return pe
```

在第一個 attention 層之前，把它加到 embedding 矩陣上。

### 步驟 2：把 RoPE 套用到 Q、K (RoPE applied to Q, K)

RoPE 直接在 Q 和 K 上就地運作。對每一對維度：

```python
def apply_rope(x, pos, base=10000):
    d = len(x)
    out = list(x)
    for i in range(d // 2):
        theta = pos / (base ** (2 * i / d))
        c, s = math.cos(theta), math.sin(theta)
        a, b = x[2 * i], x[2 * i + 1]
        out[2 * i]     = a * c - b * s
        out[2 * i + 1] = a * s + b * c
    return out
```

關鍵：對位置 $m$ 的 Q 和位置 $n$ 的 K 套用相同的函式。它們的 dot product 在每一對座標上都會多出一個 $\cos((m-n) \cdot \theta_i)$ 因子。Attention 免費就學會了相對位置。

### 步驟 3：ALiBi 的斜率與 bias (ALiBi slopes and bias)

```python
def alibi_bias(n_heads, seq_len):
    # slope_h = 2 ** (-8 * h / n_heads) for h = 1..n_heads
    slopes = [2 ** (-8 * (h + 1) / n_heads) for h in range(n_heads)]
    bias = []
    for m in slopes:
        row = [[-m * abs(i - j) for j in range(seq_len)] for i in range(seq_len)]
        bias.append(row)
    return bias  # add to attention scores before softmax
```

把 `bias[h]` 加到 head `h` 的 `(seq_len, seq_len)` attention score 矩陣上，再做 softmax。

### 步驟 4：驗證 RoPE 的相對距離性質 (verify relative-distance property of RoPE)

挑兩個隨機向量 `a, b`。以 `(pos_a, pos_b)` 旋轉。再以 `(pos_a + k, pos_b + k)` 旋轉。兩個 dot product 在浮點誤差範圍內必須一致。這個性質就是 RoPE 的全部意義：它對絕對偏移不變，只有相對間隔重要。

## 實際使用 (Use It)

PyTorch 2.5+ 在 `torch.nn.functional` 中附帶 RoPE 工具。多數正式上線的程式碼使用 `flash_attn` 或 `xformers`，RoPE 在 attention kernel 內部套用。

```python
from transformers import AutoModel
model = AutoModel.from_pretrained("meta-llama/Llama-3.2-3B")
# model.config.rope_scaling → {"type": "yarn", "factor": 32.0, "original_max_position_embeddings": 8192}
```

**2026 年的長 context 技巧：**

- **NTK-aware interpolation。** 從 4K 擴充到 16K 以上時，把 `base` 重新縮放成 `base * (scale_factor)^(d/(d-2))`。
- **YaRN。** 更聰明的 interpolation，能在長 context 上保留 attention entropy。Llama 3.1 128K 使用它。
- **LongRoPE。** Microsoft 在 2024 年提出的方法，用演化搜尋挑選每個維度的縮放因子。Phi-3-Long 使用它。
- **Position interpolation + fine-tuning。** 只要依擴充倍數縮小位置，再 fine-tune 1 到 5B 個 token。效果出奇地好。

## 交付 (Ship It)

請看 `outputs/skill-positional-encoding-picker.md`。這個 skill 會根據目標 context 長度、外推需求與訓練預算，為新模型挑選編碼策略。

## 練習 (Exercises)

1. **簡單。** 在 `max_len=512, d=128` 下，把 sinusoidal `PE` 矩陣畫成熱圖。確認「維度索引越大，條紋越寬」的模式。
2. **中等。** 實作 NTK-aware RoPE scaling。在長度 256 的序列上訓練一個小型 LM，再於長度 1024 上測試，比較有無 scaling 的差異。測量 perplexity。
3. **困難。** 在同一個 attention 模組中實作 ALiBi 和 RoPE。在長度 512 的序列上，用複製任務訓練 4 層 transformer。測試時外推到 2048。比較退化程度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Positional encoding | 「告訴 attention 順序」 | 任何加到 embedding 或 attention 上、用來編碼位置的訊號。 |
| Sinusoidal | 「最初的那個」 | 以等比頻率的 `sin/cos` 加到 embedding 上；無法外推。 |
| RoPE | 「Rotary embeddings」 | 以和位置相關的角度旋轉 Q、K；dot product 編碼了相對距離。 |
| ALiBi | 「線性 bias 把戲」 | 把 $-m \cdot \lvert i-j \rvert$ 加到 attention score；不需要 embedding，外推能力很好。 |
| base | 「RoPE 的旋鈕」 | RoPE 中的頻率縮放器；增大它即可在推論時擴充 context。 |
| NTK-aware | 「RoPE 的縮放技巧」 | 重新縮放 `base`，讓 context 擴充時高頻維度不會被擠壓。 |
| YaRN | 「花俏的那個」 | 逐維度的 interpolation 加 extrapolation，保留 attention entropy。 |
| Extrapolation | 「超出訓練長度仍可運作」 | 位置機制在超過訓練時見過的 `max_len` 之後，還能不能提供正確輸出？ |

## 延伸閱讀 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.5](https://arxiv.org/abs/1706.03762) — 最初的 sinusoidal。
- [Su et al. (2021). RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864) — RoPE 論文。
- [Press, Smith, Lewis (2021). Train Short, Test Long: Attention with Linear Biases Enables Input Length Extrapolation](https://arxiv.org/abs/2108.12409) — ALiBi。
- [Peng et al. (2023). YaRN: Efficient Context Window Extension of Large Language Models](https://arxiv.org/abs/2309.00071) — RoPE scaling 的最先進方法。
- [Chen et al. (2023). Extending Context Window of Large Language Models via Positional Interpolation](https://arxiv.org/abs/2306.15595) — Meta 的 Llama 2 長 context 論文。
- [Ding et al. (2024). LongRoPE: Extending LLM Context Window Beyond 2 Million Tokens](https://arxiv.org/abs/2402.13753) — Phi-3-Long 使用、並在「實際使用」一節引用的 Microsoft 方法。
- [HuggingFace Transformers — `modeling_rope_utils.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/modeling_rope_utils.py) — 每一種 RoPE scaling 方案（default、linear、dynamic、YaRN、LongRoPE、Llama-3）的正式等級實作。
