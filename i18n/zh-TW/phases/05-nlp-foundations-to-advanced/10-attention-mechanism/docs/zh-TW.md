# Attention 機制 (Attention Mechanism) — 突破點

> Decoder 不再瞇著眼看一個壓縮過的摘要，而是開始看整個來源。在這之後的一切，都是 attention 加上工程。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Time:** ~45 分鐘

## 問題 (The Problem)

第 09 課結束在一個量測到的失敗。在玩具複製任務上訓練的 GRU encoder-decoder，準確率從長度 5 時的 89%，掉到長度 80 時接近隨機水準。原因是結構性的，不是訓練的 bug：encoder 蒐集到的每一點資訊，都必須塞進一個固定大小的 hidden state，而 decoder 看不到其他任何東西。

Bahdanau、Cho 和 Bengio 在 2014 年發表了一個三行的修正。不要只給 decoder 最後的 encoder state，而是保留每一個 encoder state。在每個 decoder 步驟，計算 encoder state 的加權平均，權重代表「decoder 現在需要多看 encoder 的位置 `i` 嗎？」這個加權平均就是 context，而且每個 decoder 步驟都會改變。

整個想法就是這樣。Transformer 擴充了它。Self-attention 把它套用在單一序列上。Multi-head attention 讓它平行執行。但 2014 年的版本已經打破了瓶頸，一旦你擁有它，轉向 transformer 就是工程問題，不是概念問題。

## 核心概念 (The Concept)

![Bahdanau attention: decoder queries all encoder states](../assets/attention.svg)

在每個 decoder 步驟 `t`：

1. 用先前的 decoder hidden state `s_{t-1}` 當作 **query**。
2. 對每個 encoder hidden state `h_1, ..., h_T` 打分。每個 encoder 位置一個純量。
3. 對分數做 softmax，得到總和為 1 的 attention weight `α_{t,1}, ..., α_{t,T}`。
4. Context vector `c_t = Σ α_{t,i} * h_i`。encoder state 的加權平均。
5. Decoder 取 `c_t` 加上先前輸出的 token，產生下一個 token。

重點就是這個加權平均。當 decoder 需要把 "Je" 翻成 "I" 時，它給 "Je" 上的 encoder state 高權重，其他的低權重。當它需要 "not" 時，就給 "pas" 高權重。context vector 每一步都會重塑。

## 形狀 (Shapes)（每個人都會踩到的坑）

每個 attention 實作第一次幾乎都在這裡出錯。請慢慢讀。

| 項目 | 形狀 | 備註 |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | 如果是 BiLSTM，`d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | 一個向量 |
| Attention score `e_{t,i}` | 純量 | 每個 encoder 位置一個 |
| Attention weight `α_{t,i}` | 純量 | 對所有 `i` 做 softmax 之後 |
| Context vector `c_t` | `(d_h,)` | 與 encoder state 形狀相同 |

**Bahdanau（additive）分數。** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`。

- `s_{t-1}` 的形狀是 `(d_s,)`，`h_i` 的形狀是 `(d_h,)`。
- `W_a` 的形狀是 `(d_attn, d_s)`。`U_a` 的形狀是 `(d_attn, d_h)`。
- 它們在 tanh 內的和，形狀是 `(d_attn,)`。
- `v_α` 的形狀是 `(d_attn,)`。與 `v_α` 做內積後會縮成一個純量。**這就是 `v_α` 的作用。** 沒什麼神奇的，它只是把 attention 維度的向量變成純量分數的投影。

**Luong（multiplicative）分數。** 三種變體：

- `dot`：`e_{t,i} = s_t^T * h_i`。要求 `d_s == d_h`，是硬性限制。如果你的 encoder 是雙向的，就跳過。
- `general`：`e_{t,i} = s_t^T * W * h_i`，其中 `W` 的形狀是 `(d_s, d_h)`。移除了維度相等的限制。
- `concat`：基本上就是 Bahdanau 的形式。因為前兩種比較便宜，很少使用。

**一個值得點名的 Bahdanau / Luong 陷阱。** Bahdanau 使用 `s_{t-1}`（產生當前字*之前*的 decoder state）。Luong 使用 `s_t`（*之後*的 state）。把兩者搞混，會產生微妙錯誤的梯度，極難除錯。挑一篇論文，並堅持它的慣例。

```figure
attention-heatmap
```

## 動手做 (Build It)

### 步驟 1：Additive (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

把你的形狀對照上面的表格檢查。`encoder_states` 的形狀是 `(T_enc, d_h)`。`projected_enc` 的形狀是 `(T_enc, d_attn)`。`projected_dec` 的形狀是 `(d_attn,)`，並且會 broadcast。`combined` 的形狀是 `(T_enc, d_attn)`。`scores` 的形狀是 `(T_enc,)`。`weights` 的形狀是 `(T_enc,)`。`context` 的形狀是 `(d_h,)`。可以出貨了。

### 步驟 2：Luong dot 與 general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

各三行。這就是 Luong 的論文能夠受到注目的原因：在大多數任務上準確率相同，程式碼少很多。

### 步驟 3：一個數值範例

給定三個 encoder state（大致代表 "cat"、"sat"、"mat"），以及一個與第一個最吻合的 decoder state，attention 分布會集中在位置 0。如果 decoder state 轉為與最後一個吻合，attention 就會移到位置 2，context vector 也跟著移動。

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

第一列勝出。接著把 decoder state 移近第三個 encoder state，看權重如何移動。就是這樣。Attention 就是明確的對齊 (alignment)。

### 步驟 4：為什麼這是通往 transformer 的橋樑

把上面的說法翻譯成 Q/K/V：

- **Query** = decoder state `s_{t-1}`
- **Key** = encoder state（我們拿來打分的對象）
- **Value** = encoder state（我們加權並加總的對象）

在傳統 attention 中，key 和 value 是同一個東西。Self-attention 把它們分開：你可以讓序列對自己做 query，並對 K 和 V 使用不同的學習投影。Multi-head attention 以不同的學習投影平行執行它。Transformer 把整個階段堆疊很多次，並拋棄 RNN。

數學相同，形狀相同。從 Bahdanau attention 到 scaled dot-product attention 的教學跳躍，大多只是符號上的差別。

## 實際使用 (Use It)

PyTorch 和 TensorFlow 直接提供 attention。

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

這就是一個 transformer attention 層。5 個位置的 query batch，10 個位置的 key/value batch，各 128 維，8 個 head。`output` 是加入新 context 後的 query。`weights` 是你可以視覺化的 5x10 對齊矩陣。

### 傳統 attention 仍然重要的時候

- 教學。單頭、單層、以 RNN 為基礎的版本，讓每個概念都看得見。
- Transformer 放不下的裝置端序列任務。
- 任何 2014 到 2017 年的論文。不懂 Bahdanau 的慣例，你會讀錯。
- 機器翻譯中細粒度的對齊分析。即使在 transformer 模型上，原始 attention weight 也是一種可解釋性工具，而讀懂它們需要知道它們是什麼。

### 把 attention weight 當作解釋的陷阱

Attention weight 看起來可解釋。它們是在各位置上總和為一的權重；你可以把它們畫出來；高代表「看了這裡」。審查者很愛它們。

它們沒有看起來那麼可解釋。Jain 和 Wallace (2019) 顯示，對某些任務，attention 分布可以被打亂並換成任意替代品，而不改變模型的預測。在沒有 ablation 或反事實檢查的情況下，絕對不要把 attention weight 當作推理的證據來回報。

## 交付 (Ship It)

存成 `outputs/prompt-attention-shapes.md`：

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## 練習 (Exercises)

1. **簡單。** 實作 `softmax` masking，讓 encoder 中的 padding token 得到的 attention weight 為零。在變動長度序列的 batch 上測試。
2. **中等。** 把 multi-head attention 加到 Luong 的 `general` 形式。把 `d_h` 拆成 `n_heads` 組，每個 head 各自做 attention，再串接起來。驗證單頭的情況與你先前的實作相符。
3. **困難。** 在第 09 課的玩具複製任務上，訓練帶有 Bahdanau attention 的 GRU encoder-decoder。畫出準確率對序列長度的圖，並與沒有 attention 的 baseline 比較。你應該會看到差距隨長度增加而擴大，證實 attention 解除了瓶頸。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Attention | 看著東西 | value 序列的加權平均，權重由 query 與 key 的相似度計算而來。 |
| Query, Key, Value | QKV | 三種投影：Q 負責問，K 是用來比對的，V 是要回傳的。 |
| Additive attention | Bahdanau | Feed-forward 分數：`v^T tanh(W q + U k)`。 |
| Multiplicative attention | Luong dot / general | 分數是 `q^T k` 或 `q^T W k`。比較便宜，多數任務上準確率相同。 |
| Alignment matrix | 漂亮的圖 | 以 `(T_dec, T_enc)` 格子呈現的 attention weight。讀它就能看到模型注意了什麼。 |

## 延伸閱讀 (Further Reading)

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — 就是那篇論文。
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — 三種分數變體及其比較。
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — 關於可解釋性的警告。
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — 可執行的 PyTorch 逐步講解。
