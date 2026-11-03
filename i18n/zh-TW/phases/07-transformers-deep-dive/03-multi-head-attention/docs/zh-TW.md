# Multi-Head Attention

> 一個 attention head 一次只學一種關係。八個 head 就學八種。Head 幾乎是免費的，多拿一點吧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention from Scratch)
**Time:** ~75 分鐘

## 問題 (The Problem)

單一 self-attention head 只會算出一個 attention 矩陣。這個矩陣只能捕捉一種關係，通常是對目前訓練訊號而言能讓 loss 最小的那一種。如果你的資料同時糾纏著主詞動詞一致、共指、長距離語篇和句法分塊，單一 head 會把它們全部抹進同一個 soft-max 分布，損失一半的訊號。

2017 年 Vaswani 論文的解法是：平行執行好幾個 attention 函式，每個都有自己的 Q、K、V projection，再把輸出 concatenate 起來。每個 head 在維度為 $d_{\text{model}} / n_{\text{heads}}$ 的較小子空間中運作。總參數量不變，表達能力卻提高了。

Multi-head attention 是 2026 年所有 transformer 的預設配備。唯一的爭論是 head 要有*多少個*，以及 key 和 value 要不要共用 projection（Grouped-Query Attention、Multi-Query Attention、Multi-head Latent Attention）。

## 核心概念 (The Concept)

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**拆分 (Split)。** 取形狀為 `(N, d_model)` 的 `X`。投影成 Q、K、V，形狀都是 `(N, d_model)`。Reshape 成 `(N, n_heads, d_head)`，其中 $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$。再 transpose 成 `(n_heads, N, d_head)`。

**平行 attend (Attend in parallel)。** 在每個 head 內執行 scaled dot-product attention。每個 head 產生 `(N, d_head)`。各個 head 在 embedding 的不同子空間上運作，在 attention 計算過程中彼此不會溝通。

**Concatenate 並 projection (Concatenate and project)。** 把 head 疊回 `(N, d_model)`，再乘上學習得來的輸出矩陣 `W_o`，形狀為 `(d_model, d_model)`。Head 就是在 `W_o` 這裡混合。

**為什麼有效。** 每個 head 都可以專精於某件事，而不必和其他 head 爭奪表示能力的額度。2019 到 2024 年的 probing 研究顯示 head 有各自不同的角色：位置 head、關注前一個 token 的 head、複製 head、命名實體 head、induction head（in-context learning 的基礎）。

**2026 年的變體譜系：**

| 變體 | Q head 數 | K/V head 數 | 使用者 |
|---------|---------|-----------|---------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | 壓縮成 low-rank | DeepSeek-V2, V3 |

GQA 是現代的預設，因為它把 KV-cache 記憶體砍掉 $N/G$ 倍，卻幾乎保有完整品質。MLA 更進一步，把 K/V 壓縮到 latent 空間，計算時再投影回來：多花一些 FLOPs，卻省下多得多的記憶體。

```figure
multihead-split
```

## 動手做 (Build It)

### 步驟 1：從現有的單一 head attention 拆分出多個 head (split heads from the single-head attention we already have)

拿第 02 課的 `SelfAttention`，外面包上一組 split/concat。numpy 實作請看 `code/main.py`；邏輯如下：

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

一次 reshape、一次 transpose，沒有迴圈。這正是 PyTorch 在 `nn.MultiheadAttention` 底下做的事。

### 步驟 2：對每個 head 執行 scaled-dot-product attention (run scaled-dot-product attention per head)

每個 head 拿到自己那一份 Q、K、V。Attention 變成 batched matmul：

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

在真實硬體上，`Qh @ Kh.transpose(...)` 就是一次 `bmm`。GPU 看到的是一個形狀為 `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)` 的 batched matmul。增加 head 是免費的。

### 步驟 3：Grouped-Query Attention 變體 (Grouped-Query Attention variant)

只有 key 和 value 的 projection 會改變。Q 取得 `n_heads` 組；K 和 V 取得 `n_kv_heads < n_heads` 組，並重複展開以對齊：

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

推論時這能省記憶體，因為 KV cache 裡只存 `n_kv_heads` 份，而不是 `n_heads` 份。Llama 3 70B 用 64 個 query head 搭配 8 個 KV head，cache 縮小 8 倍。

### 步驟 4：探查每個 head 學到了什麼 (probe what each head learned)

在一個短句上，用 4 個 head 執行 MHA。對每個 head 印出 `(N, N)` 的 attention 矩陣。即使是隨機初始化，你也會看到不同的 head 挑出不同的結構。這部分是真正的訊號，部分則來自子空間中的旋轉對稱性。

## 實際使用 (Use It)

在 PyTorch 中，一行版本：

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

PyTorch 2.5+ 的 GQA：

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**要用幾個 head？** 2026 年正式上線模型的經驗法則：

| 模型大小 | d_model | n_heads | d_head |
|------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` 幾乎總是落在 64 或 128。它是單一 head 能「看到」多少的單位。低於 32，head 就會和縮放因子 $\sqrt{d_{\text{head}}}$ 打架；高於 256，你就失去了「許多小型專家」的好處。

## 交付 (Ship It)

請看 `outputs/skill-mha-configurator.md`。這個 skill 會根據參數預算、序列長度與部署目標，為新的 transformer 推薦 head 數量、kv-head 數量與 projection 策略。

## 練習 (Exercises)

1. **簡單。** 拿 `code/main.py` 的 MHA，固定 `d_model=64`，把 `n_heads` 從 1 改到 16。在合成的複製任務上，畫出一個很小的單層模型的 loss。更多 head 是有幫助、持平，還是有害？
2. **中等。** 實作 MQA（所有 query head 共用一個 KV head）。測量參數量比完整 MHA 少多少。計算 N=2048 時，推論的 KV-cache 大小縮小多少。
3. **困難。** 實作一個迷你版的 Multi-head Latent Attention：把 K、V 壓縮到 rank-`r` 的 latent，把 latent 存進 KV cache，在 attention 時再解壓。`r` 要多少，cache 記憶體才會降到完整 MHA 的 1/8 以下，同時品質維持在驗證 ppl 的 1 bit 以內？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Head | 「單一 attention 電路」 | 一組維度為 $d_{\text{head}} = d_{\text{model}} / n_{\text{heads}}$ 的 Q/K/V projection，擁有自己的 attention 矩陣。 |
| d_head | 「Head 維度」 | 每個 head 的 hidden 寬度；正式上線時幾乎總是 64 或 128。 |
| Split / combine | 「Reshape 把戲」 | 在 attention 前後做 `(N, d_model) ↔ (n_heads, N, d_head)` 的 reshape 加 transpose。 |
| W_o | 「輸出 projection」 | Concatenate 各 head 之後套用的 `(d_model, d_model)` 矩陣；head 在此混合。 |
| MQA | 「單一 KV head」 | Multi-Query Attention：共用單一 K/V projection。KV cache 最小，品質略有損失。 |
| GQA | 「Llama 2 以來的預設」 | Grouped-Query Attention，`n_kv_heads < n_heads`；重複展開以對齊 Q。 |
| MLA | 「DeepSeek 的招數」 | Multi-head Latent Attention：K、V 壓縮成 low-rank latent，在 attend 時解壓。 |
| Induction head | 「in-context learning 背後的電路」 | 一對 head，偵測先前出現過的內容，並複製它們後面接著的東西。 |

## 延伸閱讀 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — 最初的 multi-head 規格。
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — MQA 論文。
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — 如何在訓練後把 MHA 轉成 GQA。
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA，以及它為何在 cache 記憶體上勝過 MHA/GQA。
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — 從機制層面看 head 實際在做什麼。
