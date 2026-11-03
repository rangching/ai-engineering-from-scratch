# Attention 變體：Sliding Window、Sparse、Differential (Attention Variants — Sliding Window, Sparse, Differential)

> Full attention 是一個圓。每個 token 都看得到每個 token，記憶體為此付出代價。三種變體把圓的形狀彎曲，省下一半的成本。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head), Phase 7 · 12 (KV Cache / Flash Attention)
**Time:** ~60 分鐘

## 問題 (The Problem)

Full attention 的記憶體與運算量都是序列長度的 $O(N^2)$。對 128K context 的 Llama 3 70B 來說，每層有 160 億個 attention 項目，再乘以 80 層。Flash Attention（第 12 課）藏起了 $O(N^2)$ 的 activation 記憶體，但沒有改變算術成本，每個 token 仍然要 attend 到其他每個 token。

有三類變體改變了 attention 矩陣本身的拓樸：

1. **Sliding window attention (SWA)。** 每個 token 只 attend 到固定視窗內的鄰居，而不是整個 prefix。記憶體與運算量降到 $O(N \cdot W)$，其中 $W$ 是視窗大小。使用者有 Gemma 2/3、Mistral 7B 的前幾層、Phi-3-Long。
2. **Sparse / block attention。** 只對選定的 `(i, j)` 配對計分；其餘被強制為零權重。使用者有 Longformer、BigBird、OpenAI sparse transformer。
3. **Differential attention。** 用不同的 Q/K projection 計算兩張 attention map，把其中一張減掉另一張。消除會把權重流失到前幾個 token 的「attention sink」。Microsoft 的 DIFF Transformer（2024）。

這些可以並存。2026 年的前沿模型常常混用：大部分的層是 SWA-1024，每第五層是 global full attention，另外有少數 differential head 負責清理檢索。Gemma 3 的 5:1 SWA 對 global 比例，是目前教科書上的預設。

## 核心概念 (The Concept)

### Sliding Window Attention (SWA) (Sliding Window Attention (SWA))

位置 $i$ 的每個 query 只 attend 到 $[i - W, i]$ 內的位置（causal SWA），或 $[i - W/2, i + W/2]$（雙向）。視窗外的 token 在 score 矩陣中得到 $-\infty$。

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

當 $N = 8192$、$W = 1024$ 時，score 矩陣期望上有 1024 × 8192 個非零列，減少 8 倍。

**KV cache 隨 SWA 縮小。** 每層只需保留 K 和 V 的最後 $W$ 個 token。以 Gemma-3 類似的設定（1024 視窗、128K context）為例，KV cache 縮小 128 倍。

**品質成本。** 只用 SWA 的 transformer 在長距離檢索上會吃力。解法：把 SWA 層與 full-attention 層交錯。Gemma 3 使用 5:1 的 SWA:global。Mistral 7B 用的是 causal-SWA 堆疊，資訊透過重疊的視窗「向前流動」，每一層讓有效感受野擴大 $W$，經過 `L` 層之後，模型可以 attend 回溯 `L × W` 個 token。

### Sparse / Block Attention (Sparse / Block Attention)

事先挑選一個 `N × N` 的稀疏模式。三種典型形狀：

- **Local + strided（OpenAI sparse transformer）。** Attend 到最後 $W$ 個 token，加上在那之前每隔 `stride` 個的 token。以 $O(N \cdot \sqrt{N})$ 的運算量同時捕捉局部與長距離。
- **Longformer / BigBird。** 局部視窗 + 一小組 global token（例如 `[CLS]`），它們會 attend 所有人，也被所有人 attend，再加上 random-sparse 連結。經驗上在相同品質下有 2 倍的 context。
- **Native Sparse Attention（DeepSeek，2025）。** 學習 `(Q, K)` 的哪些 block 重要；在 kernel 層級跳過零區塊。與 FlashAttention 相容。

Sparse attention 是個 kernel 工程的故事。數學很簡單（遮住 score 矩陣）；勝利來自於永遠不把零的項目載入 SRAM。FlashAttention-3 與 2026 年的 FlexAttention API，讓自訂的 sparse 模式在 PyTorch 中成為一級公民。

### Differential Attention（DIFF Transformer，2024）(Differential Attention (DIFF Transformer, 2024))

一般的 attention 有「attention sink」問題：softmax 強迫每一列的總和為 1，所以不想特別 attend 任何東西的 token，就會把權重倒在第一個 token（或前幾個）上。這會偷走本該分給真實內容的容量。

Differential attention 透過計算**兩張** attention map 並相減來修正：

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d}) \\
\text{DiffAttn} &= (A_1 - \lambda \cdot A_2) V
\end{aligned}
```

其中 $\lambda$ 是學習得來的純量（通常為 0.5 到 0.8）。A1 捕捉真實內容的權重；A2 捕捉 sink。相減後 sink 被抵銷，權重被重新分配給相關的 token。

回報的結果（Microsoft 2024）：perplexity 低 5% 到 10%，在相同訓練長度下有效 context 長 1.5 到 2 倍，needle-in-haystack 檢索更精準。

### 變體比較 (Variant Comparison)

| 變體 | 運算量 | KV cache | 對比 full 的品質 | 正式使用情況 |
|---------|---------|----------|-----------------|----------------|
| Full attention | $O(N^2)$ | 每層 $O(N)$ | 基準 | 每個模型的預設層 |
| SWA (window 1024) | $O(N \cdot W)$ | 每層 $O(W)$ | -0.1 ppl，搭配 global 層表現良好 | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | $O(N \cdot \sqrt{N})$ | 混合 | 與 SWA 相近 | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | 約 $O(N)$ | 混合 | 在 2 倍 context 下與 full 相當 | 早期長 context BERT |
| Native Sparse (DeepSeek-V3.2) | $O(N \cdot \text{active fraction})$ | $O(N)$ | 差距在 0.05 ppl 以內 | DeepSeek-V3.2, 2025 |
| Differential | $O(2 \cdot N^2)$ | $O(2N)$ | -5 到 -10% ppl | DIFF Transformer, 2026 年初的模型 |

```figure
gqa-kv-sharing
```

## 動手做 (Build It)

請看 `code/main.py`。我們實作一個 causal mask 比較器，在玩具序列上並排展示 full、SWA、local+strided 與 differential attention。

### 步驟 1：Full causal mask（基準）(full causal mask (baseline))

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

來自第 07 課的基準。下三角；對角線以上權重為零。

### 步驟 2：Sliding window causal mask (sliding window causal mask)

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

只有一個參數 `window`。當 `window >= n` 時，就還原成 full causal attention。當 `window = 1` 時，每個 token 只 attend 自己。

### 步驟 3：Local + strided sparse mask (local + strided sparse mask)

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

密集的局部視窗，加上一路回溯到序列開頭、每隔 `stride` 個的 token。隨著層數增加，感受野以 log 步伐成長。

### 步驟 4：Differential attention (differential attention)

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

兩次 attention pass，以學習得來的混合係數相減。在程式碼中，我們比較單一與 differential 的 attention-sink 熱圖，並觀察 sink 如何消失。

### 步驟 5：KV cache 大小 (KV cache sizes)

印出每個變體在 $N = 131072$ 時每層的 cache 大小。SWA 與 sparse 變體縮小 10 到 100 倍。Differential 則加倍。請有意識地支付你的記憶體帳單。

## 實際使用 (Use It)

2026 年的正式使用模式：

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

PyTorch 2.5+ 的 FlexAttention 接受 mask 函式：

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

這會編譯成自訂的 Triton kernel。對常見模式來說，速度在 FlashAttention-3 的 10% 以內，而且 mask 函式就是一個 Python callable。

**何時選哪個：**

- **純 full attention**：context 到約 16K 的每一層，或檢索品質至上時。
- **SWA + global 混合**：長 context（>32K），訓練與推論受記憶體限制。超過 32K 時 2026 年的預設。
- **Sparse block attention**：自訂 kernel、自訂模式。保留給特殊工作負載（檢索、音訊）。
- **Differential attention**：任何 attention-sink 汙染會造成傷害的工作負載（長 context RAG、needle-in-haystack）。

## 交付 (Ship It)

請看 `outputs/skill-attention-variant-picker.md`。這個 skill 會根據目標 context 長度、檢索需求與訓練 / 推論算力概況，為新模型挑選 attention 拓樸。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。驗證 `window=4` 的 SWA 會把每一列中最後 4 個 token 以外的所有位置歸零。驗證 `window=n` 會位元完全相同地重現 full causal attention。
2. **中等。** 在第 07 課的總結專題之上，實作 `window=1024` 的 causal SWA。在 tinyshakespeare 上訓練 1,000 步。相較於 full attention，val loss 退步多少？尖峰記憶體降低多少？
3. **困難。** 在總結專題的模型中，實作 Gemma-3 風格的 5:1 層混合（5 個 SWA、1 個 global）。在相同參數量下，與純 SWA 及純 global 基準比較 loss、記憶體與生成品質。
4. **困難。** 實作每個 head 都有學習式 $\lambda$ 的 differential attention。在合成的檢索任務上訓練（一根針、2,000 個干擾項）。在相同參數量下，與單一 attention 基準比較檢索準確率。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | 「局部 attention」 | 每個 query 只 attend 它最後 $W$ 個 token；KV cache 縮小為 $O(W)$。 |
| Effective receptive field | 「模型能往回看多遠」 | 在視窗為 $W$ 的 $L$ 層 SWA 堆疊中，最多 $L \times W$ 個 token。 |
| Longformer / BigBird | 「Local + global + random」 | 帶有少數永遠參與 attention 的 global token 的 sparse 模式；早期的長 context 做法。 |
| Native Sparse Attention | 「DeepSeek 的 kernel 招數」 | 學習 block 層級的稀疏性；在 kernel 層級跳過零區塊，同時保持品質。 |
| Differential attention | 「兩張 map，一張相減」 | DIFF Transformer：從第一張 attention map 減去學習式 $\lambda$ 乘以第二張，以抵銷 attention sink。 |
| Attention sink | 「權重流向 token 0」 | Softmax normalization 強迫各列總和為 1；沒有資訊量的 query 把權重倒在位置 0。 |
| FlexAttention | 「Mask 即 Python」 | PyTorch 2.5+ 的 API，把任意 mask 函式編譯成 FlashAttention 形狀的 kernel。 |
| Layer type mix | 「5:1 SWA 對 global」 | 在堆疊中交錯 sparse 與 full attention 層，以較低的記憶體維持品質。 |

## 延伸閱讀 (Further Reading)

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — 標準的 sliding-window + global-token 論文。
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random。
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI 的 local+strided 模式。
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — 1:1 的 SWA:global 混合。
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — 視窗為 1024 的 5:1 混合，現在是教科書上的預設。
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — DIFF Transformer 論文。
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — DeepSeek-V3.2 的學習式稀疏 attention。
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — 「實際使用」一節中 mask-as-callable 模式的 API 參考。
