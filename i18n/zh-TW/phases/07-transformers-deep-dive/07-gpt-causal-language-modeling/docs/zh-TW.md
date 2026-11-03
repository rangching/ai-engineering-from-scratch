# GPT：Causal Language Modeling (GPT — Causal Language Modeling)

> BERT 兩邊都看。GPT 只看過去。這個三角形 mask，是現代 AI 裡影響最深遠的一行程式碼。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 分鐘

## 問題 (The Problem)

語言模型只回答一個問題：給定前 `t-1` 個 token，第 `t` 個 token 的機率分布是什麼？用這個訊號（next-token prediction）訓練，你就得到一個能一次生成一個 token、產出任意文字的模型。

要在整個序列上平行地端到端訓練，每個位置的預測就只能依賴更早的位置。否則模型會直接看答案作弊。

Causal mask 就是做這件事。它是一個只含 `-inf` 的上三角矩陣，在 softmax 之前加到 attention score 上。Softmax 之後，這些位置變成 0。每個位置只能 attend 到自己和更早的位置。而且因為你只對整個序列套用一次，一次 forward pass 就能得到 N 個平行的 next-token 預測。

GPT-1（2018）、GPT-2（2019）、GPT-3（2020）、GPT-4（2023）、GPT-5（2025）、Claude、Llama、Qwen、Mistral、DeepSeek、Kimi，它們全都是 decoder-only 的 causal transformer，核心迴圈相同。區分它們的是資料品質、規模、架構改良，以及 post-training（SFT、RLHF、DPO 和它們的後繼者）。

## 核心概念 (The Concept)

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mask (The mask)

給定長度為 $N$ 的序列，建立一個 $N \times N$ 的矩陣：

```math
M_{ij} = \begin{cases} 0 & \text{if } j \le i \\ -\infty & \text{if } j > i \end{cases}
```

在 softmax 之前，把 `M` 加到原始 attention score 上。$\exp(-\infty) = 0$，所以被遮住的位置貢獻的權重為零。Attention 矩陣的每一列，都是只涵蓋先前位置的機率分布。

實作成本：一次 `torch.tril()` 呼叫。計算時間：奈秒等級。對整個領域的影響：一切。

### 三角形從哪裡來 (Where the triangle comes from)

Mask 通常被呈現成接在 attention 上的一塊補丁。把推導倒過來走，它就不再神祕：attention 是 prefix average 的第三次改良，而三角形就是那個平均的迴圈範圍，寫成矩陣的樣子。

**階段 1：prefix average。** 對序列最笨的 causal 摘要：位置 $i$ 變成位置 $0 \ldots i$ 的平均。寫成迴圈就是 `out[i] = X[:i+1].mean(0)`。同樣的運算也可以是一次矩陣乘法。取一個全是 1 的下三角矩陣，每一列除以其個數，然後相乘：

```python
import numpy as np

A = np.tril(np.ones((n, n)))
A = A / A.sum(axis=1, keepdims=True)
out = A @ X
```

$A$ 的第 $i$ 列是 $[1/(i+1), \ldots, 1/(i+1), 0, \ldots, 0]$。對角線以上的零就是 causality。並沒有任何關於未來的東西被遮掉；未來從頭到尾就不在這個加總裡。

**階段 2：學習得來的權重。** 均勻平均把每個過去的 token 都當成同樣相關。把那些 1 換成學習得來的分數矩陣 `S`。現在各列不再自動加總為 1，所以改用 softmax 對每一列做 normalize，而不是除以個數。Softmax 永遠不會輸出剛好為零的值，這會破壞 causality，除非未來的分數以 `-inf` 輸入，因為 `exp(-inf) = 0`：

```python
def softmax(x, axis):
    e = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return e / e.sum(axis=axis, keepdims=True)

S = S + np.triu(np.full((n, n), -np.inf), k=1)
A = softmax(S, axis=1)
out = A @ X
```

同樣的三角形、同樣的 row-stochastic 矩陣、同樣的一次 matmul。`-inf` mask 不是新機制。它只是階段 1 的零項，翻譯到 softmax 的輸入域而已。

**階段 3：依內容而定的權重。** 在階段 2 中，`S` 在訓練後就固定了：不管 token 內容是什麼，位置 7 對位置 3 的權重永遠一樣。讓分數依賴 token 本身：`S = Q @ K.T / sqrt(d_k)`。其他一切都不變。Mask、softmax、matmul 完全相同。

三個階段，一個不變量：一個下三角的 row-stochastic 矩陣乘上序列。均勻平均、學習得來的靜態權重、依內容而定的權重。Mask 從來不是加到 attention 上的東西，它是從平均那裡一路留下來的。

```figure
mask-derivation
```

### 平行訓練，串行推論 (Parallel training, serial inference)

訓練：對整個 `(N, d_model)` 序列做一次 forward pass，計算 N 個 cross-entropy loss（每個位置一個），加總，backprop。沿著序列平行。這就是 GPT 訓練能 scale 的原因，你可以在一次 GPU pass 中處理一個 batch 的 1M 個 token。

推論：你一個 token 一個 token 地生成。餵入 `[t1, t2, t3]`，得到 `t4`。餵入 `[t1, t2, t3, t4]`，得到 `t5`。餵入 `[t1, t2, t3, t4, t5]`，得到 `t6`。KV cache（第 12 課）會儲存 `t1…tn` 的 hidden state，讓你不必每一步都重新計算。但推論時的串行深度 = 輸出長度。這就是 autoregressive 的代價，也是 decoding 成為每個 LLM 延遲瓶頸的原因。

### Loss：位移一格 (The loss — shift-by-one)

給定 token `[t1, t2, t3, t4]`：

- 輸入：`[t1, t2, t3]`
- 目標：`[t2, t3, t4]`

對每個位置 `i`，計算 `-log P(target_i | inputs[:i+1])`。加總。這就是整個序列的 cross-entropy。

你聽過的每個 transformer LM 都用這個 loss 訓練。Pre-training、fine-tuning、SFT，都是同一個 loss，只是資料不同。

### Decoding 策略 (Decoding strategies)

訓練完之後，取樣的選擇比大家想的更重要。

| 方法 | 做什麼 | 何時使用 |
|--------|--------------|-------------|
| Greedy | 每一步取 argmax | 確定性任務、程式碼補全 |
| Temperature | 把 logit 除以 T，再取樣 | 創意任務，T 越高多樣性越高 |
| Top-k | 只從前 k 個 token 取樣 | 砍掉低機率的長尾 |
| Top-p (nucleus) | 從累積機率 ≥ p 的最小集合取樣 | 2020 年之後的預設；會適應分布形狀 |
| Min-p | 保留 `p > min_p * max_p` 的 token | 2024 年之後；比 top-p 更擅長拒絕長尾 |
| Speculative decoding | Draft 模型提議 N 個 token，大模型驗證 | 同樣品質下延遲降低 2 到 3 倍 |

2026 年，對開放權重模型來說，min-p + temperature 0.7 是合理的預設。Speculative decoding 對任何正式上線的推論堆疊來說都是基本配備。

### 「GPT 配方」為何有效 (What made the "GPT recipe" work)

1. **Decoder-only。** 沒有 encoder 的額外開銷。每層只需一次 attention + FFN。
2. **Scaling。** 124M → 1.5B → 175B → 數兆。Chinchilla scaling law（第 13 課）告訴你如何分配算力。
3. **In-context learning。** 在 6B 到 13B 左右浮現。模型不必 fine-tune 就能遵循 few-shot 範例。
4. **RLHF。** 依據人類偏好做 post-training，把原始的預訓練文字轉換成聊天助理。
5. **Pre-norm + RoPE + SwiGLU。** 大規模下穩定訓練。

自 GPT-2 以來，核心架構沒有太大改變。所有有趣的事都發生在資料、規模和 post-training。

```figure
causal-mask
```

## 動手做 (Build It)

### 步驟 1：Causal mask (the causal mask)

請看 `code/main.py`。一行搞定：

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

在 softmax 之前把它加到 attention score 上。這就是整個機制。

### 步驟 2：2 層的類 GPT 模型 (a 2-layer GPT-ish model)

堆疊兩個 decoder block（masked self-attention + FFN，沒有 cross-attention）。加上 token embedding、positional encoding，以及 unembedding（與 token embedding 矩陣共用權重，這是 GPT-2 以來的標準手法）。

### 步驟 3：端到端的 next-token 預測 (next-token prediction, end-to-end)

在 20 個 token 的玩具詞彙上，於每個位置產生 logit。針對位移一格的目標計算 cross-entropy loss。不做 gradient，這是 forward pass 的健全性檢查。

### 步驟 4：取樣 (sampling)

實作 greedy、temperature、top-k、top-p、min-p。在固定的 prompt 上各跑一次並比較輸出。一個取樣函式只要 10 行。

## 實際使用 (Use It)

PyTorch，2026 年的慣用寫法：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

在底層，`generate()` 會執行 forward pass，取出最後位置的 logit，取樣下一個 token，附加上去，然後重複。每個正式上線的 LLM 推論堆疊（vLLM、TensorRT-LLM、llama.cpp、Ollama、MLX）都用大量最佳化實作同一個迴圈：batched prefill、continuous batching、KV cache paging、speculative decoding。

**GPT 對 BERT，各一句話：** GPT 預測 $P(x_t \mid x_{<t})$。BERT 預測 $P(x_{\text{masked}} \mid x_{\text{unmasked}})$。Loss 決定了模型能不能生成。

## 交付 (Ship It)

請看 `outputs/skill-sampling-tuner.md`。這個 skill 會為新的生成任務挑選取樣參數，並在需要確定性 decoding 時提出警示。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`，驗證 causal attention 矩陣在 softmax 之後是下三角。抽查：第 3 列的權重應該只出現在第 0 到 3 欄。
2. **中等。** 實作寬度為 4 的 beam search。在 10 個短 prompt 上，比較 beam-4 與 greedy 的 perplexity。Beam 總是贏嗎？（提示：翻譯通常是，開放式聊天則不是。）
3. **困難。** 實作 speculative decoding：用一個很小的 2 層模型當 draft，6 層模型當 verifier。在 100 筆長度 64 的補全上測量實際耗時的加速。確認輸出與 verifier 的 greedy 結果一致。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Causal mask | 「那個三角形」 | 加到 attention score 上的上三角 `-inf` 矩陣，讓位置 `i` 只看得到位置 `≤ i`。 |
| Next-token prediction | 「那個 loss」 | 模型分布對每個位置上真實下一個 token 的 cross-entropy。 |
| Autoregressive | 「一次生成一個」 | 把輸出餵回去當輸入；只有訓練時能平行，生成時不行。 |
| Logits | 「Softmax 之前的分數」 | LM head 在 softmax 之前的原始輸出；取樣就是對這些值進行。 |
| Temperature | 「創意旋鈕」 | 把 logit 除以 $T$；$T \to 0$ = greedy，$T \to \infty$ = 均勻分布。 |
| Top-p | 「Nucleus sampling」 | 把分布截斷到總和 $\ge p$ 的最小集合；從剩下的取樣。 |
| Min-p | 「比 top-p 更好」 | 保留 `p ≥ min_p × max_p` 的 token；截斷點會隨分布的尖銳程度調整。 |
| Speculative decoding | 「Draft + verify」 | 便宜的模型提議 N 個 token；大模型平行驗證。 |
| Teacher forcing | 「訓練技巧」 | 訓練時餵入真實的前一個 token，而不是模型的預測。每個 seq2seq LM 的標準做法。 |

## 延伸閱讀 (Further Reading)

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1。
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2。
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 與 in-context learning。
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding 論文。
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — 標準的 causal-LM 參考程式碼。
