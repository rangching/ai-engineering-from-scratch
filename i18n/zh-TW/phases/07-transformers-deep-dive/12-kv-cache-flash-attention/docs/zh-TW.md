# KV Cache、Flash Attention 與推論最佳化 (KV Cache, Flash Attention & Inference Optimization)

> 訓練是平行的，受 FLOP 限制。推論是串行的，受記憶體限制。瓶頸不同，技巧也不同。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~75 分鐘

## 問題 (The Problem)

天真的 autoregressive decoder 要生成 $N$ 個 token，需要做 $O(N^2)$ 的工作：每一步都要對整個 prefix 重新計算 attention。對 4K token 的回應來說，那是 1600 萬次 attention 運算，其中大部分是多餘的。Prefix token 的每個 hidden state 一旦算出就是確定的，你只需要把新 token 的 query 對照先前所有內容已快取的 key 和 value 執行即可。

除此之外，attention 本身搬動大量資料。標準 attention 會具體產生 N×N 的 score 矩陣、N×d 的 softmax 輸出、N×d 的最終輸出，對 HBM 的讀寫次數太多。當 N≥2K 時，attention 在變成 FLOP 受限之前，就先變成記憶體受限。經典的 attention kernel 只用到現代 GPU 的 4 到 10 分之一。

兩項最佳化，都來自 Dao et al.，把前沿推論從「慢」推到「快」：

1. **KV cache。** 儲存每個 prefix token 的 K 和 V 向量。每個新 token 的 attention 就是一個 query 對上已快取的 key。每個生成步驟的推論從 $O(N^2)$ 降到 $O(N)$。
2. **Flash Attention。** 把 attention 計算切成 tile，讓完整的 N×N 矩陣永遠不會進入 HBM。所有 softmax + matmul 都在 SRAM 內完成。在 A100 上實際耗時加速 2 到 4 倍；在使用 FP8 的 H100 上加速 5 到 10 倍。

到了 2026 年，兩者都已普及。每個正式上線的推論堆疊（vLLM、TensorRT-LLM、SGLang、llama.cpp）都預設它們存在。每個前沿模型出貨時都啟用了 Flash Attention。

## 核心概念 (The Concept)

![KV cache growth and Flash Attention tiling](../assets/kv-cache-flash-attn.svg)

### KV cache 的數學 (KV cache math)

每個 decoder 層、每個 token、跨所有 KV head：

```
bytes_per_token_per_layer = 2 * n_kv_heads * d_head * dtype_size
                          ^
                          K and V
```

對一個 7B 模型，32 層、32 個 head（K 和 V 全部使用）、d_head=128、fp16：

```
per token per layer = 2 * 32 * 128 * 2 = 16384 bytes (16 KiB)
per token (32 layers) = 512 KiB
per 32K context = 16 GiB
```

對 Llama 3 70B（80 層、d_head=128、8 個 KV head 的 GQA）：

```
per token per layer = 2 * 8 * 128 * 2 = 4096 bytes (4 KiB)
per 32K context = 10 GiB
```

32K 時的 10 GiB，到 128K 就長到 40 GiB，光是 batch size 為 1 的 KV cache 就吃掉 40 GB A100 的全部記憶體。

**GQA 是 KV-cache 的勝利。** 有 64 個 KV head 的 MHA 會大 8 倍，在 32K 時約 80 GiB。MLA 壓縮得更多。

拖動各個維度，看 cache 大小怎麼變。把序列長度或 batch 調高，看它多快就超出單張 GPU：

```figure
kv-cache-sizer
```

### Flash Attention：Tiling 的招數 (Flash Attention — the tiling trick)

標準 attention：

```
S = Q @ K^T          (HBM read, N×N, HBM write)
P = softmax(S)       (HBM read, HBM write)
O = P @ V            (HBM read, HBM write)
```

三次 HBM 往返。在 H100 上，HBM 頻寬是 3 TB/s；SRAM 是 30 TB/s。每次 HBM 往返，相較於把所有東西留在晶片上，都慢了 10 倍。

Flash Attention：

```
for each block of Q (tile size ~128 × 128):
    load Q_tile into SRAM
    for each block of K, V:
        load K_tile, V_tile into SRAM
        compute S_tile = Q_tile @ K_tile^T     (SRAM)
        running softmax aggregation             (SRAM)
        accumulate into O_tile                  (SRAM)
    write O_tile to HBM
```

每個 tile 只需一次 HBM 往返。總記憶體用量從 $O(N^2)$ 降到 $O(N)$。Backward pass 會從 forward pass 重新計算一些值，而不是儲存它們，又是一個記憶體上的勝利。

**數值技巧。** Running softmax 在各個 tile 之間維護 `(max, sum)`，讓最後的 normalization 是精確的。這不是近似：Flash Attention 計算出的輸出與標準 attention 位元完全相同（除了 fp16 的不可結合性）。

**版本演進：**

| 版本 | 年份 | 關鍵改變 | 在參考硬體上的加速 |
|---------|------|-----------|-------------------------------|
| Flash 1 | 2022 | Tiled SRAM kernel | A100 上 2× |
| Flash 2 | 2023 | 更好的平行度、causal 優先的排序 | A100 上 3× |
| Flash 3 | 2024 | Hopper asynchrony、FP8 | H100 上 1.5–2×（~740 TFLOPs FP16） |
| Flash 4 | 2026 | Blackwell 5 階段 pipeline、軟體 exp2 | 推論優先（一開始只有 forward） |

Flash 4 推出時只支援 forward pass。訓練仍使用 Flash 3。Flash 4 對 GQA 與 varlen 的支援尚待完成（2026 年中）。

### Speculative decoding：另一個延遲上的勝利 (Speculative decoding — the other latency win)

便宜的模型提議 N 個 token。大模型平行驗證全部 N 個。如果驗證接受了 k 個 token，你只花了 1 次大模型 forward pass，就換到 k 次生成。在程式碼和散文上，典型的 k=3–5。

2026 年的預設：
- **EAGLE 2 / Medusa。** 整合式的 draft head，共用 verifier 的 hidden state。加速 2 到 3 倍，品質沒有損失。
- **使用 draft 模型的 speculative decoding。** 在消費級硬體上加速 2 到 4 倍。
- **Lookahead decoding。** Jacobi 迭代；不需要 draft 模型。小眾但免費。

### Continuous batching (Continuous batching)

經典的 batched 推論：等最慢的序列完成，再開始新的 batch。短回應提早完成時會浪費 GPU。

Continuous batching（最早在 Orca 推出，現在在 vLLM、TensorRT-LLM、SGLang 中）：舊請求一完成，就把新請求換進 batch。對典型的聊天工作負載，吞吐量提升 5 到 10 倍。

### PagedAttention：把 KV cache 當成虛擬記憶體 (PagedAttention — KV cache as virtual memory)

vLLM 的招牌功能。KV cache 以 16 個 token 為一個 block 來配置；page table 把邏輯位置對應到實體 block。讓你可以在平行取樣（beam search、parallel sampling）之間共用 KV、為 prompt caching 熱抽換 prefix，並整理記憶體碎片。比天真的連續配置吞吐量提升 4 倍。

```figure
flash-attention-memory
```

## 動手做 (Build It)

```figure
iso-machine {"machine":"kv-cache"}
```

請看 `code/main.py`。我們實作：

1. 天真的 $O(N^2)$ 增量 decoder。
2. $O(N)$ 的 KV-cache decoder。
3. 模擬 Flash Attention running-max 演算法的 tiled softmax。

### 步驟 1：KV cache (KV cache)

```python
class KVCache:
    def __init__(self, n_layers, n_heads, d_head):
        self.K = [[[] for _ in range(n_heads)] for _ in range(n_layers)]
        self.V = [[[] for _ in range(n_heads)] for _ in range(n_layers)]

    def append(self, layer, head, k, v):
        self.K[layer][head].append(k)
        self.V[layer][head].append(v)

    def read(self, layer, head):
        return self.K[layer][head], self.V[layer][head]
```

很簡單：在每層、每個 head 的 list 中，持續增長每個 token 的 K、V 向量。

### 步驟 2：Tiled softmax (tiled softmax)

```python
def tiled_softmax_dot(q, K, V, tile=4):
    """Flash-attention-style softmax(qK^T)V with running max/sum."""
    m = float("-inf")
    s = 0.0
    out = [0.0] * len(V[0])
    for start in range(0, len(K), tile):
        k_block = K[start:start + tile]
        v_block = V[start:start + tile]
        scores = [sum(qi * ki for qi, ki in zip(q, k)) for k in k_block]
        new_m = max(m, *scores)
        exp_old = math.exp(m - new_m) if m != float("-inf") else 0.0
        exp_new = [math.exp(sc - new_m) for sc in scores]
        s = s * exp_old + sum(exp_new)
        for j in range(len(out)):
            out[j] = out[j] * exp_old + sum(e * v[j] for e, v in zip(exp_new, v_block))
        m = new_m
    return [o / s for o in out]
```

輸出與一次做完的 `softmax(qK) V` 位元完全相同，但任何時刻的工作集都是 `tile × d_head` 的區塊，而不是完整的 `N × d_head`。

### 步驟 3：比較天真與有快取的 decoding (compare naive vs cached decoding)

計算 attention 運算次數。兩個 decoder 做的 query-key 比較次數相同：第 `t` 步 attend 到 `t` 個 token，所以總數是 1 + 2 + ... + N = N(N+1)/2。程式碼使用 N = 10，兩個 decoder 都印出 55。對 100 個 token 的生成，兩者都做 5050 次比較。Cache 省下的是 K 和 V 的 projection。沒有 cache 時，每一步都要為整個 prefix 重新計算 K 和 V，所以 100 個 token 的生成要計算 5050 個 token state。有 cache 時，每個 token 的 K 和 V 只計算一次，所以同樣的生成只計算 100 個 token state。

## 實際使用 (Use It)

```python
# HuggingFace transformers auto-enables KV cache on decoder-only generate().
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-3.2-3B",
    attn_implementation="flash_attention_2",  # use FA3 if Hopper
    torch_dtype="bfloat16",
)
# generate() uses KV cache automatically
```

vLLM 正式使用：

```bash
pip install vllm
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --tensor-parallel-size 4 \
    --max-model-len 32768 \
    --enable-prefix-caching \
    --kv-cache-dtype fp8
```

跨請求的 prefix caching 是 2026 年的一大勝利：相同的 system prompt、few-shot 範例或長 context 文件，可以在多次呼叫之間重複使用 KV。對於有重複工具 prompt 的 agent 工作負載，prefix caching 通常能讓吞吐量提升 5 倍。

## 交付 (Ship It)

請看 `outputs/skill-inference-optimizer.md`。這個 skill 會為新的推論部署挑選 attention 實作、KV cache 策略、quantization 與 speculative decoding。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。確認天真的與有快取的 decoder 產生相同的輸出；注意運算次數的差異。
2. **中等。** 實作 prefix caching：給定一個 prompt P 與數個補全，對 P 執行一次 forward pass 來填滿 KV cache，然後對每個補全分支。測量相較於每次都重新 encode P 的加速。
3. **困難。** 實作玩具版的 PagedAttention：KV cache 放在固定的 16 個 token block 中，並使用 free-list。序列完成時，把它的 block 歸還到池中。模擬 1,000 個長度不一的聊天補全。比較記憶體碎片化程度與連續配置的差異。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| KV cache | 「讓 decoding 變快的招數」 | 儲存每個 prefix token 的 K 和 V；新的 query 去 attend 它們，而不是重新計算。 |
| HBM | 「GPU 主記憶體」 | High Bandwidth Memory；H100 上有 80 GB，B200 上有 192 GB。頻寬約 3 TB/s。 |
| SRAM | 「晶片上記憶體」 | 每個 SM 的高速記憶體，H100 上每個 SM 約 256 KB。頻寬約 30 TB/s。 |
| Flash Attention | 「Tiled attention kernel」 | 在不於 HBM 中具體產生 N×N 矩陣的情況下計算 attention。 |
| Continuous batching | 「不用等的 batching」 | 把完成的序列換出、新的換入，不必清空整個 batch。 |
| PagedAttention | 「vLLM 的招牌」 | KV cache 以固定 block 搭配 page table 配置；消除碎片化。 |
| Prefix caching | 「重複使用長 prompt」 | 跨請求快取共用 prefix 的 KV；大幅降低 agent 的成本。 |
| Speculative decoding | 「Draft + verify」 | 便宜的 draft 模型提議 token；大模型一次 pass 驗證 k 個。 |

## 延伸閱讀 (Further Reading)

- [Dao et al. (2022). FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness](https://arxiv.org/abs/2205.14135) — Flash 1。
- [Dao (2023). FlashAttention-2: Faster Attention with Better Parallelism and Work Partitioning](https://arxiv.org/abs/2307.08691) — Flash 2。
- [Shah et al. (2024). FlashAttention-3: Fast and Accurate Attention with Asynchrony and Low-precision](https://arxiv.org/abs/2407.08608) — Flash 3。
- [FlashAttention-4 release notes (Dao-AILab, 2026)](https://github.com/Dao-AILab/flash-attention) — Blackwell 5 階段 pipeline 與軟體 exp2 技巧；請閱讀 repo 的 README，了解本課提到的僅限 forward 的推出限制。
- [Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention](https://arxiv.org/abs/2309.06180) — vLLM 論文。
- [Leviathan et al. (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — speculative decoding。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — 本課引用的整合式 draft 做法的 EAGLE-1/2 論文。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — 與 EAGLE 並列提及的 Medusa 做法。
- [vLLM docs — PagedAttention](https://docs.vllm.ai/en/latest/design/kernel/paged_attention.html) — 關於 16 個 token 的 block 與 page-table 設計的標準深入解說。
