# Gradient Checkpointing 與 Activation Recomputation (Gradient Checkpointing and Activation Recomputation)

> Backprop 會保留每一個中間 activation。在 70B 參數、128K context 下，每個 rank 的 activation 高達 3 TB。Checkpointing 用 FLOPs 換記憶體：重新計算，而不是儲存。問題在於該丟掉哪些區段，而答案不是「全部」。

**Type:** Build
**Languages:** Python (with numpy, optional torch)
**Prerequisites:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Time:** ~70 分鐘

## 問題 (The Problem)

訓練 transformer 時，每一層都要儲存 backward 中每個被微分的運算的輸入：attention 的輸入、Q/K/V 投影、softmax 輸出、FFN 的輸入、norm 的輸出，以及 residual stream。對於 hidden size 為 `d`、序列長度為 `L`、batch 為 `B` 的一層，每層大約是 `12 * B * L * d` 個 float 的量級。

對於 `d=8192, L=8192, B=1`，那大約是 8.05 億個值，也就是 BF16 下每層 1.6 GB。一個 64 層的模型約有 103 GB 的 activation，而這還沒乘上 microbatch 大小，還沒加上 attention-softmax 的中間值（每個 head 是 `L^2`），也還沒考慮 tensor-parallel 的部分副本。

雙面的帳單：BF16 權重加上 optimizer state 也許放得進 80GB，但 activation 會讓你超出。Gradient checkpointing（又稱 activation recomputation）是標準的修正方式。丟掉大部分 activation；在 backward 期間重做 forward 把它們找回來。代價：額外的 FLOPs。好處：記憶體依 checkpoint 區段數與總層數的比例下降。

天真地做的話，checkpointing 讓每一步的 forward pass FLOPs 增加約 33%。做得好的話，也就是依照 Korthikanti 等人的「聰明選擇」做 selective checkpointing，你可以用不到 5% 的 FLOP 開銷省下 5 倍的記憶體。而搭配 FP8 matmul、FSDP offload 與 expert-parallel MoE，這真的很重要：你負擔不起記憶體，也負擔不起浪費的運算。

## 核心概念 (The Concept)

### Backward 實際需要什麼 (What Backward Actually Needs)

`output = layer(input)`。Backward 想要 `grad_input` 與 `grad_params`。要計算它們，它需要：

- `input`（用來計算線性層的 `grad_params = input.T @ grad_output`）
- 一些 activation 導數的中間值（ReLU/GELU/softmax 的導數取決於 activation 的值）

Forward pass 會自動把這些儲存在 autograd 圖裡。每個 `tensor.retain_grad()`，以及每個需要其輸入的運算，都會保留一個參照。

### 天真的完整 Checkpointing (Naive Full Checkpointing)

把網路切成 `N` 個區段。Forward 期間，只儲存每個區段的*輸入*。當 backward 需要中間值時，重新執行該區段的 forward pass 來具現它們，然後微分。

範例：把 32 層 transformer 切成 32 個區段，每個區段 1 層。

- 記憶體：32 個層輸入（小）對比 32 * (每層的 activation 量)（巨大）。
- 額外運算：每個區段多一次 forward，也就是總共約多 33% 的 forward FLOPs（因為 backward 是 forward 的 2 倍，完整的一步從 1 + 2 = 3 個單位變成 1 + 1 + 2 = 4 個單位）。

這是最初 Chen 等人 2016 年的配方：每 `sqrt(L)` 層設一個 checkpoint，以平衡記憶體與運算。對於 $L=64$，那就是 8 個 checkpoint。

### Selective Checkpointing（Korthikanti 2022）(Selective Checkpointing (Korthikanti 2022))

不是所有 activation 的成本都一樣。Attention softmax 的輸出是 `B*L*L*heads`，並隨序列長度*二次方*成長。FFN 的 hidden activation 是 `B*L*4d`，線性成長。對於長序列，softmax 佔主導。

Selective checkpointing 保留儲存成本低的 activation（線性投影、residual），只重新計算昂貴的那些（attention）。你用最少的 FLOPs 重新計算，卻省下 $O(L^2)$ 的記憶體。

Megatron-Core 把這個實作為「selective」activation recomputation。2024 年以後的大多數 frontier 訓練都在使用。

### Offload (Offload)

重新計算的替代方案：在 forward 與 backward 之間，把 activation 送到 CPU RAM。需要 PCIe 頻寬；當閒置的頻寬超過重新具現的成本時才有利。混合策略很常見：對某些層做 checkpoint，對其他層做 offload。

FSDP2 把 offload 作為一等公民選項提供。當 GPU 的瓶頸在記憶體，而 CPU-GPU 傳輸還有餘裕時，offload 就很出色。

### Recompute 成本模型 (Recompute Cost Model)

在 `L` 層中每 `k` 層做一次天真 checkpointing 時，每一步的 FLOPs：

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

使用 selective checkpointing 時，你只重新計算 attention kernel，而不是整層：

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### 記憶體節省模型 (Memory Savings Model)

每層的 activation 量：`A`。對於 `L` 層，總 activation 記憶體：`L * A`。

完整 checkpoint（區段大小 1）：只儲存 `L * input_volume`（標準 transformer 約為 `L * 1/10 A`）。省下約 `9 * L * A * 1/10`。

每 `k` 層做一次 checkpoint：儲存 `L/k * A`，加上目前作用中的區段內 `k-1` 層的量。

在 `k = sqrt(L)` 時，記憶體與重新計算成本都隨 `sqrt(L)` 縮放，這對於成本一致的層來說是最佳的取捨。

### 何時不該做 Checkpoint (When Not to Checkpoint)

- 已經在進行中的 pipeline 階段的最內層。反正它們都得完成。
- 如果第一層與最後一層主導了該階段的運算（在 transformer 中少見）。
- 已經使用 FlashAttention 的 attention kernel：Flash 本來就會快速重新計算 softmax，所以在其之上額外做層級的 checkpointing 增益很小。

### 實作模式 (Implementation Patterns)

1. **函式包裝：** 用 `torch.utils.checkpoint.checkpoint(fn, input)` 包住一個區段。PyTorch 只儲存 `input`，並在 backward 時重新計算其他所有東西。

2. **以 decorator 為基礎：** 把層標記為可 checkpoint；由 trainer 在設定階段決定哪些區段要被包裝。

3. **手動明確重新計算：** 自己寫 backward pass，呼叫一個自訂的 `recompute_forward`，用儲存的輸入複製 forward。

三者在功能上給出相同的結果。包裝函式是標準的慣用寫法。

### 與 TP / PP / FP8 的互動 (Interaction with TP / PP / FP8)

- **Tensor parallel：** 重新計算時，checkpoint 的輸入必須被收集 (gather) 或重新分散 (rescatter)；要處理通訊成本。
- **Pipeline parallel：** 典型的模式是對每個 pipeline 階段的 forward 做 checkpoint，讓逆序的 microbatch 能重複使用 activation 記憶體。
- **FP8 recompute：** 重新計算期間更新的 amax 歷史必須與原始 forward 的相符，否則 FP8 的 scale 會漂移。大多數框架會為 scale 做快照。

```figure
activation-recompute
```

## 動手做 (Build It)

### 步驟 1：帶有區段的玩具模型 (Step 1: A Toy Model With Segments)

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### 步驟 2：需要所有 Activation 的天真 Backward (Step 2: Naive Backward Needing All Activations)

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### 步驟 3：每 k 層做 Checkpoint 的記憶體 (Step 3: Checkpoint-Every-k Memory)

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### 步驟 4：成本模型 (Step 4: Cost Model)

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### 步驟 5：記憶體估算器 (Step 5: Memory Estimator)

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### 步驟 6：最佳區段大小 (Step 6: Optimal Segment Size)

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### 步驟 7：Selective Checkpoint 的決策 (Step 7: Selective Checkpoint Decision)

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## 實際使用 (Use It)

- **torch.utils.checkpoint**：`from torch.utils.checkpoint import checkpoint`，PyTorch 中標準的包裝函式。包住一個函式；只儲存輸入，在 backward 時重新計算。
- **Megatron-Core activation recomputation**：支援 `selective`、`full` 與 `block` 模式。是 2024 年以後 frontier 訓練的標準。
- **FSDP2 offload**：`module.to_empty(device="cpu")` 搭配 FSDP2 中的 `offload_policy`，把 activation 分片到 CPU，而不是重新計算。
- **DeepSpeed ZeRO-Offload**：針對 optimizer state 與 activation 的 CPU offload，與 checkpointing 互補。

## 交付 (Ship It)

這堂課會產出 `outputs/prompt-activation-recompute-policy.md`，這個 prompt 接收你的模型設定（層數、hidden、seq、batch）與可用的 GPU 記憶體，並輸出每層的重新計算策略（none / selective / full / offload）。

## 練習 (Exercises)

1. 驗證正確性。執行 `model_forward` + `model_backward`（完整 activation）與 `model_forward_checkpointed` + `model_backward_checkpointed`（分段）。參數 gradient 必須在機器精度內完全相同。

2. 掃過區段大小 `k` 從 1 到 `L`。畫出 FLOP 開銷與記憶體。找出曲線的轉折點。

3. 實作 selective checkpointing：儲存 attention 模組的輸入，但不儲存它的中間值。在 seq=8192 的 32 層模型上，量測相對於整層 checkpointing 的 FLOP 開銷。

4. 加入 offload。把區段輸入存到模擬的「CPU buffer」（一個獨立的 list）。把「PCIe 頻寬」量測為 bytes/時間，並找出 offload 與重新計算之間的損益平衡點。

5. 對真實的 PyTorch transformer，在有和沒有 `torch.utils.checkpoint` 的情況下做 benchmark。量測記憶體（透過 `torch.cuda.max_memory_allocated`）與每步耗時。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|----------------------|
| Gradient checkpointing | 「重做 forward 來省記憶體」 | 只儲存區段的輸入；在 backward 期間重新計算中間值，以取得支援 gradient 的 tensor |
| Activation recomputation | 「跟 checkpointing 一樣」 | 同一種技術在 HPC 領域的叫法 |
| Segment size (k) | 「每個 checkpoint 有幾層」 | 中間值一起被丟棄並重新具現的層數 |
| Selective checkpointing | 「Korthikanti 的技巧」 | 只重新計算儲存成本高的 activation（attention softmax），保留便宜的 |
| Full checkpointing | 「天真的版本」 | 在每個區段中重新計算每一層的中間值 |
| Block checkpointing | 「粗粒度」 | 對整個 transformer block 做 checkpoint，是最大的粒度 |
| FLOP overhead | 「運算稅」 | 每一步額外的 FLOPs = (重新計算 FLOPs) / (fwd + bwd FLOPs)，天真版為 33%，selective 為 5% |
| Activation offload | 「送到 CPU」 | 在 forward 到 backward 之間，把 activation 移到 CPU RAM，是重新計算的替代方案 |
| sqrt-L rule | 「經典的最佳解」 | 對成本一致的層，最佳的 checkpoint 間距是 $\sqrt{L}$ 層 |
| Attention-softmax volume | 「O(L^2) 問題」 | $L^2 \cdot \text{heads} \cdot \text{batch}$ 個 float，在長 context 下主導 activation 記憶體 |

## 延伸閱讀 (Further Reading)

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174)：將 gradient checkpointing 正式化的原始論文
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198)：selective activation recomputation 與正式的成本分析
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645)：透過 reverse-mode rematerialization 的另一種固定記憶體做法
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840)：大規模的 activation offload
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html)：標準 API
- [Megatron Bridge activation recomputation documentation](https://docs.nvidia.com/nemo/megatron-bridge/latest/training/activation-recomputation.html)：selective、full 與 block 模式
