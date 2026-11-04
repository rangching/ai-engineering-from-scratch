# Native Sparse Attention（DeepSeek NSA）(Native Sparse Attention (DeepSeek NSA))

> 在 64k 個 token 時，attention 吃掉 70-80% 的 decode 延遲。每個開放模型實驗室都有修正它的計畫。DeepSeek 的 NSA (ACL 2025 最佳論文) 是真正留下來的那一個：三條平行的 attention 分支，分別是壓縮的粗粒度 token、選擇性保留的細粒度 token，以及用於局部 context 的 sliding window，透過一個學習得來的 gate 結合在一起。它與硬體對齊（對 kernel 友善），原生可訓練（可用於 pre-training，而不是推論時才硬加上去），而且在 64k 的 decode 上，它比 FlashAttention 更快，同時追平或勝過完整 attention 的品質。這堂課端到端打造這三條分支，並說明為什麼這種稀疏性是端到端可微分的。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 12 (KV cache, flash-attention), Phase 7 · 15 (attention variants), Phase 10 · 16 (differential attention)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 NSA 的三條 attention 分支，以及每一條捕捉什麼。
- 解釋為什麼 NSA 是「原生可訓練」，而先前的 sparse-attention 方法只能用於推論。
- 以壓縮 block 大小與選擇 top-k 的函數，計算 NSA 相對於完整 attention 在 64k context 下節省的 attention 運算。
- 用標準函式庫 Python，在一條簡短的合成序列上實作三分支結合，並驗證 gate 權重的行為。

## 問題 (The Problem)

序列長度為 N 時，完整 attention 的成本是每層 `O(N^2)` 的時間與 `O(N)` 的 KV cache。在 64k 個 token 時，運算與記憶體頻寬的數字是災難性的。NSA 論文中的理論估計：在 64k 時，attention 佔了總 decode 延遲的 70-80%。下游的一切，包括 TTFT、tokens/sec、每百萬 token 的成本，都由 attention 的成本主導。

Sparse attention 是顯而易見的答案。先前的嘗試分為兩類。固定模式的稀疏性（sliding-window、strided、block-local）會丟掉資訊，在長距離回想任務上失敗。推論時的稀疏性（KV cache 剪枝、H2O、StreamingLLM）是套用在以 dense attention pre-train 的模型上，只能拿回潛在加速的一小部分，因為模型從未被要求透過稀疏模式傳遞資訊。

Native Sparse Attention (Yuan et al., DeepSeek + PKU + UW, ACL 2025 最佳論文, arXiv:2502.11089) 兩者兼顧：一個模型在 pre-training 期間學到的稀疏模式，並實作成與 kernel 對齊的演算法，在推論時真的帶來運算的節省。兩年後，NSA 或它的直系後代，會是每個 frontier 長 context 模型的預設 attention。

## 核心概念 (The Concept)

### 三條平行分支 (Three parallel branches)

對每個 query，NSA 針對 KV cache 的三種不同視角，執行三次 attention：

1. **壓縮分支 (Compressed branch)。** Token 被分成大小為 `l` 的 block（通常是 32 或 64）。每個 block 透過一個小型學習式 MLP 壓縮成單一的摘要 token。Query 對這些壓縮後的 token 做 attention，得到整條序列的粗粒度視角。

2. **選擇分支 (Selected branch)。** 利用壓縮分支的 attention 分數，找出與目前 query 最相關的前 k 個 block。讀取這些 block 中細粒度（未壓縮）的 token，query 對它們全部做 attention。可以把壓縮分支的 attention 想成是選擇的路由訊號。

3. **Sliding-window 分支。** Query 對最近的 `W` 個 token（通常是 512）做 attention，以取得局部 context。這條分支捕捉結構性很重的短距離模式（語法、局部共指），這是另外兩條可能漏掉的。

三條分支的輸出透過學習得來的每個位置 gate 結合：

```math
\text{out} = g_{\text{cmp}} \cdot \text{out}_{\text{cmp}} + g_{\text{sel}} \cdot \text{out}_{\text{sel}} + g_{\text{win}} \cdot \text{out}_{\text{win}}
```

`g_cmp, g_sel, g_win` 是來自 query 上的小型 MLP 的 gate 權重。它們不需要加總為 1，可以獨立地為各分支加權。

### 為什麼這是「原生可訓練」(Why this is "natively trainable")

選擇步驟（top-k block）是離散的。離散運算會破壞 gradient 流動。先前的 sparse-attention 工作，要不是跳過穿過選擇步驟的 backprop（限制了訓練），就是使用連續鬆弛，卻沒有在推論時給出真正的稀疏性。

NSA 繞過了這個問題：壓縮分支的 attention，本身就是對整條序列的可微分粗粒度 attention。Top-k 運算只是重複使用壓縮分支中最高的 attention 分數，來挑選要載入哪些細粒度 block。Gradient 透過壓縮分支自己的輸出，傳到壓縮分支的分數，而不是透過 block 的選擇。在被選中的 block 內，對被選中的 key 和 value 的 attention 是普通的 attention，所以它們對最終輸出的貢獻也是可微分的。`top_k` 的索引選擇本身沒有 gradient：它只控制哪些 block 從記憶體載入。

這就是 NSA 可以端到端用於 pre-training 的原因。模型學會共同地透過三條分支傳遞資訊，產生一個在推論時真的帶來承諾加速的稀疏模式。

### 與硬體對齊的 kernel (Hardware-aligned kernel)

NSA 的 kernel 是針對現代 GPU 記憶體階層設計的。這個 kernel 依 GQA group 載入 query（外層迴圈），為每個 group 取得對應的稀疏 KV block（內層迴圈），並在 SRAM 上執行 attention。因為每個 query group 看到相同的被選 block（選擇是依 query group，而不是依 query head），KV 的載入成本被整個 group 攤提。Arithmetic intensity 維持在高檔。

論文回報，Triton kernel 在 64k 的 decode 上比 FlashAttention 快 9 倍，加速倍率隨序列長度成長。forward 與 backward 的 kernel 都有提供。

### 運算預算 (The compute budget)

令 `N` 為序列長度，`l` 為壓縮 block 大小，`k` 為 top-k 選擇數量，`w` 為 sliding window，`b` 為被選 block 大小（通常等於 `l`）。

- 壓縮分支：每個 query `O(N/l)` 個 key，所以總共 `O(N * N / l)`。
- 選擇分支：每個 query `O(k * b)` 個 key，所以 `O(N * k * b)`。
- Sliding 分支：每個 query `O(w)` 個 key，所以 `O(N * w)`。

總計：`O(N * (N/l + k*b + w))`。

在 `N = 64k, l = 64, k = 16, b = 64, w = 512` 時：每個 query 的成本是 `1000 + 1024 + 512 = 2536 個 key`。完整 attention 是 `64000 個 key`。運算減少 25 倍。

在 `N = 128k, l = 64, k = 16, b = 64, w = 512` 時：每個 query 的成本是 `2000 + 1024 + 512 = 3536 個 key`。完整 attention 是 `128000 個 key`。減少 36 倍。好處隨序列長度成長，這正是重點。

### 它的比較 (How does it compare)

| 方法 | 可微分 | 真實的推論加速 | 長距離回想 |
|--------|---------------|----------------------|-------------------|
| 僅 Sliding window | 是 | 是 | 失敗 |
| Strided / block-sparse | 是 | 是 | 部分 |
| KV 剪枝 (H2O, StreamingLLM) | N/A（推論時） | 是 | 部分 |
| MoBA (Moonshot) | 部分 | 是 | 良好 |
| NSA | 是（原生） | 是（64k 時 9 倍） | 追平完整 attention |

MoBA (Moonshot, arXiv:2502.13189) 是同時發表的，採取類似的「三個比一個好」做法，把 MoE 原則套用到 attention block。NSA 與 MoBA 是 2026 年長 context pre-training 需要認識的兩個架構。

```figure
sliding-window-attention
```

## 動手做 (Build It)

`code/main.py` 在一條簡短的合成序列上實作這三條分支，並展示：

- 壓縮 MLP（為了教學清晰，使用簡單的 mean-pool baseline；真正的 NSA 使用學習式 MLP）。
- 由壓縮分支分數驅動的 top-k block 選擇。
- 對最後 `w` 個 token 的 sliding-window attention。
- 閘控組合。
- 與完整 attention 比較的運算量計數輸出。

### 步驟 1：把 token 壓縮成 block (Step 1: compress tokens into blocks)

```python
def compress(K, l):
    n = len(K)
    n_blocks = (n + l - 1) // l
    out = []
    for b in range(n_blocks):
        start, end = b * l, min((b + 1) * l, n)
        block = K[start:end]
        summary = [sum(row[d] for row in block) / len(block) for d in range(len(K[0]))]
        out.append(summary)
    return out
```

### 步驟 2：壓縮分支的 attention (Step 2: compressed-branch attention)

對 query 與壓縮後的 key 執行 softmax attention。壓縮分支的分數同時也是 top-k 選擇的訊號。

### 步驟 3：Top-k block 選擇 (Step 3: top-k block selection)

挑出壓縮 block 中分數最高的 `k` 個索引。從這些 block 載入原始未壓縮的 token，並對它們執行 attention。

### 步驟 4：Sliding-window attention (Step 4: sliding-window attention)

取最後 `w` 個 token，並對它們執行標準 attention。

### 步驟 5：Gate + 組合 (Step 5: gate + combine)

query 上的小型 MLP 產生三個 gate 權重。最終輸出是三條分支輸出的加權和。

### 步驟 6：運算量計數 (Step 6: compute counting)

印出每條分支每個 query attend 的 key 數量與總數。與 `N`（完整 attention）比較。在 1024 個 token 的合成序列、`l = 32, k = 4, w = 128` 的情況下，NSA 每個 query 看到 `32 + 128 + 128 = 288` 個 key，完整 attention 則是 1024，少了 3.5 倍。

## 實際使用 (Use It)

NSA 正出貨在 DeepSeek 自己的長 context pre-training 管線中。截至 2026 年 4 月，公開推論堆疊的整合狀態：

- **DeepSeek 內部**：原生支援，公開發布的權重使用 NSA 或它的後繼者 DSA (Deepseek Sparse Attention)。
- **vLLM**：針對 DeepSeek-V3.x 權重的實驗性 NSA 支援開發中。
- **SGLang**：已發布 NSA benchmark；正式環境的路徑跟隨 vLLM。
- **llama.cpp / CPU**：不支援；在 CPU 的吞吐量下，kernel 分解的開銷不值得。

何時該用 NSA：

- 以 64k 以上 context 為目標、並擁有可觀運算預算的 pre-training 或持續訓練。
- 推論 DeepSeek 自己的長 context checkpoint。這些權重是 NSA 原生的。

何時不該用：

- 服務一個既有的 dense-attention pre-trained 模型。不經持續訓練，你無法事後加裝 NSA。
- Context 小於 16k。三分支的開銷會蓋過節省。
- Batch-1 的互動式聊天。對延遲敏感的 decode 有好處，但只在長 context 時。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-nsa-integrator.md`。給定一個長 context pre-training 的規格，它會產出 NSA 整合計畫：壓縮 block 大小、top-k、sliding window、gate MLP 寬度、kernel 選擇，以及能證明這個架構改動合理的特定長 context eval。

## 練習 (Exercises)

1. 在 1024 個 token 的合成序列上執行 `code/main.py`。對三組預設值掃過 `(l, k, w)`，並印出運算量計數。找出在 needle-in-haystack 測試中，相對於完整 attention 維持 95% recall 的同時，達到最低每個 query key 數量的預設值。

2. 把 mean-pool 壓縮器換成一個很小的學習式 MLP（2 層、hidden 32）。在訊號是一個 block 平均值的合成任務上訓練它。量測在保留資料上，相對於 mean-pool baseline 的 perplexity 差距。

3. 實作 gate MLP。它接收 query 作為輸入，輸出三個純量。展示 gate 的行為合理：對隨機的 query 接近均勻加權，當 query 命中很久以前的 block 時，對選擇分支給予很大的權重。

4. 計算啟用 NSA 的 70B 模型在 128k context 下的 KV cache 記憶體預算。KV head 為 8，head dim 為 128，BF16。與完整 attention 以及 MLA 比較（Phase 10 · 14 展示了 MLA 的數字）。找出 NSA 細粒度分支的 KV cache 等於完整 attention 的序列長度。

5. 閱讀 NSA 論文 (arXiv:2502.11089) 的第 4 節，並用三句話解釋，為什麼壓縮分支的 attention 分數被重複用於 top-k 選擇，而不是另外計算一個路由分數。把答案與 gradient 流動連結起來。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Compressed branch | 「粗略的視角」 | 對 block 平均後的 key 做 attention，以每個 query $O(N/l)$ 個 key 提供全域 context |
| Selected branch | 「Top-k block」 | 對壓縮分支分數最高的 `k` 個 block，做細粒度的 attention |
| Sliding window | 「局部 context」 | 對最後 `W` 個 token 做 attention，處理短距離模式 |
| Native trainability | 「開著稀疏性來 pre-train」 | 稀疏模式是在 pre-training 期間學到的，而不是推論時才硬加上去 |
| Compression block size l | 「粗略視角的分組大小」 | 有多少個 token 被合併成一個摘要，通常是 32-64 |
| Top-k | 「要保留的 block」 | 其未壓縮 token 會被讀取的壓縮 block 數量，通常是 16 |
| Sliding window W | 「局部 attention 半徑」 | 通常是 512，太短會損害局部連貫性，太長則浪費運算 |
| Branch gate | 「如何混合這三者」 | 每個位置的 MLP 輸出，用來加權三條分支的貢獻 |
| Hardware alignment | 「對 kernel 友善的稀疏性」 | 選擇稀疏模式，使實際的 GPU kernel 達到理論上的加速 |
| DSA | 「NSA 的後繼者」 | Deepseek Sparse Attention，在 DeepSeek 的系譜中，繼 NSA 之後的架構 |

## 延伸閱讀 (Further Reading)

- [Yuan et al. — Native Sparse Attention: Hardware-Aligned and Natively Trainable Sparse Attention (arXiv:2502.11089, ACL 2025 Best Paper)](https://arxiv.org/abs/2502.11089)：這篇論文
- [DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：NSA 所針對的架構家族
- [Moonshot AI — MoBA: Mixture of Block Attention for Long-Context LLMs (arXiv:2502.13189)](https://arxiv.org/abs/2502.13189)：同時期的工作，對 block 做 MoE 風格的 attention
- [Beltagy et al. — Longformer: The Long-Document Transformer (arXiv:2004.05150)](https://arxiv.org/abs/2004.05150)：sliding-window 的起源
- [Xiao et al. — StreamingLLM: Efficient Streaming Language Models with Attention Sinks (arXiv:2309.17453)](https://arxiv.org/abs/2309.17453)：NSA 所改進的推論時稀疏性 baseline
- [Dao et al. — FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)：NSA kernel 在 64k 時所超越的完整 attention baseline
