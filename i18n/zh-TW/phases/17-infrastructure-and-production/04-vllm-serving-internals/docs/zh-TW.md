# Serving Engine 內部原理 (Serving Engine Internals)：PagedAttention、Continuous Batching、Chunked Prefill

> 現代 serving-engine 的 throughput 建立在三個疊加的預設值上，而不是單一技巧。PagedAttention 永遠開啟。Continuous batching 在 decode 迭代之間把新的 request 注入進行中的 batch。Chunked prefill 把長 prompt 切片，讓 decode tokens 不會被餓死。三者全開後，單張 H100 SXM5 上的 Llama 3.3 70B FP8 在 128 並行下可達 2,200-2,400 tok/s，大約比 vLLM 自己的預設高約 25%，是單純 PyTorch 迴圈的 3-4 倍。這堂課以你能畫出來的程度，閱讀 vLLM（這三項技術的參考 engine）的 scheduler 與 attention kernel，最後在 `code/main.py` 用一個玩具 continuous batcher，照 vLLM 的方式排程 prefill 與 decode。

**Type:** Learn
**Languages:** Python (stdlib, toy continuous batching scheduler)
**Prerequisites:** Phase 17 · 01 (Model Serving), Phase 11 (LLM Engineering)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 把 PagedAttention 解釋為 KV cache 配置器：blocks、block tables，以及為什麼在 production 負載下 fragmentation 能維持在 4% 以下。
- 在迭代層級畫出 continuous batching：完成的 sequence 如何離開 batch、新的 sequence 如何不需清空就加入。
- 用一句話描述 chunked prefill，並說出它保護哪個 latency 指標（提示：是並行 decode 的 ITL 尾端，不是平均 throughput）。
- 在一次啟用所有最佳化之前，先對照你的版本的相容性矩陣，檢查 vLLM 功能組合。

## 問題 (The Problem)

單純的 PyTorch serve 迴圈一次處理一個 request：tokenize、prefill、decode 到 EOS、回傳。一個使用者時沒問題。一百個使用者時，就是一排耐心等待的人。顯而易見的修法是 static batching：把每個 request padding 到視窗內最長的 prompt，把每個 decode padding 到預期最長的輸出，並讓整個 batch 卡在最慢的 sequence 上。你為從沒用到的 padding 付費，快的 request 還得等慢的。

vLLM 一次解決三個問題。PagedAttention 避免 KV cache fragmentation 像傳統連續配置那樣吃掉 60-80% 的 GPU 記憶體。Continuous batching 讓 request 在每次 decode 迭代之間加入與離開 batch，所以 batch 永遠裝滿真正的工作。Chunked prefill 把 32k-token 的 prompt 切成約 ~512-token 的片段，與 decode 交錯執行，讓長 prompt 不會凍結 GPU 上所有的 decode tokens。

2026 年的 production 預設是三者全開。你需要了解每一個在做什麼，因為失敗模式全都出在 scheduler，而不是模型。

## 核心概念 (The Concept)

### 把 PagedAttention 看成虛擬記憶體系統

每個 sequence 的 KV cache 是 `num_layers × 2 × num_heads × head_dim × seq_len × bytes_per_element`。對 8192 tokens 的 Llama 3.3 70B，BF16 下每個 sequence 約 1.25 GB。如果你替每個 request 預先保留 8192 個位置，但平均 request 只用 1500 tokens，就浪費了約 82% 你保留的 HBM。傳統 batching 就要付這筆浪費。

PagedAttention 借用了 OS 虛擬記憶體的概念。KV cache 不是每個 sequence 一塊連續空間，而是以固定大小的 block 配置（預設 16 tokens）。每個 sequence 有一張 block table，把它的邏輯 token 位置對應到實體 block ID。當 sequence 長過已配置的 blocks，就再加一個 block。完成時，它的 blocks 歸還到 pool。

Fragmentation 從 60-80%（傳統）降到 4% 以下 (PagedAttention)。你不需要用旗標啟用 PagedAttention，它是 vLLM 唯一提供的配置器。可調的旋鈕是 `--gpu-memory-utilization`（預設 0.9），告訴 vLLM 在載入權重與 activations 之後，要保留多少 HBM 給 KV blocks。

### 迭代層級的 continuous batching

舊式的「dynamic batching」會等一個視窗（例如 10 ms）湊滿一個 batch，然後執行 prefill + decode + decode + decode，直到每個 sequence 都結束。快的 sequence 提早離開，在 GPU 完成慢的 sequence 時閒置。

Continuous batching 在每個 decode 步驟之間運作。把執行中的 sequence 集合稱為 `RUNNING` 清單。每次迭代：

1. `RUNNING` 中剛好碰到 EOS 或 max_tokens 的 sequence 被移除。
2. Scheduler 查看等待佇列。如果有空的 KV blocks，就接納新的 sequence（prefill 或恢復的）。
3. 對現在 `RUNNING` 中的所有內容執行 forward pass，每個 sequence 產生一個新 token。

Batch 大小永遠不會 padding 到固定數字。處於輸出中不同位置的 sequence 共用同一個融合的 forward。在 2026 年的 vLLM 中這稱為 `V1 scheduler`。關鍵不變量是：scheduler 每次 decode 迭代執行一次，而不是每個 request 一次。

### Chunked prefill 保護 ITL 尾端

Prefill 是 compute-bound。在 Llama 3.3 70B 上，32k-token 的 prompt 在單張 H100 上需要約 ~800 ms 的純 prefill。Prefill 執行時，batch 中其他所有 sequence 的 decode tokens 都在等待。在 serving 迴圈裡，一個長 prompt 的 first-token latency (TTFT)，會變成幾十位其他使用者的 inter-token latency (ITL) 突波。

Chunked prefill 把 prefill 切成固定大小的 chunk（預設 512 tokens），並把每個 chunk 當成一個單位排程。在 chunk 之間，scheduler 可以讓 decode sequence 前進一個 token。你付出一點點絕對 prefill latency（每個 chunk 幾 ms），換取低很多的 decode 時間抖動。依已發表的 benchmark，混合負載下的 P99 ITL 從約 ~50 ms 降到約 ~15 ms。

### 三個預設值彼此互動

三項功能彼此以對方為前提。PagedAttention 給 scheduler 一個可細粒度交易的 KV 資源。Continuous batching 需要這個細粒度資源，這樣接納新 sequence 才不會引發全域重新洗牌。Chunked prefill 是 scheduler 在同一個 `RUNNING` 清單上做的決定，它只是多一個 scheduler 政策，而不是獨立的系統。

你不需要知道每個旗標。你需要知道 scheduler 在優化什麼：在 KV-block 預算下的 goodput，並受 chunked prefill 切片所約束。

### 檢查相容性矩陣

在一次啟用全部功能之前，請針對你確切的 vLLM 版本，檢查每個功能組合的相容性矩陣，因為能組合哪些會隨版本變動。在 v0.18.0 中，功能矩陣把 speculative decoding 標示為與 chunked prefill 和 prefix caching 相容，而 speculative-decoding 頁面列出兩個已知的不相容：pipeline parallelism（到 v0.15.0 為止）與 draft-model speculation（到 v0.10.0 為止）。至於 draft 方法本身，2026 年的預設常是 EAGLE-3 (`"method": "eagle3"`)，在 Phase 17 · 05 介紹。

### 該記住的數字

- Llama 3.3 70B FP8、H100 SXM5、128 並行、三者全開：2,200-2,400 tok/s。
- 同一個模型，預設 vLLM（無 chunked prefill）：~1,800 tok/s。
- 同一個模型，單純 PyTorch forward 迴圈：~600 tok/s。
- Production 負載下 PagedAttention 的 KV fragmentation 浪費：<4%。
- 混合負載下的 P99 ITL：有 chunked prefill 約 ~15 ms，沒有則約 ~50 ms。

### Scheduler 長什麼樣子

```
while True:
    finished = [s for s in RUNNING if s.is_done()]
    for s in finished: release_blocks(s); RUNNING.remove(s)

    while WAITING and have_free_blocks_for(WAITING[0]):
        s = WAITING.pop(0)
        allocate_initial_blocks(s)
        RUNNING.append(s)

    # schedule prefill chunks + decode in one batch
    batch = []
    for s in RUNNING:
        if s.in_prefill:
            batch.append(next_prefill_chunk(s))   # e.g. 512 tokens
        else:
            batch.append(decode_one_token(s))     # 1 token

    run_forward(batch)                            # one fused GPU call
```

`code/main.py` 正是這個迴圈，用 stdlib Python 搭配假的 token 數量與假的 forward latency 實作。執行它可以看到 chunked prefill 如何在長 prefill 期間讓 decode sequence 持續存活。

```figure
tensor-parallel
```

## 實際使用 (Use It)

`code/main.py` 模擬一個可切換功能的 vLLM 風格 scheduler。執行它來觀察：

- `NAIVE` 模式：一次一個 request，不做 batching。
- `STATIC` 模式：padding 並等待，傳統 batching。
- `CONTINUOUS` 模式：迭代層級的接納與釋放。
- `CONTINUOUS + CHUNKED` 模式：prefill 切片與 decode 交錯。

輸出顯示總 throughput（每虛擬秒的 tokens）、TTFT 平均值與 P99 ITL。在混合流量下，`CONTINUOUS + CHUNKED` 那一列應該居冠。

## 交付 (Ship It)

這堂課產出 `outputs/skill-vllm-scheduler-reader.md`。給定 serving 設定（batch size、KV 記憶體使用率、chunked prefill 大小、speculative 設定），它會產出 scheduler 診斷，指出三個預設值中哪一個是瓶頸，以及該調整什麼。

## 練習 (Exercises)

1. 執行 `code/main.py`。在混合短、長 request 的工作負載上比較 `STATIC` 與 `CONTINUOUS`。throughput 差距來自哪裡：prefill 效率、decode 效率，還是尾端 latency？
2. 修改玩具 scheduler，加入 `--max-num-batched-tokens`。對執行 Llama 3.3 70B FP8 的 H100，合適的值是多少？（提示：它是 KV block 大小與空閒 block 數量的函數，不是原始 HBM。）
3. 重新閱讀 vLLM v0.18.0 的 release notes。哪些旗標組合互斥？列出來。
4. 對一份包含 1,000 個 request、平均 1,500 輸出 tokens、標準差 600 tokens 的 trace，計算 KV cache fragmentation 浪費：(a) 每個 request 以最大 8192 連續配置，(b) 使用 16-token blocks 的 PagedAttention。
5. 用一段話解釋為什麼 chunked prefill 能改善 P99 ITL，但單獨來看並不提升 throughput。實務上 throughput 的提升從哪裡來？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| PagedAttention | 「KV 的小技巧」 | KV cache 的固定大小 block 配置器；fragmentation <4% |
| Block table | 「page table」 | 每個 sequence 從邏輯 token 位置到實體 KV block 的對應表 |
| Continuous batching | 「做對的 dynamic batching」 | 每次 decode 迭代都做接納/釋放決定 |
| Chunked prefill | 「prefill 切分」 | 把長 prefill 切成 512-token 片段，與 decode 交錯 |
| TTFT | 「第一個 token 的時間」 | Prefill + 佇列 + 網路；長 prompt 時以 prefill 為主 |
| ITL | 「token 間 latency」 | 連續 decode tokens 之間的時間；主要由 batch size 決定 |
| Goodput | 「符合 SLO 的 throughput」 | 仍達成 TTFT 與 ITL 目標的 requests/sec（Phase 17 · 08 以 request 占比來回報） |
| V1 scheduler | 「新的 scheduler」 | vLLM 2026 年的 scheduler；以 chunked prefill 執行 continuous batching |
| `--gpu-memory-utilization` | 「記憶體旋鈕」 | 載入權重與 activations 後，保留給 KV blocks 的 HBM 比例 |

## 延伸閱讀 (Further Reading)

- [vLLM documentation — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode/)：chunked-prefill 與 speculative-decoding 相容性的官方來源。
- [vLLM Release Notes (NVIDIA)](https://docs.nvidia.com/deeplearning/frameworks/vllm-release-notes/index.html)：2026 年的發布節奏與各版本行為。
- [vLLM Blog — PagedAttention](https://blog.vllm.ai/2023/06/20/vllm.html)：原始文章，至今仍定義了該如何思考這個配置器。
- [PagedAttention paper (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)：fragmentation 分析與 scheduler 設計。
- [Aleksa Gordic — Inside vLLM](https://www.aleksagordic.com/blog/vllm)：附 flame graph 的 V1 scheduler 詳細導覽。
