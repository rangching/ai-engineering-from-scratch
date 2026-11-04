# Pipeline Parallel 與 Bubble 分析 (Pipeline Parallel and Bubble Analysis)

> Tensor parallelism 把矩陣乘法分到各個 rank。Pipeline parallelism 把模型分到各個 rank，每個 rank 一個 stage。Microbatch 流過管線。開頭與結尾的空檔就是 bubble；把它降到最小就是整門功夫。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 19 Track C lessons 42-49
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 把序列式模型切成 N 個 stage，並在 N 個 rank 上模擬 forward 管線。
- 使用 GPipe 排程（只有 forward 的填充，然後 backward）讓 M 個 microbatch 通過管線，並計算 bubble 比例。
- 將 bubble 與 Megatron-LM 和 PipeDream 使用的交錯式 1F1B 排程比較。
- 為 stage 分配辯護：每個 stage 的運算量相等，比每個 stage 的參數數量相等更重要。

## 問題 (The Problem)

一個 fp16 的 70B 參數模型，光是參數就需要 140 GB。沒有任何消費級 GPU 放得下。ZeRO-3 把參數分片到各個 rank，但每次 forward 仍然需要每個 rank 對每一層 allgather 完整的層，每層要付 $\log(N)$ 個 hop。Pipeline parallel 走不同的路：把模型切成 N 個 stage，每個 rank 放一個 stage。第 1 層的 forward 在 rank 0 完成後，把 activation tensor 交給 rank 1；rank 1 執行第 2 層後交給 rank 2；依此類推。Backward 以相反方向流動。記憶體線性下降，因為每個 rank 只持有一個 stage；運算是循序的，這就是 bubble 問題。

Bubble 是管線開頭（等待第一個 microbatch 抵達最後一個 stage）和結尾（等待最後一個 microbatch 流回去）的閒置時間。有 M 個 microbatch 和 N 個 stage 時，每個 stage 的 bubble 比例是 (N-1)/(M+N-1)。在 M=8、N=4 時是 27%。在 M=64、N=4 時是 4.5%。當每一步有很多 microbatch 時 bubble 會縮小，這意味著每個 microbatch 的 batch size 很小，這正是驅動 microbatch 設計的限制。

## 核心概念 (The Concept)

```mermaid
flowchart LR
  R0[rank 0: stage 0 / layer 0] --> R1[rank 1: stage 1 / layer 1]
  R1 --> R2[rank 2: stage 2 / layer 2]
  R2 --> R3[rank 3: stage 3 / loss]
  R3 -.backward.-> R2
  R2 -.backward.-> R1
  R1 -.backward.-> R0
```

### GPipe 排程 (GPipe schedule)

在開始任何 backward 之前，先用全部 M 個 microbatch 填滿管線的 forward；然後反向排空 backward。每個 microbatch 的 activation 都必須保留到它的 backward，所以記憶體隨 M 線性成長。Forward 花 M+N-1 個週期，backward 再花 M+N-1 個週期。每個 stage 的有用工作是 2M 個週期；每個 stage 的 bubble 是 2(N-1) 個週期。當每次 forward 和 backward 都花一個單位時間，bubble 比例是 (N-1)/(M+N-1)。選擇遠大於 N 的 M，可以隱藏 bubble。

### 1F1B 排程 (1F1B schedule)

交錯進行：一個 microbatch 的 forward 一到達最後一個 stage，就開始它的 backward，讓它往回串流。排程在每個 stage 交替進行一次 forward 與一次 backward。Bubble 仍然是 N-1，但 activation 記憶體受管線深度限制，而不是 microbatch 數量。正式環境的管線使用 1F1B（Megatron、PipeDream）。本課先實作 GPipe，因為比較簡單，1F1B 則留作練習。

### 為什麼每個 stage 的運算量相等很重要 (Why equal compute per stage matters)

如果 stage 0 花 50 ms，stage 1 花 100 ms，每個週期都被 stage 1 卡住。其他 stage 每個週期閒置 50 ms，等 stage 1 釋放。參數數量相等是錯誤的軸：transformer 的運算量主要由每層的 attention 加 MLP 主導，而 embedding 層有很多參數卻幾乎沒有運算。Stage 的分配應該讓每個 stage 的 FLOPs 相等，而不是每個 stage 的權重相等。

### Microbatch 對 batch (Microbatch versus batch)

管線執行 M 個大小各為 B 的 microbatch。有效 batch size 是 M*B。管線一步結束時的 gradient，是合併後 M*B 個樣本上的 gradient。Bubble 比例取決於 M；optimizer 看到的是 M*B。調整 M 就是在 bubble（M 高時較低）與每個 microbatch 的記憶體（GPipe 在 M 高時 activation 記憶體較高）之間取捨。

```figure
cd-pipeline-bubble
```

## 動手做 (Build It)

`code/main.py` 實作了：

- `PipelineStage`：一個小的 `nn.Module`，持有一個 stage 的參數並暴露 `forward(activation)`。
- `Pipeline(stages, num_microbatches)`：在模擬的 stage 上，使用每個 stage 模擬的牆上時間，協調 GPipe 排程。
- `bubble_fraction(num_stages, num_microbatches)`：封閉形式的 (N-1)/(M+N-1)。
- 一個 4 stage 的示範，印出每個 microbatch 的 trace 與量測到的 bubble 比例。

執行它：

```bash
python3 code/main.py
```

輸出：一張 stage 乘 microbatch 的甘特圖，以及與封閉形式預測比較的 bubble 百分比。

## 實務上的正式環境模式 (Production patterns in the wild)

有三個模式把 pipeline parallel 強化到足以發佈。

**Activation checkpointing 與 pipeline 搭配。** 在 GPipe 上有 M 個 microbatch 進行中時，activation 記憶體是一個 microbatch 的 M 倍。Activation checkpointing 在 backward 時重新計算 forward，以運算換記憶體；這個組合讓 pipeline 對長序列變得可行。

**Stage 的平衡是量測出來的，不是假設的。** 正式環境的團隊會跑一輪 profiling，量測目標硬體上實際的每層運算（FLOPs 與牆上時間），再依該量測值做分割。Megatron-LM 的 `--num-layers-per-stage` 旗標接受一個清單，以便在各 stage 的每層成本不同時，允許不均勻的層數。

**Send-recv 排程必須避免死鎖。** 每個 stage 都先 send 再 receive 的管線，會在線路上死鎖。標準的修正方式是交錯：偶數 rank 的 stage 先 send 再 recv，奇數 rank 的 stage 先 recv 再 send。本課明確排定各個 rank，讓這個模式清楚可見。

## 實際使用 (Use It)

正式環境的做法：

- **Megatron-LM。** 大規模 pipeline parallel 的參考。使用 1F1B，並支援 tensor + pipeline + data parallel 合併使用。
- **DeepSpeed Pipeline。** 與 ZeRO 整合；ZeRO-1 + pipeline 是最大型開放模型的常見組合。
- **PyTorch Pipe。** PyTorch 原生的 pipeline wrapper，建立在 `torch.distributed.pipeline.sync.Pipe` 之上。

## 交付 (Ship It)

第 80 課把每個 stage 的參數分片存進分片 checkpoint。第 81 課在端到端示範中組合 DDP + ZeRO + pipeline（精神上如此；示範為了執行時間，讓 pipeline 保持模擬）。

## 練習 (Exercises)

1. 實作 1F1B，並驗證 bubble 比例與 GPipe 相符，但 activation 記憶體有上限。
2. 在更深的模型上剖析實際的每個 stage 時間，並依量測到的牆上時間重新平衡 stage。
3. 在 pipeline microbatch 之間加入 gradient 累積，並檢查 gradient 等於等價的完整 batch forward 的 gradient。
4. 將 pipeline 與 activation checkpointing 搭配，並量測記憶體下降相對於運算成本。
5. 把 pipeline 與 DDP 結合（每個 pipeline rank 在 data-parallel 群組中被複製），並推導 2D 排程。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Pipeline | 「沿深度做 model parallel」 | 每個 rank 一個 stage，activation 從一個 stage 流到下一個 |
| Bubble | 「管線閒置時間」 | 開頭與結尾有 (N-1) 步，其中某些 stage 沒有工作 |
| Microbatch | 「batch 的切片」 | 一個 forward/backward 單位；M 變大時 bubble 縮小 |
| GPipe | 「先填滿再排空」 | 在任何 backward 之前先做全部 M 個 forward；activation 記憶體高 |
| 1F1B | 「交錯排程」 | 每個 stage 一次 forward 一次 backward；activation 記憶體有上限 |

## 延伸閱讀 (Further Reading)

- [Huang et al, GPipe: Efficient Training of Giant Neural Networks](https://arxiv.org/abs/1811.06965)
- [Narayanan et al, PipeDream: Generalized Pipeline Parallelism for DNN Training](https://arxiv.org/abs/1806.03377)
- [Megatron-LM pipeline parallel docs](https://github.com/NVIDIA/Megatron-LM)
- Phase 19 Lesson 76 - 排程所使用的 send/recv 基本元件
- Phase 19 Lesson 78 - ZeRO 與 pipeline 正交，常被組合使用
