# Serverless LLM 的 Cold Start 緩解 (Cold Start Mitigation for Serverless LLMs)

> 大型模型從冷啟動到能服務需要好幾分鐘：全新節點上的 70B 模型，在第一個 token 之前大約需要 3-8 分鐘。在真正的 serverless 世界裡，這不是暖機，而是停機。緩解措施分布在五個層次：預先植入的節點映像（AWS 上的 Bottlerocket，雙 volume 架構）、模型 streaming（NVIDIA Run:ai Model Streamer，vLLM 原生支援）、GPU 記憶體 snapshot（Modal checkpoints，重啟最多快 10x）、warm pool (`min_workers=1`)、分層載入（ServerlessLLM 的 NVMe→DRAM→HBM pipeline，latency 降低 10-200x），另外還有第六個額外模式：live migration，搬移的是輸入 tokens (KB) 而不是 KV cache (GB)。Modal 公布的 cold start 底線是 2-4s；Baseten 預設 5-10s，預先暖機後不到一秒。這堂課教你量測、編列預算，並疊加這五個層次。

**Type:** Learn
**Languages:** Python (stdlib, toy cold-start path simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 列舉 cold-start 緩解的五個層次，並說出每層的一項工具或模式。
- 把 70B 模型的總 cold-start 時間算成（節點佈建）+（權重下載）+（權重載入 HBM）+（engine 初始化）的總和。
- 解釋為什麼 live migration 傳輸的是輸入 tokens (KB) 而不是 KV cache (GB)，以及代價是什麼（重新計算）。
- 說出 warm-pool 的取捨（為閒置 GPU 付費，或接受 cold-start 尾端），以及 `min_workers > 0` 變成必要條件的 SLA 門檻。

## 問題 (The Problem)

你的 serverless LLM endpoint 在夜間縮到零。早上 8 點流量暴增。第一個 request 要等待：

1. Karpenter 佈建 GPU 節點：45-60s。
2. 容器拉取含權重的 30 GB 映像：120-300s。
3. Engine 把權重載入 HBM：依模型大小與儲存速度為 45-120s。
4. vLLM 或 TRT-LLM 初始化 CUDA graphs、KV cache pool、tokenizer：10-30s。

總計：220-510s（大約 3-8 分鐘）才會回傳一個 token。你的 SLA 是 2s。你部署 warm-pool (`min_workers=1`)，問題似乎消失了，但現在你得 24x7 為一張閒置 GPU 付費。如果你的服務有 5 個產品，各有一個 warm replica，那就是每月 5 × 24 × 30 = 3,600 GPU 小時，不管有沒有任何使用者呼叫過。

Cold-start 緩解，就是在保有 serverless 經濟性的同時，逼近 always-on 的 latency。

## 核心概念 (The Concept)

### 第 1 層：預先植入的節點映像 (Bottlerocket)

在 AWS 上，Bottlerocket 的雙 volume 架構把 OS 與資料分開。把已預先拉好容器映像的資料 volume 做 snapshot，並在你的 `EC2NodeClass` 中引用 snapshot ID。新節點開機時，權重已經在本機 NVMe 上，步驟 2 和部分步驟 3 就消失了。與 Karpenter 原生搭配。典型節省：大型模型每次 cold start 省 2-4 分鐘。

GCP 上的對應做法：預先烘焙容器層的自訂 VM 映像。Azure 上：用同樣模式的 managed disk snapshot。

### 第 2 層：模型 streaming (Run:ai Model Streamer)

不要在回應第一個 request 之前就載入完整檔案，而是逐層把權重 streaming 進 GPU 記憶體，第一個 transformer block 一就位就開始處理。NVIDIA Run:ai Model Streamer 在 2026 年的 vLLM 中原生提供。支援 S3、GCS 與本機 NVMe。藉由讓 I/O 與運算設定重疊，大型模型的權重載入時間大約可減半。

### 第 3 層：GPU 記憶體 snapshot (Modal)

Modal 在第一次載入後，對 GPU 狀態（權重、CUDA graphs、KV cache 區域）做 checkpoint。後續重啟直接反序列化進 HBM，比重新初始化快 10x。這是最接近「2 秒內開機一張暖 GPU」的做法。取捨：snapshot 是依 GPU 拓撲而定，所以如果 Karpenter 把你遷移到不同 SKU，就得重新 checkpoint。

### 第 4 層：warm pools (min_workers=1)

最簡單的緩解：永遠保持一個 replica 就緒。成本是一張 GPU 24x7 的每小時費率。這個算術對小模型很殘酷（你付 $0.85-$1.50/hr 只為避開 30s 的 cold start），對大模型則友善（付 $4/hr 避開 5 分鐘的 cold start）。warm pool 變成必要的 SLA 門檻：通常是 70B+ 模型的 TTFT P99 < 60s。

### 第 5 層：分層載入 (ServerlessLLM)

ServerlessLLM 把儲存視為階層：NVMe（快但大）、DRAM（中等但分層）、HBM（極小但即時）。權重預先載入到 DRAM；依需求載入到 HBM。論文回報相較單純的 disk-to-HBM，cold load 的 latency 降低 10-200x。Production 採用還在早期，但已有與 vLLM 的整合。

### 第 6 層：live migration（額外模式）

當節點變得無法使用（spot 驅逐、節點 drain），傳統模式是 cold-start 另一個 replica 並排空 request 佇列。Live migration 則把輸入 tokens（幾 KB）搬到已載入模型的目的地，並在目的地重新計算 KV cache。重新計算比透過網路傳輸數 GB 的 KV cache 便宜。適用於 disaggregated 部署。

### warm-pool 的算術

對 P99 TTFT SLA 為 2s 的服務，問題不是「要不要 warm pool」，而是「要幾個 warm replica，以及哪些路徑可以有」。

- 高價值的互動路徑（即時聊天、語音 agent）：`min_workers=1-2`。
- 背景 batch 路徑（夜間分類）：接受縮到零，可容忍 5-10 分鐘的 cold start。
- Premium 方案：每個 tenant 有專屬容量的 `min_workers`。

### 先量測再優化

全新節點上 70B 模型的 cold-start 剖析（示意）：

| 階段 | 時間 | 緩解方式 |
|-------|------|-----------|
| 節點佈建 | 50s | Bottlerocket + 預先植入映像、warm pool |
| 映像拉取 | 180s | 預先植入的資料 volume（消除） |
| 權重載入 HBM | 75s | Model streamer（減半）；GPU snapshot（消除） |
| Engine 初始化 | 20s | 持久化的 CUDA graph cache |
| 第一次 forward | 3s | 固有的最小 latency |
| **冷啟動總計** | **328s** | |
| **套用緩解後總計** | **~15s** | 減少 22x |

### 該記住的數字

- Modal cold start：2-4s（搭配 GPU snapshots）。
- Baseten 預設 cold start：5-10s；預先暖機後不到一秒。
- 原始 70B cold start：3-8 分鐘。
- Run:ai Model Streamer：權重載入約 ~2x 加速。
- ServerlessLLM 分層載入：latency 降低 10-200x（論文數字）。

```figure
cold-start-pipeline
```

## 實際使用 (Use It)

`code/main.py` 模擬有無各項緩解措施的 cold-start 路徑。回報總 cold-start 時間、warm-pool 成本，以及 warm pool 開始回本的損益平衡 request 速率。

## 交付 (Ship It)

這堂課產出 `outputs/skill-cold-start-planner.md`。給定 SLA、模型大小與流量型態，挑選要疊加哪些緩解措施。

## 練習 (Exercises)

1. 執行 `code/main.py`。計算損益平衡的 request 速率：超過它之後，warm replica 比起透過 SLO 下額外掉落的 request 來付 cold-start 稅更便宜。
2. 你部署一個 13B 模型，P99 TTFT SLA 為 3s。挑出能達標的最小緩解組合（層數最少）。
3. Bottlerocket 預先植入消除了映像拉取，但權重仍需從 snapshot 載入 HBM。假設 snapshot 所在的 NVMe 讀取速度為 7 GB/s，計算 70B 模型的實際經過時間。
4. 你的 serverless 供應商提供 GPU snapshots（Modal），而你的團隊以「snapshot 會洩漏 PII」為由拒絕。請正反兩面都辯論：實際風險是什麼，緩解方式是什麼（暫時性 snapshot、加密、namespace 隔離）？
5. 設計分層的 warm-pool 政策：付費使用者、試用使用者與 batch 工作負載各要幾個 warm replica？列出計算過程。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Cold start | 「那段大停頓」 | 全新 replica 上從 request 到第一個 token 的時間 |
| Warm pool | 「常駐最小量」 | 以 `min_workers >= 1` 保持至少一個 replica 就緒 |
| Pre-seeded image | 「烘焙好的 AMI」 | 容器權重已常駐的節點映像 |
| Bottlerocket | 「AWS 節點 OS」 | 支援雙 volume snapshot 的 AWS 容器最佳化 OS |
| Model streamer | 「串流載入」 | 讓權重 I/O 與運算設定重疊 |
| GPU snapshot | 「HBM checkpoint」 | 序列化載入後的 GPU 狀態；重啟時反序列化 |
| Tiered loading | 「NVMe + DRAM + HBM」 | 儲存層級階層；依需求載入 |
| Live migration | 「搬 tokens」 | 傳輸輸入 (KB)，在目的地重新計算 KV |
| `min_workers` | 「warm replicas」 | Serverless 的最小保活數量 |
| Scale-to-zero | 「完整 serverless」 | 閒置時零成本；接受完整的 cold-start 稅 |

## 延伸閱讀 (Further Reading)

- [Modal — Cold start performance](https://modal.com/docs/guide/cold-start)：Modal 公布的 benchmark 與 checkpoint 架構。
- [AWS Bottlerocket](https://github.com/bottlerocket-os/bottlerocket)：預先植入資料 volume snapshot 的模式。
- [NVIDIA Run:ai Model Streamer](https://github.com/run-ai/runai-model-streamer)：讓權重載入與運算設定重疊。
- [Baseten — Cold starts](https://docs.baseten.co/deployment/autoscaling/cold-starts)：預先暖機的作法手冊。
- [ServerlessLLM paper (USENIX OSDI'24)](https://www.usenix.org/conference/osdi24/presentation/fu)：分層載入的設計。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)：disaggregated 部署的 live migration。
