# Kubernetes 上的 GPU Autoscaling：Karpenter、KAI Scheduler、Gang Scheduling

> 是三層，不是一層。Karpenter 動態佈建節點（不到一分鐘，比 Cluster Autoscaler 快 40%）。KAI Scheduler 負責 gang scheduling、topology awareness 與階層式佇列，避免「8 取 7」的部分配置陷阱：七個節點在等，卻因為缺一張 GPU 而空燒。應用層級的 autoscaler（NVIDIA Dynamo Planner、llm-d Workload Variant Autoscaler）依 inference 專屬訊號擴縮，例如佇列深度、KV cache 使用率，而不是 CPU/DCGM duty cycle。經典的 HPA 陷阱是：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle 量測，100% 可能是 10 個 request，也可能是 100 個。vLLM 會預先配置 KV cache 記憶體，所以記憶體永遠不會觸發 scale-down。這堂課教你組合這三層，並避開 Karpenter 預設的 `WhenEmptyOrUnderutilized` 政策，因為它會在 inference 進行到一半時終止正在執行的 GPU job。

**Type:** Learn
**Languages:** Python (stdlib, toy queue-depth autoscaler simulator)
**Prerequisites:** Phase 17 · 02 (Inference Platform Economics), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 畫出三個 autoscaling 層（節點佈建、gang scheduling、應用層級），並說出每層使用的工具。
- 解釋為什麼 `DCGM_FI_DEV_GPU_UTIL` 不適合當 vLLM 的 HPA 訊號，並說出兩個替代方案（佇列深度、KV cache 使用率）。
- 描述 gang scheduling，以及 KAI Scheduler 所避免的部分配置失敗模式（8 張 GPU 中 7 張閒置）。
- 說出會終止正在執行 GPU job 的 Karpenter consolidation 政策 (`WhenEmptyOrUnderutilized`)，並寫出 2026 年的安全替代方案。

## 問題 (The Problem)

你的團隊在 Kubernetes 上交付一個 LLM-serving 服務。你用 `DCGM_FI_DEV_GPU_UTIL` 當訊號設定 HPA。服務在上班時間固定頂在 100% 使用率，HPA 從不 scale up，因為它認為你已經滿載。你手動加一個 replica，TTFT 下降，但 HPA 還是不 scale。這個訊號在騙你。

另一件事：你用 Cluster Autoscaler 管節點。凌晨 2 點來了一個 1M-token 的 prompt，叢集花 3 分鐘佈建一個節點，request 就逾時了。

還有另一件事：你部署一個需要跨 2 個節點共 8 張 GPU 的 70B 模型。叢集有 7 張 GPU 閒置，分散在 3 個節點，還差 1 張。Cluster Autoscaler 為了缺的那 1 張 GPU 去佈建新節點。那 7 張已保留的 GPU 閒置 4 分鐘空燒錢，等 Kubernetes 把最後一張 GPU 準備好。

三層，三種不同的失敗模式。2026 年的 GPU-aware autoscaling 不是「把 HPA 打開」，而是組合節點佈建、gang scheduling 與應用訊號 autoscaling。

## 核心概念 (The Concept)

### 第 1 層：節點佈建 (Karpenter)

Karpenter 監看 pending 的 pod，並在約 ~45-60 秒內佈建節點（Cluster Autoscaler 對 GPU 節點通常要 90-120 秒）。它依 `NodePool` 的限制動態挑選 instance type。如果你的 pod 需要 8 張 H100，而叢集沒有符合的節點，Karpenter 會直接佈建一個，而不是擴充既有的群組。

**Consolidation 陷阱**：Karpenter 預設的 `consolidationPolicy: WhenEmptyOrUnderutilized` 對 GPU pool 很危險。它會為了把 pod 遷移到更便宜、大小更合適的 instance，而終止正在執行的 GPU 節點。對 inference 工作負載來說，這代表驅逐進行中的 request，並在新節點上重新載入 70B 模型。損失是數分鐘的容量加上 request 失敗。

GPU pool 的安全設定：

```yaml
disruption:
  consolidationPolicy: WhenEmpty
  consolidateAfter: 1h
```

這讓 Karpenter 在節點真正空了一小時後才 consolidate，但絕不驅逐正在執行的 job。

### 第 2 層：gang scheduling (KAI Scheduler)

KAI Scheduler（專案原名 "Karp"，後來改名）處理預設 kube-scheduler 做不到的事：

**Gang scheduling**：全有或全無的排程。一個需要 8 張 GPU 的分散式 inference pod，要嘛 8 張一起啟動，要嘛都不啟動。沒有這個，就會掉進部分配置陷阱：8 個 pod 啟動了 7 個，無限期等待，空燒錢。

**Topology awareness**：知道哪些 GPU 共用 NVLink、哪些在同一個機架、哪些之間有 InfiniBand，並據此放置 pod。DeepSeek-V3 67B 的 tensor-parallel 工作負載必須待在同一個 NVLink domain，KAI Scheduler 會遵守這點。

**階層式佇列**：多個團隊以優先順序與配額競爭同一個 GPU pool。只有在優先順序規則允許時，A 團隊的 production 尖峰才會被 B 團隊的 training job 搶佔。

KAI 以次要排程器的身分與 kube-scheduler 並行部署；你要在 workload 上加 annotation 才會使用它。Ray 與 vLLM production-stack 都有整合。

### 第 3 層：應用層級訊號

**HPA 陷阱**：`DCGM_FI_DEV_GPU_UTIL` 是 duty-cycle 指標，量的是 GPU 在每個取樣區間內有沒有在工作。100% 使用率可能代表 10 個並行 request，也可能是 100 個，GPU 都一樣忙。依 duty cycle 擴縮就是盲目擴縮。

更糟的是，vLLM 與類似的 engine 會預先配置 KV cache 記憶體（最高到 `--gpu-memory-utilization`）。即使只有一個 request，記憶體用量也維持在 90% 附近。以記憶體為基礎的 HPA 永遠不會 scale down。

**2026 年的替代訊號**：

- 佇列深度（等待 prefill 的 request 數量）。
- KV cache 使用率（有多少比例的 block 配置給進行中的 sequence）。
- 每個 replica 的 P99 TTFT（你的 SLA 訊號）。
- Goodput（每秒符合所有 SLO 的 request 數）。

NVIDIA Dynamo Planner 與 llm-d Workload Variant Autoscaler 會使用這些訊號來擴縮 replica。在 LLM serving 上，它們完全取代 HPA。

### 什麼時候用什麼

| 擴縮決策 | 工具 |
|----------------|------|
| 新增/移除節點 | Karpenter |
| 排程多 GPU job | KAI Scheduler |
| 新增/移除 replica | Dynamo Planner / llm-d WVA（或以佇列深度為訊號的自訂 HPA） |
| 選擇 GPU 類型 | Karpenter NodePool |
| 搶佔低優先順序 | KAI Scheduler 佇列 |

### Disaggregated prefill/decode 讓一切更複雜

如果你執行 disaggregated prefill/decode (Phase 17 · 17)，就會有兩類 pod，擴縮觸發條件不同：prefill pod 依佇列深度擴縮，decode pod 依 KV cache 壓力擴縮。llm-d 把它們暴露為各自獨立的 `Services`，並各有自己的 HPA。不要試著在兩者前面放同一個 HPA。

### Cold start 在這裡也很重要

Cold-start 緩解 (Phase 17 · 10) 是節點佈建時間會被使用者感受到的地方。Karpenter 的 45-60 秒暖機，加上 20GB 的模型載入，再加上 engine 初始化，代表從零開始的 request 要 2-5 分鐘。對 SLO 關鍵路徑請保留 warm pool (`min_workers=1`)，或在應用層使用 Modal 風格的 checkpointing。

### 該記住的數字

- Karpenter 節點佈建：約 ~45-60s，Cluster Autoscaler 約 ~90-120s（GPU 節點）。
- KAI Scheduler 避免部分配置浪費：8 取 7 陷阱。
- 以 `DCGM_FI_DEV_GPU_UTIL` 當 HPA 訊號：不可行；改用佇列深度或 KV 使用率。
- Karpenter `WhenEmptyOrUnderutilized`：會終止正在執行的 GPU job。Inference 請用 `WhenEmpty + consolidateAfter: 1h`。

```figure
autoscaling
```

## 實際使用 (Use It)

`code/main.py` 在突發性的 GPU 工作負載上模擬三層 autoscaler。比較單純 HPA (duty cycle)、佇列深度 HPA，以及 KAI gang-scheduled 擴縮。回報未滿足的 request、GPU 閒置分鐘數，以及綜合分數。

## 交付 (Ship It)

這堂課產出 `outputs/skill-gpu-autoscaler-plan.md`。給定叢集拓撲、工作負載型態與 SLO，它會設計出三層 autoscaling 計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。在突發工作負載下，單純 duty-cycle HPA 會丟掉多少個佇列深度 HPA 接得住的 request？差異從哪裡來？
2. 為在 H100 SXM5 上服務 Llama 3.3 70B FP8 的叢集設計 Karpenter NodePool。指定 `capacity-type`、`disruption.consolidationPolicy`、`consolidateAfter`，以及讓非 GPU 工作負載遠離這些節點的 taint。
3. 你的團隊回報 deployment 卡在 Pending，原因是「有 GPU 可用，但 pod 排不上去」。診斷一下：是 Karpenter、kube-scheduler，還是 KAI Scheduler？哪些 metrics 可以佐證？
4. 為 disaggregated 的 prefill pod 選一個 autoscale 訊號，再為 decode pod 選另一個不同的訊號。兩者都要說明理由。
5. 針對一個 24x7 的 production 服務，計算 `WhenEmptyOrUnderutilized` consolidation 陷阱的成本。該服務平均每天發生 60 次 request 掉落事件（P99 TTFT > 10s）。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Karpenter | 「節點佈建器」 | Kubernetes 節點 autoscaler；佈建時間不到一分鐘 |
| Cluster Autoscaler | 「舊的 scaler」 | Kubernetes 節點 autoscaler 的前代；較慢，以群組為單位 |
| KAI Scheduler | 「GPU 排程器」 | 負責 gang + topology + 佇列的次要排程器 |
| Gang scheduling | 「全有或全無」 | 原子性地排程 N 個 pod，否則全部延後 |
| Topology awareness | 「機架感知」 | 依 NVLink/IB/機架位置放置 pod |
| `DCGM_FI_DEV_GPU_UTIL` | 「GPU 使用率」 | Duty-cycle 指標；不是 LLM 的擴縮訊號 |
| Queue depth | 「等待中的 request」 | prefill-bound 擴縮的正確 HPA 訊號 |
| KV cache utilization | 「記憶體壓力」 | decode-bound 擴縮的正確 HPA 訊號 |
| Consolidation | 「Karpenter consolidation」 | 為了換成更便宜的 instance type 而終止節點 |
| `WhenEmpty + 1h` | 「安全的 consolidation」 | 不會驅逐正在執行 GPU job 的政策 |

## 延伸閱讀 (Further Reading)

- [KAI Scheduler GitHub](https://github.com/kai-scheduler/KAI-Scheduler)：設計文件與設定範例。
- [Karpenter Disruption Controls](https://karpenter.sh/docs/concepts/disruption/)：consolidation 政策語意與 GPU 安全預設值。
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)：Dynamo Planner 擴縮訊號。
- [Ray docs — KAI Scheduler for RayClusters](https://docs.ray.io/en/latest/cluster/kubernetes/k8s-ecosystem/kai-scheduler.html)：Ray 整合模式。
- [AWS EKS Compute and Autoscaling Best Practices](https://docs.aws.amazon.com/eks/latest/best-practices/aiml-compute.html)：託管 Kubernetes 專屬指引。
- [llm-d GitHub](https://github.com/llm-d/llm-d)：Workload Variant Autoscaler 設計。
