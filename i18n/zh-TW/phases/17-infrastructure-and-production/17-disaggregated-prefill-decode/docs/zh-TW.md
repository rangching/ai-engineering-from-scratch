# Disaggregated Prefill/Decode：NVIDIA Dynamo 與 llm-d (Disaggregated Prefill/Decode)

> Prefill 是 compute-bound；decode 是 memory-bound。把兩者放在同一張 GPU 上會浪費其中一種資源。Disaggregation 把它們拆到不同的 pool，並透過 NIXL（RDMA/InfiniBand，或 TCP fallback）在兩者之間傳輸 KV cache。NVIDIA Dynamo（GTC 2025 發表，1.0 GA）位於 vLLM/SGLang/TRT-LLM 之上，它的 Planner Profiler + SLA Planner 會自動匹配 prefill:decode 比例以達成 SLO。NVIDIA 公布的 throughput 提升約在這個範圍：developer.nvidia.com (2025-06) 顯示在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 在中等 latency 區間約有 ~6x 的提升；Dynamo 產品頁面 (developer.nvidia.com，未註明日期) 則宣稱在 GB300 NVL72 + Dynamo 上，MoE throughput 相較 Hopper 最高達 50x。「30x」這個數字是社群對完整 Blackwell + Dynamo + DeepSeek-R1 報告的彙總；我們沒有找到明確寫 30x 的單一原始來源，所以請把它視為方向性的說法。llm-d (Red Hat + AWS) 是 Kubernetes 原生：prefill / decode / router 是獨立的 Services，各有每個角色的 HPA。llm-d 0.5 新增階層式 KV offloading、cache-aware LoRA 路由、UCCL 網路、scale-to-zero。經濟面：彙整多位客戶揭露的內部資料顯示，在固定 SLA 下，從共置 serving 改成搭配 Dynamo 的 disaggregated，在 $2M 等級的 inference 支出上可省下 30–40%（即 $600-800K/年）；具體的 $2M→$600-800K 數字是內部綜合值，不是單一公開案例研究，請把它當成數量級的錨點，而不是參考引用。短 prompt（<512 tokens、輸出短）不值得付出傳輸成本。

**Type:** Learn
**Languages:** Python (stdlib, toy disaggregated-vs-colocated simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 08 (Inference Metrics)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼 prefill 與 decode 有不同的最佳 GPU 配置，並量化共置下的浪費。
- 畫出 disaggregated 架構：prefill pool、decode pool、透過 NIXL 的 KV 傳輸、router。
- 說出 disaggregation 不划算的條件（短 prompt、短輸出）。
- 區分 NVIDIA Dynamo（stack-above）與 llm-d（Kubernetes 原生），並把各自對應到營運情境。

## 問題 (The Problem)

你在 8 張 H100 上執行 Llama 3.3 70B。在混合工作負載（長 prompt + 短輸出）下，GPU 在 decode 期間閒置，因為大部分運算都花在 prefill。在不同的工作負載（短 prompt + 長輸出）下，則剛好相反。共置的 prefill + decode 意味著你兩邊都過度配置。

預算影響：20-40% 的 GPU 時間浪費在錯誤的資源上。你買 H100 的運算能力去跑 memory-bound 的 decode，或是買 H100 的 HBM 頻寬去跑 compute-bound 的 prefill。兩者都是昂貴的浪費。

Disaggregation 把 prefill 與 decode 拆到各自依瓶頸調整大小的 pool。KV cache 透過高頻寬互連從 prefill pool 傳到 decode pool。

## 核心概念 (The Concept)

### 為什麼瓶頸不同

**Prefill**：在一次 forward 中對完整的輸入 prompt 執行 transformer。矩陣乘法占主導；compute-bound。H100 FP8 可提供約 ~2000 TFLOPS 的有效 throughput。Batch 效率很好，一次 forward 處理很多 tokens。

**Decode**：一次產生一個 token，每次迭代都讀取全部權重。Memory-bandwidth-bound。HBM3 提供約 ~3 TB/s。Batch 效率只有在高並行時才好，因為權重讀取會被 batch 攤銷。

共置它們時：你買的 GPU 要同時為兩者優化。H100 兩者都擅長，但無論哪種價格都一樣。在規模化時，你會希望 prefill pool 用 H100 / 重運算；decode pool 用 H200 / 重記憶體，或搭配激進的 quantization。

### 架構

```
            ┌──────────────┐
  Request → │    Router    │ ───────────────────────┐
            └──────┬───────┘                        │
                   │                                │
                   ▼ (prompt only)                  │
            ┌──────────────┐    KV cache    ┌───────▼──────┐
            │ Prefill pool │ ─── NIXL ────► │ Decode pool  │
            │  (compute)   │                │  (memory)    │
            └──────────────┘                └──────┬───────┘
                                                   │ tokens
                                                   ▼
                                                 Client
```

NIXL 是 NVIDIA 的節點間傳輸。有 RDMA/InfiniBand 時就使用，否則退回 TCP。傳輸 latency 是真實存在的，對 70B FP8 上 4K-token prompt 的 KV cache，通常為 20-80 ms。這就是短 prompt 不值得做 disaggregation 的原因：傳輸稅超過節省。

### Dynamo vs llm-d

**NVIDIA Dynamo**（GTC 2025 發表，1.0 GA）：
- 以 orchestrator 的身分位於 vLLM、SGLang、TRT-LLM 之上。
- Planner Profiler 量測工作負載，SLA Planner 自動設定 prefill:decode 比例。
- Rust 核心，Python 可擴充。
- Throughput 提升：NVIDIA 回報在 GB200 NVL72 + Dynamo 上，DeepSeek-R1 MoE 於中等 latency 區間達 6x (developer.nvidia.com, 2025-06)；社群在完整 Blackwell + Dynamo + DeepSeek-R1 stack 上「最高 30x」的說法缺乏單一原始來源，應視為方向性。
- GB300 NVL72 + Dynamo：依 Dynamo 產品頁面 (developer.nvidia.com，未註明日期)，MoE throughput 相較 Hopper 最高 50x。

**llm-d**（Red Hat + AWS，Kubernetes 原生）：
- Prefill / decode / router 是獨立的 Kubernetes Services。
- 每個角色的 HPA，使用佇列深度 (prefill) / KV 使用率 (decode) 訊號。
- `topologyConstraint packDomain: rack` 把 prefill+decode 的 clique 放在同一個機架，以取得高頻寬的 KV 傳輸。
- llm-d 0.5 (2026)：階層式 KV offloading、cache-aware LoRA 路由、UCCL 網路、scale-to-zero。

如果你想要託管的 stack-above orchestrator，就用 Dynamo。如果你想要 Kubernetes 原生的基本元件，且全力投入 CNCF 生態系，就用 llm-d。

### 經濟面

內部綜合值（不是單一公開案例研究，僅作數量級錨點）：

- 共置 serving 的 inference 年支出 $2M。
- 改成搭配 Dynamo 的 disaggregated。
- 相同的 request 量、相同的 P99 latency SLA。
- 回報節省：$600K–$800K/年（降低 30–40%）。
- 沒有新增硬體。

這個數字是我們從多位客戶的揭露中綜合而成，不是單一可引用的案例研究；最接近的公開資料點是 Baseten 以 Dynamo KV 路由取得的 2x 更快 TTFT / 61% 更高 throughput (baseten.co, 2025-10)，以及 VAST + CoreWeave 預測在 40–60% KV 命中率下每塊錢可多 60–130% 的 tokens (vastdata.com, 2025-12)。節省來自替每個 pool 調整合適的大小；重 prefill 的工作負載（8K+ 前綴的 RAG）受益比均衡型更多。

### 什麼時候不要做 disaggregation

- Prompt < 512 tokens 且輸出 < 200 tokens：傳輸稅超過收益。
- 小叢集 (< 4 張 GPU)：pool 的多樣性不夠。
- 團隊無法營運兩個各自擴縮的 GPU pool：Dynamo 有幫助，但並不簡單。
- 沒有 RDMA fabric：TCP 傳輸稅更重。

### Router 與 Phase 17 · 11 整合

Disaggregated 的 router 是 KV-cache-aware 的 (Phase 17 · 11)。Request 會落在持有其前綴的 decode pool，如果沒有相符，就走 prefill → decode。命中率與 disaggregation 會疊加，cache-aware router 決定了是否真的需要一次新的 prefill。

### Blackwell 上的 MoE 才是真正數字所在

GB300 NVL72 + Dynamo 相較 Hopper 基準，MoE throughput 達 50x。MoE 的 expert 路由在 prefill 時重運算、在 decode 時重記憶體（expert cache），所以 disaggregation 是雙重收益。2026 年前沿模型的 serving 以 MoE 為主（DeepSeek-V3、未來的 GPT-5 變體）。

### 該記住的數字

Benchmark 數字會漂移，NVIDIA 與 inference stack 每季都會發布更新的結果。引用前請重新確認。

- GB200 NVL72 + Dynamo 上的 DeepSeek-R1：中等 latency 區間相較基準約 ~6x throughput (developer.nvidia.com, 2025-06)；社群在完整 Blackwell + Dynamo stack 上「最高 30x」的說法是缺乏單一原始來源的方向性彙總。
- GB300 NVL72 + Dynamo：相較 Hopper，MoE throughput 最高 50x (developer.nvidia.com，未註明日期)。
- 節省錨點（內部綜合值，不是單一案例研究）：在固定 SLA 下，$2M 年支出可省 $600-800K/年。
- Disaggregation 門檻：prompt >512 tokens + 輸出 >200 tokens。
- 透過 NIXL 的 KV 傳輸：70B FP8 上 4K-prompt 的 KV 需 20-80 ms。

```figure
prefill-decode-split
```

## 實際使用 (Use It)

`code/main.py` 模擬共置與 disaggregated 的 serving。回報 throughput、每個 request 的成本，以及 prompt 長度的交叉點。

## 交付 (Ship It)

這堂課產出 `outputs/skill-disaggregation-decider.md`。給定工作負載與叢集，判斷是否要做 disaggregation。

## 練習 (Exercises)

1. 執行 `code/main.py`。Prompt 長度多長時，disaggregation 會勝過共置？
2. 為 P99 前綴長度 8K、輸出 300 的 RAG 服務設計 prefill pool 和 decode pool。
3. Dynamo vs llm-d：對一個沒有 Python runtime 偏好的純 Kubernetes 團隊，挑一個。
4. 計算 KV 傳輸成本：70B FP8 上 4K prefill = 約 ~500 MB KV。RDMA 100 GB/s 時傳輸 = 5 ms。TCP 10 GB/s 時 = 50 ms。哪一個對你的 SLA 有影響？
5. MoE expert 路由會改變 KV 存取模式。當 MoE 每個 token 啟用不同的 experts 時，disaggregation 的表現如何？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Disaggregated serving | 「拆分 prefill/decode」 | 每個階段使用獨立的 GPU pool |
| NIXL | 「NVIDIA 傳輸」 | Dynamo 的節點間 KV 傳輸 (RDMA/TCP) |
| NVIDIA Dynamo | 「orchestrator」 | vLLM/SGLang/TRT-LLM 之上的 stack-above 協調器 |
| llm-d | 「Kubernetes 原生」 | Red Hat + AWS 的 K8s disaggregated stack |
| Planner Profiler | 「Dynamo 自動設定」 | 量測工作負載，設定 pool 比例 |
| SLA Planner | 「Dynamo 政策」 | 自動匹配 prefill:decode 比例以達成 SLO |
| `packDomain: rack` | 「llm-d 拓撲」 | 把 prefill+decode 放在同一機架以加速 KV |
| UCCL | 「unified collective」 | llm-d 0.5 為 scale-to-zero 提供的網路層 |
| MoE expert routing | 「每個 token 一個 expert」 | DeepSeek-V3 模式；disaggregation 有幫助 |

## 延伸閱讀 (Further Reading)

- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)
- [NVIDIA — Disaggregated LLM Inference on Kubernetes](https://developer.nvidia.com/blog/deploying-disaggregated-llm-inference-workloads-on-kubernetes/)
- [TensorRT-LLM Disaggregated Serving blog](https://nvidia.github.io/TensorRT-LLM/blogs/tech_blog/blog5_Disaggregated_Serving_in_TensorRT-LLM.html)
- [llm-d GitHub](https://github.com/llm-d/llm-d)
- [llm-d 0.5 release notes](https://github.com/llm-d/llm-d/releases)
