# 多區域 LLM Serving 與 KV Cache 局部性 (Multi-Region LLM Serving and KV Cache Locality)

> Round-robin 負載平衡對有快取的 LLM inference 反而有害。沒有落在持有其前綴節點上的 request，要付出完整的 prefill 成本：長 prompt 在 P50 約 800 ms，命中 cache 則約 ~80 ms。2026 年的 production 模式是 cache-aware router（Rust 寫的 vLLM Router、llm-d router），它消費 KV-cache 事件，並依前綴 hash 相符來路由。近期研究 (GORGO) 把跨區域網路 latency 明確納入路由目標。商業的「cross-region inference」產品（Bedrock cross-region inference、GKE multi-cluster gateways）把 inference 當成不透明的東西，處理的是可用性，而不是 TTFT。JPMorgan 與 Mayo Clinic 在 2024 年 11 月演練 us-east-1 failover，約 22 分鐘。DR 的現實：32% 的 LLM DR 失敗，是因為團隊備份了權重，卻忘了 tokenizer 檔案或 quantization 設定。

**Type:** Learn
**Languages:** Python (stdlib, toy prefix-cache-aware router simulator)
**Prerequisites:** Phase 17 · 04 (vLLM Serving), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼 round-robin 負載平衡會破壞有快取的 inference，並量化 TTFT 的損失。
- 畫出 cache-aware router：輸入（KV-cache 事件）、演算法（前綴 hash 相符）、決勝條件（GPU 使用率）。
- 說出造成 LLM 32% DR 失敗的原因（缺少 tokenizer 檔案 / quantization 設定），並寫出三檔案的 DR 檢查清單。
- 區分商業的跨區域產品（Bedrock CRI、GKE Multi-Cluster Gateway）與 KV-aware 路由。

## 問題 (The Problem)

你的服務跑在 us-east-1、us-west-2 與 eu-west-1。你在前面放了採用 round-robin 的 ALB。Production 中的前綴 cache hit rate 掉到 8%。TTFT P50 變成三倍。你的 vLLM log 顯示每個 request 都在付完整的 prefill 成本。

Round-robin 對無狀態服務是最佳的。LLM inference 在設計上是有狀態的，KV cache 編碼了模型所見的一切。盲目路由，就是路由到錯誤的 cache。

另外，你的團隊有 DR 計畫。你把模型權重跨區域備份到 S3。區域性停機發生；你嘗試 failover；replica 拒絕啟動。你忘了 tokenizer.json、quantization 設定，以及 RoPE scaling 設定，它們在另一個你沒有同步的 bucket 裡。

多區域 LLM serving 是 cache 問題、路由問題，也是 DR 衛生問題，而不是 load-balancer 問題。

## 核心概念 (The Concept)

### Cache-aware 路由

Request 帶著 prompt 抵達。Router 對前綴做 hash（例如前 512 個 tokens）；詢問每個 replica「你有快取這個前綴嗎？」。Replica 在配置與驅逐 block 時，把 KV-cache 事件發布到 pub/sub 頻道。Router 挑出相符的 replica，若沒有任何相符，就退回以 GPU 使用率為基礎的決勝條件。

**vLLM Router**（Rust，2026 production-stack）：訂閱 `kv.cache.block_added` 事件，維護前綴 hash → replica 的索引，以 $O(1)$ 查找來路由。沒有相符時退回最小佇列深度。

**llm-d router**：相同模式，Kubernetes 原生。透過 ControlPlane API 發布事件。

**SGLang RadixAttention** (Phase 17 · 06) 是 replica 內部的對應做法。跨 replica 的路由嚴格來說在它的上游。

### 數字

2K-token prompt、Llama 3.3 70B FP8、H100 的 TTFT P50：
- Cache hit（同一個 replica，前綴常駐）：~80 ms。
- Cache miss（冷 prefill）：~800 ms。

10x 差距。如果你的 router 在各 replica 間命中 60-80% 的前綴 cache，你就以 N 個 replica 的容量逼近單一 replica 的效能。如果只命中 10%，就等於單純的水平擴充。

### 跨區域有個新限制：網路 latency

區域間 RTT：
- us-east-1 ↔ us-west-2：~65 ms。
- us-east-1 ↔ eu-west-1：~75 ms。
- us-east-1 ↔ ap-southeast-1：~220 ms。

如果路由把 request 從 us-east-1 送到 ap-southeast-1 的熱門前綴，省下的 prefill（800 → 80 ms，即 720 ms）在付掉約 ~220 ms 的 RTT 後縮到約 500 ms；若遠端區域把 request 排隊，或已驅逐該前綴，優勢就會消失。GORGO（2026 研究）把這點明確化：要共同最小化 `prefill_time + network_latency`，而不是只看 prefill。答案通常是維持區域內路由，除非是 prefill 占主導的數 MB 大型前綴。

### 商業的「cross-region inference」在這裡幫不上忙

AWS Bedrock cross-region inference 在容量吃緊時，自動把 request 路由到其他區域。它優化的是可用性，不是 TTFT，並且把 inference 當成不透明的東西。GKE Multi-Cluster Gateway 也一樣：服務層級的 failover，不感知 KV cache。

即使使用這些產品，你仍然需要應用層的 cache-aware router。它們處理「us-east-1 著火了」的情況。Cache-aware 路由處理的是 TTFT 的情況。

### DR 衛生：32% 缺檔案的問題

2026 年被廣泛引用的統計：32% 的 LLM DR 失敗，是因為團隊備份了權重，卻忘了：

- `tokenizer.json` 或 `tokenizer.model`
- Quantization 設定 (`quantize_config.json`、AWQ scales、GPTQ zero-points)
- 模型專屬設定（RoPE scaling、attention masks、chat templates）
- Engine 設定 (`vllm_config.yaml`、sampling 預設值、LoRA adapter manifests)

修法是三檔案最小 DR manifest：

1. HF 模型 repo 下的所有檔案（權重 + 設定 + tokenizer）。
2. Engine 專屬的 serving 設定。
3. 部署 manifest（K8s YAML、Dockerfile、相依套件 lock）。

另外：每季跑一次 DR 演練。JPMorgan 在 2024 年 11 月的 us-east-1 演練之所以能在 22 分鐘內復原，是因為事先演練過作法手冊。

### 資料落地是另一個正交的問題

歐盟客戶的 PHI 不能離開歐盟。如果你的 cache-aware router 為了前綴相符，把源自巴黎的 request 送到 us-east-1，不管 TTFT 提升多少，你都違反了 GDPR。在優化 cache 之前，先依落地邊界切分 router。

### 該記住的數字

- Cache hit 與 miss 的 TTFT 差距：約 ~10x（2K prompt 上 80 ms vs 800 ms）。
- 美國-歐洲區域間 RTT：~75 ms。
- DR 失敗：32% 漏掉 tokenizer/quant 設定。
- JPMorgan 2024 年 11 月 us-east-1 failover：22 分鐘（30 分鐘 SLA）。

```figure
cache-aware-router
```

## 實際使用 (Use It)

`code/main.py` 在多區域工作負載上模擬三種路由策略（round-robin、cache-aware 區域內、cache-aware 全域）。回報 cache hit rate、TTFT P50/P99，以及跨區域費用。

## 交付 (Ship It)

這堂課產出 `outputs/skill-multi-region-router.md`。給定區域、資料落地限制與 SLA，設計路由計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。在 75 ms RTT 下，prompt 長度多長時，跨區域路由會勝過僅限本地的路由？
2. 你的 cache hit rate 從 70% 掉到 12%。診斷三個可能原因，以及能佐證每一個的可觀測指標。
3. 為在 vLLM 上服務、使用 5 個 LoRA adapters 的 70B AWQ 量化模型設計 DR manifest。列出每個檔案與設定。
4. 辯論 Bedrock cross-region inference 對有嚴格 TTFT SLO 的 fintech 是否「足夠」。引用具體行為。
5. 一個源自巴黎的 request 在 us-east-1 找到前綴相符。你要路由它嗎？寫出這個政策。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Cache-aware routing | 「聰明的 LB」 | 依前綴 hash 相符，路由到持有 KV-cache 的 replica |
| KV-cache events | 「cache pub-sub」 | Replica 發布 block 新增/驅逐；router 建立索引 |
| Prefix hash | 「cache key」 | 前 N 個 tokens 的 hash，作為 router 查找之用 |
| GORGO | 「跨區域路由研究」 | arXiv 2602.11688；把網路 latency 作為明確項目 |
| Cross-region inference | 「Bedrock CRI」 | AWS 產品；可用性 failover，不感知 TTFT |
| DR manifest | 「備份清單」 | 還原所需的每一個檔案，不只是權重 |
| Data residency | 「GDPR 邊界」 | 關於哪個區域可以看到使用者資料的法律限制 |
| RTT | 「往返時間」 | 網路 latency；美國-歐洲 75 ms、美國-APAC 220 ms |
| LLM-aware LB | 「cache-hit LB」 | 作為一種產品類別的 cache-aware router |

## 延伸閱讀 (Further Reading)

- [BentoML — Multi-cloud and cross-region inference](https://bentoml.com/llm/infrastructure-and-operations/multi-cloud-and-cross-region-inference)
- [arXiv — GORGO (2602.11688)](https://arxiv.org/html/2602.11688v1)：含網路 latency 項的跨區域 KV-cache 重用。
- [TianPan — Multi-Region LLM Serving Cache Locality](https://tianpan.co/blog/2026-04-17-multi-region-llm-serving-data-residency-routing)
- [AWS Bedrock Cross-Region Inference](https://docs.aws.amazon.com/bedrock/latest/userguide/cross-region-inference.html)：可用性 failover 文件。
- [vLLM Production Stack Router](https://github.com/vllm-project/production-stack)：cache-aware router 原始碼。
