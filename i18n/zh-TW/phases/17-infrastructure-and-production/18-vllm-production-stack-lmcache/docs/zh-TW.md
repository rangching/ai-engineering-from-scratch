# Production Serving Stack：KV Offloading 與 Cache-Aware Routing (Production Serving Stack)

> Production serving stack 把 router、engines 與 observability 串進同一個 Kubernetes 部署，並把 KV cache 視為可以離開 GPU 的資源。KV offloading 把 KV cache 從 GPU 記憶體抽出，在查詢與 engine 之間重用（CPU DRAM，然後是 disk/Ceph）。vLLM 的 production-stack 是參考部署；LMCache 是 offloading 層。vLLM 0.11.0 的 KV Offloading Connector（2026 年 1 月）透過 Connector API (v0.9.0+) 讓這件事變成非同步且可插拔。Offload 路徑通常對 request 路徑是隱藏的，不過 cache miss 與 promotion 仍可能增加端到端 latency。即使沒有共用前綴，LMCache 也很有價值：當 GPU 的 KV slot 用完時，被搶佔的 request 可以從 CPU 還原，而不必重新計算 prefill。在 4 台 a3-highgpu-4g 共 16x H100 (80GB HBM) 上發布的 benchmark：當 KV cache 超過 HBM 時，原生 CPU offload 與 LMCache 都大幅提升 throughput；在 KV 占用低時，所有設定都與基準相當，僅有少許開銷。

**Type:** Learn
**Languages:** Python (stdlib, toy KV-spill simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang/RadixAttention)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 畫出 vLLM production-stack 的各層：router、engines、KV offload、observability。
- 解釋 KV Offloading Connector API (v0.9.0+)，以及 0.11.0 的非同步路徑如何隱藏 offload latency。
- 量化 LMCache CPU-DRAM 在何時有幫助（KV > HBM），何時增加開銷（KV 小到放得進 HBM）。
- 在給定部署限制下，於原生 vLLM CPU offload 與 LMCache connector 之間做選擇。

## 問題 (The Problem)

你的 vLLM serving 顯示 GPU 的 HBM 100% 滿載，只要並行度一升高就出現搶佔事件。Request 被驅逐、重新排隊，你一分鐘內把同一個 2K-token 的 prompt 重新 prefill 四次。GPU 運算花在多餘的 prefill 上；goodput 遠低於原始 throughput。

增加 GPU 的成本是線性的。增加 HBM 不可能。但 CPU DRAM 很便宜：一個 socket 就有 512 GB+，latency 比 HBM 差幾個數量級，但對於「暫時保持溫熱」的 KV cache 來說完全夠用。

LMCache 把 KV cache 抽到 CPU DRAM，讓被搶佔的 request 快速恢復，並讓各 engine 之間重複的前綴共用 cache，而不必每個 engine 都重新 prefill。

## 核心概念 (The Concept)

### vLLM production-stack

`github.com/vllm-project/production-stack` 是參考的 Kubernetes 部署：

- **Router**：cache-aware (Phase 17 · 11)。消費 KV 事件。
- **Engines**：vLLM workers。每張 GPU 一個，或每個 TP/PP 群組一個。
- **KV cache offload**：LMCache 部署或原生 connector。
- **Observability**：Prometheus 抓取、Grafana 儀表板、OTel traces。
- **Control plane**：服務探索、設定、滾動更新。

以 Helm chart + operator 形式提供。

### KV Offloading Connector API (v0.9.0+)

vLLM 0.9.0 引入 Connector API，用於可插拔的 KV cache 後端。你的 engine 把 blocks offload 給 connector；connector 儲存它們（RAM、disk、物件儲存、LMCache）。Request 需要某個 block 時，connector 把它載回來。

vLLM 0.11.0（2026 年 1 月）新增非同步 offload 路徑：offload 可以在背景進行，所以 engine 在一般情況下不會被它阻塞。端到端 latency 與 throughput 仍取決於工作負載型態、KV cache 命中率與系統壓力；vLLM 自己的說明指出，客製 kernel 的 offload 在低命中率時可能降低 throughput，而非同步排程與 speculative decoding 有已知的互動問題。

### 原生 CPU offload vs LMCache

**原生 vLLM CPU offload**：engine 本地。把 KV blocks 存在主機 RAM。實作快、沒有網路跳轉。不會跨 engine。

**LMCache connector**：叢集規模。把 blocks 存在共用的 LMCache server（CPU DRAM + Ceph/S3 層）。任何 engine 都能存取這些 blocks。已發布 16x H100 的 benchmark。

單一 engine 有 HBM 壓力時，選原生。多個 engine 共用前綴時（有共同 system prompt 的 RAG、有共用 template 的多租戶），選 LMCache。

### Benchmark 行為

分布在 4 台 a3-highgpu-4g 的 16x H100 (80 GB HBM) 測試：

- KV 占用低（短 prompt、低並行）：所有設定與基準相當，LMCache 增加約 ~3-5% 開銷。
- 中等占用：LMCache 開始在跨 engine 的前綴重用上發揮作用。
- KV 超過 HBM：原生 CPU offload 與 LMCache 都大幅提升 throughput；LMCache 增益更大，因為有跨 engine 共用。

### LMCache 何時具決定性

- 多租戶 serving，system prompt 在租戶之間共用。
- RAG，文件 chunk 在查詢之間重複。
- 同一個 base 上的 fine-tuned 變體 (LoRA)，base 模型 KV 的重用可以減少多餘工作。
- 大量搶佔的工作負載：從 CPU 還原比重新 prefill 便宜。

### 何時不要啟用

- HBM 壓力小：你付出開銷卻沒有好處。
- 短 context (<1K tokens)：傳輸時間 > 重新 prefill。
- 單租戶單 prompt 的工作負載：沒有可捕捉的重用。

### 與 disaggregated serving 整合

Phase 17 · 17 的 disaggregated serving + LMCache 會疊加：從 prefill pool 傳到 decode pool 的 KV 若沒用到，會落進 LMCache；後續查詢從 LMCache 拉取。Phase 17 · 11 的 cache-aware router 可以路由到本地 cache 或 LMCache 共用 cache 相符的 engine。

### 該記住的數字

- vLLM 0.9.0：Connector API 上線。
- vLLM 0.11.0（2026 年 1 月）：非同步 offload 路徑；端到端 latency 影響取決於工作負載、KV 命中率與系統壓力（並非絕對保證）。
- 16x H100 benchmark：當 KV 占用超過 HBM 時，LMCache 有幫助。
- HBM 壓力小：3-5% 開銷而無好處。

```figure
zero-sharding
```

## 實際使用 (Use It)

`code/main.py` 在大量搶佔的工作負載下，模擬有無 LMCache 的情況。回報避免的重新 prefill 次數、throughput 增益，以及損益平衡的 HBM 使用率。

## 交付 (Ship It)

這堂課產出 `outputs/skill-vllm-stack-decider.md`。給定工作負載型態與 vLLM 部署，決定用原生、LMCache，還是都不用。

## 練習 (Exercises)

1. 執行 `code/main.py`。HBM 使用率到多少時，LMCache 開始回本？
2. 某租戶在每小時 200 次查詢間共用一個 6K-token 的 system prompt。計算每個租戶預期的 LMCache 節省。
3. LMCache server 是單點故障。設計 HA 策略（replicas、fallback 到原生）。
4. LMCache 存到旋轉硬碟上的 Ceph。對 70B FP8 上 4K-token 的 KV (500 MB)，讀取時間 vs 重新 prefill 如何？
5. 辯論 vLLM 0.11.0 的非同步路徑是否「免費」：開銷藏在哪裡？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Production-stack | 「參考部署」 | vLLM 的 Kubernetes Helm chart + operator |
| Connector API | 「KV 後端介面」 | vLLM 0.9.0+ 的可插拔 KV 儲存介面 |
| Native CPU offload | 「engine 本地溢出」 | 把 KV 存在同一個 engine 的主機 RAM |
| LMCache | 「叢集 KV cache」 | 建立在 CPU DRAM + disk 上的跨 engine KV cache server |
| 0.11.0 async | 「非阻塞 offload」 | Offload 隱藏在 engine stream 之後 |
| Preemption | 「驅逐以騰出空間」 | HBM 滿時的 KV cache 洗牌 |
| Prefix reuse | 「相同的 system prompt」 | 多個查詢共用開頭；cache 命中 |
| Ceph tier | 「disk 層」 | cache 階層中 DRAM 之下的持久儲存 |

## 延伸閱讀 (Further Reading)

- [vLLM Blog — KV Offloading Connector (Jan 2026)](https://blog.vllm.ai/2026/01/08/kv-offloading-connector.html)
- [vLLM Production Stack GitHub](https://github.com/vllm-project/production-stack)：Helm chart + operator。
- [LMCache for Enterprise-Scale LLM Inference (arXiv:2510.09665)](https://arxiv.org/html/2510.09665v2)
- [LMCache GitHub](https://github.com/LMCache/LMCache)：Connector 實作。
- [vLLM 0.11.0 release notes](https://github.com/vllm-project/vllm/releases)：非同步路徑細節。
