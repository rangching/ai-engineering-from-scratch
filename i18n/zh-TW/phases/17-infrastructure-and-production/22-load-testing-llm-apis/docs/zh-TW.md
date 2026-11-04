# LLM API 的負載測試：為什麼 k6 和 Locust 會說謊 (Load Testing LLM APIs)

> 傳統的負載測試工具不是為串流回應、可變輸出長度、token 層級的指標或 GPU 飽和而設計的。有兩個陷阱咬到大多數團隊。GIL 陷阱：Locust 的 token 層級量測在 Python GIL 下執行 tokenization，在高並行下與 request 產生互相競爭；tokenization 的積壓會讓回報的 inter-token latency 膨脹，也就是瓶頸在你的 client，而不是 server。Prompt 一致性陷阱：迴圈中使用相同的 prompt，只測到 token 分布上的一個點；真實流量有可變長度與多樣的前綴相符。LLMPerf 用 `--mean-input-tokens` + `--stddev-input-tokens` 修正這點。2026 年的工具對照：LLM 專用的（GenAI-Perf、LLMPerf、LLM-Locust、guidellm）提供 token 層級的準確度；**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025 年 9 月)**，支援串流、透過 TestRun/PrivateLoadZone CRDs 做 Kubernetes 原生的分散式測試，最適合 CI/CD 關卡；Vegeta 用於 Go 的固定速率飽和測試；Locust 2.43.3 只有搭配 LLM-Locust 擴充才支援串流。負載模式：steady-state、ramp、spike（autoscaling 測試）、soak（記憶體洩漏）。

**Type:** Build
**Languages:** Python (stdlib, toy realistic-prompt generator + latency collector)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 17 · 03 (GPU Autoscaling)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋讓一般負載測試工具對 LLM API 說謊的兩個反模式（GIL 陷阱、prompt 一致性陷阱）。
- 依用途挑選工具：LLMPerf（benchmark 執行）、k6 + streaming 擴充（CI 關卡）、guidellm（大規模合成）、GenAI-Perf（NVIDIA 參考）。
- 設計四種負載模式（steady、ramp、spike、soak），並說出每種能抓到的失敗模式。
- 用輸入 tokens 的平均值 + 標準差（而不是固定長度）建立擬真的 prompt 分布。

## 問題 (The Problem)

你用 k6 測試 LLM endpoint，500 個並行使用者，撐住了。你上線。在 production 只有 200 個實際使用者時，服務垮了，P99 TTFT 暴增，GPU 被釘滿。

發生了兩件事。第一，k6 送出 500 個相同的 prompt，你的 request 合併與 prefix caching 讓它看起來像在處理 500 個並行 decode，實際上只處理了一個。第二，k6 不會像人眼所感受到的那樣追蹤串流回應上的 inter-token latency；它看到的是一個 HTTP 連線，而不是 500 個以不同間隔抵達的 tokens。

LLM 的負載測試是一門獨立的學問。

## 核心概念 (The Concept)

### GIL 陷阱 (Locust)

Locust 使用 Python，並在 GIL 下於 client 端執行 tokenization。高並行下，tokenizer 會排在 request 產生之後。回報的 inter-token latency 包含了 client 端 tokenization 的積壓。你以為是 server 慢；其實是測試工具。

修法：LLM-Locust 擴充把 tokenization 移到獨立行程，或使用編譯語言的工具（k6、使用 tokenizers.rs 的 LLMPerf）。

### Prompt 一致性陷阱

所有已知的負載測試工具都只讓你設定一個 prompt。在 10,000 次迭代的迴圈測試中，每次送的都是完全相同的 prompt。Server 每次看到相同的前綴，prefix cache 命中率逼近 100%，throughput 看起來很棒。

修法：從 prompt 分布取樣。LLMPerf 使用 `--mean-input-tokens 500 --stddev-input-tokens 150`，長度多樣、內容多樣。

### 四種負載模式

1. **Steady-state**：固定 RPS 持續 30-60 分鐘。抓到：基準效能退化。
2. **Ramp**：在 15 分鐘內把 RPS 從 0 線性增加到目標。抓到：容量斷點、暖機異常。
3. **Spike**：突然提高 3-10x RPS 持續 2 分鐘後恢復。抓到：autoscaling latency、佇列飽和、cold-start 影響。
4. **Soak**：steady-state 持續 4-8 小時。抓到：記憶體洩漏、連線池漂移、observability 溢出。

### 2026 年的工具對照

**LLMPerf** (Anyscale)：Python，但 tokenization 由 Rust 支撐。平均值/標準差的 prompt。支援串流。效能測試的最佳預設。

**NVIDIA GenAI-Perf**：NVIDIA 的參考。使用 Triton client；指標涵蓋完整。注意它的 ITL 排除 TTFT；LLMPerf 的則包含。兩個工具對同一台 server 會產生不同的 TPOT。

**LLM-Locust** (TrueFoundry)：修正 GIL 陷阱的 Locust 擴充。熟悉的 Locust DSL + 串流指標。

**guidellm**：大規模合成 benchmark。

**k6 v2026.1.0** + **k6 Operator 1.0 GA (2025 年 9 月)**：
- k6 本身（Go、編譯式、沒有 GIL）新增了支援串流的指標。
- k6 Operator 使用 TestRun / PrivateLoadZone CRDs 做 Kubernetes 原生的分散式測試。
- 最適合 CI/CD 關卡與 SLA 測試。

**Vegeta**：Go，比 k6 簡單。固定速率的 HTTP 飽和測試。不感知 LLM，但適合 gateway / rate-limit 測試。

**Locust 2.43.3 原版**：對 LLM 有 GIL 陷阱。只有搭配 LLM-Locust 擴充才行。

### CI 中的 SLA 關卡

在 PR 上執行 k6，條件為：

- 在基準 RPS 下每個執行 30-50 次迭代。
- 關卡：P50/P95 TTFT、5xx < 5%、TPOT 低於門檻。
- 違反就讓 build 失敗。

### 擬真的 prompt 分布

用真實流量樣本（如果有）或已發表的分布（例如聊天用 ShareGPT prompts、程式碼用 HumanEval）建立。把平均值 + 標準差餵給 LLMPerf。無論如何都要避免用單一 prompt 的迴圈。

### 該記住的數字

- k6 Operator 1.0 GA：2025 年 9 月。
- k6 v2026.1.0：支援串流的指標。
- 典型 LLMPerf 執行：並行度 X 下 100-1000 個 request。
- 典型 CI 關卡：每個 PR 30-50 次迭代。
- 四種模式：steady、ramp、spike、soak。

```figure
load-pattern-waves
```

## 實際使用 (Use It)

`code/main.py` 以擬真的 prompt 分布模擬負載測試，量測有效 TPOT，並展示 prompt 一致性陷阱。

## 交付 (Ship It)

這堂課產出 `outputs/skill-load-test-plan.md`。給定工作負載與 SLA，挑選工具並設計四種負載模式。

## 練習 (Exercises)

1. 執行 `code/main.py`。比較一致與擬真的分布，差距在哪裡？
2. 為 CI 關卡寫一個 k6 腳本：100 並行下 TTFT P95 < 800 ms，執行時間 5 分鐘。
3. 你的 soak 測試顯示記憶體每小時增加 50 MB。說出三個原因，以及用來區分它們的 instrumentation。
4. 從 10 RPS 到 100 RPS 的 spike 測試。如果有 Karpenter + vLLM production-stack（Phase 17 · 03 + 18），預期的恢復時間是多少？
5. 在同一台 server 上，GenAI-Perf 回報 TPOT=6ms，LLMPerf 回報 TPOT=11ms。解釋原因。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| LLMPerf | 「LLM 測試工具」 | Anyscale 的 benchmark 工具，支援串流 |
| GenAI-Perf | 「NVIDIA 工具」 | NVIDIA 參考測試工具 |
| LLM-Locust | 「LLM 版 Locust」 | 修正 GIL 陷阱的 Locust 擴充 |
| guidellm | 「合成 benchmark」 | 大規模合成工具 |
| k6 Operator | 「K8s k6」 | 以 CRD 為基礎的分散式 k6 |
| GIL trap | 「Python client 開銷」 | Tokenization 積壓使回報的 latency 膨脹 |
| Prompt-uniformity trap | 「單一 prompt 謊言」 | 用同一 prompt 迴圈會命中 cache，使 throughput 膨脹 |
| Steady-state | 「固定負載」 | 持續 N 分鐘的平穩 RPS |
| Ramp | 「線性上升」 | 在一段時間內從 0 增加到目標 |
| Spike | 「突發測試」 | 突然乘上倍數然後恢復 |
| Soak | 「長時間測試」 | 數小時，用於偵測洩漏 |

## 延伸閱讀 (Further Reading)

- [TianPan — Load Testing LLM Applications](https://tianpan.co/blog/2026-03-19-load-testing-llm-applications)
- [PremAI — Load Testing LLMs 2026](https://blog.premai.io/load-testing-llms-tools-metrics-realistic-traffic-simulation-2026/)
- [NVIDIA NIM — Introduction to LLM Inference Benchmarking](https://docs.nvidia.com/nim/large-language-models/1.0.0/benchmarking.html)
- [TrueFoundry — LLM-Locust](https://www.truefoundry.com/blog/llm-locust-a-tool-for-benchmarking-llm-performance)
- [LLMPerf](https://github.com/ray-project/llmperf)
- [k6 Operator](https://github.com/grafana/k6-operator)
