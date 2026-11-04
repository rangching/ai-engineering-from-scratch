# Inference 指標 (Inference Metrics)：TTFT、TPOT、ITL、Goodput、P99

> 五個指標決定 inference 部署是否有效。TTFT 是 prefill 加佇列加網路。TPOT（常被稱為 ITL，不過各工具對 ITL 是否計入第一個 token 看法不一）是 memory-bound 的每 token decode 成本。End-to-end latency 是 TTFT 加上 TPOT 乘以輸出長度。Throughput 是整個 fleet 彙總的每秒 tokens。但對產品而言真正重要的是 goodput：同時滿足每一項 SLO 的 request 比例。高 throughput 搭配低 goodput，代表你處理的 tokens 沒有準時送到使用者手上。2026 年 Llama-3.1-8B-Instruct 在 TRT-LLM 上的參考數字：平均 TTFT 162 ms、平均 TPOT 7.33 ms、平均 E2E 1,093 ms。一律回報 P50、P90、P99，不要只報平均。還要小心量測陷阱：GenAI-Perf 在 ITL 計算中排除 TTFT，LLMPerf 則包含；兩個工具對同一次執行的 TPOT 看法不同。

**Type:** Learn
**Languages:** Python (stdlib, toy percentile calculator and goodput reporter)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 精確定義 TTFT、TPOT、ITL、E2E、throughput 與 goodput，並說出每個指標量測的組成部分。
- 解釋為什麼平均值是 LLM serving 的錯誤統計量，以及如何解讀 P50/P90/P99。
- 建立 SLO 多重限制（例如 TTFT<500 ms 且 TPOT<15 ms 且 E2E<2 s），並對其計算 goodput。
- 說出兩個對同一次執行的 TPOT 看法不一的 benchmark 工具，並解釋原因。

## 問題 (The Problem)

「我們的 throughput 是每秒 15,000 tokens。」那又怎樣？如果 40% 的 request 端到端超過 2 秒，使用者早就放棄這個 session 了。光看 throughput 無法告訴你產品是否運作正常。

Inference 有多個 latency 維度，每一個的失敗方式都不同。Prefill 是 compute-bound，隨 prompt 長度成長。Decode 是 memory-bound，隨 batch size 成長。佇列延遲是營運問題。網路是實體距離問題。你需要為每一項設置獨立的指標，需要百分位數，還需要一個單一的綜合指標來回答「使用者有沒有得到他們預期的」，那就是 goodput。

## 核心概念 (The Concept)

### TTFT：time to first token

`TTFT = queue_time + network_request + prefill_time`

Prompt 很長時由 prefill 主導。在 H100 上的 Llama-3.3-70B FP8，32k prompt 需要約 ~800 ms 的純 prefill。佇列時間是 scheduler 在負載下的行為。Network request 是包含 TLS 的傳輸時間。TTFT 是使用者在任何內容串流回來之前所看到的 latency。

### TPOT / ITL：inter-token latency

同一個量有很多名稱。`TPOT`（time per output token）、`ITL`（inter-token latency）、`decode latency per token` 都是同一回事。它是第一個之後，連續串流 tokens 之間的時間。

`TPOT = (decode_forward_time + scheduler_overhead) / tokens_produced`

在同樣的 H100 上 Llama-3.3-70B、啟用 chunked prefill 的 stack，平均 TPOT 約 ~7 ms。沒有 chunked prefill 時，如果鄰近的 sequence 正在做長 prefill，TPOT 可能飆到 50 ms。請看 P99，不要看平均。

### E2E latency

`E2E = TTFT + TPOT * output_tokens + network_response`

對長輸出（>500 tokens），E2E 由 TPOT 主導。對長 prompt 搭配短輸出，E2E 由 TTFT 主導。請回報以輸出長度為條件的 E2E。

### Throughput

`throughput = total_output_tokens / elapsed_time`

彙總指標。告訴你 fleet 的效率，但不會告訴你個別 request 的健康度。

### Goodput：你真正在意的指標

`goodput = fraction of requests meeting (TTFT <= a) AND (TPOT <= b) AND (E2E <= c)`

SLO 是多重限制。只有每一項限制都成立，request 才算「好」。Goodput 就是這個比例。有些論文和儀表板把 goodput 報成速率：每秒滿足所有 SLO 的 request 數。把比例乘以 request 速率即可換算。高 throughput 但 goodput 只有 60% 是失敗。較低的 throughput 但 goodput 99% 才是目標。

2026 年，goodput 是 MLPerf Inference v6.0 提交以及 AI 平台供應商內部 SLA 追蹤所使用的指標。

### 為什麼平均值是錯的統計量

LLM latency 分布是右偏的。一個鄰近有長 prefill 的 decode batch，可能送出 500 個 TPOT ~7 ms 的 tokens，以及 20 個 TPOT ~60 ms 的 tokens。平均 TPOT 是 9 ms。P99 TPOT 是 65 ms。使用者經常碰到 P99，這就是他們離開的原因。

一律回報三元組 (P50, P90, P99)。就使用者體驗而言，P99 才是你要優化的。

### 參考數字：2026 年 TRT-LLM 上的 Llama-3.1-8B-Instruct

- 平均 TTFT：162 ms
- 平均 TPOT：7.33 ms
- 平均 E2E：1,093 ms
- P99 TPOT：依 chunked-prefill 設定而異，10-25 ms。

這些是 NVIDIA 發布的參考點。它們會隨模型大小（70B 會是 3-5x）、硬體（H100 vs B200 約 3x）與負載而變。

### 量測陷阱

2026 年最常用的兩個 benchmark 工具，對同一次執行的 TPOT 看法不一致：

- **NVIDIA GenAI-Perf**：ITL 計算排除 TTFT。ITL 從 token 2 開始。
- **LLMPerf**：包含 TTFT。ITL 從 token 1 開始。

對一個 TTFT 500 ms、100 個輸出 tokens、總 decode 700 ms 的 request，GenAI-Perf 回報 `ITL = 700/99 = 7.07 ms`，LLMPerf 回報 `ITL = 1200/100 = 12.00 ms`。選哪個工具會改變數字。

一律註明用的是哪個工具。一律公布定義。

### 建立 SLO

2026 年面向消費者的 70B 聊天模型，合理的 SLO 如下：

- TTFT P99 <= 800 ms。
- TPOT P99 <= 25 ms。
- <300-token 輸出的 E2E P99 <= 3 s。
- Goodput 目標 >= 99%。

企業 SLO 會收緊 TTFT（200-400 ms）並放寬 E2E。重點是把它們寫下來，三項都量測，並把 goodput 當作單一綜合指標來追蹤。

### 如何量測

- 執行真實流量或擬真的合成流量（LLMPerf 搭配 `--mean-input-tokens 800 --stddev-input-tokens 300 --mean-output-tokens 150`）。
- Benchmark 執行時，以尖峰並行度的 2x 為目標。
- 跑 30-50 次迭代，對合併樣本取百分位數。
- 發布時附上工具名稱、工具版本、模型、硬體、並行度、prompt 分布。

```figure
throughput-latency
```

## 實際使用 (Use It)

`code/main.py` 是一個玩具 goodput 計算器。產生合成的 latency 分布，套用 SLO，並計算 goodput。同時展示 GenAI-Perf 與 LLMPerf 在同一份 trace 上的 TPOT 差異。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?users=1) 這個機器展示了在 NVIDIA L4 上，每個輸出 token 的時間從何而來。一個 decode 步驟要讀取 16.41 GB 的權重，在 300 GB/s 下至少需要 54.7 ms，而實測的步驟花了 60.16 ms。

## 交付 (Ship It)

這堂課產出 `outputs/skill-slo-goodput-gate.md`。給定工作負載與 SLO，它會產出一份可用於 CI/CD 的 benchmark 配方，以 goodput 而非 throughput 作為部署的關卡。

## 練習 (Exercises)

1. 執行 `code/main.py`。產生一個帶有 1% 尾端突波的分布。當你把 P99 TPOT 從 30 ms 收緊到 15 ms，goodput 如何變化？
2. 某廠商宣稱「Llama 3.3 70B 在 H100 上 15,000 tok/s」。說出在相信它之前要問的三個問題。
3. 為什麼 chunked prefill 能保護 P99 TPOT，卻不能保護平均 TPOT？
4. 為語音助理（第一個 token 是聽到，而不是讀到）建立一個消費者 SLO。哪個指標對使用者最可見？
5. 閱讀 LLMPerf README 與 GenAI-Perf 文件。找出這兩個工具看法不一的另外三個指標。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| TTFT | 「第一個 token 的時間」 | 佇列 + 網路 + prefill；長 prompt 時由 prefill 主導 |
| TPOT | 「每個輸出 token 的時間」 | 第一個之後，memory-bound 的每 token decode 成本 |
| ITL | 「token 間 latency」 | 在多數工具中與 TPOT 相同（並非全部，見 GenAI-Perf） |
| E2E | 「端到端」 | TTFT + TPOT * output_len；再加上回應端網路 |
| Throughput | 「tok/s」 | Fleet 效率；沒有 latency 百分位數就沒有用 |
| Goodput | 「達成 SLO 的比率」 | 同時滿足每項 SLO 限制的 request 比例 |
| P99 | 「尾端」 | 百分之一的最差情況 latency；使用者體驗指標 |
| SLO multi-constraint | 「聯合條件」 | 三個 latency 界線的 AND；任何一項被違反，request 就失敗 |
| GenAI-Perf vs LLMPerf | 「工具陷阱」 | 兩個工具對 ITL 是否包含 TTFT 看法不一 |

## 延伸閱讀 (Further Reading)

- [NVIDIA NIM — LLM Benchmarking Metrics](https://docs.nvidia.com/nim/benchmarking/llm/latest/metrics.html)：TTFT、ITL、TPOT 的標準定義。
- [Anyscale — LLM Serving Benchmarking Metrics](https://docs.anyscale.com/llm/serving/benchmarking/metrics)：另一套定義與量測配方。
- [BentoML — LLM Inference Metrics](https://bentoml.com/llm/inference-optimization/llm-inference-metrics)：在真實部署上的應用量測。
- [LLMPerf](https://github.com/ray-project/llmperf)：以 Ray 為基礎的開源 benchmark。
- [GenAI-Perf](https://github.com/triton-inference-server/perf_analyzer/blob/main/genai-perf/README.md)：NVIDIA 的 benchmark 工具。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)：業界公認、以 goodput 為基礎的 benchmark。
