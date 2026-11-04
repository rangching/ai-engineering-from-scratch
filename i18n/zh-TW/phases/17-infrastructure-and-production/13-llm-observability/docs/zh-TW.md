# LLM Observability Stack 選型 (LLM Observability Stack Selection)

> 2026 年的 observability 市場分成兩類。開發平台（LangSmith、Langfuse、Comet Opik）把 monitoring 與 evals、prompt 管理、session replays 綁在一起。Gateway/instrumentation 工具（Helicone、SigNoz、OpenLLMetry、Phoenix）專注在 telemetry。Langfuse 是 MIT 授權的核心，OSS 平衡性強（雲端免費方案每月 50K events）。Phoenix 原生支援 OpenTelemetry，採 Elastic License 2.0，非常適合 drift/RAG 視覺化，但不是持久的 production 後端。Arize AX 採用 zero-copy 的 Iceberg/Parquet 整合，宣稱比單體式 observability 便宜 100x。LangSmith 在 LangChain/LangGraph 領先，$39/user/mo，僅 Enterprise 可自架。Helicone 以 proxy 為基礎，15-30 分鐘就能設定完，每月 100K req 免費，但在 agent traces 上深度較少。常見的 production 模式：Gateway (Helicone/Portkey) + eval 平台 (Phoenix/TruLens)，以 OpenTelemetry 黏合。

**Type:** Learn
**Languages:** Python (stdlib, toy trace-sampling simulator)
**Prerequisites:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 區分開發平台（整合式：evals + prompts + sessions）與 gateway/telemetry 工具（只有 traces + metrics）。
- 把六個主要工具（Langfuse、LangSmith、Phoenix、Arize AX、Helicone、Opik）對應到它們的授權、定價與最佳使用情境。
- 解釋讓你能把 gateway 工具與獨立 eval 平台結合的 OpenTelemetry 黏合模式。
- 說出 2026 年的成本差異點（Arize AX 的 zero-copy 做法 vs 單體式 ingest），並說明大約 100x 的倍數。

## 問題 (The Problem)

你交付了一個 LLM 功能。它能運作。但你看不到 prompt 失敗、tool 迴圈、latency 退化、成本暴增，或是 prompt-cache hit rate。你 Google「LLM observability」，得到八個工具，全都宣稱用三種不同價位解決同一個問題。

它們其實並沒有解決同一個問題。LangSmith 回答「這個 LangGraph run 為什麼失敗？」Phoenix 回答「我的 RAG pipeline 有在 drift 嗎？」Helicone 回答「哪個 app 在燒 tokens？」Langfuse 回答「我能把整套都自架嗎？」不同的工具，不同的受眾。

挑選牽涉四個維度：stack（LangChain？原生 SDK？多供應商？）、授權容忍度（只能 MIT？Elastic 可以？商業的也行？）、預算（免費方案？每月 $100？每月 $1000？），以及自架（必須？最好有？絕不？）。

## 核心概念 (The Concept)

### 兩個類別

**開發平台**把 observability 與 evals、prompt 管理、dataset 版本控管、session replay 綁在一起。你執行實驗，看哪個 prompt 有效，並針對舊的優勝者對新 prompt 做 dataset 迴歸測試。LangSmith、Langfuse、Comet Opik。

**Gateway/telemetry 工具**對 inference 呼叫做 instrumentation：prompt、回應、tokens、latency、模型、成本。Helicone、SigNoz、OpenLLMetry、Phoenix。極簡。可以透過 OpenTelemetry 與獨立的 eval 工具結合。

### Langfuse：OSS 平衡

- 核心採 Apache / MIT 授權；可透過 Docker 自架。
- 雲端免費方案：每月 50K events。付費：團隊方案每月 $29。
- Evals、prompt 管理、traces、datasets。對開發平台的四項功能涵蓋合理。
- 最佳情境：你想要 LangSmith 等級的功能，但必須自架或維持 OSS 授權。

### Phoenix (Arize)：telemetry 優先、原生 OpenTelemetry

- Elastic License 2.0；自架很容易。
- 在 RAG 與 drift 視覺化上表現優異。Embedding 空間的散佈圖是一等公民功能。
- 並非設計為持久的 production 後端，主要是開發期間的 observability。
- 最佳情境：RAG pipeline 開發、drift 除錯，production 則搭配獨立的 gateway。

### Arize AX：規模化方案

- 商業產品。透過 Iceberg/Parquet 與 zero-copy 資料湖整合。
- 宣稱在規模化時比單體式 observability（Datadog 等級）便宜約 ~100x。算法是：你把 traces 存在自己 S3 上的 Parquet，Arize 直接讀取。
- 最佳情境：>10M traces/day、已有資料湖，想要 LLM 專用的儀表板又不想付 Datadog 的價格。

### LangSmith：LangChain/LangGraph 優先

- 商業產品，$39/user/month。僅 Enterprise 可自架。
- 在 LangChain 與 LangGraph stack 上同類最佳。如果你兩者都不用，吸引力就較低。
- 最佳情境：全力投入 LangChain、願意付費的團隊。

### Helicone：以 proxy 為基礎的最小可行方案

- 把 `OPENAI_API_BASE` 換成 Helicone proxy，15-30 分鐘就設定完成。
- MIT 授權；每月 100K req 免費，付費方案每月 $20+。
- 包含 failover、caching、rate limits，同時也扮演 gateway。
- 在 agent / 多步驟 traces 上深度較少。
- 最佳情境：快速上手、單一 stack 的 app，需要 gateway + observability 合一。

### Opik (Comet)：OSS 開發平台

- Apache 2.0，完全 OSS。
- 功能組合與 Langfuse 類似，承襲 Comet 的血統。
- 最佳情境：已經在用 Comet 的 ML 團隊，想在同一個面板看 LLM observability。

### SigNoz：以 OpenTelemetry 為先的完整 APM

- Apache 2.0。處理一般 APM，並透過 OpenTelemetry 支援 LLM。
- 最佳情境：跨服務與 LLM 呼叫的統一 observability。

### 黏合劑：OpenTelemetry + GenAI semantic conventions

OpenTelemetry 在 2025 年底發布了 GenAI semantic conventions (`gen_ai.system`、`gen_ai.request.model`、`gen_ai.usage.input_tokens`)。能消費 OTel 的工具之間可以互通。逐漸成形的 production 模式：

1. 每次 LLM 呼叫都以 GenAI conventions 發出 OTel。
2. 日常路由到 gateway (Helicone / Portkey)。
3. 雙寫到 eval 平台 (Phoenix / Langfuse) 做迴歸。
4. 封存到資料湖 (Iceberg)，透過 Arize AX 或 DuckDB 做長期分析。

### 陷阱：在錯誤的層次做 instrumentation

在你的 agent framework 內部做 instrumentation（例如加上 LangSmith traces），會讓你與該 framework 綁死。在 HTTP/OpenAI-SDK 層做 instrumentation（透過 OpenLLMetry 或你的 gateway）則是可攜的。

### Sampling：你不可能全部保留

在 >1M requests/day 時，保留完整 trace 的成本高於 LLM 呼叫本身。依規則取樣：錯誤 100%、高成本 100%、成功 5%。永遠保留彙總；原始資料只保留長尾。

### 該記住的數字

- Langfuse 免費雲端：每月 50K events。
- LangSmith：$39/user/month。
- Helicone 免費：每月 100K req。
- Arize AX 宣稱：規模化時比單體式便宜約 ~100x。
- OpenTelemetry GenAI conventions：2025 年發布，2026 年廣泛採用。

```figure
i4-otel-glue
```

## 實際使用 (Use It)

`code/main.py` 模擬一天 1M 筆 traces，跨不同保留策略（100% ingest、取樣、取樣 + 錯誤）。回報儲存成本，以及每種策略下遺失了什麼。

## 交付 (Ship It)

這堂課產出 `outputs/skill-observability-stack.md`。給定 stack、規模、預算、授權立場，挑選工具。

## 練習 (Exercises)

1. 你使用 LangChain 的團隊想要 OSS 自架的 observability。在 Langfuse 或 Opik 之間挑一個並說明理由。
2. 在每天 5M traces、Datadog 報價每月 $150K 的情況下，計算 Arize AX 的損益平衡。
3. 設計一組 OpenTelemetry GenAI 屬性，作為你組織的準則，要求每次 LLM 呼叫都必須帶上。
4. 辯論 Phoenix 單獨使用是否足以應付 production。什麼時候不夠？
5. Helicone 有 20ms 的 proxy 開銷。在 P99 TTFT 300 ms 時可以接受嗎？如果 SLA 是 100 ms 呢？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| OpenLLMetry | 「LLM 的 OTel」 | 開源的 LLM OpenTelemetry instrumentation |
| GenAI conventions | 「OTel 屬性」 | LLM 呼叫的標準 OTel 屬性名稱 |
| LangSmith | 「LangChain observability」 | 與 LangChain 生態系綁定的商業平台 |
| Langfuse | 「OSS 版 LangSmith」 | 功能組合類似的 MIT OSS |
| Phoenix | 「Arize 開發工具」 | 原生 OpenTelemetry 的開發/eval 平台 |
| Arize AX | 「規模化 observability」 | 商業的 zero-copy Iceberg/Parquet observability |
| Helicone | 「proxy observability」 | 收集 LLM telemetry 並附帶 gateway 功能的 HTTP proxy |
| Opik | 「Comet LLM」 | Comet 的 Apache 2.0 OSS 開發平台 |
| Session replay | 「trace 重播」 | 重播含 tool 呼叫的完整 agent session |
| Eval | 「離線測試」 | 在有標註的資料集上執行候選模型/prompt |

## 延伸閱讀 (Further Reading)

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
