# OpenTelemetry GenAI：端對端追蹤 Tool Call (OpenTelemetry GenAI — Tracing Tool Calls End-to-End)

> 一個 agent 呼叫五個 tool、三個 MCP server 和兩個子 agent。你需要涵蓋這一切的單一 trace。OpenTelemetry GenAI 語意慣例（v1.37 以上為穩定屬性）是 2026 年的標準，Datadog、Langfuse、Arize Phoenix、OpenLLMetry 和 AgentOps 都原生支援。這堂課會列出必要的屬性，走過 span 階層（agent → LLM → tool），並交付一個可接上任何 OTel exporter 的 stdlib span emitter。

**Type:** Build
**Languages:** Python (stdlib, OTel span emitter)
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 LLM span 與 tool 執行 span 所需的 OTel GenAI 屬性。
- 建構涵蓋 agent 迴圈、LLM call、tool call 與 MCP client dispatch 的 trace 階層。
- 決定哪些內容要擷取（選擇加入）、哪些要遮蔽（預設）。
- 在不重寫 tool 程式碼的情況下，把 span 送到本地 collector（Jaeger、Langfuse）。

## 問題 (The Problem)

2026 年 2 月的一次除錯：使用者回報「我的 agent 有時要 30 秒才回應，有時只要 3 秒。」沒有 trace。Log 顯示 LLM call，但沒有 tool dispatch、沒有 MCP server 的往返，也沒有子 agent。你只能猜。最後你發現：有一個 MCP server 偶爾會在 cold-start 時卡住。

沒有端對端的追蹤，你就找不到這個問題。OTel GenAI 解決了它。

這些慣例在 2025 到 2026 年間，於 OpenTelemetry semantic-conventions 工作小組底下定案。它們定義了穩定的屬性名稱，讓 Datadog、Langfuse、Phoenix、OpenLLMetry 和 AgentOps 都能解析相同的 span。只要instrument 一次，就能送到任何 backend。

## 核心概念 (The Concept)

### Span 階層 (Span hierarchy)

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

整個結構巢狀在同一個 trace id 之下。Span id 連結父子關係。

### 必要的屬性 (Required attributes)

依據 2025-2026 的 semconv：

- `gen_ai.operation.name` — `"chat"`、`"text_completion"`、`"embeddings"`、`"execute_tool"`、`"invoke_agent"`。
- `gen_ai.provider.name` — `"openai"`、`"anthropic"`、`"google"`、`"azure_openai"`。
- `gen_ai.request.model` — 請求的 model 字串（例如 `"gpt-4o-2024-08-06"`）。
- `gen_ai.response.model` — 實際提供服務的 model。
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`。
- `gen_ai.response.id` — 用於對應的 provider response id。

Tool span 的屬性：

- `gen_ai.tool.name` — tool 識別碼。
- `gen_ai.tool.call.id` — 特定的 call id。
- `gen_ai.tool.description` — tool 的描述（選擇性）。

Agent span 的屬性：

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`。

### Span 種類 (Span kinds)

- 跨越 process 邊界的 call（LLM provider、MCP server）使用 `SpanKind.CLIENT`。
- agent 自己的迴圈步驟與 tool 執行使用 `SpanKind.INTERNAL`。

### 選擇加入的內容擷取

預設情況下，span 帶有指標與時間資訊，不含 prompt 或 completion。大型 payload 與 PII 預設關閉。設定 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` 和特定的內容擷取環境變數，才會包含內容。在 prod 啟用前請仔細審查。

### Span 上的事件 (Events on spans)

Token 層級的事件可以加成 span event：

- `gen_ai.content.prompt` — 輸入訊息。
- `gen_ai.content.completion` — 輸出訊息。
- `gen_ai.content.tool_call` — 所記錄的 tool call。

事件在 span 內依時間排序，以便詳細重播。

### Exporter

OTel span 可以匯出到：

- **Jaeger / Tempo。** 開源、on-prem。
- **Langfuse。** 專為 LLM observability 設計；可視覺化 token 用量。
- **Arize Phoenix。** 結合 eval 與追蹤。
- **Datadog。** 商業產品；原生解析 `gen_ai.*` 屬性。
- **Honeycomb。** 欄導向；方便查詢。

全部都使用 OTLP 這個 wire format。你的程式碼不必在意。

### 跨 MCP 傳播

當 MCP client 呼叫 server 時，把 W3C traceparent header 注入請求中。Streamable HTTP 支援標準 header。Stdio 原生不攜帶 HTTP header；規格 2026 年的路線圖討論過，要在 JSON-RPC call 上新增 `_meta.traceparent` 欄位。

在它推出之前：手動把 traceparent 放進每個請求的 `_meta`。Server 記錄 trace id。

### 指標 (Metrics)

除了 span，GenAI semconv 也定義了指標：

- `gen_ai.client.token.usage` — histogram。
- `gen_ai.client.operation.duration` — histogram。
- `gen_ai.tool.execution.duration` — histogram。

用這些來做不需要每個 call 細節的 dashboard。

### AgentOps 層

AgentOps（創立於 2024 年）專精於 GenAI observability。它包裝熱門的框架（LangGraph、Pydantic AI、CrewAI），自動輸出 OTel span。如果你的技術堆疊使用受支援的框架，這很有用；否則使用手動 instrumentation。

```figure
t3-span-waterfall
```

## 實際使用 (Use It)

`code/main.py` 針對一個呼叫 LLM、dispatch 兩個 tool、並進行一次 MCP 往返的 agent，把 OTel 形狀的 span 輸出到 stdout（類似 OTLP-JSON 的格式）。沒有真正的 exporter，這堂課聚焦在 span 的形狀與屬性集合。你可以把輸出貼到相容 OTLP 的檢視器，或直接閱讀。

觀察重點：

- 所有 span 共用同一個 trace id。
- 父子連結透過 `parentSpanId` 編碼。
- 必要的 `gen_ai.*` 屬性都有填入。
- 內容擷取預設關閉；其中一個情境透過環境變數開啟它。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-otel-genai-instrumentation.md`。給定一個 agent 程式碼庫，這個 skill 會產出 instrumentation 計畫：在哪裡加 span、要填入哪些屬性，以及要以哪些 exporter 為目標。

## 練習 (Exercises)

1. 執行 `code/main.py`。計算 span 的數量，並指出哪些是 CLIENT、哪些是 INTERNAL。

2. 開啟內容擷取（環境變數），並確認 `gen_ai.content.prompt` 和 `gen_ai.content.completion` 事件出現。注意這對 PII 的影響。

3. 加入 tool 執行指標 `gen_ai.tool.execution.duration`，並為每個 call 輸出一個 histogram 樣本。

4. 把父 agent span 的 traceparent 傳播到 MCP 請求的 `_meta.traceparent` 欄位。驗證 MCP server 會看到相同的 trace id。

5. 閱讀 OTel GenAI semconv 規格。找出 semconv 中列出、但這堂課的程式碼沒有輸出的一個屬性。並把它加進去。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| OTel | 「OpenTelemetry」 | 用於 trace、指標、log 的開放標準 |
| GenAI semconv | 「GenAI 語意慣例」 | LLM / tool / agent span 的穩定屬性名稱 |
| `gen_ai.*` | 「屬性 namespace」 | 所有 GenAI 屬性共用的前綴 |
| Span | 「有時間的操作」 | 有開始、結束與屬性的工作單元 |
| Trace | 「跨 span 的祖先關係」 | 共用 trace id 的 span 樹 |
| SpanKind | 「CLIENT / SERVER / INTERNAL」 | 關於 span 方向的提示 |
| OTLP | 「OpenTelemetry Protocol」 | exporter 使用的 wire format |
| Opt-in content | 「Prompt / completion 擷取」 | 預設關閉；用環境變數啟用 |
| traceparent | 「W3C header」 | 跨服務傳播 trace context |
| Exporter | 「特定 backend 的傳送器」 | 把 span 送到 Jaeger / Datadog 等的元件 |

## 延伸閱讀 (Further Reading)

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — GenAI span、指標與事件的標準慣例
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — LLM 與 tool 執行 span 的屬性清單
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent 層級的 `invoke_agent` span
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — GitHub 上的權威來源
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — production 整合的逐步說明
