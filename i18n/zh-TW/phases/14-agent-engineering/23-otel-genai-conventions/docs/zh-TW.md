# OpenTelemetry GenAI Semantic Conventions (OpenTelemetry GenAI Semantic Conventions)

> OpenTelemetry 的 GenAI SIG（2024 年 4 月成立）定義了 agent telemetry 的標準 schema。Span 名稱、屬性和內容擷取規則在各廠商之間趨同，因此 agent trace 在 Datadog、Grafana、Jaeger 和 Honeycomb 中意義相同。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 24 (Observability Platforms)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 GenAI span 的類別：model/client、agent、tool。
- 區分 `invoke_agent` 的 CLIENT 與 INTERNAL span，以及各自適用的時機。
- 列出頂層的 GenAI 屬性：provider 名稱、request model、data-source ID。
- 解釋內容擷取的合約：opt-in、`OTEL_SEMCONV_STABILITY_OPT_IN`、建議使用外部參照。

## 問題 (The Problem)

每個廠商都發明自己的 span 名稱。維運團隊最後得為每個 framework 各建儀表板。OpenTelemetry 的 GenAI SIG 定義了一個整個生態系都以它為目標的標準來解決這件事。

## 核心概念 (The Concept)

### Span 類別 (Span categories)

1. **Model / client span。** 涵蓋原始的 LLM 呼叫。由 provider SDK（Anthropic、OpenAI、Bedrock）和 framework 的模型 adapter 發出。
2. **Agent span。** `create_agent`（建構 agent 時）和 `invoke_agent`（執行時）。
3. **Tool span。** 每次 tool 呼叫一個；透過父子關係連到 agent span。

### Agent span 命名 (Agent span naming)

- Span 名稱：有命名時為 `invoke_agent {gen_ai.agent.name}`；退而求其次為 `invoke_agent`。
- Span kind：
  - **CLIENT**：用於遠端 agent 服務（OpenAI Assistants API、Bedrock Agents）。
  - **INTERNAL**：用於行程內的 agent framework（LangChain、CrewAI、本地 ReAct）。

### 關鍵屬性 (Key attributes)

- `gen_ai.provider.name`：`anthropic`、`openai`、`aws.bedrock`、`google.vertex`。
- `gen_ai.request.model`：模型 ID。
- `gen_ai.response.model`：實際解析出的模型（可能因路由而與 request 不同）。
- `gen_ai.agent.name`：agent 識別碼。
- `gen_ai.operation.name`：`chat`、`completion`、`invoke_agent`、`tool_call`。
- `gen_ai.data_source.id`：用於 RAG：查詢了哪個語料庫或 store。

Anthropic、Azure AI Inference、AWS Bedrock、OpenAI 都有各自技術專屬的 convention。

### 內容擷取 (Content capture)

預設規則：instrumentation 預設不應該擷取輸入/輸出 (SHOULD NOT)。擷取是透過以下項目 opt-in：

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

建議的正式環境模式：把內容存在外部（S3、你的 log store），在 span 上記錄參照（指標 ID，而不是散文內容）。這就是接進 observability 的 Lesson 27 內容污染防禦。

### 穩定性 (Stability)

截至 2026 年 3 月，大多數 convention 仍屬實驗性質。用以下方式 opt in 到穩定預覽版：

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ 原生地將 GenAI 屬性對應到它的 LLM Observability schema。其他後端（Grafana、Honeycomb、Jaeger）支援原始屬性。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **在 span 中擷取完整 prompt。** 維運人員可讀的 trace 裡有 PII、機密、客戶資料。存放在外部。
- **沒有 `gen_ai.provider.name`。** 缺少歸屬時，多 provider 儀表板會壞掉。
- **Span 沒有父連結。** 變成孤立的 tool span。務必傳播 context。
- **沒有設定穩定性 opt-in。** 後端升級時，你的屬性可能被重新命名。

```figure
ae-genai-span-tree
```

## 動手做 (Build It)

`code/main.py` 實作符合 GenAI convention 的 stdlib span 發送器：

- 具有 GenAI 屬性 schema 的 `Span`。
- 具有 `start_span` 和巢狀 context 的 `Tracer`。
- 一次腳本化的 agent 執行，發出：`create_agent`、`invoke_agent`（INTERNAL）、每個 tool 的 span、LLM 呼叫的 `chat` span。
- 一個內容擷取模式，把 prompt 存在外部，並在 span 上記錄 ID。

執行：

```
python3 code/main.py
```

輸出：一棵具有所有必要 GenAI 屬性的 span 樹，以及一個顯示 opt-in 內容參照的「外部 store」。

## 實際使用 (Use It)

- **Datadog LLM Observability**（v1.37+）原生對應屬性。
- **Langfuse / Phoenix / Opik**（Lesson 24）：自動為生態系做 instrumentation。
- **Jaeger / Honeycomb / Grafana Tempo**：原始 OTel trace；用 GenAI 屬性建儀表板。
- **自架**：執行帶有 GenAI processor 的 OTel Collector。

## 交付 (Ship It)

`outputs/skill-otel-genai.md` 把 OTel GenAI span 接進既有的 agent，並採用內容擷取預設值與外部參照儲存。

## 練習 (Exercises)

1. 用 `invoke_agent`（INTERNAL）+ 每個 tool 的 span，為你的 Lesson 01 ReAct loop 做 instrumentation。送到一個 Jaeger 實例。
2. 加入「僅參照」模式的內容擷取：prompt 存進 SQLite，span 屬性只帶 row ID。
3. 讀 `gen_ai.data_source.id` 的規格。把它接進你的 Lesson 09 Mem0 search。
4. 設定 `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental`，並驗證你的屬性不會被 collector 重新命名。
5. 建一個儀表板：只用 GenAI 屬性，呈現「哪些 tool 錯誤與哪些模型相關」。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| GenAI SIG | 「OpenTelemetry GenAI 小組」 | 定義 schema 的 OTel 工作小組 |
| invoke_agent | 「Agent span」 | 代表一次 agent 執行的 span 名稱 |
| CLIENT span | 「遠端呼叫」 | 呼叫遠端 agent 服務的 span |
| INTERNAL span | 「行程內」 | 行程內 agent 執行的 span |
| gen_ai.provider.name | 「Provider」 | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | 「RAG 來源」 | retrieval 命中的語料庫/store |
| Content capture | 「Prompt 記錄」 | 以 opt-in 方式擷取訊息；正式環境存在外部 |
| Stability opt-in | 「預覽模式」 | 用來固定實驗性 convention 的環境變數 |

## 延伸閱讀 (Further Reading)

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：規格
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)：預設就有 GenAI span
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：內建 OTel span
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview)：W3C trace context 傳播
