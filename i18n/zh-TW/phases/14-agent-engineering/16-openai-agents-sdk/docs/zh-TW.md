# OpenAI Agents SDK：Handoff、Guardrail、Tracing (OpenAI Agents SDK: Handoffs, Guardrails, Tracing)

> OpenAI Agents SDK 是建立在 Responses API 之上的輕量 multi-agent framework。五個原語：Agent、Handoff、Guardrail、Session、Tracing。Handoff 是名為 `transfer_to_<agent>` 的 tool。Guardrail 在輸入或輸出時觸發。Tracing 預設開啟。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 06 (Tool Use)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 OpenAI Agents SDK 的五個原語。
- 解釋 handoff：為什麼把它建模成 tool、模型看到的名稱形式，以及 context 如何轉移。
- 區分 input guardrail、output guardrail 和 tool guardrail；解釋 `run_in_parallel` 與 blocking 模式。
- 用 stdlib 實作 runtime，包含 handoff + guardrail + span 風格的 tracing。

## 問題 (The Problem)

無法乾淨委派的 agent，最後會把所有東西都塞進一個 prompt。沒有 guardrail 的 agent，會外洩 PII、產出違反政策的內容，或無限 loop。OpenAI 的 SDK 把讓 multi-agent 工作變得可行的三個原語寫成了規範。

## 核心概念 (The Concept)

### 五個原語 (Five primitives)

1. **Agent。** LLM + instructions + tools + handoffs。
2. **Handoff。** 委派給另一個 agent。對模型而言，表現為名為 `transfer_to_<agent_name>` 的 tool。
3. **Guardrail。** 對輸入（僅第一個 agent）、輸出（僅最後一個 agent）或 tool 呼叫（每個 function tool）做驗證。
4. **Session。** 跨 turn 自動保存對話歷史。
5. **Tracing。** 內建針對 LLM 生成、tool call、handoff、guardrail 的 span。

### 把 handoff 當作 tool (Handoffs as tools)

模型在 tool 清單中看到 `transfer_to_billing_agent`。呼叫它會通知 runtime：

1. 複製對話 context（或透過 `nest_handoff_history` beta 將其折疊）。
2. 用目標 agent 的 instructions 初始化它。
3. 以目標 agent 繼續這次執行。

這就是產品化的 supervisor 模式（Lesson 13 / Lesson 28）。

### Guardrail

三種類型：

- **Input guardrail。** 對第一個 agent 的輸入執行。在任何 LLM 呼叫之前就拒絕不安全或超出範圍的請求。
- **Output guardrail。** 對最後一個 agent 的輸出執行。抓出 PII 外洩、政策違規、格式錯誤的回應。
- **Tool guardrail。** 對每個 function tool 執行。驗證參數、檢查權限、稽核執行。

模式：

- **Parallel**（預設）。Guardrail LLM 與主要 LLM 並行執行。尾端延遲較低。如果被觸發，主要 LLM 的工作會被丟棄（浪費 token）。
- **Blocking**（`run_in_parallel=False`）。Guardrail LLM 先執行。如果被觸發，主要呼叫不會浪費 token。

Tripwire 會拋出 `InputGuardrailTripwireTriggered` / `OutputGuardrailTripwireTriggered`。

### Tracing

預設開啟。每次 LLM 生成、tool call、handoff 和 guardrail 都會發出 span。`OPENAI_AGENTS_DISABLE_TRACING=1` 可以關閉。`add_trace_processor(processor)` 會把 span 同時分送到你自己的後端和 OpenAI 的後端。

### Session

`Session` 把對話歷史儲存在後端（SQLite、Redis、自訂）。`Runner.run(agent, input, session=session)` 會自動載入並附加。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Handoff 漂移 (Handoff drift)。** Agent A 交給 Agent B，B 又交回 Agent A。加上 hop 計數器。
- **Guardrail 被繞過 (Guardrail bypass)。** Tool guardrail 只對 function tool 觸發；內建 tool（檔案讀取、web fetch）需要另外的政策。
- **過度 tracing (Over-tracing)。** span 中含有敏感內容。搭配 OTel GenAI 內容擷取規則（Lesson 23）：存在外部，用 ID 參照。

```figure
ae-agent-handoff
```

## 動手做 (Build It)

`code/main.py` 用 stdlib 實作 SDK 的形態：

- `Agent`、`FunctionTool`、`Handoff`（作為帶有轉移語意的 function tool）。
- `Runner`：含 input/output/tool guardrail、handoff 分派和 hop 計數器。
- 一個簡單的 span 發送器，用來展示 trace 的形狀。
- 一個 triage agent，依使用者的查詢交接給 billing 或 support；guardrail 會在其中一個輸入上被觸發。

執行：

```
python3 code/main.py
```

Trace 顯示兩次成功的 handoff、一次 input guardrail 觸發，以及與真正 SDK 所發出的相符的 span 樹。

## 實際使用 (Use It)

- **OpenAI Agents SDK** 用於以 OpenAI 為優先的產品。
- **Claude Agent SDK**（Lesson 17）用於以 Claude 為優先的產品。
- **LangGraph**（Lesson 13）用於你想要明確 state 與 durable resume 時。
- **自行打造**：當你需要精確控制時（語音、多 provider、聯邦式部署）。

## 交付 (Ship It)

`outputs/skill-agents-sdk-scaffold.md` 建立 Agents SDK 應用程式的骨架，包含 triage agent、handoff、input/output/tool guardrail、session store 和 trace processor。

## 練習 (Exercises)

1. 加上 handoff hop 計數器：N 次轉移後就拒絕。追蹤這個行為。
2. 實作 `nest_handoff_history` 作為選項：轉移之前，把先前的訊息折疊成一份摘要。
3. 寫一個 blocking output guardrail。比較會觸發它的 prompt 與通過的 prompt 的延遲。
4. 把 `add_trace_processor` 接到 JSON logger。每個 span 輸出什麼形狀？
5. 讀 SDK 文件。把你的 stdlib toy 移植到 `openai-agents-python`。你哪裡建模錯了？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Agent | 「LLM + instructions」 | SDK 中的 Agent 型別；擁有 tool 與 handoff |
| Handoff | 「轉移」 | 模型為了委派給另一個 agent 而呼叫的 tool |
| Guardrail | 「政策檢查」 | 對輸入 / 輸出 / tool 呼叫的驗證 |
| Tripwire | 「Guardrail 觸發」 | Guardrail 拒絕時拋出的例外 |
| Session | 「歷史儲存」 | 在多次執行之間持久化的對話 memory |
| Tracing | 「Span」 | 涵蓋 LLM + tool + handoff + guardrail 的內建 observability |
| Blocking guardrail | 「循序檢查」 | Guardrail 先執行；觸發時不浪費 token |
| Parallel guardrail | 「並行檢查」 | Guardrail 並行執行；延遲較低，觸發時浪費 token |

## 延伸閱讀 (Further Reading)

- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：原語、handoff、guardrail、tracing
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude 風格的對應
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：究竟何時該使用 handoff
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：Agents SDK 的 span 所對應的標準
