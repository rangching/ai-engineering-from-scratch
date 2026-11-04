# Agent 的 Actor Model：非同步訊息與有型別的 Runtime (The Actor Model for Agents — Async Messages and Typed Runtimes)

> Agent 即 actor：非同步訊息交換、事件驅動的 handler、故障隔離、天然並行。AutoGen v0.4（Microsoft Research, 2025 年 1 月）圍繞這個模型重新設計了 agent 編排；這個 framework 現在處於維護模式，由 Microsoft Agent Framework（2025 年 10 月公開預覽）接手作為正式環境的後繼者。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 actor model：agent 即 actor、訊息是唯一的 IPC、每個 actor 各自隔離故障。
- 說出 AutoGen v0.4 的三個 API 層：Core、AgentChat、Extensions，以及各自的用途。
- 解釋為什麼把訊息傳遞與處理解耦，能帶來故障隔離和天然並行。
- 用 stdlib 在 Python 中實作 actor runtime，並把一個雙 agent 的 code-review 流程移植上去。

## 問題 (The Problem)

大多數 agent framework 是同步的：一個 agent 生產，另一個 agent 消費，都在同一個 call stack 裡。失敗會讓 stack 崩潰。並行是後來硬加上去的。要分散式部署就得重寫。

AutoGen v0.4 的答案：actor model。每個 agent 是一個有私有 inbox 的 actor。訊息是唯一的互動方式。Runtime 把傳遞與處理解耦。失敗隔離在單一 actor 內。並行是原生的。分散式只是換一種 transport。

## 核心概念 (The Concept)

### Actor

Actor 具有：

- 私有 state（絕不從外部直接碰觸）。
- Inbox（訊息佇列）。
- Handler：`receive(message) -> effects`，其中 effects 可以是「回覆」、「傳給其他 actor」、「產生新 actor」、「更新 state」、「停止自己」。

兩個 actor 不能共用記憶體。它們只能傳送訊息。

### 三個 API 層 (Three API layers)

AutoGen v0.4 把它的介面分成三層：

1. **Core。** 低階 actor framework。`AgentRuntime`、`Agent`、`Message`、`Topic`。非同步訊息交換，事件驅動。
2. **AgentChat。** 以任務為導向的高階 API（取代 v0.2 的 ConversableAgent）。`AssistantAgent`、`UserProxyAgent`、`RoundRobinGroupChat`、`SelectorGroupChat`。
3. **Extensions。** 整合：OpenAI、Anthropic、Azure、tool、memory。

### 為什麼解耦很重要 (Why decoupling matters)

在 v0.2 模型中，同步呼叫 `agent_a.chat(agent_b)` 會阻塞 agent_a，直到 agent_b 回傳。在 v0.4 中，`send(agent_b, msg)` 把訊息放進 agent_b 的 inbox 就返回。Runtime 稍後才傳遞。三個結果：

- **故障隔離。** Agent B 當機不會讓 Agent A 當機。Runtime 在 B 的 handler 中攔截失敗，並決定怎麼處理（log、重試、dead-letter）。
- **天然並行。** 同時有很多訊息在途中；actor 並行處理自己的 inbox。
- **便於分散式。** 不論 actor 在行程內還是在另一台主機上，inbox + transport 都是同一個抽象。

### 拓撲 (Topologies)

- **RoundRobinGroupChat。** Agent 依固定輪替順序輪流發言。
- **SelectorGroupChat。** 由 selector agent 根據對話 context 挑選下一位發言者。
- **Magentic-One。** 用於瀏覽網頁、執行程式碼、處理檔案的參考 multi-agent 團隊。建立在 AgentChat 之上。

### Observability

內建 OpenTelemetry 支援。每則訊息都會發出一個 span；tool call 依 2026 年 OTel GenAI semantic convention（Lesson 23）帶有 `gen_ai.*` 屬性。

### 狀態：維護模式 (Status: maintenance mode)

2026 年初：AutoGen v0.7.x 在研究與原型開發上穩定。Microsoft 已把積極開發轉移到 Microsoft Agent Framework，也就是正式環境的後繼者（2025 年 10 月 1 日公開預覽；1.0 GA 目標是 2026 年第一季末）。AutoGen 的模式可以順利往前移植，actor model 才是持久的想法。

```figure
actor-mailbox
```

## 動手做 (Build It)

`code/main.py` 實作 stdlib 的 actor runtime：

- `Message`：有型別的 payload，含 `sender`、`recipient`、`topic`、`body`。
- `Actor`：抽象類別，含 `receive(message, runtime)`。
- `Runtime`：帶共用佇列、傳遞與故障隔離的事件迴圈。
- 雙 actor 示範：`ReviewerAgent` 審查程式碼，`ChecklistAgent` 執行檢查清單；它們交換訊息直到達成共識。

執行：

```
python3 code/main.py
```

Trace 顯示訊息傳遞、其中一個 actor 的模擬失敗並未讓另一個 actor 當機，以及最後收斂到共同的結論。

## 實際使用 (Use It)

- **AutoGen v0.4/v0.7**（維護中）：在研究、原型開發和 multi-agent 模式上穩定。
- **Microsoft Agent Framework**：正式環境的後繼者（2025 年 10 月公開預覽）；同樣的 actor-model 想法，搭配更新的 API。
- **LangGraph swarm 拓撲**（Lesson 13）：透過共用 tool 交接達成的類似模式。
- **自訂 actor runtime**：當你需要特定的 transport（NATS、RabbitMQ、gRPC）時。

## 交付 (Ship It)

`outputs/skill-actor-runtime.md` 針對指定的 multi-agent 任務，產生最小的 actor runtime 加上團隊模板（RoundRobin 或 Selector）。

## 練習 (Exercises)

1. 加一個 dead-letter queue：當 handler 拋出例外時，把失敗的訊息暫存起來供人工檢視。你的 toy 中 DLQ 被觸發的頻率有多高？
2. 實作 `SelectorGroupChat`：由 selector actor 根據對話狀態，挑選誰來處理下一則訊息。
3. 加上分散式 transport：把行程內佇列換成 JSON-over-HTTP 伺服器，讓 actor 可以在不同行程中執行。
4. 為每則訊息接上一個 OTel span（或無操作的替代品）。依 Lesson 23 輸出 `gen_ai.agent.name`、`gen_ai.operation.name`。
5. 讀 AutoGen v0.4 的架構文章。把你的 toy 移植到真正的 `autogen_core` API。你略過了哪些在正式環境中重要的東西？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Actor | 「Agent」 | 私有 state + inbox + handler；不共用記憶體 |
| Message | 「事件」 | 有型別的 payload；actor 互動的唯一方式 |
| Inbox | 「信箱」 | 每個 actor 待處理訊息的佇列 |
| Runtime | 「Agent 主機」 | 路由訊息並隔離故障的事件迴圈 |
| Topic | 「頻道」 | actor 之間具名的 publish-subscribe 路由 |
| Fault isolation | 「Let it crash」 | 一個 actor 失敗不會讓其他 actor 當機 |
| RoundRobinGroupChat | 「固定輪替的團隊」 | Agent 依序輪流發言 |
| SelectorGroupChat | 「依 context 路由的團隊」 | 由 selector 挑選下一位發言者 |
| Magentic-One | 「參考團隊」 | 處理 web + 程式碼 + 檔案的 multi-agent 小隊 |

## 延伸閱讀 (Further Reading)

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：重新設計的文章
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：graph 形狀的替代方案
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)：AutoGen 預設發出的 span
