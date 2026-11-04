# 有狀態的 Graph 編排：Durable Execution 與 Checkpoint (Stateful Graph Orchestration — Durable Execution and Checkpoints)

> Agent 是一個狀態機；節點是函式；邊是轉換；狀態在每個節點之後做 checkpoint。從任何失敗點，在最後一次成功的 checkpoint 恢復。LangGraph 是這種低階有狀態編排模型在 2026 年的參考實作。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 描述 LangGraph 的核心模型：具型別狀態的狀態機、函式節點、條件邊，以及節點後的 checkpoint。
- 說出文件強調的四項能力：durable execution、streaming、human-in-the-loop、完整的 memory。
- 解釋 LangGraph 支援的三種編排拓撲：supervisor、peer-to-peer（swarm）、階層式（巢狀 subgraph）。
- 用 stdlib 實作有狀態 graph，包含具型別狀態、條件邊，以及 checkpoint/resume 週期。

## 問題 (The Problem)

Agent 和 workflow 有個共同的問題：當一個 40 步的執行在第 38 步失敗時，你想從第 38 步恢復，而不是從頭來過。次等的狀態模型，會讓維運人員圍繞著一個假設每次都是全新執行的函式庫，硬拼湊重試機制。

LangGraph 的設計答案：狀態是一級的具型別物件，變更是明確的，checkpoint 在每個節點之後持久化。恢復只是一次 `load_state(session_id)` 呼叫。

## 核心概念 (The Concept)

### Graph

Graph 由以下定義：

- **State 型別。** 每個節點讀取並變更的具型別 dict（或 Pydantic model）。
- **Node。** 純函式 `(state) -> state_update`。更新在回傳後被合併進 state。
- **Edge。** 節點之間的條件式或直接轉換。
- **進入與離開。** `START` 和 `END` 哨兵節點標示邊界。

範例：一個有 `classify`、`refund`、`bug`、`sales`、`done` 節點的 agent，也就是以 graph 形式表示的 routing workflow。

### Durable execution

每個節點回傳後，runtime 會序列化 state 並寫入 checkpointer（SQLite、Postgres、Redis、自訂）。在第 N 步失敗時，runtime 可以 `resume(session_id)`，用完全相同的 state 從第 N+1 步接續。

LangGraph 文件明確指出這一點重要的正式環境使用者：Klarna、Uber、J.P. Morgan。重點不在 graph 的形狀，而在於 graph 形狀加上 checkpoint，讓恢復的成本很低。

### Streaming

每個節點都可以 yield 部分輸出。Graph 會把逐節點的 delta 事件串流給呼叫端，讓 UI 隨著 graph 執行而更新。

### Human-in-the-loop

在節點之間檢視並修改 state。實作方式：在關鍵節點前暫停、把 state 呈現給人類、接受修改、然後恢復。Checkpointer 讓這件事變得容易，因為 state 本來就已經序列化了。

### Memory

短期（在一次執行內：state 中的對話歷史）與長期（跨執行：透過 checkpointer 加上獨立的長期 store 持久化）。LangGraph 透過 tool 整合外部 memory 系統（Mem0、自訂）。

### 三種拓撲 (Three topologies)

1. **Supervisor。** 中央 router LLM 把工作分派給專家 subagent。`langgraph-supervisor` 中的 `create_supervisor()`（不過 LangChain 團隊在 2026 年建議直接透過 tool call 來做，以便更好地控制 context）。
2. **Swarm / peer-to-peer。** Agent 透過共用的 tool 介面直接交接。沒有中央 router。
3. **階層式 (Hierarchical)。** Supervisor 管理子 supervisor，以巢狀 subgraph 實作。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Checkpoint 太小。** 只對對話 turn 做 checkpoint，會讓 tool 狀態和 memory 寫入無法恢復。完整的 state 都必須序列化。
- **非確定性的節點。** Resume 假設節點的輸入會產生相同的 state 更新。隨機 seed、wall-clock、外部 API 都必須被記錄下來。
- **過度使用條件邊。** 每條邊都是條件式的 graph，是一個無法推理的狀態機。偏好線性 chain，偶爾分支。

```figure
langgraph-state
```

## 動手做 (Build It)

`code/main.py` 實作 stdlib 的有狀態 graph：

- `State`：具型別的 dict，含 `messages`、`step`、`route`、`output`、`human_approval`。
- `Node`：接收 state 並回傳更新 dict 的 callable。
- `StateGraph`：節點 + 邊 + 條件邊 + run + resume。
- `SQLiteCheckpointer`（記憶體內的假實作）：每個節點後序列化 state；`load(session_id)` 負責還原。
- 示範 graph：classify -> branch(refund / bug / sales) -> human gate -> send。

執行：

```
python3 code/main.py
```

Trace 顯示第一次執行在 human gate 失敗、持久化，然後 resume 產生最終輸出。

## 實際使用 (Use It)

- **LangGraph**：參考實作，可用於正式環境。使用 `create_react_agent`、`create_supervisor`，或自己建 graph。
- **AutoGen v0.4**（Lesson 14）：適合高並行情境的 actor model 替代方案。
- **Claude Agent SDK**（Lesson 17）：內建 session store 的託管 harness。
- **自行打造**：當你需要精確控制 state 形狀或 checkpointer 後端時。

## 交付 (Ship It)

`outputs/skill-state-graph.md` 在任何目標 runtime 中產生 LangGraph 形狀的 state graph，並接好 checkpoint 與 resume。

## 練習 (Exercises)

1. 當分類信心低於門檻時，加一條從 `classify` 到 `end` 的條件邊。在人類手動設定 `route` 後恢復執行。
2. 把類似 SQLite 的假實作換成真正的 SQLite checkpointer。量測每一步的序列化額外開銷。
3. 實作平行邊：兩個節點同時執行，透過自訂 reducer 合併。不可變 state 在這裡帶來什麼好處？
4. 讀 `langgraph-supervisor` 的參考文件。把 toy 移植成 `create_supervisor`。比較 trace 的形狀。
5. 加入 streaming：每個節點在執行時 yield 部分 state。在 delta 到達時印出。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| State graph | 「Agent 即狀態機」 | 具型別 state + 節點 + 邊 + reducer |
| Checkpointer | 「持久化後端」 | 每個節點後序列化 state；讓 resume 成為可能 |
| Reducer | 「State 合併器」 | 把目前 state 與節點更新結合的函式 |
| Conditional edge | 「分支」 | 由 state 的函式選擇的邊 |
| Subgraph | 「巢狀 graph」 | 在另一個 graph 內部當作節點使用的 graph |
| Durable execution | 「從失敗恢復」 | 用完全相同的 state，從最後一個成功的節點重新開始 |
| Supervisor | 「Router LLM」 | 專家 subagent 的中央調度器 |
| Swarm | 「P2P agent」 | Agent 透過共用 tool 交接；沒有中央 router |

## 延伸閱讀 (Further Reading)

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：參考文件
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/)：supervisor 模式的 API
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：actor-model 替代方案
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：session store 與 subagent
