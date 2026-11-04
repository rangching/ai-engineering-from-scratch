# Agent 狀態機 — Graph、Node、Checkpoint (Agent State Machines)

> 手寫的 ReAct 迴圈就是一個 `while True`。同樣的迴圈如果寫成明確的 graph，你就可以對它做 checkpoint、中斷、分支，甚至時光回溯。Agent 沒有變，變的是它外圍的 harness。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 14 (Model Context Protocol)
**Time:** ~75 分鐘

## 問題 (The Problem)

你上線了一個 function-calling agent。它跑了三輪都正常，然後出事了：模型呼叫的 tool 回傳 500、使用者在任務進行到一半時改變主意，或是 agent 在沒有人簽核的情況下決定退款。`while True:` 迴圈沒有任何 hook。你不能暫停它、不能倒帶，也不能分支出去看「如果模型選了另一個 tool 會怎樣」。一旦你把它推出 demo 階段，這個 agent 就會變成黑盒子，只有「成功」或「失敗」兩種結果。

一旦你看清楚，下一步就很明顯了。Agent 本來就是一個狀態機 — system prompt 加上訊息歷史、待處理的 tool call，以及下一個動作。把這個狀態機明確化：為「模型思考」、「tool 執行」、「人類核准」建立 node，並為它們之間的條件轉換建立 edge。一旦 graph 明確了，harness 就免費得到四樣東西：checkpoint（在步驟之間儲存狀態）、interrupt（為人類而暫停）、streaming（串流 token 與中間事件），以及 time-travel（倒回先前的狀態並嘗試不同的分支）。

這個抽象的參考實作是 LangGraph。它不是 LangChain 那種意義上的 agent 框架（「這裡有個 AgentExecutor，祝你好運」）。它是一個 graph runtime，具備一等公民的 state、一等公民的持久化，以及一等公民的 interrupt。Agent 迴圈是你畫出來的，而不是手寫出來的。

## 核心概念 (The Concept)

![LangGraph StateGraph: nodes, edges, and the checkpointer](../assets/langgraph-stategraph.svg)

`StateGraph` 有三樣東西。

1. **State。**一個在 graph 中流動的具型別字典（TypedDict 或 Pydantic model）。每個 node 接收完整的 state，並回傳部分的更新，LangGraph 會用每個欄位各自的 *reducer* 來合併 — 需要累積的 list 使用 `operator.add`，預設則是覆寫。
2. **Node。**Python 函式 `state -> partial_state`。每個都是離散的步驟：「呼叫模型」、「執行 tool」、「摘要」。
3. **Edge。**Node 之間的轉換。靜態 edge 只通往一個地方。條件式 edge 接受一個路由函式 `state -> next_node_name`，讓 graph 可以依據模型輸出進行分支。

你編譯 (compile) 這個 graph。Compile 會綁定拓樸結構、附加 checkpointer（可選，但對 production 而言必不可少），並回傳一個可執行物件。你用初始 state 和 `thread_id` 來呼叫它。每個執行步驟都會把 checkpoint 持久化，以 `(thread_id, checkpoint_id)` 為 key。

### 四個超能力 (The four superpowers)

**Checkpointing。**每次 node 轉換都會把新的 state 寫入儲存體（測試用記憶體內、production 用 Postgres/Redis/SQLite）。用同一個 `thread_id` 再次呼叫 graph 就能恢復。Graph 會從暫停的地方接著繼續。

**Interrupts。**用 `interrupt_before=["human_review"]` 標記某個 node，執行就會在那個 node 執行之前停止。State 會被持久化。你的 API 回應使用者「等待核准中」。之後對同一個 `thread_id` 帶著 `Command(resume=...)` 發出的請求，會恢復執行。

**Streaming。**`graph.stream(state, mode="updates")` 會在發生時產出 state 的差異 (delta)。`mode="messages"` 串流 model node 內的 LLM token。`mode="values"` 產出完整的快照。你可以挑選要在 UI 中呈現什麼。

**Time-travel。**`graph.get_state_history(thread_id)` 回傳完整的 checkpoint 紀錄。把任何先前的 `checkpoint_id` 傳給 `graph.invoke`，你就能從那個點分叉。適合除錯（「如果模型選了 tool B 會怎樣？」）以及重播 production trace 的 regression test。

### Reducer 才是重點 (Reducers are the point)

每個 state 欄位都有一個 reducer。大多數預設值都沒問題 — 新值覆蓋舊值。但訊息 list 需要 `operator.add`，讓新訊息附加而不是取代。平行的 edge 透過 reducer 合併它們的更新。如果兩個 node 都更新 `messages`，而你忘了 `Annotated[list, add_messages]`，後者會無聲地勝出，你就會丟掉半個回合。Reducer 是這個函式庫中唯一微妙的地方；把它做對，其餘的就能組合起來。

### 四個 node 組成的 ReAct graph

Production 的 ReAct agent 是四個 node 和兩條 edge：

1. `agent` — 用目前的訊息歷史呼叫 LLM。回傳 assistant 訊息（可能包含 tool_calls）。
2. `tools` — 執行最後一則 assistant 訊息中的所有 tool_calls，並把 tool 結果附加為 tool 訊息。
3. 一條從 `agent` 出發的條件式 edge：如果最後一則訊息有 tool_calls 就導向 `tools`，否則導向 `END`。
4. 一條從 `tools` 回到 `agent` 的靜態 edge。

就這樣。你用大約 40 行程式碼，就得到完整的 ReAct 迴圈（Thought → Action → Observation → Thought → …），同時具備 checkpointing、interrupt 與 streaming。

### StateGraph 與 Send（fanout）

`Send(node_name, state)` 讓 node 派發平行的 subgraph。範例：agent 決定同時查詢三個 retriever。每個 `Send` 會產生目標 node 的一次平行執行；它們的輸出透過 state reducer 合併。這就是 LangGraph 在不使用 threading 基本元件的情況下，表達 orchestrator-workers 模式的方式。

### Subgraph

編譯後的 graph 可以成為另一個 graph 中的 node。外層 graph 看到的是單一 node；內層 graph 有自己的 state 和自己的 checkpoint。團隊就是這樣建構 supervisor-worker agent：supervisor graph 把使用者意圖路由到各領域的 worker subgraph。

```figure
l5-state-graph-ledger
```

## 動手做 (Build It)

### 步驟 1：state 與 node

```python
from typing import Annotated, TypedDict
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage
from langgraph.graph import StateGraph, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver

class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]

def agent_node(state: State) -> dict:
    response = llm.invoke(state["messages"])
    return {"messages": [response]}

def should_continue(state: State) -> str:
    last = state["messages"][-1]
    return "tools" if getattr(last, "tool_calls", None) else END

tool_node = ToolNode(tools=[search_web, read_file])

graph = StateGraph(State)
graph.add_node("agent", agent_node)
graph.add_node("tools", tool_node)
graph.set_entry_point("agent")
graph.add_conditional_edges("agent", should_continue, {"tools": "tools", END: END})
graph.add_edge("tools", "agent")

app = graph.compile(checkpointer=MemorySaver())
```

`add_messages` 是讓訊息 list 累積而不是覆寫的 reducer。忘了它是 LangGraph 最常見的 bug。

### 步驟 2：搭配 thread 執行

```python
config = {"configurable": {"thread_id": "user-42"}}
for event in app.stream(
    {"messages": [HumanMessage("find the Anthropic headquarters address")]},
    config,
    stream_mode="updates",
):
    print(event)
```

每個更新都是一個字典 `{node_name: state_delta}`。你的前端可以把這些串流給 UI，讓使用者看到「agent 正在思考… 呼叫 search_web… 取得結果… 回答中」。

### 步驟 3：加入 human-in-the-loop 的 interrupt

標記一個 node，讓執行在它執行之前暫停。

```python
app = graph.compile(
    checkpointer=MemorySaver(),
    interrupt_before=["tools"],  # pause before every tool call
)

state = app.invoke({"messages": [HumanMessage("delete the production database")]}, config)
# state["__interrupt__"] is set. Inspect proposed tool calls.
# If approved:
from langgraph.types import Command
app.invoke(Command(resume=True), config)
# If denied: write a rejection message and resume
app.update_state(config, {"messages": [AIMessage("Blocked by human reviewer.")]})
```

State、checkpoint 與 thread 都會在 interrupt 期間持續保存。除了執行當下，沒有任何東西留在記憶體中。

### 步驟 4：用 time-travel 除錯

```python
history = list(app.get_state_history(config))
for snapshot in history:
    print(snapshot.values["messages"][-1].content[:80], snapshot.config)

# Fork from a prior checkpoint
target = history[3].config  # three steps back
for event in app.stream(None, target, stream_mode="values"):
    pass  # replay from that point forward
```

傳入 `None` 作為輸入，會從指定的 checkpoint 重播；傳入一個值，則會在恢復之前，把它作為更新附加到該 checkpoint 的 state。這就是你不用重跑整段對話，就能重現一次糟糕的 agent 執行的方法。

### 步驟 5：為 production 更換 checkpointer

```python
from langgraph.checkpoint.postgres import PostgresSaver

with PostgresSaver.from_conn_string("postgresql://...") as checkpointer:
    checkpointer.setup()
    app = graph.compile(checkpointer=checkpointer)
```

SQLite、Redis 與 Postgres 都有提供。`MemorySaver` 是給測試用的。任何要跨重啟持久保存的情況，都需要真正的儲存體。

## 技能 (The Skill)

> 你把 agent 建構成 graph，而不是 `while True` 迴圈。

在你伸手去拿 LangGraph 之前，先做 60 秒的設計：

1. **為 node 命名。**每個離散的決策或有副作用的動作都是一個 node。「Agent 思考」、「tool 執行」、「reviewer 核准」、「回應串流」。如果你列不出它們，代表這個任務還不是 agent 的形狀。
2. **宣告 state。**精簡的 TypedDict，每個 list 欄位都有 reducer。不要把所有東西都塞進 `messages`；把任務專屬的欄位（工作中的 `plan`、`budget` 計數器、`retrieved_docs` 清單）提升到最上層。
3. **畫出 edge。**除非下一步取決於模型輸出，否則使用靜態 edge。每個條件式 edge 都需要有具名分支的路由函式。
4. **事先選好 checkpointer。**測試用 `MemorySaver`，其他一律用 Postgres/Redis/SQLite。沒有 checkpointer 就不要上線 — 沒有 checkpointer 就沒有 resume、沒有 interrupt、沒有 time-travel。
5. **在 tool 執行之前決定 interrupt，而不是之後。**核准放在通往有副作用 node 的 edge 上，這樣你可以在造成傷害之前取消；驗證放在離開模型的 edge 上，這樣你可以低成本地拒絕不好的呼叫。
6. **預設啟用 streaming。**UI 用 `mode="updates"`，model node 內的 token 層級串流用 `mode="messages"`，eval 期間的完整快照用 `mode="values"`。

拒絕上線沒有 checkpointer 的 LangGraph agent。拒絕上線在副作用*之後*才 interrupt 的 agent。拒絕上線沒有以 `add_messages` 作為 reducer 的 `messages` 欄位。

## 練習 (Exercises)

1. **簡單。**用計算機 tool 與網頁搜尋 tool，實作上面四個 node 的 ReAct graph。驗證對於兩輪的對話，`list(app.get_state_history(config))` 回傳至少四個 checkpoint。
2. **中等。**加入一個在 `agent` 之前執行的 `planner` node，並把結構化的 `plan: list[str]` 寫入 state。讓 `agent` 把 plan 的步驟標記為完成。如果 `plan` 在 checkpoint 恢復後遺失（reducer 錯誤），就讓測試失敗。
3. **困難。**建立 supervisor graph，使用 `Send` 在三個 subgraph（`researcher`、`writer`、`reviewer`）之間路由。每個 subgraph 都有自己的 state 和 checkpointer。在外層 graph 加上 `interrupt_before=["writer"]`，讓人類能核准研究簡報。確認從先前的 checkpoint 做 time-travel，只會重新執行被分叉的分支。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| StateGraph | 「LangGraph 的 graph」 | 在 compile 之前，你用來新增 node 和 edge 的 builder 物件。 |
| Reducer | 「欄位怎麼合併」 | 當 node 為某個欄位回傳更新時所套用的函式 `(old, new) -> merged`；預設是覆寫，`add_messages` 則是附加。 |
| Thread | 「對話 ID」 | 一個 `thread_id` 字串，作為單一 session 所有 checkpoint 的範圍。 |
| Checkpoint | 「暫停的狀態」 | 在 node 轉換之後，對完整 graph state 的持久化快照，以 `(thread_id, checkpoint_id)` 為 key。 |
| Interrupt | 「為人類暫停」 | `interrupt_before` / `interrupt_after` 在 node 邊界停止執行；用 `Command(resume=...)` 恢復。 |
| Time-travel | 「從先前的步驟分叉」 | `graph.invoke(None, config_with_old_checkpoint_id)` 從該 checkpoint 往前重播。 |
| Send | 「平行 subgraph 派發」 | node 可以回傳的一個建構子，用來產生目標 node 的 N 次平行執行。 |
| Subgraph | 「作為 node 的已編譯 graph」 | 在另一個 graph 中當作 node 使用的已編譯 StateGraph；保有自己的 state 範圍。 |

## 延伸閱讀 (Further Reading)

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、reducer、checkpointer 與 interrupt 的標準參考。
- [LangGraph concepts: state, reducers, checkpointers](https://langchain-ai.github.io/langgraph/concepts/low_level/) — 這堂課使用的心智模型，直接來自原始來源。
- [LangGraph Persistence and Checkpoints](https://langchain-ai.github.io/langgraph/concepts/persistence/) — 關於 Postgres/SQLite/Redis 儲存體、checkpoint namespace 與 thread ID 的細節。
- [LangGraph Human-in-the-loop](https://langchain-ai.github.io/langgraph/concepts/human_in_the_loop/) — `interrupt_before`、`interrupt_after`、`Command(resume=...)`，以及 edit-state 模式。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting in Language Models" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每個 LangGraph agent 所實作的模式；想了解推理軌跡的設計理由請讀它。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 該偏好哪些 graph 形狀（chain、router、orchestrator-workers、evaluator-optimizer），以及何時使用。
- Phase 11 · 09 (Function Calling) — 每個 LangGraph agent node 都會重複使用的 tool-call 基本元件。
- Phase 11 · 14 (Model Context Protocol) — 透過 MCP adapter 接上 LangGraph `ToolNode` 的外部 tool 發現機制。
- Phase 11 · 17 (Agent framework tradeoffs) — 何時該選 LangGraph 而不是 CrewAI、AutoGen 或 Agno。
