# Agent 框架的取捨 — Graph、Role 與 Actor 編排 (Agent Framework Tradeoffs)

> 每個框架賣的都是同一個 demo（research agent 建出一份報告），也藏著同一個 bug（state schema 與編排層互相打架）。請挑選抽象與你問題形狀相符的框架；其他一切都是你得寫兩遍的黏合程式碼。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 16 (LangGraph)
**Time:** ~45 分鐘

## 問題 (The Problem)

你有一個需要不只一次 LLM 呼叫的任務。也許是研究工作流程（規劃、搜尋、摘要、引用）。也許是 code review pipeline（解析 diff、批評、修補、驗證）。也許是一個多輪的助理，負責訂機票、寫 email 和提交費用報銷。你挑了一個框架。

三天後，你發現框架的抽象漏了。CrewAI 給你角色，但當「researcher」需要把結構化的計畫交給「writer」時，它就跟你作對。AutoGen 給你 agent 之間的聊天，但沒有一等公民的 state，所以你的 checkpoint 就是一份對話 log 的 pickle。LangGraph 給你 state graph，但強迫你在還不知道 agent 會做什麼之前，就得為每個轉換命名。Agno 給你單一 agent 的抽象，一旦你想要 fan out 到三個並行的 worker，它就會大聲抱怨。

解法不是「挑最好的框架」，而是把框架的核心抽象，對上你問題的形狀。這堂課畫出這張地圖。

## 核心概念 (The Concept)

![Agent framework matrix: core abstraction vs problem shape](../assets/framework-matrix.svg)

四個框架主宰了 2026 年的版圖。它們的核心抽象並不相同。

| 框架 | 核心抽象 | 最適合 | 最不適合 |
|-----------|------------------|----------|-----------|
| **LangGraph** | `StateGraph` — 具型別的 state、node、條件式 edge、checkpointer。 | 具有明確 state 與 human-in-the-loop interrupt 的工作流程；需要 time-travel 除錯的 production agent。 | 拓樸未知、鬆散的角色驅動腦力激盪。 |
| **CrewAI** | `Crew` — 角色（目標、背景故事）、任務、流程（sequential 或 hierarchical）。 | 角色扮演或 persona 驅動、計畫簡短且為線性/階層式的工作流程。 | 超出 crew 回合歷史的任何有狀態需求；複雜的分支。 |
| **AutoGen** | `ConversableAgent` 配對 — 兩個以上的 agent 輪流發言，直到滿足結束條件。 | 多 agent *對話*（老師-學生、提案者-批評者、執行者-審查者），思考從聊天中浮現。 | 已知 DAG 的確定性工作流程；任何需要跨重啟持久 state 的情況。 |
| **Agno** | `Agent` — 單一 LLM + tool + 記憶，可組合成 team。 | 快速建構的單一 agent 與輕量 team；強大的多模態與內建儲存 driver。 | 具有自訂 reducer、深層且明確分支的 graph。 |

### 「抽象」到底是什麼意思 (What "abstraction" actually means)

框架的核心抽象，就是你在白板上推銷架構時畫出來的東西。

- **LangGraph** → 你畫一張 graph。Node 是步驟，edge 是轉換，而每個時間點的 state 物件都有型別。心智模型是狀態機。
- **CrewAI** → 你畫一張組織圖。每個角色都有職務說明，由經理分派任務。心智模型是一個小型的專家團隊。
- **AutoGen** → 你畫一個 Slack 私訊。兩個 agent 互傳訊息；如果需要主持人，就加入第三個。心智模型是聊天。
- **Agno** → 你畫一個單一的方框，上面掛著 tool。把方框並排放就是一個 team。心智模型是「電池全附的 agent」。

### State 的問題 (The state question)

State 是多數框架選擇在 production 中崩潰的地方。

- **LangGraph。**具型別的 state（`TypedDict` 或 Pydantic model）、每個欄位各自的 reducer、一等公民的 checkpointer（SQLite/Postgres/Redis）。Resume、interrupt 與 time-travel 都是免費的。*（見 Phase 11 · 16。）*
- **CrewAI。**State 透過 `context` 欄位以字串的形式在任務之間流動，或透過 `output_pydantic` 以結構化方式傳遞。開箱即用沒有持久的 per-crew 儲存；如果 crew 必須撐過重啟，你得自己外掛。
- **AutoGen。**State 就是聊天歷史和任何使用者定義的 `context`。對話逐字稿會持久保存；任意的工作流程 state 除非你寫 adapter，否則不會。
- **Agno。**內建儲存 driver（SQLite、Postgres、Mongo、Redis、DynamoDB），透過 `storage=` 附加到 `Agent` — 對話 session 與使用者記憶會自動持久保存。不是完整的 graph checkpointer，而是 session 儲存體。

### 分支的問題 (The branching question)

每個不簡單的 agent 都會分支。由誰決定分支很重要。

- **LangGraph** — 由你決定，透過條件式 edge。路由是一個帶有具名分支的 Python 函式。分支是已編譯 graph 中的一等公民；checkpointer 會記錄走了哪個分支。
- **CrewAI** — 在 hierarchical 模式下由經理決定；在 sequential 模式下由你在建構時決定。路由隱含在任務清單中；除了經理的 prompt 之外，沒有一等公民的「if」。
- **AutoGen** — 由 agent 透過聊天決定。分支是由下一個發言者自然浮現的。`GroupChatManager` 選擇下一位發言者；你可以手寫 `speaker_selection_method`，但預設是由 LLM 驅動。
- **Agno** — 由 agent 決定下一個要呼叫哪個 tool。Team 有 coordinator/router/collaborator 模式；超出這些的分支，由開發者負責。

### Observability 的問題 (The observability question)

- **LangGraph** — 透過 LangSmith 或任何 OTel exporter 提供 OpenTelemetry。每次 node 轉換都是一個 trace span；checkpoint 同時也是可重播的 trace。LangSmith 是官方的選項；Langfuse/Phoenix 也有 adapter。
- **CrewAI** — 自 2025 年底起提供一等公民的 OpenTelemetry；整合 Langfuse、Phoenix、Opik、AgentOps。
- **AutoGen** — 透過 `autogen-core` 整合 OpenTelemetry；AgentOps 和 Opik 有 connector。Tracing 的粒度是每則 agent 訊息，而不是每個 node。
- **Agno** — 內建 `monitoring=True` 旗標加上 OpenTelemetry exporter；與 Langfuse 緊密整合，用於 session trace。

### 成本與延遲 (Cost and latency)

四個框架都會增加每次呼叫的額外負擔（框架邏輯、驗證、序列化）。額外負擔由低到高的粗略順序：Agno ≈ LangGraph < CrewAI ≈ AutoGen。差異主要取決於框架額外做了多少 LLM 路由。CrewAI 的 hierarchical 經理要花 token 來決定下一個是誰；AutoGen 的 `GroupChatManager` 也一樣。LangGraph 只在你寫 `llm.invoke` 的地方花 token。Agno 的單一 agent 路徑很薄。

當每次執行的成本很重要時，請偏好明確的路由（LangGraph edge、AutoGen `speaker_selection_method`），而不是由 LLM 選擇的路由。

### 互通性 (Interoperability)

- **LangGraph** ↔ **LangChain** 的 tool、retriever、LLM。一等公民的 MCP adapter（tool 以 MCP server 的形式匯入）。
- **CrewAI** ↔ tool 繼承自 `BaseTool`；LangChain tool、LlamaIndex tool 與 MCP tool 都能接進來。透過 `allow_delegation=True` 做 crew 對 crew 的委派。
- **AutoGen** → `FunctionTool` 包裝任何 Python callable；有 MCP adapter。在 agent 對 agent 的模式上，與 AG2 生態系緊密耦合。
- **Agno** → `@tool` decorator 或 BaseTool 子類別；MCP adapter；tool 可以在 agent 與 team 之間共用。

## 技能 (The Skill)

> 你能用一句話解釋，為什麼某個框架適合某個 agent 問題。

建構前檢查清單：

1. **畫出形狀。**這是 graph（具型別的 state、具名的轉換）嗎？角色扮演（專家之間交接工作）嗎？聊天（agent 互相交談直到完成）嗎？還是帶有 tool 的單一 agent？
2. **決定誰來分支。**開發者決定的分支 → LangGraph。經理 agent 決定 → CrewAI hierarchical。聊天中浮現 → AutoGen。由 tool call 決定 → Agno。
3. **檢查 state 預算。**你需要從 checkpoint 恢復嗎？Time-travel 嗎？執行中途的人類 interrupt 嗎？如果需要，LangGraph 是預設選擇；Agno 的 session 涵蓋以對話為範圍的 state。
4. **檢查成本預算。**由 LLM 選擇的路由，每一輪都要多花 token。如果 agent 每天執行數千次，請偏好明確的路由。
5. **為框架的額外負擔編列預算。**每個框架都是多一個相依套件。如果任務只是兩次 LLM 呼叫加一個 tool，就寫 30 行純 Python；沒有任何框架比沒有框架更便宜。

在你能畫出 graph、組織圖、聊天或 agent 方框之前，拒絕去用框架。拒絕選擇那種為了你真正需要的東西，逼你和它的 state 模型對抗的框架。

## 決策矩陣 (The Decision Matrix)

| 問題形狀 | 偏好的框架 | 原因 |
|---------------|---------------------|-----|
| 具有型別 state、人類核准、長時間執行的工作流程 DAG | LangGraph | 一等公民的 state、checkpointer、interrupt、time-travel。 |
| 具有不同角色的研究/寫作 pipeline | CrewAI (sequential) 或 LangGraph subgraph | 在 CrewAI 中，每個任務一個角色很容易表達；當分支變複雜時，再用 LangGraph 擴展。 |
| 提案者-批評者或老師-學生的對話 | AutoGen | 兩個 agent 的聊天是它的原生形狀。 |
| 帶有 tool、session、記憶的單一 agent | Agno | 設定最精簡，內建儲存與記憶。 |
| 帶有 reducer 的數千個平行 fanout | LangGraph + `Send` | 唯一擁有一等公民平行派發 API 的框架。 |
| 快速原型，不想綁定框架 | 純 Python + 供應商 SDK | 沒有框架就是最快的框架。 |

```figure
l5-framework-fit
```

## 練習 (Exercises)

1. **簡單。**拿同一個任務 — 「研究 Anthropic 的總部，撰寫 200 字的簡報，並引用來源」 — 分別用 LangGraph（四個 node：plan、search、write、cite）和 CrewAI（三個角色：researcher、writer、editor）實作。回報每次執行的 token 成本與程式碼行數。
2. **中等。**用 AutoGen（researcher ↔ writer 聊天，editor 透過 `GroupChat` 加入）與 Agno（一個帶有 `search_tools` 和 `write_tools` 的單一 agent，加上 session 儲存體）建構同一個任務。依 (a) 每次執行的成本、(b) 當機後恢復的能力、(c) 在寫作步驟之前插入人類核准的能力，為這四種實作排名。
3. **困難。**建立決策樹腳本 `pick_framework.py`，它接收簡短的問題描述（JSON：`{has_typed_state, has_roles, has_dialogue, has_parallel_fanout, needs_resume}`），並回傳附帶一句話理由的建議。用你自己設計的六個案例驗證它。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Orchestration | 「agent 如何協調」 | 決定下一個執行哪個 node/角色/agent 的那一層。 |
| Durable state | 「重啟後恢復」 | 能撐過行程終止、附加在 checkpoint 或 session 儲存體上的 state。 |
| LLM-selected routing | 「讓模型決定」 | 由規劃者 LLM 在每一輪挑選下一步；彈性高，但每個決定都要花 token。 |
| Explicit routing | 「開發者決定」 | 由 Python 函式或靜態 edge 挑選下一步；便宜且可稽核。 |
| Crew | 「CrewAI 的團隊」 | 角色 + 任務 + 流程（sequential 或 hierarchical）綁定成單一可執行物件。 |
| GroupChat | 「AutoGen 的多 agent 聊天」 | 由 N 個 agent 組成、帶有發言者選擇器的受管理對話。 |
| Team (Agno) | 「Agno 的多 agent」 | 對一組 agent 採用 route / coordinate / collaborate 模式。 |
| StateGraph | 「LangGraph 的 graph」 | 具型別 state、node、條件式 edge、checkpointer 的抽象。 |

## 延伸閱讀 (Further Reading)

- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — StateGraph、checkpointer、interrupt、time-travel。
- [CrewAI documentation](https://docs.crewai.com/) — Crew、Flow、Agent、Task、Process。
- [AutoGen documentation](https://microsoft.github.io/autogen/) — ConversableAgent、GroupChat、team、tool。
- [Agno documentation](https://docs.agno.com/) — Agent、Team、Workflow、儲存、記憶。
- [Anthropic — Building effective agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — 與框架無關的模式庫（prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer）。
- [Yao et al., "ReAct: Synergizing Reasoning and Acting" (ICLR 2023)](https://arxiv.org/abs/2210.03629) — 每個框架都在包裝的那個迴圈。
- [Wu et al., "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation" (2023)](https://arxiv.org/abs/2308.08155) — AutoGen 的設計論文。
- [Park et al., "Generative Agents: Interactive Simulacra of Human Behavior" (UIST 2023)](https://arxiv.org/abs/2304.03442) — CrewAI 風格 persona 堆疊所建立在其上的角色扮演基礎。
- Phase 11 · 16 (LangGraph) — 這堂課拿來作為比較基準的框架。
- Phase 11 · 19 (Reflexion) — 一種能乾淨對應到 LangGraph、但對應到 CrewAI 卻很彆扭的模式。
- Phase 11 · 22 (Production observability) — 如何為你挑選的任何框架加上 instrument。
