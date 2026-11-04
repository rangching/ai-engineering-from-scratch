# Multi-Agent 原語模型 (The Multi-Agent Primitive Model)

> 四個原語，不多不少：agent、handoff、shared state、orchestrator。它們撐起一個四維的設計空間，而 2026 年上市的主要 multi-agent 框架（AutoGen、LangGraph、CrewAI、OpenAI Agents SDK、Microsoft Agent Framework）都只是這個空間裡的一個點。這堂課從零開始把它們建出來，用一個玩具系統跑過全部四個，再把每個主流框架對應到同樣的軸上，讓你讀任何新版本都只需要一段話。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 (Agent Engineering), Phase 16 · 01 (Why Multi-Agent)
**Time:** ~60 分鐘

## 問題 (Problem)

每隔六個月就有新的 multi-agent 框架推出。2023 年有 AutoGen。2024 年有 CrewAI。2024 年有 LangGraph 和 OpenAI Swarm。2025 年 4 月有 Google ADK。2026 年 2 月有 Microsoft Agent Framework RC。每則新聞稿都宣稱自己是「正確的抽象」。

如果你一個一個學，會學到累。API 長得都不一樣。文件對「agent」是什麼各說各話。一個框架把它的 shared memory 叫做「blackboard」，另一個叫「message pool」，第三個叫「StateGraph」。你開始懷疑這個領域只是在空轉。

其實不是。在行銷詞彙底下，這四個原語是穩定的。學一次，之後讀任何新框架都只要一段話。

## 核心概念 (Concept)

### 四個原語 (The four primitives)

1. **Agent** — 一個 system prompt 加上一份 tool 清單。無狀態；每次執行都從它的 system prompt 和目前的訊息歷史開始。
2. **Handoff** — 把控制權從一個 agent 結構化地轉給另一個。機制上，就是回傳新 agent 的 tool call，或是依條件走的圖邊 (graph edge)。
3. **Shared state** — 超過一個 agent 能讀取（有時能寫入）的任何資料結構。Message pool、blackboard、key-value store、vector memory。
4. **Orchestrator** — 決定下一個誰發言的角色。選項有：明確的圖（確定性）、由 LLM 擔任的 speaker-selector（軟性）、上一位發言者的 handoff 呼叫（OpenAI Swarm），或是在 queue 上運作的 scheduler（swarm 架構）。

這就是整個設計空間。每個框架對每條軸挑選預設值；其餘都只是表面語法。

### 每個 2026 框架如何對應 (How every 2026 framework maps to it)

| 框架 | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool 回傳 Agent | 由呼叫者自己處理 | LLM 的下一次 handoff 呼叫 |
| AutoGen v0.4 / AG2 | `ConversableAgent` | GroupChat 上的 speaker-selector | message pool | selector 函式（LLM 或 round-robin） |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | 串接起來的 Task 輸出 | manager LLM 或固定順序 |
| LangGraph | node 函式 | graph edge + 條件 | `StateGraph` reducer | 圖本身，確定性 |
| Microsoft Agent Framework | agent + orchestration pattern | 依 pattern 而定 | thread / context | 依 pattern 而定 |
| Google ADK | agent + A2A card | A2A task | A2A artifact | 由 host 決定 |

表面差異看起來很大。底層都一樣：同樣四個旋鈕。

### 為什麼這很重要 (Why this matters)

一旦你看見這些原語，框架比較就變成一張簡短的檢查清單：

- Orchestrator 是相信 LLM 來路由（Swarm），還是把路由固定寫在程式碼裡（LangGraph）？
- Shared state 是完整歷史（GroupChat），還是經過投影的（StateGraph reducer）？
- Agent 能修改彼此的 prompt（CrewAI manager），還是只能 handoff（Swarm）？

這三個問題就能回答 80% 的「哪個框架適合這個問題」。你不再去找「最好的 multi-agent 框架」，而是針對你真正在意的那條軸來設計。

### 無狀態的洞見 (The stateless insight)

除了 shared state 之外，每個原語都是無狀態的。Agent 是 (prompt, tools) 的函式。Handoff 是一次函式呼叫。Orchestrator 是一個 scheduler。**整個系統裡唯一有狀態的東西是 shared state。** 所有有趣的 bug 都住在那裡：記憶體汙染 (memory poisoning)（Lesson 15）、訊息順序、版本控制、寫入競爭。

隱藏 shared state 的框架（Swarm）把問題丟給呼叫者。把它集中管理的框架（LangGraph checkpoint、AutoGen pool）讓它可以被檢視，但會把協調成本轉嫁到 shared-state 的實作上。

### 單一原語的剖析 (Anatomy of a single primitive)

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

沒有記憶。沒有狀態。兩個 system prompt 和 tool 相同的 agent 可以互換。看起來像是每個 agent 專屬的狀態，其實都在 shared state 或 handoff protocol 裡。

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

有三種實作方式佔主導：

- **函式回傳 (Function return)** — tool 回傳下一個 agent。這是 OpenAI Swarm 的 pattern。Agent 把路由寫在自己的 tool schema 裡。
- **圖邊 (Graph edge)** — LangGraph。邊是宣告式的。LLM 產生一個值；由條件選出下一個 node。
- **Speaker selection** — AutoGen GroupChat。由 selector 函式（有時它本身就是一次 LLM 呼叫）讀取 pool，挑出下一個發言者。

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

最少是一份訊息清單。通常更多：結構化的 artifact（CrewAI Task 輸出）、有型別的 context（LangGraph reducer）、外部記憶體（MCP、vector DB）。

兩種拓撲：**完整 pool**（每個 agent 看得到每則訊息）和 **投影式 (projected)**（agent 看到依角色限定範圍的視圖）。完整 pool 簡單但擴展性差。投影式能擴展，但需要事先設計 schema。

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

四種類型：

- **靜態 (Static)** — 圖在建置時就固定（LangGraph 確定性、CrewAI Sequential）。
- **LLM 選擇 (LLM-selected)** — 由 LLM 讀取 pool 並挑出下一個發言者（AutoGen、CrewAI Hierarchical）。
- **Handoff 驅動 (Handoff-driven)** — 目前的 agent 呼叫 handoff tool 來決定（Swarm）。
- **Queue 驅動 (Queue-driven)** — worker 從共享的 queue 拉取工作；沒有明確的下一位發言者（swarm 架構、Matrix）。

### 框架之間真正不同的地方 (What changes between frameworks)

原語固定之後，剩下的設計決策是：

- **記憶體策略** — 暫存 vs 持久的 checkpoint（LangGraph checkpointer）。
- **安全邊界** — 誰能核准 handoff（human-in-the-loop）。
- **成本計算** — 每個 agent 的 token 預算。
- **可觀測性** — 追蹤 handoff、保存狀態以便重播。

全都能建在這些原語之上。沒有一個是新的原語。

```figure
a5-primitive-radar
```

## 動手做 (Build It)

`code/main.py` 用約 150 行的標準函式庫 Python 實作這四個原語。沒有真正的 LLM，每個 agent 都是寫好腳本的策略，讓重點留在協調結構上。

這個檔案匯出：

- `Agent` — 一個 dataclass，包含 name、system prompt、tools、policy 函式。
- `Handoff` — 回傳新 agent 的函式。
- `SharedState` — 執行緒安全的訊息 pool。
- `Orchestrator` — 三個變體：`StaticOrchestrator`、`HandoffOrchestrator`、`LLMSelectorOrchestrator`（模擬的）。

這個 demo 讓同一條三個 agent 的 pipeline（research → write → review）跑過全部三種 orchestrator，並在最後印出訊息 pool。你可以看到輸出的差異只在於 *誰挑選下一個*；agent 和 shared state 在每次執行中都是相同的。

執行：

```
python3 code/main.py
```

預期輸出：三次 orchestrator 的執行，每種 pattern 一次。每次都會印出最終的訊息 pool。如果 researcher 認為自己提早完成，handoff 驅動的那次會觸及較少的 agent，這就是 LLM 路由取捨的縮影。

## 實際使用 (Use It)

`outputs/skill-primitive-mapper.md` 是一個 skill，它會讀任何 multi-agent 程式庫或框架文件，並回傳四個原語的對應。新框架發布時就用它跑一遍，在深入讀文件之前先得到一段話的理解。

## 交付 (Ship It)

採用新框架之前，先為它寫出原語對應。如果寫不出來，要嘛是文件不完整，要嘛是這個框架在發明第五個原語（很少見，請檢查是否有你沒見過的 shared-state 變體）。

把這份對應釘在你的架構文件裡。新成員加入時，先給他對應表，再給 API 文件。框架版本更新時，比對的是這份對應，而不是 changelog。

## 練習 (Exercises)

1. 用不同的 agent policy 執行 `code/main.py` 三次。觀察 orchestrator 的選擇如何改變哪些 agent 會執行。
2. 實作第四種 orchestrator：queue 驅動的版本，agent 輪詢 shared state 來找工作。會發生什麼 deadlock？你要怎麼偵測？
3. 拿 LangGraph 的 quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents)，用四個原語改寫。LangGraph 的哪些抽象是 1:1 對應，哪些只是方便的包裝？
4. 閱讀 OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。找出 Swarm 讓四個原語中的哪一個最好用，又把哪一個丟給了呼叫者。
5. 在這張表裡找一個完全隱藏 shared state 的框架。說明當 agent 需要跨 handoff 協調、又不想重讀歷史時，會壞掉什麼。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Agent | 「有 tool 的 LLM」 | 一個 `(system_prompt, tools, model)` 三元組。無狀態。 |
| Handoff | 「控制權轉移」 | 一個結構化的呼叫，指名下一個 agent 與可選的 payload。三種實作：函式回傳、graph edge、speaker selection。 |
| Shared state | 「記憶體」／「context」 | multi-agent 系統中唯一有狀態的部分。Message pool 或 blackboard。 |
| Orchestrator | 「協調者」 | 決定下一個誰執行的角色。靜態圖、LLM selector、handoff 驅動，或 queue 驅動。 |
| Primitive | 「抽象」 | 每個框架都會參數化的四條軸之一。不是框架的功能。 |
| Message pool | 「共享聊天歷史」 | 完整歷史的 shared state。容易推理，擴展性差。 |
| Projected state | 「限定範圍的視圖」 | 對 shared state 依角色做的視圖。能擴展，需要設計 schema。 |
| Speaker selection | 「下一個誰說話」 | Orchestrator pattern：由一個函式（通常是 LLM）從一群 agent 中挑出下一個。 |

## 延伸閱讀 (Further Reading)

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 對 handoff 驅動 orchestration 最清楚的闡述
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection 是 LLM 選擇式 orchestration 的參考
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration 與以 reducer 為基礎的 shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory 的 agent、Sequential / Hierarchical 流程
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — Microsoft 把 v0.4 轉入維護模式後，仍持續活躍的 AutoGen v0.2 分支
