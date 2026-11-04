# 以角色為基礎的 Agent 團隊：Role、Task、Process (Role-Based Agent Teams — Roles, Tasks, Processes)

> 四個原語：Agent、Task、Crew、Process。兩種頂層形態：Crews（自主、以角色為基礎的協作）和 Flows（事件驅動、確定性）。CrewAI 是 2026 年的參考實作，它的文件說得很直接："for any production-ready application, start with a Flow."

**Type:** Learn + Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 14 (Actor Model)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 CrewAI 的四個原語（Agent、Task、Crew、Process），以及各自擁有什麼。
- 區分 Sequential、Hierarchical 和規劃中的 Consensus process；依工作負載各選一個。
- 區分 Crews（自主、以角色為基礎）和 Flows（事件驅動、確定性），並解釋文件對正式環境的建議。
- 用 `@tool` decorator 和 `BaseTool` 子類別接上 tool；推理結構化輸出與自由文字的差異。
- 說出 CrewAI 的四種 memory 類型，以及各自何時划算。
- 用純 Python 實作三個 agent 的 crew（researcher、writer、editor），產出一份簡報。
- 辨認 CrewAI 的四種失敗模式：prompt 膨脹、manager-LLM 稅、脆弱的交接、Crew 直接上正式環境。

## 問題 (The Problem)

採用 multi-agent framework 的團隊都會撞上同一面牆。「自主協作」在 demo 裡聽起來很棒。然後客戶回報一個 bug，你需要確定性的重播。或者財務問你，由 LLM 路由的 crew 每次執行要花多少錢。或者值班人員需要知道凌晨 3 點是哪個 agent 卡住了。

自由形式、由 LLM 路由的 crew，沒有一個問題能乾淨地回答。純 DAG 全都能回答，卻失去了腦力激盪型 agent 所需的探索性。

CrewAI 的切分對取捨很誠實。Crews 用於協作、以角色為基礎、探索性的工作。Flows 用於事件驅動、由程式碼擁有、可稽核的正式環境。同一個 framework，兩種形態，依使用面選擇。

## 核心概念 (The Concept)

### 四個原語 (Four primitives)

CrewAI 的介面很小。記住這個，其餘就只是設定。

- **Agent。** `role + goal + backstory + tools + (optional) llm`。Backstory 是承重的。它塑造語氣、判斷，以及 agent 何時停下來。Tool 是 agent 可以呼叫的函式（下面有更多說明）。
- **Task。** `description + expected_output + agent + (optional) context + (optional) output_pydantic`。可重複使用的工作單位。`expected_output` 是合約。`context` 列出輸出會被傳進來的上游 task。`output_pydantic` 強制結構化的形狀。
- **Crew。** 容器。擁有 `agents` 清單、`tasks` 清單、`process`，以及選用的 `memory` + `verbose` + `manager_llm` 設定。
- **Process。** 執行策略。Sequential、Hierarchical、Consensus（規劃中）。決定這次執行的形狀。

Agent 彼此不會直接看見對方。Task 參照 agent。Crew 為 task 排序。Process 決定誰來挑下一個 task。整個心智模型就是這樣。

> **驗證版本：** CrewAI 0.86 (2026-05)。較新的版本可能重新命名或合併 process 類型；依賴特定形態之前，請先查閱 [CrewAI Processes docs](https://docs.crewai.com/concepts/processes)。

### Sequential、Hierarchical 與 Consensus

- **Sequential。** Task 依宣告順序執行。Task N 的輸出可作為 task N+1 的 `context`。成本最低。最可預測。順序固定時使用。
- **Hierarchical。** 由 manager Agent（獨立的 LLM 呼叫）在專家之間做路由。CrewAI 會依你的 `manager_llm` 設定或預設值產生 manager。Manager 每一輪挑選下一個 task，並可以拒絕或重新路由。當你有四個以上的專家，且順序確實取決於先前的輸出時使用。
- **Consensus。** 規劃中，目前公開 API 尚未實作。文件把這個名稱保留給未來以投票為基礎的 process。今天不要依賴它。

Hierarchical 會在每次專家呼叫之上，再多加每一輪一次的 LLM 呼叫（manager）。五步驟的執行，token 成本可能變三倍。只有在你需要路由時才付這個代價。

### Crews 對 Flows (Crews vs Flows)

這是文件在 2026 年一開始就強調的框架。

- **Crew。** 由 LLM 驅動的自主性。Framework 在執行期決定形狀。適合：研究、腦力激盪、初稿，也就是路徑本身就是答案一部分的地方。難以重播。難以測試。原型製作很便宜。
- **Flow。** 你擁有的事件驅動 graph。`@start` 標示進入點。`@listen(topic)` 標示在另一個步驟發出該 topic 時觸發的步驟。每個步驟都是普通的 Python（內部可以呼叫 Crew）。適合：正式環境。可觀測。可測試。確定性。

文件在 2026 年對正式環境的建議：從 Flow 開始。當自主性值回票價時，再以 Flow 步驟內的 `Crew.kickoff()` 呼叫，把 Crew 折進來。Flow 給你稽核軌跡，Crew 給你探索。要組合，不要二選一。

### Tool 整合 (Tool integration)

給 Agent 一個 tool 有三種方式。選最簡單且合適的那個。

1. **`@tool` decorator。** 純函式變成 tool。簽名就是 schema；docstring 就是 LLM 看到的 description。最適合一次性的輔助函式。

   ```python
   from crewai.tools import tool

   @tool("Search the web")
   def search(query: str) -> str:
       """Return top results for the query."""
       return run_search(query)
   ```

2. **`BaseTool` 子類別。** 以類別為基礎的 tool，有明確的參數 schema、async 支援、重試。當 tool 有狀態（client、cache）或需要結構化參數時使用。

   ```python
   from crewai.tools import BaseTool
   from pydantic import BaseModel

   class SearchArgs(BaseModel):
       query: str
       limit: int = 10

   class SearchTool(BaseTool):
       name = "web_search"
       description = "Search the web and return top results."
       args_schema = SearchArgs

       def _run(self, query: str, limit: int = 10) -> str:
           return self.client.search(query, limit=limit)
   ```

3. **內建 toolkit。** CrewAI 提供第一方 adapter：`SerperDevTool`、`FileReadTool`、`DirectoryReadTool`、`CodeInterpreterTool`、`RagTool`、`WebsiteSearchTool`。一行 import 就能接上。

結構化輸出使用 Pydantic。在 Task 上傳入 `output_pydantic=MyModel`。CrewAI 會用該 model 驗證 LLM 的回應，然後轉型或重試。搭配精簡的 `expected_output` 字串使用。草稿用自由文字輸出沒問題；下游的 Flow 能消費的是結構化輸出。

### Memory hook

CrewAI 內建四種 memory 類型。它們可以組合：一個 Crew 可以同時啟用四種。

> **驗證版本：** CrewAI 0.86 (2026-05)。近期版本把一切都導向一個包住這四個 store 的統一 `Memory` 系統。下面的概念模型仍然成立，但在較新版本中，公開的類別介面可能收斂成單一的 `Memory` 入口；目前的 API 請查閱 [CrewAI memory docs](https://docs.crewai.com/concepts/memory)。

- **Short-term。** 單次執行內的對話 buffer。結束時清除。
- **Long-term。** 跨執行持久化。存在 vector DB（預設是 Chroma，可替換）。依與目前 task 的相似度取回。
- **Entity。** 每個實體的事實。「客戶 X 使用 enterprise 方案。」以實體為 key，而不是以相似度。跨執行保留。
- **Contextual。** 組裝時的 retrieval。在 Agent 需要的那一刻才取回相關 memory，而不是預先載入。

在 Crew 上用 `memory=True` 或依類型設定來啟用。底層是你設定的 embeddings provider（預設是 OpenAI，可換成本地的）。Memory 是 CrewAI 相較於較薄的 framework 值回票價的地方之一；純 LangGraph 需要你自己接上每一個。

### 何時適合角色為基礎的團隊 (When role-based teams fit)

- 三到六個有明確角色與協作 workflow 的 agent。撰寫草稿、審查、規劃、腦力激盪。
- LLM 對下一步的判斷本身就是價值一部分的路由（Hierarchical）。
- 團隊讀 `role + goal + backstory` 比讀 graph 定義更自在的任何地方。

### 何時不適合 (When they do not)

- 有嚴格順序的確定性 DAG。使用 LangGraph（Lesson 13）。Graph 的形狀才是正確的抽象；CrewAI 的角色框架是阻力。
- 延遲預算在秒以下。Hierarchical 會增加來回次數。即使是 Sequential，也會把含有 backstory 和先前輸出的 prompt 串行化。
- 單一 agent 的 loop。跳過 framework；agent loop（Lesson 1）加上 tool registry 更短。

Lesson 28（Orchestration Patterns）會並排比較各種拓撲。簡短版本：CrewAI 位於「協作式、以角色為基礎」的角落。

### 相依性形態 (Dependency shape)

獨立於 LangChain。Python 3.10 到 3.13。使用 `uv`。Star 數：見 [crewAIInc/crewAI](https://github.com/crewAIInc/crewAI)（2026-05 的快照）。AWS Bedrock 整合有文件記載；廠商 benchmark 回報在 QA 工作負載上相較 LangGraph 有大幅加速，但方法論（資料集、硬體、評估指標）並未公開，所以請把 framework 廠商的數字視為僅供參考的方向。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Backstory 造成 prompt 膨脹。** 每個 agent 2000 字的 backstory，加上五個 agent 的 crew，在第一次 tool call 之前就把 context 預算燒光。Backstory 控制在 200 字以內。跨 agent 重複使用片語；不要把 house style 重複寫五次。
- **Manager-LLM 的 token 稅。** Hierarchical process 在每次專家呼叫之前，都會加一次 manager LLM 呼叫。五個 task 的 crew 會變成十次 LLM 呼叫而不是五次，而且 manager 呼叫還帶著完整的 task 清單加上先前的輸出。除非路由取決於輸出，否則改用 Sequential。
- **脆弱的交接 (Brittle handoffs)。** Task N 的 `expected_output` 是「一份大綱」。Task N+1 把它當作 `context` 讀取，並試著解析三個段落。LLM 產生了四個。下游的 Agent 就即興發揮。修法是在 Task N 上用 `output_pydantic`，讓 Task N+1 讀到的是有型別的物件，而不是自由文字。
- **Crew 直接上正式環境 (Crew-as-prod)。** 沒有 Flow 包裝就把自由形式的 Crew 上線。輸出變異很大；無法重播；值班人員無法把壞的執行和好的執行做 diff。用 Flow 包起來。

```figure
ae-crew-vs-flow
```

## 動手做 (Build It)

`code/main.py` 以純 Python 實作兩種形態，加上一個三 agent 的 crew。除了標準函式庫之外唯一的相依性是 numpy，用於模擬的相似度。

結構：

- `Agent`、`Task` dataclass，對應 CrewAI 的介面。
- `SequentialCrew.kickoff(inputs)` 依宣告順序執行 task，並把輸出以 `context` 串接起來。
- `HierarchicalCrew.kickoff(topic)` 加入 manager Agent，每一輪挑選下一位專家，遇到 "done" 就停止。
- `Flow`：附有 `@start` 和 `@listen(topic)` decorator、一個小型事件迴圈，以及 trace。
- `tool(name)` decorator，對應 CrewAI 的 `@tool` 形態。
- `Memory`：有 `short_term`、`long_term`、`entity` 三個 store；模擬的相似度使用 numpy。
- 模擬的 LLM 回應是硬編碼的字串，以 role 加上輸入前綴作為 key。沒有網路。確定性。

具體 demo：researcher、writer、editor 的 crew，產出一份關於「agent engineering 2026」的簡報。Researcher 取得（模擬的）來源。Writer 撰寫草稿。Editor 精簡。同一個 crew 也透過 Flow 執行，以展示確定性的形態。

執行：

```bash
python3 code/main.py
```

Trace 涵蓋：sequential crew 透過 `context` 串接輸出、hierarchical crew 由 manager 挑選（researcher、writer、editor，然後 "done"）、flow 用明確的 topic（`researched`、`drafted`、`edited`）執行同樣的三個步驟、透過 `@tool` 路由的 tool call，以及跨兩次 kickoff 保留的 long-term memory。

Crew 的 trace 是流動的；manager 原則上可以重新排序。Flow 的 trace 是固定的。這個選擇就是這一課的重點。

## 實際使用 (Use It)

- **CrewAI Flow** 用於正式環境。即使 Flow 只有一個呼叫 `Crew.kickoff()` 的步驟。Flow 提供稽核邊界。
- **CrewAI Crew (Sequential)** 用於順序明確的協作工作，特別是初稿和審查 loop。
- **CrewAI Crew (Hierarchical)** 用於路由取決於輸出，且你有四個以上專家時。
- **LangGraph**（Lesson 13）用於明確的狀態機、durable resume、嚴格的順序。
- **AutoGen v0.4**（Lesson 14）用於 actor-model 並行與故障隔離。
- **OpenAI Agents SDK**（Lesson 16）用於以 OpenAI 為優先、需要 handoff 與 guardrail 的產品。
- **Claude Agent SDK**（Lesson 17）用於以 Claude 為優先、需要 subagent 與 session store 的產品。

## 交付 (Ship It)

`outputs/skill-crew-or-flow.md` 為任務挑選 Crew 或 Flow，並建立最小實作骨架。硬性拒絕：沒有 backstory 的 Crew、沒有明確 topic 的 Flow、專家少於三個的 Hierarchical。

## 陷阱 (Pitfalls)

- **把 backstory 當成調味。** 它會塑造輸出。每個 agent 測三個變體；變異是真實存在的。選一個，然後凍結。
- **跳過 `expected_output`。** 沒有每個 task 的合約，下游 task 會接收 LLM 產出的任何東西。Crew 能跑；稽核失敗。
- **Memory 一直開著。** Long-term 每次執行都寫入。Vector DB 越長越大。Retrieval 變得雜訊很多。把寫入限縮在事實是持久性的 task。
- **Manager prompt 漂移。** Hierarchical 的 manager prompt 是隱含的。如果路由變得怪異，用 verbose 模式把它傾印出來閱讀。
- **Crew 中 tool 的副作用。** Crew 呼叫 tool 的次數可能超出預期。POST、DELETE、付款屬於 Flow 步驟，絕不是 Crew tool。

## 練習 (Exercises)

1. 把 Sequential crew 轉成 Flow。數一數變異性降低的接觸點。記下可讀性下降的地方。
2. 為 crew 加上 entity memory：關於某位客戶的事實在多次 kickoff 之間持續保留。驗證 retrieval 取回的是正確的實體。
3. 實作一個 Hierarchical process，讓 manager 在 writer 的輸出至少有三段之前，拒絕路由給 editor。追蹤重試過程。
4. 為（模擬的）web search 接上 `BaseTool` 子類別。比較它與 `@tool` decorator 版本的 trace 形狀。
5. 在 editor task 上加 `output_pydantic=Brief`，其中 `Brief` 有 `title`、`summary`、`sections`。讓 writer task 輸出一次格式錯誤的 JSON；在 trace 中驗證 CrewAI 的重試行為。
6. 讀 CrewAI 文件簡介。把 toy 移植到真正的 `crewai` API。stdlib 版本略過了哪些保證？
7. 把 AgentOps 或 Langfuse（Lesson 24）接到一次真實的執行。你在 stdlib 版本中漏掉了哪些 trace？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Agent | 「Persona」 | Role + goal + backstory + tools |
| Task | 「工作單位」 | Description + 預期輸出 + 負責人 + 選用的結構化輸出 |
| Crew | 「Agent 團隊」 | Agents + Tasks + Process 的容器 |
| Process | 「執行策略」 | Sequential / Hierarchical / Consensus（規劃中） |
| Flow | 「確定性 workflow」 | 事件驅動、由程式碼擁有、可測試 |
| Backstory | 「Persona prompt」 | 塑造 Agent 語氣與判斷的內容 |
| `@tool` | 「函式 tool」 | 把函式變成 Agent 可呼叫 tool 的 decorator |
| `BaseTool` | 「類別 tool」 | 以類別為基礎、含參數 schema、重試、async 支援的 tool |
| Entity memory | 「每個實體的事實」 | 限定於某位客戶 / 帳號 / issue 的 memory |
| Long-term memory | 「跨執行 memory」 | 由 vector 支撐、在多次 kickoff 之間保留的 memory |
| Contextual memory | 「即時 retrieval」 | 在 Agent 需要的那一刻才取回的 memory |
| Manager LLM | 「Router agent」 | Hierarchical process 中挑選下一個 task 的額外 LLM |
| `expected_output` | 「Task 合約」 | 告訴 Agent（與稽核）該回傳什麼形狀的字串 |

## 延伸閱讀 (Further Reading)

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction)：概念與建議的正式環境路徑
- [CrewAI Flows guide](https://docs.crewai.com/en/concepts/flows)：事件驅動的形態、`@start`、`@listen`
- [CrewAI tools reference](https://docs.crewai.com/en/concepts/tools)：`@tool`、`BaseTool`、內建 toolkit
- [CrewAI memory](https://docs.crewai.com/en/concepts/memory)：short-term、long-term、entity、contextual
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：multi-agent 何時有幫助、何時沒有
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：狀態機的替代方案
