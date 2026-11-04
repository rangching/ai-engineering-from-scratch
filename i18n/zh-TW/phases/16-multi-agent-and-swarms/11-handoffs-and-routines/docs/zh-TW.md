# Handoff 與 Routine：無狀態的 Orchestration (Handoffs and Routines — Stateless Orchestration)

> OpenAI 的 Swarm（2024 年 10 月）把 multi-agent orchestration 精煉成兩個原語：**routine**（指令 + tool，作為 system prompt）和 **handoff**（一個回傳另一個 Agent 的 tool）。沒有狀態機，沒有分支 DSL，LLM 靠呼叫正確的 handoff tool 來路由。OpenAI Agents SDK（2025 年 3 月）是正式環境的繼任者。Swarm 本身仍是最乾淨的概念參考，整份原始碼只有幾百行。這個 pattern 會擴散開來，是因為 API 介面大致就是「agent = prompt + tools；handoff = 回傳 agent 的函式」。限制：無狀態，所以記憶體是呼叫者的問題。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 分鐘

## 問題 (Problem)

每個 multi-agent 框架都要你學它的 DSL：LangGraph 的 node 和 edge、CrewAI 的 crew 和 task、AutoGen 的 GroupChat 和 manager。這些 DSL 是真正的抽象，但它們讓這件事感覺比實際需要的還要沉重。

Swarm 往相反的方向推：使用模型本來就有的 tool-calling 能力。Handoff 變成 tool call。Orchestrator 就是目前握有對話的那個 agent。狀態機隱含在 agent 的 system prompt 之中。

## 核心概念 (Concept)

### 兩個原語 (Two primitives)

**Routine。** 定義 agent 角色與可用 tool 的 system prompt。把它想成限定範圍的一組指令：「你是 triage agent；如果使用者問退款，就 handoff 給退款 agent。」

**Handoff。** Agent 可以呼叫的 tool，會回傳新的 Agent 物件。Swarm runtime 偵測到回傳值是 Agent，就把下一輪的 active agent 切換過去。

這就是整個抽象。

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

Triage agent 的 system prompt 讓它依使用者訊息選擇正確的 handoff。路由是由 LLM 的 tool-calling 完成的。

### 為什麼會擴散 (Why it is viral)

- **API 很小。** 只有兩個概念要學。
- **利用模型本來就會的事。** Tool calling 在各家 provider 上都已達到正式環境等級。
- **沒有狀態機的負擔。** 你不用描述圖；agent 的 prompt 就描述了它們會 handoff 給誰。

### 無狀態的取捨 (The stateless trade)

Swarm 明確地在執行之間是無狀態的。框架在一次執行中保留訊息歷史，但不持久化任何東西。記憶體、連續性、長時間執行的任務，全是呼叫者的問題。

在正式環境中（OpenAI Agents SDK，2025 年 3 月），這是改變最主要的事情之一：SDK 在保留 handoff 原語的同時，加入內建的 session 管理、guardrail 和 tracing。

### Swarm／Handoff 適合的時候 (When Swarm/handoffs fit)

- **Triage pattern。** 第一線 agent 把使用者導向專家。
- **以技能為基礎的 handoff。** 「如果任務需要程式碼，呼叫 coder；如果需要研究，呼叫 researcher。」
- **簡短、有界限的對話。** 客服、FAQ 轉工單、簡單的工作流程。

### Swarm 吃力的時候 (When Swarm struggles)

- **有共享記憶體的長 session。** Handoff 會把對話狀態重設為新 agent 的 prompt 加上歷史。沒有呼叫者管理的記憶體，就沒有跨 agent 的持久狀態。
- **平行執行。** Handoff 一次只有一個，active agent 被切換。平行化需要呼叫者去編排多次 Swarm 執行。
- **稽核與重播。** 無狀態的執行很難精確重播；LLM 的 handoff 選擇不是確定性的。

### OpenAI Agents SDK（2025 年 3 月）(OpenAI Agents SDK (March 2025))

正式環境的繼任者增加了：

- **Session 狀態。** 跨執行的持久 thread。
- **Guardrail。** 輸入／輸出驗證的 hook。
- **Tracing。** 每次 tool call 和 handoff 都會被記錄。
- **Handoff filter。** 控制 handoff 時哪些 context 會被轉移。

Handoff 原語留了下來；正式環境的使用體驗則是圍繞它補上的。

### Swarm 與 GroupChat (Swarm vs GroupChat)

兩者都使用 LLM 驅動的路由，但差別在於 **誰挑選下一個**：

- GroupChat：由 selector（函式或 LLM）從外部挑選下一位發言者。
- Swarm：目前的 agent 透過呼叫 handoff tool 挑選它的繼任者。

Swarm 是「agent 決定下一步」；GroupChat 是「manager 決定下一步」。Swarm 的決定存在於 active agent 的 tool call 中；GroupChat 的決定則存在 `GroupChatManager`。

```figure
sw-handoff-routing
```

## 動手做 (Build It)

`code/main.py` 從零實作 Swarm：一個 Agent dataclass、一個 handoff 機制（tool 回傳 Agent），以及一個偵測 agent 切換的執行迴圈。

Demo：一個 triage agent 把使用者導向退款、銷售或支援專家。每位專家有自己的 tool。執行迴圈會印出每一次 handoff。

執行：

```
python3 code/main.py
```

## 實際使用 (Use It)

`outputs/skill-handoff-designer.md` 為指定任務設計 handoff 拓撲：有哪些 agent、它們可以呼叫哪些 handoff、哪些 context 會被轉移。

## 交付 (Ship It)

檢查清單：

- **Handoff 紀錄。** 每次 handoff 都寫入一筆追蹤事件，包含 from-agent、to-agent、context 快照。
- **Context 轉移規則。** 決定 handoff 時什麼會移動：完整歷史（昂貴）、最後 N 則訊息，或摘要。
- **Handoff 的 guardrail。** Handoff 到工具權限不同的專家時，必須經過驗證，否則 prompt injection 可能強迫發生不想要的 handoff。
- **迴圈偵測。** 兩個 agent 來回互相 handoff 是常見的失敗；用簡單的最近 K 筆環狀檢查來偵測。
- **備援 agent。** 如果 handoff 的目標不存在，就退回安全的預設值。

## 練習 (Exercises)

1. 執行 `code/main.py`，從 triage 到退款 agent。確認第二輪的 active agent 是 refund。
2. 加入迴圈偵測規則：如果同樣兩個 agent 連續 handoff 3 次，就強制離開。設計備援。
3. 閱讀 OpenAI Agents SDK 關於 handoff filter 的文件。實作「handoff 時摘要」的版本：交出的 agent 在接手的 agent 接管之前，先把 context 壓縮成條列摘要。
4. 比較 Swarm 的 handoff 和 GroupChatManager 的 selector。哪一種 pattern 會讓 prompt injection 更嚴重？為什麼？
5. 閱讀 Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents)。找出 Swarm 做出的一項明確設計決定，並說明 OpenAI Agents SDK 是改變了它還是保留了它。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Routine | 「Agent 的 prompt」 | System prompt + tool 清單。定義角色與可用的 handoff。 |
| Handoff | 「轉給另一個 agent」 | Active agent 可呼叫、會回傳新 Agent 的 tool。Runtime 會切換 active agent。 |
| Stateless | 「執行之間沒有記憶」 | Swarm 不持久化任何東西；記憶體由呼叫者負責。 |
| Active agent | 「現在是誰在說話」 | 目前握有對話的 agent。Handoff 會改變它。 |
| Context transfer | 「Handoff 時什麼會移動」 | 接手的 agent 看到什麼歷史的政策：完整、最後 N 則，或摘要。 |
| Handoff loop | 「Agent 打乒乓球」 | 失敗模式：兩個 agent 不斷把控制權交回給對方。 |
| OpenAI Agents SDK | 「正式版的 Swarm」 | 2025 年 3 月的繼任者；在 handoff 原語之上加入 session、guardrail、tracing。 |
| Handoff filter | 「轉移時的關卡」 | SDK 功能，用來在 handoff 邊界檢查並修改 context。 |

## 延伸閱讀 (Further Reading)

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — 參考性的闡述
- [OpenAI Swarm repo](https://github.com/openai/swarm) — 原始實作，保留作為概念參考
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — 具備 session 與 tracing 的正式環境繼任者
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — Claude Code subagent 如何透過 `Task` 使用類似 handoff 的 pattern
