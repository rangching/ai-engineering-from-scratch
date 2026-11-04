# Orchestration 模式：Supervisor、Swarm、Hierarchical (Orchestration Patterns: Supervisor, Swarm, Hierarchical)

> 四種 orchestration 模式在 2026 年的 framework 中反覆出現：supervisor-worker、swarm / peer-to-peer、hierarchical、debate。Anthropic 的指引："It's about building the right system for your needs." 從簡單開始；只有當單一 agent 加上五種 workflow 模式不夠時，才加入拓撲。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 25 (Multi-Agent Debate)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出四種反覆出現的 orchestration 模式，以及各自適用的時機。
- 描述 2026 年 LangChain 的建議：以 tool call 為基礎的 supervision，對比 supervisor 函式庫。
- 解釋 Anthropic 的「打造對的系統」規則，以及它如何限制拓撲的選擇。
- 針對共同的腳本化 LLM，用 stdlib 實作全部四種。

## 問題 (The Problem)

團隊在需要之前就伸手去拿「multi-agent」。四種模式在各 framework 中反覆出現；一旦你能叫出它們的名字，就能選對的那個，或是乾脆完全跳過拓撲。

## 核心概念 (The Concept)

### Supervisor-worker

- 由中央路由 LLM 把工作分派給專家 agent。
- 決定：繞回自己、交接給專家，或終止。
- 專家之間不互相溝通；所有路由都經過 supervisor。

Framework：LangGraph `create_supervisor`、Anthropic orchestrator-workers、CrewAI Hierarchical Process。

**2026 年 LangChain 的建議：** 透過直接的 tool call 來做 supervision，而不是用 `create_supervisor`。這讓你對 context engineering 有更細緻的控制，也就是你自己決定每位專家看到什麼。

### Swarm / peer-to-peer

- Agent 透過共用的 tool 介面直接交接。
- 沒有中央 router。
- 延遲比 supervisor 低（跳數較少）。
- 更難推理（沒有單一控制點）。

Framework：LangGraph swarm 拓撲、OpenAI Agents SDK handoff（當所有 agent 都能交接給其他所有 agent 時）。

### Hierarchical

- Supervisor 管理子 supervisor，再管理 worker。
- 在 LangGraph 中以巢狀 subgraph 實作；在 CrewAI 中是巢狀 crew。
- 能擴展到大量 agent，代價是營運複雜度。

何時需要：當單一 supervisor 的 context 預算放不下所有專家的描述時。

### Debate

- 平行的提案者 + 迭代的交叉批評（Lesson 25）。
- 嚴格來說不是 orchestration，更像驗證，但在 framework 中會以拓撲選項的形式出現。

### 自主 crew 對確定性 flow (Autonomous crews vs deterministic flows)

CrewAI 將兩種部署模式正式化：

- **Flow** 用於確定性的事件驅動自動化（正式環境建議的起點）。
- **Crew** 用於自主的、以角色為基礎的協作。

這與上述四種模式正交，但可以對應到拓撲：Flow 通常是 supervisor 或 hierarchical；Crew 通常是帶有 LLM router 的 supervisor。

### Anthropic 的指引 (Anthropic's guidance)

"Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

決策順序：

1. 單一 agent + workflow 模式（Lesson 12）：從這裡開始。
2. Supervisor-worker：當你有 2-4 位專家時。
3. Swarm：當延遲比推理清晰度更重要時。
4. Hierarchical：只有在 supervisor 的 context 預算不夠時。
5. Debate：當準確率比成本更重要時。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **拓撲優先的思考。** 還沒弄清楚 multi-agent 要解決什麼問題，就說「我們需要 multi-agent」。
- **Swarm 中來回彈跳的 handoff。** A -> B -> A -> B。使用 hop 計數器。
- **假的階層。** 因為「企業級」而做三層，實際只有兩個團隊。把它們壓平。

```figure
orchestration-pattern
```

## 動手做 (Build It)

`code/main.py` 針對腳本化的 LLM，用 stdlib 實作全部四種模式：

- `Supervisor`：中央 router。
- `Swarm`：peer-to-peer，直接 handoff。
- `Hierarchical`：supervisor 的 supervisor。
- `Debate`：平行提案者 + 批評。

每種模式都處理同樣的三個意圖任務（退款 / bug / 業務）。Trace 的形狀不同。

執行：

```
python3 code/main.py
```

輸出：每種模式的 trace + 操作次數。Supervisor 最乾淨；swarm 最短；hierarchical 最深；debate 最昂貴。

## 實際使用 (Use It)

- **LangGraph** 用於 supervisor 和 hierarchical（巢狀 subgraph）。
- **OpenAI Agents SDK** 用於把 handoff 當作 tool（supervisor 形狀）。
- **CrewAI Flow** 用於正式環境的確定性。
- **自行打造** 用於 debate，或當你想要精確控制時。

## 交付 (Ship It)

`outputs/skill-orchestration-picker.md` 挑選拓撲並實作它。

## 練習 (Exercises)

1. 移除 router，把 supervisor-worker 轉成 swarm。什麼會壞掉？什麼會變好？
2. 為 swarm 加上 hop 計數器：3 次 handoff 後就拒絕。它抓得到 A->B->A 的彈跳嗎？
3. 為有 12 位專家的領域建一個兩層的 hierarchical 系統。沒有巢狀的話，context 預算在哪裡失效？
4. 在接近正式環境形狀的工作負載上分析這四種模式。哪個在哪個指標（延遲、成本、準確率、可除錯性）上勝出？
5. 讀 Anthropic 的 "Building Effective Agents" 文章。把你每個正式環境的流程對應到四種之一。有沒有無法乾淨對應的？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Supervisor-worker | 「Router + 專家」 | 中央 LLM 分派給專家；專家之間不互相溝通 |
| Swarm | 「Peer-to-peer」 | 透過共用 tool 直接 handoff；沒有中央 router |
| Hierarchical | 「Supervisor 的 supervisor」 | 為大量 agent 設計的巢狀 subgraph |
| Debate | 「提案者 + 批評」 | 平行提案者，交叉批評（Lesson 25） |
| Tool-call-based supervision | 「沒有函式庫的 supervisor」 | 以直接 tool call 實作 supervisor，以控制 context |
| Crew | 「自主團隊」 | CrewAI 以角色為基礎的協作模式 |
| Flow | 「確定性 workflow」 | CrewAI 事件驅動的正式環境模式 |

## 延伸閱讀 (Further Reading)

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：五種模式 + agent 對 workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：supervisor、swarm、hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction)：Crew 對 Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)：debate 模式
