# 正式環境的 Agent Runtime：快速實例化與有型別的 Workflow (Production Agent Runtimes — Fast Instantiation and Typed Workflows)

> 正式環境的 agent runtime 最佳化的是原型 framework 所忽略的東西：實例化成本、有型別的 workflow 介面，以及可直接上線服務的後端。2026 年的一對組合：Agno（Python）目標是微秒等級的 agent 實例化與 stateless 的 FastAPI 後端。Mastra 則在 Vercel AI SDK 的基底上，提供 agent、tool、workflow、統一的模型路由和複合式儲存。

**Type:** Learn
**Languages:** Python, TypeScript
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 13 (LangGraph)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 辨識 Agno 的效能目標，以及它們何時重要。
- 說出 Mastra 的三個原語（Agents、Tools、Workflows）以及支援的 server adapter。
- 解釋為什麼 stateless、以 session 為範圍的 FastAPI 後端，是建議的 Agno 正式環境路徑。
- 針對特定技術堆疊（Python 優先或 TypeScript 優先）在 Agno 與 Mastra 之間做選擇。

## 問題 (The Problem)

LangGraph、AutoGen、CrewAI 都是 framework 很重的選擇。想要「只要 agent loop，要快，而且在我的 runtime 裡」的團隊，會去找 Agno（Python）或 Mastra（TypeScript）。兩者都犧牲一部分由 framework 擁有的原語，換取原始速度，以及與周圍技術堆疊更緊密的貼合。

## 核心概念 (The Concept)

### Agno

- Python runtime，前身是 Phi-data。
- "No graphs, chains, or convoluted patterns — just pure python."
- 他們文件中的效能目標：約 2μs 的 agent 實例化、每個 agent 約 3.75 KiB 記憶體、約 23 個模型 provider。
- 正式環境路徑：stateless、以 session 為範圍的 FastAPI 後端。每個請求啟動一個全新的 agent；session state 存在 DB 裡。
- 原生多模態（文字、圖片、音訊、影片、檔案）與 agentic RAG。

當你每秒有數千個短命的 agent 時（chat fan-in、評估 pipeline），這些速度目標就很重要。當一個 agent 要跑 10 分鐘時，就沒那麼重要。

### Mastra

- TypeScript，建立在 Vercel AI SDK 之上。
- 三個原語：**Agents**、**Tools**（Zod 型別）、**Workflows**。
- 統一的 Model Router：橫跨 94 個 provider 的 3,300 多個模型（2026 年 3 月）。
- 複合式儲存：memory、workflow、observability 各自接不同後端；大規模 observability 建議使用 ClickHouse。
- Apache 2.0，但 `ee/` 目錄採用 source-available 的企業授權。
- 支援 Express、Hono、Fastify、Koa 的 server adapter；一級的 Next.js 和 Astro 整合。
- 附帶 Mastra Studio（localhost:4111）供 debug 使用。
- 1.0（2026 年 1 月）時有 22k 以上的 GitHub star、每週 300k 以上的 npm 下載量。

### 定位 (Positioning)

兩者都不是想成為 LangGraph。它們競爭的是：

- **語言契合度。** Agno 給 Python 優先的團隊；Mastra 給 TypeScript 優先的團隊。
- **Runtime 使用體驗。** Agno = 近乎零額外開銷；Mastra = 與 Vercel 生態系整合。
- **Observability。** 兩者都整合 Langfuse/Phoenix/Opik（Lesson 24），但 Mastra Studio 是第一方的。

### 何時選哪個 (When to pick each)

- **Agno**：Python 後端、很多短命 agent、對效能要求高、FastAPI 團隊。
- **Mastra**：TypeScript 後端、Next.js / Vercel 部署、統一的多 provider 模型路由、Zod 型別的 tool。
- **LangGraph**（Lesson 13）：當 durable state 和明確的 graph 推理比原始速度更重要時。
- **OpenAI / Claude Agent SDK**：當你想要 provider 產品化的形態時（Lessons 16–17）。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **為效能而效能 (Perf-for-perf's-sake)。** 因為「2μs」聽起來很棒就選 Agno，但工作負載是每個請求一次緩慢的 agent 呼叫。額外開銷並不是瓶頸。
- **生態系綁定 (Ecosystem lock-in)。** Mastra 帶有 Vercel 風味的整合，在 Vercel 上是加分，在別處是減分。
- **企業授權混淆 (Enterprise license confusion)。** Mastra 的 `ee/` 目錄是 source-available，不是 Apache 2.0。如果你打算 fork，請閱讀授權條款。

```figure
wb-runtime-spawn
```

## 動手做 (Build It)

這一課主要是比較性質，沒有單一程式碼成品能同時公平對待兩個 framework。參見 `code/main.py` 的並排 toy：一個最小的「執行 agent、串流輸出、持久化 session」流程，實作兩次（一次 Agno 形狀，一次 Mastra 形狀）。

執行：

```
python3 code/main.py
```

兩份結構不同但功能等價的 trace。

## 實際使用 (Use It)

- **Agno**：需要速度和 FastAPI 形態的 Python 後端。
- **Mastra**：有許多 provider 和 workflow 原語的 TypeScript 後端。
- 兩者都提供第一方 observability hook。兩者都整合 Langfuse。

## 交付 (Ship It)

`outputs/skill-runtime-picker.md` 依技術堆疊、延遲預算和營運形態，挑選 Agno、Mastra、LangGraph 或 provider SDK。

## 練習 (Exercises)

1. 讀 Agno 的文件。把 stdlib ReAct loop（Lesson 01）移植到 Agno。什麼消失了？什麼保留了？
2. 讀 Mastra 的文件。把同一個 loop 移植到 Mastra。在 tool 型別上（Zod 對沒有）有什麼改變？
3. Benchmark：在你的技術堆疊上量測 agent 實例化延遲。Agno 的 2μs 對你的工作負載重要嗎？
4. 設計一次遷移：如果你一直在 Python 中執行 CrewAI，搬到 Agno 會壞掉什麼？
5. 讀 Mastra 的 `ee/` 授權條款。哪些限制會影響開源 fork？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Agno | 「快速的 Python agent」 | Stateless、以 session 為範圍的 agent runtime |
| Mastra | 「Vercel AI SDK 上的 TypeScript agent」 | Agents + Tools + Workflows + Model Router |
| Unified Model Router | 「多 provider 存取」 | 單一 client 存取橫跨 94 個 provider 的 3,300 多個模型 |
| Composite storage | 「多個後端」 | Memory/workflow/observability 各自接不同的 store |
| Mastra Studio | 「本地 debugger」 | 用來檢視 agent 內部的 localhost:4111 UI |
| Source-available | 「不是開源」 | 授權允許閱讀原始碼，但限制商業使用 |

## 延伸閱讀 (Further Reading)

- [Agno Agent Framework docs](https://www.agno.com/agent-framework)：效能目標、FastAPI 整合
- [Mastra docs](https://mastra.ai/docs)：原語、server adapter、Model Router
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：有狀態 graph 的替代方案
- [Comet Opik](https://www.comet.com/site/products/opik/)：Mastra 整合所引用的 observability 比較
