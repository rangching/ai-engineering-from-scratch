# 正式環境 Runtime：Queue、Event、Cron (Production Runtimes: Queue, Event, Cron)

> 正式環境的 agent 以六種 runtime 形態執行：request-response、streaming、durable execution、以 queue 為基礎的背景、事件驅動，以及排程。在選 framework 之前先選形態。Observability 在每一種形態下都是承重的。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 13 (LangGraph), Phase 14 · 22 (Voice)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出六種正式環境 runtime 形態，並將每一種對應到 framework / 產品模式。
- 解釋為什麼 durable execution（LangGraph）對長時程任務很重要。
- 描述事件驅動 runtime，以及 Claude Managed Agents 何時適用。
- 解釋對多步驟 agent 而言「observability 是承重的」這項主張。

## 問題 (The Problem)

正式環境的 agent 會以 Jupyter notebook 不會浮現的方式失敗：第 37 步網路 timeout、使用者在語音通話中途掛斷、cron job 在機器重開機時掛掉、背景 worker 記憶體耗盡。Runtime 形態決定哪些失敗是可以承受的。

## 核心概念 (The Concept)

### Request-response

- 同步 HTTP。使用者等待完成。
- 只適用於短任務（<30 秒）。
- 技術堆疊：Agno（Python + FastAPI）、Mastra（TypeScript + Express/Hono/Fastify/Koa）。
- Observability：標準 HTTP access log + OTel span。

### Streaming

- 用 SSE 或 WebSocket 做漸進式輸出。
- LiveKit 把它延伸到語音/影片的 WebRTC（Lesson 22）。
- 技術堆疊：任何支援 streaming 的 framework + 處理 SSE/WS 的前端。
- Observability：每個 chunk 的時間、first-token 延遲、尾端延遲。

### Durable execution

- 每一步之後都做 state checkpoint；失敗時自動恢復。
- AutoGen v0.4 的 actor model 把失敗隔離在單一 agent（Lesson 14）。
- LangGraph 的核心差異化特色（Lesson 13）。
- 當步數未知且恢復成本很高時不可或缺。

### 以 queue 為基礎 / 背景 (Queue-based / background)

- 工作進入 queue，worker 接手，結果透過 webhook 或 pub/sub 回傳。
- 對長時程 agent 不可或缺（依 Anthropic 的 computer use 公告，每個任務數十到數百步）。
- 技術堆疊：Celery（Python）、BullMQ（Node）、SQS + Lambda（AWS）、自訂。
- Observability：queue 深度、每個 job 的延遲分佈、DLQ 大小。

### 事件驅動 (Event-driven)

- Agent 訂閱觸發條件：新 email、PR 開啟、cron 觸發。
- Claude Managed Agents 開箱即用地涵蓋這種情況（Lesson 17）。
- CrewAI Flows（Lesson 15）組織事件驅動的確定性 workflow。
- Observability：觸發來源、事件到啟動的延遲、agent 延遲。

### 排程 (Scheduled)

- 定期執行、cron 形狀的 agent。
- 與 durable execution 結合，讓失敗的夜間執行在下一個 tick 恢復。
- 技術堆疊：Kubernetes CronJob + durable framework；託管（Render cron、Vercel cron）。

### 2026 年的部署模式 (2026 deployment patterns)

- **CrewAI Flows** 用於事件驅動的正式環境。
- **Agno** stateless FastAPI，用於 Python 微服務。
- **Mastra** server adapter（Express、Hono、Fastify、Koa）用於嵌入。
- **Pipecat Cloud / LiveKit Cloud** 用於託管語音（Lesson 22）。
- **Claude Managed Agents** 用於託管的長時間非同步工作。

### Observability 是承重的 (Observability is load-bearing)

沒有 OpenTelemetry GenAI span（Lesson 23）加上 Langfuse/Phoenix/Opik 後端（Lesson 24），你就無法 debug 在第 40 步失敗的多步驟 agent。這在正式環境不是可選的。這是「我們 debug 得很快」和「我們加更多 logging 然後從頭重播」之間的差別。

### 正式環境 runtime 在哪裡失敗 (Where production runtimes fail)

- **形態選錯。** 對 5 分鐘的任務選了 request-response。使用者掛斷；worker 堆積；重試疊加。
- **沒有 DLQ。** 沒有 dead-letter 的 queue worker。失敗的 job 憑空消失。
- **不透明的背景工作。** 背景 agent 執行時沒有匯出 trace。失敗在使用者回報之前都看不見。
- **跳過 durable state。** 任何超過 30 秒、又無法承受重啟的執行，都需要 durable execution。

```figure
wb-runtime-shapes
```

## 動手做 (Build It)

`code/main.py` 是 stdlib 的多形態 demo：

- Request-response 端點（普通函式）。
- Streaming handler（generator）。
- 帶 DLQ 的 queue worker。
- 事件觸發 registry。
- Cron 形狀的排程器。

執行：

```bash
python3 code/main.py
```

輸出：五份 trace，顯示每種形態在同一任務上的行為。同樣的 agent 邏輯，不同的外殼。Durable execution（第六種形態）刻意留在 Lesson 13 用 LangGraph checkpoint 涵蓋。

## 實際使用 (Use It)

- **Request-response** 用於聊天式 UX。
- **Streaming** 用於漸進式回應。
- **Durable** 用於長時程任務。
- **Queue** 用於批次 / 非同步 / 長時間執行。
- **Event** 用於 agent 的反應性。
- **Cron** 用於例行維護（memory 整併、eval、成本報告）。

## 交付 (Ship It)

`outputs/skill-runtime-shape.md` 為任務挑選 runtime 形態，並接上 observability 需求。

## 練習 (Exercises)

1. 把你的 Lesson 01 ReAct loop 移植到你技術堆疊中的全部六種形態。哪種形態適合哪個產品介面？
2. 為 queue 版 demo 加上 DLQ。模擬 10% 的 job 失敗；呈現 DLQ 大小。
3. 寫一個由 cron 觸發的 eval agent，每晚針對當天最重要的 20 筆 trace 執行。
4. 實作帶 backpressure 的 streaming：如果 client 很慢，就暫停 agent。這與 turn budget 如何互動？
5. 讀 Claude Managed Agents 文件。你什麼時候會把自架的長時程 agent 搬到託管環境？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Request-response | 「同步」 | 使用者等待；僅適用於短任務 |
| Streaming | 「SSE / WS」 | 漸進式輸出；UX 較好；每個 chunk 的延遲可觀測 |
| Durable execution | 「從失敗恢復」 | Checkpoint 的 state；從最後一步重新開始 |
| Queue-based | 「背景 job」 | 生產者 / worker pool / DLQ |
| Event-driven | 「觸發式」 | Agent 對外部事件做出反應 |
| DLQ | 「Dead-letter queue」 | 失敗 job 的停放區 |
| Claude Managed Agents | 「託管 harness」 | Anthropic 託管的長時間非同步，含 caching + 壓縮 |

## 延伸閱讀 (Further Reading)

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：durable execution 的細節
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)：託管的長時間非同步
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)："dozens-to-hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/)：actor-model 故障隔離
