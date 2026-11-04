# 正式環境的擴展：Queue、Checkpoint 與 Durability (Production Scaling — Queues, Checkpoints, Durability)

> 要把 multi-agent 系統擴展到數千個並行的執行，需要 **durable execution**：工作 queue 加上 checkpoint，這樣只要 lease 處理、冪等的副作用和確定性的重播都到位，任何 worker 都能在任何當機之後接手任何一次執行。LangGraph 的 runtime 是參考範例：它在每個 super-step 之後，以 `thread_id` 為鍵寫入一個 checkpoint（預設存到 Postgres）；worker 當機時會釋放 lease，由另一個 worker 接手。Agent 可以無限期地睡著等待人類輸入。**MegaAgent** (arXiv:2408.09955) 為每個 agent 執行一個 producer-consumer queue，有三種狀態（Idle / Processing / Response），並採兩層協調（群組內聊天 + 群組間 admin 聊天）。對 LLM streaming 而言，**fiber/async** 勝過一個工作一個執行緒：執行緒 99% 的時間都閒置在等 token，而 fiber 會在 I/O 時協作式地讓出。反方觀點：Ashpreet Bedi 的 "Scaling Agentic Software" 主張在負載證明有必要之前，**只用 FastAPI + Postgres，別的都不要**，簡單的架構能走得比預期更遠。這堂課建立一個 durable 的 checkpoint log、一個帶狀態轉換的每 agent 工作 queue、一個 async 對 thread 的 demo，並落實務實的「從簡單開始」原則。

**Type:** Learn + Build
**Languages:** Python (stdlib, `asyncio`, `sqlite3`)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Time:** ~75 分鐘

## 問題 (Problem)

一個 multi-agent 系統的原型，在一台筆電上、用記憶體內的 event loop 跑三個 agent 就能運作。你要搬到正式環境：

- Agent 有時會跑好幾個小時（長時間的研究、human-in-the-loop 的等待）。
- Worker 行程會當機。重新啟動會丟失狀態。
- 尖峰負載是平均值的 10 倍；你需要水平擴展。
- 使用者依 agent 執行次數付費；你需要計費的 exactly-once 語意。

記憶體內的 event loop 一項都做不到。你需要底下有一個 durable 的執行層。2026 年的標準選項有：

1. 帶 checkpoint 的 workflow engine（Temporal、LangGraph runtime）。
2. 搭配狀態儲存的 message queue（Postgres + SQS/RabbitMQ）。
3. Actor 模型框架（MegaAgent 每個 agent 一個 producer-consumer）。
4. 手刻的 FastAPI + Postgres（Bedi 的論點）。

這堂課為每一種建一個縮小版。

## 核心概念 (Concept)

### Durable execution 的 pattern (Durable execution, the pattern)

Durable-execution engine 會在每個「步驟」（用 LangGraph 的說法是 super-step）之後，持久化完整的程式狀態。當機時：

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

要讓這個機制運作的需求：

- **可序列化的狀態。** 所有 agent 狀態都必須能持久化。帶著活的資料庫連線的函式 closure 無法存活。
- **確定性的恢復。** 給定相同的狀態與相同的輸入，agent 產生相同的行動（或對 LLM 呼叫交給外部的確定性 oracle）。
- **冪等的副作用。** 外部呼叫（tool call、付款）必須是冪等的，或使用去重 key。

LangGraph 在每個 super-step 之後寫入 checkpoint；Temporal 在每個 activity 之後寫入；Restate 使用事件溯源的 journal。三者實作的都是同一個 pattern。

### 每步一個 checkpoint 的 runtime (A checkpoint-per-step runtime)

LangGraph 的 runtime 是實際的範例：每個 agent 有一個 `thread_id`；狀態是帶型別的 dict；每個 super-step 對 checkpoint 資料表寫入一列。恢復時，runtime 從最後一個 checkpoint 重播，而不是從頭開始。Agent 可以 `interrupt()` 等待人類輸入；runtime 持久化後釋放 worker。輸入到達時，任何 worker 都可以接手。

這是 2026 年 4 月的參考正式設計。

### MegaAgent 的每 agent queue (MegaAgent's per-agent queue)

arXiv:2408.09955 描述了一個規模實驗：一個叢集中有數千個並行的 agent。架構：

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

兩層協調讓群組內的對話可以密集進行，而群組間保持稀疏，這是讓成本在數千個 agent 下維持線性的 pattern。

### Async 對一個工作一個執行緒 (Async vs thread-per-job)

LLM 呼叫是 I/O 密集的。等待下一個 token 的執行緒有 99% 的時間是閒置的。每個執行緒大約花 ~1MB 的 RAM；在 10,000 個並行呼叫時，光是 stack 就要 10GB。

Fiber（Python `asyncio`、Go goroutine、Rust `tokio`）在 I/O 時協作式地讓出。同樣的 10,000 個呼叫可以輕鬆放進行程中。在 LLM-agent 的規模下，async 不是最佳化，而是架構本身。

例外：CPU 密集的後處理（embedding、tokenizer 技巧）仍然需要執行緒或行程。把你的 I/O 層和 CPU 層分開。

### Bedi 的反方觀點 (Bedi's counterpoint)

"Scaling Agentic Software"（Ashpreet Bedi，2026）主張，大多數團隊在量測負載之前就過度工程。務實的預設值：

- FastAPI + Postgres。
- 每次 agent 執行是一列資料；用樂觀並行控制 (optimistic concurrency) 原地更新狀態。
- 背景工作透過 `pg_notify` 或簡單的 Celery worker。
- 重試政策放在應用程式碼中。

對於低於約 ~100 個並行 agent 執行、任務可掌控的負載，這往往就是你需要的全部。等你量測到它失敗了，再升級。

原則：當你碰到簡單架構無法解決的具體問題時，才採用 durable-execution 框架。過早採用，會把時間燒在沒有回報的儀式上。

### Exactly-once 語意 (Exactly-once semantics)

對於付費的 agent 執行，你需要「有效的 exactly-once」（至少一次傳遞 + 冪等的 consumer）。工程上的做法：

- **每次執行一個去重 key。** 在每個副作用呼叫中都包含它。
- **Outbox pattern。** 副作用先寫入資料表，再由獨立的行程執行。兩個步驟都是冪等的。
- **補償交易 (Compensating transaction)。** 當副作用成功但它的追蹤寫入失敗時，排程一次補償。

這些是資料庫工程的 pattern，不是 LLM 專屬的。LLM 的稅只在於 LLM 呼叫很慢；其他一切都是標準的分散式系統。

### Rainbow deployment

Anthropic 的 multi-agent 研究系統使用「rainbow deployment」：多個版本的 agent runtime 並行執行，讓長時間執行的 agent 不必在每次程式碼部署時被終止。把新版本 canary 到一小部分流量；等舊版本的 agent 完成後再汰除。

這對長時間執行、有狀態的系統是標準做法；2026 年的調整是 agent 可以活上好幾個小時，所以部署週期必須配合。

### 標準的正式環境檢查清單 (The canonical production checklist)

- Durable 狀態（checkpoint、snapshot，或 outbox + 可重播的 log）。
- 冪等的副作用。
- 給 LLM 呼叫用的 async I/O 層。
- 搭配去重的至少一次傳遞。
- 針對有狀態工作負載的 rainbow/canary 部署。
- 可觀測性：每個 agent 的追蹤、super-step 稽核、重試計數器。

```figure
sw-checkpoint-replay
```

## 動手做 (Build It)

`code/main.py` 實作：

- `CheckpointStore` — 以 SQLite 為後盾、用 thread-id 為鍵的 checkpoint log。每個 super-step 附加一列。
- `run_with_checkpoint(agent, thread_id)` — 模擬執行中途當機；第二個 worker 從最後的 checkpoint 接手。
- `AgentQueue` — 每個 agent 的 Idle / Processing / Response 狀態機，附一個小型的工作 queue。
- `demo_async_vs_threads()` — 用 asyncio 與執行緒各跑 500 個並行的模擬「LLM 呼叫」；回報實際耗時與尖峰記憶體（近似值）。

執行：

```
python3 code/main.py
```

預期輸出：在模擬當機後 checkpoint 恢復成功；async 版本在 < 1 秒內處理 500 個並行呼叫；執行緒版本要花好幾秒，且每個並行單位用掉的記憶體多了好幾個數量級。

## 實際使用 (Use It)

`outputs/skill-scaling-advisor.md` 針對 durable-execution 的選擇提供建議：FastAPI + Postgres、LangGraph runtime、Temporal，或自訂。依負載、狀態保留需求和部署頻率校準。

## 交付 (Ship It)

標準的正式環境強化：

- **從簡單開始（Bedi 的原則）。** 在你量測到它失敗之前，用 FastAPI + Postgres。
- **最佳化之前先把一切裝上儀表。** 每次執行的延遲直方圖、每步的時間、重試次數、失敗分類。
- **副作用使用 outbox pattern。** 尤其是付款和外部 API 呼叫。
- **Rainbow deploy。** 部署期間絕不終止進行中的 agent 執行。
- **當你碰到具體問題時，才採用 durable-execution engine（Temporal / LangGraph / Restate）：** 長達數小時的 human-in-the-loop 等待、跨區域協調、複雜的重試／補償政策。
- **I/O 層用 async。** 只有 CPU 密集的後處理才用執行緒。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 checkpoint 恢復有效；測量 async 與執行緒在並行上的差異。
2. 實作 **outbox** 資料表：每次 tool call 先寫入 outbox，再由獨立的 goroutine/task 執行。將 tool call 執行兩次來驗證冪等性。
3. 模擬 **rainbow deploy**：兩個並行的 runtime 版本；把新的 thread_id 各分一半給它們；確認舊版本上進行中的 thread 沒有被中斷。
4. 閱讀 LangGraph 的 runtime 文件（連結在下方）。找出 runtime 的哪些功能，在手刻的 FastAPI + Postgres 版本中最久才能複製出來。這是採用的理由，還是你可以延後？
5. 閱讀 MegaAgent (arXiv:2408.09955) 第 3 節。兩層協調（群組內 + 群組間 admin 聊天）是明確的。勾勒你會如何把它對應到有兩組 queue 家族的 message queue。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Durable execution | 「持久化程式狀態」 | Engine 在每個 super-step 之後寫入狀態；當機復原是確定性的。 |
| Super-step | 「交易邊界」 | 兩個 checkpoint 之間的工作單位。LangGraph 的用語。 |
| thread_id | 「Agent 執行的識別碼」 | 綁定 checkpoint 與恢復邏輯的 key。 |
| Idempotency | 「重試也安全」 | 重複一個副作用，產生的結果與只做一次相同。 |
| Outbox pattern | 「解耦副作用」 | 把意圖寫進資料表；由獨立的執行者執行並標記完成。 |
| At-least-once delivery | 「可能有重複」 | Message queue 的語意；去重 key 讓 consumer 達到有效的一次。 |
| Rainbow deploy | 「重疊的版本」 | 長時間執行的工作負載期間，多個 runtime 版本並行。 |
| Async fiber | 「協作式讓出」 | 使用者模式的並行；對 I/O 密集的負載而言，比執行緒便宜。 |
| Checkpoint | 「狀態快照」 | 在 super-step 邊界序列化的狀態；恢復的關鍵。 |

## 延伸閱讀 (Further Reading)

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime 的設計
- [MegaAgent](https://arxiv.org/abs/2408.09955) — 每個 agent 的 producer-consumer queue；數千個並行 agent 下的兩層協調
- [Matrix](https://arxiv.org/abs/2511.21686) — 以 message queue 作為協調基礎的去中心化框架
- [Temporal docs](https://docs.temporal.io/) — durable execution 的參考 workflow engine
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 包含 rainbow deployment 的正式環境經驗
