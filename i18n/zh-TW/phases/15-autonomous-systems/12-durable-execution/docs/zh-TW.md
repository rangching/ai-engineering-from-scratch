# 長時間執行的背景 Agent：Durable Execution (Long-Running Background Agents)

> Production 的 long-horizon agent 不會跑在 `while True` 裡。每一次 LLM 呼叫都會變成一個帶有 checkpoint、retry 與 replay 的 activity。Temporal 與 OpenAI Agents SDK 的整合在 2026 年 3 月 GA。Claude Code Routines (Anthropic) 不需要常駐的本機行程，就能執行排程的 Claude Code 呼叫。Session 可以在等待人類輸入時暫停、撐過部署，並從以 `thread_id` 為 key 的最新 checkpoint 繼續。新的使用體驗背後，是一個老模式：workflow orchestration，加上一個新的輸入：LLM 呼叫是非確定性的 activity，在復原時必須被確定性地 replay。

**Type:** Learn
**Languages:** Python (stdlib, minimal durable-execution state machine)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~60 分鐘

## 問題 (The Problem)

想像一個跑四小時的 agent。它呼叫三個工具、詢問使用者兩次，並做了四十次 LLM 呼叫。跑到一半，它所在的主機重新開機。會發生什麼事？

- 在天真的 `while True` loop 中：一切都沒了。執行從頭開始。三次工具呼叫（有真實的副作用）再執行一次。使用者又被問到他們已經核准過的事。四十次 LLM 呼叫再被計費一次。
- 使用 durable execution：執行從最近的 checkpoint 繼續。已完成的 activity 不會重新執行；它們的結果從持久的 log 中 replay。使用者不需要重新核准已核准過的事。已經做過的 LLM 呼叫不會重複計費。

這是 workflow engine 已經用了十年的同一套模式（Temporal、Cadence、Uber 的 Cherami）。新的地方在於，LLM 呼叫現在是一種 activity：非確定性、昂貴、有副作用，而且很貼合這個模式。

這堂課反覆出現的主題：long-horizon 的可靠度會衰退（METR 觀察到「35 分鐘退化」，成功率大致隨 horizon 以二次方下降）。Durable execution 讓執行時間可以超過可靠度特性所能支撐的長度，如果設計正確，這是一種新的安全失敗方式；如果設計錯誤，就是不安全的方式。

## 核心概念 (The Concept)

### Activity、workflow 與 replay

- **Workflow**：確定性的 orchestration 程式碼。定義 activity 的順序、分支、等待。必須是確定性的，這樣才能從 event log replay，而不會出現令人意外的分歧。
- **Activity**：非確定性、可能失敗的工作單元。LLM 呼叫、工具呼叫、檔案寫入、HTTP 請求。每個 activity 都會連同輸入，以及（完成後的）輸出一起被記錄。
- **Event log**：持久的後端儲存。每個 activity 的開始、完成、失敗、重試，以及每個 workflow 決策都會被記錄。
- **Replay**：復原時，workflow 程式碼從頭重新執行；每個已完成的 activity 直接回傳記錄下來的結果，不會重新執行。只有尚未完成的 activity 才真的會被執行。

這和 React 對照 virtual DOM 重新渲染，或 Git 從 commit 重建工作樹是同樣的形狀。Orchestrator 的確定性，是讓持久性變便宜的關鍵。

### 為什麼 LLM 呼叫符合這個模式

LLM 呼叫是：
- 非確定性的（temperature > 0；即使 temperature 0，不同模型版本之間也會漂移）。
- 昂貴的（金錢與延遲）。
- 可能失敗的（rate limit、timeout）。
- 有副作用的（如果它們會呼叫工具）。

這正是 activity 的特徵。把每個 LLM 呼叫包成 activity，就能得到帶有 exponential backoff 的 retry、跨重啟的 checkpoint，以及可 replay 的偵錯軌跡。

### 以 `thread_id` 為 key 的 checkpoint

LangGraph、Microsoft Agent Framework、Cloudflare Durable Objects 與 Claude Code Routines，全都收斂到同一種 API 形狀：用 `thread_id`（或等價物）識別 session；每次狀態轉換都持久化到後端（預設 PostgreSQL，開發用 SQLite，快取用 Redis）；resume 時讀取最新的 checkpoint。

後端的選擇很重要：

- **PostgreSQL**：持久、可查詢、撐得過部署。LangGraph 的預設。
- **SQLite**：僅限本機開發；跨主機會遺失資料。
- **Redis**：很快，但除非設定 AOF/snapshot，否則是暫時性的。
- **Cloudflare Durable Objects**：透明地分散式；以唯一的 key 劃定範圍；可存活數小時到數週。

### 把人類輸入當作一級狀態

Propose-then-commit（第 15 課）需要一個持久的「等待人類」狀態。Workflow 暫停，外部佇列保存待處理的請求，核准後會從那個確切的點繼續。沒有持久性，這只是盡力而為；有了它，隔夜的核准抵達後，workflow 會在早上接續。

### 35 分鐘退化

METR 觀察到，每一類被量測的 agent，在連續運作約 35 分鐘之後，可靠度都會衰退。任務時間加倍，失敗率大約變成四倍。Durable execution 無法修正這點；它只是讓你能跑得比可靠度特性支撐的更久。安全的模式，是把持久性與「重新進入時需要新的 HITL」的 checkpoint 結合，並搭配預算 kill switch（第 13 課），不論牆上時間多長，都能限制總運算量。

### 何時 durable execution 是錯誤的答案

- 短於幾分鐘、沒有人類輸入的執行。額外負擔大於好處。
- 嚴格唯讀的資訊檢索。
- 正確性需要在單一 context window 內端到端完成的任務（某些推理任務；某些一次性生成）。

```figure
memory-consolidation
```

## 實際使用 (Use It)

`code/main.py` 用 stdlib Python 實作了一個最小的 durable-execution engine。它支援：

- `@activity` decorator，把輸入與輸出記錄到 JSON event log。
- 一個把 activity 排序的 workflow 函式。
- 一個 `run_or_replay(workflow, event_log)` 函式，replay 已完成的 activity 而不重新執行它們。

驅動程式模擬一個三個 activity 的 workflow，在中途當機，並顯示 (a) 天真的 retry 重新執行所有東西，與 (b) replay 只執行缺少的那個 activity。

## 交付 (Ship It)

`outputs/skill-durable-execution-review.md` 審查提議中的長時間執行 agent 部署，檢查 durable execution 的形狀是否正確：activity、確定性、checkpoint 後端、人類輸入狀態，以及 resume 時的 HITL 政策。

## 練習 (Exercises)

1. 執行 `code/main.py`。觀察天真 retry 與 replay 之間 activity 執行次數的差異。改變當機點，並展示 replay 的次數也隨之改變。

2. 把玩具 engine 轉成明確使用 `thread_id`。模擬兩個共用 engine 的並行 session，並確認它們的 event log 不會互相衝突。

3. 挑玩具 engine 中的一個 activity。引入一個非確定性（在 workflow 決策中使用牆上時鐘的時間戳記）。展示 replay 時的分歧。說明真實的 engine 如何處理這點（副作用註冊、`Workflow.now()` API）。

4. 閱讀 LangChain 的「Runtime behind production deep agents」文章。列出 runtime 持久化的每個狀態，並指出每個狀態涵蓋哪一種失敗模式。

5. 為一個 6 小時的自主 coding 任務設計 checkpoint 政策。你在哪裡做 checkpoint？當機後的 resume 長什麼樣子？什麼需要新的 HITL？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Workflow | 「Agent 的腳本」 | 確定性的 orchestration 程式碼；可從 event log replay |
| Activity | 「一個步驟」 | 非確定性的單元（LLM 呼叫、工具呼叫）；前後都會被記錄 |
| Event log | 「後端儲存」 | 每個狀態轉換的持久記錄 |
| Replay | 「Resume」 | 重新執行 workflow；已完成的 activity 回傳記錄的結果，不重新執行 |
| Checkpoint | 「存檔點」 | 以 thread_id 為 key 的持久化狀態；resume 時以最新者為準 |
| thread_id | 「Session key」 | 界定持久狀態範圍的識別碼 |
| 35-minute degradation | 「可靠度衰退」 | METR：成功率大致隨 horizon 以二次方下降 |
| Non-determinism | 「Replay 時的漂移」 | 牆上時鐘、隨機、LLM 輸出；必須註冊為副作用 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 預算、輪數與 resume 語意。
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — RequestInfoEvent 的形狀。
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — 具體的 runtime 需求。
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — LLM 呼叫的 activity 形狀。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 35 分鐘退化的參考來源。
