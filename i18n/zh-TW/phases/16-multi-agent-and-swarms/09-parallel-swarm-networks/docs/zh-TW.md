# 平行／Swarm／網路化架構 (Parallel / Swarm / Networked Architectures)

> 與 supervisor 對比：沒有中央決策者。Agent 讀取共享的 event bus、非同步地接工作，再把結果寫回去。LangGraph 明確支援用於去中心化、動態環境的「Swarm Architecture」。Matrix (arXiv:2511.21686) 把控制流與資料流都表示成序列化的訊息，經由分散式 queue 傳遞，以消除 orchestrator 的瓶頸。取捨很明確：用確定性與可追蹤性，換取可擴展性。Swarm 適合有許多獨立子問題的任務；不適合需要單一連貫計畫的任務。

**Type:** Learn + Build
**Languages:** Python (stdlib, `threading`, `queue`)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Time:** ~75 分鐘

## 問題 (Problem)

Supervisor 可以擴展到幾個 worker。那數百個呢？Supervisor 本身會變成瓶頸：每一個關於誰做什麼的決定都要擠過同一個 agent。一個緩慢的計畫步驟就會讓整個系統停擺。

Swarm 架構把設計反過來。不再是中央 planner 派工，而是 worker 從共享的 queue 取工作。「協調」被內建在 event bus 的語意中。沒有 orchestrator；系統可以擴展到 queue 自己的極限。

## 核心概念 (Concept)

### 形狀 (The shape)

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

沒有 orchestrator。每個 worker 重複：拉一個任務、處理、寫入結果（可選擇地再排入後續任務）。

### Swarm 適合的時候 (When swarm fits)

- **許多獨立任務。** 爬取、轉換、分類。任務之間沒有相依關係。
- **執行時間不定的工作。** 如果有些任務花 100ms，有些花 10s，swarm 會自動平衡負載：快的 worker 會去拉下一個工作。Supervisor 則必須預測執行時間。
- **吞吐量優先於確定性。** 你在意的是總完成時間，而不是嚴格的順序。

### Swarm 失敗的時候 (When swarm fails)

- **有順序的工作流程。** 如果第 3 步需要第 2 步的輸出，swarm 有風險讓第 3 步在第 2 步完成前就觸發。
- **需要全域計畫的任務。** 複雜的研究問題受益於 planner。一群 researcher 組成的 swarm 產生的是獨立的事實，而不是連貫的報告。
- **除錯。** 沒有中央 log，加上非同步的工作，重現 bug 的成本很高。

### Matrix (arXiv:2511.21686)

Matrix 是 2025 年的論文，把 swarm 推到它的自然結論：控制流與資料流都是分散式 queue 上的序列化訊息。沒有中央協調者。容錯來自訊息的持久性 (durability)。可擴展性是 message broker 的問題，而不是系統的問題。

貢獻：一種程式設計模型，multi-agent 協調變成「這個 agent 訂閱哪個訊息 topic？」，而不是「supervisor 接下來挑哪個 agent？」。這讓系統看起來像是 pub/sub 的事件網格 (event mesh)。

### 圖框架中的 Swarm (Swarm in graph frameworks)

LangGraph 2025 的文件明確將「Swarm Architecture」列為 multi-agent pattern 之一：agent 是 node，但 edge 形成一張有環的有向圖，任何 node 都可以從 pool 中被啟動。Worker 依條件，而不是依 supervisor 的指派，從可用的工作中挑選。

### 失敗模式：飢餓與熱點 (Failure mode: starvation and hot-spotting)

如果所有 worker 都去拉最快可得的任務，長時間執行的任務就永遠不會被挑中，直到它們是唯一剩下的。這是典型的 queue 飢餓 (starvation)。

緩解方式：
- 帶有明確老化機制的優先權 queue（優先權隨等待時間增加）。
- Worker 專門化：有些 worker 只接「長」任務。
- Back-pressure：限制有多少快速任務進入 queue。

### 與內容導向路由的連結 (The content-based routing link)

Swarm 很自然地與內容導向路由 (content-based routing) 搭配（Lesson 22）。不用通用的 queue，而是每種訊息型別一個 queue。專家 worker 只訂閱自己的型別。這是能擴展到數千個 agent 的 message-bus 架構的基礎。

```figure
sw-work-stealing
```

## 動手做 (Build It)

`code/main.py` 實作一個由 4 個 worker 執行緒組成的 swarm，從共享的 `queue.Queue` 拉工作。任務的執行時間不一（有些快，有些慢）。這個 demo 比較：

- **循序基準：** 一個 worker 循序處理所有任務。
- **固定指派：** 每個任務預先指派給特定的 worker（supervisor 風格）。
- **Swarm：** worker 從共享的 queue 拉取。

Swarm 自動平衡負載；固定指派在被指派的任務很慢時，會讓快的 worker 閒置。

執行：

```
python3 code/main.py
```

輸出顯示每個 worker 的任務數量（swarm 分配不均但最佳）和實際耗時。

## 實際使用 (Use It)

`outputs/skill-swarm-fit.md` 評估某個任務該用 swarm 還是 supervisor。輸入：任務獨立性、執行時間變異、順序需求、可除錯性需求。

## 交付 (Ship It)

檢查清單：

- **帶老化的優先權 queue。** 防止長任務飢餓。
- **Worker 的冪等性 (idempotency)。** 如果 worker 在執行中途當機，任務可能被拉取不只一次。Worker 必須是冪等的。
- **持久的 queue。** 正式環境使用 Kafka、Redis Streams，或以資料庫為後盾的 queue。`queue.Queue` 只存在記憶體裡。
- **每個任務的可觀測性。** 每個任務有一個 trace ID；每個 worker 都用它記錄開始／結束。
- **Back-pressure。** 如果 queue 成長的速度比 worker 排空的速度快，就放慢 producer。

## 練習 (Exercises)

1. 執行 `code/main.py`。在執行時間不定的工作負載上，swarm 比循序快多少？比固定指派快多少？
2. 加入優先權 queue 的變體（使用 `queue.PriorityQueue`）。依任務的「importance」欄位指派優先權。觀察在持續負載下，低優先權的任務會不會飢餓。
3. 實作一個熱點偵測器：當任何 worker 處理的任務數是最慢 worker 的 3× 時，記錄下來。這說明了任務執行時間分布的什麼資訊？
4. 閱讀 Matrix 論文 (arXiv:2511.21686) 的摘要與第 3 節。找出 Matrix 接受的一項具體取捨（可擴展性的增益）和放棄的一項（可追蹤性、確定性）。
5. 把 swarm demo 改成使用 (task_type, payload) 元組的 `queue.Queue`，讓 worker 只訂閱特定型別。當任務是異質的時候，哪些路由規則合理？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Swarm architecture | 「去中心化 agent」 | Worker 從共享的 queue 拉取；沒有中央 orchestrator。 |
| Event bus | 「Agent 訂閱 topic」 | 依型別或內容把任務路由給 worker 的 message broker。 |
| Starvation | 「任務永遠不會執行」 | 低優先權任務永遠不會被挑中，因為較高優先權的工作持續到來。 |
| Hot-spotting | 「一個 worker 被淹沒」 | 負載不平衡，一個 worker 拿到大部分任務。 |
| Back-pressure | 「讓 producer 慢下來」 | 當 queue 滿了，通知上游停止生產的機制。 |
| Idempotent worker | 「重跑也安全」 | 任務處理兩次會產生相同的結果。必須如此，因為 worker 可能在執行中途當機。 |
| Durable queue | 「當機後仍存活」 | 以磁碟或複寫儲存為後盾的 queue；worker 當機時任務不會遺失。 |
| Matrix framework | 「完整的 message-passing swarm」 | 資料與控制流都是分散式 queue 上的序列化訊息。 |

## 延伸閱讀 (Further Reading)

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — 明確的 swarm 支援
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — 完整的 message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — 為什麼某個正式系統明確選擇 supervisor 而不是 swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — 事件驅動的 actor 重寫版，比 v0.2 的 GroupChat 更接近 swarm
