# Checkpoint 與 Rollback (Checkpoints and Rollback)

> 每一次 graph-state 轉換都會持久化。當 worker 當機，它的 lease 會過期，另一個 worker 會從最新的 checkpoint 接手。Cloudflare Durable Objects 可以跨數小時或數週保存狀態。Propose-then-commit（第 15 課）為每個動作定義 rollback 計畫。動作後的驗證讓整個迴圈閉合。EU AI Act 第 14 條規定高風險系統必須有有效的人類監督，實務上這表示 checkpoint 必須可查詢、rollback 必須演練過，而且稽核軌跡必須撐得過一次部署。最尖銳的失敗模式是：沒有 idempotency key 與前置條件檢查，暫時性失敗後的 retry 可能讓已核准的動作被執行兩次。動作後驗證就是用來抓這個的。

**Type:** Learn
**Languages:** Python (stdlib, checkpoint and rollback state machine)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 15 (Propose-then-commit)
**Time:** ~60 分鐘

## 問題 (The Problem)

Durable execution（第 12 課）讓當機的 agent 可以繼續執行。Propose-then-commit（第 15 課）讓已核准的動作可以被稽核。這堂課把兩者接起來：當一個已核准的動作只執行了一部分、當機，然後繼續時，會發生什麼事？Rollback 何時執行，又是針對什麼狀態？

真實的系統接線方式各不相同：

- **LangGraph** 把每次 graph-state 轉換都 checkpoint 到 PostgreSQL。Worker 當機時，lease 釋放，另一個 worker 從最新的 checkpoint 繼續。Workflow 在 `interrupt()` 上暫停，而 `interrupt()` 本身也會持久化。
- **Cloudflare Durable Objects** 跨數小時或數週保存每個 key 的狀態。把已核准動作的運算與儲存放在同一處。
- **Microsoft Agent Framework** 在 workflow API 中提供 `Checkpoint` 基本元件；replay 加上 idempotency 涵蓋了 retry。

在每個案例中，真正有效的組合是：idempotency key（防止重複執行）+ 前置條件檢查（狀態仍是我們核准時所依據的狀態）+ 動作後驗證（副作用真的發生了）+ 驗證失敗時 rollback。

## 核心概念 (The Concept)

### 每次轉換都持久化

Graph-state 轉換是指任何把 workflow 從一個具名狀態移到另一個具名狀態的步驟。天真的實作只在特定的 commit 點持久化；production 的實作則持久化每次轉換。成本（多幾次寫入）相對於可靠度的收益（replay 可以落在任何地方、lease 復原很精確）來說很小。

### Lease 復原

Worker 當機時，workflow 不會遺失；lease（一個短暫的宣告，表示這個 worker 正在執行這次 run）只是過期了。另一個 worker 接手最新的 checkpoint 並繼續。Lease 機制讓 production 系統能撐過滾動部署，而不遺失正在處理中的工作。

### Idempotency 加上前置條件

只有 idempotency 並不夠。考慮：某個 workflow 被核准「當餘額 > $1000 時，從 A 轉 $100 到 B」。Workflow 已 commit，在執行中途當機，然後繼續。如果只檢查 idempotency key，而執行繼續，轉帳只會執行一次（正確）。但想想看，在當機與繼續之間，A 的餘額因為另一個 workflow 降到了 $500。Idempotency 檢查仍然通過；前置條件卻不成立。沒有前置條件檢查，我們就出貨了一筆透支。

每個有後果的動作都需要兩者：

- **Idempotency key**：防止重複執行。
- **前置條件檢查**：確認狀態仍與被核准的內容一致。

### 動作後驗證

「工具回傳 200」不是驗證。真正的驗證會重新讀取目標狀態，確認副作用真的發生了。模式：

- 資料庫更新：`UPDATE ... RETURNING *`，然後斷言回傳的資料列符合預期狀態。
- 寄送電子郵件：提交後，在寄件備份中檢查訊息 ID。
- 寫入檔案：讀回檔案並計算 hash。
- API 呼叫：對目標資源做後續的 `GET`。

如果 verify 失敗，workflow 處於已知的壞狀態。Rollback 啟動。

### Rollback 計畫

Propose-then-commit（第 15 課）中的每個有後果的動作，都帶有 rollback 計畫。類型：

- **In-band rollback**：直接反轉副作用（`INSERT` 之後 `DELETE`、寄出後再寄更正信）。
- **Compensating transaction**：一個新的動作，用來抵銷原本的動作（標準的 SAGA 模式）。
- **Out-of-band rollback**：通知人類、暫停 workflow、把壞狀態留著供調查。

無法 rollback（「我們無法復原這個」）必須在提議中明確寫出。沒有 rollback 的動作，在 commit 時需要更強的 HITL（第 15 課的 challenge-and-response）。

### EU AI Act 第 14 條的實作解讀

第 14 條要求高風險系統必須有「有效的人類監督」。在實作層面，實作者將它解讀為：

- Checkpoint 可以被稽核者查詢。
- Rollback 經過演練（至少端到端測試過一次）。
- 稽核軌跡撐得過一次部署（checkpoint 後端不是暫時性的）。
- 驗證失敗會發出警報，而不是默默地記在 log 裡。

一個在 commit 中途當機、繼續後完成副作用、卻沒有 verify + rollback 路徑的 workflow，通不過第 14 條的檢驗。

### 尖銳的失敗模式：重複執行

這個領域最常見的 production 事故：

1. 動作被核准，idempotency key 為 k。
2. Commit 開始、執行、回傳 200。
3. Workflow 在持久化「已 commit」狀態之前當機。
4. Workflow 繼續；看到「已核准但未 commit」；重新執行。
5. 副作用觸發兩次。

緩解方式：在執行前先持久化「執行中」的意圖，用 idempotency key 執行，只有在動作後驗證成功後才標記「已 commit」。如果動作觸發了但狀態寫入失敗，你就知道要去驗證，並（必要時）重新觸發。如果狀態寫入成功但動作失敗，你透過復原路徑驗證，並只觸發一次。

```figure
checkpoint-replay
```

## 實際使用 (Use It)

`code/main.py` 實作了一個帶有 idempotency、前置條件、verify 與 rollback 的 checkpoint workflow。驅動程式模擬四種情境：乾淨的執行、當機後 retry（idempotency 抓到）、前置條件失敗（workflow 在觸發前中止）、verify 失敗（rollback 觸發）。

## 交付 (Ship It)

`outputs/skill-rollback-rehearsal.md` 為提議中的 workflow 設計 rollback 演練測試，並稽核 checkpoint 後端的稽核軌跡是否持久。

## 練習 (Exercises)

1. 執行 `code/main.py`。驗證這四種情境。針對 commit 期間當機的案例，確認動作在多次 retry 中恰好觸發一次。

2. 修改「先標記完成，再執行」的模式，讓狀態寫入在動作之後才觸發。重新執行當機情境。量測有多少重複的動作被觸發。

3. 為一個特定的 production 動作（例如「發文到 Slack 頻道」）設計 rollback 計畫。分類為 in-band、compensating 或 out-of-band。說明選擇的理由。

4. 選一個你熟悉的 workflow。找出每一個狀態轉換。為每個轉換標上持久性需求（持久化 / 不持久化）。數一數你目前沒有持久化的有幾個。

5. 演練 rollback 的測試：設計一個端到端測試，執行真實的 workflow、讓它當機，並確認 rollback 路徑會觸發。這個測試要斷言什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Checkpoint | 「存檔點」 | 每次 graph-state 轉換都持久化到持久儲存 |
| Lease | 「Worker 宣告」 | 短暫的宣告，表示某個 worker 正在執行一次 run；當機時過期 |
| Precondition | 「狀態閘門」 | 斷言狀態仍與已核准的動作一致 |
| Post-action verify | 「重新讀取檢查」 | 確認副作用真的在目標系統中發生了 |
| In-band rollback | 「直接復原」 | 用反向操作反轉副作用 |
| Compensating transaction | 「SAGA 復原」 | 一個抵銷原本動作的新動作 |
| Mark-as-done-first | 「狀態寫入順序」 | 在 commit 回傳之前，先持久化已 commit 的狀態 |
| Article 14 | 「EU AI Act 人類監督」 | 實作面：可查詢的 checkpoint、演練過的 rollback、可稽核的軌跡 |

## 延伸閱讀 (Further Reading)

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint 基本元件與 lease 復原。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — 以 Durable Objects 作為狀態基底。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 法規基準。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon workflow 的可靠度框架。
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — Claude Code Routines 的 workflow 形狀。
