# MCP 的可靠性、取消與流量控制 (MCP Reliability, Cancellation, and Flow Control)

> Request ID 只用來對應訊息。它不會讓副作用變安全、不會停止 worker，也不會保護 stream 不受慢速消費者影響。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 09 and 13
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 為 stdio 與 Streamable HTTP 實作正確的取消訊號。
- 解決完成與取消之間的 race，且不在取消之後送出訊息。
- 把請求取消與持久化的 `tasks/cancel` 語意分開。
- 根據副作用與明確的冪等性 key 建立重試決定。
- 在保留最終 response 的同時，為 progress queue 設定上限。
- 透過重新連線、重新取得與帶 jitter 的 backoff 來復原 stream。

## 問題 (The Problem)

順利的路徑會藏住最昂貴的分散式系統 bug。

Client 呼叫 tool。Server 開始工作。Progress 抵達。Proxy 緩衝了 stream。Client 達到 timeout 並中斷連線。Server 在一毫秒後完成。Client 用新的 JSON-RPC id 重試。變更被執行了兩次。

每個元件在本地的行為都正常。整個系統卻在全域上失敗了。

MCP 定義訊息與 transport 的行為，但你的應用程式仍然擁有：

- 時間預算；
- 商業冪等性；
- 有界限的 queue；
- 重試分類；
- 持久化的 task 狀態；
- 重新連線與重新取得的政策。

這堂課把這些決定建構到一個確定性的 simulator 中。沒有 sleep、沒有 socket、也沒有隨機失敗。你直接控制取消事件的順序。一個同步的 thread 測試會強迫兩個 ledger client 競爭同一個冪等性 key。

## 請求取消取決於 transport (Request Cancellation Is Transport-Specific)

意圖在每個 transport 上都一樣：client 不再需要進行中的結果。Wire 上的訊號則不同。

### stdio

stdio 使用一條共用的雙向通道。Client 送出 notification：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/cancelled",
  "params": {
    "requestId": 41,
    "reason": "User closed the operation"
  }
}
```

這個 notification 是送出後不管（fire-and-forget）。Server 不會對它發出 JSON-RPC response。

Server 應該停止工作、釋放資源，並避免為已取消的請求送出 response。當請求未知、已經完成，或無法安全停止時，它可以忽略取消。

格式錯誤、未知與已完成的取消 notification 會被忽略。把這些 race 轉成新的錯誤，只會製造更多 race。

### Streamable HTTP

現代的 Streamable HTTP 讓每個請求都有自己的 HTTP response 或 SSE response 串流。Client 透過關閉該請求的 response 串流來取消。

不要為一般的 HTTP 請求 POST `notifications/cancelled`。關閉串流就是取消訊號。

Server 一旦觀察到斷線，就應該停止工作，而且不得再為該請求送出更多訊息。

### Server 發送的取消範圍很窄

Server 不會用 `notifications/cancelled` 取消任意的 client call。在 stdio 上，server 發送的取消僅保留給終止 `subscriptions/listen` 請求。讓這條路徑與一般的 client 請求取消保持分開。

## 取消是一場 race (Cancellation Is a Race)

兩種事件順序都是有效的。

### 取消勝出

```text
request starts
client sends cancellation signal
server marks request cancelled
worker reaches completion
server suppresses the response
```

### 完成勝出

```text
request starts
worker commits the result
server sends the response
cancellation arrives late
server ignores the late notification
```

Client 也必須忽略已被它放棄的請求的遲到 response。網路延遲意味著任何一方都無法證明對方先觀察到哪個事件。

```figure
mcp-reliability-race
```

這堂課的 `RequestCoordinator` 只儲存一個終結狀態。`complete()` 在取消之後不回傳 response。遲到的取消無法改變已完成的紀錄。

## Timeout 需要兩個時鐘 (Timeouts Need Two Clocks)

單一個 idle 計時器是不夠的。

使用兩個限制：

1. **Idle timeout。** 請求可以在多長時間內沒有任何有用的活動。
2. **Maximum timeout。** 從請求開始算起的絕對 wall-clock 預算。

Progress 可以重設 idle 時鐘。但它絕不能移除最大期限。

```text
start: 0 ms
progress: 400 ms
progress: 800 ms
progress: 1200 ms
idle timeout: 500 ms
maximum timeout: 2000 ms
```

在 1500 ms 時，請求仍然是 active 的，因為最新的 progress 只有 300 ms 前。在 2000 ms 時，即使另一個 progress 事件在 1999 ms 抵達，最大期限也會取消它。

Progress 是選擇性的。Server 可以接受 progress token 卻不送出任何更新。絕不要因為 token 存在，就把它變成無限的 timeout。

MCP 的 progress 值必須遞增。完成或取消之後，notification 就停止。對 progress 做 rate limit，讓快速的 worker 無法灌爆 transport。

## 請求取消不是 `tasks/cancel` (Request Cancellation Is Not `tasks/cancel`)

這些機制解決的是不同的生命週期。

| 機制 | 目標 | 訊號 | 成功的意義 |
|-----------|--------|--------|--------------------|
| stdio 上的請求取消 | 一個進行中的 RPC | `notifications/cancelled` | Client 放棄了請求；server 在可行時應該停止 |
| HTTP 上的請求取消 | 一個進行中的 response 串流 | 關閉串流 | Client 放棄了請求；server 在可行時應該停止 |
| `tasks/cancel` | 一個持久化的 Task | 一般的 MCP 請求 | Server 確認了取消的意圖 |

成功的 `tasks/cancel` 結果並不證明 worker 已經停止。Task 可能一直維持 `working`，直到 worker 的 checkpoint 觀察到旗標。工作也可能在該 checkpoint 之前就完成。

當 HTTP 連線關閉時，不要抹除持久化的 task 狀態。建立 Task 的理由，正是它的生命週期比一個請求和一條連線更長。

## 新的 JSON-RPC ID 不是冪等性 (A New JSON-RPC ID Is Not Idempotency)

JSON-RPC id 用來對應請求與 response。它們不識別商業操作。

假設 client 用 id `41` 提交一筆扣款，遺失了 response，並用 id `42` 重試。Server 看到兩則不同的訊息。沒有應用程式的 key，它無法知道這代表同一次結帳。

冪等性 key 識別的是商業意圖：

```json
{
  "name": "charge_account",
  "arguments": {
    "account": "acct-7",
    "cents": 1200,
    "idempotencyKey": "checkout-7"
  }
}
```

Server 儲存：

- key；
- 操作參數的指紋；
- 已 commit 的結果。

相同的 key 與相同的參數，會回傳儲存的結果。相同的 key 但參數不同，則會被拒絕。這避免意外的 key 重用，去改動不同的商業操作。

### Ledger 邊界必須是 atomic 且持久的

這個順序是不安全的：

```text
check key
run mutation
store result
```

兩個 worker 可能都觀察到 key 不存在，然後都執行變更。在效果發生之後、儲存之前當機，重試時會產生同樣的模糊性。

這堂課使用以檔案為基礎的 SQLite ledger。`BEGIN IMMEDIATE` 把 key 檢查、模擬的商業效果、執行計數器與儲存的結果序列化到同一個 transaction 中。因此，兩個獨立的 ledger 連線以相同的 key 競爭時，會觀察到一個已 commit 的結果與一次執行。關閉並重新開啟 ledger 會保留那筆紀錄。

每個回傳值都是從儲存的 JSON 重建的。呼叫者永遠不會收到 ledger 所持有的可變物件，所以改動回傳的 dictionary，不會破壞之後 replay 的結果。

Simulator 的商業效果，是同一個 SQLite transaction 內的收據與執行計數器。真正的付款、部署或外部 API call，不會只因為寫入了本地資料表就變成 atomic。Production 需要持久的共用資料庫 transaction、transactional outbox，或強制執行相同冪等性 key 的上游 provider。單靠 process lock 無法保護多個 replica，也撐不過重啟。

### 重試矩陣

在實作重試之前，先分類。

| 類別 | 範例 | 重試規則 |
|------|---------|------------|
| 安全 | 沒有副作用的確定性讀取 | 在了解失敗邊界之後，用新的 JSON-RPC id 重試 |
| 有條件 | 帶有持久冪等性 key 的變更 | 用相同的 key 與相同的參數重試 |
| 不安全 | 沒有商業去重的變更 | 不要自動重試；先對帳 |

`readOnlyHint` 與 `idempotentHint` 這類 tool annotation，仍是不受信任的提示。重試是否安全，由應用程式合約與 server 實作決定。

## Backpressure 是正確性的一部分 (Backpressure Is Part of Correctness)

SSE 的生產者產生 progress 的速度，可能比 client、proxy 或網路能消耗的更快。沒有上限的 queue 會把慢速變成記憶體耗盡。

使用有界限的 queue，並定義什麼可以遺失。

Progress 是可取代的。對同一個 token，較晚的 progress 值取代較早的。最終的 JSON-RPC response 則不可取代。

這堂課的 buffer 套用這個政策：

1. 合併同一個 token 相鄰的 progress。
2. 達到容量時，丟棄最舊的 progress。
3. 標記該串流需要重新取得具權威性的資料。
4. 保留最終 response。
5. 拒絕那種為了保留最終 response 而必須丟棄另一個最終 response 的狀態。

這是帶有明確復原機制的有界限遺失。默默遺失不是策略。

### Proxy 緩衝

Server 可能正確地串流，而 reverse proxy 卻把事件留在 buffer 中。

對於 SSE response，送出：

```http
Content-Type: text/event-stream
Cache-Control: no-cache
X-Accel-Buffering: no
```

2026 年的 Streamable HTTP 規格建議使用 `X-Accel-Buffering: no`，讓相容的 proxy 立即遞送事件。

對於安靜的長壽命串流，定期送出 SSE 註解：

```text
:
```

Client 忽略註解行。中介者看到流量，比較不會關閉閒置的連線。

Keepalive 不是 progress。不要只因為收到 transport 註解，就重設操作語意上的 idle timeout。

## 重新連線就是重新取得 (Reconnect Means Refetch)

現代的 Streamable HTTP 不支援透過 `Last-Event-ID` 恢復的 SSE。

在 `subscriptions/listen` 串流中斷之後：

1. 用新的 JSON-RPC id 開啟新的 listen 請求。
2. 還原想要的訂閱 filter。
3. 從具權威性的 method 重新取得受影響的 tool、resource、prompt 或 Task。
4. 依穩定的識別碼對應用程式狀態去重。
5. 不要只因為 response 遺失，就重放不安全的變更。

範例復原計畫明確地把 `sendLastEventId` 設為 false，並列出要重新取得的 resource。

### 防止重新連線的驚群效應

如果 10,000 個 client 剛好在一秒時重新連線，正在復原的 server 會再次失敗。

使用帶 jitter 與上限的指數 backoff。這堂課依 client id 與嘗試次數計算確定性的 jitter，讓測試可重現：

```text
attempt 0: up to 250 ms
attempt 1: up to 500 ms
attempt 2: up to 1000 ms
...
cap: 8000 ms
```

Production 可以使用密碼學安全的隨機數或 runtime 的隨機數。不變量是分布，而不是特定的公式。

## 動手做 (Build It)

`code/main.py` 建構五個小型的可靠性元件。

### `RequestCoordinator`

- 以 idle 與最大期限啟動一個進行中的請求；
- 發出單調遞增的 progress notification；
- 產生正確的 stdio 或 HTTP 取消訊號；
- 忽略無效的取消 notification；
- 讓取消與完成的終結 race 變得明確；
- 把 server 發送的取消保留給 stdio 訂閱。

### `MutationLedger`

- 證明在沒有商業 key 的情況下，兩個 JSON-RPC id 會執行兩次；
- 對 key 檢查、模擬的效果、執行計數器與結果 commit，使用以檔案為基礎的 SQLite transaction；
- 跨獨立的 ledger 連線，對同一個冪等性 key 下相符的參數去重；
- 拒絕以不同參數重用同一個 key；
- 回傳防禦性副本，並在重新開啟後保留已 commit 的紀錄。

### `DurableTaskService`

- 確認取消請求；
- 讓 task 維持 `working`，直到 worker 的 checkpoint；
- 示範為什麼確認不等於最終狀態。

### `BoundedSseBuffer`

- 在壓力下合併或丟棄 progress；
- 記錄需要重新取得具權威性的資料；
- 絕不丟棄最終 response。

### 復原輔助工具

- 回傳對 proxy 安全的 SSE header 與 keepalive 註解；
- 建立重新連線與重新取得的計畫；
- 用確定性的指數 backoff 加 jitter 來分散重試。

## 實際使用 (Use It)

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/code
python3 main.py
python3 -m unittest discover tests -v
```

Demo 會執行核心 race 的兩邊、在暫時的以檔案為基礎的 ledger 中執行有 transaction 去重的變更、讓有界限的 progress buffer 過載，並顯示持久化的 Task 如何從已確認的取消，走到 worker 觀察到的取消。

## 互動實驗 (Interactive Lab)

在不加入 sleep 的情況下，執行四種事件順序。

1. 啟動請求 `A`，取消它，然後呼叫 `complete()`。
2. 啟動請求 `B`，完成它，然後遞送取消。
3. 啟動請求 `C`，在每個 idle 期限之前送出 progress，然後越過最大期限。
4. 透過 Streamable HTTP 啟動請求 `D`，並關閉它的 response 串流。

為每個情境記錄：

- 請求的終結狀態；
- 是否存在最終 response；
- 放在 wire 上的取消訊號；
- client 應該忽略哪個事件。

然後把 `D` 改成 stdio。操作相同，但取消訊號必須改變。

## 實作練習 (Practice Lab)

為 `MutationLedger` 新增 `reserve_inventory` 變更。

需求：

1. Key 綁定 SKU、數量、tenant 與操作名稱。
2. 用相同的 key 與相同的參數重試，會回傳第一次的預留。
3. 用改變的數量重試會失敗，且不會再次預留。
4. 已 commit 但遺失 response 的執行，可以依 key 對帳。
5. 結果不記錄機密或付款資料。
6. 當 client 沒有提供 key 時，停用自動重試。
7. 加入模擬的訂閱中斷，並在決定下一步之前，重新取得庫存紀錄。
8. 在一個 barrier 啟動兩個 ledger 連線，並同時提交相同的 key。斷言只 commit 了一次預留。
9. 改動第一個回傳的預留物件。重放該 key，並證明儲存的結果沒有改變。
10. 關閉並重新開啟 ledger 檔案，然後依 key 對帳該預留。

保持實驗誠實：如果庫存存在於另一個服務中，請說明該服務是否接受相同的冪等性 key，或是否由 transactional outbox 把本地的 commit 橋接到遠端的效果。

## 交付的成品 (Shipped Artifact)

`outputs/skill-mcp-reliability-reviewer.md` 是單一檔案的可靠性審查 skill。給它一個 MCP 操作、transport、timeout 政策、重試行為、queue 政策與復原計畫。它會回傳 race 表、重試分類、冪等性邊界、流量控制檢查，以及失敗 fixture。

## 驗證 (Verify It)

當以下敘述都為真時，這堂課就完成了：

- stdio 取消會送出 `notifications/cancelled`，且不會收到 response。
- Streamable HTTP 取消會關閉請求串流，且不送出取消 POST。
- 先取消、後完成，會抑制最終 response。
- 先完成、後取消，會保留 response 並忽略遲到的取消。
- Progress 可以重設 idle timeout，但絕不能重設最大 timeout。
- 單單是新的 JSON-RPC id，會讓變更再次執行。
- 在兩個連線同時 race 的情況下，一個冪等性 key 加上相同的參數只會執行一次。
- 已 commit 的紀錄在重新開啟後仍然存在，而 replay 會回傳防禦性副本。
- 改動一個回傳的結果，無法改變儲存的結果。
- 有界限的 buffer 維持在容量之內，並保留最終 response。
- 重新連線使用新的請求，不送出 `Last-Event-ID`，並重新取得受影響的狀態。
- `tasks/cancel` 的確認會讓 task 維持非終結狀態，直到 worker 觀察到它。

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 可觀察的症狀 | 正確的回應 |
|---------|--------------------|------------------|
| HTTP client POST 取消 notification | Server 與 client 對請求的生命週期看法不一 | 關閉請求的 SSE response 串流 |
| Server 在接受取消之後仍然回應 | Client 收到無法使用的遲到結果 | 取消勝出時，停止工作並抑制後續訊息 |
| Progress 重設每個期限 | 卡住的工作永遠存活 | 保留獨立的絕對最大 timeout |
| 新的 RPC id 被當成去重 | 扣款、部署或刪除執行兩次 | 加入持久的應用程式冪等性 key |
| Key 檢查與效果是分開的 | 並行的 worker 都觀察到 key 不存在 | 以 atomic 方式 commit key claim、效果紀錄與結果 |
| 跨 replica 使用記憶體內的 ledger | 重啟或另一個 worker 忘記先前的 commit | 使用共用的持久化儲存或上游冪等性 |
| 直接回傳儲存的可變結果 | 呼叫者的改動破壞之後的 replay | 序列化已 commit 的結果並回傳防禦性副本 |
| Key 在參數改變時被重用 | 一個 key 對應到兩個商業意圖 | 儲存並比對參數指紋 |
| 沒有上限的 progress queue | 記憶體隨慢速消費者上升 | 在上限內合併並丟棄可取代的 progress |
| 最終 response 在壓力下被丟棄 | Client 無法得知請求的結果 | 預留容量或淘汰 progress，絕不淘汰最終 response |
| Proxy 緩衝 SSE | Progress 成批抵達，或在 timeout 之後才到 | 停用緩衝，並設定相容的 proxy timeout |
| 假設有 `Last-Event-ID` | Client 從 server 不支援的狀態恢復 | 以新的請求重新連線並重新取得 |
| 每個 client 立即重新連線 | 復原造成另一次中斷 | 使用帶上限的指數 backoff 加 jitter |
| Task 確認被當成最終取消 | UI 說已停止，worker 卻仍在執行 | 輪詢 Task 直到終結狀態 |

## 與 Capstone 的連結 (Capstone Connection)

工具生態系的 capstone 應該把可靠性視為可執行的證據，而不是架構圖中的一段文字。

要求這些成品：

- 每個 transport 一份取消 race 的 transcript；
- 每個公開變更的重試表；
- 冪等性 key 紀錄與不一致的 fixture；
- 同一個 key 並行的 transcript、重新開啟檢查，以及變更別名檢查；
- 有界限 buffer 過載的結果；
- reverse-proxy 的 SSE header 與 idle 政策；
- 指名具權威性重新取得 method 的重新連線計畫；
- 當 capstone 使用 Tasks 時，持久化 Task 的取消 trace。

本地 process 中一個綠燈的請求，只證明了順利的路徑。當遺失的 response、遲到的取消、慢速消費者與重新連線的驚群效應都有確定性的結果時，capstone 才算準備好上 production。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Request cancellation | 放棄一個進行中的 MCP 請求 |
| Cancellation race | 終結的完成事件與取消事件之間的競爭 |
| Idle timeout | 自上次有用的請求活動起算的限制 |
| Maximum timeout | 從請求開始起算的絕對限制，不受 progress 影響 |
| Idempotency key | 對一個商業意圖去重的應用程式識別碼 |
| Atomic ledger | 把 key claim、效果紀錄與結果作為一個單位 commit 的持久邊界 |
| Backpressure | 當生產者快過消費者時所套用的控制 |
| Progress coalescing | 以較新的具權威性值取代較舊的 progress |
| Refetch | 在串流出現缺口之後，重新讀取目前的狀態 |
| Jitter | 刻意的變化，讓重試分散在不同時間 |

## 延伸閱讀 (Further Reading)

- [MCP Cancellation](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/cancellation)
- [MCP Progress](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/progress)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Tasks Extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
