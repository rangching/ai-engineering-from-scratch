# MCP Tasks Extension：建立在無狀態核心上的持久化工作 (MCP Tasks Extension: Durable Work on a Stateless Core)

> 無狀態的 MCP 並不表示每個操作都必須在一個請求內完成。官方的 Tasks extension 為長時間執行的工作提供明確的持久化 handle。Server 可以從 `tools/call` 回傳該 handle，任何實例都能回應 `tasks/get`，而 client 的輸入透過 `tasks/update` 送達，不需要重新啟用 protocol session。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 11 (stateless MRTR), Phase 13 · 12 (elicitation)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 區分無狀態的 protocol transport 與持久化的應用程式 task 狀態。
- 在每請求的 capabilities 與 `server/discover` 中協商 `io.modelcontextprotocol/tasks` extension。
- 只在持久化建立完成之後，才回傳由 server 主導、帶有 `resultType: "task"` 的 `CreateTaskResult`。
- 用 `tasks/get` 輪詢、用 `tasks/update` 滿足 task 的輸入，並用 `tasks/cancel` 請求協作式取消。
- 移除較舊的 `tasks/status`、`tasks/result` 與 `tasks/list` 假設。
- 透過 POST response SSE 串流上的 `subscriptions/listen`，訂閱選擇性的 task notification。
- 正確地建模 task 的過期、重啟復原、input-key 去重，以及執行錯誤。

## 為什麼 Task 是 Extension (Why Tasks Are an Extension)

Task 最早在 2025-11-25 以實驗性核心功能的形式出現。2026 年 7 月的重新設計把它們移進官方的 `io.modelcontextprotocol/tasks` extension，讓 client 與 server 可以選擇加入額外的生命週期，而不必為所有人擴充核心 protocol。

雖然 extension 規格是目前 Task 在官方的歸屬，它仍然是 draft 的介面。請固定你的 SDK 所支援的 extension 版本、執行 conformance 情境，並將 wire adapter 與你的 worker 和儲存領域隔離。

當操作具備以下一項或多項特性時，使用 task：

- 它可能比一般請求的 timeout 活得更久。
- Worker queue 或外部的 job 系統已經擁有執行權。
- Client 需要在自己重啟後能夠復原。
- 操作在執行期間會暫停，等待使用者或模型的輸入。
- 取消與持久化的結果取回，是產品需求。

不要為便宜、確定性的查詢建立 task。Handle、持久化、輪詢、過期與取消，都是真正的複雜度。

## 無狀態核心，有狀態的應用程式 (Stateless Core, Stateful Application)

MCP 2026-07-28 移除了 `initialize`、`notifications/initialized`、protocol session 與 `Mcp-Session-Id`。這並不禁止有狀態的產品。

Task id 是明確的應用程式狀態：

- Server 在回傳之前先持久化它。
- Client 可以儲存它，並在重啟後再次輪詢。
- 該 id 可以路由到由同一個持久化 store 支援的任何 replica。
- 每個 task method 都會檢查授權。
- 過期與刪除由 task 欄位定義，而不是 transport 的生命週期。

這在運作上，和附加在連線上的隱藏狀態截然不同。

請把四種生命週期分開：

| 狀態 | 生命週期 | 歸屬位置 |
|---|---|---|
| Protocol metadata | 一個請求 | `params._meta`，每次 call 都重新驗證 |
| Transport 工作 | 一個 stdio 請求或 HTTP response | 有界限期限的 in-flight coordinator |
| MRTR continuation | 一次重試序列 | 有完整性保護的 `requestState`，必要時加上 replay 控制 |
| 持久化 task | 跨請求、replica、重啟與重新連線 | 以已授權的 `taskId` 為鍵的共用應用程式 store |

把 task 紀錄移進 process 記憶體，並不會讓 MCP 變成有狀態。它只會讓應用程式變得不可靠。Protocol 仍是無狀態的，但之後被路由到另一個 replica 的 `tasks/get`，無法復原該紀錄。請在回傳 handle 之前先持久化，並讓每個 task method 在 tenant 與 principal 檢查之下，解析到同一筆共用紀錄。

## Capability 協商 (Capability Negotiation)

Client 在每個符合資格的請求上公告支援：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "extensions": {
        "io.modelcontextprotocol/tasks": {}
      }
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "lesson-client",
      "version": "1.0.0"
    }
  }
}
```

Server 從 `server/discover` 回傳精確的 `supportedVersions`、capabilities、`ttlMs` 與 `cacheScope`，並在 capabilities 底下列出同一個 extension。因為它公告了 tools，所以也實作強制要求的 `tools/list`。該結果回傳確定性的 `generate_report` descriptor、有效的物件 `inputSchema`、`resultType: "complete"`、server identity metadata，以及 public 的 cache 提示。

沒有宣告 extension 的 client 所送出的 task method，會回傳 `-32021`（Missing Required Client Capability），且 `data.requiredCapabilities` 設為 `{"extensions":{"io.modelcontextprotocol/tasks":{}}}`。不支援的 protocol 字串會回傳 `-32022`，並附上精確的 `supported` 與 `requested` 資料；缺少或不是字串的 version 則回傳 `-32602`。

沒有 JSON-RPC `id` 的 envelope 是 notification。接收者可以處理它，但不發出 JSON-RPC 的 result 或 error。對於被接受的 notification，Streamable HTTP adapter 會回傳沒有 body 的 `202 Accepted`。

目前只有 `tools/call` 支援以 task 擴充的執行。請設計你內部的抽象，讓未來的請求類型不需要重寫儲存層。

## Server 主導的 Task 建立 (Server-Directed Task Creation)

舊的 client 旗標 `params._meta.task.required` 已經沒有了。Client 宣告 extension 支援，再由 server 決定特定的 `tools/call` 是否變成 task。

Request：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "generate_report",
    "arguments": {"size": "large"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

Response：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "task",
    "taskId": "tsk_786512e29e0d",
    "status": "working",
    "statusMessage": "Preparing report outline.",
    "createdAt": "2026-08-21T10:30:00Z",
    "lastUpdatedAt": "2026-08-21T10:30:00Z",
    "ttlMs": 900000,
    "pollIntervalMs": 1000
  }
}
```

在 `tasks/get` 能夠解析該 id 之前，server 不得回傳這個 handle。在最終一致性（eventually consistent）的 store 中，要等到讀取可見之後再回應。否則 client 可能收到看起來有效的 id，卻立刻得到「not found」。

就 client 並未請求 task 模式這點而言，task response 是主動送出的。但它不是未經協商的：當前的請求仍必須公告該 extension。

## Task 的形狀 (The Task Shape)

每個 task 都帶有：

- `taskId`：穩定、由 server 產生的識別碼；
- `status`：`working`、`input_required`、`completed`、`cancelled` 或 `failed`；
- `createdAt` 與 `lastUpdatedAt`：ISO 8601 時間戳記；
- `ttlMs`：從建立時間起算的過期期間，或 `null` 表示沒有公告的上限；
- 選擇性的 `pollIntervalMs`：server 目前建議的最小輪詢節奏；
- 選擇性的 `statusMessage`：給使用者或模型看的 context。

特定狀態的欄位只在相關時才出現：

- `input_required` 包含 `inputRequests`。
- `completed` 包含原始請求的 `result` 形狀。
- `failed` 包含 JSON-RPC 的 `error` 物件。

Client 應該遵守 `pollIntervalMs`。Server 可以對更積極的輪詢做 rate limit，也可以在 task 的生命週期中改變這個間隔。

## 用 `tasks/get` 輪詢 (Poll with `tasks/get`)

Client 要求目前的快照：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/get
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tasks/get",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

`tasks/get` 本身已經完成，所以它的結果永遠是 `resultType: "complete"`。內嵌的 task 仍然可以是 `status: "working"` 或 `status: "input_required"`。

這個區分可以避免常見的 parser bug：

```text
result.resultType = complete    means the tasks/get RPC finished
result.status = working        means the represented job is still running
```

沒有 `tasks/result` 這個 call。Task 完成時，下一個 `tasks/get` response 會把原始的 `CallToolResult` 內嵌在 `result` 之下：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "completed",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:34:12Z",
  "ttlMs": 900000,
  "result": {
    "resultType": "complete",
    "content": [
      {"type": "text", "text": "Generated large report with approved outline."}
    ],
    "structuredContent": {"size": "large", "approved": true},
    "isError": false,
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "tasks-demo",
        "version": "1.0.0"
      }
    }
  },
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "tasks-demo",
      "version": "1.0.0"
    }
  }
}
```

外層的 `resultType` 表示 `tasks/get` RPC 已完成。內嵌的 `result.resultType` 表示原本的 tool call 已完成。那個內嵌的 discriminator 是必要的。內嵌的 `CallToolResult` 也應該帶有自己的 `io.modelcontextprotocol/serverInfo`；這堂課包含了它，而不是儲存沒有型別的 payload。

也沒有 `tasks/list`。無 session 的 server 無法安全地推斷哪些 task 屬於某個以連線為範圍的清單。需要歷史紀錄的應用程式，應該公開帶有明確 filter 與擁有權規則、經過授權的領域 tool。

## Task 執行期間的輸入 (Input During Task Execution)

Task 輸入與核心的 MRTR 看起來類似，但使用不同的 continuation。

### 在 task 建立之前需要輸入

從原本的 `tools/call` 回傳核心的 `resultType: "input_required"`。Client 滿足它，並重試原本的 call。只有在那些同步的 MRTR 回合結束之後，才建立 task。

### 在 task 建立之後需要輸入

把 task 設為 `input_required`。`tasks/get` 會公開尚未處理的 `inputRequests`，client 透過 `tasks/update` 送出 response。Client 不會重試原本的 `tools/call`。

快照：

```json
{
  "resultType": "complete",
  "taskId": "tsk_786512e29e0d",
  "status": "input_required",
  "createdAt": "2026-08-21T10:30:00Z",
  "lastUpdatedAt": "2026-08-21T10:31:00Z",
  "ttlMs": 900000,
  "inputRequests": {
    "approve_outline": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Approve the generated report outline?",
        "requestedSchema": {
          "type": "object",
          "properties": {"approved": {"type": "boolean"}},
          "required": ["approved"]
        }
      }
    }
  }
}
```

更新：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/update
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 4,
  "method": "tasks/update",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "inputResponses": {
      "approve_outline": {
        "action": "accept",
        "content": {"approved": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

成功的 response 是空的確認，加上 `resultType: "complete"`。狀態變更可能是最終一致的，所以 client 要繼續輪詢或 listen。

每個 `inputRequests` 的 key，在整個 task 生命週期中都必須唯一。重複的 `tasks/get` 快照可能顯示同一個尚未處理的 key；client 會把 UI 去重，server 則忽略針對未知、已被取代或已被滿足的 key 的 response。部分更新可以讓 task 維持在 `input_required`，直到所有必要的 key 都被回答。

## 取消是協作式的 (Cancellation Is Cooperative)

`tasks/cancel` 表達意圖，並回傳空的 complete 確認。該確認不保證 worker 已經停止。工作可能先完成、忽略取消，或之後才轉換狀態。

```http
POST /mcp HTTP/1.1
Content-Type: application/json
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tasks/cancel
Mcp-Name: tsk_786512e29e0d
```

```json
{
  "jsonrpc": "2.0",
  "id": 5,
  "method": "tasks/cancel",
  "params": {
    "taskId": "tsk_786512e29e0d",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/tasks": {}
        }
      }
    }
  }
}
```

對於這三個 task method，`Mcp-Name` 鏡射的是 `params.taskId`。它不會重複 JSON-RPC 的 method 名稱。`code/main.py` 把這條規則集中在 `make_http_request`。

這堂課的 worker 會立即遵守取消，讓重複的 call 具有冪等性。Production 的 client 仍然必須把取消視為協作式的，而不是從確認訊息推斷最終的 task 狀態。

不要使用 `notifications/cancelled` 來取消 task。那個 notification 屬於請求取消，不屬於持久化的 Task。

這個區分在路由邊界上很重要。請求取消針對的是一個進行中的 JSON-RPC 操作，或它在請求範圍內的 HTTP response。如果 `tools/call` 已經回傳了 `resultType: "task"`，該請求就已完成，關閉它的 transport 無法指名或停止那個持久化的 job。`tasks/cancel` 是一個新的、經過授權的 RPC。它帶有 `params.taskId`、在 `Mcp-Name` 中鏡射該 id、解析 task 所屬的 backend、記錄協作式取消的意圖，並回傳確認，而不宣稱 worker 已經停止。

因此，gateway 必須把 request coordinator 與 task route 放在不同的表裡。Request 表可以在 response 結束時消失。Task route 則必須存活到終結狀態與保留期限過期為止。[Lesson 29: MCP Reliability, Cancellation, and Flow Control](../../29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 為這兩條路徑建構 race、timeout、冪等性、backpressure 與重試的規則。

## 選擇性的 Notification (Optional Notifications)

輪詢是基準做法。想要推送更新的 client，會送出帶有 task id 的 `subscriptions/listen`。在 Streamable HTTP 上，這是一個 POST，其 response 是請求範圍的 SSE 串流。沒有獨立的 GET 事件串流，也沒有需要維持的 protocol session。

Server 用 `notifications/subscriptions/acknowledged` 確認被接受的 id，之後可以透過 `notifications/tasks` 傳送完整的快照。確認訊息和每個 task notification，都在 `_meta` 中帶有 `io.modelcontextprotocol/subscriptionId`，其值等於 `subscriptions/listen` 的請求 id。每個 task notification 在其他方面，都等同於 `tasks/get` 在當下會回傳的內容。

Client 仍然必須宣告 Tasks extension。它們應該重新連線，並從持久化的 task id 繼續，而不是依賴事件 replay 或 `Last-Event-ID`。

## 失敗語意 (Failure Semantics)

正確使用這兩層錯誤。

### Protocol 錯誤

無效的 method 參數或未知的 task id，會回傳 JSON-RPC 錯誤，通常是 `-32602`。缺少 extension 支援則回傳 `-32021`，並附上必要的 capability 物件。

### Task 的執行結果

- 帶有 `isError: true` 的一般 tool 結果，仍然是 `completed` 的 task，因為 tool call 產生了它定義的結果。
- 延後執行期間的 JSON-RPC 錯誤，會讓 task 變成 `failed`，並把該 JSON-RPC 錯誤存在 `error` 之下。
- 使用者拒絕可以產生 `cancelled`、已完成的拒絕結果，或其他特定領域的安全結果。請記錄你的選擇。

## 持久性、過期與擁有權 (Durability, Expiry, and Ownership)

至少要持久化 task id、狀態、時間戳記、ttl、輪詢間隔、原始操作的擁有權、結果或錯誤、尚未處理的 input request，以及所有已發出的 input key。

儲存的 key 必須包含或解析出具權威性的 tenant 與 principal。知道 task id 不得授予存取權。在每個 `tasks/get`、`tasks/update`、`tasks/cancel` 和訂閱上檢查擁有權。

`ttlMs` 從建立時間起算，而且可能改變。當 task 不再產生可觀察的更新時，client 可以把它當作保底。Server 可以讓過期的 task 失敗，之後再刪除。不要把它描述成承諾在完成之後保留已完成結果那麼多毫秒。

使用 atomic 寫入或 transaction。這堂課寫入暫存檔，再以 atomic 的方式 rename。多 replica 的服務應該使用共用的持久化 store，加上 worker lease 或等效的並行控制。

```figure
tp-task-lifecycle
```

## 動手做 (Build It)

`code/main.py` 實作一個確定性的 task 服務：

- `server/discover` 回傳 `supportedVersions`、cache 提示以及 Tasks extension。
- `tools/list` 回傳確定性、可快取、帶有有效 input schema 的 `generate_report` descriptor。
- `tools/call` 在回傳 `resultType: "task"` 之前，先建立並持久化 task。
- 新的服務實例會重新載入同一個 task，示範重啟復原。
- `tasks/get` 回傳完整的 task 快照。
- Worker 從 `working` 轉移到 `input_required`。
- `tasks/update` 接受 form response，並回傳空的 complete 確認。
- Worker 儲存內嵌的 `CallToolResult`，帶有自己的 `resultType` 與 server identity，然後轉移到 `completed`。
- 在這個實作中，`tasks/cancel` 具有冪等性。
- HTTP builder 對 `tasks/get`、`tasks/update` 和 `tasks/cancel`，把 `Mcp-Name` 設為 `params.taskId`。
- Notification helper 使用 `notifications/subscriptions/acknowledged` 與 `notifications/tasks`，兩者都標記了 listen 的請求 id。
- 沒有 id 的 notification 不會產生 JSON-RPC response。

Worker 是明確地推進，而不是在背景 thread 中 sleep。這讓每個狀態轉換都具有確定性，並讓 protocol 範例與 queue 機制分開。

## 實際使用 (Use It)

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/13-mcp-async-tasks/code
python3 main.py
python3 -m unittest discover tests -v
```

預期的結果序列：

```text
id=0 resultType=complete status=ack
id=1 resultType=task status=working
id=2 resultType=complete status=working
id=3 resultType=complete status=input_required
id=4 resultType=complete status=ack
id=5 resultType=complete status=completed
```

也請驗證，在現代服務中，`tasks/status`、`tasks/result` 與 `tasks/list` 會回傳 method-not-found。
並驗證 `tools/list` 具有確定性，而且目前每個 HTTP task method，都透過 `Mcp-Name` 鏡射它的 task id。

## 交付 (Ship It)

`outputs/skill-task-store-designer.md` 現在會產出能感知 extension 的設計：capability 協商、先持久化再回傳的建立方式、目前的 method、輸入更新流程、擁有權、過期、取消、訂閱，以及從已移除的實驗性 method 遷移。

## 練習 (Exercises)

1. 新增第二個尚未處理的 input key。送出部分的 `tasks/update`，並證明 task 會維持在 `input_required`，直到兩個 key 都被回答。
2. 在 store 中加入 tenant 擁有權，並拒絕由錯誤的已驗證 principal 所出示的有效 task id。
3. 新增帶有過期時間的 worker lease。示範兩個服務實例不能同時完成同一個 task。
4. 為 `subscriptions/listen` 實作 POST-response 的 SSE adapter。不要加入 GET、`Last-Event-ID` 或 session header。
5. 新增過期清理。區分已過期的 task 與格式錯誤的 task id，同時不洩漏跨 tenant 的存在性。

## 關鍵術語 (Key Terms)

| 術語 | 在目前 extension 中的意義 |
|------|----------------------------------|
| Tasks extension | 用於持久化非同步工作、選擇性的 `io.modelcontextprotocol/tasks` capability |
| `CreateTaskResult` | 對符合資格的請求，由 server 主導、`resultType: "task"` 的 response |
| `tasks/get` | 輪詢完整的目前 task 快照，包含終結結果或待處理的輸入 |
| `tasks/update` | 對 task 尚未處理的 `inputRequests` 提交 response |
| `tasks/cancel` | 確認協作式取消的意圖 |
| `input_required` | 表示仍在等待 client 輸入的 task 狀態 |
| `pollIntervalMs` | Server 建議的、下次輪詢前的最小延遲 |
| `ttlMs` | 從 task 建立時間起算的過期期間 |
| Durable-before-return | task id 必須在送出 handle 之前就能解析的規則 |
| `notifications/tasks` | 在已訂閱的 SSE response 上傳送、選擇性的完整 task 快照 |

## Legacy 相容性 (Legacy Compatibility)

2025-11-25 的實驗性介面使用 client 請求的 task 擴充、`tasks/status`、`tasks/result` 與選擇性的 `tasks/list`。請只把這些名稱留在固定的 legacy adapter 裡。目前的 client 使用 extension capability、接受由 server 主導的 handle、輪詢 `tasks/get`、用 `tasks/update` 提供輸入，並從 task 快照讀取最終結果。

## 延伸閱讀 (Further Reading)

- [Official MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
