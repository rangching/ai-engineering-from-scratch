# 無狀態 MCP Gateway 與 Registry 接納 (Stateless MCP Gateways and Registry Admission)

> Gateway 應該讓每一條路由都明確。2026-07-28 的 protocol 在沒有 transport session 的情況下，提供了 method、name、version、capability、identity、cache 與 trace 的邊界。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 15 (security), Phase 13 · 16 (authorization)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 在沒有 session affinity 的情況下，把多個 MCP server 聚合在單一 2026-07-28 endpoint 之後。
- 在套用政策或轉發之前，先驗證每請求的 metadata 與路由 header。
- 以穩定的 namespace、確定性的順序、descriptor pin、RBAC 與 private 快取來合併 tool。
- 把 registry 紀錄視為仍需要接納（admission）政策的 discovery 證據。
- 正確路由請求範圍的 SSE、`subscriptions/listen`、MRTR 重試與 Tasks extension 的 call。
- 把 legacy 的 handshake 與 session 支援，與現代路徑隔離。

## 問題 (The Problem)

讓一個 client 直接連到一個 server 很簡單。較大的部署需要對更難的問題給出一致的答案：

- 哪些 server 被允許？
- 哪個 principal 可以看到並呼叫哪個 tool？
- 兩個 backend 公開相同名稱時會怎樣？
- Descriptor 的變更如何審查？
- Rate limit 與稽核事件套用在哪裡？
- 任何實例都能處理下一個請求嗎？

Gateway 位於 client 與 backend MCP server 之間。它公開一個 MCP endpoint、套用橫切關注點的政策，並轉發經過核准的請求。

較舊的 gateway 設計經常把一個 client session 多工成數個 backend session，並改寫 `Mcp-Session-Id`。那是 legacy 的相容性設計。2026-07-28 的核心沒有 protocol session。

## 核心概念 (The Concept)

### 現代的 gateway 路徑

對每個請求：

1. 從 transport 授權驗證 principal 的身分。
2. 驗證 `MCP-Protocol-Version`、`Mcp-Method`、`Mcp-Name` 與 `params._meta`。
3. 對 principal、resource、method、tool 與參數做授權。
4. 套用 descriptor、registry、rate 與資料政策。
5. 為選定的 backend 建立全新、自成一體的請求。
6. 驗證 backend 的結果，並回傳 gateway 的結果。
7. 記錄稽核事件，但不記錄機密。

沒有任何步驟需要隱藏的 protocol session。應用程式狀態仍可存在資料庫、明確的 handle、Tasks，或有完整性保護的 MRTR 狀態中。

### 執行期政策是 gateway 的首要決定 (Runtime policy is the primary gateway decision)

接納（admission）決定哪個 backend 版本可以進入 gateway。它不授權即時的 call。對每個請求，gateway 都會根據已驗證的 principal、issuer 與 resource、tenant、比對到的 method 與 name、正規化的參數、已接納的 descriptor pin、目前 backend 的健康狀態、capability 的交集、資料分類、rate 狀態，以及任何綁定動作的核准，重新計算政策。

這個順序很重要。Registry 紀錄可以保持 active，而使用者的角色卻已被撤銷。Descriptor 可以保持被 pin 住，而某個目的地參數卻跨越了 tenant 邊界。Backend 可以保持已核准，而事件政策卻隔離了會改變狀態的 call。因此，執行期政策是首要的允許或拒絕決定，Registry 與 descriptor 的證據只是輸入。

不要把允許的決定，快取在連線或已被移除的 session 識別碼底下。如果政策不可用，請依操作類別遵循宣告好的失敗政策。安全的預設是對會改變狀態的操作與敏感讀取 fail closed，而明確核准的公開讀取路徑，只有在其風險模型允許時，才可以使用短暫的最後已知政策。記錄是哪個政策版本與失敗路徑做出了決定，然後在回傳之前驗證 backend 的結果。

### 一個 POST endpoint

現代的 Streamable HTTP 透過 POST 傳送每則 JSON-RPC 訊息：

```text
POST /mcp
Authorization: Bearer <gateway-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.search
Accept: application/json, text/event-stream
```

Gateway 可以對該 POST 回傳 JSON 或請求範圍的 SSE。對現代請求，GET 與 DELETE 回傳 405。`Mcp-Session-Id` 與 `Last-Event-ID` 不會建立權限、affinity 或 replay 行為。

Header 與 body 的值必須一致。在查找 backend 之前，就用 `-32020` 拒絕不一致。這讓 load balancer、gateway 與 rate limiter 能在不解析完整 body 的情況下路由，同時保有端對端的完整性。

以一個確切的順序驗證：JSON-RPC 與 metadata 的型別、header 與 body 相等、然後才是對一致 version 的支援。不一致時，回傳 HTTP 400 與 `-32020`。如果 header 與 body 對一個不支援的 version 看法一致，回傳 HTTP 400 與 `-32022`，且 `data` 精確為 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知的 method 回傳 HTTP 404 與 `-32601`。

`ProtocolError` 帶有選擇性的 `data`，gateway 會把它序列化進 JSON-RPC 的 error 物件。Notification 沒有 `id`，所以永遠不會收到 JSON-RPC 的成功或錯誤。被接受的 HTTP notification 回傳 202 與空的 body。

### 在每一層實作 discovery

Gateway 為 client 實作 `server/discover`。它也探索每個 backend，以便知道 protocol version、capabilities 與 extension。

Gateway 結果範例：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": true}
  },
  "ttlMs": 30000,
  "cacheScope": "private",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "enterprise-gateway",
      "version": "2.0.0"
    }
  }
}
```

只公告 gateway 能端對端兌現的 capability 交集。Backend 的功能並不會自動就能安全地公開。沒有 backend 路徑的 gateway 功能，公告出去也沒有用。

`serverInfo` 是自行申報的顯示與診斷資料。不要把它當作 registry 或發布者的證明。

### 每請求的 client capabilities

每個轉發的請求都需要當前的 `_meta` envelope：

```json
{
  "io.modelcontextprotocol/protocolVersion": "2026-07-28",
  "io.modelcontextprotocol/clientCapabilities": {},
  "io.modelcontextprotocol/clientInfo": {
    "name": "enterprise-gateway",
    "version": "1.0.0"
  }
}
```

不要盲目地把外層 client 的 capabilities 複製給 backend。Gateway 才是 backend 的 client。只公告 gateway 能正確居中處理的功能。

### 確定性的 namespace

把 backend 的 tool 合併到穩定的公開名稱底下：

```text
notes.search
notes.create
issues.list
issues.open
```

保留從公開名稱到 backend 與原始 tool 名稱的對應。絕不要選擇第一個或最後一個衝突者。公開名稱是核准與稽核合約的一部分，所以更改它就是一次遷移。

`tools/list` 必須是確定性的。當可見性隨 principal 而異時，回傳 `cacheScope: private`。有界限的 `ttlMs` 可以減少 backend discovery 的負載，又不會讓使用者專屬的清單洩漏到其他授權 context。

每個公開的 tool descriptor 都包含穩定的 name、description 與物件根的 `inputSchema`。Namespace 不能移除必要的 descriptor 欄位。完整的 list 結果也包含 `resultType`、server identity metadata 與 cache 提示。

### Pin 住已核准的 descriptor

在接納時，把完整的 descriptor 正規化，並把 digest 存在限定的公開名稱底下。在 list 與 call 時，把即時的 descriptor 與已核准的 digest 比對。

如果它改變了：

- 從 `tools/list` 移除它。
- 拒絕直接的 call。
- 發出稽核事件。
- 在更新 pin 之前，要求政策或人工重新核准。

Gateway 是有用的集中式強制執行點，但它不會把初次看到的 descriptor 變成安全的。初次審查仍然必要。

### Registry 幫助探索，不做決定

Registry 的 `server.json` 提供發布 metadata。以套件為基礎的紀錄可能長這樣：

```json
{
  "$schema": "https://static.modelcontextprotocol.io/schemas/2025-12-11/server.schema.json",
  "name": "com.example/notes",
  "description": "Example notes MCP server.",
  "version": "1.0.0",
  "packages": [
    {
      "registryType": "npm",
      "identifier": "@example/notes-mcp",
      "version": "1.0.0",
      "transport": {"type": "stdio"}
    }
  ]
}
```

發布 metadata 不攜帶 gateway 的安全決定。請把已驗證的發布者與出處證據，放在獨立的接納狀態中：

```json
{
  "registryName": "com.example/notes",
  "registryVersion": "1.0.0",
  "publisher": {"namespace": "com.example", "status": "verified"},
  "provenance": {
    "source": "registry.modelcontextprotocol.io",
    "recordId": "com.example/notes@1.0.0"
  },
  "admission": {"status": "approved", "reviewedBy": "gateway-policy"}
}
```

Gateway 檢查 `server.json` 的形狀，並將它與外部狀態結合。Gateway 仍然需要接納政策。

對每個已接納的 backend，記錄：

- 確切的 registry 與紀錄識別碼。
- 已驗證的發布者 namespace 或 domain 證據。
- 允許的 transport 與 endpoint。
- 固定的 version 或核准的升級政策。
- Artifact 或 descriptor 的 digest。
- 授權 issuer 與 resource。
- 審查者、核准時間與到期日。

不要因為顯示名稱類似熟悉的產品，就接受某個 server。不要把出現在 registry 當成營運面的安全審查。私有 server 即使從未出現在公開 registry，也可以透過同一套證據 schema 被接納。

這堂課實作 gateway 的接縫：在 backend 變成可路由之前，把發布證據與本地的接納結合。[Lesson 30: MCP Registry Supply Chain, Admission, Drift, and Rollback](../../30-mcp-registry-supply-chain-and-drift/docs/en.md) 建構完整的控制平面，涵蓋確切的 namespace 證明、artifact 出處、不可變的 pin、即時 descriptor 漂移、Registry 狀態對帳、防竄改的接納帳本，以及有證據支持的 rollback。請把供應鏈狀態，與上述每請求的執行期決定分開。

### 憑證居中處理 (Credential mediation)

Gateway 驗證呼叫者的身分，並另外向 backend 驗證自己的身分。Backend 的憑證絕不會交給 client。

保持這些綁定明確：

```text
outer principal -> gateway role and policy
backend issuer + resource -> backend registration and token
```

絕不要把外層 gateway 的 token 傳給 backend。絕不要在不同的 issuer 或 resource 重用 backend 的 token。如果 tool 代表終端使用者行動，請用設計過的 exchange 或 claims 模型保留該委派，而不是用共用的服務憑證冒充使用者。

### 沒有 session 的 rate limit

以已驗證的 principal、issuer、resource、公開 tool、成本類別與時間窗口作為 limit 的 key。沒有 session id，而且即使有，也很容易輪換。

在消耗昂貴的工作之前，先套用便宜的驗證。決定被拒絕的 call 是否計入濫用限制、商業配額，或兩者皆計入。

### 稽核決策鏈

記錄足以重建 call 的資訊：

- 請求與 trace 識別碼。
- 已驗證的 principal 與 issuer。
- 公開的 tool 與 backend 路由。
- Descriptor pin 的版本。
- 政策決定與原因。
- 延遲與結果類別。
- 適用時的 MRTR 回合或 task 識別碼。

遮蔽 bearer token、authorization code、refresh token、原始機密，以及不必要的敏感參數。

### 請求範圍的 SSE (Request-scoped SSE)

當工作在單一請求期間以串流方式進行時，一般的 POST 可以回傳請求範圍的 SSE。關閉 response 串流會取消該進行中的現代 HTTP 請求。

不要建立獨立的 GET 串流，也不要承諾 Last-Event-ID 的 replay。那些是較舊的 transport 假設。

### 長壽命的變更 notification

對於 list 與 resource 變更的 notification，目前的 client 透過 POST 送出 `subscriptions/listen`，並收到 SSE response。Notification 的 filter 使用確切的扁平欄位 `toolsListChanged`、`promptsListChanged`、`resourcesListChanged` 與 `resourceSubscriptions`：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-tools",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

第一個事件確認所支援的子集合。它的 subscription 識別碼就是開啟該串流的請求的 JSON-RPC id：

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": "listen-tools"
    },
    "notifications": {
      "toolsListChanged": true
    }
  }
}
```

接著 gateway 只轉發已確認的變更類型。該串流上的每個 notification，都在 `params._meta` 中帶有相同的 `io.modelcontextprotocol/subscriptionId`。沒有自動 replay，也沒有自動重新 listen。重新連線時，client 重新開啟訂閱，並重新整理它所依賴的清單。Server 發起的正常關閉，會回傳一個標有相同 subscription id 的最終 complete 結果。

現代路徑取代了 `resources/subscribe`、`resources/unsubscribe`，以及主動送出的獨立 GET 串流。這些只保留在受 version 管控的舊路徑中。

### 透過 gateway 的 MRTR

當 backend 回傳 `resultType: input_required` 時，只有在外層 client 支援所需的 input request 時，gateway 才可以轉發該結果。除非 gateway 刻意終止並重新發起這次互動，否則要逐位元組保留 `requestState`。

Client 用新的 JSON-RPC id 與 `inputResponses` 重試原本的公開 tool。Gateway 重新授權該重試、檢查相同的公開路由，然後轉發全新的 backend 請求。它不得假設較早的回合授予了無限的核准。

### Tasks extension 路由

Task 是以 `io.modelcontextprotocol/tasks` 識別的官方 extension。它們不是核心 session 的替代品。

Client 在每請求的 client capabilities 中宣告該 extension，而 gateway 只有在能端對端保留其生命週期時，才在 discovery 中公告它。對於受支援的 `tools/call`，由 backend 單獨決定是回傳一般結果，還是 `resultType: task`。Task 結果直接在結果中帶有 `taskId`、`status`、時間戳記、`ttlMs`，以及選擇性的 `pollIntervalMs`。在送出該結果之前，task 必須已經可以持久地讀取。

Gateway 為不透明的 task 識別碼記錄已驗證的 principal 與 backend 路由。後續的 `tasks/get`、`tasks/update` 與 `tasks/cancel` call，都使用 `params.taskId` 作為 `Mcp-Name`，這給了中介者一個路由 key。`tasks/get` 回傳 `resultType: complete` 與目前的 task 狀態，並在終結狀態內嵌最終結果或 protocol 錯誤。`tasks/update` 為尚未處理的 task 輸入送出以 key 區分的 `inputResponses`，並回傳空的 complete 確認。`tasks/cancel` 是協作式的意圖，附有空的 complete 確認，不保證工作會停止。

不要實作新的 `tasks/list` 或 `tasks/result` method。它們屬於較舊的實驗性模型。需要輸入的 task，會透過 `tasks/get` 公開完整內嵌的請求；client 透過 `tasks/update` 回答它們，而不是重試原本的 tool call。Client 仍以建議的間隔輪詢；task 的建立仍由 server 主導。

持久的 task 路由狀態是以 task handle 為 key 的應用程式資料，不是 protocol session。

### 相容性邊界

如果 gateway 必須服務較舊的 client 或 backend：

- 明確地偵測世代。
- 把初始化、transport session、GET 串流、resource 訂閱與舊的 task 詞彙，保留在 legacy adapter 內。
- 絕不要讓 legacy 的 session id 漏進現代的路由或授權。
- 優先使用有界限的 discovery 探測與明確的 fallback 政策，而不是默默降級。

```figure
t3-gateway-funnel
```

## 動手做 (Build It)

`code/main.py` 實作一個 in-process 的 protocol gateway 和兩個 backend server。每個 backend 都收到全新的、符合目前 protocol 的請求。Gateway 提供 discovery、依使用者過濾且確定性的 `tools/list`、帶 namespace 的路由、Registry `server.json` 加上外部接納狀態、descriptor pin、RBAC、以 principal 為 key 的 rate limit、稽核決定，以及模擬的 `subscriptions/listen` SSE 確認。

這個模型接收已解析的請求 body、路由 header，以及已驗證的 bearer identity。它不是完整的 HTTP adapter，也不解析 `Content-Type` 或完整的 `Accept` 合約。請把它接到課程 09 的 Streamable HTTP adapter，它要求 `Content-Type: application/json`，以及同時包含 `application/json` 和 `text/event-stream` 的 `Accept` 值。

執行它：

```bash
cd phases/13-tools-and-protocols/17-mcp-gateways-and-registries
python3 code/main.py
python3 -m unittest discover code/tests -v
```

Demo 會印出外層的 request id 與全新的 backend request id，讓無狀態的轉送過程看得見。

## 實際使用 (Use It)

把 in-process 的 backend 物件換成真正符合目前 protocol 的 client。保持相同的接縫：

- 連線之前先有接納紀錄。
- 公開 capability 之前先做 backend discovery。
- 授權之前先有限定的公開名稱。
- list 或 call 之前先做 descriptor pin。
- 轉發之前先有全新的每請求 metadata。
- 回傳之前先做結果驗證。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-gateway-bootstrap.md`。它會產出現代的 gateway 設計，涵蓋 ingress、discovery、接納、namespace、授權、快取、串流、訂閱、MRTR、Tasks、observability，以及 legacy 隔離。

## 練習 (Exercises)

1. 在外層與轉發請求的 metadata 中加入 trace context，並在稽核事件中記錄其關聯。
2. 新增一個支援 Tasks 的 backend，並依 `Mcp-Name` 中的 task id 路由 `tasks/get`。
3. 修改其中一個 backend 的 descriptor，並證明 discovery 與直接 call 都被阻擋。
4. 新增一個依 principal 而異的 server capability，並說明為什麼 discovery 必須維持 private 快取。
5. 寫一個 legacy adapter 介面，且不在現代的 `Gateway` class 中加入任何 legacy 狀態。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| MCP gateway | 位於 client 與 backend MCP server 之間的政策與路由 server |
| Admission record | 允許一個 backend 進入 gateway 的證據與政策決定 |
| Qualified tool name | 穩定的公開路由，例如 `notes.search` |
| Descriptor pin | 在 discovery 與 dispatch 期間檢查的已核准 digest |
| Private cache scope | 限制於單一授權 context 的快取結果 |
| Request-scoped SSE | 附屬於單一 POST 請求的串流 response |
| `subscriptions/listen` | 為選定的長壽命變更 notification，由 client 開啟的 SSE 串流 |
| Task route | 從不透明的 task id 到其 backend 的應用程式對應 |
| Legacy adapter | 為舊的 handshake 與 session 行為設立、受 version 管控的明確邊界 |

## 延伸閱讀 (Further Reading)

- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [MCP Tasks extension](https://tasks.extensions.modelcontextprotocol.io/specification/draft/tasks)
