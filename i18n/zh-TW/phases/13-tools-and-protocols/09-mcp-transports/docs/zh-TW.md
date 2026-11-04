# MCP Transports：stdio 與無狀態的 Streamable HTTP (MCP Transports: stdio and Stateless Streamable HTTP)

> Transport 負責傳送 MCP 訊息。它不會補上缺少的 protocol 狀態。在 `2026-07-28` 中，本地的 stdio 與遠端的 Streamable HTTP 都傳送自我描述的請求。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07 and 08
**Time:** ~65 分鐘

## 學習目標 (Learning Objectives)

- 本地子 process 選擇 stdio，網路服務選擇 Streamable HTTP。
- 實作現代的單一 endpoint、僅限 POST 的 Streamable HTTP 合約。
- 對照 JSON-RPC body，鏡射並驗證 MCP 的 version、method 與 name header。
- 正確傳送請求範圍的 SSE 與長壽命的 `subscriptions/listen` 串流。
- 遷移以 session 為基礎的部署與 legacy HTTP+SSE 部署，且不把 legacy 行為當成現代行為呈現。

## 問題 (The Problem)

較早的 Streamable HTTP revision 把 protocol 協商與連線、session 行為混在一起。Server 可以產生 `Mcp-Session-Id`、公開獨立的 GET 串流、接受 DELETE 來終止 session，並用 `Last-Event-ID` 恢復 SSE。

MCP `2026-07-28` 把這些機制從現代的 wire 上移除。每個請求都可以落在任何健康的 worker 上，因為它的 protocol version 與 client capabilities 就在請求 body 裡。HTTP header 會鏡射部分欄位，用於路由與政策，但 server 在執行前，會先對照 body 驗證這些 header。

結果是更容易擴展，也更容易推理。這也代表，如果 server 把 2025 年的 transport 當作現行版本來教，教的就是錯誤的失敗模型與安全模型。

## 核心概念 (The Concept)

### stdio

stdio binding 適用於由 client 啟動的子 process：

- Client 把每行一則的 UTF-8 JSON-RPC 訊息寫到 stdin。
- Server 把每行一則的 UTF-8 JSON-RPC 訊息寫到 stdout。
- Server 把診斷訊息寫到 stderr。
- Server 在 stdin EOF 時迅速結束。
- 每個現代請求都在 `params._meta` 中帶有 version 與 client capabilities。

Process 可能存活很多次 call，但它不是現代的 protocol session。如果它意外結束，進行中的請求就會遺失。請重啟 process、重新 discovery、重新 list、重新開啟訂閱，並以新的 request id 重試安全的操作。

### 2026-07-28 的 Streamable HTTP

現代 server 公開單一 MCP endpoint（例如 `/mcp`），接受 POST。

每個 JSON-RPC 請求或 notification 都是新的 HTTP POST。Body 包含一則 JSON-RPC 訊息。Client 不會把 JSON-RPC response 送給 server。

對於請求，server 會回傳以下其中一種：

- `Content-Type: application/json`，內含一個 JSON-RPC response；或
- `Content-Type: text/event-stream`，內含與該請求相關的 notification，最後接著 JSON-RPC response。

對於被接受的 notification，server 回傳沒有 body 的 `202 Accepted`。

Client 會同時公告兩種 response 類型：

```http
Accept: application/json, text/event-stream
```

### 僅限 POST 就是僅限 POST

現代的 Streamable HTTP 沒有獨立的 GET 串流，也沒有 DELETE session endpoint。

- `GET /mcp` 回傳 `405 Method Not Allowed`。
- `DELETE /mcp` 回傳 `405 Method Not Allowed`。
- `Mcp-Session-Id` 會被忽略，絕不產生也不回傳。
- `Last-Event-ID` 會被忽略，因為現代串流不可恢復。

如果請求範圍的串流在最終 response 之前中斷，client 就失去了那個進行中的請求。在可以安全重試時，它可以用新的 JSON-RPC id 發出新請求。它不得嘗試恢復串流。

### Origin 驗證

Server 會驗證進入連線的 `Origin`，以防範 DNS rebinding。如果 header 存在但沒有被明確允許，就回傳 `403 Forbidden`。非瀏覽器的 client 可以省略 `Origin`，官方 transport 規則允許這麼做。

本地 server 應該 bind 到 `127.0.0.1`，而不是所有介面。網路服務仍然需要在每個請求上做驗證身分與授權。Origin 驗證不是身分驗證。

在正規化設定之後，使用精確的 origin 比對。像 `origin.startswith("https://trusted.example")` 這樣的前綴檢查並不安全，因為它可能接受攻擊者控制的後綴。

### 必要的 HTTP metadata header

每個現代的 POST 請求都包含：

```http
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes_search
```

Header 規則：

- `MCP-Protocol-Version` 是必要的，而且必須等於 `params._meta.io.modelcontextprotocol/protocolVersion`。
- `Mcp-Method` 是必要的，而且必須等於 JSON-RPC 的 `method`。
- `tools/call`、`resources/read` 和 `prompts/get` 需要 `Mcp-Name`。
- `Mcp-Name` 等於 `params.name`，`resources/read` 則等於 `params.uri`。
- Header 的值區分大小寫，雖然 header 名稱不區分。

不安全或非 ASCII 的 `Mcp-Name` 值，使用精確的 UTF-8 Base64 sentinel：

```text
=?base64?{Base64EncodedValue}?=
```

Server 會先解碼該值，再與 body 比對。

缺少、格式錯誤，或不一致的鏡射 header，會回傳 HTTP `400` 與 JSON-RPC 代碼 `-32020`。如果 header 與 body 一致，但指定了 server 不支援的 version，就回傳 HTTP `400` 與 `-32022`，並附上精確的錯誤資料，例如 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。

未知的現代 method 會回傳 HTTP `404` 與 JSON-RPC `-32601`。JSON-RPC body 很重要，因為雙世代的 client 會用它來區分現代錯誤與 legacy endpoint 找不到。

### 請求範圍的 SSE (Request-scoped SSE)

Server 可以為一個長時間執行的請求選擇 SSE：

```text
POST tools/call id=41
  <- notifications/progress related to id=41
  <- notifications/progress related to id=41
  <- JSON-RPC response id=41
stream closes
```

Server 不得在這個串流上送出獨立的 JSON-RPC 請求。Sampling、elicitation 和 roots 的互動，使用 Multi Round-Trip Request 的結果。關閉 response 串流就會取消該請求。

不要為了 replay 而加入 SSE event id。`Last-Event-ID` 的恢復機制不屬於現代 revision。

### 長壽命的變更使用 subscriptions/listen

變更通知使用 client 開啟的請求，而不是獨立的 GET：

```json
{
  "jsonrpc": "2.0",
  "id": "listen-1",
  "method": "subscriptions/listen",
  "params": {
    "notifications": {
      "toolsListChanged": true,
      "resourceSubscriptions": ["notes://note-1"]
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    }
  }
}
```

POST response 是一個長壽命的 SSE 串流。它的第一則 protocol 訊息是 `notifications/subscriptions/acknowledged`。確認訊息、每個變更 notification 以及最終結果，都在 `_meta` 中帶有 `io.modelcontextprotocol/subscriptionId`，其值等於 listen 請求的 id。Server 可以發送 SSE 註解作為 keepalive。當串流中斷時，client 會用新的 request id 重新發出 `subscriptions/listen`，並重新取得受影響的資料。

`resources/subscribe` 和 `resources/unsubscribe` 屬於 legacy 世代。不要在現代連線上使用它們。

### 明確的應用程式狀態 (Explicit application state)

移除 protocol session，並不代表禁止有狀態的 workflow。Server 可以產生不透明的狀態 handle，並把它當作一般的 tool 結果回傳。Client 在之後的 call 中，把該 handle 當作明確的參數傳回。

把 handle 綁定到已驗證身分的主體，讓它難以猜測、會過期，並對每次使用都做授權。這讓狀態在應用程式層可見，而不是藏在 transport affinity 裡。

隱藏的 replica 狀態所造成的失敗，是機械式的：

1. 請求 A 到達 replica 1，並在該 process 的記憶體中建立一個 draft。
2. Response 沒有回傳 draft handle，因為實作假設連線就能識別 draft。
3. 請求 B 是全新的 POST，到達 replica 2。
4. Replica 2 有有效的 protocol metadata，卻無法命名或載入該 draft，所以 workflow 失敗，或讀到錯誤的本地物件。
5. Sticky routing 看起來修好了症狀，直到重啟、rollout、重新排程或 failover，把下一個請求移走。

正確的邊界有兩個部分。Protocol context 留在每個請求中。持久的應用程式狀態則存在共用的 store 裡，由 server 產生的 handle 回傳給 client。下一次 call 帶上該 handle，任何 replica 都能載入同一筆紀錄，而授權會把這筆紀錄綁定到已驗證身分的主體和 tenant。Replica 記憶體可以快取紀錄，但不能是正確性所依賴的唯一副本。

依生命週期選擇狀態機制。請求內的區域變數可服務單一 call。短暫的 MRTR continuation 可以使用有完整性保護的 `requestState`。Draft 或持久的 task 則需要明確的 handle，加上共用的持久化、過期、並行控制與冪等性。這些物件都不是 MCP protocol session。

### HTTP 雙世代相容性

同時支援現代與 legacy server 的 client，會先嘗試現代的 POST。如果它收到 HTTP `400`、`404` 或 `405`，就檢查 body：

- 可辨識的現代 JSON-RPC 錯誤，證明 server 是現代的。修正請求，或用公告的 version 重試。不要降級。
- 空的 body 或無法辨識的 response，可能表示是 legacy HTTP+SSE server。只有在這種情況下，才嘗試舊的 GET endpoint，並預期它的 legacy `endpoint` 事件。

Server 可以在遷移期間同時支援兩個世代：把帶有現代 metadata 的請求路由到現代僅限 POST 的實作，並為舊 client 保留獨立的 legacy endpoint。絕不要把 legacy 的 GET、DELETE、session id 或 replay 行為，說成是 `2026-07-28` 的一部分。

```figure
tp-transport-handshake
```

## 實際使用 (Use It)

`code/main.py` 用 Python 標準函式庫，實作一個有限的現代 Streamable HTTP server。它會驗證 Origin 和鏡射的 header、忽略已移除的 session header、對一般 call 回傳 JSON，並示範有限的 `subscriptions/listen` SSE 串流。

```bash
cd code
python3 main.py --probe
python3 -m unittest discover tests -v
```

這個探測檢查：

- 無效的 Origin 會被拒絕；
- 沒有 session id 的 discovery 會成功；
- `Mcp-Session-Id` 與 `Last-Event-ID` 會被忽略；
- header 不一致會回傳 `-32020`；
- 不支援的 version 會回傳 `-32022`，並附上精確的 `supported` 與 `requested` 資料；
- 被接受、沒有 id 的 notification 會回傳沒有 body 的 HTTP `202`；
- GET 與 DELETE 會回傳 `405`；
- `subscriptions/listen` 是 POST 的 response 串流，其確認訊息、notification 與最終結果都帶有它的 subscription id。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-transport-migrator.md`。它會移除現代的 protocol session、加入 header 與 body 的驗證、用 `subscriptions/listen` 取代獨立的 GET，並讓任何 legacy bridge 保持明顯分離。

## 練習 (Exercises)

1. 從 POST 中移除 `Mcp-Method`。確認 HTTP `400` 與錯誤 `-32020`。
2. 傳送相符的 header 與 body version `2027-01-01`。確認 HTTP `400`、錯誤 `-32022`，以及精確的資料 `{"supported":["2026-07-28"],"requested":"2027-01-01"}`。
3. 為非 ASCII 的 resource URI 傳送 Base64 sentinel 的 `Mcp-Name`。確認解碼後的值會與 `params.uri` 比對。
4. 在最終 response 之前中斷有限的 listen 串流。用新的 JSON-RPC id 重新發出，並重新取得 tool。
5. 在 ping tool 加入明確的 workflow handle。把它綁定到授權主體，而不使用連線 affinity。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| stdio | 透過 client 啟動的子 process 傳送、以換行分隔的 JSON-RPC |
| Streamable HTTP | 單一 endpoint，每則現代訊息都是新的 POST |
| Request-scoped SSE | 包含相關 notification 與最終 response 的 POST response 串流 |
| `subscriptions/listen` | 用於選擇加入的變更 notification 的長壽命 POST 請求 |
| Header mismatch | 鏡射的 header 與 body 不一致時，回傳 HTTP `400` 與 JSON-RPC `-32020` |
| Origin validation | 針對進入連線的 DNS-rebinding 防禦，不是身分驗證 |
| Explicit state handle | 當作一般參數傳遞的應用程式 token，取代隱藏的 session 狀態 |
| Legacy bridge | 僅為相容性而保留、獨立存在的較早世代行為 |

## 延伸閱讀 (Further Reading)

- [MCP Transport Overview](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
