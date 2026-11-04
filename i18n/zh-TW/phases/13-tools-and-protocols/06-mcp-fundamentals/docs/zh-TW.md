# MCP 基礎：無狀態請求與 JSON-RPC (MCP Fundamentals: Stateless Requests and JSON-RPC)

> 現代的 MCP 沒有 handshake，也沒有 protocol session。每個請求都必須自帶足夠的 metadata，才能獨立被理解、授權、路由和重試。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 01 through 05
**Time:** ~55 分鐘

## 學習目標 (Learning Objectives)

- 區分 MCP 的 server primitive 與 client 端功能。
- 為 MCP `2026-07-28` 建構有效的 JSON-RPC 2.0 request 與 response。
- 在每個請求上附帶 protocol version、client capabilities 與 client identity。
- 使用 `server/discover`，並在沒有 handshake 的情況下處理 `UnsupportedProtocolVersionError`。
- 追蹤一個獨立請求，從驗證一路到完整結果。

## 問題 (The Problem)

一個 MCP server 可能在同一個 process 或 HTTP worker 上，連續收到兩個來自不同 client、具備不同 capabilities 的請求。如果 server 記住了前一個請求宣告的內容，就可能套用錯誤的權限，或回傳錯誤的 wire 形狀。

MCP `2026-07-28` 消除了這種歧義。Protocol 核心是無狀態的。Server 必須根據當前的請求，來決定如何處理當前的請求，而不是根據連線歷史。

這改變了心智模型。舊的順序是先連線、再 handshake、最後才操作。現代的順序更簡單：

1. Client 送出一個自我描述的請求。
2. Server 驗證該請求的 version 與 capabilities。
3. Server 處理該 method。
4. Server 回傳有型別的結果，或 JSON-RPC 錯誤。

下一個請求會從頭重複同樣的流程。

## 核心概念 (The Concept)

### Server primitive

MCP server 公開三種主要的 primitive：

1. **Tools** 是由模型控制的動作，用 `tools/list` 探索，用 `tools/call` 呼叫。
2. **Resources** 是以 URI 定址的資料，用 `resources/list` 探索，用 `resources/read` 取得。
3. **Prompts** 是可重複使用的 template，用 `prompts/list` 探索，用 `prompts/get` 轉譯。

Roots、sampling 和 logging 為了相容性仍保留在 `2026-07-28` 的 schema 中，但已被棄用。新的實作應該用明確的 tool 或 resource 輸入來取代 roots、用直接的 model-provider API 取代 sampling，並用 stderr 或 OpenTelemetry 取代 logging。Elicitation 仍可透過 Multi Round-Trip Requests 使用：server 回傳一個 input request，client 再重試原本的操作。現代的 server 絕不會自行發起獨立的 JSON-RPC 請求。

### JSON-RPC envelope

MCP 使用 JSON-RPC 2.0：

- Request：`{jsonrpc, id, method, params}`
- Response：`{jsonrpc, id, result}` 或 `{jsonrpc, id, error}`
- Notification：沒有 `id` 的 `{jsonrpc, method, params}`

請求的 `id` 用來對應一個回應。它不會建立 protocol session。

### 必要的請求 metadata

每個現代請求都在 `params` 裡帶有一個 `_meta` 物件：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "method": "tools/list",
  "params": {
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

Protocol version 與 client capabilities 是必要的。Client identity 則是建議提供。它是自行申報的顯示與除錯資料，不是安全憑證。

Server 不得從先前的請求、stdio process、HTTP 連線或單獨的 transport header 推論出這些值。

### 完整結果與 server identity

每個成功的現代結果都包含 `resultType`。一般的最終結果使用 `"complete"`。Server 也應該在結果 metadata 中表明自己的身分：

```json
{
  "jsonrpc": "2.0",
  "id": 7,
  "result": {
    "resultType": "complete",
    "tools": [],
    "ttlMs": 30000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "notes-server",
        "version": "1.0.0"
      }
    }
  }
}
```

`tools/list`、`resources/list`、`prompts/list`、`resources/templates/list`、`resources/read` 和 `server/discover` 都是可快取的結果。它們包含 `ttlMs` 和 `cacheScope`。安全的預設值是 `ttlMs: 0` 和 `cacheScope: "private"`。清單項目應該有確定性的排序，讓等價的回應產生穩定的 cache key 和穩定的模型 context。

### 沒有 handshake 的 discovery

每個現代 server 都必須實作 `server/discover`。Client 可以在呼叫其他 method 之前先呼叫它，以取得：

- `supportedVersions`
- server 的 `capabilities`
- 選擇性的使用 `instructions`
- 結果 `_meta` 中的 server identity
- cache 提示

Discovery 很有用，但它不是閘門。Client 可以先送 `tools/list`，因為該請求本身就已經帶有 protocol version 與 capabilities。

如果請求的 version 不被支援，server 會回傳 JSON-RPC 代碼 `-32022`，並附帶：

```json
{
  "requested": "2027-01-01",
  "supported": ["2026-07-28"]
}
```

Client 選擇雙方都支援的現代 version，並以新的 JSON-RPC request id 重試。

### 一個請求的生命週期

依以下順序追蹤一個現代請求：

1. 解析一個 JSON-RPC envelope。
2. 確認 `jsonrpc` 是 `"2.0"`、有 `id`、`method` 是字串，而且 `params` 是物件。
3. 要求 `params._meta` 中必須有 version 字串和 capability 物件；格式錯誤或缺少 metadata，會是 `-32602`。
4. 在 HTTP 邊界上，把 version、method 和適用的 name header 與 body 比對。即使兩個 version 值之一不被支援，不一致就是 `-32020`。
5. 確認一致之後，拒絕雖然一致但不被支援的 version，回傳 `-32022`。
6. 檢查必要的 capabilities，然後依 `method` 路由，並驗證該 method 專屬的參數。
7. 在 handler 執行之前，先對具體的操作做驗證身分與授權。
8. 回傳附有 server identity 的完整結果。
9. 忘掉請求範圍內的 protocol metadata。

這個順序能避免兩個元件對不同的 call 做出不同的解讀。Gateway 絕不能在 origin 執行 `params.name: notes.delete` 的同時，授權 `Mcp-Name: notes.read`。它也讓格式錯誤的輸入、header 混淆、version 協商、capability 失敗、授權和 handler 失敗，各自成為獨立的證據。

關閉 stdin 或結束 HTTP response，只是結束 transport 的活動。它不會終止 protocol session，因為現代的 MCP 沒有 protocol session。

### 明確的 legacy 相容性 (Explicit legacy compatibility)

直到 `2025-11-25` 的各個 version，都使用 `initialize`、`notifications/initialized`、以連線為範圍的 capabilities，而在較早的 Streamable HTTP 上，還有選擇性的 protocol session。當雙世代的 client 與舊 server 對話時，這種行為仍然相關。

請把兩個世代分開。現代請求靠必要的每請求 metadata 來辨識。Legacy 連線只能透過文件記載的 fallback 路徑來選擇。不要把 `initialize` 當成 `2026-07-28` server 的預設行為。

因此，「Stateless」在不同世代有不同的意義。在 `2026-07-28` 中，它是 protocol 的不變量：每個一般請求都能獨立解讀，而且不存在 MCP session。在直到 `2025-11-25` 的版本中，初始化和協商出來的 capabilities 屬於連線，所以相容性 adapter 可以保留那份 legacy 連線狀態。雙世代的實作不是一個寬鬆的單一狀態機，而是一個無狀態的現代核心，旁邊接一個隔離的 legacy adapter，並在任何一個 parser 執行之前，先做出明確的選擇決定。

這兩種意義都不禁止持久的應用程式狀態。Workflow、task 或 draft 可以藏在不透明的 handle 後面，存在共用的 store 裡。Client 把該 handle 當作一般輸入送出，每個 replica 都會驗證身分並授權其使用。Protocol context 不得洩漏到那個 store 裡，充當已被移除的 session 的替代品。

```figure
mcp-tool-call
```

## 實際使用 (Use It)

`code/main.py` 不靠任何框架，就能建構、驗證、追蹤並分派現代的 MCP 訊息。執行：

```bash
python3 code/main.py
python3 -m unittest discover code/tests -v
```

在輸出中留意三個不變量：

- 每個請求都會重複它的 `_meta` 欄位。
- 每個成功的結果都是 `resultType: "complete"`，並包含 server identity。
- 清單結果有確定性的排序，並帶有明確的 cache 提示。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-handshake-tracer.md`。檔名沿用歷史名稱以保持穩定，但這份成品現在是無狀態的請求追蹤器。它會獨立稽核每則訊息，只有在真的出現 legacy handshake 流量時，才會加以標記。

## 練習 (Exercises)

1. 把其中一個請求的 protocol version 改成 `2027-01-01`。確認錯誤代碼是 `-32022`，而且 data 中會公告所支援的 version。
2. 從第二個請求中移除 `io.modelcontextprotocol/clientCapabilities`。確認 server 沒有重用第一個請求的 capabilities。
3. 把記憶體內的 tool registry 反轉。確認 `tools/list` 仍然回傳同樣的確定性順序。
4. 把 `cacheScope` 從 `public` 改成 `private`。說明兩種情況下，分別有哪些授權 context 可以重用該回應。
5. 新增一個省略 optional `clientInfo` 的測試。該請求應該仍然有效，因為 client identity 是建議提供，不是必要的。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Stateless protocol | 每個請求都自帶解讀它所需的 metadata |
| Request metadata | `params._meta` 中的 version、client capabilities，以及建議提供的 client identity |
| `server/discover` | 用於取得 version、capabilities、instructions 與 identity 的強制 server method |
| `resultType` | 每個成功的現代結果上的 discriminator |
| Cacheable result | 包含必要的 `ttlMs` 與 `cacheScope` 提示的結果 |
| Protocol era | 現代的每請求 metadata，或 legacy 以連線為範圍的初始化 |
| Transport lifetime | Process、連線或 response-stream 的生命週期，不是 protocol session 狀態 |
| `-32022` | 不支援的 protocol version 錯誤，附帶請求的與支援的 version |

## 延伸閱讀 (Further Reading)

- [MCP Architecture](https://modelcontextprotocol.io/specification/2026-07-28/architecture)
- [MCP Base Protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP 2026-07-28 Changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
