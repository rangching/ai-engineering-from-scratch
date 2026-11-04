# Model Context Protocol (MCP)

> MCP 為 AI host 提供一套協定，用來發現並呼叫 tool、resource 與 prompt。2026-07-28 版本讓這套協定變成無狀態 (stateless)：capability 與版本資訊隨著每個請求一起傳送，而不是綁在連線的 handshake 上。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 09 (Function Calling), Phase 11 · 03 (Structured Outputs)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 區分 MCP 的 host、client、server、transport 與 server primitive。
- 建立帶有 MCP 2026-07-28 所需 metadata 的 JSON-RPC 請求。
- 使用 `server/discover` 檢視版本、身分與 capability。
- 從 tool、resource 與 prompt 回傳具型別且感知快取的結果。
- 說明現代的無狀態 MCP 如何與 handshake 時代的 server 互通。
- 為 server 選擇安全的狀態、transport 與核准邊界。

## 問題 (The Problem)

你的應用程式需要資料庫查詢、行事曆操作和檔案讀取器。沒有共用協定的話，每個 AI host 都需要為這些相同的功能，自訂發現、呼叫、錯誤、transport 與授權的黏合程式碼。

MCP 縮減了這個整合矩陣。Server 發布標準的 JSON-RPC 介面。相容的 client 可以發現這個介面、呈現給模型或使用者、呼叫它，並解讀結果，不需要針對特定 server 的 adapter。

有一條重要的界線很容易被忽略。MCP 標準化的是通訊。它不會決定模型該呼叫哪個 tool、不會讓不受信任的內容變安全，也不會把無狀態的請求變成持久的應用程式狀態。這些決定仍然由你的 host 和 server 負責。

## 核心概念 (The Concept)

![MCP host, stateless request, and server primitives](../assets/mcp-architecture.svg)

### 三種 server primitive

1. **Tools** 是可呼叫的動作。每個 tool 都有名稱、描述、JSON Schema 輸入與 handler。
2. **Resources** 是具名、以 URI 定址、client 可以讀取的內容。
3. **Prompts** 是 host 可以暴露給使用者的可重複使用 template。

Host 是 AI 應用程式。Host 內的 MCP client 與一個 server 溝通。Transport 在兩者之間傳送 JSON-RPC 訊息。

### 無狀態請求取代 Handshake

MCP 2026-07-28 移除了 `initialize` 與 `notifications/initialized`。它也移除了協定層級的 session。每個請求都在 `params._meta` 中攜帶解讀它所需的 context：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

協定版本與 client capability 是必填的。Client 身分則是建議提供。缺少 `_meta`、缺少必填欄位，或必填欄位型別錯誤，都屬於格式錯誤，會回傳 Invalid Params（`-32602`）。格式正確但 server 不支援的版本字串，會回傳 `UnsupportedProtocolVersionError`（`-32022`）。Server 可以處理有效的請求，而不需要找回先前的協商紀錄。

無狀態不代表應用程式永遠不能維護狀態。它的意思是，狀態不會藏在 MCP 連線或 `Mcp-Session-Id` 背後。如果工作流程需要連續性，server 會產生一個不透明的 handle，client 在之後的呼叫中把這個 handle 當作一般的 tool 引數傳入。每個請求仍然必須檢查授權。

### 發現與版本選擇 (Discovery and version selection)

每個現代 server 都實作 `server/discover`。結果會公告支援的版本、capability 與 server 身分：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "complete",
    "supportedVersions": ["2026-07-28"],
    "capabilities": {
      "tools": {},
      "resources": {},
      "prompts": {}
    },
    "ttlMs": 3600000,
    "cacheScope": "public",
    "_meta": {
      "io.modelcontextprotocol/serverInfo": {
        "name": "demo-server",
        "version": "1.0.0"
      }
    }
  }
}
```

Client 可以直接呼叫其他方法並處理版本錯誤，但 discovery 讓 capability 的顯示與版本選擇更明確。不支援的版本會回傳 `UnsupportedProtocolVersionError`，code 為 `-32022`。它的 data 包含 `supported`（server 支援的版本陣列）與 `requested`（被拒絕的版本）。

在 stdio 上，同時支援兩個時代的 client 會用 `server/discover` 探測。Discovery 的結果，或是像 `UnsupportedProtocolVersionError` 這類被認得的現代錯誤，代表這是現代 server。任何未被認定為現代的錯誤或逾時，都允許回退到 2025-11-25 的 `initialize` 流程。舊版行為是相容性程式碼，不是現代的預設。

### 結果是明確的 (Results are explicit)

2026-07-28 的每個核心結果都有 `resultType`：

- `complete` 代表操作已完成。
- `input_required` 代表 server 需要透過 Multi Round-Trip Requests 模式再來回一次。核心 server 只能從 `tools/call`、`resources/read` 或 `prompts/get` 回傳它。

Client 必須把省略 `resultType` 的舊版結果視為 complete。

Server 應該在每個結果的 `_meta` 中包含 `io.modelcontextprotocol/serverInfo`。這個身分是自行回報的，用於顯示、logging 與除錯，不應用於安全決策。

List 與 read 的結果也帶有 `ttlMs` 與 `cacheScope`。確定性的 `tools/list` 順序加上新鮮度提示，讓 client 能安全地快取 discovery，並提升 prompt-cache 的穩定性。`cacheScope: public` 允許共用快取；`private` 則把重複使用限制在呼叫端的 context 內。

### 傳輸格式與 transport (The wire format and transport)

MCP 在 stdio 或 Streamable HTTP 上使用 JSON-RPC 2.0。

- 請求包含 `jsonrpc`、`id`、`method` 與 `params`。
- 回應有相符的 `id`，以及 `result` 或 `error` 其中之一。
- Notification 沒有 `id`，也不預期回應。

現代的 Streamable HTTP 暴露一個接受 POST 的 endpoint。每則 JSON-RPC 訊息都有自己的 POST。請求的 POST 會收到單一 JSON 物件，或是一個以最終回應結尾、限定於該請求的 Server-Sent Events 串流。被接受的 notification POST 會收到 HTTP 202 且沒有回應 body；這個核心版本沒有定義透過 Streamable HTTP 的 client 到 server 的 notification。

2026-07-28 中沒有獨立的 MCP GET 串流、DELETE session endpoint、`Mcp-Session-Id`，或 `Last-Event-ID` 重播。長時間存活的變更通知，使用 `subscriptions/listen` POST，其回應會以 SSE 串流維持開啟。

### 沒有 server 發起請求的 Client 輸入 (Client input without server-initiated requests)

較舊的版本讓 server 可以透過串流送出 `sampling/createMessage`、`roots/list` 或 `elicitation/create` 這類請求。目前的協定改用 Multi Round-Trip Requests。符合資格的 tool call、resource read 或 prompt get 會回傳 `resultType: input_required`，並至少帶有 `inputRequests` 或 `requestState` 其中之一。Client 收集所要求的輸入，以新的 JSON-RPC ID 與對應的 `inputResponses` 重試原本的方法，並在有提供 `requestState` 時原封不動地回傳。如果沒有 `inputRequests`，重試時就省略 `inputResponses`。

Roots、Sampling 與 Logging 仍可運作但已被棄用，所以新的實作不應該採用它們。既有的 Roots 或 Sampling 請求會放在 MRTR 的 `inputRequests` 內傳送，絕不會作為獨立的 server 到 client 的 JSON-RPC 請求。請優先使用明確的檔案或目錄參數、resource URI、server 設定，以及直接的模型供應商整合。stdio 的診斷訊息請用 stderr，production 的遙測請用 OpenTelemetry。

```figure
mcp-nxm-collapse
```

## 動手做 (Build It)

### 步驟 1：註冊 server 介面

即使請求的契約改變了，註冊仍然很簡單：

```python
server = MCPServer("demo-server")

@server.tool(
    "add",
    "Add two integers.",
    {
        "type": "object",
        "properties": {
            "a": {"type": "integer"},
            "b": {"type": "integer"}
        },
        "required": ["a", "b"]
    }
)
def add(a: int, b: int) -> dict:
    return {"sum": a + b}
```

`code/main.py` 中附帶的實作也註冊了一個 resource 和 prompt。它刻意只使用標準函式庫，讓你能看到每個 envelope，而不是把協定交給 SDK 處理。

### 步驟 2：在每個請求上附加 metadata

```python
def request(method, params=None):
    body_params = dict(params or {})
    body_params["_meta"] = {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {},
        "io.modelcontextprotocol/clientInfo": {
            "name": "demo-client",
            "version": "1.0.0"
        }
    }
    return {
        "jsonrpc": "2.0",
        "id": 1,
        "method": method,
        "params": body_params
    }
```

不要只把這份 metadata 快取在連線物件裡。Server 會在每個請求上驗證它。

### 步驟 3：列出之前可選擇性地先做 discovery

呼叫 `server/discover`，選擇一個支援的版本，然後呼叫 `tools/list`。如果你已經知道版本並且能處理 `-32022`，直接呼叫 `tools/list` 也是有效的。

這個 demo 會依名稱順序回傳 tool 清單，並附上 `ttlMs`、`cacheScope`、`resultType` 與 server 身分。Tool call 會回傳 complete 且不可快取的結果，因為它的輸出可能取決於目前的狀態。

### 步驟 4：把同一個請求對應到 HTTP

遠端的 `tools/call` POST 包含與 JSON-RPC body 對應的 header：

```http
POST /mcp HTTP/1.1
Content-Type: application/json
Accept: application/json, text/event-stream
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: add
```

`MCP-Protocol-Version` header 必須與 `_meta` 中的版本一致。每個 JSON-RPC 請求都需要 `Mcp-Method`，且必須與 `method` 相符。`Mcp-Name` 只在 `tools/call`、`resources/read` 與 `prompts/get` 時需要，此時它必須與 tool 名稱、resource URI 或 prompt 名稱相符。缺少必要的 header 或不一致，會回傳 HTTP 400 與 `HeaderMismatch`，code 為 `-32020`。

### 步驟 5：在協定狀態之外落實安全性

- 在每個 HTTP 請求上驗證授權與 audience。
- 把本機 server 綁定到 localhost，並在 Streamable HTTP 上驗證 `Origin`。
- 把會產生變更的 tool 標記 `destructiveHint: true`，並要求 host 核准。
- 明確傳入目錄與檔案範圍，而不是依賴已被棄用的 Roots。
- 把 resource 與 tool 輸出視為不受信任的資料。
- 在 stdio 下，stdout 保留給 JSON-RPC 使用；診斷訊息寫到 stderr。

## 實際使用 (Use It)

從這堂課的目錄執行：

```bash
python3 code/main.py
cd code
python3 -m unittest discover tests -v
```

第一行應該回報在協定 `2026-07-28` 上發現了 `demo-server`。接著檢視 `MCPClient.request`：它為每次呼叫重新建構 `_meta`。把其中一個請求的 metadata 移除，並觀察 server 拒絕它。

## 交付 (Ship It)

`outputs/skill-mcp-server-designer.md` 把一個領域轉換成無狀態的 MCP 設計。它的驗收門檻要求：discovery 結果、每個請求的 metadata 政策、確定性且感知快取的清單、明確的狀態 handle、transport header、授權與核准規則。

## 繼續深入 MCP (Continue the MCP Deep Dive)

這堂課給你協定模型。Phase 13 把四條 production 邊界拆成各自獨立的建構與驗證課程：

1. [MCP Tool Contracts and Content](../../../13-tools-and-protocols/28-mcp-tool-contracts-and-content/docs/en.md) 涵蓋封閉的輸入 schema、結構化內容、路由 metadata、不透明的分頁、completion 授權，以及協定錯誤與 tool 領域錯誤的差異。
2. [MCP Reliability, Cancellation, and Flow Control](../../../13-tools-and-protocols/29-mcp-reliability-cancellation-and-flow-control/docs/en.md) 涵蓋請求取消、持久任務取消、deadline、idempotency、backpressure、proxy 緩衝與重新連線行為。
3. [MCP Registry Supply Chain, Admission, Drift, and Rollback](../../../13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/docs/en.md) 涵蓋 namespace 證明、artifact 來源、不可變的 pin、即時 drift、Registry 狀態、准入證據與 rollback。
4. [MCP Conformance Engineering](../../../13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/docs/en.md) 涵蓋 golden 與負面的 wire transcript、嚴格的版本時代、SDK 差異比對、proxy 證據、遮蔽、健康閘門與發布 rollback。

當 server 要跨越團隊或信任邊界時，請依序學習。它們一起帶你從「這個方法能用」走到「這份契約在部署過程中仍然安全且可診斷」。

## 練習 (Exercises)

1. 新增一個 `subtract` tool，並確認 `tools/list` 仍維持字母順序。
2. 移除協定版本的 key，並驗證 Invalid Params（`-32602`）。然後送出格式正確但不被支援的版本 `2025-11-25`，驗證 `-32022`，確認 `requested` 回傳的是那個版本，並從 `supported` 中選擇。
3. 在建立操作中加入 server 產生的 `draftId`，然後要求更新時把它當作引數。解釋為什麼這是應用程式狀態，而不是協定 session。
4. 讓一個需要使用者確認的 tool 回傳 `input_required`。用新的 ID、一筆 `inputResponses` 項目與原封不動的 `requestState` 重試原本的呼叫，而不是自創 server 到 client 的 JSON-RPC 請求。
5. 草擬一個同時支援兩個時代的 stdio client。把結果或被認得的現代錯誤視為現代，並且只在遇到未被認得的錯誤或逾時時，才允許回退到 `initialize`。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| MCP | 「給 LLM 用的 tool 協定」 | 用於 server 發現、tool、resource、prompt 與擴充功能的 JSON-RPC 協定 |
| Host | 「AI 應用程式」 | 擁有模型與 UI，並掛載一個或多個 MCP client |
| Client | 「連接器」 | 代表 host 與一個 server 以 MCP 溝通 |
| Stateless MCP | 「沒有 session」 | 每個請求都攜帶版本與 capability；沒有任何協定狀態是以連線為 key |
| `server/discover` | 「Capability 探測」 | 必要的 server 方法，公告版本、capability 與身分 |
| `resultType` | 「結果狀態」 | 將結果標記為 `complete` 或 `input_required` |
| State handle | 「工作流程 ID」 | 由 server 產生、當作一般引數傳入的應用程式識別碼 |
| Streamable HTTP | 「遠端 transport」 | 一個 POST endpoint，搭配 JSON 或限定於請求的 SSE 回應 |
| MRTR | 「詢問後重試」 | 嵌入在結果中的輸入請求，接著重試原本的操作 |

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 key changes](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [MCP Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
