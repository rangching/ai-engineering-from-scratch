# MCP Resources 與 Prompts：無狀態 Server 的可定址 Context (MCP Resources and Prompts: Addressable Context for Stateless Servers)

> Tool 執行操作。Resource 公開可定址的內容。Prompt 封裝由使用者選擇的訊息 template。好的 MCP server 會讓這三種合約彼此分離，而且行為可預測。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07 (Building an MCP Server), Phase 13, Lesson 09 (MCP Transports)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 從使用者（consumer）的意圖，在 tool、resource 與 prompt 之間做選擇。
- 透過強制要求的 `server/discover` 公告 resource 與 prompt 的介面。
- 建構確定性的 `resources/list` 與 `prompts/list` 結果。
- 套用 `ttlMs` 與 `cacheScope`，而不洩漏使用者專屬的資料。
- 對無效或未知的 resource URI，回傳 JSON-RPC 錯誤 `-32602`。
- 開啟 `subscriptions/listen` 的 POST-response 串流，並以 subscription ID 對應每個事件。
- 把 resource 內容與 prompt template 視為不受信任的 server 輸出。

## 從使用者出發 (Start With the Consumer)

誤用 MCP 最簡單的方式，就是從實作程式碼開始。資料庫查詢因為函式很熟悉，就變成了 tool。可重複使用的 workflow 因為存在檔案裡，就變成了 resource。Prompt 因為 host 可以注入，就變成了隱藏的政策。

請從「誰來選擇」和「他們期待什麼」開始。

| Primitive | 主要意圖 | 選擇者 | 典型結果 |
|---|---|---|---|
| Tool | 執行一個操作 | 模型或應用程式 | 結構化的動作結果 |
| Resource | 讀取某個 URI 的內容 | Host、應用程式或使用者 | 文字或二進位內容 |
| Prompt | 啟動可重複使用的訊息 workflow | 使用者，透過 host UI | 一則或多則 prompt 訊息 |

`notes://note-1` 的 note 是 resource，因為它是可定址的內容。`delete_note` 是 tool，因為它會改變狀態。`review_note` 是 prompt，因為使用者選擇的是預先準備好的審閱 workflow。

不要為了看起來完整，就把同一個操作同時公開成三種。每多一個介面，就需要 discovery、授權、快取、錯誤處理、測試和文件。

## 2026-07-28 的無狀態 Envelope (The 2026-07-28 Stateless Envelope)

這堂課以 MCP protocol revision `2026-07-28` 為目標。在這個 profile 中沒有初始化 handshake，也沒有 protocol session。每個請求都在保留的 `_meta` key 中帶有它的 protocol version 與 client capabilities。

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "resources/list",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      },
      "io.modelcontextprotocol/clientCapabilities": {}
    }
  }
}
```

Server 必須實作 `server/discover`。它的結果會公告支援的
version、resource 與 prompt 的 capabilities、實作的 identity，以及
cache 提示。Client 可以直接呼叫其他 method，但 discovery 能在它建構 UI 之前，
提供一份穩定的快照。

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "resources": {"listChanged": true, "subscribe": true},
    "prompts": {"listChanged": true}
  },
  "ttlMs": 3600000,
  "cacheScope": "public"
}
```

一般的結果會宣告 `"resultType": "complete"`。Response 的 `_meta` 用 `io.modelcontextprotocol/serverInfo` 表明提供服務的實作。這項資訊對診斷有用，但它不是用於身分驗證的 identity。帶有不支援 revision 的請求，會回傳 `-32022`，同時附上請求的 revision 和 server 支援的 revision。

無狀態合約會改變你的設計直覺。清單不能依賴同一條連線上先前的 call。授權可能改變可見的集合，因為憑證是請求的輸入，但連線歷史不行。

## Resource 是穩定的 URI 合約 (Resources Are Stable URI Contracts)

Resource 是由 URI 識別的內容。請在寫 handler 之前先設計 URI。

好的 URI 具備的特性：

- 穩定到可以加書籤，或在請求之間傳遞。
- 以 server 的領域作為 namespace。
- 獨立於 process ID 或連線。
- 在存取儲存空間之前先驗證。
- 每次讀取都要授權。

`notes://note-1` 比 `note-1` 好，因為它的 namespace 很明確。檔案 server 可以使用 `file://` URI，但仍然必須在解析 symlink 和相對路徑片段之後，檢查設定好的目錄邊界。

`resources/list` 回傳呼叫者目前可見的 resource。請依穩定的 key（例如 URI）排序。確定性的順序可避免嘈雜的 cache miss、快照變動，以及在重新整理之間跳動的 host UI。

```json
{
  "resultType": "complete",
  "resources": [
    {
      "uri": "notes://note-1",
      "name": "Architecture decision",
      "description": "Why the service uses a stateless boundary",
      "mimeType": "text/markdown"
    }
  ],
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

`resources/read` 回傳一個或多個內容項目。未知的 URI 不是成功的空讀取。目前的 Resources 規格，把無效或未知的 resource URI 歸類為 JSON-RPC 的無效參數，代碼 `-32602`。

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "error": {
    "code": -32602,
    "message": "Unknown or invalid resource URI",
    "data": {
      "uri": "notes://missing"
    }
  }
}
```

這個區分讓 client 能把「不存在」和「有效的空文件」分開。它也能防止意外退回到範圍更廣的查詢。

### Resource template

Resource template 描述一整族有參數的 URI。當列出每個具體項目的成本很高或沒有上限時，就使用它。例如，`notes://projects/{project}/decisions/{decision}` 告訴 client 如何組出有效的位址，而不必回傳每一個 decision。

Template 不會削弱驗證。要解析變數、套用授權、強制長度與字元限制，並用有型別的參數建構儲存空間的查詢。絕不要把任意的 URI 尾段串接到檔案系統路徑或資料庫語句裡。

### 內容不是受信任的指令 (Content is not trusted instruction)

Resource 文字可能含有 prompt injection、機密、誤導性的指令或格式錯誤的標記。Host 應該保留來源（provenance），並把 resource 內容視為資料。Server 應該限制內容大小、回傳準確的 MIME type、遮蔽呼叫者無權存取的欄位，並避免回傳不相關的紀錄。

## Prompt 是由使用者控制的 Template (Prompts Are User-Controlled Templates)

MCP prompt 是為了讓使用者明確選擇而設計的。Host 可以把它們呈現為 slash command、選單項目或 workflow 按鈕。Protocol 並不要求特定的 UI。

在相同的請求授權下，`prompts/list` 應該是確定性的。每個 prompt 都需要穩定的 name、有用的 description，以及參數宣告，讓 host 能在 `prompts/get` 之前收集輸入。

```json
{
  "resultType": "complete",
  "prompts": [
    {
      "name": "review_note",
      "title": "Review a note",
      "description": "Review one note for a named concern",
      "arguments": [
        {
          "name": "uri",
          "description": "The note resource URI",
          "required": true
        }
      ]
    }
  ],
  "ttlMs": 600000,
  "cacheScope": "public"
}
```

`prompts/get` 把參數解析成訊息。它不會取代 host 的 system instruction。Host 決定回傳的訊息如何進入模型 context，並讓自己受信任的政策保持較高的優先順序。

在 server 的邊界驗證 prompt 參數。Prompt URI 應該通過與直接讀取 resource 相同的授權檢查。不要讓 prompt 變成繞過 resource 存取的旁路。

## Cache 提示是正確性的一部分 (Cache Hints Are Part of Correctness)

`ttlMs` 告訴 client 結果可以重用多久。`cacheScope` 描述誰可以共用該快取值。

| Scope | 意義 | 典型用途 |
|---|---|---|
| `public` | 在授權允許時，可跨使用者重用 | 公開的 prompt 目錄 |
| `private` | 綁定到發出請求的使用者或憑證 context | 使用者自己的 note 內容 |

依資料的變動速度，以及過期資料造成的損害來選擇 TTL。五分鐘可能適合公開的 prompt 目錄。私人 note 的讀取可以用一分鐘。

MCP 只定義 `public` 與 `private` 作為 `cacheScope` 的值。對含有機密或變動快速的結果，回傳 `cacheScope: "private"` 搭配 `ttlMs: 0`，再在 host 的 cache 政策中套用更嚴格的 no-store 規則。`no-store` 本身不是 MCP 的 `cacheScope` 值。

Cache 提示絕不能取代授權。Cache key 必須包含所有會改變可見性的請求維度，包括 tenant、使用者、scope、locale 和分頁 cursor。如果共用的 cache 無法安全地表達這些維度，就使用 `private` 搭配零 TTL，並採用 host 層級的 no-store 政策。

## 訂閱使用由 Client 開啟的 Response 串流 (Subscriptions Use a Client-Opened Response Stream)

現代的訂閱模式取代了先前的 `resources/subscribe` RPC 和舊的 HTTP GET 事件 endpoint。

Client 把 `subscriptions/listen` 當作一般的 JSON-RPC 請求送出。在 Streamable HTTP 上，這是一個 POST，其 response 會以 SSE 串流的形式保持開啟。`notifications` 物件是 allowlist。Server 不得傳送沒有被請求的 notification 類型。

```json
{
  "jsonrpc": "2.0",
  "id": 17,
  "method": "subscriptions/listen",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "course-client",
        "version": "1.0.0"
      }
    },
    "notifications": {
      "resourcesListChanged": true,
      "promptsListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

Request ID 就是 subscription ID。在任何被請求的事件之前，server 會先送出 `notifications/subscriptions/acknowledged`。它的 filter 只包含 server 接受的子集合。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/subscriptions/acknowledged",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "notifications": {
      "resourcesListChanged": true,
      "resourceSubscriptions": [
        "notes://note-1"
      ]
    }
  }
}
```

該串流上之後的每個事件，都帶有相同的 metadata。

```json
{
  "jsonrpc": "2.0",
  "method": "notifications/resources/updated",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/subscriptionId": 17
    },
    "uri": "notes://note-1"
  }
}
```

這個 notification 表示 resource 已變更。Client 會透過 `resources/read` 再讀一次，並受當前授權限制。它不會假設事件中就包含新的文件。

數個訂閱可以共用一條 stdio 通道。Subscription ID 讓 client 能分流它們。在 HTTP 上，關閉 response 串流就會取消訂閱。Server 若正常結束串流，會回傳最終的 `resultType: "complete"` response，並對應到原本的請求。

不要把訂閱串流當作 protocol session。之後的讀取仍然是一個完整的請求，可以抵達任何健康的 server 實例。

```figure
t3-primitive-sort
```

## 互動實驗 (Interactive Lab)

用這張圖，把專案追蹤工具的五項功能分類：issue 詳細資料、建立 issue、sprint 審閱 template、專案政策，以及關閉 issue。接著判斷哪些清單可以公開快取、哪些讀取必須保持 private，以及哪些 resource 值得更新通知。

每一項分類都要指出選擇者。如果是模型執行動作，就用 tool。如果是 host 讀取以 URI 定址的內容，就用 resource。如果是使用者啟動預先準備好的訊息 workflow，就用 prompt。

## 實作練習 (Practice Lab)

從 repository 根目錄執行 simulator：

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

依以下順序檢查 transcript：

1. 確認 `server/discover` 公告了目前的 revision 以及兩種 capability。
2. 確認兩個 list 結果都已排序，並使用 `resultType: "complete"`。
3. 確認 list 與 read 結果帶有刻意設定的 cache 提示。
4. 把 read 的 URI 改成 `notes://missing`，並觀察 `-32602`。
5. 確認訂閱的確認訊息出現在 resource 事件之前。
6. 確認事件與正常關閉都帶有 subscription ID `5`。

這個 Python 模型不會開啟真正的 HTTP 連線。它代表 SDK 必須放到請求範圍 response 串流上的訊息。在 production 中，請使用官方 SDK 處理 framing 與 transport。

## 交付的成品 (Shipped Artifact)

`outputs/skill-primitive-splitter.md` 是可重複使用的設計審查，用於選擇 MCP primitive。它現在會檢查確定性的 discovery、cache scope、無效 URI 的行為，以及現代的訂閱 filter。

這堂課也附上 `assets/primitive-split.svg`，是 primitive 與訂閱邊界的靜態版本，可供離線學習。

## 驗證 (Verify It)

```bash
cd phases/13-tools-and-protocols/10-mcp-resources-and-prompts/code
python3 main.py
python3 -m unittest discover tests -v
```

預期結果：主程式會印出 JSON transcript，而測試指令會回報至少十二個通過的測試。

## 與 Capstone 的連結 (Capstone Connection)

當你的 capstone server 在動作之外，也公開可定址的知識時，請使用這份合約。包含一份確定性的目錄快照、一次經過授權的 resource 讀取、一次 prompt 解析、一個無效 URI 案例，以及一份訂閱 transcript。

你的證據應該顯示，沒有任何清單依賴連線歷史，而且訂閱事件絕不會授予對底層 resource 的存取權。

## 練習 (Exercises)

1. 新增 `notes://projects/{project}/notes/{id}` resource template，並驗證兩個變數。
2. 為 `resources/list` 加入分頁，同時保持確定性的順序。
3. 把一個 resource 改成 `cacheScope: "private"` 搭配 `ttlMs: 0`，加入 host 層級的 no-store 政策，並說明哪個威脅讓這兩項控制都有必要。
4. 新增 prompt-list 變更的訂閱，並證明當 filter 省略 `promptsListChanged` 時，不會送出任何事件。
5. 建立兩個同時進行的訂閱，並證明每個事件都帶有正確的 request ID。
6. 在 read handler 加入授權主體，並證明 cache 項目不能跨主體使用。

## 關鍵術語 (Key Terms)

- **Resource：** MCP server 公開、以 URI 定址的內容。
- **Prompt：** MCP server 公開、由使用者控制的訊息 template。
- **Deterministic list：** 在相同請求輸入下，成員與順序都穩定的 discovery 結果。
- **`ttlMs`：** 以毫秒為單位的 cache 新鮮度期間。
- **`cacheScope`：** 快取結果的共用邊界。
- **`subscriptions/listen`：** 長壽命的請求，其 response 串流會傳送明確過濾過的 notification。
- **Subscription ID：** 原始 listen 請求的 ID，會在 notification metadata 中重複出現。
- **Invalid parameters：** JSON-RPC 錯誤 `-32602`，用於無效或未知的 resource URI。
- **Unsupported protocol version：** JSON-RPC 錯誤 `-32022`，包含 `supported` 與 `requested` 的 revision。
- **`server/discover`：** 強制要求的 server method，回傳支援的 revision、capabilities、identity，以及選擇性的 cache 提示。

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP 2026-07-28 Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP 2026-07-28 Subscriptions](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/subscriptions)
- [MCP 2026-07-28 Caching](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/caching)
