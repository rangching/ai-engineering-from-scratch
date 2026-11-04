# 無狀態 Protocol 上的 MCP Apps (MCP Apps on the Stateless Protocol)

> 互動式的結果，本質上仍是 MCP 的 tool 與 resource 交換。2026-07-28 的核心讓這個交換變得自成一體，而 Apps extension 則加上了位於 sandbox 內的瀏覽器介面。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 透過 `server/discover` 與每請求的 extension capabilities 公告 MCP Apps。
- 在 tool 被呼叫之前，就在 tool 上宣告 `ui://` resource。
- 在 2026-07-28 的無狀態 wire 上回傳完整的 tool 與 resource 結果。
- 區分 Apps 的 `ui/initialize` bridge 訊息，與已被移除的 MCP 核心 handshake。
- 套用 origin 驗證、sandbox、CSP 與最小權限的 permission。

## 問題 (The Problem)

文字結果可以描述一條時間軸，卻沒辦法給使用者一條可以篩選、檢視或據以行動的時間軸。

MCP Apps 用一個選擇性的 extension 解決呈現問題。Tool definition 指向一個 `ui://` resource。Host 可以在 tool 執行之前，先取得並審查該 resource，把它轉譯在 sandbox 的 iframe 中，並透過 JSON-RPC bridge 居中處理所有 App 的動作。

核心 protocol 在 2026-07-28 改變了。不要把 App 包進舊的連線生命週期：

- 沒有核心的 `initialize` 請求或 `notifications/initialized` notification。
- 沒有 `Mcp-Session-Id` header。
- 每個請求都在 `params._meta` 中帶有 protocol version 與 client capabilities。
- Server 實作 `server/discover`，讓 client 可以檢視 version、核心 capabilities 與 extension。
- 每個成功的結果都有 `resultType` discriminator。
- Streamable HTTP 每個請求使用一個 POST。現代的 GET 與 DELETE 入口會回傳 405。

Apps bridge 仍然有一個叫做 `ui/initialize` 的 method。它屬於 iframe postMessage 的方言。它不會重新建立核心的 MCP session。

## 核心概念 (The Concept)

### 兩個 protocol，一個功能

把各層明確地分開：

1. MCP 核心承載 `server/discover`、`tools/list`、`tools/call`、`resources/list` 與 `resources/read`。
2. MCP Apps extension 宣告 UI，並定義 iframe 到 host 的 bridge。
3. 瀏覽器的 sandbox 規則限制 UI 能碰到什麼。

Extension 的識別碼是 `io.modelcontextprotocol/ui`。雙方 peer 都要選擇加入。Client 在每個請求的 capabilities 物件中送出 extension 支援：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "server/discover",
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "extensions": {
          "io.modelcontextprotocol/ui": {}
        }
      },
      "io.modelcontextprotocol/clientInfo": {
        "name": "timeline-host",
        "version": "1.0.0"
      }
    }
  }
}
```

建議提供 `clientInfo` 供診斷使用。它是自行申報的資料，不是授權用的 identity。

### 轉譯之前先 discover (Discover before rendering)

Server 的 discovery 結果會公告這個 extension：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {},
    "resources": {},
    "extensions": {
      "io.modelcontextprotocol/ui": {}
    }
  },
  "ttlMs": 300000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "timeline-app-server",
      "version": "2.0.0"
    }
  }
}
```

Server 必須支援 discovery。Client 不必在每個動作之前都呼叫 discovery，因為每個動作都帶有自己的 capabilities。

### 在 tool definition 上宣告 UI

現代的 Apps 合約，在 `tools/list` 中把 UI 綁定到 tool：

```json
{
  "name": "notes_timeline",
  "description": "Render a timeline of notes.",
  "inputSchema": {
    "type": "object",
    "properties": {}
  },
  "_meta": {
    "ui": {
      "resourceUri": "ui://notes/timeline.html"
    }
  }
}
```

這刻意是呼叫之前的 metadata。Host 可以在結果要求顯示之前，預先載入、快取並對 HTML 做安全審查。較舊的扁平 metadata key 或許能被相容性程式碼接受，但新的 server 應該輸出巢狀的 `_meta.ui.resourceUri` 形式。

`tools/list` 在目前的核心中是可快取的。請包含確定性的排序、`ttlMs` 與 `cacheScope`。當可見的 tool 會隨使用者或 token 而不同時，使用 `private`。

### 回傳資料，再讓 host 綁定 view

Tool call 回傳一般的內容加上結構化的資料：

```json
{
  "resultType": "complete",
  "content": [
    {"type": "text", "text": "Timeline ready."}
  ],
  "structuredContent": {
    "notes": [
      {"id": "note-1", "title": "Discover", "created": "2026-07-28"}
    ]
  },
  "isError": false
}
```

Host 已經知道哪個 view 屬於這個 tool。不要只為了重複 URI，就發明新的 content block。

### 把 app 當作 resource 提供

Server 在 discovery 中公告了 `resources`，所以也實作強制要求的 `resources/list` 操作。它確定性的 list 項目包含 canonical URI、穩定的 name、description 與 MIME type。List 結果包含 `resultType`、server identity metadata、`ttlMs` 與 `cacheScope`，就像確定性的 tool list 一樣。

Host 送出 `resources/read`。在 Streamable HTTP 上，請求帶有：

```text
POST /mcp
MCP-Protocol-Version: 2026-07-28
Mcp-Method: resources/read
Mcp-Name: ui://notes/timeline.html
```

Header 的值必須與 JSON-RPC body 相符。不一致是 protocol 錯誤 `-32020`。

結果包含 HTML resource 與 cache 提示：

```json
{
  "resultType": "complete",
  "contents": [
    {
      "uri": "ui://notes/timeline.html",
      "mimeType": "text/html;profile=mcp-app",
      "text": "<!doctype html>...",
      "_meta": {
        "ui": {
          "csp": {
            "connectDomains": [],
            "resourceDomains": [],
            "frameDomains": [],
            "baseUriDomains": []
          },
          "permissions": {}
        }
      }
    }
  ],
  "ttlMs": 60000,
  "cacheScope": "public"
}
```

### 把 UI resource 當作可執行內容來快取

App resource 與一般的散文不可互換。它的 cache 項目可以執行 bridge 程式碼、轉譯 tool 資料，並請求由 host 居中處理的動作。請以 canonical `ui://` URI、已接納的 server identity 與 version、resource 內容 digest，以及（當 `cacheScope` 是 private 時）授權 context 作為 key。絕不要在不同 principal 之間重用 private 的 App resource，因為即使 URI 相同，HTML 或它的政策 metadata 也可能不同。

當它的 `ttlMs` 到期、tool 的 `_meta.ui.resourceUri` 綁定改變、server version 或已接納的 descriptor pin 改變，或已確認的 resource 變更訂閱點名該 URI 時，讓該項目失效。在重新掛載之前，重新取得並重新套用 CSP 與 permission 審查。過期的 iframe 不能只因為新的 resource version 還沒載入，就保有更廣的權限。

### 在功能政策之前先拒絕 wire 的歧義

驗證有刻意安排的順序。先驗證 JSON-RPC 的形狀，並要求有字串形式的 protocol metadata，以及物件形式的 client capability map。接著比對路由 header 與 body。然後才決定一致的 protocol version 是否被支援。這個順序可以避免 proxy 與 server 解讀出不同的請求。

| 條件 | HTTP | JSON-RPC 錯誤 |
|-----------|------|----------------|
| Header 與 body 的 version、method 或 name 不一致 | 400 | `-32020` |
| Header 與 body 對一個不支援的 version 看法一致 | 400 | `-32022`，`data` 精確為 `{"supported":["2026-07-28"],"requested":"<actual>"}` |
| `resources/read` 缺少 Apps extension capability | 400 | `-32021`，帶有 `data.requiredCapabilities.extensions.io.modelcontextprotocol/ui` |
| Method 未知 | 404 | `-32601` |

JSON-RPC notification 沒有 `id`，所以 server 絕不會為它發出 JSON-RPC response。被接受的 HTTP notification 回傳 202 與空的 body。錯誤可以改變 HTTP 狀態，但仍不能為 notification 建立 JSON-RPC 錯誤 body。

### Sandbox 是邊界，不是信任的裁決

Host 控制 iframe。App 不能直接讀取 host 的 cookie、local storage 或頁面 DOM。所有需要特權的工作都必須跨過 bridge。

使用這些預設值：

- 讓所有 CSP 的 domain 清單保持空白，然後只加入 App 需要的 origin。`connectDomains` 用於 fetch、XHR 和 WebSocket；`resourceDomains` 用於 script、style、image 和 font。
- 可行時，把程式碼與資料一起打包。
- 除非有明顯的功能需要，否則不要請求相機、麥克風或位置的 permission。
- 把 `postMessage` 固定到精確的 peer origin，並拒絕來自其他任何 origin 的事件。
- 把 tool 參數、tool 結果、resource 文字與 bridge 訊息都視為不受信任的輸入。
- 讓使用者同意留在 host 中。Iframe 不能核准自己的 consequential 動作。

不要把教學文章裡固定的 `sandbox` 屬性，複製到每一個 host。Host 必須根據 App 的 origin 模型和自己的隔離設計來選擇旗標。

被允許的 domain 仍然是外洩路徑。`connectDomains: ["https://api.example.com"]` 表示任何在 App 內執行的 script，都可以把被允許的資料送到那裡。精確的 origin 比對能避免目的地混淆，但不能決定 payload 是否適當。預設讓 connect 存取保持空白，避免把 bearer token 放進 iframe，可行時透過 host 以窄範圍的操作做 proxy，限制 response 與 request 的大小，並稽核每個對外請求是由哪個使用者動作引起。把 `resourceDomains` 與 `connectDomains` 分開對待；載入 font 或 script 的 permission，不應該授予任意的資料上傳。

### Apps bridge 有自己的生命週期

Apps bridge 是透過 `postMessage` 的 JSON-RPC 方言。它可以交換 `ui/initialize` 與 `ui/*` notification，也可以 proxy 看起來像核心的 method，例如 `tools/call`。

View 送出帶有 `appInfo` 與 `appCapabilities` 物件的 `ui/initialize`。Host 回傳它的 capabilities 與 host context。只有在該 response 之後，View 才送出 `ui/notifications/initialized`。Host 必須等到這個 Apps notification 之後，才能傳送訊息給 View。

這個本地 handshake 在一個 iframe 與一個 host frame 之間建立 bridge。它不會協商 MCP protocol version、不會建立 server 狀態，也不會產生 transport session。請注意精確的前綴：核心的 `notifications/initialized` 已被移除，而 Apps 的 `ui/notifications/initialized` 仍然存在。由 bridged tool call 產生的核心請求，是一個新的、自成一體的請求，帶有新的 JSON-RPC id 與完整的請求 metadata。

### Host context、動作與撤銷

Bridge 初始化之後，host 仍是權威。View 只能透過 host 公告過的 capability，請求 tool 動作、導覽、剪貼簿使用或其他特權效果。Host 驗證有型別的請求、當前使用者、目標與參數，套用核准政策，並可以拒絕。按鈕點擊和有效的 bridge 訊息表達的是意圖；兩者都不授予權限。

把主題、大小與無障礙，視為會改變的 host context，而不是一次性的轉譯輸入：

- 套用 host 提供的色彩與字型 token，然後在主題或對比偏好改變時做出反應。
- 讓 View 回報想要的尺寸，但由 host 限制並套用 iframe 的大小，讓內容無法逃出它的版面，或製造欺騙性的覆蓋。
- 在 iframe 內保留鍵盤順序、可見的 focus、無障礙名稱、螢幕閱讀器狀態、足夠的對比、縮放，以及 reduced-motion 行為。
- 在 resize 與重新轉譯之後，重新測試 host 控制項與 View 控制項之間的 focus 轉移。

當 App 開啟時，capability 可能被撤銷，原因包括使用者切換帳號、政策改變、server 被隔離，或 host 縮小同意的範圍。要在動作當下檢查 capability 與授權，而不是只在 `ui/initialize` 時檢查。撤銷時，拒絕待處理的特權 call，停止不再符合政策的網路活動，清除敏感的已轉譯狀態，並在 UI resource 本身不再被接納時，重新掛載或退回到文字。View 必須把拒絕當作一般結果來處理，而不是一直重試到 host 讓步。

### Fallback 是合約的一部分

支援 Apps 的 server，仍然可以服務沒有公告 UI extension 的 host：

- 在 `tools/list` 中回傳同一個 tool，但不帶 `_meta.ui`。
- 為 `tools/call` 保留有用的文字結果。
- 以缺少 capability 的錯誤，拒絕針對 UI 的 `resources/read`。
- 在判斷 tool 是否完成時，絕不要假設存在 iframe。

```figure
t3-ui-sandbox
```

## 動手做 (Build It)

`code/main.py` 不用 SDK，建構一個小型的 in-process protocol 模型。它驗證目前的請求 envelope 與 Streamable HTTP 的路由值，透過 `server/discover` 公告 Apps，列出 tool 與 resource，執行 tool，並提供自成一體的 HTML resource。

這個模型接收的是已經解析好的 body 與路由 header。它不是完整的 HTTP adapter，也不解析 `Content-Type` 或 `Accept`。完整的 Streamable HTTP adapter 請見課程 09，它要求 `Content-Type: application/json`，以及同時包含 `application/json` 和 `text/event-stream` 的 `Accept` 值。

執行它：

```bash
cd phases/13-tools-and-protocols/14-mcp-apps
python3 code/main.py
python3 -m unittest discover code/tests -v
```

在輸出中檢查五件事：

1. 每個 call 都是獨立的。
2. 每個請求都有 `_meta` capabilities。
3. `resources/list` 在任何 resource 讀取之前，就先回傳穩定的 descriptor。
4. 每個結果都有 `resultType` 與 server identity metadata。
5. 沒有出現任何核心的 session 識別碼。

## 實際使用 (Use It)

從 `server/discover` 開始。確認 `io.modelcontextprotocol/ui` 出現在 server 的 extension map 中。接著呼叫 `tools/list` 兩次，一次帶 Apps capability，一次不帶。第一個 response 宣告 resource。第二個仍是可用的純文字 tool。

讀取 `ui://notes/timeline.html`。在 HTML 中搜尋 `hostOrigin` 和 `event.origin` 檢查。這兩行是 bridge 不使用萬用字元目標的最低限度可見證明。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-apps-spec.md`。在寫框架程式碼之前，用它來審查 App 合約。它迫使作者說明目前的核心 envelope、extension 協商、fallback、UI resource、cache 政策、CSP、permission、bridge method 與同意邊界。

## 練習 (Exercises)

1. 把 client capability 改成空的 extension map。確認 `tools/list` 保留 tool，但移除 UI 綁定。
2. 傳送 `Mcp-Name: ui://notes/other.html`，body 則是讀取 timeline。確認錯誤 `-32020`。
3. 把 resource 改成 `cacheScope: private`。說明讓這麼做合理的使用者專屬條件。
4. 把 script 移到 `https://static.example.com/app.js`。把該 origin 加進 `resourceDomains`，並說明新的供應鏈風險。
5. 新增 `notes_open` tool，並讓按鈕點擊透過 host 路由。讓使用者核准留在 host 中。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| MCP Apps | 由 MCP host 轉譯互動式 HTML 的選擇性 extension |
| `io.modelcontextprotocol/ui` | 雙方 peer 都會公告的 extension 識別碼 |
| `ui://` | App 的 UI template 所用的 resource scheme |
| `text/html;profile=mcp-app` | MCP App HTML 的 MIME type |
| `server/discover` | 用於 protocol 與 capability discovery 的目前 RPC |
| `resources/list` | server 公告 resources 時，強制要求的 resource 列表 method |
| `resultType` | 現代成功結果必備的 discriminator |
| `ui/initialize` | 第一個 Apps bridge 請求，與已被移除的核心初始化無關 |
| `ui/notifications/initialized` | host 回應之後，由 Apps View 送出的就緒 notification |
| CSP | 限制 script、style、image 與網路 origin 的瀏覽器政策 |
| Text fallback | 為不支援 Apps 的 host 保留的 tool 行為 |

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP Apps overview](https://modelcontextprotocol.io/extensions/apps/overview)
- [MCP Apps build guide](https://modelcontextprotocol.io/extensions/apps/build)
- [Official extension support matrix](https://modelcontextprotocol.io/extensions/client-matrix)
