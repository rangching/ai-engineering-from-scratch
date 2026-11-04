# 建構 MCP Server：無狀態的 Python 與 TypeScript (Building an MCP Server: Stateless Python and TypeScript)

> 現代的 MCP server 不會記住 handshake。它在每個請求上驗證 metadata、執行一個 handler，然後回傳一個有型別的結果。

**Type:** Build
**Languages:** Python, TypeScript
**Prerequisites:** Phase 13, Lesson 06
**Time:** ~85 分鐘

## 學習目標 (Learning Objectives)

- 為 MCP `2026-07-28` 實作強制要求的 `server/discover`。
- 在每個請求上驗證 protocol version 與 client capabilities。
- 公開 tools、resources 與 prompts，並讓清單順序具有確定性。
- 在正確的結果上回傳 `resultType`、server identity 與 cache 提示。
- 在 Python 與 TypeScript 中，透過以換行分隔的 stdio 提供相同的無狀態合約。

## 問題 (The Problem)

一個在第一則訊息之後就儲存 client capabilities 的 server，很容易寫，卻很難維運。同一個 process 可能先後服務不同的 client。遠端請求可能落在不同的 worker 上。過期的 capability 宣告，可能讓行為跨越授權邊界外洩。

MCP `2026-07-28` 讓每個請求都能自我描述，解決了這個問題中屬於 protocol 的部分。你的應用程式仍然可以保留持久的 note、job 或明確的狀態 handle。它不能保留的，是會改變後續請求解碼方式的隱藏 protocol 狀態。

這堂課會把 notes server 建兩次。Python 與 TypeScript 版本的 protocol 核心，都只用各自的標準函式庫。兩者公開相同的 method，並強制執行相同的 wire 合約。

## 核心概念 (The Concept)

### 現代的 dispatch 迴圈

```text
read one JSON-RPC line
parse the envelope
if it is a notification, do not respond
validate params._meta for this request
route by method
wrap success with resultType and serverInfo
write one JSON-RPC response line
forget request-scoped metadata
```

仍有三條 stdio 規則很重要：

- 只把 JSON-RPC 訊息寫到 stdout。診斷訊息送到 stderr。
- 用換行分隔訊息，並在每個回應之後 flush。
- 當 stdin 到達 EOF 時迅速結束。

Process 的生命週期就是 transport 的生命週期。它不是現代的 MCP session。

### 請求驗證 (Request validation)

每個請求都必須有：

```json
{
  "params": {
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "notes-client",
        "version": "1.0.0"
      }
    }
  }
}
```

前兩個欄位是必要的。`clientInfo` 是建議提供。若有提供身分資訊，請驗證其形狀，但不要把它當作身分驗證。

如果 version 不被支援，回傳代碼 `-32022`，並附上 `requested` 與 `supported`。缺少請求 metadata 屬於無效參數，代碼為 `-32602`。絕不要用前一次 call 的內容補齊缺少的欄位。

### 強制的 discovery

現代 server 必須實作 `server/discover`。完整的 discovery 結果包含支援的現代 version、capabilities、選擇性的 instructions、cache 提示，以及結果 `_meta` 中的 server identity：

```json
{
  "resultType": "complete",
  "supportedVersions": ["2026-07-28"],
  "capabilities": {
    "tools": {"listChanged": false},
    "resources": {"listChanged": false, "subscribe": false},
    "prompts": {"listChanged": false}
  },
  "ttlMs": 3600000,
  "cacheScope": "public",
  "_meta": {
    "io.modelcontextprotocol/serverInfo": {
      "name": "notes-server",
      "version": "2.0.0"
    }
  }
}
```

Discovery 不會解鎖 server。Client 可以不呼叫 discovery 就呼叫 `tools/list`，因為 `tools/list` 本身就帶有相同的請求 metadata。

### Tools

`tools/list` 回傳確定性的 tool descriptor 清單。穩定的排序能改善回應快取，也讓模型 context 保持穩定。結果同樣需要 `ttlMs` 與 `cacheScope`。

`tools/call` 回傳 content block 和 `isError`。當 protocol envelope 或 method 參數無效時，使用 JSON-RPC 錯誤。當有效的 tool 呼叫執行了，但 tool 本身失敗時，使用 `isError: true`。

Tool annotation 仍然只是提示，不是強制：

- `readOnlyHint`
- `destructiveHint`
- `idempotentHint`
- `openWorldHint`

Host 應該用它們來做確認與呈現。Server 仍然必須強制執行真正的授權。

### Resources

`resources/list` 回傳穩定的 URI descriptor。`resources/read` 回傳有型別的內容。兩者在 `2026-07-28` 中都是可快取的，所以都包含 `ttlMs` 與 `cacheScope`。

對使用者專屬的 note 資料，請使用 `cacheScope: "private"`。共用的 cache 不得在不同授權 context 之間重用 private 回應。

現代的變更通知不使用 `resources/subscribe`。Client 會開啟 `subscriptions/listen`，並請求 `resourceSubscriptions` 或清單變更類別。課程 10 會建構這個流程。

### Prompts

`prompts/list` 是可快取且具確定性的。`prompts/get` 用參數轉譯指定名稱的 prompt。轉譯後的 prompt 結果是 complete，但它不屬於需要 cache 提示的可快取 list 或 read 結果。

### 每個成功的結果都有型別 (Every successful result is typed)

範例對每個成功結果使用同一個 wrapper：

```python
def complete(payload):
    return {
        "resultType": "complete",
        **payload,
        "_meta": {SERVER_INFO_KEY: SERVER_INFO},
    }
```

List、read 和 discovery 的 handler 會加上 `ttlMs` 與 `cacheScope`。把這個 wrapper 集中管理，可以避免某個 handler 默默漏掉現代結果的欄位。

### 沒有 server 發起的請求 (No server-initiated requests)

現代 server 可以送出與某個 client 請求相關的 notification，或是在 client 開啟的 `subscriptions/listen` 串流上送出 notification。它不得發送自己的 JSON-RPC 請求。

當 handler 需要 sampling、elicitation 或 roots 的輸入時，它會回傳 `input_required` 結果。Client 滿足其中內嵌的 input request，再以新的 request id 重試原本的 method。課程 11 涵蓋這個 Multi Round-Trip Request 模式。

### 明確的 legacy 相容性 (Explicit legacy compatibility)

雙世代的 server 也可以在明確分離的 legacy 分支上，實作 `2025-11-25` 的 handshake。當存在必要的現代 `_meta` 欄位時，它選擇現代行為；收到 `initialize` 時，則選擇 legacy 行為。

不要讓 `2026-07-28` 的請求走過 legacy handshake 路徑。不要把現代的 `resultType` 欄位蓋到 legacy 的初始化結果上。這堂課的程式碼刻意只支援現代版本，好讓它的不變量一目了然。

```figure
t3-dispatch-loop
```

## 實際使用 (Use It)

執行 Python server 的有限 demo 與測試：

```bash
cd code
python3 main.py --demo
python3 -m unittest discover tests -v
```

用 TypeScript runner 執行 TypeScript 版本：

```bash
npx tsx main.ts --demo
```

這個 demo 會送出 `server/discover`、列出每種 primitive、呼叫 tool，並顯示 unsupported-version 錯誤。每個現代請求都會重複 metadata。每個成功結果都包含 server identity。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-server-scaffolder.md`。它會產出現代 server 計畫，內含 discovery 合約、每請求驗證、確定性且可快取的清單，以及選擇性的隔離 legacy adapter。

## 練習 (Exercises)

1. 從一個請求中移除 capabilities，並證明 server 沒有重用前一個請求的宣告。
2. 反轉 `TOOLS`、`PROMPTS` 和 note 的插入順序。確認所有清單結果仍然穩定。
3. 新增一個具破壞性的 `notes_delete` tool，並要求在 executor 內做授權檢查。讓 `destructiveHint` 只作為 UX 提示。
4. 新增 `resources/templates/list`，包含 `ttlMs`、`cacheScope` 和確定性排序。
5. 為 `2025-11-25` 建一個獨立的 legacy adapter。新增測試證明現代請求永遠不會進入它。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Stateless server | 依每個請求自己的 metadata 處理，沒有 protocol-session 記憶 |
| `server/discover` | 公告 version 與 capabilities 的強制現代 method |
| Complete result | 帶有 `resultType: "complete"` 的成功現代結果 |
| Cacheable result | 帶有 `ttlMs` 與 `cacheScope` 的 discovery、list 或 resource-read 結果 |
| Deterministic list | 相同的邏輯 registry 會產生相同的項目順序 |
| Server identity | 結果 `_meta` 中建議提供的 `io.modelcontextprotocol/serverInfo` |
| Tool error | 有效的 tool call，回傳帶有 `isError: true` 的內容 |
| Protocol error | 無效的 JSON-RPC 或 MCP 請求，透過 `error` 回傳 |

## 延伸閱讀 (Further Reading)

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Resources](https://modelcontextprotocol.io/specification/2026-07-28/server/resources)
- [MCP Prompts](https://modelcontextprotocol.io/specification/2026-07-28/server/prompts)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
