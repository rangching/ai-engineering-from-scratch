# 建構 MCP Client：Discovery、Routing 與雙世代 Fallback (Building an MCP Client: Discovery, Routing, and Dual-Era Fallback)

> 現代的 MCP client 在每個請求上都重述它的合約。它最難的相容性決定，是判斷舊 server 什麼時候是真的舊，以及現代 server 什麼時候只是回報了一個可以修正的錯誤。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lesson 07
**Time:** ~85 分鐘

## 學習目標 (Learning Objectives)

- 用最新的 metadata 建構每一個 MCP `2026-07-28` 請求。
- 用 `server/discover` 探測 stdio server，並選出雙方都支援的 version。
- 只對明確列入 allowlist 的 peer，授權有界限的 legacy 探測。
- 只有在驗證過支援 revision 的正向 `initialize` 結果之後，才接受 legacy 世代。
- 合併確定性的 tool 清單，不要默默覆寫名稱衝突。
- 把 call 路由到擁有各 tool 的 peer，而不自行發明 protocol session。

## 問題 (The Problem)

Agent host 通常會同時與不只一個 MCP server 溝通。它必須探索每個 server、合併 tool 目錄、解決重複的名稱、路由 call，並從 transport 失敗中復原。

`2026-07-28` 版讓穩定狀態更簡單，因為每個請求都是獨立完整的。相容性則讓啟動階段更微妙。Client 可能遇到：

- 支援偏好 version 的現代 server；
- 回傳可辨識的 version 或 header 錯誤的現代 server；
- 從未聽過 `server/discover` 的 legacy server；
- 在收到 `initialize` 之前保持沉默的 legacy server。

把每個探測錯誤都當成 legacy 是危險的。格式錯誤的現代請求、過載的 server、已死的 process，以及舊 server，都可能產生相同的 timeout 或連線關閉。這些訊號是含糊的。Client 必須結合操作者明確的意圖，和正向的 protocol 證據，才能選擇 legacy 世代。

## 核心概念 (The Concept)

### 一個 peer，而不是 protocol session

為每個 server process 或 endpoint 保存一筆 transport peer 紀錄：

- transport handle 或 send 函式；
- 選定的 protocol 世代與 version；
- 最近一次探索到的 server capabilities；
- 最近一次的確定性 tool 清單；
- 用於對應的 pending request id；
- transport 健康狀態。

這是 client 的簿記。它不是 protocol session 狀態。在現代 MCP 上，server 仍然會在每個請求上收到最新的 version 與 capabilities。

### 從頭建構每一個現代請求

```python
def modern_request(request_id, method, params, version, capabilities):
    return {
        "jsonrpc": "2.0",
        "id": request_id,
        "method": method,
        "params": {
            **params,
            "_meta": {
                "io.modelcontextprotocol/protocolVersion": version,
                "io.modelcontextprotocol/clientCapabilities": capabilities,
                "io.modelcontextprotocol/clientInfo": CLIENT_INFO,
            },
        },
    }
```

不要把 metadata 只附加在連線物件上一次，就假設它已經到了 wire 上。要對最終序列化後的請求蓋章並檢查。

### 現代 discovery

`server/discover` 回傳支援的 version、server capabilities、instructions、cache 提示，以及建議提供的 server identity。Client 選擇雙方都支援的最高現代 version。

對只支援現代版本的 client 來說，discovery 是選擇性的，但在 stdio 上建議使用。有些 legacy server 在初始化之前就接受操作，所以先送 `tools/list` 可能產生含糊的成功。`server/discover` 則創造出清楚的世代界線。

### Stdio 相容性探測

雙世代的 stdio client 會在任何其他請求之前，先送出帶有偏好現代 metadata 的 `server/discover`。結果有三類：

1. **DiscoverResult。** Server 是現代的。選出雙方都支援的 version，並以每請求 metadata 繼續。
2. **可辨識的現代錯誤。** Server 是現代的。遇到 `-32022` 時，從 `data.supported` 中挑選，並以新的 request id 重試。遇到 header 或 capability 錯誤時，修正請求。不要送 `initialize`。
3. **含糊的訊號。** 無法辨識的 JSON-RPC 錯誤、timeout、連線關閉或空回應，都不能辨識出世代。除非那個確切的 peer 被設定為 legacy 相容，否則一律 fail closed。

可辨識的現代 protocol 錯誤包括：

- `-32020` HeaderMismatch
- `-32021` MissingRequiredClientCapability
- `-32022` UnsupportedProtocolVersion

即使 peer 在 legacy allowlist 上，可辨識的現代錯誤仍然代表現代世代。一旦 server 證明它懂現代的錯誤詞彙，再送 `initialize` 就是降級。

不要把 `-32601` 當成正向的 legacy 證據。它只會讓明確列入 allowlist 的 peer 有資格接受一次 legacy 探測。timeout、連線關閉或空回應也適用同樣的規則。

### Allowlist 是操作者的意圖，不是證據

Legacy 相容性必須是某個固定 peer 設定的明確屬性：

```python
client.add_server("archive", archive_transport, allow_legacy=True)
```

把這個選擇綁定到設定好的 command 或 endpoint。不要使用萬用字元，讓任意的 server 自己選擇進入較弱的語意。沒有 `allow_legacy=True` 的 peer，在含糊的 discovery 結果之後就會失敗，而且永遠不會收到 `initialize`。

Allowlist 授予的是探測的許可。它不決定世代。Client 在 transport 強制執行的期限內送出一次 `initialize`，然後要求以下全部條件：

- 一個 JSON-RPC `2.0` 的回應，且 request id 相符；
- 恰好一個 `result`，沒有 `error`；
- `protocolVersion` 屬於 client 設定的 legacy revision 集合；
- 值為物件的 `capabilities` 欄位；
- 一個 `serverInfo` 物件，其中 `name` 與 `version` 欄位都是非空字串。

Timeout、連線關閉、錯誤回應、格式錯誤的結果、id 不符或不支援的 revision，都會 fail closed。只有結構有效的正向結果，才會選定 legacy 世代。程式碼會把 `legacy_probe_timeout_ms` 傳給 transport adapter；真正的 stdio 或 HTTP adapter 必須強制執行這個期限，而不是只記錄它。

為 transport peer 快取選定的世代。不要在每次 call 之前都重新探測。

### Legacy 是一條相容性分支

一旦有界限的探測回傳有效的正向 legacy 證據，client 就完全依照該 revision 的定義來使用選定的 legacy version：

1. 驗證回應 envelope 與對應 id。
2. 驗證協商出來的 revision 在設定的 legacy 集合中。
3. 記錄驗證過的 capabilities 與 server identity。
4. 只有在所有檢查都通過之後，才送出 `notifications/initialized`。
5. 在該 transport 的生命週期內，使用 legacy 的請求形狀。

這條分支是為了與已知的 peer 互通而存在。它不是新 server 或新請求的預設設計。如果 transport 重啟或 endpoint 改變，要丟掉 peer 世代快取，重新協商。

### 探索與快取 tool

對每個作用中的 peer 呼叫 `tools/list`。現代的結果包含 `resultType`、`ttlMs` 與 `cacheScope`。在正確的授權 context 內，遵守新鮮度提示。到期或收到已訂閱的清單變更事件後，重新取得。

Client 必須把 legacy server 缺少的 `resultType` 視為 `"complete"`。不要要求較早協商出的世代的回應帶有現代的 cache 欄位。

Server 應該回傳確定性的排序。Client 也應該在合併前先排序，讓本地 registry 的順序不依賴 process 的啟動時序。

### 避免衝突的 namespace 合併

兩個 server 可能都公開 `search`。請選擇一個明確宣告的策略：

1. **衝突時加前綴。** 保留第一個 canonical 名稱，把後來衝突的公開為 `<server>/<tool>`。
2. **衝突時拒絕。** 不載入重複的項目，並顯示清楚的設定錯誤。
3. **默默覆寫。** 絕不使用。它會隱藏由模型選擇的動作會送到哪個 server。

同時保存 canonical 名稱和本地名稱。模型看到的是 canonical 名稱。送出的 `tools/call` 則使用擁有它的 server 所宣告的本地名稱。

### 路由 call

路由是純粹的查表：

```text
canonical tool name
  -> peer name + local tool name
  -> new JSON-RPC request id
  -> modern request metadata or explicit legacy shape
  -> matching response id
```

當擁有它的 transport 不可用時，不要送出 call。重新連線或重啟 transport，然後重新執行 discovery 和 `tools/list`。在壞掉的 transport 上遺失的現代進行中請求，只要該操作的安全政策允許，就可以用新的 JSON-RPC id 重試。

### Notification 與訂閱

現代的清單與 resource 變更，只會出現在 client 開啟的 `subscriptions/listen` 串流上。Client 送出 notification 過濾條件，等待 `notifications/subscriptions/acknowledged`，並用 notification metadata 中的 listen request id 來對應事件。

斷線時，開啟新的 listen 請求，並重新取得相關的清單或 resource。現代串流不會用 `Last-Event-ID` 恢復。

### 沒有 server 發起的請求

現代 server 不會為了 sampling、elicitation 或 roots，用獨立的 JSON-RPC 請求呼叫 client。它們會回傳 `input_required`，client 則在滿足內嵌的 input request 之後，重試原本的請求。

在滿足 input 的期間，不要阻塞 peer 的 response reader。保留對應關係，並為重試建立新的 JSON-RPC id。

```figure
tp-client-merge
```

## 實際使用 (Use It)

`code/main.py` 使用 in-process 的 peer 函式，讓 protocol 的決策一目了然。它連接兩個現代 peer 和一個刻意列入 allowlist 的 legacy peer，然後合併並路由它們的 tool。Transport 的 callable 會收到 timeout 預算，所以相容性分支無法藏住沒有界限的探測。

```bash
cd code
python3 main.py
python3 -m unittest discover tests -v
```

這些測試證明了一般 demo 會漏掉的邊界：

- 現代請求會重複 metadata；
- `-32022` 會重試現代 discovery，而不會初始化；
- 可辨識的現代錯誤永遠不會降級，即使是 allowlist 上的 peer；
- 沒有 allowlist 時，timeout、連線關閉、空回應和無法辨識的錯誤，都不會觸發 `initialize`；
- 列入 allowlist 的 peer 只有在收到有效、受支援的 `initialize` 結果之後，才會變成 legacy；
- 格式錯誤與不支援的 legacy 結果，會讓 peer 維持不可用；
- 成功選定的世代，會在 transport 的生命週期內被快取。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-client-harness.md`。它會產生現代請求蓋章、stdio 世代協商、確定性 namespace 合併、路由，以及 fail-closed 的 legacy 相容性分支的雛形。

## 練習 (Exercises)

1. 讓假 server 回傳 `-32022`，且沒有雙方都支援的 version。確認 client 會失敗，而不是送出 `initialize`。
2. 把一個假的 legacy server 列入 allowlist，讓它有界限的 `initialize` 探測 timeout，並證明 peer 維持 `unknown` 且不可用。
3. 為兩個授權 context 新增 `cacheScope: "private"` 的 tool 清單。確認 client 絕不會把一個 context 快取的結果分享給另一個。
4. 把衝突策略改成拒絕，讓啟動失敗，且錯誤訊息中包含兩個 peer 的名稱。
5. 新增一個有限的 `subscriptions/listen` 模擬器。串流中斷時，用新的 request id 重新 listen，並重新取得 tool。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Peer | Client 端針對一個 server transport 及其探索到的資料所做的紀錄 |
| Protocol era | 現代的每請求 metadata，或 legacy 的初始化語意 |
| Discovery probe | 用來辨識 stdio 世代的初始 `server/discover` |
| Recognized modern error | 證明是現代行為並禁止 legacy fallback 的錯誤 |
| Legacy allowlist | 操作者設定，允許對固定 peer 做一次有界限的相容性探測 |
| Positive legacy evidence | 針對明確支援的 legacy revision、有效且對應的 `initialize` 結果 |
| Merged namespace | 所有作用中 peer 的 canonical tool 名稱 |
| Collision policy | 重複 tool 名稱的加前綴或拒絕規則 |
| Era cache | 為單一 transport peer 儲存的現代或 legacy 行為選擇 |
| Transport recovery | 重啟或重新連線、重新 discovery、重新 list，並以新 id 安全地重試 |

## 延伸閱讀 (Further Reading)

- [MCP Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28/)
- [MCP Server Discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
- [MCP stdio Transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/stdio)
- [MCP Versioning](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
