# MCP Conformance 工程：版本、證據與營運 (MCP Conformance Engineering: Versioning, Evidence, and Operations)

> Server 不會因為它的順利路徑在某個 SDK 上跑通，就算符合規範（conformant）。Conformance 存在於 wire 上、版本邊界上、穿過中介者的路徑上，以及 rollback 期間。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 17 (gateways), Phase 13 · 30 (registry admission)
**Time:** ~100 分鐘

## 學習目標 (Learning Objectives)

- 把規範性的 MCP 規則轉成 golden 與 negative 的 wire transcript。
- 讓嚴格的 `2026-07-28` 行為，與有界限的 legacy fallback 保持分開。
- 區分可加性的未知欄位，與無效的未知 `resultType`。
- 將原始的 JSON-RPC 證據，與經過 SDK 正規化的視圖做比較。
- 透過真實的 proxy 邊界，證明 header 與 body 的完整性。
- 用經過遮蔽的 transcript、健康與 rollback 證據，作為發布的閘門。

## 問題 (The Problem)

你的 client 透過 SDK 呼叫 `tools/list`，並取得 tool。整合測試通過了。

這個結果留下了重要的問題沒有回答：

- 請求是否帶有現代的每請求 protocol metadata？
- `MCP-Protocol-Version`、`Mcp-Method` 與 `Mcp-Name` 是否與 JSON-RPC body 相符？
- Response 在 wire 上是否包含有效的 `resultType`，還是 SDK 自己合成了一個？
- Client 會保留未來新增的欄位嗎？
- 被辨識的現代錯誤會不會意外觸發 legacy handshake？
- Proxy 是否保留了 origin 的狀態與 JSON-RPC 錯誤？
- Notification 的序列化器是否送出了被禁止的 response？
- 營運人員能不能在不儲存機密的情況下，證明為什麼某次發布被升級或 rollback？

Conformance 是一組可觀察的不變量。請在 production 流量不得不發現它們之前，先建構一個擷取這些不變量的 harness。

```figure
mcp-conformance-operations
```

## 從版本世代開始 (Start With Version Eras)

MCP `2026-07-28` 使用自成一體的每請求 metadata。現代請求帶有 `params._meta.io.modelcontextprotocol/protocolVersion` 與 `params._meta.io.modelcontextprotocol/clientCapabilities`。確切的 namespace key 很重要；單純的 `protocolVersion` 或 `clientCapabilities` 別名是格式錯誤的。當 HTTP 邊界上出現鏡射的路由 header 時，它們的值必須與 JSON-RPC body 一致。現代的成功結果帶有 `resultType`。

到 `2025-11-25` 為止的版本，使用較早的初始化世代。沒有 `resultType` 的 legacy 結果，只有在 client 已選定那個較早的世代之後，才被解讀為 complete。

不要建立一個同時接受兩種形狀的寬鬆 validator。使用兩條分支：

| 分支 | 進入的證據 | 缺少 `resultType` | 初始化 |
|---|---|---|---|
| 現代 | 成功的 `server/discover` 或被辨識的現代 response | 無效 | 不是預設路徑 |
| Legacy | 設定好的 allowlist，加上在不確定的現代探測之後有效的 legacy `initialize` 結果 | 解讀為 complete | 該世代要求 |

這樣的分離，可以避免格式錯誤的現代 peer 因較弱的驗證而得到獎勵。

### 嚴格模式 (Strict mode)

嚴格模式要求現代行為的證明。成功的 `server/discover` 證明現代分支。被辨識的現代 JSON-RPC 錯誤也證明這點。修正請求，或停止。絕不要因為 server 回傳了 `-32020`、`-32021` 或 `-32022` 就降級。

### Fallback 模式

Fallback 模式會進行一次有界限的現代探測。Timeout、空的回覆、關閉的連線或無法辨識的 response，都是不確定的。它們並不證明 peer 是 legacy。只有明確設定或列入 allowlist 以做相容性的 endpoint，才可以接著收到有界限的 legacy 探測，而且 client 只有在驗證過該探測的 `initialize` 結果與協商出的 legacy revision 之後，才選擇 legacy 分支。

Fallback 不是「任何錯誤之後都嘗試 legacy」。被辨識的現代錯誤包含有用的修正資訊。在它之後降級，可能會隱藏 header 不一致、缺少 capability 宣告或不支援的 version。

這可以防止攻擊者、中斷或過濾型 proxy 透過丟掉現代 response，來強迫降級。把 endpoint 政策、不確定的現代觀察、確切的正向 legacy 證據與選定的世代，一起記錄。

把選定的世代記錄在每份 transcript 旁邊。沒有這項事實，缺少的欄位在某次測試執行中看起來可以接受，在另一次卻是無效的。

## 建構 transcript 語料庫 (Build a Transcript Corpus)

Transcript fixture 記錄的是跨越邊界的東西，而不只是 SDK 的呼叫：

```json
{
  "name": "golden-modern-list",
  "era": "modern",
  "headers": {
    "MCP-Protocol-Version": "2026-07-28",
    "Mcp-Method": "tools/list"
  },
  "request": {
    "jsonrpc": "2.0",
    "id": 1,
    "method": "tools/list",
    "params": {
      "_meta": {
        "io.modelcontextprotocol/protocolVersion": "2026-07-28",
        "io.modelcontextprotocol/clientCapabilities": {}
      }
    }
  },
  "responseStatus": 200,
  "responseBody": {
    "jsonrpc": "2.0",
    "id": 1,
    "result": {
      "resultType": "complete",
      "tools": []
    }
  }
}
```

保留兩類 fixture。

### Golden transcript

Golden transcript 證明被接受的行為：

- 現代 discovery 或 method 請求，metadata 與 header 相符
- 具備必要欄位的 complete 結果
- 當 method 可以請求更多輸入時的 `input_required` 結果
- 只有在對應的 capability 被公告之後才出現的 extension 結果
- 沒有 `resultType` 的 legacy 結果，但僅限於選定的 legacy 世代
- 沒有 JSON-RPC response 的 notification 處理

Golden transcript 要精確，而不是大。讓易變的 ID 與時間戳記保持確定性，或在比較之前將其正規化。

### Negative transcript

Negative transcript 證明拒絕的行為：

- header 與 body 不一致
- 缺少每請求的 capabilities
- 不支援但相符的 protocol version
- 缺少現代的 `resultType`
- 未知或未公告的 `resultType`
- response 的 `jsonrpc` 不是 `2.0`，或 ID 在值或 JSON 型別上不同
- 同時包含 `result` 與 `error`，或兩者皆無的 response
- 沒有整數 `code` 與字串 `message` 的錯誤
- 已知的 protocol 錯誤被對應到錯誤的 HTTP 狀態
- 為 notification 送出的 response
- 格式錯誤的 JSON-RPC envelope
- Proxy 把 protocol 錯誤折疊掉

對每個 negative 案例，斷言拒絕的邊界與穩定的錯誤代碼。「呼叫失敗了」太弱。Proxy 產生的 500 與 origin 的 `-32020` 看起來都像失敗，卻對營運人員說著完全不同的故事。

Header 不一致的 fixture 必須包含 server 實際的 HTTP 400 JSON-RPC response，帶有相符的 request ID 與錯誤代碼 `-32020`。只要本地 validator 觀察到 `HeaderMismatch`，就自動強制執行這點；不要把 response 驗證做成選擇性的 fixture 旗標。即使本地的拒絕代碼是正確的，帶有 HTTP 500 且沒有 body 的案例也會失敗。在自己的請求 validator 丟出例外之後就停止的 harness，只測試了自己，而不是 server 的 wire 行為。

官方的 MCP conformance 專案可作為外部套件與有版本的參考。也請保留你自己的本地 transcript。它們擷取你的 proxy、SDK、驗證身分、extension 與發布路徑，而這是通用套件無從得知的。

## Header 的值必須與 RPC body 相符 (Header Values Must Match the RPC Body)

在現代的 Streamable HTTP 中，中介者可以使用鏡射的 header 進行路由或強制執行政策。JSON-RPC body 仍是 protocol 的權威來源。不一致是完整性的失敗，而不是要你選擇其中一個值的提示。

依此順序驗證：

1. 解析並驗證 JSON-RPC envelope 與 metadata 的型別。
2. 比較 `MCP-Protocol-Version` 與 `params._meta.io.modelcontextprotocol/protocolVersion`。
3. 比較 `Mcp-Method` 與 `method`。
4. 當 method 有 routing name 時，比較 `Mcp-Name` 與對應的 body 值。
5. 確立相等之後，再決定相符的 version 與 capability 集合是否被支援。

這個順序區分了不一致的 `-32020` 與不支援 version 的 `-32022`。它也避免 gateway 授權了 header 的名稱，而 origin 卻執行了不同的 body 名稱。

HTTP 欄位名稱不分大小寫，而它們的值則分大小寫。查找之前先正規化 header 名稱，並拒絕互相衝突的重複項。對於不安全、非 ASCII 或前後有空白的 `Mcp-Name`，在與 body 比較之前，先解碼確切的 `=?base64?{Base64EncodedValue}?=` UTF-8 sentinel。對不完整的 sentinel、無效的 Base64、無效的 UTF-8 或原始的不安全值，以 `-32020` 拒絕。即使 body 包含相同的字元，原始的前後空白也是無效的，因為該值在 transport 之前需要 sentinel 編碼。

中介者可以在請求到達 MCP server 之前，就拒絕格式錯誤的 HTTP，所以它的失敗可能是沒有 JSON-RPC 的 HTTP 錯誤。請擷取拒絕是來自中介者還是 origin。當 origin MCP server 處理的是有效的 JSON-RPC 請求時，應該使用 protocol 的錯誤合約。

## 未知欄位不是未知結果 (Unknown Fields Are Not Unknown Results)

向前相容需要兩條不同的規則。

### 可加性的未知欄位

結果物件與 `_meta` map 可以新增欄位。Validator 應該依其角色保留或忽略可加性的欄位，除非該欄位違反保留的合約。範例在證據中保留完整的原始結果，並接受已知結果旁邊的 `futureHint`。

如果你是透明的 proxy，保留未知欄位通常比較安全。如果你是應用程式 client，忽略它可以是有效的。你的差異測試仍應揭露 SDK 省略了它，讓這個行為是刻意的。

### 未知的 `resultType`

`resultType` 是 discriminator。核心現代結果使用 `complete` 或 `input_required`。Extension 只有在其 capability 已公告時，才可以新增另一個值。例如 Tasks extension，可以在協商出的 capability context 中新增 `task`。

未知或未公告的 discriminator，無法被安全地視為 complete。Client 不知道自己正在丟棄什麼生命週期。拒絕它。

因此，同一個原始 response 可以同時包含可接受的未知欄位與不可接受的未知結果類型。兩種案例都要測試。

Discriminator 只是第一層。在它之後驗證 method 專屬的 payload。Complete 的 `tools/list` 結果需要一個 `tools` 陣列，其中的 descriptor 具有唯一、非空的名稱、有用的 description，以及物件根的 `inputSchema` 值。`task` 結果只對符合資格、具備 Tasks capability 的 `tools/call` 有效，並且需要 `taskId`、已知的狀態、建立與更新的時間戳記和 `ttlMs`，以及有效的選擇性輪詢間隔。Complete 的 `completion/complete` 結果需要 `completion` 物件，其中最多 100 個字串值、選擇性且不小於所回傳值數量的非負整數 `total`，以及選擇性的布林值 `hasMore`。拼寫正確的 `resultType` 無法讓格式錯誤的 payload 符合規範。

## Notification 不變量 (The Notification Invariant)

JSON-RPC notification 沒有 `id`。接收者不得送出 JSON-RPC 的成功或錯誤 response。

對於被接受的 HTTP notification 形狀，harness 預期 HTTP `202` 與空的 body。MCP `2026-07-28` 在 Streamable HTTP 上沒有定義核心的 client 到 server notification。範例使用帶 namespace 的課程 extension notification，僅用來測試單向序列化器的不變量。不要把它呈現為新的核心 method。

測試序列化器，而不只是 handler。Handler 可能回傳 `None`，而 middleware 卻把它包成 JSON 成功物件。請擷取最終的 egress 位元組。

## 加入 SDK 差異比較 (Add an SDK Differential)

SDK 經常把 wire 物件轉成方便使用的語言型別。這很有用，但經過正規化的物件無法證明實際收到了什麼。

對每個高風險的 fixture，擷取：

1. SDK 解碼之前的原始狀態、header 與 response body。
2. SDK 正規化後的回傳值或例外。
3. 針對選定世代的預期語意投影。
4. SDK 提升、合成、剝除或改變的欄位。

範例在比較應用程式 payload 時，允許 SDK 僅移除已知的 wire 簿記資訊，例如 `resultType`、`_meta`、`ttlMs` 和 `cacheScope`。它會回報被丟棄的 `futureHint`，因為那個未知的語意欄位消失了。

不要假設每個差異都是 SDK 的 bug。重點是讓轉換看得見。決定你的元件是應用程式 endpoint（可以忽略可加性欄位），還是透明中介者（應該保留它）。

針對你交付的每個 SDK 與版本執行差異比較。如果兩個 SDK 對同一份 transcript 的正規化結果不同，發布政策應該說明哪種行為可以接受，而不是事後挑選最方便的輸出。

## 擷取 proxy 證據 (Capture Proxy Evidence)

大多數 production 的 MCP 失敗，發生在不只一個 process 之間。記錄三個視圖：

| 視圖 | 最少的證據 |
|---|---|
| Ingress | 請求 header、JSON-RPC body、content type、已驗證的路由、接收時間 |
| Origin | 轉發的 header 與 body digest、origin 狀態、response header 與 body |
| Egress | client 看得到的狀態、header、body 與送出時間 |

範例偵測兩種常見的轉換：

- origin 的 HTTP 400 或 404 JSON-RPC 錯誤，變成通用的 proxy 500
- egress 的 JSON-RPC body 與 origin 的 body 不同

為 content type、`Accept`、壓縮、請求範圍的 SSE、cache header 與 trace 關聯，加入針對部署的斷言。在政策允許時，擷取 TLS 終止的兩邊。絕不要只為了證明路徑，而記錄憑證。

## 在證據離開記憶體之前先遮蔽 (Redact Before Evidence Leaves Memory)

遮蔽是 conformance 營運的一部分，不是事後的清理工作。在序列化、hash、log、測試成品或失敗上傳之前就套用。

範例會先對 key 名稱做大小寫摺疊並移除分隔符號再比對，然後遞迴地取代 `Authorization`、`Cookie`、`Set-Cookie`、`X-Api-Key`、`accessToken`、`clientSecret`、`registrationAccessToken`、`token`、`password`、`secret` 和 `api_key` 等 key 底下的值。正規化與 denylist 必須使用相同的形式，這樣 camelCase、連字號、底線與點分隔的變體，就無法彼此繞過對方的政策。Production 的 collector 應該加入 method 專屬的參數政策，因為像 `query` 這種無害的 key，仍然可能包含個人或受管制的資料。

對經過遮蔽的證據套件做 hash。只有在特定調查需要時，才把原始擷取保留在經核准的短期系統中。Digest 證明是哪個經過遮蔽的套件驅動了決定；它不會揭露被移除的值。

## 讓健康與 rollback 成為閘門的一部分 (Make Health and Rollback Part of the Gate)

Protocol conformance 是發布的必要條件，但並不充分。符合規範的候選版本仍然可能 timeout、洩漏記憶體，或壓垮相依元件。

在 rollout 之前定義健康視窗：

- 最少的樣本數
- 最大的錯誤率
- 最大的延遲百分位數
- 飽和度或資源限制
- 觀察期間
- 與已接納 baseline 的比較

也要在 rollout 之前定義 rollback 證據：

- 確切的先前版本
- Admission 證據 digest
- SHA-256 的 artifact 與 descriptor pin
- 目前的 Registry 狀態
- 目前的健康結果
- 路由還原程序
- 由受信任的 release-controller identity 針對這些確切欄位所做的 attestation

要求該 rollback 目標在升級之前就已被驗證且健康，而不只是在候選版本失敗之後。沒有可用復原路徑的成功發布，並未準備好上 production。

如果候選版本失敗，而 rollback 目標缺少該證據，就暫停流量，而不是用猜的。「Rollback 到原本有的東西」不是營運上的控制。

不要把就緒狀態簡化為 truthiness 檢查，例如非空的 version、`healthy: "yes"` 或任意的證據字串。範例要求確切的型別、active 的狀態、三個 SHA-256 digest、受信任的簽署者，以及針對完整 rollback payload 的有效 HMAC-SHA-256 attestation。它確定性的 demo key 是非機密的 fixture。在 production 中，請在發布邊界注入受保護的 key、KMS 驗證結果，或公鑰 attestation 驗證器。

發布閘門也拒絕空的 transcript、SDK 差異或 proxy 證據。每個來源都必須帶有有效的證據 digest。綠色的健康視窗無法填補從未被觀察過的邊界。

## 動手做 (Build It)

執行標準函式庫的 harness：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
```

Demo 恰好執行十五份 golden 與 negative 的 transcript，包括有效與格式錯誤的 completion 結果，將原始結果與 SDK 視圖做比較，檢查一個折疊了 origin 錯誤的 proxy，評估健康狀態，驗證 rollback 證據，並選出該目標。

預期的形狀：

```json
{
  "transcriptsPassed": 15,
  "transcriptsTotal": 15,
  "sdkDroppedFields": ["futureHint"],
  "proxyIssues": [
    "proxy collapsed a protocol error into HTTP 500",
    "proxy changed the origin JSON-RPC body"
  ],
  "releaseAction": "rollback",
  "evidenceDigest": "..."
}
```

依此順序閱讀 `code/main.py`：

1. `validate_request()` 強制執行各世代專屬的請求與 header 規則。
2. `validate_result()` 區分缺少的 legacy discriminator、有效的現代值、extension 與未知值。
3. `select_era()` 實作嚴格與有界限的 fallback 政策。
4. `run_transcript()` 評估 golden 與 negative 的 fixture。
5. `compare_sdk_view()` 揭露正規化的差異。
6. `inspect_proxy()` 比較 ingress、origin 與 egress 的證據。
7. `redact()` 在證據 hash 之前移除明顯的機密。
8. `rollback_evidence_ready()` 驗證確切的 pin 欄位與受信任的發布 attestation。
9. `ReleaseGate.evaluate()` 結合非空的 conformance、SDK、proxy、健康與 rollback 證據。

## 實際使用 (Use It)

在四個時間點執行 harness：

1. 每次實作變更時，搭配 in-process 的測試 adapter。
2. 對建置好的 client 與 server 二進位檔，透過真實的 transport 執行。
3. 在 staging 環境中，透過已部署的 proxy 或 gateway 執行。
4. 在 canary rollout 期間，搭配即時的健康與 rollback 證據。

跨層保持相同穩定的案例名稱。`negative-header-body-mismatch` 在單元、端對端、proxy 與 canary 報告中，應該代表同一個不變量。證據 digest 會因為邊界不同而不同；需求則不應改變。

將 fixture schema 存放在版本控制中。把經過遮蔽的執行證據存放在你的發布系統中。短期的原始擷取，只能存放在事件存取控制之下。

## 互動實驗 (Interactive Lab)

### 實驗 A：證明世代邊界

從 `code` 目錄開啟 Python：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations/code
python3 -q
```

執行：

```python
from main import *
validate_result({"tools": []}, "legacy")
validate_result({"tools": []}, "modern")
```

Legacy 呼叫推斷為 `complete`。現代呼叫則丟出 `ProtocolViolation`。現在測試 fallback：

```python
select_era({"kind": "timeout"}, "fallback")
select_era(
    {"kind": "timeout"},
    "fallback",
    legacy_allowed=True,
    legacy_evidence={"kind": "initialize_success", "protocolVersion": LEGACY_VERSION},
)
select_era({"kind": "jsonrpc_error", "code": -32021}, "fallback")
```

第一次 timeout 會 fail closed，因為沉默不是 legacy 的證據。第二次呼叫選擇 legacy，只因為設定允許，而且觀察到有效的 legacy 初始化結果。被辨識的缺少 capability 錯誤，則證明現代分支。

### 實驗 B：可加性欄位 vs discriminator

```python
validate_result({"resultType": "complete", "tools": [], "futureHint": True}, "modern")
validate_result({"resultType": "future_mode", "tools": []}, "modern")
```

第一個結果保留 `futureHint`。第二個被拒絕，因為生命週期 discriminator 是未知的。

### 實驗 C：檢查 SDK 的轉換

```python
compare_sdk_view(
    {"resultType": "complete", "tools": [], "futureHint": {"mode": "new"}},
    {"tools": []},
)
```

決定你的元件可以忽略 `futureHint`，還是必須轉發它。把這個選擇寫進發布政策。不要默默抹去差異。

### 實驗 D：修復 proxy

修改 demo 的交換，讓 egress 保留 origin 的狀態與 body。再次執行 `python3 main.py`。Proxy 的問題應該消失，但 SDK 差異仍會阻擋升級。接著把 `futureHint` 放進 SDK 視圖，觀察當每個證據來源都通過時，動作變成 `promote`。

## 實作練習 (Practice Lab)

為 harness 加入請求範圍 SSE 的 transcript。

需求：

- 擷取 response 狀態、content type、有序的 SSE 事件與串流終止。
- 證明每個 JSON-RPC 事件都有符合該世代的有效結果或錯誤。
- 為在轉發之前緩衝整個串流的 proxy，加入 negative 案例。
- 為 JSON-RPC id 與請求不同的 SSE 事件，加入 negative 案例。
- 在寫入證據之前，遮蔽事件資料。
- 把串流持續時間、首個事件的延遲與事件數量，納入健康視窗。
- 當串流失敗時，讓發布閘門只選擇有證據支持的 rollback 目標。

成功的意思是，同一個案例能直接執行，也能透過 proxy 執行，並產生一份指出究竟是哪個邊界改變了行為的報告。

## 交付的成品 (Shipped Artifact)

這堂課會交付 `outputs/skill-mcp-conformance-release-gate.md`。用它把 server、client、gateway 或 SDK 的變更，轉成有版本的 conformance 矩陣與發布決定。這份成品要求原始的 wire 證據、negative 案例、明確的世代選擇、SDK 差異比較、proxy 證明、遮蔽、健康門檻與 rollback 證據。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/31-mcp-conformance-versioning-and-operations
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- 每份包含的 golden 與 negative transcript 都達到預期的結果
- 現代請求要求確切的 namespace metadata key
- HTTP header 名稱不分大小寫比對，而編碼過的 `Mcp-Name` 值被精確解碼
- header 與 body 不一致會回傳現代的不一致代碼
- Response 的 version、ID、result 或 error 的互斥性、錯誤形狀與 HTTP 對應都被驗證
- method 專屬的 tool-list、task 與 completion payload 需求被強制執行
- 每個觀察到的 `HeaderMismatch`，都需要實際的 HTTP 400 JSON-RPC `-32020` response
- 原始的 `Mcp-Name` 空白被拒絕，而精確以 sentinel 編碼的空白可以來回轉換
- 缺少 `resultType` 只在選定的 legacy 世代中有效
- 可加性欄位在原始驗證中保留，而未知的結果類型會失敗
- extension 的結果類型需要其公告的 capability
- 被辨識的現代錯誤絕不會造成 legacy fallback
- notification 不產生 JSON-RPC response
- SDK 簿記資訊的移除與語意欄位的遺失被區分開來
- proxy 的錯誤折疊被偵測，而憑證在 camelCase 與分隔符號變體之間被遞迴遮蔽
- 升級需要非空的 transcript、SDK、proxy 與健康的營運證據
- 升級與 rollback 都需要經過驗證、已 pin、active 且健康的 rollback 目標

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 薄弱的測試回報什麼 | Harness 必須證明什麼 |
|---|---|---|
| SDK 合成了缺少的 discriminator | 「tools/list 通過了」 | 原始的現代結果缺少 `resultType`，因此無效 |
| Client 在 `-32021` 之後降級 | 「Legacy 重試成功了」 | 被辨識的現代錯誤禁止 fallback |
| 未知的結果類型被當成 complete | 「Response 解析了」 | 未公告的生命週期 discriminator 被拒絕 |
| Proxy 授權了一個 tool，而 origin 執行了另一個 | 「請求到達了 server」 | `Mcp-Name` 在每一跳都等於 body 的 routing name |
| Harness 在讀取 server response 之前就丟出例外 | 「Header 不一致測試通過了」 | HTTP 400 與 JSON-RPC `-32020` 的 response 被擷取並驗證 |
| Proxy 把 origin 的 400 變成通用的 500 | 「上游錯誤」 | Origin 與 egress 的狀態及 JSON-RPC body 都被保留 |
| Notification middleware 送出 `{result: null}` | 「Handler 回傳了 none」 | 最終的 egress body 是空的，且不存在 JSON-RPC response |
| SDK 剝除了可加性欄位 | 「型別化物件相符」 | 原始與正規化的視圖顯示確切被丟棄的欄位 |
| 失敗成品洩漏了 bearer token | 「Debug 套件已上傳」 | 遮蔽發生在 hash、log 或上傳之前 |
| 憑證 key 的風格繞過遮蔽 | 「Denylist 包含 api_key」 | CamelCase 與分隔符號變體共用同一個正規 denylist 形式 |
| Canary 沒有樣本卻看起來健康 | 「零個錯誤」 | 強制執行最少樣本數 |
| Rollback 選了未知的 build | 「已還原先前的部署」 | 目標 version、admission digest、pin、狀態與健康都存在 |

## 營運規則 (Operational Rule)

測試你送出的位元組、每個中介者轉發的位元組、每個 SDK 公開的語意，以及營運人員在壓力下會使用的證據。相容性是明確的分支。Rollback 是有證據支持的發布動作。兩者都不應該是寬鬆 parser 的意外副作用。

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 base protocol](https://modelcontextprotocol.io/specification/2026-07-28/basic)
- [MCP version negotiation](https://modelcontextprotocol.io/specification/2026-07-28/basic/versioning)
- [MCP Streamable HTTP](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Official MCP conformance project](https://github.com/modelcontextprotocol/conformance)
