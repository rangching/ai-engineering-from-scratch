# MCP Tool 合約與內容 (MCP Tool Contracts and Content)

> 只有當 discovery、參數、結果、分頁與 transport metadata 都同意同一份合約時，tool 才能安全地自動化。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13, Lessons 07, 09, and 10
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 用 JSON Schema 2020-12 定義 tool 的輸入與輸出。
- 驗證結構化結果，且不假設它們是 JSON 物件。
- 在 text、image、audio、resource link 與 embedded resource 之間做選擇。
- 在 tool 到達模型之前，拒絕不安全的 `x-mcp-header` 定義。
- 編碼參數 header 的值，並驗證 header 與 body 完全一致。
- 走訪 cursor 分頁，且不解讀 cursor 的值。
- 限制並授權 `completion/complete` 的建議。

## 問題 (The Problem)

呼叫 Python 函式很容易。透過 AI host 呼叫遠端的 capability，則是合約問題。

Server 發布 descriptor。Client 把 descriptor 變成模型 context 和使用者介面。模型產生參數。Gateway 可能根據鏡射的 header 路由請求。Server 執行 tool。接著 client 決定結果是否夠安全、夠有效，可以回傳給模型。

任何一個薄弱的邊界，都會破壞整條鏈。

考慮五種失敗：

- Descriptor 說結果是物件，但 server 回傳的是陣列。
- 當 `nextCursor` 是空字串時，client 停止了分頁。
- Token 參數被鏡射到 HTTP header，對中介者可見。
- Unicode 的路由值被當成原始 header 送出，之後 gateway 與 origin 解讀出不同的位元組。
- Completion endpoint 向無法存取 production 環境的呼叫者建議了 production 環境。

這些失敗沒有一個能靠更好的 prompt 修正。它們需要明確的 protocol 與應用程式合約。

## 合約管線 (The Contract Pipeline)

把每個 tool call 視為五道閘門：

1. **Discover。** 讀取確定性、有分頁的 tool 清單。
2. **Admit。** 驗證每個 descriptor 並套用本地安全政策。
3. **Invoke。** 驗證參數並建構 transport metadata。
4. **Execute。** 執行 handler 並正確地分類失敗。
5. **Consume。** 在模型使用之前，驗證 content block 與結構化輸出。

```figure
mcp-contract-pipeline
```

Host 擁有 admission 與 consumption 的閘門。Server 無法強迫 client 信任它的 annotation、schema 或輸出。

## JSON Schema 是 runtime 的邊界 (JSON Schema Is a Runtime Boundary)

在 MCP `2026-07-28` 中，`inputSchema` 和 `outputSchema` 使用 JSON Schema。當 `$schema` 不存在時，預設方言是 2020-12。

Input schema 必須是 schema 物件。沒有參數的 tool 仍然應該明確說明它接受什麼：

```json
{
  "type": "object",
  "additionalProperties": false
}
```

這比 `{ "type": "object" }` 更嚴格，後者接受任意的 property。

Output schema 是選擇性的。一旦 server 發布了它，每個 complete 的 tool 結果都承諾回傳符合的 `structuredContent`，包括帶有 `isError: true` 的結果。錯誤旗標分類的是執行結果；它不會免除已發布的輸出合約。Client 應該驗證結果，而不是信任 descriptor。

### Structured content 可以是任何 JSON 值

不要把 `structuredContent` 寫死為 dictionary。它可以是：

- 物件；
- 陣列；
- 字串；
- 數字；
- 布林值；
- `null`。

這個 tool 回傳陣列：

```json
{
  "name": "tag_catalog",
  "inputSchema": {
    "type": "object",
    "additionalProperties": false
  },
  "outputSchema": {
    "type": "array",
    "items": {"type": "string"}
  }
}
```

它成功的結果是有效的：

```json
{
  "resultType": "complete",
  "content": [
    {
      "type": "text",
      "text": "[\"contracts\", \"mcp\", \"stateless\"]"
    }
  ],
  "structuredContent": ["contracts", "mcp", "stateless"],
  "isError": false
}
```

為了相容性，結構化結果也應該在 text block 中包含序列化的 JSON。文字不是驗證的來源。`structuredContent` 才是。

### 小型 validator 仍然能說明邊界

這堂課刻意使用 JSON Schema 的子集，因為這樣可以留在 Python 標準函式庫之內。它檢查範例 tool 所用到的機制：

- object、array、string、integer、number、boolean 與 null 型別；
- required property；
- `additionalProperties: false`；
- array items；
- enum 值；
- 最小字串長度。

這不能取代完整的 production validator。可重複使用的教訓是驗證發生的位置：descriptor 在 discovery 之後、參數在執行之前、結構化結果在消耗之前。

## Content block 承擔不同的成本 (Content Blocks Carry Different Costs)

`content` 陣列可以組合數種內容類型。

| 類型 | 用途 | 主要邊界 |
|------|------------|---------------|
| `text` | 人類與模型可讀的摘要 | 把文字視為不受信任的輸出 |
| `image` | 以 base64 編碼的視覺證據 | 驗證 media type 與大小 |
| `audio` | 以 base64 編碼的語音或錄音輸出 | 驗證 media type 與時長限制 |
| `resource_link` | client 之後可以取得的 URI | 對之後的 resource 讀取重新授權 |
| `resource` | 直接內嵌在結果中的資料 | 立即強制執行 payload 與內容限制 |

Resource link 並不證明該 resource 出現在 `resources/list` 中。它是這次 tool call 回傳的參照。Client 在跟隨該 URI 時，仍然要套用它的 resource 政策。

內嵌的 resource 省去另一次往返，但會增加目前 response 的大小。對大型或各自獨立變動的成品，使用 link。對必須與結果一起原子性傳遞的小型證據，使用內嵌的 resource。

這堂課的 `evidence_bundle` 結果包含全部五種類型。Client 在接受結果之前，驗證每個 block。

## `x-mcp-header` 是 routing metadata

`inputSchema` 內的 property 可以宣告 `x-mcp-header`。透過 Streamable HTTP，client 會把該參數鏡射到 `Mcp-Param-{name}`。

```json
{
  "region": {
    "type": "string",
    "x-mcp-header": "Region"
  }
}
```

當 `region: "eu-west"` 時，transport 可以送出：

```http
Mcp-Param-Region: eu-west
```

這個 annotation 的存在，是為了讓 load balancer、gateway 或政策引擎能在不解析 JSON body 的情況下進行路由。它不是放憑證的地方。

Protocol 對這個 annotation 有以下限制：

- header 名稱非空，並遵循 HTTP field-name 的 token 語法；
- header 名稱不分大小寫地唯一；
- property 的型別是 string、integer 或 boolean；
- 不允許 `number`；
- annotation 只出現在 `inputSchema.properties` 的直接成員上；
- integer 值維持在 `-9007199254740991` 到 `9007199254740991` 之間。

位置規則是語法性的，而且 fail closed。走訪整個 schema 樹，而不只是你的 validator 剛好理解的 property。拒絕出現在巢狀物件的 `properties`、`oneOf` 分支、`items`、透過 `$ref` 到達的 definition，或任何 output schema 之下的 annotation。解析 reference 並不會讓被參照的節點變成直接的頂層 property。

這堂課加入了部署政策：拒絕鏡射像 `password`、`secret`、`token`、`api_key` 或 `authorization` 這類名稱的 descriptor。官方規格建議 server 作者不要鏡射敏感參數。Client 可以把這個建議變成硬性的 admission 規則。

稽核 header 的名稱，而不是它的值。範例程式碼會記錄 `Mcp-Param-Region`，同時讓 `eu-west` 不進入稽核事件。

### 在建構 HTTP header 之前先編碼值

參數值只有在它是由 `!` 到 `~` 的可見 ASCII 字元組成的非空字串，而且不像編碼 sentinel 時，才能以純文字傳送。其他一切都使用這個確切的形式：

```text
=?base64?{Base64UTF8}?=
```

`Base64UTF8` 是對確切 UTF-8 位元組做的標準 base64。不要先修剪、正規化或取代該值。要編碼 Unicode、空字串、空格、tab、控制字元、CR 或 LF、前後的空白，以及任何以 `=?base64?` 開頭的值。把看起來像 sentinel 的值再編碼一次，才能讓接收端還原出字面的原始文字，而不是把它當作 transport 語法解碼。

布林值轉譯成小寫的 `true` 或 `false`。Integer 以十進位轉譯，且必須維持在 JavaScript 安全 integer 範圍之內。超出該範圍的值會被拒絕，而不是被中介者四捨五入。

### Server 檢查鏡射的副本

產生 header 只是 client 的那一半。在 Streamable HTTP 的邊界，server 必須：

1. 不分大小寫地找出被辨識的 `Mcp-Param-*` 名稱；
2. 當存在時，解碼確切的 base64 sentinel 形式；
3. 把解碼後的文字與對應的 JSON body 參數精確比對；
4. 在 dispatch 之前，拒絕缺少、重複、非預期、格式錯誤或不一致的被辨識 header。

拒絕的方式是 HTTP `400` 搭配 JSON-RPC 錯誤代碼 `-32020`。Body 的值及其編碼後的 header 形式，都不應出現在稽核紀錄中。只記錄被辨識的 header 名稱與拒絕的類別。

`code/main.py` 直接模擬這道邊界。[Lesson 09](../../09-mcp-transports/) 涵蓋更廣的 Streamable HTTP 驗證順序，包括 method 與 protocol-version 的一致性。

## 分頁 cursor 是不透明的 (Pagination Cursors Are Opaque)

MCP 的 list 操作使用 cursor 分頁。由 server 選擇頁面大小與 cursor 格式。Client 只有一個決定：

```python
if result.get("nextCursor") is None:
    break
cursor = result["nextCursor"]
```

不要這樣寫：

```python
if not result.get("nextCursor"):
    break
```

空字串是有效的 cursor。用 truthiness 判斷會太早停止。

Client 不得解碼 cursor、遞增它、將它與先前的 cursor 比較順序，或推斷頁碼。Server 可以對 cursor 簽章、把它綁定到目錄版本，或把它對應到私有的狀態。那是 server 的實作細節。

範例 server 刻意在第一頁之後回傳 `""`。Client 必須在第二個請求中，原樣送出該值。它的 trace 是：

```text
<first request with no cursor>
<second request with cursor "">
```

無效的 cursor 會產生 JSON-RPC 的無效參數，代碼 `-32602`。

## Completion 是授權的介面 (Completion Is an Authorization Surface)

`completion/complete` 為 prompt 參數與 resource-template 參數提供建議。它對互動式表單很有用，但可能洩漏一般 list method 所保護的名稱。

Completion 請求會指名一個 reference 以及正在補全的參數：

```json
{
  "method": "completion/complete",
  "params": {
    "ref": {
      "type": "ref/prompt",
      "name": "deployment_review"
    },
    "argument": {
      "name": "environment",
      "value": "st"
    }
  }
}
```

結果最多回傳 100 個值，並可以回報 `total` 加上 `hasMore`。

套用與被參照的 prompt 或 resource 相同的授權邊界。範例中的 analyst 會收到 `development` 和 `staging`。只有 operator 能收到 `production`。

Production 的 completion 還需要：

- 輸入驗證；
- 感知呼叫者的過濾；
- client 端的請求 debounce；
- server 端的 rate limit；
- 有界限的結果數量；
- 不暴露敏感建議值的 log。

Completion 是輔助，不是繞過 discovery 的方法。

## 兩層錯誤 (Two Error Layers)

把 protocol 錯誤與 tool 執行錯誤分開。

當 MCP 請求無法被正確 dispatch 時，使用 JSON-RPC 錯誤：

- 未知的 tool 名稱；
- 格式錯誤的請求形狀；
- 缺少請求 metadata；
- 無效的 cursor。

當呼叫已到達 tool，而 tool 回報可處理的失敗時，使用帶有 `isError: true` 的 complete tool 結果：

- 報告來源不可用；
- 日期超出支援的範圍；
- 商業規則拒絕了所請求的操作。

模型通常可以修復 tool 的執行錯誤。它們無法修復違反自己輸出 schema 的 server。

如果 tool 宣告了 output schema，請在該 schema 內建模可處理的失敗。範例中的 `route_report` 失敗會回傳它所請求的 region，並帶有 `accepted: false`，同時附上人類可讀的錯誤文字與 `isError: true`。

## 動手做 (Build It)

`code/main.py` 用 Python 標準函式庫建構邊界的兩端。

Server 實作：

- 每請求的 MCP metadata 驗證；
- 帶有 tools 與 completions capability 的 `server/discover`；
- 確定性的 `tools/list` 分頁；
- 四個 tool descriptor，其中一個必須被拒絕；
- 陣列的結構化輸出；
- 目前所有的 content block 類型；
- Streamable HTTP 一致性閘門，會解碼被辨識的參數 header，並在不一致時回傳 HTTP `400` 加上 JSON-RPC `-32020`；
- 經過授權且有 rate limit 的 completion。

Client 實作：

- descriptor 的 admission；
- 完整樹狀的 `x-mcp-header` 位置驗證與敏感欄位政策；
- 精確的純可見 ASCII 或 base64 UTF-8 值編碼；
- 會跟隨空字串的不透明 cursor 迴圈；
- 參數與結果驗證；
- content-block 驗證；
- 包含名稱但不含值的 header 稽核事件。

刻意不安全的 descriptor 是教學資料。它證明一個被拒絕的 tool，不會阻止有效的 tool 載入。

## 實際使用 (Use It)

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/28-mcp-tool-contracts-and-content/code
python3 main.py
python3 -m unittest discover tests -v
```

Demo 會印出被接納的 tool、被拒絕的 descriptor、兩個分頁請求、結構化的陣列內容、content-block 類型、鏡射的 header 名稱、值是否需要編碼、HTTP 一致性狀態，以及依呼叫者過濾的 completion 值。

## 互動實驗 (Interactive Lab)

開啟 `code/main.py` 並找到 `TOOLS`。

1. 把 `tag_catalog.outputSchema.type` 從 `array` 改成 `object`。
2. 執行 demo。Client 應該拒絕回傳的陣列。
3. 還原 schema。
4. 讓第一頁的 `nextCursor` 保持為 `""`，然後讓最後一頁回傳 `nextCursor: None`，而不是省略該欄位。
5. 執行測試並比較 cursor 的 trace。
6. 在字串 property 上加入 `x-mcp-header: "Authorization"`。
7. 確認 descriptor 的 admission 在呼叫之前就拒絕它。
8. 嘗試包含 Unicode、換行、前後空格，以及字面文字 `=?base64?SGVsbG8=?=` 的 `region` 值。解碼每個送出的 header，並證明原始值完整保留。
9. 把 annotation 移到 `oneOf`、`items` 或 `$ref` definition 之下。確認每個 descriptor 都被拒絕，即使該分支從未被 demo 使用。
10. 移除被辨識的 header，或改變它解碼後的值。確認 HTTP 邊界回傳狀態 `400` 與 JSON-RPC 代碼 `-32020`。

重點不是記住 JSON 的形狀，而是看著每道閘門在擁有它的邊界上失敗。

## 實作練習 (Practice Lab)

為合約實驗擴充一個 `search_evidence` tool。

需求：

1. 它的 input schema 接受 `query`、`limit` 與安全的 `region` routing 欄位。
2. 它的 output schema 是由 `uri`、`title` 和 `score` 組成的物件陣列。
3. 結果包含相容性文字，以及每個項目的 resource link。
4. 參數會拒絕未知的 property。
5. `limit` 由應用程式驗證設上限。
6. 沒有某個 URI 存取權的呼叫者，絕不會透過 completion 或 tool 輸出看到該 URI。
7. 測試包含不符合的 score、無效的 header annotation，以及兩頁的清單。
8. Header 值的測試涵蓋可見 ASCII、Unicode、控制字元、空白、看起來像 sentinel 的文字，以及 JavaScript 安全 integer 的兩個邊界。
9. HTTP fixture 接受不分大小寫的 header 名稱，但會以狀態 `400` 與代碼 `-32020` 拒絕缺少或不一致的被辨識值。

## 交付的成品 (Shipped Artifact)

`outputs/skill-mcp-contract-reviewer.md` 是單一檔案、可重複使用的審查 skill。給它 tool descriptor、範例結果、分頁行為與 completion 政策。它會回傳 admission 決定、結果驗證計畫、header 政策，以及具體的失敗測試。

## 驗證 (Verify It)

當以下敘述都為真時，這堂課就完成了：

- `tools/list` 在重複呼叫時回傳相同的邏輯順序。
- 當 `nextCursor` 是 `""` 時，client 會發出第二個請求。
- 不安全的敏感 header descriptor 被排除，而其他 tool 仍可用。
- 陣列通過它的陣列 output schema。
- 物件無法通過同一個陣列 schema。
- 錯誤結果不能省略或違反已發布的 output schema。
- Text、image、audio、resource link 與 embedded resource 的 block 都能通過驗證。
- Header 稽核事件包含名稱，不含值。
- 純可見 ASCII 維持純文字；Unicode、控制字元、有填充、空的，以及看起來像 sentinel 的值，都透過精確的 base64 UTF-8 編碼來回轉換。
- 超出 JavaScript 安全範圍的鏡射 integer 會被拒絕。
- 位於 `oneOf`、`items`、巢狀物件、`$ref` definition 或 output schema 之下的 annotation，在 admission 時被拒絕。
- 不分大小寫的被辨識 header 名稱，只有在解碼後的值與 body 完全相符時才會通過；缺少或不一致的副本會產生 HTTP `400` 與 JSON-RPC `-32020`。
- Analyst 的 completion 絕不回傳 `production`。
- Tool 失敗使用 `isError: true`；格式錯誤的 protocol call 使用 JSON-RPC 的 `error`。

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 學習者看到什麼 | 正確的回應 |
|---------|-----------------------|------------------|
| Client 假設輸出是物件 | 有效的陣列失敗或被默默包起來 | 依已發布的 schema 驗證，不使用僅限物件的型別 |
| 空 cursor 被當成 false | 最後幾頁消失 | 只要 `nextCursor` 存在且非 null 就繼續 |
| 敏感的值被鏡射 | 機密出現在 proxy、WAF 或 trace 資料中 | 拒絕該 descriptor，並讓機密留在受保護的請求資料中 |
| 原始 Unicode 或空白被鏡射 | Gateway 與 origin 看法不一，或值被正規化 | 使用精確的 base64 UTF-8 sentinel 編碼，並在解碼後比對 |
| Annotation 藏在 schema 分支中 | Client 在 admission 時漏掉 routing metadata | 走訪整個 schema 樹，只允許直接的頂層 property |
| 大型 integer 被鏡射 | JavaScript 中介者把 routing 值四捨五入 | 拒絕超出 JavaScript 安全 integer 範圍的值 |
| Header 與 body 不一致 | Gateway 路由到一個目標，origin 卻執行另一個 | 在 dispatch 之前以 HTTP `400` 與 JSON-RPC `-32020` 拒絕 |
| Output schema 被忽略 | 下游程式碼消耗損壞的結構 | 在模型或應用程式使用之前驗證 |
| Resource link 被自動信任 | Client 跟隨了未授權的 URI | 對每次 resource 讀取重新授權 |
| Completion 共用全域建議 | 隱藏的 tenant 名稱洩漏 | 依呼叫者、reference 與授權過濾 |
| Tool annotation 被當成政策 | 破壞性操作繞過確認 | 在 annotation 之外強制執行授權與核准 |
| 一個格式錯誤的 tool 破壞 discovery | 整個 server 變得不可用 | 拒絕壞的 descriptor，並獨立地接納有效的 tool |

## 與 Capstone 的連結 (Capstone Connection)

Phase 13 的 capstone 需要能合併多個 server 的 tool 的 gateway。這堂課提供它的 admission 核心。

用這份成品為四項 capstone 證據評分：

- 確定性且完整的分頁 discovery；
- 在模型看到之前的 descriptor 驗證；
- 經過驗證的結構化輸出，加上有界限的 content block；
- 保留授權邊界的 completion 與 routing metadata。

不要只因為 `tools/call` 成功，就宣稱 gateway 相容。請擷取 descriptor、頁面 trace、被接納的 tool 集合、被拒絕的 tool 集合，以及一個經過驗證的結果。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| `inputSchema` | 定義 tool 可接受參數的 JSON Schema 物件 |
| `outputSchema` | 定義 `structuredContent` 的選擇性 JSON Schema |
| `structuredContent` | tool 結果所產生的任何 JSON 值 |
| Content block | 有型別的 text、image、audio、resource link 或 embedded resource |
| `x-mcp-header` | 將基本型別參數鏡射到 Streamable HTTP metadata 的 schema annotation |
| Opaque cursor | 由 server 簽發、client 不解讀其值的分頁 token |
| Completion reference | 正在補全其參數的 prompt 名稱或 resource URI/template |
| Admission | Client 決定公開或拒絕已探索到的 descriptor |

## 延伸閱讀 (Further Reading)

- [MCP Tools](https://modelcontextprotocol.io/specification/2026-07-28/server/tools)
- [MCP Completion](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/completion)
- [MCP Pagination](https://modelcontextprotocol.io/specification/2026-07-28/server/utilities/pagination)
- [MCP Streamable HTTP Parameter Headers](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http#custom-headers-from-tool-parameters)
