# MCP 安全：被汙染的 Metadata、Routing 與 MRTR 狀態 (MCP Security: Poisoned Metadata, Routing, and MRTR State)

> 無狀態不代表不需要信任。它的意思是，每個請求都攤開 server 與 gateway 需要的證據，讓它們能獨立驗證該次 call。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 08 (MCP client)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 把 tool description、annotation、client 資訊與 server 資訊，都視為不受信任的資料。
- 偵測 metadata 汙染、descriptor 變更，以及跨 server 的名稱衝突。
- 驗證 2026-07-28 的請求 metadata 與 Streamable HTTP 路由 header。
- 保護 MRTR 的 `requestState` 不被竄改，並把確認綁定到確切的參數。
- 對 principal 套用授權與 rate limit，而不是對已被移除的 protocol session。

## 問題 (The Problem)

模型讀取 tool description 來決定要呼叫什麼。Router 讀取 tool name 來決定把請求送到哪裡。使用者讀取標籤來決定要核准什麼。一個惡意的 descriptor 可以同時攻擊這三者。

官方的 MCP 安全指引說得很直接：除非來自受信任的 server，否則 description 與 annotation 都應視為不受信任。即使是受信任的 server，部署的信任也可能改變。Server 更新、被入侵的套件、registry 的錯誤，或 gateway 的合併，都可能改變模型看到的內容。

目前的 protocol 也改變了安全邊界。在 2026-07-28 中，沒有核心 handshake，也沒有 transport session。只用 `Mcp-Session-Id` 作為核准、rate limit 或稽核歷史的 key，並不是目前的設計。

## 核心概念 (The Concept)

### 值得檢查的七個攻擊面 (Seven attack surfaces worth checking)

使用具體的清單，而不是「要小心」這種模糊的指示。

1. **Metadata 汙染。** Description 包含與宣告的 tool 行為無關的指令。
2. **Descriptor rug pull。** 先前已核准的 name、description、schema 或 annotation 被改變。
3. **跨 server 遮蔽（shadowing）。** 兩個 backend 公開相同、沒有限定的 tool 名稱，而路由默默選了其中一個。
4. **Header 與 body 混淆。** `Mcp-Method` 或 `Mcp-Name` 與 JSON-RPC 請求不一致。
5. **Capability 提權。** Peer 聲稱擁有某個 extension 或 client 功能，而 server 把該宣告誤認為授權。
6. **MRTR 狀態竄改。** Client 改變 `requestState`、回答不同的問題，或把確認重用在不同的參數上。
7. **供應鏈 identity 混淆。** 把熟悉的顯示名稱，當作發布者或 server identity 的證明。

這些攻擊面會互相重疊。Hash pinning 有助於偵測 descriptor 變更，但無法證明第一個 descriptor 是安全的。靜態掃描能抓到明顯的片語，卻抓不到微妙的指令。Namespace 能避免一類衝突，卻擋不住惡意但有 namespace 的 server。請把各種控制疊加起來。

### 目前的請求 envelope 是證據，不是 identity

每個 2026-07-28 請求都包含：

```json
{
  "_meta": {
    "io.modelcontextprotocol/protocolVersion": "2026-07-28",
    "io.modelcontextprotocol/clientCapabilities": {
      "elicitation": {"form": {}}
    },
    "io.modelcontextprotocol/clientInfo": {
      "name": "security-lab",
      "version": "1.0.0"
    }
  }
}
```

在每個請求上驗證 version 與 capability 的形狀。用 capabilities 來選擇相容的 response 形狀。不要把 `clientInfo` 當作已驗證的 principal。它是自行申報的。

同樣的警告也適用於結果 metadata 中的 `io.modelcontextprotocol/serverInfo`。它對 log 與除錯有用，但不是憑證、registry 證明或授權決定。

### 在政策之前先驗證路由 (Validate routing before policy)

對於 `tools/call`，Streamable HTTP 包含：

```text
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.export
```

Header 的 method 必須等於 body 的 method。Header 的 name 必須等於 `params.name`。在選擇 backend、套用 RBAC 或消耗 rate-limit token 之前，就用 `-32020` 拒絕不一致。

這個順序封住了一種常見的歧義：一個元件授權 body，另一個元件卻依 header 路由。

Wire 驗證遵循一個確切的順序。驗證 JSON-RPC 與 metadata 的型別，把 header 的值與 body 比對，然後才檢查一致的 version 是否被支援。Header 不一致時，回傳 HTTP 400 與 `-32020`。如果 header 與 body 對一個不支援的 version 看法一致，回傳 HTTP 400 與 `-32022`，且 `data` 精確為 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知的 method 回傳 HTTP 404 與 `-32601`。

當合約需要結構化的復原資訊時，每個 error 物件都會包含選擇性的 `data`。Notification 沒有 `id`，所以永遠不會收到 JSON-RPC 的成功或錯誤 response。被接受的 HTTP notification 回傳 202 與空的 body。

### 固定整個 descriptor (Pin the whole descriptor)

只對 description 做 hash，會漏掉 schema 與 annotation 的變更。請將使用者核准過的 descriptor 欄位正規化後再做 hash：

```python
normalized = json.dumps(tool, sort_keys=True, separators=(",", ":"))
digest = hashlib.sha256(normalized.encode()).hexdigest()
```

把 digest 存在帶有限定名稱的 key（例如 `notes.export`）下，並連同發布者證據與核准時間一起存放，這些在這個簡化範例之外。

每次重新整理時：

- 未知的 key：隔離，直到審查完成。
- 相同的 key、不同的 digest：視為 rug pull 加以隔離，直到重新核准。
- 重複的未限定名稱：要求使用確定性的 namespace。
- 掃描器命中：阻擋，並審查完整的 descriptor。

Hash 相等證明的是穩定，不是安全。被汙染的 descriptor，即使被完美地 pin 住，仍然是被汙染的。

### 靜態掃描是絆線

簡單的模式可以標出角色標籤、指令覆寫、隱匿、機密存取和被遮蔽的網路目的地。它們的成本低到足以在安裝時和 CI 中執行。

它們不是語意上的證明。安全的 description 可能在合法的警告中含有被標記的片語。惡意的 description 則可以避開所有片語。請把掃描器輸出視為審查證據，而不是自動的清白分數。

### 合併之前先加 namespace (Namespace before merging)

假設兩個 server 都公開 `search`。絕不要讓 discovery 的順序決定誰勝出。

```text
notes.search
issues.search
```

限定名稱是 gateway 對外的公開名稱。另外記錄 backend 的對應。穩定的名稱，讓核准、稽核、hash pin 與 `Mcp-Name` 路由，都指向同一個物件。

### Capability 是相容性宣告

每請求的 `clientCapabilities` 告訴 server，client 能處理哪些 protocol 功能。它不會授予 client 對 tool、資料或動作的存取權。

授權仍然來自已驗證的 principal 與 resource 政策。順序是：

1. 驗證 transport 的憑證。
2. 驗證 version、header 與請求的形狀。
3. 檢查 capability 相容性。
4. 對 principal、tool、resource 與參數做授權。
5. 執行，或請求使用者輸入。

### 保護無狀態的 MRTR 確認 (Protect stateless MRTR confirmation)

具 consequential 性質的 tool 可能需要使用者確認。目前的 MCP 使用 Multi Round-Trip Requests，而不是 server 到 client 的 callback。

第一個 response：

```json
{
  "resultType": "input_required",
  "inputRequests": {
    "confirm": {
      "method": "elicitation/create",
      "params": {
        "mode": "form",
        "message": "Export notes to archive?",
        "requestedSchema": {
          "type": "object",
          "properties": {
            "confirm": {"type": "boolean"}
          },
          "required": ["confirm"]
        }
      }
    }
  },
  "requestState": "opaque-integrity-protected-value"
}
```

Client 取得輸入後，用新的 JSON-RPC id 重試原本的 method：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes.export",
    "arguments": {"query": "private", "destination": "archive"},
    "requestState": "opaque-integrity-protected-value",
    "inputResponses": {
      "confirm": {
        "action": "accept",
        "content": {"confirm": true}
      }
    },
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

每個 `inputRequests` 的值，都是帶有 `method` 與 `params` 的完整內嵌請求。它的 key 必須與 `inputResponses` 中對應的項目相符。Form elicitation 使用物件根的 `requestedSchema`，而且 client 必須在 server 請求之前，就已宣告 form elicitation 的 capability。

目前的 capability 有兩種有效的 form 宣告。`{"elicitation":{}}` 隱含支援 form elicitation，而 `{"elicitation":{"form":{}}}` 則是明確聲明。僅宣告 URL 的 `{"elicitation":{"url":{}}}` 不支援 form 請求。Server 會回傳 HTTP 400 與 `-32021`，且 `data.requiredCapabilities` 等於 `{"elicitation":{"form":{}}}`。

把 `requestState` 視為惡意輸入。對它簽章或加密、驗證它，並把它綁定到 method、tool、確切的參數、用途、過期時間、principal，以及在 replay 重要時的一次性 nonce。這堂課的程式碼使用 HMAC 與確切的參數比對，讓邊界清楚可見。

Nonce 帳本不能只存在一個 gateway 物件裡。可執行的模型注入一個有界限、依 TTL 清理的 replay store，可由多個 gateway 實例共用。它的 atomic claim 就是執行邊界：只有經過驗證的接受，或明確的終結 decline，才會消耗 state。格式錯誤的 response 或 `cancel` 不會執行任何事，並在過期前維持可重試。Production 的 fleet 需要在共用的持久化儲存中，使用相同的條件式 claim。

不要把隱藏的確認 context 存在 protocol session 裡。任何 server 實例都應該能驗證重試。

### 高風險 call 的 Rule of Two

沿著三條軸線為 call 分類：

- 它會消耗不受信任的輸入。
- 它可以存取敏感資料。
- 它會造成 consequential 的外部動作。

單一個自動步驟不應該同時結合這三者。把它拆開、降低權限，或透過 MRTR 請求明確的使用者輸入。這是一種設計啟發法，不是 protocol 的 capability。

### 執行之前先降低權限 (Reduce authority before execution)

單靠無狀態並不安全。它移除了隱藏的 protocol 歷史，但自成一體的請求仍然可能要求一個權限過大的 handler 洩漏資料，或做出無法回復的變更。安全來自在每個邊界降低權限：

1. **有型別的動詞。** 公開一個有界限的操作，例如 `archive_note`，而不是能表達不相關權力的通用 `run` 或 `request` tool。
2. **經過驗證的參數。** 在可行的地方使用封閉的 schema，拒絕未知欄位，只正規化識別碼一次，限制大小，並在政策評估之前，驗證目的地、tenant 與 resource 擁有權。
3. **當前的授權。** 把已驗證的 principal，綁定到確切的動詞、resource、環境與正規化後的參數。Tool annotation 與 client capability 不授予這項權限。
4. **綁定動作的核准。** 對 consequential 的 call，把核准綁定到有型別動詞與正規化參數的 digest，加上 principal、過期時間與一次性政策。任何欄位改變，都需要新的決定。
5. **一等公民的拒絕。** 把 deny、核准過期、使用者 decline 與不安全的目的地，建模為不執行任何副作用的一般結果。不要把拒絕轉成較弱的 fallback tool。
6. **經過遮蔽的稽核證據。** 記錄誰提出請求、使用了哪個已接納的 descriptor 與政策版本、授權了哪個正規化的目標、決定允許或拒絕的原因，以及執行是否已開始。儲存 digest 或遮蔽後的值，而不是機密。

每個步驟都會縮小下一個元件能做的事。最後的 handler 應該收到已驗證的領域指令，而不是原始的模型文字加上廣泛的憑證。在 MRTR 重試、task 更新或 gateway 轉發的 call 上，要重複整條鏈。先前的核准，不會讓之後的請求變成受信任的 session 流量。

### 目前與 legacy 的互動路徑

Roots、Sampling 和 Logging 對新的 2026-07-28 實作而言已被棄用。Gateway 可以保留較舊的 request-channel 程式碼，但只能作為受 version 管控的相容性路徑。

不要圍繞每個 session 的 sampling limiter 來建構新的防禦。請對已驗證的 principal、issuer、resource、tool 與時間窗口套用配額。對於目前的互動式工作，請檢查 MRTR 的 input request 與 response。

### 無狀態 transport 檢查

- 在單一 POST endpoint 接受現代的 MCP 訊息。
- 對現代的 GET 與 DELETE 回傳 405。
- 不要產生或依賴 `Mcp-Session-Id`。
- 把 legacy 的 session 與 replay header 當作權限輸入時要忽略。
- 對該 POST 回傳 JSON 或請求範圍的 SSE。
- 只對選擇加入的長壽命變更 notification 使用 `subscriptions/listen`。

```figure
tp-tool-poisoning
```

## 動手做 (Build It)

`code/main.py` 實作一個小型的 in-process 安全 gateway 模型。它會正規化並 pin 完整的 tool descriptor、回報 metadata 汙染與 shadowing、驗證現代的請求 envelope 與路由值，並執行帶有簽章 `requestState` 與注入共用 replay store 的兩回合確認匯出。

這個模型從 HTTP adapter 已解析完 JSON body 與路由 header 之後開始。它不驗證 `Content-Type` 或 `Accept`。請把同一個 dispatcher 接到課程 09 的完整 Streamable HTTP adapter，它要求 `Content-Type: application/json`，以及同時包含 `application/json` 和 `text/event-stream` 的 `Accept` 值。

執行它：

```bash
cd phases/13-tools-and-protocols/15-mcp-security-tool-poisoning
python3 code/main.py
python3 -m unittest discover code/tests -v
```

範例刻意修改了一個 descriptor。掃描器與 digest 比對各自產生獨立的發現。接著匯出示範了 `input_required` response 與無狀態的重試。

## 實際使用 (Use It)

把 `SAFE_TOOLS` 換成你自己核准的 server 的正規化快照。不要讓憑證和機密進入快照。在更新 digest 之前，審查每個新增或變更的 descriptor。

在 gateway，於 discovery 期間以及 dispatch 之前，各執行一次相同的檢查。Cache 可以減少 discovery 的工作量，但快取的核准在 descriptor 改變時，必須過期或失效。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-mcp-threat-model.md`。它會針對 metadata、路由、capability、授權、MRTR、快取、registry 與相容性邊界，產出符合目前 protocol 的威脅模型。

## 練習 (Exercises)

1. 把已驗證的 principal 與當前授權決定，綁定到密封的 MRTR state，然後拒絕在不同 principal 下的重試。
2. 把記憶體內的 replay store 換成持久化的條件式 insert，並證明兩個 process 無法都 claim 同一個 nonce。
3. 在 replay claim 之後、模擬匯出之前注入一個失敗。定義並測試讓復原安全的 transaction 或冪等性規則。
4. 在不改變 description 的情況下，改變某個 tool 的 `inputSchema`。確認整個 descriptor 的 pin 能抓到它。
5. 新增一項政策，當 `tools/list` 隨 principal 不同而不同時，拒絕 public 快取。
6. 為 gateway 後方較舊的 server 建模。把所有 handshake 與 session 行為，放在明確的 `2025-11-25` 相容性分支之後。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Metadata poisoning | 嵌入 tool descriptor 的指令或欺騙性聲明 |
| Rug pull | 對先前已核准 descriptor 的變更 |
| Tool shadowing | 重複的未限定名稱所造成的路由歧義 |
| Header mismatch | 路由 header 與 JSON-RPC body 不一致，錯誤 `-32020` |
| Hash pin | 完整核准 descriptor 的 digest |
| MRTR | 用於 server 請求輸入的無狀態 response 與重試模式 |
| `requestState` | 必須被視為不受信任輸入的不透明往返值 |
| Capability declaration | protocol 相容性的聲明，不是授權 |
| Implicit form support | 空的 `elicitation` capability 物件，等同於支援 form |
| Qualified tool name | 穩定的 gateway 名稱，例如 `notes.search` |

## 延伸閱讀 (Further Reading)

- [MCP security and trust guidance](https://modelcontextprotocol.io/specification/2026-07-28#security-and-trust--safety)
- [Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [Streamable HTTP transport](https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http)
- [Deprecated features](https://modelcontextprotocol.io/specification/2026-07-28/deprecated)
