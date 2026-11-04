# MCP Model Input：Sampling 遷移與無狀態 MRTR (MCP Model Input: Sampling Migration and Stateless MRTR)

> MCP 2026-07-28 對新的設計棄用 Sampling，並移除 server 到 client 的請求通道。如果現有的 workflow 仍然需要 client 的模型，server 會回傳 `input_required` 結果，client 則帶著模型輸出重試原本的請求。推理迴圈在 protocol 層變得明確、有界限且無狀態。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 10 (resources and prompts)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說明為什麼 Sampling 在 MCP 2026-07-28 中被棄用，並為新的 server 選擇直接整合模型的預設做法。
- 實作相容性 workflow，透過 Multi Round-Trip Requests (MRTR) 承載 `sampling/createMessage`。
- 在每個請求的 `_meta` 物件中放入 protocol revision 與 client capabilities。
- 回傳 `resultType: "input_required"`，並用新的 JSON-RPC id 重試原本的 method。
- 對 `requestState` 做完整性保護，並將它綁定到 principal、method、參數與過期時間。
- 用 capability 檢查、核准、response 驗證與回合上限，為模型輔助的迴圈設定界限。

## 在 Protocol 之前的決定 (The Decision Before the Protocol)

像 `summarize_repo` 這樣的 tool 需要兩種工作：

1. 確定性的工作：列出檔案、讀取允許的檔案、驗證路徑，並組合內容。
2. 模型的工作：挑選有代表性的檔案，並綜合出摘要。

你現在有兩種有效的架構。

### 新 server：直接與模型 provider 整合

這是目前的預設做法。Server 擁有模型選擇、憑證、預算、重試和 observability。它對 MCP client 回傳一個一般的 `tools/call` 結果。

當 server 本來就是託管服務，或可預測的模型行為比使用 host 的模型更重要時，選擇這種做法。

### 現有的 Sampling workflow：遷移到 MRTR

Sampling 在棄用期間仍然存在。以 2026-07-28 為目標的 server，不能把即時的 `sampling/createMessage` 請求送回 client。它改為把該請求內嵌在 `InputRequiredResult` 裡。

只有在使用 client 的模型和憑證是真正的產品需求時，才選擇這條相容性路徑。請記錄移除計畫，因為新的實作不應該採用已棄用的 Sampling。

## 無狀態合約 (The Stateless Contract)

2026 年 7 月的 protocol 沒有 `initialize` 交換、沒有 `notifications/initialized`，也沒有 `Mcp-Session-Id`。每個請求都攜帶過去存在於 handshake 中的資訊：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}},
      "io.modelcontextprotocol/clientInfo": {
        "name": "lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Server 在每個請求上都驗證 revision。缺少或不是字串的 version 屬於無效參數，`-32602`。不支援的字串會回傳 `-32022`，並附上精確的資料 `{"supported":["2026-07-28"],"requested":"<client version>"}`。缺少 Sampling capability 則會回傳 `-32021`，且 `data.requiredCapabilities` 設為 `{"sampling":{}}`。

沒有 JSON-RPC `id` 的 envelope 是 notification。接收者可以處理它，但既不發出成功 response，也不發出錯誤 response。對於被接受的 notification，Streamable HTTP adapter 會回傳沒有 body 的 `202 Accepted`。

Server 也實作 `server/discover`，包含精確的 `supportedVersions` key、capabilities、`ttlMs` 和 `cacheScope`，讓 client 能在呼叫 tool 之前，先學習並快取 server 的合約。因為 discovery 公告了 `tools`，server 也實作強制要求的 `tools/list`。它確定性的 `summarize_repo` descriptor 包含有效的物件 `inputSchema`、`resultType: "complete"`、server identity metadata，以及 public 的 cache 提示。

每個成功的現代結果都有 discriminator：

- `resultType: "complete"` 表示操作已完成。
- `resultType: "input_required"` 表示 client 必須滿足內嵌的請求並重試。
- Extension 可以定義額外的結果類型。Tasks extension 在課程 13 中新增了 `"task"`。

## 一個 MRTR 回合 (One MRTR Round)

Server 在處理請求時無法呼叫 client。它改為回傳這個結果：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "pick_files": {
        "method": "sampling/createMessage",
        "params": {
          "messages": [
            {
              "role": "user",
              "content": {
                "type": "text",
                "text": "Choose three representative files and return a JSON array."
              }
            }
          ],
          "systemPrompt": "Return only the requested value.",
          "modelPreferences": {
            "costPriority": 0.8,
            "intelligencePriority": 0.2
          },
          "maxTokens": 400
        }
      }
    },
    "requestState": "opaque-integrity-protected-value"
  }
}
```

Client 驗證自己支援 Sampling，套用它的核准與模型政策，並取得模型 response。接著它用不同的 JSON-RPC id 送出新的請求：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "summarize_repo",
    "arguments": {"audience": "developer"},
    "inputResponses": {
      "pick_files": {
        "role": "assistant",
        "content": {
          "type": "text",
          "text": "[\"README.md\", \"server.py\", \"docs/intro.md\"]"
        },
        "model": "host-model",
        "stopReason": "endTurn"
      }
    },
    "requestState": "opaque-integrity-protected-value",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {"sampling": {}}
    }
  }
}
```

重試不是 protocol session 的延續。它是一個新的請求，重述原本的 method 與參數，只加上當前回合的 `inputResponses`，並逐位元組原樣回傳 `requestState`。

MRTR 只允許用在 `tools/call`、`prompts/get` 和 `resources/read`。Server 不得從不相關的 method 回傳 `input_required`。

## 多回合狀態 (Multi-Round State)

這堂課需要兩次模型呼叫：

1. `pick_files` 回傳 JSON 陣列。
2. `summary` 回傳最終的散文。

每次重試只攜帶該回合的 response。因此 server 把階段與驗證過的中間資料放進下一個 `requestState`。

把這個值視為受攻擊者控制。只對原始的階段名稱簽章是不夠的。請將狀態綁定到：

- 已驗證的 principal，而不是自行申報的 `clientInfo`；
- 發起的 method；
- 原始參數的 digest；
- 短暫的過期時間；
- 當前階段與驗證過的中間值。

不需要保密時使用 HMAC。當 client 不應讀取該狀態時，使用 authenticated encryption。遇到簽章錯誤、已過期的值、principal 改變或參數改變，用 `-32602` 拒絕。

Client 不得解析或修改 `requestState`。它唯一的工作，就是在重試時原樣回傳那個字串。

## 模型偏好只是提示 (Model Preferences Are Hints)

`costPriority`、`speedPriority` 和 `intelligencePriority` 是各自獨立的偏好。它們不是機率分布，也不需要加總為一。Client 可以忽略它們，因為模型政策由 client 擁有。

如果你維護 legacy 的 Sampling flow，請讓 `includeContext` 維持在 `"none"`。其他 context 模式會增加洩漏風險，而且本身也已被棄用。請在請求中傳入最少的明確 context。

## 安全不變量 (Safety Invariants)

對於內嵌的 Sampling 請求，client 就是信任邊界。

- 當政策要求核准時，向使用者顯示 server 要求模型做什麼。
- 限制 MRTR 的回合數。否則惡意 server 可以製造模型花費的迴圈。
- 在把每個 sampling response 用作檔名、URL 或 tool 輸入之前，先驗證它。
- 限制每個回合的 bytes 與 token。
- 拒絕當前 client capabilities 中沒有宣告的 input request。
- 不要讓模型輸出參與授權決策。
- 記錄發起的 method 與 input-request 的 key，但不要記錄敏感的 prompt 內容。

`clientInfo` 和 `serverInfo` 是顯示與診斷用的 metadata。絕不要把任何一個當作已驗證的 identity。

```figure
t3-sampling-flip
```

## 動手做 (Build It)

`code/main.py` 不用任何第三方套件，就實作完整的兩回合流程：

- `server/discover` 回傳 `supportedVersions`、公告 tool 支援，並回傳 cache 提示。
- `tools/list` 回傳確定性、可快取、帶有物件 input schema 的 `summarize_repo` descriptor。
- `tools/call` 驗證每請求的 metadata。
- 第一個結果內嵌用於挑選檔案的 `sampling/createMessage`。
- 第一次重試驗證模型結果，並內嵌第二個請求。
- 以 HMAC 保護的 `requestState` 在獨立的請求之間攜帶階段。
- 最終結果使用 `resultType: "complete"`。

假的 host 模型讓範例具有確定性。連接真正的 host 時，只需要替換 `fake_host_model`。Server 端的狀態機應保持確定性且可測試。

## 實際使用 (Use It)

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/11-mcp-sampling/code
python3 main.py
python3 -m unittest discover tests -v
```

預期的檢查點：

- Discovery 回傳帶有 `ttlMs` 與 `cacheScope` 的 complete 結果。
- Tool discovery 回傳相同排序的 descriptor，帶有 `resultType`、server identity 與 cache 提示。
- 缺少 capabilities 與不支援的 version，使用精確的 `-32021` 與 `-32022` 錯誤資料。
- 沒有 id 的 notification 不會產生 JSON-RPC response。
- Request id 為 `[1, 2, 3]`，證明每個 MRTR 回合都是獨立的。
- 前兩個結果是 `input_required`。
- 最終結果是 `complete`，並包含所選的檔案加上摘要。
- 在重試時改變原始參數，會讓 request-state 檢查失敗。

## 交付 (Ship It)

`outputs/skill-sampling-loop-designer.md` 現在是遷移規劃器。它先判斷是否應該移除 Sampling，改採直接模型整合。如果需要相容性，它會產出 MRTR 回合、狀態綁定、capability 閘門、預算、驗證和移除計畫。

## 練習 (Exercises)

1. 把檔案選擇的 response 改成無效的 JSON。確認 server 回傳 `-32602`，而不是信任模型輸出。
2. 在第一次 call 和重試之間改變 `audience`。說明為什麼密封的狀態能阻擋跨請求重用。
3. 新增第三個回合，請 host 對摘要做評論。把先前的摘要放在簽章過的狀態裡，並把整個流程限制在三個回合。
4. 把假的 host callback 換成 server 自有的模型 adapter，藉此移除 Sampling。列出哪些核准、計費和 observability 的責任會移到 server 身上。
5. 用一個比期限晚一秒的狀態值，新增過期測試。

## 關鍵術語 (Key Terms)

| 術語 | 在 2026-07-28 中的意義 |
|------|------------------------|
| Sampling | 向 client 的模型要求 completion 的已棄用功能 |
| MRTR | 請求期間需要 client 輸入時，使用的無狀態重試模式 |
| `InputRequiredResult` | 帶有 `resultType: "input_required"` 的結果 |
| `inputRequests` | Server 指派的 map，內含內嵌的 elicitation、sampling 或 roots 請求 |
| `inputResponses` | 當前回合 client 的結果，以與 `inputRequests` 相同的 key 為鍵 |
| `requestState` | 由 client 原樣回傳、由 server 驗證的不透明 server 狀態 |
| `resultType` | 現代 MCP 結果必備的 discriminator |
| Direct model integration | 對需要模型推論的新 server 的建議替代做法 |
| Capability gate | 防止送出 client 未公告的內嵌請求的規則 |
| Loop budget | 該操作允許的最大回合、token、bytes、時間與花費 |

## Legacy 相容性 (Legacy Compatibility)

固定在 2025-11-25 的 client，仍可能在即時連線上使用舊的 server 發起的 `sampling/createMessage` flow。請只把這種行為放在特定 version 的 adapter 中。不要讓有 session 的路徑成為 2026-07-28 server 的架構。

官方 SDK 可以為較舊的 peer 轉譯現代的 `input_required` handler。那個 shim 是相容性邊界，不是新增依賴 session 的邏輯的許可。

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog)
- [MCP Sampling deprecation](https://modelcontextprotocol.io/seps/2577-deprecate-roots-sampling-and-logging)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
