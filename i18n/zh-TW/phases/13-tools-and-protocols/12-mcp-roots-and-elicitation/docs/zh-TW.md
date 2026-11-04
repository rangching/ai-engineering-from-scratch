# 明確的 Scope 與無狀態 Elicitation (Explicit Scope and Stateless Elicitation)

> Roots 在 MCP 2026-07-28 中已被棄用，而且從來就不是安全 sandbox。把 scope 放在看得見的 tool 參數或 resource URI 裡，在 server 上授權它，並在 tool 真的需要使用者輸入時使用 MRTR。使用者看到決定，模型看到 handle，任何一個 server 實例都能處理重試。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 07 (MCP server), Phase 13 · 11 (stateless MRTR)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 用明確的 workspace 參數、resource URI 或 server 設定，取代已棄用的 Roots。
- 把 scope 提示，與授權、路徑限制（containment）和作業系統層級的 sandbox 區分開來。
- 透過 MRTR 的 `input_required` 結果，傳送 form 模式的 `elicitation/create`。
- 在每請求的 client capabilities 中公告 elicitation 支援，並拒絕不支援的模式。
- 把 `accept`、`decline` 和 `cancel` 驗證為各自獨立的結果。
- 將破壞性操作的確認，綁定到已驗證的 principal、原始參數、候選集合與過期時間。

## 兩個看起來相似的問題 (Two Problems That Look Similar)

筆記 tool 收到這個請求：「刪除舊的 TPS report。」

Server 必須回答兩個不同的問題。

1. 這個操作可以碰哪個 workspace？
2. 三筆相符的 note 中，使用者指的是哪一筆？

第一個是 scope 與授權。第二個是互動式的釐清。把兩者混在一起，會導致危險的設計，例如把 client 提供的資料夾，當成呼叫者可以刪除其中所有內容的證明。

## Roots 是遷移介面 (Roots Are a Migration Surface)

較早的 MCP revision 讓 client 公告 Roots，並在清單變更時通知 server。Roots 是資訊性的指引。它們不會限制 server process 可以讀取什麼，不會授權呼叫者，也不會建立作業系統層級的 sandbox。

MCP 2026-07-28 對新的設計棄用 `roots/list` 與 `notifications/roots/list_changed`。請優先選擇以下明確的替代做法之一：

- 當 scope 隨每次 call 而不同時，使用 `workspaceUri` 或 `directory` tool 參數。
- 當操作本來就以某個 resource 為目標時，使用 resource URI。
- 當一個部署擁有一個固定的 workspace 時，使用 server 設定。
- 當程式碼在技術上必須無法逃離時，使用 process sandbox 或 jailed filesystem。

如果現有的 2026-07-28 整合在棄用期間仍然需要 `roots/list`，server 會把它內嵌在 MRTR 的 `inputRequests` 中。它不得送出即時的反向請求。那是遷移用的 adapter；新的 handler 應該改為接受明確的 scope。

模型可以看見並重複明確的 handle。隱藏在 transport session 裡的 scope，則更難檢查、重放、稽核和路由。

### 三層規則 (The three-layer rule)

明確的 URI 仍然不會自己授權自己。請強制執行這三層：

1. **授權 (Authorization)：** 這個已驗證的 principal 是否被允許使用這個 workspace？
2. **限制 (Containment)：** 正規化後的目標 URI 是否留在已授權的 workspace 邊界內？
3. **Sandbox：** 作業系統能不能在 server 被入侵時，仍然阻止它逃脫？

可執行的 server 保存一份已授權 workspace URI 的 allowlist，正規化經過 percent-encoding 的路徑，檢查真正的路徑元件邊界，並在刪除之前立刻重新檢查 containment。

單純的字串前綴檢查是錯的：

```text
allowed:   file:///work/notes
attacker:  file:///work/notes-evil/secret.md
traversal: file:///work/notes/%2e%2e/private.md
```

兩個惡意路徑都以具誤導性的字串開頭。先正規化，再比對路徑元件。Production 的檔案系統 server 還必須防禦 symbolic-link race 和平台特有的路徑語意。

## Elicitation 仍然存在，但傳送方式改變了 (Elicitation Still Exists, but Delivery Changed)

Elicitation 是目前讓 client 在 `tools/call`、`prompts/get` 或 `resources/read` 期間蒐集使用者輸入的功能。Method 名稱仍是 `elicitation/create`。改變的是 wire flow 的方向。

2026-07-28 的 server 不會送出反向的 JSON-RPC 請求。它回傳 `InputRequiredResult`：

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "resultType": "input_required",
    "inputRequests": {
      "delete_choice": {
        "method": "elicitation/create",
        "params": {
          "mode": "form",
          "message": "Choose one matching note and confirm deletion.",
          "requestedSchema": {
            "type": "object",
            "properties": {
              "note_id": {
                "type": "string",
                "enum": ["note-3", "note-7", "note-14"]
              },
              "confirm": {"type": "boolean"}
            },
            "required": ["note_id", "confirm"]
          }
        }
      }
    },
    "requestState": "integrity-protected-delete-state"
  }
}
```

Host 轉譯表單。使用者可以接受、明確拒絕，或直接關掉。接著 client 用新的 id 重試原本的 `tools/call`：

```json
{
  "jsonrpc": "2.0",
  "id": 2,
  "method": "tools/call",
  "params": {
    "name": "notes_delete",
    "arguments": {
      "workspaceUri": "file:///Users/alice/Documents/Notes",
      "title": "TPS report"
    },
    "inputResponses": {
      "delete_choice": {
        "action": "accept",
        "content": {"note_id": "note-14", "confirm": true}
      }
    },
    "requestState": "integrity-protected-delete-state",
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {
        "elicitation": {"form": {}}
      }
    }
  }
}
```

兩次 call 之間沒有 protocol session。Server 驗證回傳的 state、依預期 schema 驗證 response、檢查所選的 note 在簽章過的候選集合中、重新授權 workspace、重新檢查 containment，然後才刪除。

## Capability 協商是每個請求各自進行 (Capability Negotiation Is Per Request)

支援 form 模式 elicitation 的 client 會宣告：

```json
{
  "io.modelcontextprotocol/clientCapabilities": {
    "elicitation": {"form": {}}
  }
}
```

為了相容性，空的 elicitation capability `"elicitation": {}` 仍等同於僅支援 form。明確的 `"elicitation": {"form": {}}` 同樣支援 form 模式。僅宣告 URL 的 `"elicitation": {"url": {}}` 則不支援。Server 不得內嵌當前請求 capabilities 中沒有的模式，即使先前的請求曾公告過。

每個請求也都帶有 `io.modelcontextprotocol/protocolVersion`。缺少或不是字串的 version 會回傳 `-32602`。不支援的字串會回傳 `-32022`，並附上精確的 `supported` 與 `requested` 資料。缺少或僅支援 URL 的 elicitation，會回傳 `-32021`，且 `data.requiredCapabilities` 設為 `{"elicitation":{"form":{}}}`。

沒有 JSON-RPC `id` 的 envelope 是 notification。處理它，但不發出 JSON-RPC 的成功或錯誤 response。在 Streamable HTTP 上，被接受的 notification 會收到沒有 body 的 `202 Accepted`。

應該包含 `clientInfo` 供診斷使用，但它是自行申報的，不能用來識別授權的使用者。

Server 實作 `server/discover`，並回傳 `supportedVersions`、capabilities、`ttlMs` 與 `cacheScope`，以及 `resultType: "complete"`。對這個現代設計，它不公告 Roots。因為它公告了 tools，所以也實作強制要求的 `tools/list`。該結果回傳確定性的 `notes_delete` descriptor、有效的物件 `inputSchema`、server identity metadata，以及 public 的 cache 提示。

## Form 模式 (Form Mode)

Form 模式使用為可用對話框設計的受限 JSON Schema。根是一個物件，它的 property 是扁平的基本型別欄位，或受支援的 enum 陣列。深層巢狀的物件和通用的文件 schema，不適合放在確認對話框裡。

Form 模式適用於：

- 在幾個候選項目中選擇一個；
- 確認破壞性操作；
- 蒐集非敏感的偏好設定；
- 蒐集少量必須由使用者（而不是模型）決定的值。

不要用 form 模式處理密碼、API key、access token 或付款憑證。這些機密會經過 MCP client，可能進入 log 或模型 context。

Server 會再次驗證回傳的內容。Client 端的表單驗證能改善 UX，但不會建立信任。

## URL 模式 (URL Mode)

URL 模式會送出一個安全的網頁 URL，用於帶外（out-of-band）互動：

```json
{
  "method": "elicitation/create",
  "params": {
    "mode": "url",
    "message": "Connect the report service to continue.",
    "url": "https://mcp.example.com/connect/report-service"
  }
}
```

當敏感資訊必須直接送到 server 控制的網頁流程時使用，例如第三方授權。Client 會顯示完整的目的地，並在開啟之前取得同意。它不得預先抓取（prefetch）該 URL。

`accept` response 表示使用者同意開啟 URL。它不能證明外部流程已完成。重試時，server 檢查自己的狀態，然後完成，或回傳另一個 `input_required` 結果。

URL elicitation 不是 MCP client 與 MCP server 之間授權的替代品。它用於 MCP server 需要代表使用者執行的外部互動。Server 必須把瀏覽器使用者，綁定到發起這次 MCP 操作的同一個已驗證 principal。

## Response 分支 (Response Branches)

把這些 action 視為產品決策，而不是別名：

| Action | 意義 | 安全的 server 行為 |
|--------|---------|----------------------|
| `accept` | 使用者提交了互動 | 驗證內容並繼續 |
| `decline` | 使用者明確拒絕 | 回傳 complete、非錯誤的拒絕結果 |
| `cancel` | 使用者關掉，或無法完成 | 安全地停止，並允許稍後重試 |

絕不要把缺少內容解讀成同意。絕不要把 decline 轉成重複提示的迴圈。

## 保護破壞性的 MRTR 狀態 (Protecting Destructive MRTR State)

候選清單不能只存在於 prompt 或未簽章的 Base64 值裡。Client 控制它送回來的一切。

這堂課對一份 state payload 簽章，內容包括：

- 已驗證的 principal；
- 發起的 method；
- `workspaceUri` 與 `title` 的 digest；
- 表單中顯示的允許 note id；
- 操作階段；
- 短暫的過期時間。

在變更之前，server 也會檢查即時的 note 紀錄。這樣可以抓出刪除的 race，以及表單顯示之後被移出 workspace 的目標。

對於一次性的金融或不可逆的操作，單靠 HMAC 無法防止有效的 state 在過期前被重放。請在所有 handler 實例共用的 replay store 中，只儲存並消耗 nonce 一次。這堂課注入一個有界限、會依 TTL 清理的 store，並在執行記憶體內刪除時，持有它的 atomic claim。Production 資料庫應該把 nonce claim 與變更放在同一個 transaction，或等效的條件式寫入邊界裡。

在 claim nonce 之前先驗證互動。格式錯誤的 response 或 `cancel` 不會執行任何變更，並讓 state 在過期前仍可重試。明確的 `decline` 是終結狀態，所以這堂課會消耗 nonce，但不刪除任何東西。

```figure
t3-roots-boundary
```

## 動手做 (Build It)

`code/main.py` 示範現代的 `notes_delete` tool：

- `tools/list` 回傳確定性、可快取、帶有必要 workspace 與 title schema 的 descriptor。
- Scope 是明確的 `workspaceUri` 參數。
- Server 設定為這堂課的 principal 授權該 workspace。
- URI 正規化會拒絕前綴混淆與編碼過的路徑穿越。
- 每次破壞性刪除都需要 form 模式的 elicitation。
- Elicitation 包在 `resultType: "input_required"` 之中傳送。
- 簽章過的 `requestState` 綁定確切的候選清單與原始參數。
- 注入的 replay store 會拒絕同一個已接受或已拒絕的 state，跨 server 實例也一樣。
- 重試使用新的 request id，並回傳 `resultType: "complete"`。

資料 store 放在記憶體中，讓 protocol 行為容易檢查。換成資料庫，安全規則仍然相同。

## 實際使用 (Use It)

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/12-mcp-roots-and-elicitation/code
python3 main.py
python3 -m unittest discover tests -v
```

預期的檢查點：

- Discovery 公告 tools，沒有 Roots。
- Tool discovery 回傳 `notes_delete`，帶有 `resultType`、server identity 與 cache 提示。
- Request id `1` 在 `inputRequests.delete_choice` 中回傳表單。
- Request id `2` 回傳簽章過的 state，並完成刪除。
- 前綴路徑與編碼過的穿越路徑，都無法通過 containment。
- 改過的 title 無法重用原本的確認 state。
- Decline 會讓 note 保持不變。
- 共用 note 與 replay 狀態的兩個 server 物件，無法都執行同一個確認。
- 空的與明確的 form 宣告都能運作，而僅支援 URL 會回傳精確的 `-32021` form 需求。
- 不支援 version 的失敗，使用精確的 `-32022` 資料形狀。
- 沒有 id 的 notification 不會產生 JSON-RPC response。

## 交付 (Ship It)

`outputs/skill-elicitation-form-designer.md` 設計明確的 scope、授權檢查、MRTR 表單、response 分支與 state 綁定。它拒絕把已棄用的 Roots 當成 sandbox，也拒絕透過 form 模式蒐集機密。

## 練習 (Exercises)

1. 把記憶體內的 replay store 換成 SQLite。用一個 transaction 同時 claim nonce 並刪除 note，然後證明兩個 process 無法都 commit。
2. 新增 `url` capability 協商與帶外的設定流程。不要讓第三方憑證出現在 `inputResponses` 中。
3. 把記憶體內的 note map 換成暫時的 SQLite 資料庫。在變更的 transaction 內重新檢查授權與 containment。
4. 為真正的檔案系統實作加入 symbolic-link 政策。說明為什麼只靠 URI 的字面 containment，無法阻止 symlink 逃脫。
5. 設計一個 2025-11-25 的 adapter，把現代 MRTR handler 的輸出對應到 legacy 由 server 發起的 elicitation。讓它與目前的 handler 保持隔離。

## 關鍵術語 (Key Terms)

| 術語 | 在 2026-07-28 中的意義 |
|------|------------------------|
| Roots | 已棄用的資訊性 workspace 提示，不是授權或 sandbox |
| Explicit scope | 在請求參數中可見的 workspace、目錄或 resource handle |
| Containment | 讓目標留在邊界內的正規化路徑元件檢查 |
| Elicitation | 在 MCP 操作期間取得使用者輸入的 client 功能 |
| Form mode | 使用受限扁平 schema 的帶內結構化使用者輸入 |
| URL mode | 用於敏感或外部 workflow 的帶外互動 |
| MRTR | 無狀態的 input-required 結果，接著是全新的重試 |
| `requestState` | 由 server 檢查完整性、需原樣回傳的不透明狀態 |
| Decline | 使用者明確拒絕 |
| Cancel | 沒有核准的關閉或未完成的互動 |

## Legacy 相容性 (Legacy Compatibility)

對於固定在 2025-11-25 的 peer，`roots/list`、`notifications/roots/list_changed` 與即時由 server 發起的 `elicitation/create` 可能仍然存在。請把該 adapter 標示為 legacy。不要讓 legacy 的 Root 清單繞過 server 授權，也不要把 protocol-session 的假設帶進現代的 handler。

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 Elicitation](https://modelcontextprotocol.io/specification/2026-07-28/client/elicitation)
- [MCP 2026-07-28 Multi Round-Trip Requests](https://modelcontextprotocol.io/specification/2026-07-28/basic/patterns/mrtr)
- [MCP 2026-07-28 Roots deprecation](https://modelcontextprotocol.io/specification/2026-07-28/client/roots)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
