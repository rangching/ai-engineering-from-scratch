# MCP 授權：CIMD、Issuer 綁定、PKCE 與 Step-Up (MCP Authorization: CIMD, Issuer Binding, PKCE, and Step-Up)

> 遠端的 MCP 請求是無狀態的，但它的授權不是匿名的。把每個憑證綁定到建立它的 issuer，並把每個 token 綁定到接收它的 resource。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 09 (transports), Phase 13 · 15 (security)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 透過 protected-resource metadata 探索 authorization server。
- 優先使用 Client ID Metadata Documents，而不是已棄用的 Dynamic Client Registration。
- 當無法避免 DCR 相容路徑時，宣告正確的 `application_type`。
- 驗證 authorization response 的 `iss`，並依 issuer 隔離憑證。
- 使用 PKCE、resource indicator、audience 驗證與漸進式的 scope。
- 在沒有 protocol session 的情況下，送出經過授權的 MCP 2026-07-28 請求。

## 問題 (The Problem)

遠端的 MCP server 可能讀取私人紀錄、寫入外部系統，或觸發昂貴的工作。驗證身分（authentication）告訴它是誰出示了憑證。授權（authorization）還必須回答：

- 哪個 authorization server 簽發了這個憑證？
- 這個 token 是給哪個 MCP resource 的？
- 哪個 client 與 redirect URI 完成了這個流程？
- 使用者核准了哪些操作？
- 這個確切的請求是否仍然符合該核准？

2026-07-28 的授權 profile 強化了 client 註冊與 issuer 的處理。它偏好 Client ID Metadata Documents，棄用 Dynamic Client Registration，要求 DCR 必須帶有正確的 `application_type`，驗證 RFC 9207 的 issuer response，並禁止跨 issuer 重用憑證。

這些規則與無狀態核心互補。它們不會恢復核心 handshake 或 `Mcp-Session-Id`。

## 核心概念 (The Concept)

### 認識三個角色

- **MCP client：** 代表 resource owner 送出請求。
- **MCP resource server：** 接受 access token 並提供 MCP endpoint。
- **Authorization server：** 驗證 resource owner、蒐集同意，並簽發 token。

Resource server 與 authorization server 可以一起營運，但要把它們的識別碼與驗證責任分開。

### 授權適用於 HTTP

MCP 的授權規格適用於以 HTTP 為基礎的 transport。本地的 stdio server 在 process 與作業系統的信任邊界下執行。不要只為了對稱，就為 stdio 加上假的瀏覽器 OAuth 流程。

對遠端的 Streamable HTTP，在每個請求的 `Authorization` header 中送出 bearer token。絕不要把它放進 URL。

### 從 protected-resource metadata 開始

Resource server 發布 RFC 9728 的 metadata：

```json
{
  "resource": "https://notes.example.com/mcp",
  "authorization_servers": ["https://auth.example.com"],
  "scopes_supported": ["notes:delete", "notes:read", "notes:write"]
}
```

Client 從 MCP resource 的 URL 開始，取得這份文件，選擇一個公告的 authorization server，然後取得該 server 的 OAuth 或 OpenID Connect metadata。

建構 RFC 9728 的 well-known URL 時，要保留 resource 的路徑。對於 resource `https://notes.example.com/mcp`，這堂課使用 `https://notes.example.com/.well-known/oauth-protected-resource/mcp`。如果去掉 `/mcp` 後綴，可能會選到同一個 origin 上另一個受保護 resource 的 metadata。

不要從 hostname 猜測 authorization server。不要跟隨從未經驗證的錯誤 body 中發現的 issuer。請保有一份 client 願意信任哪些 issuer 的政策。

### 驗證 authorization server metadata

Metadata 應該公開 endpoint 與支援的控制項：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "code_challenge_methods_supported": ["S256"],
  "authorization_response_iss_parameter_supported": true,
  "client_id_metadata_document_supported": true
}
```

PKCE 要求 S256。記錄確切的 issuer 字串。這個確切的值，會成為註冊與 token 儲存的 key。

### 遵循註冊的優先順序

當 client 與所選 issuer 已有明確的關係時，使用預先註冊的 client 資訊。否則，在 authorization server 公告支援時，優先使用 Client ID Metadata Documents。只把 DCR 當作已棄用的相容性 fallback，如果以上機制都不可用，再提示輸入 client 資訊。

### 優先使用 Client ID Metadata Documents

Client ID Metadata Document 給 authorization server 一個 HTTPS URL，它同時是 client 識別碼，也是它 metadata 的所在位置：

```json
{
  "client_id": "https://client.example.com/oauth/metadata.json",
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

Authorization server 會取得並驗證這份文件。`client_id` 必須是帶有路徑的 HTTPS URL，而且文件內的值必須與該 URL 完全相等。文件必要的欄位是 `client_id`、`client_name` 和 `redirect_uris`。`application_type` 出現在這個範例中，但它不是 CIMD 的要求。它新的強制使用，專指 DCR 路徑。

把取得文件視為對 SSRF 敏感的操作。解析並驗證目的地，拒絕 loopback、private、link-local 以及其他不允許的位址，在 redirect 與 DNS 變更之後重新檢查，限制 redirect、bytes 與時間，要求 JSON，並只依據經過驗證的 HTTP cache 控制來快取。把 `client_name` 和其他顯示欄位視為不受信任的文字。

CIMD 免除了為每次首次接觸都產生全新動態識別碼的需要。它不會免除 redirect URI 驗證、issuer 政策或使用者同意。

### DCR 是相容性路徑

Dynamic Client Registration 對較舊的 authorization server 仍然可用，但對新的 MCP 實作已被棄用。

使用 DCR 時，宣告 `application_type`：

```json
{
  "client_name": "Notes desktop client",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:8765/callback"],
  "grant_types": ["authorization_code"],
  "response_types": ["code"]
}
```

- 桌面、行動、命令列與 loopback 的 client 使用 `native`。
- 遠端託管的瀏覽器應用程式使用 `web` 與遠端 HTTPS redirect。

省略這個欄位，在 OpenID Connect 註冊實作中可能預設為 `web`，並讓合法的 loopback redirect 失敗。

把 DCR 程式碼放在明確的 fallback 決定之後。不要在任意的 CIMD 驗證失敗之後，就默默 fallback。那可能把安全失敗，變成較弱的註冊路徑。

### 把憑證綁定到 issuer

把由 issuer 簽發的註冊資料，存在確切的 issuer 底下：

```text
issuer_credentials[issuer] = pre_registered_or_dcr_client
tokens[(issuer, resource)] = access_token
```

如果 protected-resource discovery 從 `https://auth-one.example` 變成 `https://auth-two.example`，要重新評估信任。絕不要把第一個 issuer 的 client secret、DCR client id、registration access token、refresh token 或 access token，送給第二個。預先註冊的 client 與 DCR client 必須使用為新 issuer 簽發的憑證。

CIMD 的 client id 不同，因為它是自行託管的 HTTPS URL，不是由 authorization server 簽發的憑證。同一個 CIMD URL 是可攜的：新的受信任 issuer 會取得並驗證該文件，不需要 DCR 重新註冊。Authorization response 與 token 仍然在新的 issuer 底下驗證並儲存。

### 搭配 PKCE 的 Authorization code

互動流程是：

1. 產生高熵的 `code_verifier`。
2. 推導出 S256 的 `code_challenge`。
3. 送出授權請求，帶有確切的 `client_id`、`redirect_uri`、`scope`、`code_challenge` 與 `resource`。
4. 收到包含 `code` 的 authorization response，若有提供，還有 `iss`。
5. 在使用任何 response 欄位之前，先把 `iss` 與記錄下來的確切 issuer 比對驗證。
6. 用 `code_verifier`、相同的 redirect URI 和相同的 `resource` 交換 code。
7. 把得到的 token 存在 `(issuer, resource)` 底下。

RFC 8707 的 `resource` 參數，同時出現在授權請求和 token 請求中。它識別的是 canonical 的 MCP server URI。

### 精確驗證 `iss`

RFC 9207 防止來自某個 issuer 的 authorization response，被誤認為來自另一個 issuer 的 response。

當 `iss` 存在時，把它與記錄下來的 issuer 做比對，不做大小寫折疊、結尾斜線變更、預設 port 移除或 percent-encoding 正規化。不一致時，不要對 code 採取行動，甚至不要顯示該 response 中由攻擊者控制的錯誤細節。

包含 `iss` 的 authorization server，會公告 `authorization_response_iss_parameter_supported: true`。目前的 client 即使缺少這個公告，也仍然會驗證出現的 `iss`。

### 在 MCP server 驗證 audience

Resource server 只接受簽發給自己的 token：

```text
token.issuer == configured_authorization_server
token.audience == canonical_mcp_resource
```

無效、過期、錯誤 issuer 或錯誤 audience 的 token，會收到 401。MCP server 不得接受或轉送給其他服務使用的 token。

### 請求最小的當前 scope

從當下需要的 scope 開始。如果之後的 tool 需要更多，server 會回傳 403 與具權威性的 scope challenge：

```text
WWW-Authenticate: Bearer error="insufficient_scope",
  scope="notes:delete",
  resource_metadata="https://notes.example.com/.well-known/oauth-protected-resource/mcp"
```

Client 說明新的 permission、取得同意、用合併後的 scope 集合執行新的授權流程，並以新的 JSON-RPC id 重試 MCP 請求。

不要假設被 challenge 的 scope 是 `scopes_supported` 的子集合。該 challenge 對當前的操作具有權威性。

### 授權與無狀態的 MCP wire

經過授權的 tool call 仍然攜帶完整的當前請求 envelope：

```text
POST /mcp
Authorization: Bearer <access-token>
MCP-Protocol-Version: 2026-07-28
Mcp-Method: tools/call
Mcp-Name: notes.delete
```

```json
{
  "jsonrpc": "2.0",
  "id": 12,
  "method": "tools/call",
  "params": {
    "name": "notes.delete",
    "arguments": {"id": "note-7"},
    "_meta": {
      "io.modelcontextprotocol/protocolVersion": "2026-07-28",
      "io.modelcontextprotocol/clientCapabilities": {},
      "io.modelcontextprotocol/clientInfo": {
        "name": "oauth-lesson-client",
        "version": "1.0.0"
      }
    }
  }
}
```

Token 授權 principal。請求 metadata 協商 protocol 行為。兩者都不能取代對方。

以固定的順序驗證 wire：JSON-RPC 與 metadata 的型別、header 與 body 相等、然後才是 protocol 支援。路由或 version header 不一致，會回傳 HTTP 400 與 `-32020`。如果 header 與 body 對一個不支援的 version 看法一致，回傳 HTTP 400 與 `-32022`，且 `data` 精確為 `{"supported":["2026-07-28"],"requested":"<actual>"}`。未知的 method 回傳 HTTP 404 與 `-32601`。

每個請求錯誤，包括 401 無效 token 與 403 scope 不足，都是帶有原始請求 `id` 的 JSON-RPC 錯誤 envelope。結構化的復原資訊屬於選擇性的錯誤 `data`；`WWW-Authenticate` 仍是 HTTP response header。Notification 沒有 `id`，所以不會收到 JSON-RPC body。被接受的 HTTP notification 回傳 202 與空的 body。

Server 實作 `server/discover` 並公告 tools，所以也實作強制要求的 `tools/list` method。它的 tool descriptor 有穩定的 name、description 與物件根的 `inputSchema` 值。清單具有確定性，並回傳 `resultType`、server identity metadata、有界限的 `ttlMs` 與 `cacheScope`。Discovery 和與使用者無關的 tool 清單，可以在授權之前取得。如果其中任何一個隨 principal 而不同，請套用一般政策與 private 快取。

### 不要 token passthrough

MCP server 不得把 client 的 MCP access token 轉送給下游 API。請取得具有正確 audience 的獨立下游 token，或使用明確的 token-exchange 設計。Audience 驗證只有在各服務拒絕為別人簽發的 token 時才有效。

### Refresh token

Refresh token 是選擇性的。簽發後，請以機密方式儲存，並以 issuer 與 resource 作為 key。不要假設它們存在。當 authorization server 支援輪換時就輪換，並偵測已失效的值被重用。

```figure
t3-scope-stepup
```

## 動手做 (Build It)

`code/main.py` 是 in-process 的 protocol 與授權 simulator。它實作了 protected-resource discovery、authorization server metadata、CIMD 註冊、受 version 管控的 DCR fallback、application type 檢查、PKCE、issuer 驗證、綁定 resource 的 token、scope step-up、`server/discover`、`tools/list`，以及無狀態的 tool 請求。

這個模型接收的是已解析的請求 body 與路由 header。它不是完整的 HTTP adapter，也不解析 `Content-Type` 或 `Accept`。請把它接到課程 09 的 Streamable HTTP adapter，它要求 `Content-Type: application/json`，以及同時包含 `application/json` 和 `text/event-stream` 的 `Accept` 值。

執行它：

```bash
cd phases/13-tools-and-protocols/16-mcp-security-oauth-2-1
python3 code/main.py
python3 -m unittest discover code/tests -v
```

輸出依序顯示 discovery、CIMD 註冊、一次一般讀取、兩次獨立的 scope step-up，以及依 issuer 為 key 的憑證儲存。

## 實際使用 (Use It)

把 simulator 的物件對應到 production 的元件：

- `ResourceServer.protected_resource_metadata` 變成 RFC 9728 的 endpoint。
- `AuthorizationServer.metadata` 變成 RFC 8414 或 OpenID Connect 的 discovery。
- `Client.enroll` 變成 CIMD 解析，加上明確的 DCR 相容性分支。
- 由 issuer 簽發的 client 憑證與 `tokens_by_issuer_resource` 變成加密的紀錄。CIMD URL 可以保持可攜，而它的授權結果仍然綁定 issuer。
- `ResourceServer.handle` 變成 middleware，在 dispatch 之前驗證目前的 MCP header、token 與 tool scope，同時讓每個請求錯誤都留在對應的 JSON-RPC envelope 中。

## 交付 (Ship It)

這堂課會交付 `outputs/skill-oauth-scope-planner.md`。它現在會設計註冊的優先順序、綁定 issuer 的憑證儲存、application type、PKCE、resource indicator、scope challenge，以及目前無狀態的請求邊界。

## 練習 (Exercises)

1. 加入 refresh-token 輪換，並拒絕重用先前的 refresh token。
2. 加入 issuer allowlist。Issuer 改變時，只重用可攜的 CIMD URL；拒絕所有先前由 issuer 簽發的憑證與 token。
3. 為 authorization code 加入過期時間，並確認太晚的交換會失敗。
4. 建構使用遠端 HTTPS redirect 的 web client 變體，並將它的 DCR metadata 與 native client 比較。
5. 在同一個 issuer 底下加入第二個 resource。確認它的 access token 不能在第一個 resource 使用。

## 關鍵術語 (Key Terms)

| 術語 | 意義 |
|------|---------|
| Protected-resource metadata | 識別 resource 與 authorization server 的 RFC 9728 文件 |
| CIMD | 其 URL 就是 OAuth client 識別碼的 HTTPS metadata 文件 |
| DCR | 為相容性而保留、已棄用的動態 client 註冊 |
| `application_type` | `native` 或 `web`，用來驗證 redirect URI 規則 |
| PKCE | 保護被攔截的 authorization code 的 verifier 與 S256 challenge |
| `iss` | RFC 9207 的 authorization response issuer 識別碼 |
| Resource indicator | 將 token 請求綁定到 MCP resource 的 RFC 8707 參數 |
| Audience | token 有效的 resource |
| Step-up | 為額外的當前操作 scope 重新取得同意並簽發 token |
| Issuer-bound credentials | 依確切 authorization server issuer 隔離的註冊與 token 紀錄 |

## 延伸閱讀 (Further Reading)

- [MCP 2026-07-28 authorization specification](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization)
- [RFC 9728: OAuth 2.0 Protected Resource Metadata](https://www.rfc-editor.org/rfc/rfc9728)
- [RFC 8707: Resource Indicators for OAuth 2.0](https://www.rfc-editor.org/rfc/rfc8707)
- [RFC 9207: OAuth 2.0 Authorization Server Issuer Identification](https://www.rfc-editor.org/rfc/rfc9207)
- [OAuth Client ID Metadata Document draft](https://datatracker.ietf.org/doc/draft-ietf-oauth-client-id-metadata-document/)
