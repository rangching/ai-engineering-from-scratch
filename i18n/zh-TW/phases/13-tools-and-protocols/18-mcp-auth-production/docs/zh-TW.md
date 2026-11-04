# Production 中的 MCP Auth：綁定 Issuer 的註冊與 Token (MCP Auth in Production: Issuer-Bound Enrollment and Tokens)

> 課程 16 建立了 OAuth 2.1 的狀態機。這堂課針對 MCP 2026-07-28，強化它在 production 的邊界：優先使用 Client ID Metadata Documents，已棄用的動態註冊只為相容性保留，驗證 authorization response 的 issuer，以 issuer 為 key 的 client 憑證，JWKS 重新整理，以及在每個無狀態請求上固定 audience 的 token。
>
> **規格說明 (2026-07-28)：** Dynamic Client Registration 已被棄用，改由 Client ID Metadata Documents 取代。DCR 仍是相容性機制。使用時，client 要宣告正確的 `application_type`。Client 會驗證出現的 RFC 9207 `iss` 值，並且絕不跨 authorization-server issuer 重用憑證。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 16 (OAuth 2.1 state machine), Phase 13 · 17 (gateways)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 透過 RFC 8414 metadata 探索 authorization server，並驗證其合約。
- 透過 Client ID Metadata Document 註冊，並把已棄用的 DCR 隔離成 fallback。
- 驗證 RFC 9207 的 `iss`，以 authorization-server issuer 為 key 管理註冊，並以 issuer 加 resource 為 key 管理綁定 resource 的 token。
- 依排程快取並重新整理 JWKS key，讓簽章驗證能撐過 key 輪換。
- 使用 RFC 8707 resource indicator 把 token 固定到單一 MCP resource，並拒絕 confused-deputy 式的重用。
- 在 JWT 驗證與 token introspection 之間做選擇，定義撤銷的新鮮度，並在身分相依元件不可用時安全地失敗。
- 把 authorization server、resource server 與 client 分開，讓每一個只強制執行屬於自己的檢查。
- 依部署檢查清單稽核 authorization server，並拒絕不安全的註冊或 token 重用。

## 問題 (The Problem)

課程 16 的 simulator 在記憶體中執行 OAuth 2.1。Production 有三個營運上的缺口，是只在記憶體裡跑的 simulator 看不到的。

第一個缺口是註冊與憑證隔離。真實的組織可能運行數百個 MCP server 和數千個 MCP client。2026-07-28 revision 偏好 **Client ID Metadata Document**：client 使用自己控制、帶有路徑的 HTTPS URL 作為識別碼，由 authorization server 去拉取 metadata。RFC 7591 的動態註冊只作為已棄用的相容性路徑保留。當無法避免 DCR 時，請求要宣告正確的 `application_type`。Client 把註冊資料存在 authorization-server issuer 底下，把 access token 存在 `(issuer, resource)` 這一組底下。Issuer 改變，代表要重新註冊；resource 不同，代表要另外綁定 audience 的 token。

第二個缺口是 key 輪換。JWT 驗證依賴 authorization server 的簽章 key，這些 key 以 JSON Web Key Set (JWKS) 發布。Authorization server 依排程輪換它們（常常是每小時，事件應變時甚至更快）。只在開機時取得一次 JWKS 的 MCP server，在輪換視窗之前都驗證正常，之後每個請求都會失敗，直到重啟。Production 會把 JWKS 接成快取值，加上一個在舊 key 過期前就覆寫快取的重新整理 job，另外還有快取 miss 時的 fall-back 取得，用來處理到達的 token 是由比快取更新的 key 所簽的情況。

第三個缺口是 audience 綁定。課程 16 介紹了 RFC 8707 的 resource indicator。在 production 中，該 indicator 會變成每個請求上硬性的 claim 檢查。MCP server 把 `token.aud` 與自己的 canonical resource URL 比對，並以 HTTP 401 拒絕不一致。這是防止上游 MCP server（或持有某個 server 專用 token 的惡意 client）把該 token 重放到同一個信任網格中另一個 server 的唯一防線。

這堂課把每個缺口對應到介面上具體的一塊。Metadata 文件是一個 HTTP endpoint。JWKS 快取重新整理是排程 job 加上 key-value 快取。JWT 驗證是 resource server 在分派任何 tool 之前執行的例行程序。讓三個角色保持分離，每個角色只強制執行自己負責的檢查：authorization server 簽發並輪換 key，resource server 快取並驗證，client 探索並註冊。

## 範圍：課程 16 之後的 Production 強制執行 (Scope: Production Enforcement After Lesson 16)

[Lesson 16: MCP Security with OAuth 2.1](../../16-mcp-security-oauth-2-1/docs/en.md) 負責 authorization-code 狀態機、PKCE、protected-resource discovery、resource indicator 與 scope 決定。這堂課不會定義第二套 OAuth 流程。它從那些合約已經存在之後開始，探討已部署的 resource server 如何在 key 輪換、opaque-token 驗證、撤銷、相依元件失敗、rollout 與事件應變期間，持續強制執行它們。

Production 的邊界更窄，也更偏營運：

- JWT 路徑在每個請求上，驗證固定的 issuer、演算法、簽章 key、audience、時間 claim 與 scope，同時安全地重新整理 JWKS。
- Opaque-token 路徑呼叫 issuer 經過驗證的 introspection endpoint，並驗證回傳的 active 狀態、audience 或 resource、到期時間、subject 與 scope。
- 撤銷政策定義憑證必須多快停止運作，以及哪個快取可以延遲這件事。
- 失敗政策決定當 discovery、JWKS、introspection 或撤銷基礎設施不可用時，該怎麼做。
- 證據記錄了是哪個 issuer metadata、key set 或 introspection response、token claim、政策版本與拒絕原因，導出了這個結果，而且不儲存 token。

這個區分讓各堂課可以組合。課程 16 證明流程。課程 18 證明 token 在進入真正的 MCP 請求路徑之後，仍然值得信任，或被拒絕。

## 核心概念 (The Concept)

### RFC 8414：OAuth Authorization Server Metadata

位於 `/.well-known/oauth-authorization-server` 的文件，描述 client 需要的一切：

```json
{
  "issuer": "https://auth.example.com",
  "authorization_endpoint": "https://auth.example.com/authorize",
  "token_endpoint": "https://auth.example.com/token",
  "jwks_uri": "https://auth.example.com/.well-known/jwks.json",
  "client_id_metadata_document_supported": true,
  "registration_endpoint": "https://auth.example.com/register",
  "authorization_response_iss_parameter_supported": true,
  "response_types_supported": ["code"],
  "grant_types_supported": ["authorization_code", "refresh_token"],
  "code_challenge_methods_supported": ["S256"],
  "scopes_supported": ["mcp:tools.read", "mcp:tools.invoke"],
  "token_endpoint_auth_methods_supported": ["none", "private_key_jwt"]
}
```

拿到 MCP resource URL 的 client 會串連 discovery：RFC 9728 的 `oauth-protected-resource`（resource server 的文件）指出 issuer，然後 `oauth-authorization-server`（本 RFC）指出每個 endpoint。Client 絕不寫死授權 URL。

對於帶有路徑的 resource 識別碼，把 well-known 片段插入該路徑之前。例如，`https://mcp.example.com/team/server` 的 protected-resource metadata 解析到 `https://mcp.example.com/.well-known/oauth-protected-resource/team/server`。把 `/.well-known/...` 附加在 resource 路徑之後是不正確的。

在信任某個 IdP 用於 MCP 之前，你要驗證的合約：

- `code_challenge_methods_supported` 包含 `S256`（RFC 7636 的 PKCE）。規格很明確：如果這個欄位**不存在**，authorization server 就不支援 PKCE，client **必須**拒絕繼續。
- `grant_types_supported` 包含 `authorization_code`，並拒絕 `password` 和 `implicit`。
- 至少有一條註冊路徑可用：`client_id_metadata_document_supported: true`（CIMD，建議）、預先註冊的 client，或 `registration_endpoint`（已棄用的 RFC 7591 相容性）。
- 如果 `authorization_response_iss_parameter_supported` 為 true，client 要求回傳 RFC 9207 的 `iss`，並與重新導向前記錄的 issuer 精確比對。
- 對 OAuth 2.1 而言，`response_types_supported` 恰好是 `["code"]`。

如果缺少 `S256`，MCP server 拒絕針對這個 IdP 部署，PKCE 沒有降級模式。如果*兩種*註冊路徑都沒有公告，而你又沒有預先註冊的 `client_id`，你同樣無法註冊；錯的是部署清單，不是程式碼。

### RFC 9728（回顧）：Protected Resource Metadata

課程 16 涵蓋了 RFC 9728。Production 中的差異在於：這份文件是 client 尋找*這個* MCP server 所信任的 authorization server 的唯一地方。單一 MCP server 可能接受來自多個 IdP 的 token（一個給員工，一個給合作夥伴）。RFC 9728 宣告那個集合；RFC 8414 則說明每個 IdP 支援什麼。

```json
{
  "resource": "https://notes.example.com",
  "authorization_servers": ["https://auth.example.com", "https://partners.example.com"],
  "scopes_supported": ["mcp:tools.invoke"],
  "bearer_methods_supported": ["header"],
  "resource_documentation": "https://notes.example.com/docs"
}
```

### Client ID Metadata Documents（建議的預設做法）

CIMD 把註冊從*推*反轉成*拉*。Client 不是請 authorization server 產生 `client_id`，而是把它自己控制的 HTTPS URL **當作** `client_id`。該 URL 解析到一份 JSON metadata 文件；authorization server 在 OAuth 流程中按需取得它。信任的根基是 DNS：如果 server 營運者信任 `app.example.com`，就信任從 `https://app.example.com/client.json` 提供的 client。沒有註冊往返、沒有會耗盡的 `client_id` namespace、也沒有需要同步的每 server 狀態。

Client 託管的 metadata 文件：

```json
{
  "client_id": "https://app.example.com/oauth/client.json",
  "client_name": "Example MCP Client",
  "client_uri": "https://app.example.com",
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback", "http://localhost:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none"
}
```

文件中的 `client_id` 值**必須**等於提供它的 URL（authorization server 會驗證這點；不一致會被拒絕）。Authorization server 在其 RFC 8414 metadata 中以 `client_id_metadata_document_supported: true` 公告支援。

依目前的 CIMD 合約，`client_id`、`client_name` 與非空的 `redirect_uris` 陣列是必要的。Client 識別碼是帶有路徑的絕對 HTTPS URL。可以包含 `application_type`，但它不是強制的 CIMD 欄位。不要把 DCR 對 `application_type` 的要求，複製到建議的 CIMD 路徑。

規格直言不諱的兩個安全事實：

- **SSRF。** Authorization server 會取得由攻擊者提供的 URL。它必須防禦 server-side request forgery（不要取得內部或管理 endpoint）。
- **localhost 冒充。** 單靠 CIMD 無法阻止本地攻擊者宣稱擁有合法 client 的 metadata URL，並綁定任何 `localhost` 的 redirect。Authorization server **必須**在同意畫面清楚顯示 redirect URI 的 hostname，並**應該**對僅有 `localhost` 的 redirect 發出警告。

因為 CIMD 不需要 server 端的狀態，所以不必像 DCR 那樣建立一個註冊器。Client 這邊是唯讀的：從靜態的 HTTPS endpoint 提供你的 metadata 文件，讓 authorization server 去拉。

如果 authorization server 的營運者已經配置好 client 識別碼，請在嘗試自動註冊之前，先使用該 issuer 範圍的註冊。否則優先使用 CIMD。只有當 issuer 既不能使用預先註冊、也不能使用 CIMD 時，才使用已棄用的 DCR。

### RFC 7591：已棄用的相容性註冊

DCR 在 2026-07-28 revision 中已被棄用。只在 authorization server 無法使用 CIMD、且預先註冊不切實際時保留它。相容性 client 會 POST：

```json
POST /register
Content-Type: application/json

{
  "application_type": "native",
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "response_types": ["code"],
  "token_endpoint_auth_method": "none",
  "scope": "mcp:tools.invoke",
  "client_name": "Cursor",
  "software_id": "com.cursor.cursor",
  "software_version": "0.42.0"
}
```

Server 回應 `client_id` 與供之後更新用的 `registration_access_token`：

```json
{
  "client_id": "c_3e7f1a",
  "client_id_issued_at": 1769472000,
  "redirect_uris": ["http://127.0.0.1:7333/callback"],
  "grant_types": ["authorization_code", "refresh_token"],
  "registration_access_token": "regt_b2...",
  "registration_client_uri": "https://auth.example.com/register/c_3e7f1a"
}
```

`application_type` 不是裝飾。Loopback 的桌面 client 宣告 `native`；由 server 託管的 client 宣告 `web`，並使用 HTTPS 的 redirect URI。`token_endpoint_auth_method: none` 是 public native client 的正確預設。它只拿到 `client_id`，由 PKCE 提供持有證明。

三個 production 陷阱：

- 註冊 endpoint 必須依來源 IP 做 rate limit。沒有的話，惡意行為者會用腳本產生數百萬筆假註冊，耗盡 `client_id` namespace。在註冊器處理請求之前，先執行 rate-limit 檢查。
- 部分企業 IdP 要求 `software_statement`（為 client 背書的簽章 JWT）。這堂課的 mock 略過它；production 要接上驗證步驟，拒絕除了 localhost redirect URI 之外、任何未簽章的註冊。
- `registration_access_token` 必須以 hash 儲存，而不是明文。這個 token 被偷，代表攻擊者可以改寫 client 的 redirect URI。

### RFC 8707（回顧）：Resource Indicators

課程 16 建立了它的形狀。Production 的規則：每個 token 請求都包含 `resource=<canonical-mcp-url>`，而 MCP server 在每次 call 都驗證 `token.aud` 與自己的 resource URL 相符。Canonical URI 是 server *最具體*的識別碼：使用小寫的 scheme 與 host，沒有 fragment，慣例上也沒有結尾斜線。依規則，路徑元件**不會**被剝除，當需要它來識別個別 MCP server 時，規格會保留它。`https://mcp.example.com`、`https://mcp.example.com/mcp`、`https://mcp.example.com:8443` 和 `https://mcp.example.com/server/mcp` 都是有效的 canonical URI。每個 server 選一個，並把 `aud` 精確固定在它上面。（為了簡潔，這堂課的 mock 使用單純 host 的 audience，例如 `https://notes.example.com`；在同一個 origin 下共同託管多個 MCP server 的部署，則以路徑區分它們。）

### RFC 7636（回顧）：PKCE

在 OAuth 2.1 中，PKCE 是強制的。這堂課的 authorization-code 流程一律帶有 `code_challenge` 和 `code_verifier`。Server 會拒絕沒有 verifier 的 token 請求，或 verifier 的 hash 與儲存的 challenge 不符的請求。

### MCP 2026-07-28 授權 profile

目前的 MCP revision 保留了 OAuth resource-server 的邊界，同時讓 MCP transport 變成無狀態。沒有可以用來快取身分決定的 protocol session。因此授權層要獨立驗證每個請求：

- 實作 RFC 9728 的 protected-resource metadata，並透過 401 上的 `WWW-Authenticate: Bearer resource_metadata="..."` header，**或**透過 well-known URI `/.well-known/oauth-protected-resource` 提供其位置（SEP-985 讓 header 成為選擇性的，並以 well-known 作為 fallback）。Metadata 的 `authorization_servers` 欄位**必須**指出至少一個 server。
- 只在**每一個**請求上，透過 `Authorization: Bearer ...` 接受 token，絕不放在 query string，也絕不只在 session 開始時驗證。
- 對每個請求驗證 `aud`、`iss`、`exp` 與必要的 scope。Server **必須**驗證 token 是專門簽發給自己的（audience）；缺少或不一致的 `aud` 會被拒絕，絕不視為萬用字元。
- 對 401/403，回傳 `WWW-Authenticate: Bearer`，帶有 `error=...`、`resource_metadata="<PRM-URL>"` 參數（metadata 文件的 URL，*不是*單純的 resource），以及在 `insufficient_scope`（403）時的 `scope="..."`。注意：參數是 `resource_metadata`，這是 discovery 的指標，challenge 中沒有 `resource` 參數。
- Authorization-server discovery 接受 RFC 8414 的 OAuth metadata，**或** OpenID Connect Discovery 1.0；client 必須依優先順序嘗試兩種 well-known 後綴。
- 由 client（而不是 server）防禦 **mix-up 攻擊**：它在重新導向前記錄預期的 `issuer`，並在兌換 code 之前，驗證實際 authorization response 中回傳的 `iss` 值（RFC 9207）。單靠 PKCE 無法阻止 mix-up，因為 client 會把它的 `code_verifier` 交給它被導向的任何 token endpoint。
- Client 憑證屬於一個 authorization-server issuer。如果 discovery 解析到不同的 issuer，client 要重新註冊，而不是出示舊的 `client_id`、registration token 或 access token。
- CIMD 是建議的註冊機制。DCR 已被棄用；相容性的 DCR 請求仍要宣告正確的 `application_type`。

OAuth 2.1 draft 是基底；RFC 8414/7591/8707/9728/9207 + RFC 7636 + CIMD 是介面；MCP 規格則是 profile。

### 部署 capability 檢查清單

廠商的功能表很快就過時。請檢查你實際要部署的 authorization server 所回傳的 metadata。這道關卡是機械式的：

| 檢查 | 必要的決定 |
|---|---|
| 探索到的 issuer | 政策預期的確切 HTTPS issuer |
| PKCE | 公告 `S256`；否則停止 |
| 註冊 | 建議 CIMD，接受預先註冊，DCR 僅作為已棄用的相容性 |
| Authorization response | 存在或有公告時，驗證 RFC 9207 的 `iss` |
| Resource 綁定 | Token 請求帶有 `resource`；resource server 要求相符的 `aud` |
| 憑證儲存 | 以 issuer 為 key 管理 client ID 與註冊憑證；以 issuer 加 resource 為 key 管理 access token |
| DCR 相容性 | 宣告 `native` 或 `web`；拒絕不符合所宣告 application type 的 redirect URI |

不要從產品名稱或價格層級推斷支援狀況。把探索到的文件擷取為部署證據，並在缺少強制欄位時 fail closed。

### JWKS 重新整理模式（在 AS 輪換，在 resource server 重新整理）

把兩個動詞分開，因為把它們混為一談是真實的 production bug：

- **輪換（Rotate）** 是 *authorization server* 做的事：產生新的簽章 key、發布到 JWKS，稍後淘汰舊的。Resource server 不參與，也做不到，它沒有 IdP 的私鑰。
- **重新整理（Refresh）** 是 *resource server* 做的事：重新 `GET` 已發布的 JWKS 到它的快取中。這是 resource server 唯一會執行的 JWKS 動作。

Production 的失敗模式是過期的快取。用排程的重新整理 job 加上 key-value 快取來解決。Resource server 執行一個 job（cron、計時器，或你的 runtime 提供的任何東西），以固定的間隔取得 `<issuer>/.well-known/jwks.json` 並覆寫 `cache[issuer] = {keys, fetched_at}`。Validator 從該快取讀取。`kid` 不在快取中的 token，會觸發**一次**同步的重新整理作為 fall-back，然後重新檢查。這同時處理兩種情況：排程的重新整理，以及 key 重疊視窗中，由全新 key 所簽的 token 在下一次排程重新整理之前就到達。

這個 fall-back **必須是重新取得（re-fetch），絕不能是輪換（rotate）**。如果你把快取 miss 的路徑接成輪換並產生新 key，有兩件事會壞掉：(1) 產生新的 key 會得到一個*仍然*與 token 不符的 `kid`，所以查找還是失敗；(2) 用隨機 `kid` 值灑出 token 的攻擊者，會迫使系統無限地建立 key，這是自找的 DoS。重新取得是冪等的，所以偽造的 `kid` 頂多浪費一次取得。

快取的形狀：

```json
{
  "https://auth.example.com": {
    "keys": [
      {"kid": "k_2026_03", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"},
      {"kid": "k_2026_04", "kty": "RSA", "n": "...", "e": "AQAB", "alg": "RS256", "use": "sig"}
    ],
    "fetched_at": 1772668800
  }
}
```

同時有兩個 key 是穩定狀態。Authorization server 的輪換方式是先引入下一個 key（`k_2026_04`），再淘汰前一個（`k_2026_03`），所以用舊 key 簽發的 token 在過期前仍然有效。快取保存聯集；validator 依 `kid` 挑選。

### 驗證例行程序 (The validation routine)

MCP server 在分派任何 tool 之前執行驗證。`code/main.py` 使用的形狀：

```python
result = server.validate(bearer_token, required_scope="mcp:tools.invoke")
if not result["valid"]:
    return {"status": result["status"], "WWW-Authenticate": result["www_authenticate"]}
```

`validate` 解碼 JWT，從 JWKS 快取解析簽章 key（miss 時重新整理一次），驗證簽章，然後依序檢查 `iss` 是否在 allow-list 中、`aud` 是否為此 server 的 canonical resource、`exp` 以及必要的 scope，並在第一個失敗時回傳 `WWW-Authenticate` challenge。把它保持為 resource server 上的單一例行程序，意味著每個進入點（每個 tool call、每個 transport）都會經過相同的檢查；沒有任何路徑可以不經驗證就到達 tool。

### Opaque token 使用 introspection，而不是猜測

並非每個 access token 都是 JWT。如果 issuer 文件說明它是 opaque token，resource server 無法把它解碼成值得信任的 claim。它會透過經過驗證的 backchannel，把 token 送到 issuer 的 RFC 7662 introspection endpoint，並要求 `active: true`、預期的 issuer context、確切的 MCP audience 或 resource、未過期的時間 claim，以及具體 tool 所需的 scope。

依 issuer、單向的 token digest 與 MCP resource 為 key 快取 introspection。絕不要把明文 token 當作 log 或快取的標籤。正向快取項目的上限，是 token 到期時間、issuer 快取指引，以及部署的撤銷新鮮度目標三者中最早的一個。負向快取要保持夠短，讓新簽發的 token 不會一直被錯誤地視為 inactive。針對某個 resource 的結果，無法授權另一個 resource，即使 opaque token 字串完全相同。

不要從攻擊者控制的 token 內容選擇驗證模式。把 JWT 或 introspection 的行為，固定到經過驗證的 issuer metadata 與部署設定。在 JWT 路徑上，固定接受的演算法與受信任的 `jwks_uri`；絕不要跟隨僅由 token header 選定的 key URL 或演算法。

### 撤銷是一份新鮮度合約

RFC 7009 讓 client 能請求 authorization server 撤銷 token。該請求不會抹除每個 resource server 已快取的副本。定義最大可接受的撤銷延遲，並讓每個快取都遵守它。

Opaque-token 的部署可以透過在每個高風險 call 上做 introspection，或使用短暫的正向快取，達到更緊的撤銷。自包含的 JWT 部署通常結合短的 access-token 存活期、refresh-token 撤銷、針對 issuer 全面事件的 key 淘汰，以及用於緊急本地拒絕的選擇性 subject、session 或 token-id denylist。除非 resource server 有目前的外部撤銷證據，否則簽章過的 JWT 在到期前，密碼學上都仍然有效。

登出、帳號停用、同意撤回與事件應變是不同的觸發條件，但必須收斂到同一個可量測的陳述：最多在宣告的撤銷視窗之後，每個 replica 都拒絕該憑證。請透過 load balancer 測試這個陳述，而不是只對一個暖機的 process 測試。

### 相依元件失敗需要宣告好的決定 (Dependency failure needs a declared decision)

絕不要在 exception handler 裡臨時發明可用性政策。

| 失敗 | 安全的 production 行為 |
|---|---|
| 排程的 JWKS 重新整理失敗，已知的 `kid` 仍在有效且有界限的快取中 | 只在宣告的 stale-on-error 視窗內繼續，並發出降級的健康證據 |
| Token 的 `kid` 未知，而允許的那一次重新整理也失敗 | 拒絕；絕不接受無法驗證的簽章 |
| Introspection 不可用 | 對受保護的 call fail closed；不要把網路失敗轉成 `active: true` |
| Protected-resource 或 issuer metadata 意外改變 | 停止新的註冊與 token 取得；只在有界限的事件政策下，保留明確固定且未過期的設定 |
| 撤銷 endpoint 不可用 | 回報登出或撤銷未完成，盡可能在本地把憑證保留為不可用，並且不要宣稱全域撤銷已成功 |
| 時鐘來源或 claim 型別無效 | 拒絕，而不是放寬 skew 直到 token 通過 |

把失敗與無效的憑證分開分類。相依元件的中斷是帶有健康狀態與重試政策的營運錯誤。簽章、issuer、audience、到期或 scope 錯誤，則是授權拒絕。兩者都不會到達 tool handler，也都不應該讓 token 內容洩漏到稽核證據中。

### Audience 重放的逐步說明（access-token 權限限制）

Server A（`notes.example.com`）和 Server B（`tasks.example.com`）都向同一個 authorization server 註冊。Server A 被入侵。攻擊者拿走使用者的 notes token，並重放到 Server B。

Server B 的 validator：

1. 解碼 JWT，依 `kid` 取得 JWKS，驗證簽章。
2. 把 `iss` 與它的 protected-resource metadata 的 `authorization_servers` 比對。（通過，同一個 IdP。）
3. 檢查 `aud == "https://tasks.example.com"`。（失敗，token 的 `aud` 是 `https://notes.example.com`。）
4. 回傳 401 與 `WWW-Authenticate: Bearer error="invalid_token", error_description="audience mismatch", resource_metadata="https://tasks.example.com/.well-known/oauth-protected-resource"`。

在 protocol 層，audience claim 是防禦這種攻擊的唯一手段。為了效能而跳過它，是最常見的 production 錯誤；validator 必須在每個請求上執行，而不只是在 session 開始時。規格把這稱為 **access-token privilege restriction**：MCP server `必須`拒絕任何 audience 中沒有指名自己的 token。

> **命名說明。** 規格把 *confused deputy* 一詞保留給一個相關但不同的問題：MCP server 作為第三方 API 的 OAuth **proxy**，使用靜態的 client ID，並在沒有取得每個 client 的使用者同意的情況下轉發 token。Audience 綁定修正的是上述的重放；confused deputy 的修正，則是每個 client 的同意，**加上**絕不把進來的 token 傳遞給上游 API（MCP server `必須`取得自己獨立的上游 token）。

### Mix-up 攻擊（server 無法提供的 client 端防禦）

Client 一生中會與許多 authorization server 溝通。惡意的 AS 可以嘗試讓 client 把誠實 AS 的 authorization code，兌換到攻擊者的 token endpoint。Audience 綁定在這裡沒有幫助，因為攻擊發生在任何 token 存在之前。防禦位於 client（RFC 9207）：

1. 在重新導向之前，client 從已驗證的 AS metadata 記錄預期的 `issuer`。
2. 收到 authorization response 時，client 在把 code 送到任何地方之前，先把回傳的 `iss` 參數與記錄的 issuer 比對（單純的字串比較，不做正規化）。
3. 不一致（或 AS 已公告 `authorization_response_iss_parameter_supported` 卻缺少 `iss`）→ 拒絕，甚至不要顯示 `error` 欄位。

單靠 PKCE 無法阻止 mix-up，因為 client 會把它的 `code_verifier` 交給它被導向的任何 token endpoint。這就是為什麼規格會把每個請求的 issuer，與 PKCE verifier 和 `state` 一起記錄。

### 失敗模式 (Failure modes)

- **過期的 JWKS。** AS 輪換 key 之後，validator 拒絕有效的 token。修正方式是上面的 cron 重新整理加上 cache-miss 重新取得模式。絕不要在沒有重新整理 job 的情況下快取 JWKS。
- **以輪換作為 fall-back。** 把 cache-miss 的路徑接成輪換並產生 key，而不是重新取得，是真實的 bug：它永遠不會產生缺少的 `kid`，還會把攻擊者控制的 `kid` 值變成 key 建立的 DoS。Fall-back 必須是冪等的 `refresh-jwks`。
- **缺少 `aud` claim。** 有些 IdP 預設會省略 `aud`，除非 token 請求中帶有 `resource`。Validator 必須拒絕缺少 `aud` 的 token，而不是把缺席視為萬用字元。
- **缺少 `iss` 檢查造成的 mix-up。** Client 如果沒有把 RFC 9207 的 `iss` authorization-response 參數，與重新導向前記錄的 issuer 比對驗證，就可能被導向，把誠實 AS 的 code 兌換到攻擊者的 token endpoint。這是 client 端的失敗；resource server 無法彌補。
- **Scope 升級 race。** 同一個使用者的兩個並行 step-up 流程可能都成功，並產生兩個 scope 不同的 access token。Validator 必須使用請求上出示的 token，而不是查找「使用者目前的 scope」，那會造成 TOCTOU 視窗。
- **Registration token 被偷。** 洩漏的 `registration_access_token` 讓攻擊者可以改寫 redirect URI。靜態儲存時要 hash；要求 client 在每次更新時出示明文；懷疑時就輪換。
- **`iss` 沒有固定。** 接受任何 `iss` 的 validator，會讓攻擊者架設自己的 authorization server、為目標 audience 註冊 client 並簽發 token。Protected-resource metadata 的 `authorization_servers` 清單就是 allow-list；請強制執行它。
- **憑證或 token 快取衝突。** 只以 resource 為 key 儲存註冊的 client，可能把一個 authorization server 的身分出示給另一個。只以 issuer 為 key 儲存 access token 的 client，可能把 token 重放到錯誤的 audience。以已驗證的 issuer 為 key 管理註冊，以 `(issuer, resource)` 為 key 管理 access token，並在 issuer 改變時一律重新註冊。

```figure
t3-jwks-rotate
```

## 實際使用 (Use It)

`code/main.py` 用 stdlib Python 走過完整的 production 流程，包含三個角色：`AuthorizationServer`、`ResourceServer` 和 `Client`。流程如下：

從 repository 根目錄執行：

```bash
cd phases/13-tools-and-protocols/18-mcp-auth-production
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

第一個指令會印出綁定 issuer 的註冊與 token 驗證 transcript。第二個回報十八項通過的檢查。兩個指令都不會開啟網路 listener，也不會寫入憑證。

1. Authorization server 在 `/.well-known/oauth-authorization-server` 發布 RFC 8414 metadata。
2. MCP client 呼叫 metadata endpoint，並檢查它的註冊選項（CIMD 用 `client_id_metadata_document_supported`，DCR 用 `registration_endpoint`）以及 `S256` PKCE 支援。
3. Client 檢查是否有 issuer 範圍的預先註冊，否則用它的 HTTPS Client ID Metadata Document 註冊。已棄用的 DCR 仍是可獨立測試的相容性方法。
4. Client 記錄已驗證的 issuer，建立 S256 challenge，收到一次性的 authorization code 加上 `iss`，驗證回傳的 issuer，然後用原本的 verifier 與 RFC 8707 的 `resource` indicator 兌換 code。
5. MCP client 帶著 `Authorization: Bearer ...` 呼叫 MCP server 上的 tool。
6. MCP server 執行 `validate`，從 JWKS 快取解析簽章 key。
7. IdP 輪換 key；排程的重新整理把 JWKS 重新拉進快取。
8. 下一次 call 用重新整理後的 key 驗證，不需重啟，而先前的 token 在重疊視窗內仍然通過驗證。
9. 對不同 MCP resource 的 audience 重放嘗試，會得到 401、`audience mismatch` 以及 `resource_metadata` 指標。

這裡的 JWT 使用帶共用機密的 HS256（讓這堂課只靠 stdlib 就能執行）。Production 使用 RS256 或 EdDSA 搭配上述的 JWKS 模式；其餘的驗證邏輯相同。由於 IdP 和 resource server 在同一個 process 中，`refresh_jwks` 直接讀取 authorization server 的 key 清單；透過網路時，它是對 `jwks_uri` 的 HTTP `GET`。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-mcp-auth.md`。給定 MCP server 設定與 IdP 的 capability 集合，這個 skill 會輸出要架設的 auth 介面：protected-resource metadata、要使用的註冊路徑（CIMD、預先註冊或 DCR fallback）、JWKS 重新整理排程、scope 對應，以及當 IdP 不支援完整 RFC profile 時要套用的拒絕規則。

## 練習 (Exercises)

1. 執行 `code/main.py`。追蹤整個流程。注意 IdP 在腳本輸出的第 6 步（上述流程的第 7 步）輪換 key，排程的 `refresh_jwks` 重新拉取已發布的集合，而舊 token（重疊視窗）與新 token 都在不重啟的情況下通過驗證。

2. 在 protected-resource metadata 的 `authorization_servers` 清單中加入新的 IdP。簽發一個由新 IdP 簽章的 token，並確認 validator 接受它。簽發一個由未列出的 IdP 簽章的 token，並確認 validator 以 `WWW-Authenticate: Bearer error="invalid_token", error_description="iss not allowed"` 拒絕。

3. 為 `register_client` 加入 rate-limit 檢查，在註冊器接受請求之前執行。使用以 IP 為 key、放在小 dict 裡的每來源 IP token bucket。

4. 閱讀 RFC 7591，找出這堂課的 `/register` handler 沒有驗證的兩個欄位。加入驗證。（提示：`software_statement` 與 `redirect_uris` 的 URI scheme。）

5. 新增第二個 authorization server。確認 client 儲存獨立的、以 issuer 為 key 的註冊資料，並拒絕重用第一個 issuer 的 token 或 `client_id`。

6. 證明 DoS 的修正。送給 validator 一個帶隨機 `kid` 的 token，確認 `refresh_jwks` 最多執行一次，且 authorization server 的 key 數量沒有增加。然後刻意把 fall-back 重接成輪換並產生 key，觀察每個偽造 token 都讓 key 數量增加，之後再恢復重新取得的做法。

7. 用 `native` 與 `web` 兩種 client 測試已棄用的 DCR。確認使用 HTTP redirect URI 的 web client，以及沒有精確 loopback redirect 的 native client 都會被拒絕。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| ASM | 「OAuth metadata 文件」 | RFC 8414 的 `/.well-known/oauth-authorization-server` JSON |
| CIMD | 「Client metadata URL」 | Client ID Metadata Document：用作 `client_id` 的 HTTPS URL；由 AS 拉取 JSON。是 MCP 2026-07-28 中建議的註冊方式 |
| DCR | 「自助式 client 註冊」 | RFC 7591 的 `POST /register`；對目前的 MCP 已棄用，僅為相容性保留 |
| JWKS | 「驗證 JWT 的公鑰」 | JSON Web Key Set，從 `jwks_uri` 取得，以 `kid` 為索引 |
| Rotate vs refresh | 「更新 key」 | *Rotate* = AS 產生並淘汰簽章 key；*refresh* = resource server 重新取得已發布的集合。Resource server 永遠只做 refresh |
| Resource indicator | 「Audience 參數」 | RFC 8707 的 `resource` 參數，把 token 固定到一個 server |
| `aud` claim | 「Audience」 | validator 與 canonical resource URL 比對的 JWT claim |
| Audience replay | 「Token 重放」 | 為 Server A 簽發的 token 被出示給 Server B；以 audience 驗證防禦（規格：access-token privilege restriction） |
| Confused deputy | 「Proxy token 誤用」 | 帶靜態 client ID 的 MCP proxy 在沒有每個 client 同意的情況下轉發 token；與 audience replay 不同 |
| Mix-up attack | 「錯誤的 token endpoint」 | Client 被導向把誠實 AS 的 code 兌換到攻擊者的 endpoint；以 RFC 9207 的 `iss` 在 client 端防禦 |
| `iss` allow-list | 「受信任的 authorization server」 | protected-resource metadata 的 `authorization_servers` 中列出的集合 |
| `resource_metadata` | 「PRM 文件在哪裡」 | 在 401/403 時，指出 RFC 9728 metadata URL 的 `WWW-Authenticate` 參數 |
| Public client | 「原生或瀏覽器 client」 | 沒有 `client_secret` 的 OAuth client；由 PKCE 補償 |
| `WWW-Authenticate` | 「401/403 response header」 | 攜帶驅動 client 復原的 `Bearer error=...` 指示 |

## 延伸閱讀 (Further Reading)

- [MCP authorization specification (2026-07-28)](https://modelcontextprotocol.io/specification/2026-07-28/basic/authorization) - 目前的 MCP 授權 profile
- [MCP 2026-07-28 changelog](https://modelcontextprotocol.io/specification/2026-07-28/changelog) - CIMD、issuer 驗證、DCR 棄用與以 issuer 為 key 的憑證變更
- [OAuth Client ID Metadata Document (draft-ietf-oauth-client-id-metadata-document-00)](https://datatracker.ietf.org/doc/html/draft-ietf-oauth-client-id-metadata-document-00) — CIMD
- [RFC 8414 — OAuth 2.0 Authorization Server Metadata](https://datatracker.ietf.org/doc/html/rfc8414) — discovery 合約
- [RFC 7591 — OAuth 2.0 Dynamic Client Registration Protocol](https://datatracker.ietf.org/doc/html/rfc7591) — DCR（fallback 路徑）
- [RFC 7636 — Proof Key for Code Exchange (PKCE)](https://datatracker.ietf.org/doc/html/rfc7636) — public client 的持有證明
- [RFC 8707 — Resource Indicators for OAuth 2.0](https://datatracker.ietf.org/doc/html/rfc8707) — audience 固定
- [RFC 9728 — OAuth 2.0 Protected Resource Metadata](https://datatracker.ietf.org/doc/html/rfc9728) — resource server 的 discovery
- [RFC 9207 — OAuth 2.0 Authorization Server Issuer Identification](https://datatracker.ietf.org/doc/html/rfc9207) — 防禦 mix-up 攻擊的 `iss` 參數
- [RFC 7662: OAuth 2.0 Token Introspection](https://datatracker.ietf.org/doc/html/rfc7662)
- [RFC 7009: OAuth 2.0 Token Revocation](https://datatracker.ietf.org/doc/html/rfc7009)
