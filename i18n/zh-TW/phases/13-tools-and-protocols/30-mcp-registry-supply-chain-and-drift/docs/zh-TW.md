# MCP Registry 供應鏈：接納、漂移與 Rollback (MCP Registry Supply Chain: Admission, Drift, and Rollback)

> Registry 紀錄告訴你發布者宣告了什麼。Production 的接納（admission）則證明你取得了什麼、觀察到什麼、核准了什麼，以及你能安全地還原什麼。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 13 · 17 (gateways and registries), Phase 13 · 18 (production authentication)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 把 Registry 發布、套件出處、runtime discovery 與本地核准分開。
- 驗證 MCP server 的 namespace，而不信任其自身紀錄中的名稱。
- 固定不可變的發布、執行來源、出處與即時 descriptor 證據。
- 偵測接納之後的 registry 狀態變更與 runtime 漂移。
- 把 routing rollback 到先前已接納的版本，而不改寫歷史。
- 維護防竄改的接納帳本，說明每一個決定。

## 問題 (The Problem)

你在 registry 中找到 `com.example/inventory`。它的 description 看起來沒問題。它的套件存在。Server 回應了 `server/discover`。

這不是單一個事實。它是來自不同權威的一串事實：

1. 對某個 namespace 已通過驗證的發布者，提交了一筆紀錄。
2. 套件 registry 提供了具有特定 identity 與 digest 的 artifact。
3. 執行中的 endpoint 回報了 protocol version、capabilities、tool 與診斷用的 server 資訊。
4. 你的組織決定，這個確切的組合是被允許的。

把這些事實壓縮成「它在 registry 裡，所以信任它」，會製造供應鏈的盲點。有效的發布仍然可能被棄用。如果你沒有固定它的 digest，套件 tag 可能指向意外的 artifact。Server 可能在審查之後新增具破壞性的 tool。Rollback 可能默默選了一個從未被接納的版本。

解決方式是在每個邊界都有證據的接納控制器。

## Registry 是索引，不是你的核准系統 (The Registry Is an Index, Not Your Approval System)

官方的 MCP Registry 儲存 server 的 metadata。它的 `server.json` 紀錄指出 server 的版本，並宣告一個或多個套件或遠端 endpoint。發布規則加入了 namespace 驗證、套件擁有權檢查、受限的 registry 規則，以及狹窄的發布者 metadata 位置。

這些控制回答的是發布的問題。你的 production 政策仍然要回答部署的問題：

| 邊界 | 問題 | 證據的擁有者 |
|---|---|---|
| Namespace | 發布者是否被允許使用這個名稱？ | Registry 驗證，加上你已驗證的 namespace 輸入 |
| 紀錄 | 發布者為這個版本宣告了什麼？ | 不可變的 `server.json` digest |
| 執行來源 | 哪個套件或遠端 endpoint 會執行？ | 宣告的來源欄位、已驗證的擁有權結果、transport 與受信任的 digest |
| Runtime | endpoint 現在公開了什麼？ | `server/discover` 與 tool descriptor |
| 接納 | 你的政策是否核准了這個確切的集合？ | 本地 pin 與帳本項目 |
| 營運 | 它仍然安全嗎？什麼可以取代它？ | 漂移檢查、狀態同步、健康檢查與 rollback 路由 |

Registry 的 schema 版本與 MCP protocol 版本是互相獨立的。紀錄可以使用已發布的 `2025-12-11` server schema，而即時的 server 支援 MCP `2026-07-28`。絕不要從其中一個推斷另一個。

```figure
mcp-registry-admission
```

## 一個接納決定中的七項控制 (Seven Controls in One Admission Decision)

### 1. Namespace 驗證

官方 Registry 的名稱使用經過驗證的 namespace。已驗證的 domain 可以對應到反轉的 domain 前綴。例如，控制 `example.com` 可以建立 `com.example/*`。

不要接受字串前綴檢查：

```python
server_name.startswith("com.example")
```

那也會接受 `com.exampleevil/tool`。在 `/` 處切開名稱，要求 slug 非空，並精確比對 namespace 片段。更重要的是，從驗證結果把已驗證的 namespace 傳入 admission。不要從不受信任的紀錄推導出信任。

以 GitHub 為基礎的 namespace 與以 domain 為基礎的 namespace，使用不同的驗證路徑。把任一條路徑正規化成單一個 admission 輸入：確切的已驗證 namespace 字串。

### 2. 出處結合 (Provenance join)

對於套件紀錄，宣告與取得的 artifact 必須在明確的欄位上結合：

- 套件 registry 類型
- 套件識別碼
- 套件版本
- 已驗證的擁有權結果
- 下載的 artifact digest

也要驗證宣告的套件 transport。只有遠端 endpoint 的紀錄是有效的，不應因為缺少套件而被拒絕。對於遠端來源，把宣告的 URL 與 transport 類型，結合獨立驗證過的 endpoint 擁有權，以及受信任的連線或部署證據的 digest。

這堂課的程式碼支援兩種來源類型，並把選定的來源與 Registry 來源、server 名稱、Registry 版本、紀錄 digest 和證據 digest 一起 hash。得到的出處 digest 是指向完整證據集合的精簡指標。它不能取代保留證據。

絕不要接受只由你正在驗證的 artifact 所提供的 digest。在受信任的取得邊界計算它，或從你驗證其驗證結果的套件服務接收它。

### 3. 固定決定，而不只是版本

Registry 版本是唯一的發布識別碼。已發布的 metadata 是不可變的。變更的紀錄需要新的版本。建議使用語意化版本，但 Registry 並不要求，也不接受版本範圍。

這表示 `^1.4` 不是 admission pin。「latest」也不是。有用的 pin 包含：

```json
{
  "server": "com.example/inventory",
  "version": "1.0.0",
  "recordDigest": "...",
  "source": {"kind": "package", "registryType": "pypi"},
  "sourceDigest": "...",
  "toolsetDigest": "...",
  "provenanceDigest": "...",
  "registryStatus": "active"
}
```

固定多個層次，讓你能辨識是哪個邊界改變了。相同 Registry 版本下的紀錄 digest 改變，是 Registry 完整性失敗。相同套件座標或遠端部署下的來源 digest 改變，是執行來源完整性失敗。Toolset digest 改變則是 runtime 漂移。

### 4. 即時漂移偵測

Admission 應該觀察實際會收到流量的 server。呼叫 `server/discover`，透過你受信任的路徑列出或取得公開的 tool descriptor，並驗證：

- `2026-07-28` 在 `supportedVersions` 中
- 所有本地必要的 capabilities 都存在
- 每個 tool descriptor 都具備必要的 identity 與 schema 介面
- 在之後的檢查中，正規化後的 descriptor digest 與已接納的 pin 相符

選擇性的結果 `_meta["io.modelcontextprotocol/serverInfo"]` 值，是自行申報的顯示、log 與除錯 context。把它記錄為診斷證據，但絕不要用它來建立 namespace、套件擁有權、endpoint 擁有權、admission 或任何其他安全決定。位於 `_meta` 之外的直接 `serverInfo` 別名，不是合約欄位，不應被提升為診斷證據。

只正規化順序沒有意義的欄位。範例在 hash 之前，依穩定的名稱對 tool 清單排序，所以無害的清單順序變更不會造成漂移。它不會丟棄 descriptor 的欄位。新的 tool、變更的 schema、變更的 description 或新的 annotation，都會改變 pin。

範例把格式錯誤的 descriptor 與任何 descriptor digest 變更都視為漂移，隔離該 pin、移除它作用中的路由，並禁止該版本作為 rollback 目標。Production 政策可以只允許透過新的審查來接受編輯性變更，因為 description 會影響模型對 tool 的選擇。「外觀上的」metadata 也可能改變 agent 的行為。

### 5. Registry 狀態是即時狀態

Registry API 會在每筆 server 紀錄旁邊附上 response 層級的 `_meta` 物件。由 Registry 管理的欄位位於 `_meta["io.modelcontextprotocol.registry/official"]` 之下。把 response 的 `_meta` 物件傳給 admission，並讀取 `_meta["io.modelcontextprotocol.registry/official"].status`。直接的 `_meta.status` 值不是官方的 wire 形狀。不要把 response metadata 與發布紀錄自己的 `_meta` 混淆。狀態可以是：

- `active`：預設會回傳，且有資格被本地接納
- `deprecated`：仍可被探索並附帶警告，但不再是安全的自動選擇
- `deleted`：預設被隱藏，而其歷史紀錄仍可透過已刪除或增量的視圖取得

在接納之後同步狀態。如果 active 的版本變成 deprecated 或 deleted，就隔離它的 pin，並停止把新的工作路由給它。保留證據。從預設清單中刪除，並不是抹除你稽核軌跡的許可。

發布者提供的自訂 metadata，只屬於發布紀錄中的 `_meta.io.modelcontextprotocol.registry/publisher-provided`。由 Registry 管理的 response metadata 是分開的。不要讓發布者自己設定官方狀態。

### 6. Rollback 意味著還原路由

不可變的發布在 rollback 期間不會被編輯。Rollback 是選擇先前已接納、目前仍符合資格的 pin，並改變作用中的路由。

安全的目標必須：

1. 有完成的 admission 紀錄。
2. 在你的政策下，仍有 active 的 Registry 狀態。
3. 沒有被 runtime 或安全證據隔離。
4. 仍然解析到被固定的套件與即時 descriptor 集合。
5. 通過目前的健康檢查。

範例專注於前三項條件。真正的 reconciler 應該在啟用之前，重新取得套件並重新檢查即時 endpoint。

### 7. 附加接納帳本 (Append an admission ledger)

接納資料庫說明什麼是作用中的。帳本說明為什麼。

每個範例項目包含序號、時間、事件、server、版本、結果、原因、證據、前一個項目的 hash，以及它自己的 hash。改變較舊的結果，會破壞該項目與之後每個連結的驗證。

這是防竄改（tamper-evident），不是神奇的不可竄改（tamper-proof）。把定期的帳本 head 錨定在獨立的信任領域，例如已簽章的 release metadata 或一次寫入的儲存空間。限制誰可以附加。讓授權 token、套件憑證、tool 參數與私有 endpoint 資料不進入證據。

## 動手做 (Build It)

可執行的控制器在 `code/main.py`。它只使用 Python 標準函式庫。

從有限的 demo 開始：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
```

這個 demo 執行五個操作：

1. 接納 `1.0.0`，其 namespace、套件出處、protocol、capabilities 與 tool 都相符。
2. 接納 `1.1.0` 並讓它成為 active。
3. 在 runtime 觀察到非預期的 delete tool。
4. 觀察 `1.1.0` 的 Registry 狀態變成 `deprecated`。
5. 把 routing 還原到仍已接納的 `1.0.0` pin。

預期的形狀：

```json
{
  "admitted": [true, true],
  "driftAllowed": false,
  "rollbackAllowed": true,
  "activeVersion": "1.0.0",
  "ledgerValid": true
}
```

依以下順序閱讀實作：

1. `namespace_for_domain()` 與 `namespace_matches()` 建立精確的命名權威。
2. `digest()` 與 `normalized_tools()` 產生確定性的證據。
3. `RegistryAdmissionController.admit()` 結合發布、出處、runtime 與政策。
4. `check_live()` 將新的觀察與 pin 比較。
5. `observe_registry_status()` 隔離 Registry 狀態改變的版本。
6. `rollback()` 只啟用先前已接納且符合資格的目標。
7. `AdmissionLedger.verify()` 偵測對已記錄歷史的變更。

## 實際使用 (Use It)

把控制器放在 discovery 與 routing 之間：

```text
Registry sync -> artifact verifier -> live discovery -> admission controller -> route table
                                               |                 |
                                               v                 v
                                          evidence store    admission ledger
```

為這些工作使用獨立的 identity。Registry 同步的 worker 需要 metadata 的讀取權限。Artifact 驗證器需要套件取得權限。路由 reconciler 需要啟用已核准 pin 的權限。它們都不需要每一種憑證。

讓 rollout 狀態明確。「Approved」表示證據通過政策。「Active」表示路由目前選擇它。「Quarantined」表示它不能接收新的工作。「Superseded」表示另一個已接納的版本是 active。不要把這四種意義編碼在單一個布林值中。

在 `tools/list` 中公開 server 之前，先執行 admission。否則 client 可能在發布與政策評估之間的空隙中，探索到某個 tool。

## 互動實驗 (Interactive Lab)

你將一次觀察一個邊界失敗。

### 實驗 A：namespace 衝突

從 code 目錄開啟 Python shell：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift/code
python3 -q
```

然後執行：

```python
from main import namespace_matches
namespace_matches("com.example/inventory", "com.example")
namespace_matches("com.exampleevil/inventory", "com.example")
```

第一個結果是 `True`；第二個是 `False`。在本地把精確比對換成 `startswith`，並觀察為什麼第二個名稱會越過邊界。在繼續之前，還原成精確比對。

### 實驗 B：descriptor 漂移

```python
from main import *
times = iter(f"2026-08-21T12:00:{n:02d}+00:00" for n in range(10))
c = RegistryAdmissionController(clock=lambda: next(times))
meta = {OFFICIAL_META_KEY: {"status": "active"}}
c.admit(sample_record("1.0.0"), meta, "com.example", evidence_for("1.0.0"), sample_live("1.0.0"))
c.check_live("com.example/inventory", "1.0.0", sample_live("1.0.0", True))
```

檢查原因與路由狀態。套件與 Registry 紀錄沒有改變。改變的是 runtime 的 tool 介面，所以控制器隔離並停用了該 pin。這就是為什麼供應鏈控制在安裝之後必須持續。

### 實驗 C：狀態與 rollback

接納 `1.1.0`，把它標記為 deprecated，並嘗試兩個 rollback 目標：

```python
c.admit(sample_record("1.1.0"), meta, "com.example", evidence_for("1.1.0"), sample_live("1.1.0"))
c.observe_registry_status("com.example/inventory", "1.1.0", "deprecated")
c.rollback("com.example/inventory", "1.1.0", "unsafe retry")
c.rollback("com.example/inventory", "1.0.0", "restore known release")
c.ledger.verify()
```

被隔離的目標被拒絕。較早的 active pin 被接受。帳本仍然有效。

## 實作練習 (Practice Lab)

為控制器擴充雙人核准閘門。

需求：

- 將核准儲存為已簽章的證據參照，而不是 pin 中可變的名稱。
- 對包含 `destructiveHint: true` 的 tool 的 toolset，要求兩個不同的審查者 identity。
- 拒絕重複的審查者 identity。
- 當核准不完整時，在帳本中保留原始的 admission 嘗試。
- 為零個、一個、重複與兩個不同核准新增測試。
- 不要記錄簽章、憑證或完整的私有 tool 參數。

成功的意思是：在兩個 identity 都核准確切的紀錄、套件與 toolset digest 之前，破壞性的 tool 不能變成 active。

## 交付的成品 (Shipped Artifact)

這堂課會交付 `outputs/skill-mcp-registry-admission.md`。在審查新的 Registry 版本或調查漂移時，把它當作單一檔案、可重複使用的 runbook。它定義輸入、拒絕規則、證據套件、狀態對帳與 rollback 證明，而不依賴範例的 class 名稱。

## 驗證 (Verify It)

執行 demo 與確定性的測試套件：

```bash
cd phases/13-tools-and-protocols/30-mcp-registry-supply-chain-and-drift
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

驗證應該證明：

- 精確的 namespace 邊界會拒絕外觀相似的前綴
- 只有官方 namespace 下的 Registry 狀態，才能讓某個版本符合資格
- 未驗證或不一致的套件與遠端證據會被拒絕
- 發布者 metadata 無法冒充由 Registry 管理的 metadata
- tool 的順序被正規化，而不隱藏 descriptor 的變更
- 格式錯誤的套件與 tool 結構會安全地拒絕
- `serverInfo` 仍是診斷用途，絕不提供 admission 的權威
- descriptor 漂移會隔離、停用，並阻止 rollback 到該 pin
- 狀態變更會隔離 active 的 pin
- rollback 不能選擇被隔離或未知的版本
- 帳本被竄改時會被偵測

## Production 的失敗模式 (Production Failure Modes)

| 失敗 | 為什麼發生 | 必要的回應 |
|---|---|---|
| 名稱看起來有效，但 namespace 從未被驗證 | 政策信任了紀錄文字 | 拒絕，直到受信任的 namespace 驗證器提供確切的前綴 |
| 相同的套件座標回傳新的位元組 | 上游可變，或散布管道被入侵 | 停止啟用，保留兩個 digest，調查取得邊界 |
| 「Latest」在沒有審查的情況下改變 | 浮動的選擇逃出了 pin | 只解析確切已接納的版本與 digest |
| 核准之後出現新的 tool | Runtime 漂移，或是不同的部署 | 隔離路由，並擷取新的 descriptor 觀察 |
| Deprecated 的版本仍然 active | 狀態同步缺少或延遲 | 依排程並在啟用之前，對帳狀態 |
| 已刪除的紀錄從預設同步中消失 | Client 只請求了 active 的紀錄 | 使用增量或能感知刪除的對帳，並保留本地歷史 |
| Rollback 目標從未被接納 | 路由控制與核准狀態彼此脫節 | 拒絕 rollback，並對目標執行新的 admission |
| 攻擊者改寫所有項目後，帳本在本地仍驗證通過 | Hash 鏈沒有外部錨點 | 把已簽章的帳本 head 發布到獨立的信任領域 |
| 證據包含 bearer token 或 tool 參數 | Log 複製了整個請求 | 在收集時就遮蔽，只儲存最少的證明 |

## 營運規則 (Operational Rule)

發布回答「這個 identity 能發布這個名稱嗎？」接納回答「我們會執行這個確切的 artifact，並公開這個確切的行為嗎？」把這些決定分開，固定每一個結合，並讓 rollback 選擇證據，而不是憑記憶。

## 延伸閱讀 (Further Reading)

- [Official Registry server.json requirements](https://github.com/modelcontextprotocol/registry/blob/main/docs/reference/server-json/official-registry-requirements.md)
- [Official Registry OpenAPI contract](https://registry.modelcontextprotocol.io/openapi.yaml)
- [MCP 2026-07-28 server discovery](https://modelcontextprotocol.io/specification/2026-07-28/server/discover)
