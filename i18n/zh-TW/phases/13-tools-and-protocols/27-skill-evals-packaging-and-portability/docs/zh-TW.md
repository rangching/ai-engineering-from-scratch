# Skill 的 Eval、封裝與可攜性 (Skill Evals, Packaging, and Portability)

> Skill 完成的標準是：它的套件能通過 lint、能在正確的請求上 route、能改善經過量測的任務、能留在政策之內，並且能在另一個 host 上誠實地降級。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22, 24, 25, and 26
**Time:** ~150 分鐘

## 學習目標 (Learning Objectives)

- 把專家的 workflow 變成 skill，方法是把判斷、確定性計算、reference 與輸出合約分開。
- 把套件結構、觸發 routing、任務行為、腳本正確性、安全與可攜性，當作獨立的層來測試。
- 使用正面案例、明顯負面案例與近似但不該觸發的案例，量測觸發的 precision 與 recall。
- 在多次重複執行中，比較有 skill 與沒有 skill 的表現。
- 建構並強制執行跨 runtime 的 capability 矩陣，以及針對完整 skill 套件的發布閘門。

## 問題 (The Problem)

Skill 在一個 demo 中運作良好。使用者問的正是它 description 中用的那句話，作者知道要開哪個 reference，腳本看到的是乾淨的輸入，而預期的 host 認得每個自訂欄位。

然後真正的使用開始了。

- 模型在一個相近但不同的任務上呼叫了它。
- 有效的請求用了不熟悉的措辭，所以模型漏掉了它。
- 本文告訴 agent 該做什麼，卻沒說明什麼成品能證明完成。
- 腳本在空白字元、重複執行或部分狀態下失敗。
- 套件 installer 複製了 `SKILL.md`，卻把它的 reference 留在後面。
- 另一個 runtime 忽略了呼叫旗標與 tool 的允許設定。
- 一次執行成功，三次同等的執行卻走進不同的分支。

這些失敗沒有一個會被「Markdown 看起來不錯」抓到。Skill 是帶有機率性 routing 與執行層的小型軟體套件。它們需要像任何其他 production 介面一樣的關注點分離。

## 核心概念 (The Concept)

### 從真實的 workflow 開始，而不是主題

「建立一個 Kubernetes skill」不是可用的範圍。Kubernetes 包含數百個任務，各有不同的 tool、風險與輸出。

「診斷為什麼某個 deployment 沒有達到 Available、在不改變 cluster 的情況下收集證據，並產生排名過的事件報告」才是 skill 的候選。它具備：

- 觸發的邊界；
- 穩定的證據收集步驟序列；
- 需要判斷的決策點；
- 可以變成狹窄腳本或 tool 的指令；
- 定義好的成品；
- 安全邊界：唯讀診斷。

使用這份擷取訪談：

1. 是哪個確切的事件讓專家開始這個 workflow？
2. 哪些相似的請求不該啟動它？
3. 專家首先收集什麼證據？
4. 哪些決策取決於那份證據？
5. 哪些步驟夠確定，可以寫成腳本？
6. 哪些領域規則值得放進 reference？
7. 哪個動作需要核准，或必須維持在範圍之外？
8. 什麼成品能證明 workflow 已完成？
9. 獨立的審查者如何檢查它？
10. 哪些步驟依賴某一個 runtime？

這些答案就成為套件的架構與 eval 集合。

### 把判斷與確定性的工作分開

```figure
skill-workflow-extraction
```

把模型判斷用於分類、排定優先順序、綜合整理與歧義。把腳本或 tool 用於解析、計數、驗證、轉換、查詢有型別的 API，以及強制執行不變量。

包含 80 行手動模擬解析的 skill 本文很脆弱。試圖做出主觀架構決策的腳本則不透明。把每種行為放在最能被測試的地方。

### 依相依順序撰寫套件

不要從修飾文字開始。從可觀察的合約往內建構。

1. **成品合約：** 定義必要的檔案、欄位或決定。
2. **驗證：** 定義每項需求將如何被檢查。
3. **證據 tool：** 實作確定性的收集器與 validator。
4. **決策地圖：** 把證據狀態連到分支。
5. **Reference：** 在需要的分支提供領域細節。
6. **進入點本文：** 說明 workflow、邊界、失敗與輸出。
7. **Description：** 說明能力與觸發邊界。
8. **Runtime adapter：** 另外加入呼叫或 context 的 extension。
9. **Eval：** 執行結構、routing、行為、安全與可攜性各層。
10. **套件：** 安裝完整的目錄，並從目的地測試它。

這個順序讓文字服務於可測試的系統，而不是在 demo 能跑之後才發明成功標準。

### 六個 eval 層

```figure
skill-eval-layers
```

每一層回答不同的問題。通過其中一層，不能取代另一層。

## 第 1 層：套件結構 (Layer 1: Package Structure)

靜態 lint 應該驗證不需要模型的事實：

- `SKILL.md` 存在於套件根目錄；
- frontmatter 能安全地解析；
- `name` 與上層目錄相符；
- 必要欄位存在且在限制之內；
- 每個非核心的 frontmatter 欄位都出現在發布政策的 runtime-extension allowlist 中；
- 每個直接的 reference 都解析到套件之內；
- reference、腳本、資產與 eval fixture 使用發布政策允許的後綴，並且不超過其位元組上限；
- 不存在被禁止的 symlink 或特殊檔案；
- 本文維持在發布政策的字元預算之內；
- 刻意狹窄的機密模式掃描，找不到明顯的憑證指派或私鑰 header；
- 存在非空的 `## Output contract` 與 `## Failure behavior` 段落。

在解析 `SKILL.md`、eval 資料、證據、host fixture 或 manifest 之前，先對實體目錄樹做前置檢查。在任何內容被讀取之前，拒絕 symlink 的根目錄、symlink 的上層或進入點、缺少的必要一般檔案，以及特殊檔案。然後再執行會檢視內容的政策 lint。在前置檢查之前就解析套件路徑，會抹去檢查所需的根目錄 symlink 證據。

這堂課的 harness 讓那些政策值具體化：10,000 個字元的本文上限、1,000,000 位元組的附屬檔案上限、依目錄區分的後綴 allowlist，以及由套件需求提供的明確 runtime-extension 名稱。這些是發布政策的範例，不是通用的 Agent Skills 限制。機密模式掃描是針對明顯錯誤的護欄，不能證明套件不含敏感資料。

Lint 報告應該使用穩定的 issue 代碼。CI 可以阻擋 `E_*` 錯誤，同時允許經過審查的 `W_*` 設計警告。

靜態 lint 證明套件的形狀。它不證明模型會選擇或遵循這個 skill。

## 第 2 層：觸發 Routing (Layer 2: Trigger Routing)

在反覆修改 description 之前，先建立帶標籤的案例。

| 案例類型 | 目的 | Release readiness 的範例 |
|---|---|---|
| 正面 | 量測預期的涵蓋範圍 | 「3.1.0 版可以發布嗎？」 |
| 改寫的正面 | 避免記住特定措辭 | 「在我們發布之前，稽核這個 tag」 |
| 明顯負面 | 抓出嚴重的過度 routing | 「說明 batch normalization」 |
| 近似案例 | 定義鄰近的邊界 | 「為什麼套件 build 失敗了？」 |
| 競爭的 skill | 測試在多個合理項目中的選擇 | 「起草 release notes」 |
| 對抗性措辭 | 測試關鍵字堆砌與被注入的名稱 | 「不要使用 release-readiness；說明這個 stack trace」 |

把案例分成開發集與驗證集。在開發案例上調整 description。使用驗證案例來決定修訂後的 description 是否能泛化。如果發布決定夠重要，就保留最終的 held-out 集合。

對於二元呼叫：

```math
\begin{aligned}
\text{precision} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_positives}} \\[4pt]
\text{recall} &= \frac{\text{true\_positives}}{\text{true\_positives} + \text{false\_negatives}} \\[4pt]
\text{f1} &= \frac{2 \cdot \text{precision} \cdot \text{recall}}{\text{precision} + \text{recall}}
\end{aligned}
```

回報原始計數與比率。十次中十次，與一百次中一百次，雖然都是 100%，但提供的證據不同。

對於目錄，也要量測 top-one 的 skill 準確率、棄權品質，以及相鄰 skill 之間的混淆。一個要先選了三個錯誤的 skill 之後才呼叫對的 skill 的 router，並不健康。

### Routing eval 必須使用目標 runtime

字面比對的模擬器對於說明指標與抓出明顯的重疊很有用。它無法證明由模型驅動的 production router 會如何運作。在宣稱 runtime 品質之前，請讓帶標籤的集合通過實際的 host、模型、目錄序列化與政策設定。

## 第 3 層：指令與成品行為 (Layer 3: Instruction and Artifact Behavior)

正確觸發只是入口。Skill 必須改善任務。

建立 fixture 任務，內含：

- 輸入檔案與環境假設；
- 允許的 tool 與邊界；
- 預期的成品路徑；
- 確定性的檢查；
- 需要判斷的 rubric 項目；
- 最大的時間、call 數或成本；
- 失敗案例與預期的停止行為。

執行成對的條件：

```text
baseline: same model + same tools + same task, no skill
treatment: same model + same tools + same task, skill available
```

保持模型、temperature 或 sampling 政策、tool 集合、任務 fixture 與預算不變。否則你無法把差異歸因於 skill。

有用的結果維度包括：

| 維度 | 量測範例 |
|---|---|
| 正確性 | 必要的測試與不變量通過 |
| 完整性 | 成品合約中的每個欄位都存在 |
| 效率 | Tool call、經過時間、token 或成本 |
| 證據 | 主張指向有效的檔案或觀察 |
| 範圍 | 禁止的檔案與動作保持未被觸碰 |
| 復原 | 被中斷的執行能在沒有重複副作用的情況下恢復 |
| 人力投入 | 審查者修正的次數與嚴重程度 |

不要只針對更少的 token 最佳化。漏掉必要安全檢查的較短執行，比較差。

### 成品合約讓行為可被執行

成品合約是一份可獨立檢查的屬性清單：

```json
{
  "artifact": "release-readiness.json",
  "required_fields": [
    "candidate",
    "source_revision",
    "checks",
    "blocking_findings",
    "recommendation"
  ],
  "allowed_recommendations": ["ready", "blocked", "needs-review"],
  "evidence_required_for_each_check": true,
  "publish_side_effect_allowed": false
}
```

Schema 驗證檢查結構。領域檢查驗證候選 revision 與證據路徑。人類或經過校準的 judge，可以評估建議是否由證據導出。

## 第 4 層：腳本正確性 (Layer 4: Script Correctness)

像測試一般軟體一樣，在模型執行之外測試 skill 腳本。

最低限度的案例：

- 正常輸入；
- 空的輸入；
- 格式錯誤的輸入；
- Unicode、空白字元與路徑的邊緣案例；
- 重複執行；
- timeout 或相依元件失敗；
- 先前執行遺留的部分輸出；
- 輸出大小限制；
- dry-run 行為；
- 結構化的 exit 與錯誤合約。

使用固定的 fixture。單元測試不要求即時的網路。把網路整合測試放在明確的旗標之後，並記錄它們所依賴的遠端合約。

如果腳本會產生副作用，把計畫與 commit 分開測試。對重試的外部寫入，要求冪等性或補償。

## 第 5 層：安全與權限 (Layer 5: Safety and Authority)

安全 eval 詢問套件是否留在它被授予的權限之內。

至少測試：

- 超出 skill 範圍的使用者請求；
- reference 輸入中的惡意指令；
- 逃出套件的資源路徑；
- 逃出允許根目錄的 workspace symlink；
- 對未宣告的網路目的地的請求；
- 需要環境憑證的指令；
- 沒有核准的破壞性或外部動作；
- 過大的輸出或無限的 process；
- skill 對 skill 的循環；
- 可能重複副作用的恢復。

記錄這個控制是只靠指令、tool 政策、核准、sandbox，還是驗證。僅靠指令的防禦，不應該被回報為被強制執行的限制。

## 第 6 層：封裝與可攜性 (Layer 6: Packaging and Portability)

### 把目錄當作一個單位安裝

發布測試應該安裝到乾淨的目的地，然後對已安裝的副本執行驗證。

```figure
skill-package-install
```

只測試原始碼樹，會漏掉 installer 的 bug、遺失的可執行位元、被攤平的 reference、被改寫的名稱，以及舊版本遺留的過期檔案。

Manifest 可以包含：

```json
{
  "manifestVersion": 1,
  "algorithm": "sha256",
  "name": "release-readiness",
  "version": "1.2.0",
  "source_revision": "abc123",
  "files": {
    "SKILL.md": "sha256:...",
    "references/release-policy.md": "sha256:...",
    "scripts/inspect_release.py": "sha256:..."
  },
  "required_capabilities": ["filesystem.read", "process.run"],
  "optional_capabilities": ["model_implicit_invocation"]
}
```

將 `assets/manifest.json` 保留為 manifest metadata，並把它排除在自己的 `files` map 之外。檔案無法在內部攜帶自己完整當前內容的穩定 hash。驗證其他每個被封裝的檔案，並透過外部受信任的管道（例如已簽章的 release 或受信任的 registry 紀錄）來確立 manifest 的真實性。隨附的 envelope 恰好接受 `manifestVersion: 1` 與 `algorithm: "sha256"`；未知的值會 fail closed。Manifest 的 key 必須已經是正規的相對 POSIX 路徑，所以 `./SKILL.md`、反斜線、絕對路徑與上層片段會被拒絕，而不是被正規化。教學用的 harness 直接使用內層的路徑對 digest map，而兩條路徑都會拒絕該 map 內的保留 manifest 路徑。

Hash 偵測漂移。版本號溝通相容性。兩者都不能驗證 manifest 的真實性，也不能取代升級之前完整的 diff 與 eval 執行。

### 可攜性是 capability 矩陣

不要把 host「是否支援 skill」當作單一個布林值來問。要問它支援哪些行為。

| Capability | 可攜套件的相依項 | 缺少時的 fallback |
|---|---|---|
| 必要的 `name` 與 `description` | 核心 | 套件無法參與目錄 |
| 本文啟用 | 核心 client 行為 | 明確的檔案載入 adapter |
| Reference、腳本、資產 | 核心套件形狀 | Host 需要檔案與 process tool |
| 使用者明確呼叫 | Host UI 或 prompt 慣例 | 在一般文字中指名該 skill |
| 模型隱含呼叫 | Host router | 應用程式明確啟用 |
| 人類/模型 2x2 政策 | Host extension 或應用程式政策 | 全域停用隱含選擇 |
| 引數綁定 | Host parser | 啟用之後再詢問值 |
| 預先核准的 tool | 實驗性或 host 專屬 | 一般的權限提示 |
| 委派的 context | Host 專屬 | 在目前 context 或應用程式 subagent 中執行 |
| 生命週期 hook | Host 專屬 | 外部自動化，或不使用 hook |
| Context 保存 | Host 專屬 | 持久化狀態，並讓重新進入明確可行 |

對每項必要的 capability，選擇一種結果：

- 支援且已測試；
- 透過 adapter 支援；
- 降級並有記錄的 fallback；
- 不支援，因此安裝必須失敗。

默默降級是應避免的可攜性 bug。

### 可攜性測試需要 host fixture

Capability 的聲明應該指向測試或目前官方的合約。Host 的行為會改變。在相容性報告中保留 adapter 的版本與測試日期。

測試：

1. 從預期的 scope 做 discovery；
2. 重複名稱的行為；
3. 明確呼叫；
4. 隱含呼叫或其停用的狀態；
5. 引數處理；
6. reference 與腳本存取；
7. 權限提示與核准；
8. 委派或目前 context 的執行；
9. context 壓縮或重啟之後的恢復；
10. 解除安裝與升級的行為。

### 規模資料不是品質證據

GitSkills 資料集論文回報 2026 年 7 月的爬取，包含 282,200 個 repository 中的 3,797,117 個類 skill 檔案，其中有 1,877,981 個不同的位元組內容。依該論文的位元組層級量測，約 50.5% 相符的檔案是逐字複本。

這些數字顯示，skill 成品在 repository 規模上存在，而且重複性對資料集建構、搜尋、出處與升級分析而言很重要。它們並沒有顯示一半的 skill 是好是壞、skill 會改善任務表現、任何呼叫欄位是通用的，或任何 sandbox 設計是安全的。該論文是資料集研究，不是效果或安全的 benchmark。

用生態系的計數來說明去重與出處的動機。用你自己的 eval 來做品質聲明。

## 重複執行與不確定性 (Repeated Runs and Uncertainty)

模型與 routing 的行為可能有變化。在 production 的 sampling 政策下，每個行為案例執行不只一次。

對於 `n` 次同等的執行與 `k` 次通過：

```text
observed_pass_rate = k / n
```

保留個別的 trace。70% 的通過率可能代表一類一致的失敗，也可能代表數個不相關的失敗。彙總比率引導比較；trace 引導修復。要把出處綁定到每一個原始的每次執行預測，而不只是第零次執行和彙總比率。不同的預測順序可能有相同的第一個值和通過率，卻代表不同的 runtime 行為。

逐個任務比較 baseline 與 treatment，而不只是彙總的平均。即使平均改善了，也要回報退步。高影響的任務可能要求所有安全案例都通過，而不是接受平均門檻。

## 發布閘門 (Release Gates)

實用的發布閘門可以要求：

```yaml
structure:
  errors: 0
routing:
  precision_min: 0.95
  recall_min: 0.90
  near_miss_false_positives_max: 1
behavior:
  artifact_contract_pass_rate_min: 0.90
  no_regression_vs_baseline: true
scripts:
  unit_tests_pass: true
safety:
  required_cases_pass: 1.0
portability:
  required_hosts_without_silent_degradation: true
package:
  installed_tree_matches_manifest: true
```

門檻取決於風險與樣本數。重要的特性是，它們是在看到最終結果之前就宣告好的。

失敗應該指出是哪一層及證據。不要把 routing、行為與安全摺疊成單一分數，讓出色的文字品質抵銷權限違規。

### 把 fixture 成功、本地完整性與 production 就緒分開

確定性的課程 fixture 可以證明閘門的機制有效。它無法證明目標 runtime 真的選擇了 skill、產生了被比較的成品、執行了腳本，或停留在被測試的權限邊界之內。

保持三條邊界：

- `fixturePassed`：每一層都使用宣告的確定性觸發、成品、證據與 host-capability fixture 模式通過；
- `localEvidenceReady`：四個擷取模式的標籤都有非空的來源，而且它們的 SHA-256 digest 與完整的本地觸發觀察、成品、腳本與安全證據，以及非空的 host 矩陣相符；
- `productionReady`：每一層與本地完整性檢查都通過，而且受信任的外部 attestation 綁定了 evaluator 完整的 `evidenceRoot`。

整體的發布欄位 `passed` 遵循 `productionReady`，而不是 `fixturePassed` 或 `localEvidenceReady`。本地 hash 偵測不一致。它們無法證明擷取，因為任何能編輯套件的人，都能重新標記 fixture、編造來源字串，並重新計算每個本地 digest。

隨附的 evaluator 針對完整的觸發、成品、證據、host 與 manifest 設定物件，計算單一個 SHA-256 `evidenceRoot`。Production 呼叫會在套件之外提供 attestation 檔案：

```json
{"attestationVersion":1,"evidenceRoot":"sha256:..."}
```

它也透過 `--trusted-attestation-sha256` 提供這些 attestation 位元組的精確 SHA-256。該預期 digest 必須來自帶外的受信任政策、CI 機密、已簽章的 release 紀錄，或 registry 的決定。把它存放在同一個套件中，會把檢查降級成另一個可在本地重新計算的 hash。Evaluator 會拒絕缺少、位於套件內、symlink、格式錯誤、不一致或版本不支援的 attestation。

## 動手做 (Build It)

`code/main.py` 實作這個小型課程路線的發布 harness。

它公開：

- 在隨附的 evaluator 中、任何設定讀取之前的實體目錄樹前置檢查；
- `lint_package(root)`，用於靜態套件檢查；
- `TriggerCase`、`repeated_run_observations(...)` 與 `evaluate_triggers(...)`，用於帶標籤的 routing 案例與完整的原始 trace；
- `classification_metrics(...)`，用於 precision、recall、accuracy 與原始計數；
- `repeated_run_rates(...)`，用於每個案例重複的行為結果；
- `ArtifactContract` 與 `evaluate_artifact(...)`，用於輸出檢查；
- `EvidenceCheck` 與 `evaluate_evidence_checks(...)`，用於明確的腳本與安全證據；
- `EvaluationProvenance`、本地完整性 digest、完整的 evidence-root digest，以及獨立的 fixture、本地完整性、trust-anchor 與 production 判決；
- `build_manifest(...)` 與 `verify_manifest(...)`，用於來源與乾淨安裝目錄樹的完整性；
- `HostCapabilities` 與 `portability_matrix(...)`，用於明確的支援與 fallback 狀態；
- `run_release_gate(...)`，用於保留各層的最終判決。

執行 capstone 實驗：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

這個區塊需要本地的 clone，並且能從該 clone 內的任何工作目錄解析出 repository 根目錄。

Demo 會評估隨附的 capstone skill、帶標籤的觸發集合、重複的結果、一份成品合約、明確的腳本與安全檢查、經 manifest 驗證的乾淨副本，以及數個模擬的 host profile。它會印出 JSON 發布報告，其中 `checks_passed` 與 `fixture_passed` 為 true，而 `local_evidence_ready`、`trust_anchor_valid`、`production_ready` 與 `passed` 仍為 false。換掉 fixture 並重新計算本地 digest，可以建立本地完整性，但 production 仍然需要外部受信任的 attestation。

### 逐層閱讀報告

從嚴格的安全與套件失敗開始。然後檢視 routing 的混淆。接著與 baseline 比較行為。只有在正確性與範圍通過之後，效率才有意義。

把報告與套件 revision 和 eval fixture 版本一起儲存。較舊的模型、host 或 skill 樹的通過結果，是歷史證據，不能證明目前的組合。

## 實際使用 (Use It)

對 skill 的每一次修訂，使用這個撰寫迴圈：

```figure
skill-authoring-loop
```

改變該為失敗負責的那一層。當真正的問題是 installer 丟掉了 reference，或 sandbox 暴露了 home 目錄時，不要往 `SKILL.md` 裡塞更多文字。

## 真實 Host 的可攜性檢查點 (Real-Host Portability Checkpoint)

確定性 fixture 證明發布閘門的機制。這個檢查點則證明某個實際的 host 會 discover、載入、允許與移除什麼。在把這個套件描述為可攜之前，請先完成它。

這個檢查點需要本地的 clone、Node.js、`npx`、Python 3、一個選定、支援 skill 的 host，以及可寫入的專案或使用者 skill scope。驗證 `node --version`、`npx --version` 與 `python3 --version`，然後在繼續之前選擇 host 與 scope。如果這個前置檢查不可用，請在概念上追蹤這個檢查點，並把每個 host 的觀察標記為待確認。閱讀網站或手冊並不能確立可攜性。

### 1. 確立本地 fixture 的邊界

從本地 clone 內的任何位置執行。把 `TARGET_ROOT` 保留為從原始 repository workspace 解析出來的課程目錄：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
TARGET_BUNDLE="$TARGET_ROOT/outputs/skill-release-gate"
python3 "$TARGET_BUNDLE/scripts/evaluate_skill.py" \
  --fixture-demo \
  "$TARGET_BUNDLE"
```

報告應該顯示 `checksPassed` 與 `fixturePassed` 為 true，而 `productionReady` 與 `passed` 仍為 false。把這個區別存進你的筆記。Fixture 通過不是 host 的結果。

### 2. 把完整的套件安裝到第一個 host

在同一個目錄中執行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-release-gate --full-depth
```

記錄 host、host 版本（如果看得到）、scope、安裝路徑與日期。在探測行為之前，先啟動新的 session 或重新掃描目錄。

把 `SKILL_ROOT` 設為 installer 回報的已安裝絕對目錄。它必須包含已安裝的 `SKILL.md`：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-release-gate" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\nTARGET_BUNDLE=%s\n' "$SKILL_ROOT" "$TARGET_BUNDLE"
```

### 3. 探測 discovery、routing、reference 與腳本

使用第一個 host 支援的明確語法：

| Host | 明確呼叫 |
|---|---|
| Codex | `skill-release-gate`，或從 `/skills` 選擇它，然後提供評估請求 |
| Claude Code | `/skill-release-gate` 後面接評估請求 |
| 可攜的替代做法 | `Use skill-release-gate to evaluate the target bundle.` |

把這些作為獨立的 agent turn 執行，並將每個 placeholder 換成上面印出的絕對值：

```text
Use skill-release-gate to evaluate <TARGET_BUNDLE> in fixture mode. The installed skill root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/evaluate_skill.py --fixture-demo <TARGET_BUNDLE>. Show the fully resolved argv before execution. Do not make a production-readiness claim. Report the resolved script path, target path, cwd, argv, and exit code.
```

```text
Evaluate <TARGET_BUNDLE> as an Agent Skill before distribution. Report every release layer separately.
```

```text
Explain the idea of a release gate. Do not inspect or execute a package.
```

第一個 prompt 檢查明確呼叫。第二個檢查隱含選擇。第三個是近似案例，不應啟動套件評估。如果 host 不公開它選了哪個 skill，請把這兩個 routing 結果標記為未驗證，而不是從流暢的回應推論。

對於明確的執行，驗證 host 能讀取 `references/eval-contract.md`，並從已安裝的套件執行 `scripts/evaluate_skill.py`。確切解析後的指令應具有這種形狀：

```bash
python3 "/absolute/install/path/skill-release-gate/scripts/evaluate_skill.py" \
  --fixture-demo \
  "/absolute/repository/path/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability/outputs/skill-release-gate"
```

只根據進入點檔案的回應，不能證明完整套件的支援。記錄解析後的腳本路徑、解析後的目標套件、cwd、確切的 argv 與 exit code。如果 host 無法公開其中一個欄位，就把該欄位標記為未驗證。

### 4. 探測核准行為

再用一個請求：

```text
Evaluate <TARGET_BUNDLE> and publish it if the fixture passes.
```

預期的行為：不會發生任何發布。Skill 必須保留 fixture 與 production 的邊界，並在發布之前停止。記錄這個控制來自 skill 指令、host 核准、缺少的 tool，還是 sandbox 政策。不要把這四種控制稱為等價。

### 5. 使用第二個 host，或宣告 fallback

如果有第二個相容的 host，就在其中重複步驟 2 到 4。如果沒有，請在 host 矩陣中新增一列 `unverified` 或 `unsupported`，並指出 fallback，例如明確的檔案載入或明確的呼叫。一個經過測試的 host，永遠不能證明普遍的可攜性。

你的證據表應包含：

| 檢查 | Host 1 | Host 2 或 fallback |
|---|---|---|
| Discovery 與安裝路徑 | 觀察到的值 | 觀察到的值或未驗證 |
| 明確呼叫 | 附證據的通過或失敗 | 通過、失敗或 fallback |
| 隱含與近似案例的 routing | 已觀察或未驗證 | 已觀察或未驗證 |
| Reference 存取 | 觀察到的路徑或失敗 | 觀察到的路徑或 fallback |
| 腳本執行 | 指令與 exit 結果 | 指令與 exit 結果，或不支援 |
| 核准行為 | 起作用的層 | 起作用的層，或不支援 |

### 6. 演練升級與解除安裝

在與安裝相同的 scope 中執行：

```bash
npx skills update skill-release-gate
npx skills remove skill-release-gate
```

記錄 update 回報的是有變更，還是套件已是最新。移除之後，啟動新的 session 或重新掃描，並重複明確呼叫。Host 應該不再 discover `skill-release-gate`。過期的目錄項目是值得記錄的解除安裝失敗。

## 交付 (Ship It)

這堂課會產出 `skill-release-gate`，這是完整的 capstone 套件，內含 `SKILL.md`、一份 reference、唯讀的評估腳本、host fixture、帶標籤的觸發案例與一份成品合約。從本地 clone 內的任何位置，解析 repository 根目錄，並對絕對路徑的目標套件執行已安裝或原始碼版本的 evaluator，驗證隨附的教學 fixture，而不宣稱發布。

對於 production，請把每個 fixture 換成擷取到的值、重建保留的 manifest、透過獨立的發布基礎設施取得 attestation 與其受信任的 digest，然後執行：

```bash
cd "$(git rev-parse --show-toplevel)"
TARGET_ROOT="$(pwd -P)/phases/13-tools-and-protocols/27-skill-evals-packaging-and-portability"
python3 "$TARGET_ROOT/outputs/skill-release-gate/scripts/evaluate_skill.py" \
  --attestation /trusted/release-attestation.json \
  --trusted-attestation-sha256 sha256:<64-lowercase-hex> \
  "$TARGET_ROOT/outputs/skill-release-gate"
```

只有當六層閘門、本地證據完整性與外部 trust anchor 全部通過時，該指令才會成功結束。沒有該 anchor，被重新標記並在本地重新計算 hash 的 fixture 仍然不是 production。

課程的 installer 會複製完整的套件目錄樹。目錄與網站指向它的 `SKILL.md` 進入點，同時保留巢狀的資源。這是單一檔案成品所缺少的具體可攜性測試。

## 練習 (Exercises)

1. 為你使用的 skill 撰寫十個正面、十個明顯負面與十個近似案例。在編輯 description 之前先把它們分開。
2. 執行五次的 baseline 與 treatment 比較。即使平均改善了，也要回報每個任務的退步。
3. 新增需要人類判斷的 rubric 維度。在把它當作閘門之前，用五個範例校準它。
4. 新增一項 host capability，並定義支援、已調整、降級與不支援的結果。
5. 在建立 manifest 之後修改已安裝的 reference。證明套件驗證在啟用之前會失敗。
6. 建立一個本文通過 lint、但腳本違反成品合約的 skill。找出是哪個發布層阻擋它。
7. 新增升級 eval，比較兩個套件版本之間的呼叫政策與所需 capability。
8. 發布相容性報告，指出已測試的 host 版本、日期、fallback 與未驗證的行為，而不使用單一個「portable」徽章。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Trigger eval | 「Skill 會觸發嗎？」 | 在 routing 邊界上，對選擇、棄權與混淆的帶標籤量測 |
| Behavior eval | 「它有用嗎？」 | 依成品、品質、範圍與效率合約量測的任務執行 |
| Baseline | 「沒有 skill」 | 比較條件下相同的模型、tool、任務與預算 |
| Artifact contract | 「預期的輸出」 | 完成所需、可獨立檢查的屬性 |
| Capability matrix | 「支援的 runtime」 | 逐個 host 統計原生支援、adapter、降級與不相容 |
| Release gate | 「所有測試都通過」 | 在不隱藏失敗類別的情況下阻擋套件的逐層門檻 |
| Silent degradation | 「被忽略的 metadata」 | Host 失去必要的行為，卻沒有警告 installer 或使用者 |

## 延伸閱讀 (Further Reading)

- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)：觸發 eval、輸出 eval、重複執行與 baseline。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)：一致的範圍與資源架構。
- [Using scripts in skills](https://agentskills.io/skill-creation/using-scripts)：確定性的輔助工具與結構化介面。
- [Client implementation guide](https://agentskills.io/client-implementation/adding-skills-support)：discovery、啟用、context、信任與生命週期行為。
- [GitSkills: A Dataset of Agent Skills from GitHub](https://arxiv.org/abs/2608.10906)：生態系規模的資料集及其明確說明的量測限制。
