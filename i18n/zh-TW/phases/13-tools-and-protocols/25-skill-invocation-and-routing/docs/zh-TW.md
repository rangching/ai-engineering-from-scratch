# Skill 的呼叫與 Routing (Skill Invocation and Routing)

> 呼叫是先做權限決定、再做相關性決定。好的 description 幫助模型選擇；好的政策決定這個選擇是否被允許。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 24 (Skill Discovery and Progressive Disclosure)
**Time:** ~105 分鐘

## 學習目標 (Learning Objectives)

- 區分使用者明確呼叫、模型隱含呼叫、應用程式呼叫，以及 skill 對 skill 的呼叫。
- 把人類可見性與模型資格，建模為彼此獨立的政策維度。
- 撰寫帶有正面觸發條件與近似但不該觸發的邊界的 routing description。
- 在 trace 與測試中，將資格、選擇、activation、引數綁定與執行分開。
- 調整特定 runtime 的呼叫欄位，且不把它們當作可攜的 frontmatter 呈現。

## 問題 (The Problem)

你安裝了 `database-migration` skill。使用者可以用名稱執行它，但模型也看得到它的 description，並在有人詢問一般的資料庫問題時選中它。接著這個 skill 為一個只需要說明的任務，提出了 schema 變更。

你加上 `user-invocable: false`，期望阻止人們手動執行它。在另一個 runtime 中，該欄位被忽略。你加上 `disable-model-invocation: true`，期望 skill 完全消失。在理解它的 runtime 中，使用者仍然可以明確呼叫它。

欄位名稱沒有問題。問題出在模型。「使用者看得到它」、「模型可以選它」、「應用程式可以預先載入它」和「它裡面的 tool 可以執行」，是各自獨立的事實。單一個叫做 `invocable` 的布林值無法表達它們。

Routing 還有第二種失敗模式。如果 description 含糊，數個 skill 都顯得可能。如果 description 塞滿關鍵字，不相關的任務就會觸發它們。目錄是機率性的介面：小到放得下，具體到足以 route。

## 核心概念 (The Concept)

### 五個管道可以啟動生命週期

| 行為者 | 呼叫形狀 | 典型用途 | 主要風險 |
|---|---|---|---|
| 人類使用者 | 在 UI 或 prompt 中指名 skill | 刻意選擇 workflow | 使用者期待 host 並未授予的可用性或權限 |
| 模型或自主 agent | 從任務 context 選擇目錄項目 | 自動的專家程序 | 誤報的 routing |
| 應用程式 | 透過 runtime 程式碼啟用或預先載入 skill | 固定的產品 workflow | 與某個 host 的隱藏耦合 |
| 另一個 skill 或 subagent | 把確切的 skill 當作 workflow 相依項請求 | 組合 | 循環、缺少相依項或 context 滲漏 |
| 評估 harness | 在固定情境下啟用確切的 skill | 可重複的量測 | 測試 skill 時，意外繞過了正在研究的 production 政策 |

可攜的 Agent Skills 規格定義的是套件。它沒有標準化單一通用的 slash-command UI、隱含 routing 旗標、應用程式 API 或 subagent 生命週期。

### 五個呼叫階段 (The five invocation stages)

```figure
skill-invocation-stages
```

請精確地使用這些詞：

- **Eligible（符合資格）** 表示政策允許這個行為者請求該 skill。
- **Selected（被選中）** 表示使用者指名了它，或 router 判斷它相關。
- **Activated（已啟用）** 表示它的指令進入了工作 context。
- **Executing（執行中）** 表示 agent 已在這些指令下開始模型或 tool 的工作。
- **Completed（已完成）** 表示輸出通過了獨立的成功檢查。

只記錄 `skill_used=true` 的 trace，會隱藏失敗發生的邊界。

### 人類與模型的呼叫形成 2x2 矩陣

| 人類可呼叫 | 模型可呼叫 | 模式 | 適合的範例 |
|:---:|:---:|---|---|
| 是 | 是 | 共用 | 程式碼說明、測試規劃、文件審查 |
| 是 | 否 | 僅限人類 | 發布準備、帳單匯出、破壞性清理計畫 |
| 否 | 是 | 僅限模型 | 內部風格指南、領域 reference、自動化的支援程序 |
| 否 | 否 | 停用或僅限應用程式 | 分階段 rollout、已棄用的套件、程式化的預先載入 |

這個矩陣是政策模型，不是標準的 YAML。

某個目前的 host 用 `disable-model-invocation: true` 表示僅限人類的那一列，用 `user-invocable: false` 表示僅限模型的那一列。預設是兩者皆可。另一個 host 使用 `agents/openai.yaml` 搭配 `allow_implicit_invocation: false`，在停用隱含選擇的同時保留明確呼叫。這些是 runtime adapter。未知的 host 可能忽略它們。

容易混淆的細節很重要：`user-invocable: false` 並不表示「模型不能使用這個」。它只是在定義它的 host 中，移除使用者的直接呼叫。`disable-model-invocation: true` 並不表示「這個 skill 被停用了」。它移除由模型發起的選擇，同時保留使用者的明確存取。

### 明確呼叫以 identity 為先

明確呼叫直接提供 identity：

```text
/release-readiness v2.4.0
```

或：

```text
release-readiness check v2.4.0 without publishing
```

目前的 Codex 介面記載了用 `/skills` 來選擇，以及在請求中使用單純的 skill 名稱來做明確呼叫。Claude Code 記載了 `/skill-name` 以及特定 host 的引數展開。確切的語法、選單可見性、引號規則與變數展開，屬於 host。

明確的請求仍然要通過政策。指名一個 skill，不應該繞過缺少的權限、workspace 限制、核准閘門或 runtime 隔離。

### 隱含呼叫以 description 為先

對於隱含的 routing，模型一開始看到的是目錄 metadata，而不是完整的本文。因此 description 就是 skill 的 routing 介面。

薄弱的：

```yaml
description: Helps with releases.
```

過於寬泛的：

```yaml
description: Use for release, version, package, build, deploy, publish, tag, changelog, GitHub, CI, or software tasks.
```

有界限的：

```yaml
description: Inspect an already prepared release candidate and produce a readiness report. Use when the user asks whether a version, tag, package, or image is ready to publish; do not use for ordinary build failures or feature development.
```

有界限的版本包含：

1. **能力：** 檢視已準備好的候選版本。
2. **輸出：** 就緒報告。
3. **正面邊界：** 詢問某個發布成品是否就緒。
4. **負面邊界：** 一般的 build 與開發不在範圍內。

當兩個相鄰的 skill 共用詞彙時，負面邊界很有用。它們不能取代近似但不該觸發的 eval。

### Routing 是帶有棄權選項的分類

對於 skill `s` 與請求 `x`，想像一個 router 分數：

```text
score(s, x) = capability_match + trigger_match + context_match - exclusion_match - ambiguity_penalty
```

確切的評分可能是 LLM 的決定，而不是算術。工程原則仍然成立：選擇應該勝過門檻，也勝過競爭的 skill。當證據薄弱時，就棄權。

```figure
skill-routing-abstention
```

對高影響的 skill 來說，即使有很強的 description，隱含 routing 可能也不適當。當誤報的代價超過自動選擇的便利時，請使用僅限人類的政策。

### 資格必須先於排名

不要對每個被探索到的 skill 評分、挑出最強的匹配，然後才檢查那一個 skill 的政策。被阻擋的最高匹配，會錯誤地讓分數較低、但符合資格的候選無法被考慮。

隱含 routing 使用這個順序：

1. 依發出請求的行為者與啟用中的 host adapter，過濾被探索到的 skill。
2. 只對符合資格的候選評分。
3. 如果最強的符合資格匹配通過門檻與歧義規則，就選擇它。
4. 當沒有符合資格的候選，或沒有符合資格的分數夠強時，棄權。

假設 `incident-triage` 得分 `0.80`，但它的 host extension 停用了模型呼叫。`incident-review` 得分 `0.55`，且允許模型呼叫。Router 應該把 `incident-review` 評估為最佳的符合資格候選。它不應該選擇 `incident-triage`、拒絕它，然後停止。

這個順序也避免政策變更改變相關性分數的意義。資格定義選擇的集合。相關性則為該集合排名。

### Routing eval 需要近似但不該觸發的案例

正面案例證明 recall：

```json
{"prompt":"Is version 2.4.0 ready to publish?","expected":"release-readiness"}
```

明顯的負面案例證明基本的 precision：

```json
{"prompt":"Explain rotary position embeddings.","expected":null}
```

近似但不該觸發的案例揭露邊界的品質：

```json
{"prompt":"Why did today's package build fail?","expected":"build-diagnostics"}
```

近似案例與 release skill 共用 `package` 和 `build`，但屬於別處。只由明顯正面案例和不相關負面案例組成的 routing 集合，會高估品質。

### 引數有三種表示形式

呼叫的引數會跨越數個邊界：

```figure
skill-argument-boundaries
```

在每個邊界，都要保留意圖，同時不把文字當作程式碼。

- Host 的 parser 決定指令語法與引號。
- Skill 依 host 規則收到綁定的文字或變數。
- 指令驗證必要的值與預設值。
- Tool call 把值轉換成有型別的 schema，並重新驗證它們。

不要把原始引數插值到 shell 指令中。優先使用以引數向量呼叫的腳本，或有型別的 MCP tool。

### 應用程式呼叫是明確的編排

產品可以啟用 skill，因為它的 workflow 已經知道任務類型。例如，pull-request 審查服務可以在使用者按下 Review 之後，預先載入 `pull-request-risk-review`。

這移除了 routing 的不確定性，但會產生對 runtime API 的相依。請把該 adapter 保持在可攜的本文之外：

```figure
skill-host-adapter
```

Skill 在被不同的相容 client 開啟時，仍應保持可理解。

### Skill 對 skill 的呼叫是類似 tool 的邊

假設 `release-readiness` 在相依檔案變更時，要求 `security-change-review`。

呼叫者應該提供：

- 目標 skill 的 identity；
- 有界限的任務與成品路徑；
- 預期的回應合約；
- 呼叫的原因；
- 不可用時的 fallback；
- 最大深度或循環規則。

```json
{
  "target_skill": "security-change-review",
  "task": "Review dependency changes in the candidate diff",
  "inputs": ["artifacts/release.diff"],
  "expected": "risk-report.json",
  "max_depth": 2
}
```

第二個 skill 不會被盲目地貼進第一個。Host 決定如何啟用它，以及它是共用 context、在 fork 中執行，還是透過 tool 結果回傳。

### Context 生命週期取決於 host

啟用之後，skill 本文可能留在對話中、在壓縮期間被摘要，或在委派的 context 中執行。Tool 的允許設定可能只持續一個 turn，而指令持續更久。Subagent 可能收到 skill，但沒有收到父層的完整歷史。

不要寫出依賴看不見的生命週期假設的 skill。把持久的輸出放在檔案或有型別的狀態中，讓重新進入是安全的，並說明中斷之後必須重新載入什麼。

```markdown
On resume, read `artifacts/release-readiness.json` if it exists.
Revalidate the candidate commit before continuing.
Do not repeat an external write whose idempotency key is already recorded.
```

## 動手做 (Build It)

`code/main.py` 把政策與 routing 實作為獨立的 adapter。

模型包含：

- `Actor`，代表人類、模型、自主 agent、應用程式、skill 與 harness 呼叫者；
- `SkillMetadata`，代表 routing identity；
- `InvocationPolicy`，代表人類/模型矩陣；
- `InvocationRequest` 與 `InvocationDecision`，代表可追蹤的輸入與結果；
- `CorePolicyAdapter`，代表沒有 host extension 的可攜行為；
- `ExtensionPolicyAdapter`，代表被辨識的 runtime 欄位；
- `build_invocation_matrix(policy)`，產生 2x2 視圖；
- `route_request(skills, request, adapter)`，在相關性排名、選擇與拒絕之前，先做資格過濾。

執行它：

```bash
cd phases/13-tools-and-protocols/25-skill-invocation-and-routing
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

Demo 會印出一個矩陣，以及明確人類、隱含模型、自主 agent、應用程式、skill 組合與 harness 管道的決定。它的 extension-adapter 結果顯示，被阻擋的最高字面匹配，會在符合資格的替代項被排名之前就被移除。它也包含精確名稱的 allowlist。不需要模型 API。這個確定性的 router 是為了讓政策邊界可被檢視，而不是主張字面比對能重現 production 的模型 routing。

### 為什麼 core 與 extension adapter 要分開

如果一個 parser 對每個觀察到的 frontmatter 欄位都賦予意義，它就會默默地把 runtime 的慣例提升為假的標準。分開的 adapter 迫使呼叫者說明啟用的是哪個 host 的語意。

`CorePolicyAdapter` 只使用應用程式提供的政策。`ExtensionPolicyAdapter` 辨識一組明確的 host 欄位，並記錄是哪個欄位改變了決定。

## 實際使用 (Use It)

在發布 skill 之前，先寫下呼叫合約：

```yaml
actors:
  human: allow
  model: deny
  application: allow
  skill: deny
explicit_name: release-readiness
arguments:
  candidate: required
  publish: fixed_false
ambiguity: ask_user
missing_dependency: stop
context:
  durable_state: artifacts/release-readiness.json
  max_composition_depth: 2
```

這份合約是給 adapter 與測試用的設計文件。除非某個標準明確採用它，否則它不是可攜的 `SKILL.md` frontmatter。

## 交付 (Ship It)

這堂課會產出 `skill-invocation-router` 套件。它包含呼叫模型的 reference、範例 host 政策，以及一個不執行的 CLI，用來評估一個人類、模型、自主 agent、應用程式、skill 組合或 harness 的請求，並回傳包含管道、adapter、分數與原因的 JSON 決定。

單一請求的 CLI 是政策探測，不是完整的觸發評估。請使用課程 27 的帶標籤正面與近似案例設計，來計算混淆計數、precision、recall 與重複執行的穩定性。

## 練習 (Exercises)

1. 建立人類/模型矩陣的全部四列，並為每一列寫出一個合理的使用案例。
2. 在 `CorePolicyAdapter` 中加入僅限應用程式的啟用。證明人類與模型呼叫者仍被拒絕。
3. 為部署 skill 寫出十個近似但不該觸發的案例。每個 prompt 都必須與該 skill 共用詞彙，卻屬於不同的 workflow。
4. 在前兩名 routing 分數之間加入歧義邊際。當邊際太小時，回傳 `ask`。
5. 為 skill 對 skill 的請求加入最大組合深度，並偵測兩個 skill 的循環。
6. 讓同一組帶標籤的集合通過 core 與 extension adapter。說明每一個改變的決定。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Explicit invocation | 「Slash command」 | 行為者直接提供 skill identity，須受政策約束 |
| Implicit invocation | 「模型自己選」 | Router 根據任務 context，從符合資格的目錄 metadata 中選擇 |
| User-invocable | 「人類可以用它」 | 特定 host 的選單或直接呼叫屬性，不是核心欄位 |
| Model-invocable | 「Agent 可以用它」 | 在 host 政策下，可被隱含模型選擇的資格 |
| Invocation adapter | 「Frontmatter parser」 | 把 host 的欄位與 API 對應到宣告好的政策模型的程式碼 |
| Near miss | 「困難的負例」 | 與 skill 預期輸入相似、但不應觸發的請求 |
| Abstention | 「沒有選擇 skill」 | 當證據缺乏或含糊時，刻意產生的 routing 結果 |

## 延伸閱讀 (Further Reading)

- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)：正面觸發、具體性與評估。
- [Evaluating skills](https://agentskills.io/skill-creation/evaluating-skills)：觸發與輸出 eval 的設計。
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)：目前 Codex 的明確與隱含呼叫控制。
- [Claude Code skills](https://code.claude.com/docs/en/skills)：某個 host 的 `user-invocable`、`disable-model-invocation`、引數與委派 context。
