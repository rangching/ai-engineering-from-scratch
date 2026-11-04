# Agent Skills：可攜的合約與 Runtime 邊界 (Agent Skills: Portable Contract and Runtime Boundary)

> Skill 不是換了個更好檔名的長 prompt。它是一個可被探索的套件，內含指令、資源與可執行的輔助工具，並透過 runtime 合約進入 agent 的 context。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 01 (The Tool Interface), Phase 13 · 05 (Tool Schema Design)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 定義 agent skill，且不把它與 prompt、repository 指令、tool、hook、subagent 或 plugin 混淆。
- 閱讀可攜的 `SKILL.md` 合約，並將它與特定 runtime 的 extension 分開。
- 說明 discovery、selection、activation、資源載入、tool 使用與驗證，是各自獨立的生命週期階段。
- 在 runtime 把 skill 套件放進 agent 的目錄之前，先驗證它。
- 針對具體任務，在 skill、MCP tool、hook、subagent 或一般程式碼之間做選擇。

## 十分鐘的第一次成功 (Ten-Minute First Success)

在長篇說明之前，先做這件事。你會建立一個小 skill，把完整的 reviewer 套件安裝到真正的 agent host，呼叫它、驗證結果，然後移除。這會用可觀察的結果，證明整個生命週期。

### 真實 host 實驗的前置檢查

真實 host 的檢查點需要 Node.js、`npx`、Python 3、一個你選定、支援 skill 的 host，以及你在 installer 中選擇的專案或使用者 scope 的寫入權限。請先驗證本地指令：

```bash
node --version
npx --version
python3 --version
```

安裝之前先決定你要使用的 host 與 scope。如果有任何需求無法滿足，請在網站上閱讀這堂課，或繼續進行下面的手動套件練習。該替代方案教的是合約，但無法證明 host 的 discovery、呼叫、隨附腳本的執行或解除安裝的行為。請把這些觀察標記為待確認。

### 1. 從空的工作目錄開始

在你存放學習工作的任何上層目錄中執行這些指令：

```bash
mkdir -p agent-skills-first-run
cd agent-skills-first-run
TARGET_ROOT="$(pwd -P)"
printf 'TARGET_ROOT=%s\n' "$TARGET_ROOT"
ls -A
```

最後一個指令應該什麼都不印出。如果它印出檔案，請選擇另一個空目錄，讓審查有清楚的邊界。

為你的第一個 skill 建立目錄：

```bash
mkdir -p my-first-skill
```

用以下內容建立 `my-first-skill/SKILL.md`：

```markdown
---
name: my-first-skill
description: Turn rough meeting notes into a compact decision record when the user asks to capture a technical decision.
---

# Decision record

Extract the decision, context, alternatives, owner, and next review date.
If the notes do not contain a decision, ask one clarifying question instead
of inventing one.
```

驗證你是否把檔案建立在預期的目錄：

```bash
test -f my-first-skill/SKILL.md
```

沒有輸出且 exit code 為 0，代表檔案存在。

### 2. 安裝完整的 reviewer 套件

停留在 `agent-skills-first-run`，並執行：

```bash
npx skills add rohitg00/ai-engineering-from-scratch --skill skill-contract-reviewer --full-depth
```

選擇你正在使用的 agent host 與 scope。Installer 應該列出 `skill-contract-reviewer` 以及它寫入的目的地。需要 `--full-depth`，因為這堂課的 skill 是帶有 reference、腳本與資產的巢狀套件。

把 `SKILL_ROOT` 設為 installer 回報的絕對目錄。它必須是包含已安裝 `SKILL.md` 的目錄，不是這堂課的原始碼目錄，也不是目前的工作區：

```bash
# Replace the placeholder with the destination printed by the installer.
SKILL_ROOT="$(cd "/absolute/path/to/skill-contract-reviewer" && pwd -P)"
test -f "$SKILL_ROOT/SKILL.md"
printf 'SKILL_ROOT=%s\n' "$SKILL_ROOT"
```

如果 agent session 本來就是開啟的，請啟動新的 session，或使用該 host 的 skill 重新掃描指令。不要假設每個 host 都會熱重新載入它的目錄。

### 3. 明確地呼叫它

在已安裝的 agent 中，以 `agent-skills-first-run` 作為工作目錄，使用該 host 支援的語法：

| Host | 明確呼叫 |
|---|---|
| Codex | `skill-contract-reviewer`，或從 `/skills` 選擇它，然後提供審查請求 |
| Claude Code | `/skill-contract-reviewer` 後面接審查請求 |
| 可攜的替代做法 | `Use skill-contract-reviewer to review the target package.` |

在請求中使用為 `SKILL_ROOT` 與 `TARGET_ROOT` 印出的絕對值。要求 host 在執行前展開它們，並顯示確切解析後的指令，而不是依賴 process 工作目錄的指令：

```text
Use skill-contract-reviewer to review <TARGET_ROOT>/my-first-skill. The installed bundle root is <SKILL_ROOT>. Run python3 <SKILL_ROOT>/scripts/check_skill.py <TARGET_ROOT>/my-first-skill. Before running it, show the fully resolved argv. Return the validation report, selected primitives, and one sentence for each selection. Include the resolved script path, resolved target path, cwd, argv, and exit code as execution evidence.
```

解析後的指令應具有這種形狀，且不留下任何 placeholder：

```bash
python3 "/absolute/install/path/skill-contract-reviewer/scripts/check_skill.py" \
  "/absolute/workspace/path/agent-skills-first-run/my-first-skill"
```

成功的結果具備三項特性：

1. Host 依名稱找到 `skill-contract-reviewer`。
2. Reviewer 讀取套件合約，並執行其隨附的 validator。
3. 回應包含一份驗證報告，範例沒有結構性錯誤，並附有合理的 primitive 選擇理由。

執行證據也必須指出腳本路徑、目標路徑、cwd、確切的引數向量與 exit code。沒有這些欄位、卻很流暢的報告，無法證明已安裝的附屬腳本確實執行過。

如果 host 回報 skill 不可用，請驗證安裝目的地、重新掃描或重啟一次，再重試明確的請求。不要為了掩蓋安裝失敗而改寫 skill 的 description。

### 4. 探測隱含的選擇

開始一個新的 agent turn，輸入同樣的任務，但不指名 skill：

```text
Review <TARGET_ROOT>/my-first-skill as a reusable agent package and tell me whether its package contract is valid.
```

如果 host 會公開所選的 skill，請記錄它是否選了 `skill-contract-reviewer`。如果 host 不公開 routing，就把隱含選擇標記為未驗證。明確呼叫是可攜的替代做法。

### 5. 清理

只移除已安裝的 reviewer 套件：

```bash
npx skills remove skill-contract-reviewer
```

選擇與安裝時相同的 host 與 scope。重新掃描或開新 session 之後，對 `skill-contract-reviewer` 的明確請求應該回報它不可用。保留 `my-first-skill` 供之後的課程使用，或在完成這條路線之後移除實驗目錄。

## 問題 (The Problem)

假設你的團隊有一套可靠的發布 workflow。它會找出已合併的變更、檢查 migration 筆記、更新 changelog、執行打包指令，並產生審查檢查清單。

把這個 workflow 放在一個 prompt 裡，很容易貼上，卻很難營運。這個 prompt 沒有穩定的識別、沒有 discovery 規則、沒有資源邊界、沒有可測試的套件形狀，也無法回答一些基本問題：誰可以呼叫它？模型何時該選擇它？它能執行哪些腳本？哪些檔案是受信任的？context 被壓縮時，哪些東西會留下來？

相反的錯誤，是把每個可重複使用的指令都當成 skill。Repository 慣例、確定性的自動化、外部 tool、事件 hook 與委派的 agent，解決的是不同的問題。把它們全塞進 `SKILL.md`，會產生一個看起來可攜、實際上卻依賴某個 host 未公開行為的目錄。

第一個工程任務是分類。在決定如何封裝之前，先決定這個成品是什麼。

## 核心概念 (The Concept)

### Skill 編碼了程序性知識

Agent skill 是以 `SKILL.md` 作為進入點的目錄。進入點檔案包含 YAML frontmatter，後面接 Markdown 指令。這個目錄還可以包含 reference、腳本與資產。

```figure
skill-package-anatomy
```

可部署的單位是目錄，而不只是 Markdown 檔案。複製來的 `SKILL.md` 如果缺少 reference，即使 frontmatter 可以解析，仍是損壞的套件。

### 相鄰的抽象概念

| 成品 | 主要工作 | 何時載入或執行 | 它不該冒充什麼 |
|---|---|---|---|
| Prompt | 塑造一次模型互動 | 由應用程式或使用者帶入 | 帶有資源的版本化套件 |
| Repository 指令 | 說明一個程式碼庫的常設規則 | coding runtime 進入該 scope 時 | 可重複使用的任務 workflow |
| Agent skill | 提供可重複使用的程序性知識 | 明確或隱含的 activation | 硬性的授權邊界 |
| MCP tool | 公開有型別的遠端 capability | 由模型或應用程式呼叫 | 詳細的操作程序 |
| Hook | 在事件發生時執行確定性的邏輯 | 宣告的事件發生時 | 機率性的模型 routing |
| Subagent | 以獨立的 context 與狀態委派工作 | 由 orchestrator 建立或呼叫 | 靜態的指令套件 |
| Plugin | 散布較大的 runtime extension | host 安裝或啟用時 | 可攜的 skill 合約本身 |
| Learned skill library | 儲存經由經驗發現的行為 | policy 擷取先前的程式或軌跡時 | 符合標準的 `SKILL.md` 套件 |

發布 skill 可以告訴 agent 如何檢視一次發布。MCP server 可以公開發布 registry。Hook 可以禁止直接 push。Subagent 可以獨立稽核候選版本。這些部分可以組合，因為它們各有不同的責任。

### 「Skill」這個詞指的是兩種不同的概念

研究系統有時會把習得的程式、成功的軌跡或特定環境的 policy 片段稱為 skill。Agent 可以在探索期間建立這些成品，依任務相似度擷取它們、執行它們，並根據回饋修訂函式庫。Phase 14 · 10 建構了那種終身學習的函式庫。

在這個小型課程路線中的 Agent Skill 則不同。它是撰寫出來的套件，具有宣告的檔案系統合約、目錄 metadata、漸進式揭露、由 runtime 居中的呼叫，以及由 host 控制的 tool。它可以由 agent 產生或改進，但這個格式並不要求學習。

| 面向 | Agent Skill 套件 | Learned skill library |
|---|---|---|
| 主要單位 | `SKILL.md` 目錄 | 程式、policy、軌跡或記憶紀錄 |
| 建立方式 | 撰寫、產生或策展 | 通常從環境經驗中發現 |
| 選擇 | 目錄 description 加上 runtime policy | 對任務狀態的擷取或 policy |
| 執行 | 模型遵循指令並呼叫 host 的 tool | 環境執行儲存的行為或程式成品 |
| 可攜性 | 套件合約可跨相容的 host | 通常綁定於一個環境與動作空間 |
| 評估 | Routing、成品、安全與 host 相容性 | 獎勵、成功率、遷移與函式庫成長 |

兩種概念都封裝了可重複使用的能力。它們不該只因為名稱相同，就共用實作上的主張。

### 可攜的核心

Agent Skills 規格要求兩個 frontmatter 欄位：

```yaml
---
name: release-readiness
description: Inspect a release candidate when the user asks whether a version is ready to publish.
---
```

`name` 是穩定的識別碼。它必須符合規格的命名規則，並與上層目錄相符。`description` 既是文件，也是 routing 的 metadata。它應該說明這個 skill 做什麼，以及何時適用。

可攜的選擇性欄位有：

| 欄位 | 目的 | 可攜性說明 |
|---|---|---|
| `license` | 說明套件的條款 | 核心規格 |
| `compatibility` | 說明環境需求 | 核心規格 |
| `metadata` | 攜帶字串值的 extension 資料 | 核心規格 |
| `allowed-tools` | 建議預先核准的 tool | 實驗性；各 host 的支援不一 |

Markdown 本文承載操作指令。它應該定義 workflow、決策點、失敗行為，以及通往支援資源的直接路徑。

```markdown
# Release readiness

Use this workflow for a release candidate, not for ordinary development builds.

1. Read `references/release-policy.md`.
2. Run `python3 scripts/inspect_release.py --format json`.
3. Stop if the report contains a blocking failure.
4. Produce the checklist from `assets/release-checklist.md`.
5. Ask for approval before any publish or tag action.
```

### Runtime extension 是第二層

有些 host 接受額外的 frontmatter 或附屬設定。這些欄位可能有用，但不會自動可攜。

| 行為 | host extension 範例 | 屬於可攜核心？ |
|---|---|:---:|
| 在保留使用者直接呼叫的同時，讓 skill 不出現在模型 routing 中 | `disable-model-invocation` | 否 |
| 在允許模型 routing 的同時，讓 skill 不出現在使用者的指令選單中 | `user-invocable` | 否 |
| 在指令選單中顯示引數說明 | `argument-hint` | 否 |
| 在委派的 context 中執行 skill | `context`、`agent` | 否 |
| 固定 model 或推理設定 | `model`、`effort` | 否 |
| 註冊生命週期自動化 | `hooks` | 否 |
| 在 Codex 中停用隱含呼叫 | `agents/openai.yaml` policy | 否 |

把每個 extension 視為 adapter。讓核心 workflow 在沒有它的情況下仍然有效，記錄替代做法，並測試使用它的 host。Runtime 可能忽略未知欄位、拒絕它，或保留它但沒有實作該行為。

### Frontmatter 是可執行的 metadata

Metadata 在 skill 本文被讀取之前，就會改變系統的行為。

- 格式錯誤的 `name` 可能讓 discovery 失敗。
- 含糊的 `description` 可能把請求路由到錯誤的地方。
- 僅限人類使用的旗標可能把 skill 從模型的目錄中移除。
- Tool 的允許設定可能改變 host 是否詢問權限。
- Context 設定可能把執行移到獨立的 agent session。

像檢視設定程式碼一樣審查 frontmatter。驗證它、為它做版本管理，並把它的行為納入 eval。

### Skill 生命週期

```figure
skill-runtime-lifecycle
```

每個箭頭都是一個邊界，各有自己的失敗模式。

1. **Discovery** 在設定的位置找出可能的套件。
2. **Validation** 在目錄發布之前，拒絕格式錯誤或不安全的套件。
3. **Cataloging** 公開精簡的 `name` 與 `description`，而不是整個套件。
4. **Selection** 決定這個 skill 是否相關。
5. **Activation** 把本文載入模型可見的 context。
6. **Disclosure** 只在某個分支需要時，才讀取 reference 或資產。
7. **Execution** 在 host 的權限與隔離規則下使用 host 的 tool。
8. **Verification** 獨立於模型的主張，檢查所產生的成品。

把這些階段混為一談，會造成錯誤的心智模型。被探索到的 skill 不等於已啟用。已啟用的 skill 並不被授權去做它所描述的一切。被允許的 tool call，也不能證明結果是正確的。

### Skill 與 tool 是正交的

MCP 回答：「這個應用程式可以呼叫哪些 capability，它們的 schema 是什麼？」Skill 回答：「agent 該如何處理這一類任務？」

```figure
skill-tool-orthogonality
```

Skill 可以指名 tool，但實際的 capability registry 由 host 擁有。如果 tool 不存在，skill 應該說明替代做法，或明確失敗。它絕不該暗示光是指名一個 capability，就能創造出它。

### Skill 與 repository 指令是不同的 scope

Repository 指令描述你已身處其中的環境：指令、慣例、產生的檔案與邊界。Skill 則為可能在許多 repository 中發生的任務，提供可重複使用的程序。

當兩者都適用時，目前的使用者請求與 repository 規則會限制 skill。通用的重構 skill 不得覆寫禁止編輯產生檔案的 repository 規則。

### Skill 之間不會互相 import

一個 skill 可以指示 agent 呼叫另一個，但這不是語言層級的 import。第二個 skill 仍然要經過 runtime 的 discovery、資格、activation、權限與 context 處理。

把跨 skill 的相依性寫成可觀察的 workflow 邊：

```markdown
After producing the candidate changelog, invoke the `release-risk-review` skill.
Pass the candidate path and require a blocking or non-blocking verdict.
If that skill is unavailable, stop and report the missing dependency.
```

這讓相依性可被測試，也讓 host 有機會強制執行政策。

## 動手做 (Build It)

`code/main.py` 實作一個小型、以標準為導向的 validator 和一個成品選擇器。它只用 stdlib，讓每條規則都看得見。

Validator 公開：

- `parse_frontmatter(text)`，用來把 metadata 與本文分開。
- `validate_skill_text(text, directory_name, allowed_runtime_extensions=())`，用來檢查必要欄位、命名、未知的 extension、本文是否存在，以及可攜的限制。
- `ValidationIssue` 與 `SkillReport`，用來回傳結構化的證據，而不是單一個不透明的布林值。
- `FrontmatterSyntaxError`，用於無法被安全解讀的輸入。

選擇器公開 `TaskShape` 和 `select_primitives(task)`。它把任務的需求對應到一般程式碼、repository 指令、skill、hook、subagent 或 MCP tool。

執行實驗：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/22-skills-and-agent-sdks
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

這個指令區塊需要本地的 clone，而且必須從該 clone 內的任何位置開始，這樣 `git rev-parse --show-toplevel` 才能解析出 repository 的根目錄。

Demo 會印出 JSON，涵蓋一個有效的可攜 skill、一個由 host 擴充的 skill、一個無效的套件，以及幾個任務形狀的決定。檢查 issue 代碼。套件 validator 應該說明如何修正成品，而不是代替作者猜測。

### 驗證的順序很重要

在較深入的內容規則之前，先驗證便宜的結構事實：

```figure
skill-validation-order
```

這個順序避免次要的錯誤掩蓋第一個被破壞的不變量。

## 實際使用 (Use It)

在撰寫 skill 之前，先填寫這張決策卡：

| 問題 | 如果是 | 可能的 primitive |
|---|---|---|
| 這是否需要跨多個步驟、可重複使用的模型判斷？ | 程序穩定，但決策會變 | Skill |
| 每次事件觸發時都必須發生嗎？ | 漏掉一次執行是不可接受的 | Hook 或應用程式程式碼 |
| 模型是否需要具有型別輸入的外部 capability？ | 操作存在於模型 context 之外 | Tool 或 MCP server |
| 這項工作是否需要獨立的 context、狀態或擁有權？ | 獨立的 worker 回傳有界限的結果 | Subagent |
| 這份指引是否專屬於某一個 repository？ | 它描述本地的指令與限制 | Repository 指令 |
| 一次互動就夠嗎？ | 不需要套件生命週期 | Prompt |

許多 production workflow 會用到不只一列。這張卡可避免一個成品假裝提供每一種特性。

## 交付 (Ship It)

這堂課會在 `outputs/` 底下產出 `skill-contract-reviewer` 套件。它包含：

- 可攜的 `SKILL.md`，用來審查提議的 skill 套件；
- 針對可攜合約與 primitive 選擇的 reference 檢查清單；
- 確定性的驗證腳本；
- 涵蓋 prompt、skill、tool、hook、一般程式碼與 subagent 的任務形狀 fixture。

安裝完整的套件，而不只是它的進入點檔案：

```bash
cd "$(git rev-parse --show-toplevel)"
python3 scripts/install_skills.py /tmp/aiefs-skills --phase 13 --type skill
```

課程的 installer 會回報每個被複製的 Phase 13 skill，並寫入 `/tmp/aiefs-skills/manifest.json`。這個乾淨的目的地用來檢查套件形狀；上面第一次成功的流程，則檢查在真實 host 中的 discovery 與呼叫。

接下來的課程會深入每個生命週期階段。課程 24 建構 discovery 與漸進式揭露。課程 25 建構呼叫政策與 routing。課程 26 把權限與 sandbox 分開。課程 27 把整個套件變成經過評估的發布成品。

## 練習 (Exercises)

1. 用 `TaskShape` 分類你自己團隊的五個 workflow。為每一個你選擇多於一個 primitive 的案例辯護。
2. 新增邊界測試，證明 500 個字元的 `compatibility` 值會通過，而 501 個字元的值會以規格錯誤失敗。
3. 在 allowlist 中加入一個 runtime extension。寫一個測試，證明同一個檔案仍然可以與僅可攜的 skill 區分。
4. 把 400 行的 prompt 拆成 `SKILL.md`、一個 reference、一個腳本合約和一個輸出 template。讓每個檔案只負責一種資訊。
5. 為引用了不可用 MCP tool 的 skill 設計失敗回應。不要默默替換成權限更廣的 tool。
6. 審查一個現有的 skill，並把每個句子標記為 routing、程序、政策、reference 指標或輸出合約。把不屬於的內容移走。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Agent skill | 「儲存下來的 prompt」 | 可被探索的目錄，內含程序性指令與選擇性的資源 |
| Portable core | 「每個 runtime 共有的欄位」 | 由 Agent Skills 規格定義的合約 |
| Runtime extension | 「額外的 frontmatter」 | 特定 host 的設定，其行為需要相容的 adapter |
| Activation | 「Skill 執行了」 | Skill 本文進入模型可見的 context；執行可能稍後才發生 |
| Skill dependency | 「Import 另一個 skill」 | 由 runtime 居中、帶有可用性與政策檢查的呼叫邊 |
| Tool contract | 「函式 schema」 | 一個 capability 的輸入、輸出、權限、副作用、錯誤與證據 |

## 延伸閱讀 (Further Reading)

- [Agent Skills specification](https://agentskills.io/specification)：可攜的目錄與 frontmatter 合約。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)：scope、指令與資源組織。
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)：目前 Codex 的 discovery 與呼叫行為。
- [Claude Code skills](https://code.claude.com/docs/en/skills)：某個 runtime 的呼叫、引數、tool 與委派 context 的 extension。
