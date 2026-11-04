# Skill Discovery 與漸進式揭露 (Skill Discovery and Progressive Disclosure)

> Skill 在本文被載入之前就已經有用。它的 name 和 description 贏得目錄中的位置；較深層的檔案，只有在任務走到那一步時，才會佔用 context。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 13 · 22 (Agent Skills: Portable Contract and Runtime Boundary)
**Time:** ~105 分鐘

## 學習目標 (Learning Objectives)

- 建構檔案系統 discovery 管線，把 scope、驗證、衝突政策與目錄發布分開。
- 說明三個揭露層級：目錄 metadata、啟用中的指令，以及特定任務的資源。
- 設計 reference，讓 agent 不必載入整個套件，就能直接取得所需的細節。
- 讓目錄空間的預算，獨立於啟用中 skill 的 context。
- 當 skill 讀取自己的資源時，拒絕路徑穿越與 symlink 逃脫。

## 問題 (The Problem)

你的 agent 安裝了 200 個 skill。在 session 開始時載入每個 `SKILL.md`、reference 檔案、腳本與 template，會讓當前任務被不相關的程序淹沒。什麼都不載入，則會迫使使用者記住確切的檔案系統路徑。

通常的折衷做法是目錄：向模型展示每個符合資格的 skill 的精簡 identity 與 routing description，選定之後才載入完整本文。這會帶來兩個新的工程問題。

第一，discovery 不只是遞迴的檔案搜尋。Skill 可以存在於專案、使用者、管理者、plugin 或內建 scope。兩個套件可能同名。Symlink 可能指向受信任根目錄之外。格式錯誤的套件可能耗用目錄空間，或變得無法呼叫。

第二，漸進式揭露可能變成漸進式的混亂。如果 `SKILL.md` 寫著「閱讀相關的指南」，而套件中有十二份指南，模型就必須用猜的。如果每份指南又指向另外三個檔案，載入就會變成沒有界限的圖走訪。

好的 runtime 讓 discovery 具有確定性，也讓揭露出於刻意。

## 核心概念 (The Concept)

### Discovery 是編譯器管線 (Discovery is a compiler pipeline)

把檔案系統視為原始碼輸入。不要把原始路徑直接發布給模型。

```figure
skill-discovery-pipeline
```

每個階段都應該產生結構化的資料與結構化的失敗。Discovery log 應該能回答：

- 搜尋了哪些根目錄？
- 找到了哪些候選？
- 哪些候選被拒絕，原因是什麼？
- 哪個套件在衝突中勝出？
- 哪些目錄項目因為預算而被縮短或省略？

沒有這些證據，「模型沒有使用我的 skill」幾乎不可能診斷。

### Scope 是 runtime 政策

可攜的規格定義的是 skill 套件，而不是一個通用的安裝路徑或優先順序。Host 決定它在哪裡搜尋。

通用的 runtime 可能使用這些 scope：

| Scope | 範例根目錄 | 預期的擁有者 |
|---|---|---|
| Workspace | `<repo>/.agents/skills/` | 專案維護者 |
| User | `<user-data>/skills/` | 單一開發者 |
| Administrator | `<system>/skills/` | 機器或組織政策 |
| Plugin | 已簽章的 plugin 套件 | Plugin 發布者與安裝者 |
| Built-in | Runtime 套件 | Runtime 廠商 |

截至 2026 年 8 月，Codex 文件記載專案的 discovery 從 `$CWD/.agents/skills` 開始，經過祖先目錄一直到 repository 根目錄，另外還有使用者、管理者與內建的位置。它支援 symlink 的 skill 目錄。重複的名稱可能同時出現，而不是被合併。這些是 Codex 的行為，不是 `SKILL.md` 的要求；撰寫 adapter 時，請查證目前的 [Codex skill documentation](https://learn.chatgpt.com/docs/build-skills)。

絕不要從目錄名稱杜撰優先順序。把它宣告為政策並測試它。這堂課的實驗為每個 `Scope` 使用明確的整數 rank，讓相同的候選集合永遠以相同方式解析。

### 衝突需要 `name` 以外的 identity

兩個都叫 `release-readiness` 的套件可能都是合法的。一個可能是 workspace 的覆寫，另一個是使用者預設。因此目錄項目至少需要：

```json
{
  "name": "release-readiness",
  "description": "Inspect a release candidate for this repository.",
  "scope": "workspace",
  "source": "/repo/.agents/skills/release-readiness",
  "selected": true
}
```

常見的衝突政策包括：

| 政策 | 好處 | 風險 |
|---|---|---|
| 保留每個候選 | 沒有東西被隱藏 | 模型看到含糊的名稱 |
| 最高優先順序的 scope 勝出 | 呼叫簡單 | 本地套件可能遮蔽受信任的套件 |
| 拒絕重複 | 沒有默默的遮蔽 | 合法的覆寫停止運作 |
| 依來源限定名稱 | 明確的 identity | 面向使用者的名稱變長 |

為 host 選擇一個政策。即使被拒絕或被遮蔽的候選不在模型目錄中，也要在診斷資訊中保留它們。

### 三個揭露層級 (Three disclosure levels)

Agent Skills 規格描述了分階段載入。重點在於每個層級的目的都不同。

```figure
skill-disclosure-levels
```

#### 第 1 層：目錄 metadata

模型需要足夠的資訊，才能把這個 skill 與鄰近的 skill 區分開來。規格估計每個目錄項目大約 100 個 token，但實際的序列化與 tokenization 屬於 host。

有用的 description 有兩個子句：

```yaml
description: Validate a release candidate and produce a readiness report. Use when the user asks whether a version, tag, or package is ready to publish.
```

第一個子句說明能力。第二個子句說明觸發的邊界。課程 25 會用正面與近似但不該觸發的 prompt 來評估這個邊界。

#### 第 2 層：啟用中的指令

啟用之後，本文應該同時扮演地圖與程序。規格建議把 `SKILL.md` 保持在 500 行以下。這是設計上的訊號，不是要填滿的目標。

本文應該包含：

- 任務邊界；
- 預設的 workflow；
- 分支條件；
- 通往較深層檔案的直接 reference；
- tool 與腳本的合約；
- 失敗與停止的行為；
- 預期的輸出及其驗證。

不要只為了讓進入點檔案變短，就把核心 workflow 移到 reference 裡。啟用時必須給模型足夠的 context，讓它能正確地開始。

#### 第 3 層：支援資源

Reference 提供散文或資料。腳本提供確定性的計算。資產會被複製、填寫或轉換成交付物，而不是被當作指令。

| 目錄 | 模型會讀取嗎？ | 模型會執行嗎？ | 典型內容 |
|---|:---:|:---:|---|
| `references/` | 會，需要時 | 否 | schema、政策、領域指南 |
| `scripts/` | 可能會檢視 | 透過被允許的 tool | validator、轉換器、收集器 |
| `assets/` | 僅在有用時 | 否 | template、fixture、圖片、起始檔案 |

這些名稱是慣例，不是神奇的 capability。Host 仍然需要檔案存取與執行 tool。

### 特定分支的 reference 勝過主題傾倒 (Branch-specific references beat topic dumps)

把進入點檔案寫成決策地圖：

```markdown
## Choose the path

- For a Python package, read `references/python-release.md`.
- For a container image, read `references/container-release.md`.
- For a documentation-only release, read `references/docs-release.md`.
- If the release combines artifact types, read only the guides for those artifacts.
```

這讓每個 reference 都有可觀察的載入條件。「閱讀 `references/` 取得更多資訊」則沒有。

讓 reference 圖保持淺層。官方指引建議從 `SKILL.md` 直接連結，並避免深層的鏈。一跳讓可達性可被測試，也降低所需約束從未進入 context 的機會。

```figure
skill-reference-map
```

### 目錄預算與啟用中的 context 是不同的預算

令 `c_i` 為 skill `i` 序列化後的目錄成本，`B_c` 為目錄預算，`b_j` 為啟用中的本文成本，`r_k` 為實際載入的資源。

```text
catalog_cost = sum(c_i for every published skill)
active_cost = sum(b_j for every activated skill) + sum(r_k for every disclosed resource)
```

降低其中一個預算，並不會自動降低另一個。簡短的 description 可以節省目錄空間，但啟用的 900 行本文仍然會淹沒任務。把本文拆成 reference，只有在 runtime 和指令真的避免載入不相關的分支時，才能降低啟用成本。

當 context window 的大小已知時，Codex 目前會把初始 skill 清單的預算設為 context window 的 2%。8,000 個字元的值只有在大小未知時才是 fallback；它不是與 2% 規則並用的第二個上限。當目錄超過適用的預算時，description 可能被縮短或省略。請把這些數字視為目前 Codex 的政策，而不是 Agent Skills 標準的屬性。

### 資源路徑是信任邊界

Skill 應該只讀取其套件內的檔案。單純的字串前綴檢查是不夠的：

```text
references/../../../../.ssh/config
references/external-link -> /private/company-secrets
```

以檔案系統的語意解析套件根目錄與候選，拒絕絕對路徑的輸入，並驗證解析後的候選仍在解析後的根目錄底下。在 discovery 之前就決定是否允許 symlink。如果允許，每次都要檢查解析後的目標。

```figure
skill-resource-containment
```

路徑限制並不建立內容的信任。套件內有效的 reference 仍然可能含有惡意指令。課程 26 處理那個威脅。

### 載入必須可觀察

記錄揭露事件，但不記錄機密：

```json
{
  "event": "skill.resource.loaded",
  "skill": "release-readiness",
  "resource": "references/python-release.md",
  "reason": "candidate contains pyproject.toml",
  "bytes": 2840
}
```

原因把 context 的選擇，變成可審查的證據。它也有助於找出那些導致 agent「以防萬一」就載入每個檔案的指令。

## 動手做 (Build It)

`code/main.py` 建構確定性的 discovery 與揭露引擎。

Discovery 介面包括：

- `Scope`，用於來源與優先順序 metadata；
- `SkillCandidate`，表示尚未驗證的檔案系統候選；
- `discover_scope(scope)`，用來列舉直接子層的 skill 目錄；
- `resolve_collisions(candidates, precedence)`，用來套用一個宣告好的政策；
- `CatalogEntry` 與 `build_catalog(...)`，用來發布有界限的 metadata；
- `CatalogBudget`，用來計算序列化項目，而不假裝字元是通用的 token。

揭露介面包括：

- `load_skill_body(entry, ...)`，用於第 2 層的啟用；
- `validate_reference(skill_dir, reference)`，用於路徑限制；
- `load_reference(...)`，用於有界限的第 3 層讀取。

執行實驗：

```bash
cd "$(git rev-parse --show-toplevel)"
cd phases/13-tools-and-protocols/24-skill-discovery-and-progressive-disclosure
python3 code/main.py
python3 -m unittest discover -s code/tests -v
```

這個區塊需要本地的 clone，並且能從該 clone 內的任何工作目錄解析出 repository 根目錄。

Demo 會建立暫時的專案與使用者 scope、插入一個衝突、在刻意縮小的預算下建構目錄、啟用一個 skill，並嘗試一次有效的 reference 讀取與一次穿越逃脫。不會安裝任何永久檔案。

### 為什麼 discovery 是淺層的

`discover_scope` 檢查直接子層目錄中的 `SKILL.md`。它不會遞迴地把每個巢狀的 `SKILL.md` 都當作獨立的套件。這保留了套件邊界，並避免意外發布安裝的 skill 內的範例或 fixture。

### 為什麼實驗不解析任意的 YAML

實驗只支援其目錄所需的純量 frontmatter。Production 的 runtime 應該使用安全的 YAML parser，搭配明確的 schema、大小限制，以及停用自訂物件建構。「只用 stdlib」是教學上的限制，不是默默發明部分 YAML 方言的許可。

## 實際使用 (Use It)

把這份檢查清單套用到任何 discovery adapter：

1. 列出每個設定的根目錄，以及誰可以寫入它。
2. 說明是否允許 symlink 的套件。
3. 驗證套件名稱、目錄名稱、必要的 metadata 與進入點本文的大小。
4. 在內部 identity 中保留來源與 scope。
5. 宣告並測試重複名稱的行為。
6. 測量送給模型的確切序列化目錄。
7. 記錄本文或資源被載入的原因。
8. 讓資源讀取保持在解析後的套件根目錄之內。
9. 當被引用的檔案不存在時，明確地失敗。
10. 當安裝或政策改變時，重建目錄。

## 交付 (Ship It)

這堂課會產出 `skill-catalog-builder` 套件。它掃描明確排序的根目錄、拒絕 symlink 的進入點檔案與名稱和目錄不符的情況、解決跨 scope 的衝突、拒絕優先順序相同的重複項目，並把選定的 metadata 放進宣告的項目、description 與序列化字元預算之內。

它的 JSON 報告包含選定的項目、被遮蔽的候選、被省略的項目、驗證錯誤、優先順序與預算使用情況。本文與 reference 的載入仍是獨立的 runtime 操作，所以目錄建構器不會執行腳本，也不會讓整個套件進入 context。

## 練習 (Exercises)

1. 新增 plugin scope，並把它放在 user 與 built-in 優先順序之間。用測試證明衝突的結果。
2. 把衝突政策從最高優先順序，改成限定名稱。在目錄中保留兩個項目。
3. 為 `load_reference` 加入位元組大小限制。測試恰好在限制上的檔案，以及多一個位元組的檔案。
4. 建立兩個聽起來幾乎相同的 description。重寫它們，讓觸發邊界不重疊。
5. 新增一份 manifest，包含每個 reference 與腳本的 hash。在載入之前偵測被修改的資源。
6. 為 demo 加上 instrumentation，分別回報第 1 層、第 2 層與第 3 層的位元組數。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Skill discovery | 「找出每個 SKILL.md」 | 搜尋設定的 scope、驗證套件、附上出處並套用政策 |
| Skill catalog | 「已安裝 skill 的清單」 | 符合資格的套件的精簡、模型可見的 routing metadata |
| Collision policy | 「哪個重複項勝出」 | 針對不同來源同名候選的宣告規則 |
| Progressive disclosure | 「延遲載入」 | 從目錄到本文再到特定分支資源的分階段 context 接納 |
| Reference graph | 「被 skill 連結的檔案」 | 可到達的資源結構及其載入條件 |
| Path containment | 「待在資料夾內」 | 驗證解析後的資源目標仍在解析後的套件根目錄之內 |

## 延伸閱讀 (Further Reading)

- [Agent Skills specification](https://agentskills.io/specification)：套件形狀與漸進式揭露層級。
- [Optimizing skill descriptions](https://agentskills.io/skill-creation/optimizing-descriptions)：目錄的 routing metadata。
- [Agent Skills best practices](https://agentskills.io/skill-creation/best-practices)：直接 reference 與進入點檔案大小。
- [OpenAI: Build skills](https://learn.chatgpt.com/docs/build-skills)：目前 Codex 的 discovery scope 與目錄限制。
