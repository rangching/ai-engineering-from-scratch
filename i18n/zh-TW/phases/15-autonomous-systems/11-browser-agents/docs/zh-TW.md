# Browser Agent 與 Long-Horizon 網頁任務 (Browser Agents and Long-Horizon Web Tasks)

> ChatGPT agent（2025 年 7 月）把 Operator 和 deep research 合併成一個瀏覽器/終端機 agent，並以 68.9% 刷新 BrowseComp 的 SOTA。OpenAI 在 2025 年 8 月 31 日關閉了 Operator，這是產品層的整併。Anthropic 收購 Vercept 之後，Claude Sonnet 在 OSWorld 上從不到 15% 提升到 72.5%。WebArena-Verified（ServiceNow，ICLR 2026）修正了原始 WebArena 約 11.3% 的 false-negative 率，並推出 258 個任務的 Hard 子集。這些數字是真的，攻擊面也是真的：OpenAI 的 preparedness 負責人公開表示，對瀏覽器 agent 的間接 prompt injection「不是一個能被完全修補的 bug」。2025–2026 年已記錄的攻擊包括 Tainted Memories（Atlas CSRF）、HashJack（Cato Networks），以及 Perplexity Comet 的一鍵劫持。

**Type:** Learn
**Languages:** Python (stdlib, indirect prompt-injection attack surface model)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Time:** ~45 分鐘

## 問題 (The Problem)

Browser agent 是一種 long-horizon agent，它讀取不受信任的內容，並採取有後果的動作。Agent 造訪的每個頁面，都是使用者沒有寫過的輸入。每個頁面上的每個表單，都可能是指令通道。2025–2026 年的攻擊樣本顯示這不是假設：Tainted Memories 讓攻擊者透過精心設計的頁面，把惡意指令綁進 agent 的記憶；HashJack 把指令藏在 agent 造訪的 URL fragment 中；Perplexity Comet 的劫持只需要一次點擊。

防禦的局面令人不安。OpenAI 的 preparedness 負責人把沒人想說的話說了出來：間接 prompt injection「不是一個能被完全修補的 bug」。這是因為攻擊存在於 agent 的「讀取 vs. 行動」邊界上，而這個邊界在架構上是模糊的：模型讀到的每個 token，原則上都可能被當作指令來讀。

這堂課列出攻擊面、列出 benchmark 全景（BrowseComp、OSWorld、WebArena-Verified），並建立一個最小的間接 prompt injection 情境模型，讓你能為第 14 課與第 18 課的真實防禦做推理。

## 核心概念 (The Concept)

### 2026 年全景，每個系統一段話

**ChatGPT agent (OpenAI)。** 2025 年 7 月推出。整合 Operator（瀏覽）與 Deep Research（數小時的研究）。獨立的 Operator 在 2025 年 8 月 31 日關閉。在 BrowseComp 上以 68.9% 達到 SOTA；在 OSWorld 與 WebArena-Verified 上也有強勁的數字。

**Claude Sonnet + Vercept (Anthropic)。** Anthropic 收購 Vercept，聚焦於 computer-use 能力。讓 Claude Sonnet 在 OSWorld 上從 <15% 提升到 72.5%。Claude Computer Use 以工具 API 的形式提供。

**Gemini 3 Pro with Browser Use (DeepMind)。** Browser Use 整合提供 computer-use 控制；FSF v3（2026 年 4 月，第 20 課）專門追蹤 ML R&D 領域的自主性。

**WebArena-Verified (ServiceNow, ICLR 2026)。** 修正一個有充分記錄的問題：原始 WebArena 有約 11.3% 的 false-negative 率（被標為失敗、實際上已解決的任務）。Verified 版本用人工策展的成功標準重新評分，並新增 258 個任務的 Hard 子集（ICLR 2026 論文，openreview.net/forum?id=94tlGxmqkN）。

### BrowseComp vs. OSWorld vs. WebArena

| Benchmark | 衡量什麼 | Horizon |
|---|---|---|
| BrowseComp | 在時間壓力下，於開放網路上找到特定事實 | 數分鐘 |
| OSWorld | Agent 操作完整桌面（滑鼠、鍵盤、shell） | 數十分鐘 |
| WebArena-Verified | 模擬網站中的交易式網頁任務 | 數分鐘 |
| Hard subset | 具有多頁面狀態轉換的 WebArena-Verified 任務 | 數十分鐘 |

這是不同的面向。BrowseComp 分數高，代表 agent 找得到事實；不代表 agent 能訂機票。OSWorld 分數比較接近「它在我的桌面上能不能用」。WebArena-Verified 比較接近「它能不能走完一個流程」。任何 production 決策，都需要符合任務分布的那個 benchmark。

### 列出攻擊面

1. **間接 prompt injection。** 不受信任的頁面內容含有指令。Agent 讀到它們。Agent 執行它們。公開的例子：2024 年 Kai Greshake et al.、2025 年 Tainted Memories 論文、2026 年 HashJack（Cato Networks）。
2. **URL fragment / query 注入。** 被爬取 URL 的 `#fragment` 或 query string 含有指令。從不會被視覺化呈現，但仍在 agent 的 context 之內。
3. **記憶綁定攻擊。** 頁面指示 agent 寫入持久的記憶（第 12 課涵蓋持久狀態）。下一個 session，記憶在沒有可見觸發的情況下啟動 payload。
4. **針對已驗證 session 的 CSRF 型攻擊。** Tainted Memories 類別：agent 在某處已登入；攻擊者的頁面發出會改變狀態的請求，agent 用使用者的 cookie 執行它們。
5. **一鍵劫持。** 一個視覺上無害的按鈕夾帶 agent 會跟著執行的 payload。Comet 類別。
6. **Agent 宿主面的 Content-Security-Policy 漏洞。** 渲染層和工具層本身也可能是攻擊向量；browser-agent 中包著 browser 的技術棧範圍很廣。

### 為什麼「無法完全修補」

這種攻擊與 agent 的能力同構。Agent 必須讀取不受信任的內容才能完成工作。Agent 讀到的任何內容都可能含有指令。Agent 遵循的任何指令都可能與使用者實際的請求不一致。防禦（信任邊界、classifier、工具 allowlist、對有後果動作的 HITL）提高攻擊成本並縮小其 blast radius。它們無法終結這整個類別。

這與 Lob's theorem（第 8 課）是同一種推理模式：agent 無法證明下一個 token 是安全的；它只能建立一個讓不安全的 token 更容易被偵測的系統。

### 真正上線的防禦姿態

- **讀 / 寫邊界。** 讀取從來不是有後果的。寫入（送出表單、發布內容、呼叫有副作用的工具），如果起始內容來自信任邊界之外，就需要新的人類核准。
- **每個任務一份工具 allowlist。** Agent 可以瀏覽；除非該工具已為此任務明確啟用，否則它不能發起匯款。第 13 課涵蓋預算。
- **Session 隔離。** Browser agent 的 session 只用範圍受限的憑證執行。沒有 production auth，沒有個人電子郵件。保留每個 HTTP 請求的 log 供稽核。
- **內容 sanitizer。** 抓取的 HTML 在串接進模型 context 之前，先移除已知的惡意模式。（減少簡單的攻擊；擋不住精密的 payload。）
- **對有後果動作的 HITL。** Propose-then-commit 模式（第 15 課）。
- **記憶上的 canary token。** 如果某筆記憶被觸發，使用者會看到（第 14 課）。

```figure
injection-boundary
```

## 實際使用 (Use It)

`code/main.py` 針對三個合成頁面，建立一個小型 browser-agent 執行的模型。一個頁面是良性的，一個在可見文字中有直接的 prompt-injection 區塊，一個有 URL-fragment 注入（看不見，但在 agent 的 context 之內）。腳本顯示 (a) 天真的 agent 會做什麼、(b) 讀/寫邊界能抓到什麼、(c) sanitizer 能抓到什麼、(d) 兩者都抓不到什麼。

## 交付 (Ship It)

`outputs/skill-browser-agent-trust-boundary.md` 為一個提議中的 browser-agent 部署劃定範圍：它會碰到哪些信任區域、它被授權寫入什麼，以及第一次執行前必須就位的防禦措施。

## 練習 (Exercises)

1. 執行 `code/main.py`。找出 sanitizer 抓得到、但讀/寫邊界抓不到的攻擊，以及只有讀/寫邊界抓得到的攻擊。

2. 擴充 sanitizer，使其能偵測一類 HashJack 風格的 URL-fragment 注入。在帶有合法 fragment 的良性 URL 上，量測 false-positive 率。

3. 挑一個你熟悉的真實 browser-agent 工作流程（例如「訂機票」）。列出每個讀取與每個寫入。標出哪些寫入需要 HITL，以及原因。

4. 閱讀 WebArena-Verified 的 ICLR 2026 論文。找出原始 WebArena 評分不可靠的一類任務，並說明 Verified 子集如何解決它。

5. 為 browser-agent 的情境設計一個記憶 canary。你會儲存什麼、存在哪裡、什麼會觸發警報？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Indirect prompt injection | 「壞掉的頁面文字」 | Agent 讀取的頁面中，不受信任的內容含有 agent 會執行的指令 |
| Tainted Memories | 「記憶攻擊」 | Agent 把攻擊者提供的指令寫入持久記憶；下個 session 被觸發 |
| HashJack | 「URL fragment 攻擊」 | 藏在 URL fragment / query string 中的 payload 在 agent 的 context 內，但不會被視覺化呈現 |
| One-click hijack | 「壞按鈕」 | 可見的介面元素夾帶 agent 會執行的後續 payload |
| BrowseComp | 「網路搜尋 benchmark」 | 在開放網路上找到特定事實；以分鐘計的 horizon |
| OSWorld | 「桌面 benchmark」 | 完整的 OS 控制；多步驟的 GUI 任務 |
| WebArena-Verified | 「修正過的網頁任務 benchmark」 | ServiceNow 重新評分的 WebArena，附 Hard 子集 |
| Read/write boundary | 「副作用閘門」 | 讀取從不具後果；如果內容不受信任，寫入需要新的核准 |

## 延伸閱讀 (Further Reading)

- [OpenAI — Introducing ChatGPT agent](https://openai.com/index/introducing-chatgpt-agent/) — Operator 與 deep research 的合併；BrowseComp SOTA。
- [OpenAI — Computer-Using Agent](https://openai.com/index/computer-using-agent/) — Operator 的系譜，以及後來成為 ChatGPT agent 的架構。
- [Zhou et al. — WebArena](https://webarena.dev/) — 原始 benchmark。
- [WebArena-Verified (OpenReview)](https://openreview.net/forum?id=94tlGxmqkN) — ICLR 2026 修正子集論文。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 包含對 computer-use agent 攻擊面的討論。
