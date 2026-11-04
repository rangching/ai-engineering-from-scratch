# Human-in-the-Loop：Propose-Then-Commit

> 2026 年對 HITL 的共識是具體的。它不是「agent 問，使用者按 Approve」。它是 propose-then-commit：提議的動作連同 idempotency key 一起持久化到持久儲存；連同意圖、資料來源脈絡 (data lineage)、涉及的權限、blast radius 與 rollback 計畫一起呈現給審查者；只有在得到明確的肯定確認後才 commit；執行後再驗證副作用是否真的發生。LangGraph 的 `interrupt()` 加上 PostgreSQL checkpointing、Microsoft Agent Framework 的 `RequestInfoEvent`，以及 Cloudflare 的 `waitForApproval()`，實作的都是同一種形狀。典型的失敗模式是橡皮圖章式核准 (rubber-stamp approval)：沒有審查就點了「Approve?」。有記錄的緩解方式，是附帶明確檢查清單的 challenge-and-response。

**Type:** Learn
**Languages:** Python (stdlib, propose-then-commit state machine with idempotency)
**Prerequisites:** Phase 15 · 12 (Durable execution), Phase 15 · 14 (Tripwires)
**Time:** ~60 分鐘

## 問題 (The Problem)

Agent 採取一個動作。使用者必須決定：核准或不核准。如果決定是瞬間做出的，那大概不是審查。如果決定是有結構的，它會慢，但值得信任。工程上的問題是，如何讓有結構的審查成為阻力最小的路徑。

2023 年代的 HITL 模式是同步提示：「Agent 想寄信給 X，內文是 Y，核准嗎？」使用者點 Approve。每個人都覺得系統很安全。實務上這個介面被大量橡皮圖章式地蓋章：使用者核准得很快，核准幾乎無法預測什麼，而當 agent 出錯時，稽核軌跡顯示的是一長串使用者想不起來的核准紀錄。

2026 年的模式 propose-then-commit，把 HITL 搬到持久的基底上，附上結構化的 metadata，並且要求明確的 commit。每個託管 agent SDK 都有自己的版本：LangGraph `interrupt()`、Microsoft Agent Framework `RequestInfoEvent`、Cloudflare `waitForApproval()`。API 名稱不同；形狀不變。

## 核心概念 (The Concept)

### Propose-then-commit 狀態機

1. **Propose。** Agent 產生一個提議的動作。持久化到持久儲存（PostgreSQL、Redis、Durable Object）。包含：
   - 意圖（agent 為什麼要這麼做）
   - 資料來源脈絡 (data lineage)（是什麼來源導致這項提議）
   - 涉及的權限（哪些 scope / 檔案 / endpoint）
   - blast radius（最壞的情況是什麼）
   - rollback 計畫（如果已 commit，我們如何復原）
   - idempotency key（每個提議唯一；重新提交會回傳同一筆記錄）
2. **Surface。** 審查者看到帶有所有 metadata 的提議。審查者是人（不是 agent 審查自己）。
3. **Commit。** 明確的肯定確認。動作執行。
4. **Verify。** 執行後，讀回副作用並確認。如果 verify 步驟失敗，系統處於已知的壞狀態，警報啟動。

### Idempotency key

沒有 idempotency key，暫時性失敗後的 retry 可能讓已核准的動作被執行兩次。具體例子：使用者核准「從 A 轉 $100 到 B」。網路閃斷。Workflow retry。使用者只核准了一次，轉帳卻執行了兩次。Idempotency key 把核准綁定到單一、唯一的副作用；第二次執行是 no-op。

這和 Stripe 與 AWS API 使用的 idempotency 模式相同。Microsoft Agent Framework 文件明確提到把它重用在 agent 核准上。

### 持久性：為什麼核准能比行程活得更久

核准的等候室是 agent 不擁有的一份狀態。Workflow 被暫停（第 12 課）。核准抵達時，workflow 從那個確切的點繼續。這就是為什麼 LangGraph 把 `interrupt()` 配上 PostgreSQL checkpointing，而不只是記憶體內的狀態：兩天後的核准，仍然能找到完好無缺的 workflow。

### 橡皮圖章式核准與 challenge-and-response 緩解方式

HITL 的預設 UI（「Approve」/「Reject」按鈕）產生快速、沒有實質審查的核准。有記錄的緩解方式：一份 challenge-and-response 檢查清單，要求在 Approve 按鈕啟用之前，先對特定問題給出肯定的答案。具體形狀：

- 「你了解這會動到哪項資源嗎？[ ]」
- 「你已確認 blast radius 是可以接受的嗎？[ ]」
- 「如果失敗，你有 rollback 計畫嗎？[ ]」

這不是為官僚而官僚，而是一種強制機制。無法勾選這些框框的審查者，要嘛要求釐清（升級），要嘛拒絕（安全的預設）。Anthropic 的 agent 安全研究明確引用以檢查清單驅動的 HITL，作為對橡皮圖章式核准模式的緩解方式。

### 什麼算有後果

不是每個動作都需要 propose-then-commit。2026 年的指引：

- **有後果的動作**（一律 HITL）：不可逆的寫入、金融交易、對外通訊、production 資料庫變更、破壞性的檔案系統操作。
- **可逆的動作**（有時 HITL）：編輯本機檔案、staging 環境的變更、有清楚 rollback 的可逆寫入。
- **讀取與檢視**（絕不 HITL）：讀取檔案、列出資源、呼叫唯讀 API。

### 動作後驗證

「commit 跑完了」不等於「副作用發生了」。網路分區與 race condition 可能造成 workflow 以為成功了，但後端並沒有持久化。Verify 步驟在 commit 之後重新讀取目標資源來確認。這和資料庫交易使用 `RETURNING` 子句，或 AWS 在 `PutObject` 後做 `GetObject` 是同一種模式。

### EU AI Act 第 14 條

第 14 條要求歐盟的高風險 AI 系統必須有有效的人類監督。「有效」不是裝飾。法規用語明確排除橡皮圖章式的模式。在 Microsoft Agent Governance Toolkit 的合規文件中，帶有 challenge-and-response 的 propose-then-commit 是能通過第 14 條檢驗的形狀。

```figure
mx-propose-then-commit
```

## 實際使用 (Use It)

`code/main.py` 用 stdlib Python 實作一個 propose-then-commit 狀態機。持久儲存是一個 JSON 檔案。Idempotency key 是 (thread_id, action_signature) 的 hash。驅動程式模擬三種情況：乾淨的核准流程、暫時性失敗後的 retry（不得重複執行），以及橡皮圖章式的預設流程對比 challenge-and-response 流程。

## 交付 (Ship It)

`outputs/skill-hitl-design.md` 審查提議中的 HITL 工作流程是否符合 propose-then-commit 的形狀，並標出缺少的 metadata、idempotency、驗證或 challenge-and-response 層。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認已核准提議的 retry 會使用持久記錄，且不會重新執行。現在把 idempotency key 改成包含時間戳記，並展示 retry 會重複執行。

2. 為提議記錄擴充一個 `rollback` 欄位。模擬一次 verify 步驟失敗的執行。展示 rollback 自動觸發。

3. 閱讀 Microsoft Agent Framework 的 `RequestInfoEvent` 文件。找出 API 包含、但玩具 engine 缺少的一個 metadata 欄位。加入它，並說明它能防範什麼。

4. 為一個特定動作（例如「發文到公開的 Twitter 帳號」）設計 challenge-and-response 檢查清單。審查者必須回答哪三個問題？為什麼是這三個？

5. 挑一個同步的「Approve?」提示就足夠的情況（不需要持久儲存）。說明原因，並指出你所接受的風險類別。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Propose-then-commit | 「兩階段核准」 | 持久化的提議 + 明確的 commit + verify |
| Idempotency key | 「可安全重試的 token」 | 每個提議唯一；第二次執行是 no-op |
| Data lineage | 「它從哪裡來」 | 導致這項提議的特定來源內容 |
| Blast radius | 「最壞情況」 | 動作出錯時的影響範圍 |
| Rubber-stamp | 「快速核准」 | 沒有實質審查就點了「Approve」 |
| Challenge-and-response | 「強制檢查清單」 | 審查者必須對特定問題給出肯定的確認 |
| RequestInfoEvent | 「MS Agent Framework 的基本元件」 | 帶有結構化 metadata 的持久 HITL 請求 |
| `interrupt()` / `waitForApproval()` | 「框架基本元件」 | LangGraph / Cloudflare 中相同形狀的對應物 |

## 延伸閱讀 (Further Reading)

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`、持久核准。
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` 與 Durable Objects。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — 把 HITL 當作 long-horizon 風險的緩解方式。
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — 高風險系統的法規基準。
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — 圍繞監督的 constitutional 框架。
