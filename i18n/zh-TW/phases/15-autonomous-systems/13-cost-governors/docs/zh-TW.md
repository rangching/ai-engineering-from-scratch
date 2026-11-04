# 動作預算、迭代上限與 Cost Governor (Action Budgets, Iteration Caps, and Cost Governors)

> 一個中型電商 agent 的每月 LLM 成本，在團隊啟用「訂單追蹤」skill 之後，從 $1,200 跳到 $4,800。這不是定價 bug。這是一個 agent 找到了新的 loop，並在裡面持續花錢。Microsoft 的 Agent Governance Toolkit（2026 年 4 月 2 日）把針對這類問題的防禦整理成規範：每個請求的 `max_tokens`、每個任務的 token 與金額預算、每日/每月上限、迭代上限、分層模型路由、prompt caching、context windowing、對昂貴動作的 HITL checkpoint，以及預算超標時的 kill switch。Anthropic 的 Claude Code Agent SDK 以不同的名稱提供相同的基本元件。財務速率限制，例如 10 分鐘內超過 $50 就切斷存取，比每月上限更快抓到 loop。

**Type:** Learn
**Languages:** Python (stdlib, layered cost-governor simulator)
**Prerequisites:** Phase 15 · 10 (Permission modes), Phase 15 · 12 (Durable execution)
**Time:** ~60 分鐘

## 問題 (The Problem)

自主 agent 每一輪都在花真金白銀。Chatbot 的壞輸出只是一則壞回覆；agent 的壞 loop 則是一張帳單。業界有記錄的這種失敗模式稱為「Denial of Wallet」：agent 一直推理、一直呼叫工具、一直計費，而沒有東西能阻止它，因為從來沒人設計過。

解法不是一個數字。而是一疊在不同時間尺度與粒度上的限制：每個請求、每個任務、每小時、每天、每月。設計良好的堆疊，能在幾分鐘內抓到失控的 loop，在幾小時內抓到緩慢的洩漏，在一天內抓到有問題的版本。當 agent 是 long-horizon 且自主的時候，這疊限制也才能真正守住預算。

這是一堂工程課：數學很簡單，團隊失敗的地方在紀律。下面列出的限制，都出現在 Microsoft Agent Governance Toolkit 或 Anthropic Claude Code Agent SDK 的文件中。

## 核心概念 (The Concept)

### Cost-governor 堆疊

1. **每個請求的 `max_tokens`。** 簡單。防止任何一次呼叫輸出無上限的 completion。
2. **每個任務的 token 預算。** 整個執行期間不超過 N 個 token。到上限就硬性停止。
3. **每個任務的金額預算。** 和 token 預算相同，但以貨幣計。Claude Code 中的 `max_budget_usd`。
4. **每個工具的呼叫上限。** `WebFetch` 呼叫不超過 N 次、`shell_exec` 呼叫不超過 N 次，等等。
5. **迭代上限 (`max_turns`)。** Agent loop 的總迭代次數；防止無限推理 loop。
6. **每分鐘 / 每小時 / 每天 / 每月上限。** 滾動視窗。在不同時間尺度上抓到洩漏。
7. **財務速率限制。** 例如「如果 10 分鐘內花費超過 $50，就切斷存取」。在每月上限觸發前，先抓到由 loop 造成的燒錢。
8. **分層模型路由。** 預設用較小的模型；只有當 classifier 判斷任務需要時，才升級到較大的模型。
9. **Prompt caching。** System prompt 與穩定的 context 存在供應商的快取中；重送的 token 成本接近零。
10. **Context windowing。** 壓縮 / 摘要，讓活躍的 context 保持在門檻以下；直接降低 token 成本。
11. **對昂貴動作的 HITL checkpoint。** 在已知昂貴的動作（長時間的工具呼叫、大型下載、昂貴的模型升級）之前，要求人類點一下確認。
12. **預算超標時的 kill switch。** 任何一個上限被觸發，session 就中止。上限會被記錄；需要另一條獨立的重新啟用路徑。

### 為什麼是一整疊，而不是一個上限

單一的每月上限，只會在錢包空了之後才抓到失控的 agent。單一的每個請求上限，在 session 層級什麼都抓不到。不同的失敗模式需要不同的時間尺度：

- **失控的 loop**（agent 卡在 5 秒的 retry）：由速率限制抓到。
- **緩慢的洩漏**（agent 每個任務做的工作約是預期的 2 倍）：由每日上限抓到。
- **有問題的版本**（新版本用 5 倍的 token）：由每週 / 每月上限抓到。
- **合理的暴增**（真實需求，不是 bug）：由附有清楚 log 的每小時 / 每日上限抓到。

### Harness 的預算介面

Claude Code Agent SDK 提供（公開文件）：

- `max_turns`：迭代上限。
- `max_budget_usd`：金額上限；超標時 session 中止。
- `allowed_tools` / `disallowed_tools`：工具 allowlist 與 denylist。
- 工具使用前的 hook 點，用於自訂成本計算。

與權限模式階梯（第 10 課）搭配使用。沒有 `max_budget_usd` 的 `autoMode` session，就是不受管理的自主性。Anthropic 明確指出 Auto Mode 需要預算控制；classifier 與成本是正交的。

### EU AI Act、OWASP Agentic Top 10

Microsoft 的 Agent Governance Toolkit 涵蓋 OWASP Agentic Top 10，以及 EU AI Act 第 14 條（人類監督）的要求。在歐盟的 production 中，logging 與上限的強制執行不是選用的。

### 實際觀察到的 $1,200 → $4,800 案例

Microsoft 文件中的真實案例：一個電商 agent 的每月成本，在新增一個工具之後變成四倍。這個工具讓 agent 在每個 session 中輪詢訂單狀態。沒有 loop 偵測。沒有每個工具的上限。沒有對週對週成長的警報。修正方式是每個工具的上限，加上每日成長警報。這是一個範本：每個新的工具介面都是一個新的潛在 loop；每個新工具都需要自己的上限和自己的警報。

```figure
cost-governor-stack
```

## 實際使用 (Use It)

`code/main.py` 模擬有無分層 cost-governor 堆疊的 agent 執行。模擬的 agent 在幾輪之後漂移進入輪詢 loop；分層堆疊在速率限制的視窗內就抓到它，而單一的每月上限要幾天後才會觸發。

## 交付 (Ship It)

`outputs/skill-agent-budget-audit.md` 稽核提議中 agent 部署的 cost-governor 堆疊，並標出缺少的層。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認在輪詢 loop 的 trajectory 上，速率限制在迭代上限之前觸發。現在停用速率限制，並量測 agent 在迭代上限抓到它之前「花掉」多少。

2. 為 browser agent（第 11 課）設計一組每個工具的上限。哪個工具需要最嚴格的上限？哪個工具可以無上限執行而沒有風險？

3. 閱讀 Microsoft Agent Governance Toolkit 的文件。列出 toolkit 提到的每一種上限類型。把每一種對應到某個失敗模式（失控 loop、緩慢洩漏、有問題的版本、暴增）。

4. 為一個實際的任務（例如「分類一個 repo 中的 50 個 issue」）估算一次整夜無人看管執行的價格。把 `max_budget_usd` 設為你的點估計值的 2 倍。說明為何是 2 倍。

5. Claude Code 的 `max_budget_usd` 是依 session 的總成本觸發。設計一個你會在外部強制執行的互補速率限制。什麼會觸發切斷，重新啟用又是什麼樣子？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| Denial of Wallet | 「失控的帳單」 | Agent loop 產生花費，卻沒有上限能阻止它 |
| max_tokens | 「每個請求的上限」 | 單次 completion 大小的上限 |
| max_turns | 「迭代上限」 | 一個 session 中 agent loop 迭代次數的上限 |
| max_budget_usd | 「金額 kill switch」 | Session 的成本上限；超標時中止 |
| Velocity limit | 「速率上限」 | 短視窗內的花費限制（例如 $50 / 10 分鐘） |
| Tiered routing | 「先用小模型」 | 預設使用便宜的模型；只在 classifier 認為需要時才升級 |
| Prompt caching | 「快取的 system prompt」 | 供應商端的快取，把重送的 token 成本降到接近零 |
| HITL checkpoint | 「人類核准閘門」 | 在昂貴的動作之前，需要人類點一下確認 |

## 延伸閱讀 (Further Reading)

- [Anthropic Claude Code Agent SDK — agent loop and budgets](https://code.claude.com/docs/en/agent-sdk/agent-loop) — `max_turns`、`max_budget_usd`、工具 allowlist。
- [Microsoft Agent Framework — human-in-the-loop and governance](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — cost-governor checkpoint。
- [Anthropic — Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — 供應商端的成本控制。
- [Anthropic — Prompt caching (Claude API docs)](https://platform.claude.com/docs/en/build-with-claude/prompt-caching) — 快取機制。
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — long-horizon agent 的成本特性。
