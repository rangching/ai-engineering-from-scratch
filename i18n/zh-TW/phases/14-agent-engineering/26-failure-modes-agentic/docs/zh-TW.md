# 失敗模式：Agent 為什麼會壞 (Failure Modes: Why Agents Break)

> MASFT（Berkeley, 2025）把 14 種 multi-agent 失敗模式分成 3 類。Microsoft 的 Taxonomy 記錄了既有的 AI 失敗如何在 agentic 情境中被放大。產業實地資料收斂到五種反覆出現的模式：幻覺 action、範圍蔓延、連鎖錯誤、context 遺失、tool 誤用。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 05 (Self-Refine and CRITIC), Phase 14 · 24 (Observability)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 MASFT 的三個失敗類別，以及每一類中至少四種具體模式。
- 解釋為什麼 agentic 失敗會放大既有的 AI 失敗模式（偏見、hallucination）。
- 描述產業中反覆出現的五種模式及其緩解方式。
- 用 stdlib 實作偵測器，為 agent trace 標上失敗模式標籤。

## 問題 (The Problem)

團隊上線的 agent 在 90% 的 trace 上都能運作。那 10% 的失敗不是隨機雜訊，而是落在少數幾個反覆出現的類別中。一旦你能叫出它們的名字，就能監控並修復它們。

## 核心概念 (The Concept)

### MASFT (Berkeley, arXiv:2503.13657)

Multi-Agent System Failure Taxonomy。14 種失敗模式分成 3 類。標註者之間的 Cohen's Kappa 為 0.88，代表這些類別能被可靠地區分。

核心主張：失敗是 multi-agent 系統的根本設計缺陷，不是靠更好的基礎模型就能修好的 LLM 限制。

### Microsoft Taxonomy of Failure Mode in Agentic AI Systems

- 既有的 AI 失敗（偏見、hallucination、資料外洩）在 agentic 情境中會被放大。
- 新的失敗從自主性中浮現：大規模的非預期 action、tool 誤用、任務偏離。
- 這份白皮書是 agentic 產品的風險登記冊。

### Characterizing Faults in Agentic AI (arXiv:2603.06847)

- 失敗源自 orchestration、內部狀態演變和環境互動。
- 不只是「壞程式碼」或「壞的模型輸出」。

### LLM Agent Hallucinations Survey (arXiv:2509.18970)

兩種主要表現：

1. **Instruction-following Deviation**：agent 沒有遵循 system prompt。
2. **Long-range Contextual Misuse**：agent 忘記或誤用較早 turn 的 context。

子意圖錯誤 (Sub-intention errors)：Omission（漏掉步驟）、Redundancy（重複步驟）、Disorder（步驟順序錯亂）。

### 產業中反覆出現的五種模式 (The five industry-recurring modes)

Arize、Galileo、NimbleBrain 在 2024-2026 的實地分析收斂到：

1. **幻覺 action (Hallucinated actions)。** Agent 呼叫不存在的 tool，或捏造參數。
2. **範圍蔓延 (Scope creep)。** Agent 把任務擴大到超出使用者的要求（建立額外的 PR、寄出額外的 email）。
3. **連鎖錯誤 (Cascading errors)。** 一次錯誤的呼叫觸發下游效應。一個不存在 SKU 的 hallucination 觸發四個 API 呼叫，造成多系統事故。
4. **Context 遺失 (Context loss)。** 長時程任務忘記早期 turn 的限制。
5. **Tool 誤用 (Tool misuse)。** 用錯誤的參數呼叫對的 tool，或根本用錯 tool。

連鎖錯誤是殺手。Agent 分不清「我失敗了」和「任務不可能完成」，而且常常在遇到 400 錯誤時 hallucinate 出成功訊息來結束 loop。

### 緩解：每一步都設閘門 (Mitigation: gates at every step)

在推理鏈的每一步都設自動驗證閘門，對照環境狀態檢查事實依據。具體做法：

- 逐步安全分類器（Lesson 21）。
- Tool call 參數驗證（Lesson 06）。
- 將取回的內容與已知事實交叉檢查（Lesson 05，CRITIC）。
- 透過重新探測狀態來偵測成功幻覺（檔案真的建立了嗎？）。

### 失敗監控哪裡會出錯 (Where failure monitoring goes wrong)

- **只標記當機。** 大多數 agent 失敗會產生看起來有效的輸出。需要內容層級的檢查。
- **沒有 baseline。** Drift 偵測需要最後已知的良好狀態；沒有它，你就無法說「情況正在惡化」。
- **過度告警。** 每次失敗都產生一次呼叫通知。要分群並限制速率。

```figure
failure-cascade
```

## 動手做 (Build It)

`code/main.py` 實作 stdlib 的失敗模式標記器：

- 涵蓋五種模式的合成 trace 資料集。
- 每種模式的偵測函式（針對 tool call、輸出、重複 action 的特徵樣式）。
- 為每筆 trace 貼標籤並回報模式分佈的標記器。

執行：

```
python3 code/main.py
```

輸出：每筆 trace 的標籤 + 彙總分佈，以低成本重現 Phoenix 的 trace 分群所呈現的內容。

## 實際使用 (Use It)

- **Phoenix** 用於正式環境的 drift 分群（Lesson 24）。
- **Langfuse** 用於 session 重播 + 標註。
- **自行打造** 用於你的 observability 平台無法偵測的領域專屬特徵。

## 交付 (Ship It)

`outputs/skill-failure-detector.md` 產生針對你的領域客製化、並接上 trace store 的失敗模式偵測器。

## 練習 (Exercises)

1. 加一個「成功幻覺」的偵測器：agent 回傳成功，但目標狀態沒有改變。
2. 對你做過的某個產品，標記 100 筆真實 trace。哪個模式佔多數？修復它的成本是多少？
3. 實作「連鎖半徑」指標：給定第 N 步的失敗，它影響了多少下游步驟？
4. 讀 MASFT 的 14 種失敗模式。挑三個適用於你產品的。為它們寫偵測器。
5. 把其中一個偵測器接進 CI job：如果有 >=5% 的 trace 被標到某個模式，就讓 build 失敗。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MASFT | 「Multi-agent 失敗分類法」 | Berkeley 的 14 種模式分類 |
| Cascading error | 「漣漪式失敗」 | 一個早期錯誤傳播經過 N 個步驟 |
| Context loss | 「忘了限制」 | 長時程的 turn 丟失早期 turn 的事實 |
| Tool misuse | 「錯的 tool / 錯的參數」 | 有效的呼叫，錯誤的使用方式 |
| Success hallucination | 「假裝完成」 | Agent 在 400 錯誤時宣稱成功；狀態沒變 |
| Scope creep | 「越權」 | Agent 做的比被要求的多 |
| Instruction-following deviation | 「不聽話」 | 忽略 system prompt 或使用者的限制 |
| Sub-intention errors | 「計畫 bug」 | 計畫執行中的漏掉、重複、順序錯亂 |

## 延伸閱讀 (Further Reading)

- [Cemri et al., MASFT (arXiv:2503.13657)](https://arxiv.org/abs/2503.13657)：14 種失敗模式、3 個類別
- [Microsoft, Taxonomy of Failure Mode in Agentic AI Systems](https://cdn-dynmedia-1.microsoft.com/is/content/microsoftcorp/microsoft/final/en-us/microsoft-brand/documents/Taxonomy-of-Failure-Mode-in-Agentic-AI-Systems-Whitepaper.pdf)：風險登記冊
- [Arize Phoenix](https://docs.arize.com/phoenix)：drift 分群的實務
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：何時用較簡單的模式就能完全避開這些模式
