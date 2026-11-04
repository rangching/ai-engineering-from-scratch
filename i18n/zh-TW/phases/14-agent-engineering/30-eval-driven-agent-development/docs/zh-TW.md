# 以 Eval 驅動的 Agent 開發 (Eval-Driven Agent Development)

> Anthropic 的指引："start with simple prompts, optimize them with comprehensive evaluation, and add multi-step agentic systems only when needed." 評估不是最後一步。它是驅動 Phase 14 中其他每個選擇的外層 loop。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** All of Phase 14.
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出三個評估層級：靜態 benchmark、自訂離線、線上正式環境，以及各自的用途。
- 解釋 evaluator-optimizer 的緊密 loop。
- 描述 2026 年的最佳實務：eval 與程式碼放在一起、在 CI 中執行、擋住 PR。
- 把 Phase 14 的每一課連結到它所產生的 eval 案例。

## 問題 (The Problem)

Agent 能通過 demo。它們在正式環境失敗的方式，是 demo 無法預測的。Benchmark 回答的是「這個模型整體能力夠強嗎？」，而不是「這個 agent 為我的產品出貨的 patch 對嗎？」。答案是：在三個層級上持續執行評估，並讓每個 guardrail 和學到的規則都對應到一個 eval 案例。

## 核心概念 (The Concept)

### 三個評估層級 (Three evaluation layers)

1. **靜態 benchmark**：程式碼用 SWE-bench Verified（Lesson 19）、瀏覽 / 桌面用 WebArena/OSWorld（Lesson 20）、通才用 GAIA（Lesson 19）、tool use 用 BFCL V4（Lesson 06）。用於跨模型比較和退化把關。資料污染是真實存在的：SWE-bench+ 發現了 32.67% 的解答洩漏。一律回報 SWE-bench Verified 或經過洩漏稽核的集合（例如 SWE-bench+）的分數。

2. **自訂離線 eval**：你產品的形態：
   - LLM-as-judge（Langfuse、Phoenix、Opik，見 Lesson 24）。
   - 以執行為基礎（執行 patch，檢查測試）。
   - 以 trajectory 為基礎（將 action 序列與黃金序列比較；OSWorld-Human 顯示頂尖 agent 比黃金多 1.4-2.7 倍）。

3. **線上 eval**：正式環境：
   - Session 重播（Langfuse）。
   - Guardrail 觸發的告警（Lesson 16、21）。
   - 逐步成本 / 延遲追蹤（Lesson 23 的 OTel span）。

### Evaluator-optimizer (Anthropic)

緊密的 loop：

1. Proposer 產生輸出。
2. Evaluator 評判。
3. 精進直到 evaluator 通過。

這是一般化的 Self-Refine（Lesson 05）。任何你在意的 agent 流程，都可以用 evaluator-optimizer 包起來以提升可靠度。

### 2026 年最佳實務 (2026 best practice)

- Eval 與程式碼放在一起。
- 在每個 PR 的 CI 中執行。
- 以 eval 分數把關 merge（例如「相對 main 的退化不超過 5%」）。
- 每個 guardrail 都對應一個 eval 案例。
- 每條學到的規則（Reflexion、pro-workflow learn-rule）都對應一個失敗案例。

### 把 Phase 14 串起來 (Tying Phase 14 together)

Phase 14 的大多數課程都會產生 eval 案例。下表列出各課最明確的那一個：

| Lesson | 它產生的 eval 案例 |
|--------|------------------------|
| 01 Agent Loop | 預算耗盡、無限 loop 防護 |
| 02 ReWOO | Tool 失敗時 planner 能正確 replan |
| 03 Reflexion | 學到的反思在重試時生效 |
| 05 Self-Refine/CRITIC | Judge 通過精進後的輸出 |
| 06 Tool Use | 參數轉型有效；未知的 tool 被拒絕 |
| 07-10 Memory | Retrieval 引用與來源相符；過時事實會失效 |
| 12 Workflow Patterns | 每種模式都產生正確輸出 |
| 13 LangGraph | Resume 能精確重現 state |
| 14 AutoGen Actors | DLQ 能接住當機的 handler |
| 16 OpenAI Agents SDK | Guardrail 在正確的輸入上觸發 |
| 17 Claude Agent SDK | Subagent 結果回到 orchestrator |
| 19-20 Benchmarks | SWE-bench Verified 分數、WebArena 成功率、OSWorld 效率 |
| 21 Computer Use | 逐步安全能抓到被注入的 DOM |
| 23 OTel | Span 發出必要的屬性 |
| 26 Failure Modes | 偵測器能標記已知的失敗 |
| 27 Prompt Injection | PVE 拒絕被污染的 retrieval |
| 28 Orchestration | Supervisor 路由到正確的專家 |
| 29 Runtime Shapes | DLQ 處理 N% 的失敗 |

如果你的 eval 套件對每一列都有一個案例，你就涵蓋了 Phase 14 的核心。

### Eval 驅動開發在哪裡失敗 (Where eval-driven development fails)

- **沒有 baseline。** 沒有最後已知良好狀態的 eval 是讀不懂的。要儲存 baseline。
- **沒有依據的 LLM-judge。** Judge 也會 hallucinate。CRITIC 模式（Lesson 05）：judge 以外部 tool 為依據。
- **對 eval 過度擬合。** 為 eval 最佳化會偏離正式環境的實用性。輪替案例。
- **不穩定的 eval。** 非確定性的案例會造成誤報。固定 seed，為 state 做快照。

```figure
ae-eval-three-layers
```

## 動手做 (Build It)

`code/main.py` 是 stdlib 的 eval harness：

- 帶有類別（benchmark、custom、online）的案例 registry。
- 一個受測的腳本化 agent。
- Evaluator-optimizer loop：提案、評判、精進，直到通過或達到最大輪數。
- CI 閘門：彙整通過率 + 對照 baseline 的退化。

執行：

```
python3 code/main.py
```

輸出：每個案例的通過/失敗、退化旗標、CI 閘門的判定。

## 實際使用 (Use It)

- 把 eval 案例寫在與 agent 程式碼相同的 repo 中。
- 在每個 PR 上透過 CI 執行。
- 退化時讓 build 失敗。
- 追蹤通過率隨時間的變化。
- 把每個正式環境的失敗都連結到一個新案例。

## 交付 (Ship It)

`outputs/skill-eval-suite.md` 為 agent 產品建立三層 eval 套件，含 CI 閘門與退化追蹤。

## 練習 (Exercises)

1. 拿你其中一個正式環境的失敗。寫一個能重現它的 eval 案例。你的 agent 現在能通過嗎？
2. 為你的領域建立有三個維度（事實、語氣、範圍）的 LLM-judge 評分標準。為 50 個 session 打分。
3. 把 eval 套件接進 CI。退化 >=5% 時讓 build 失敗。
4. 加一個 trajectory 效率指標：agent 用了多少步，對比黃金 trajectory？
5. 把 Phase 14 的每一課對應到你套件中的一個 eval 案例。有缺的嗎？那就是要補的缺口。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Static benchmark | 「現成的 eval」 | SWE-bench、GAIA、AgentBench、WebArena、OSWorld |
| Custom offline eval | 「領域 eval」 | 針對你產品形態的 LLM-as-judge / 執行 / trajectory |
| Online eval | 「正式環境 eval」 | Session 重播、guardrail 告警、成本/延遲追蹤 |
| Evaluator-optimizer | 「提案-評判-精進」 | 迭代直到 judge 通過 |
| CI gate | 「Merge 阻擋器」 | eval 退化時讓 build 失敗 |
| Baseline | 「最後已知良好狀態」 | 用來偵測退化的參考分數 |
| Trajectory efficiency | 「超出黃金的步數」 | Agent 步數除以人類專家最少步數 |

## 延伸閱讀 (Further Reading)

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)："start simple, optimize with evals"
- [OpenAI, SWE-bench Verified](https://openai.com/index/introducing-swe-bench-verified/)：經整理的 benchmark
- [Berkeley Function Calling Leaderboard](https://gorilla.cs.berkeley.edu/leaderboard.html)：tool-use benchmark
- [Langfuse docs](https://langfuse.com/)：實務中的 eval + session 重播
