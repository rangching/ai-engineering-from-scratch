# Anthropic 的 Workflow 模式：簡單勝過複雜 (Anthropic's Workflow Patterns: Simple Over Complex)

> Schluntz 與 Zhang（Anthropic, 2024 年 12 月）區分了 workflow（預先定義的路徑）和 agent（動態的 tool use）。五種 workflow 模式涵蓋大多數情況。先從直接呼叫 API 開始。只有在步驟無法預測時才加入 agent。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 Anthropic 的五種 workflow 模式：prompt chaining、routing、parallelization、orchestrator-workers、evaluator-optimizer。
- 解釋 agent 與 workflow 的差異，以及各自的工程成本。
- 判斷何時該選 workflow 而非 agent（反之亦然）。
- 用 stdlib 針對腳本化的 LLM 實作全部五種模式。

## 問題 (The Problem)

團隊常常為了一個其實只要一次 function call 的問題，去找 multi-agent framework。成本是真的：framework 增加的層級會遮蔽 prompt、隱藏控制流程，還會招來過早的複雜度。Schluntz 與 Zhang 在 2024 年 12 月的文章是被引用最多的業界反思：從簡單開始，只有在複雜度值回票價時才加入。

## 核心概念 (The Concept)

### Workflow 對 agent (Workflows vs agents)

- **Workflow。** LLM 和 tool 透過預先定義的程式碼路徑來編排。工程師擁有這張圖。
- **Agent。** LLM 動態地指揮自己的 tool、決定自己的步驟。模型擁有這張圖。

兩者各有其位。Workflow 更便宜、更快、更容易 debug。Agent 能解開開放式問題，但失敗模式更難推理。

### Augmented LLM

所有五種模式的基礎：一個 LLM 接上三種能力，分別是 search（retrieval）、tools（actions）、memory（persistence）。任何 API 呼叫都可以使用這些。

### 五種模式 (The five patterns)

1. **Prompt chaining。** 第 1 次呼叫的輸出是第 2 次呼叫的輸入。適用於任務有乾淨的線性分解時。步驟之間可以有選用的程式化 gate。

2. **Routing。** 由分類器 LLM 挑選要呼叫哪個下游 LLM 或 tool。適用於類別上不同的輸入需要不同處理時（tier-1 支援、退款、bug、業務）。

3. **Parallelization。** 同時執行 N 次 LLM 呼叫，再彙整結果。兩種形態：sectioning（不同的區塊）和 voting（同一個 prompt 跑 N 次，多數決/綜合）。

4. **Orchestrator-workers。** Orchestrator LLM 動態決定要執行哪些 worker（同樣是 LLM），並綜合它們的輸出。類似 agent loop，但 orchestrator 不會無限 loop。

5. **Evaluator-optimizer。** 一個 LLM 提出答案，另一個 LLM 評估。重複到 evaluator 通過。這是一般化的 Self-Refine（Lesson 05）。

### Workflow 勝過 agent 的地方 (Where workflows beat agents)

- **可預測的任務。** 如果你能列舉出步驟，就該這麼做。
- **受成本限制的任務。** Workflow 的步數有上限；agent 可能失控。
- **受合規限制的任務。** 稽核人員想讀這張圖，而不是從 trajectory 推敲出來。

### Agent 勝過 workflow 的地方 (Where agents beat workflows)

- **開放式研究。** 下一步取決於上一步回傳了什麼。
- **長度不定的任務。** 幾分鐘到數小時的工作，步數未知。
- **全新的領域。** 當你還不知道正確的 workflow 時，先探索，之後再固化。

### Context engineering 的搭檔 (The context-engineering companion)

「Effective context engineering for AI agents」（Anthropic 2025）把相鄰的學問正式化：200k window 是預算，不是容器。要放什麼、何時壓縮、何時讓 context 成長。細節請見 Phase 14 關於 context 壓縮的課程（本課程重新編號之前的 Phase 14 較早的 lesson 06）。

```figure
workflow-chain
```

## 動手做 (Build It)

`code/main.py` 針對 `ScriptedLLM` 實作全部五種 workflow 模式：

- `prompt_chain(input, steps)`：依序執行。
- `route(input, classifier, handlers)`：分類 + 分派。
- `parallel_vote(prompt, n, aggregator)`：跑 N 次，彙整。
- `orchestrator_workers(task, workers)`：orchestrator 挑選 worker。
- `evaluator_optimizer(task, proposer, evaluator, max_iter)`：loop 到通過。

執行：

```
python3 code/main.py
```

每個模式都會印出自己的 trace。每個模式的程式碼大約 10 到 15 行；framework 的成本則是以千行計。

## 實際使用 (Use It)

- 大多數任務直接呼叫 API。
- 只有當模式真的需要 durable state（LangGraph）、actor-model 並行（AutoGen v0.4）或角色模板（CrewAI）時，才用 framework。
- 想要 Claude Code 的 harness 形態又不想重建時，就用 Claude Agent SDK。

## 交付 (Ship It)

`outputs/skill-workflow-picker.md` 針對給定的任務描述挑選合適的模式，包含決策理由，以及 workflow 不夠用時改成 agent 的重構路徑。

## 練習 (Exercises)

1. 用信心門檻實作 routing。低於門檻 -> 升級給人工。對 tier-1 支援的情境，門檻落在哪裡？
2. 為 `parallel_vote` 加上 timeout。當其中一次呼叫卡住時會怎樣？缺少投票時要如何彙整？
3. 把 `evaluator_optimizer` 變成 bandit：在迭代間保留前 2 名輸出，這樣較晚出現的好結果不會被較晚出現的壞結果覆蓋。
4. 把 prompt chaining 與 routing 結合：由 router 在三條 chain 中挑一條。與單一大 prompt 的替代方案比較 token 成本。
5. 挑一個你正式環境中的功能。畫出 workflow 圖，數一數步驟。這裡 agent 真的會比較好嗎？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Workflow | 「預先定義的流程」 | 由工程師擁有的 LLM 與 tool 呼叫圖 |
| Agent | 「自主 AI」 | 由模型擁有的圖；動態指揮 tool |
| Augmented LLM | 「有 tool 的 LLM」 | LLM + search + tools + memory；最小單位 |
| Prompt chaining | 「循序呼叫」 | 第 N 次呼叫的輸出是第 N+1 次呼叫的輸入 |
| Routing | 「分類器分派」 | 挑選由哪條 chain/模型處理輸入 |
| Parallelization | 「Fan out」 | N 個並行呼叫；以 sectioning 或 voting 彙整 |
| Orchestrator-workers | 「調度 agent」 | Orchestrator LLM 動態挑選專家 LLM |
| Evaluator-optimizer | 「提案者 + 評審」 | 迭代到 evaluator 通過；一般化的 Self-Refine |

## 延伸閱讀 (Further Reading)

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents)：五種 workflow 模式
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：搭配的學問
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：有狀態的圖何時值得
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)：產品化的 orchestrator-workers 模式
