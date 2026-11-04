# Self-Refine 與 CRITIC：迭代改進輸出 (Self-Refine and CRITIC: Iterative Output Improvement)

> Self-Refine（Madaan et al., 2023）讓同一個 LLM 在 loop 中扮演三個角色：generate、feedback、refine。7 個任務平均提升 +20 個百分點。CRITIC（Gou et al., 2023）則透過外部 tool 進行驗證，強化 feedback 這一步。2026 年這個模式出現在每個 framework 裡，名稱是「evaluator-optimizer」（Anthropic）或 guardrail loop（OpenAI Agents SDK）。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 Self-Refine 的三個 prompt（generate、feedback、refine），並解釋為什麼 history 對 refine prompt 很重要。
- 解釋 CRITIC 的關鍵洞見：沒有外部依據時，LLM 的自我驗證並不可靠。
- 用 stdlib 實作帶有 history 與選用外部 verifier 的 Self-Refine loop。
- 把這個模式對應到 Anthropic 的「evaluator-optimizer」workflow 和 OpenAI Agents SDK 的 output guardrail。

## 問題 (The Problem)

Agent 產生了一個幾乎正確的答案。也許有一行程式碼語法錯誤。也許摘要太長。也許計畫漏掉一個邊界情況。你想要的是：agent 批評自己的輸出，然後修好它。

Self-Refine 顯示這用單一模型就能做到，不需要訓練資料，也不需要 RL。但有個問題：LLM 在困難的事實上不擅長自我驗證。CRITIC 指出了解法：把 verify 這一步交給外部 tool（搜尋、code interpreter、計算機、測試執行器）。

這兩篇論文合起來，定義了 2026 年迭代改進的預設做法：generate、verify（盡可能用外部）、refine，在 verifier 通過時停止。

## 核心概念 (The Concept)

### Self-Refine (Madaan et al., NeurIPS 2023)

一個 LLM，三個角色：

```
generate(task)            -> output_0
feedback(task, output_0)  -> critique_0
refine(task, output_0, critique_0, history) -> output_1
feedback(task, output_1)  -> critique_1
refine(task, output_1, critique_1, history) -> output_2
...
stop when feedback says "no issues" or budget exhausted.
```

關鍵細節：`refine` 會看到完整 history，也就是所有先前的輸出與批評，所以不會重複犯錯。論文做了 ablation：拿掉 history，品質會大幅下降。

重點結果：在 7 個任務（數學、程式碼、縮寫、對話）上平均絕對提升 +20，包含 GPT-4。不需訓練、不需外部 tool，只要單一模型。

### CRITIC (Gou et al., arXiv:2305.11738, v4 Feb 2024)

Self-Refine 的弱點：feedback 這一步是 LLM 替自己打分。對事實性的說法，這並不可靠（hallucination 對產生它的模型來說，常常看起來很有說服力）。CRITIC 把 `feedback(task, output)` 換成 `verify(task, output, tools)`，其中 `tools` 包括：

- 用來查證事實說法的搜尋引擎。
- 用來檢查程式碼正確性的 code interpreter。
- 用來算數的計算機。
- 特定領域的 verifier（unit test、type checker、linter）。

Verifier 會根據 tool 結果產生結構化的批評。Refiner 再以這份批評為條件。

重點結果：在事實性任務上 CRITIC 勝過 Self-Refine，因為批評有依據。在沒有外部 verifier 的任務（創意寫作、格式化）上，CRITIC 退化成 Self-Refine。

### 停止條件 (The stop condition)

兩種常見形態：

1. **Verifier 通過。** 外部測試回傳成功。有的話優先選這個（unit test、type checker、guardrail assertion）。
2. **沒有發出 feedback。** 模型說「輸出沒問題」。比較便宜但不可靠；要搭配 max-iteration 上限。

2026 年的預設：兩者結合。「Verifier 通過就停止，或是模型說沒問題且 iterations >= 2，或是 iterations >= max_iterations。」

### Evaluator-Optimizer (Anthropic, 2024)

Anthropic 2024 年 12 月的文章把這個列為五種 workflow 模式之一。兩個角色：

- Evaluator：為輸出打分並產生批評。
- Optimizer：根據批評修訂輸出。

Loop 直到 evaluator 通過。這就是 Anthropic 框架下的 Self-Refine/CRITIC。Anthropic 補充的關鍵工程細節：evaluator 和 optimizer 的 prompt 應該有明顯差異，這樣模型才不會只是蓋橡皮圖章。

### OpenAI Agents SDK 的 output guardrail

OpenAI Agents SDK 把這個模式以「output guardrail」提供。Guardrail 是在 agent 最終輸出上執行的 validator。如果 guardrail 被觸發（拋出 `OutputGuardrailTripwireTriggered`），輸出就被拒絕，agent 可以重試。Guardrail 可以呼叫 tool（CRITIC 風格），也可以是純函式（Self-Refine 風格）。

### 2026 年的陷阱 (2026 pitfalls)

- **橡皮圖章 loop (Rubber-stamp loops)。** 同一個模型用相同 prompt 風格做 generation 和 critique，會收斂到「我覺得不錯」。使用結構上不同的 prompt，或用較小較便宜的模型做 critique。
- **過度 refine (Over-refinement)。** 每次 refine 都會增加延遲與 token。預算抓 1 到 3 次；超過就升級給人工審查。
- **在簡單任務上用 CRITIC。** 如果沒有外部 verifier，CRITIC 會退化成 Self-Refine；不要為了一個空殼 verifier 付出延遲。

```figure
self-refine
```

## 動手做 (Build It)

`code/main.py` 在一個 toy 任務上實作 Self-Refine 和 CRITIC：給定主題，產生簡短的項目清單。Verifier 檢查格式（3 個項目，每個少於 60 字元）。CRITIC 加上一個外部「事實 verifier」，會懲罰已知的 hallucination。

元件：

- `generate`：腳本化的產生器。
- `feedback`：LLM 風格的自我批評。
- `verify_external`：CRITIC 風格、有依據的 verifier。
- `refine`：根據 history 重寫輸出。
- 停止條件：verifier 通過，或最多 4 次迭代。

執行：

```
python3 code/main.py
```

比較 Self-Refine 與 CRITIC 的執行結果。CRITIC 抓到了 Self-Refine 漏掉的事實錯誤，因為外部 verifier 有自我批評所沒有的依據。

## 實際使用 (Use It)

Anthropic 的 evaluator-optimizer 就是這個模式，用對 Claude 友善的說法表達。OpenAI Agents SDK 的 output guardrail 是 CRITIC 形狀（guardrail 可以呼叫 tool）。LangGraph 提供讀起來像 Self-Refine 的 reflection 節點。Google 的 Gemini 2.5 Computer Use 加入了逐步的安全 evaluator，是 CRITIC 的變體：每個 action 在提交前都會被驗證。

## 交付 (Ship It)

`outputs/skill-refine-loop.md` 依任務形態、verifier 可用性和迭代預算，設定 evaluator-optimizer loop。它會輸出 generator、evaluator/verifier 和 optimizer 的 prompt，以及停止策略。

## 練習 (Exercises)

1. 用 max_iterations=1 執行 toy。CRITIC 還有幫助嗎？
2. 把外部 verifier 換成有雜訊的（隨機 30% 偽陽性）。Loop 會怎麼表現？這是大多數 guardrail 堆疊在 2026 年的現實。
3. 實作「generator 與 critic 用不同模型」的變體：大模型產生，小模型批評。它比同一個模型更好嗎？
4. 讀 CRITIC 第 3 節（arXiv:2305.11738 v4）。說出三類驗證 tool，並各舉一個例子。
5. 把 OpenAI Agents SDK 的 `output_guardrails` 對應到 CRITIC 的 verifier 角色。SDK 做錯了什麼，又做對了什麼？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Self-Refine | 「會自己修正的 LLM」 | 單一模型內帶 history 的 generate -> feedback -> refine loop |
| CRITIC | 「Tool-grounded verification」 | 用外部 verifier（搜尋、程式碼、計算、測試）取代 feedback |
| Evaluator-Optimizer | 「Anthropic 的 workflow 模式」 | 兩個角色：evaluator 打分、optimizer 修訂，反覆到收斂 |
| Output guardrail | 「事後檢查」 | OpenAI Agents SDK 在 agent 產生輸出後執行的 validator |
| Verify step | 「批評階段」 | 承重的決定：有依據，還是自己評分 |
| Refine history | 「模型已經試過什麼」 | 放在 refine prompt 前面的先前輸出 + 批評；拿掉就品質崩壞 |
| Rubber-stamp loop | 「自我同意的失敗」 | 相同 prompt 的批評回傳「看起來不錯」；用結構不同的 prompt 修復 |
| Stop condition | 「收斂測試」 | Verifier 通過，或沒有 feedback 且達到迭代上限；絕不只用單一條件 |

## 延伸閱讀 (Further Reading)

- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：經典論文
- [Gou et al., CRITIC (arXiv:2305.11738)](https://arxiv.org/abs/2305.11738)：tool-grounded verification
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：evaluator-optimizer workflow 模式
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：把 output guardrail 當作 CRITIC 形狀的 verifier
