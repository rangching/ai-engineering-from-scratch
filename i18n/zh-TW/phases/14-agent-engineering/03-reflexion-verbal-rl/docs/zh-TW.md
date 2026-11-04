# Reflexion：語言形式的強化學習 (Reflexion: Verbal Reinforcement Learning)

> 以 gradient 為基礎的 RL，要修一個失敗模式得跑上千次 trial，還要 GPU cluster。Reflexion（Shinn et al., NeurIPS 2023）用自然語言來做：每次 trial 失敗後，agent 寫下一段反思，存進 episodic memory，下一次 trial 就以這份 memory 為條件。Letta 的 sleep-time compute、Claude Code 的 CLAUDE.md learnings、pro-workflow 的 learn-rule，背後都是這個模式。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 02 (ReWOO)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 Reflexion 的三個元件（Actor、Evaluator、Self-Reflector）以及 episodic memory 的角色。
- 用 stdlib 實作 Reflexion loop，包含二元 evaluator、reflection buffer 和全新的重新嘗試。
- 針對特定任務，在 scalar、heuristic 和 self-evaluated 三種回饋來源中做選擇。
- 解釋為什麼語言形式的強化，能抓到 gradient-based RL 需要上千次 trial 才能修好的錯誤。

## 問題 (The Problem)

Agent 任務失敗了。在標準 RL 裡，你會再跑上千次 trial、計算 gradient、更新權重。昂貴、緩慢，而且大多數正式環境的 agent 沒有預算為每次失敗做訓練。

Reflexion（Shinn et al., arXiv:2303.11366）問了不同的問題：如果 agent 只是想一想為什麼失敗，然後把這個想法放進 prompt 再試一次呢？不更新權重，沒有 gradient，只是在 trial 之間存放自然語言。

結果：在 ALFWorld 上，它贏過 ReAct 和其他沒有 fine-tune 的 baseline。在 HotpotQA 上，它比 ReAct 進步。在程式碼生成（HumanEval/MBPP）上，它在當時創下 state of the art。全程沒有任何一步 gradient。

## 核心概念 (The Concept)

### 三個元件 (The three components)

```
Actor         : generates a trajectory (ReAct-style loop)
Evaluator     : scores the trajectory — binary, heuristic, or self-eval
Self-Reflector: writes a natural-language reflection on the failure
```

再加上一個資料結構：

```
Episodic memory: list of prior reflections, prepended to the next trial's prompt
```

一次 trial 會執行 Actor。Evaluator 給它打分。如果分數低，Self-Reflector 就產生一段反思（「我選錯 tool，因為我把問題誤讀成問 X，其實是在問 Y」）。反思進入 episodic memory。下一次 trial 從頭開始，但會看到這段反思。

### 三種 evaluator (Three evaluator types)

1. **Scalar**：外部的二元訊號。ALFWorld 成功或失敗，HumanEval 測試通過或失敗。最簡單，訊號最強。
2. **Heuristic**：預先定義的失敗特徵。「如果 agent 連續兩次產生相同 action，標記為卡住。」「如果 trajectory 超過 50 步，標記為沒有效率。」
3. **Self-evaluated**：LLM 替自己的 trajectory 打分。在沒有 ground truth 時需要。訊號較弱；適合搭配 tool-grounded verification（Lesson 05，CRITIC）。

2026 年的預設做法是混合：有 scalar 就用 scalar，沒有就用 self-eval，heuristic 當作安全護欄。

### 為什麼能一般化 (Why this generalizes)

與其說 Reflexion 是新演算法，不如說它是一個有名字的模式。幾乎每個正式環境的「自我修復」agent 都在跑某種變體：

- Letta 的 sleep-time compute（Lesson 08）：獨立的 agent 回顧過去的對話，並寫入 memory block。
- Claude Code 的 `CLAUDE.md` / 「save memory」模式：把反思記成 learnings，放到未來 session 的開頭。
- pro-workflow 的 `/learn-rule` 指令：把修正記成明確的規則。
- LangGraph 的 reflection 節點：一個為輸出打分的節點，必要時導向 refine。

全都源自同一個洞見：自然語言是夠豐富的媒介，足以在不同次執行之間攜帶「我從失敗中學到什麼」。

### 何時有效，何時無效 (When it works and when it does not)

Reflexion 有效的情況：

- 有明確的失敗訊號（測試失敗、tool 錯誤、答案錯誤）。
- 任務類型可重現（同類型的問題可以再問一次）。
- 反思有空間改善 trajectory（有足夠的 action budget）。

Reflexion 沒有幫助的情況：

- Agent 第一次就成功了。
- 失敗來自外部（網路斷線、tool 壞掉）：「網路斷了」這種反思對未來的執行沒有幫助。
- 反思變成迷信：把一次偶發的不穩定執行寫成一套說法存起來。

2026 年的陷阱：memory rot。反思會累積；有些已經過時或有錯；episodic buffer 越大，重新執行就越慢。緩解方式：定期壓縮（Lesson 06）、為反思設 TTL，或使用獨立的 sleep-time 清理 agent（Letta）。

```figure
react-trace
```

## 動手做 (Build It)

`code/main.py` 在一個 toy 謎題上實作 Reflexion：產生一個總和等於目標值的 3 元素清單。Actor 輸出候選清單；Evaluator 檢查總和；Self-Reflector 寫一行說明哪裡出錯。反思進入 episodic memory，供下一次 trial 使用。

元件：

- `Actor`：腳本化的 policy，看到反思後會改進。
- `Evaluator.binary()`：對目標總和做通過/失敗判斷。
- `SelfReflector`：產生失敗的一行診斷。
- `EpisodicMemory`：帶有 TTL 語意的有界清單。

執行：

```
python3 code/main.py
```

Trace 顯示三次 trial。Trial 1 失敗，反思被存起來；trial 2 看到反思後有改進但仍失敗；trial 3 成功。與 baseline 執行（沒有反思）比較，它會一直卡在 trial 1 的答案。

## 實際使用 (Use It)

LangGraph 把 reflection 當作節點模式提供。Claude Code 的 `/memory` 指令和 pro-workflow 的 `/learn-rule`，把 episodic buffer 外部化成 markdown 檔案。Letta 的 sleep-time compute 在閒置時間執行 Self-Reflector，讓主要 agent 的延遲不受影響。OpenAI Agents SDK 沒有直接提供 Reflexion；你要用自訂 Guardrail（依分數拒絕 trajectory）加上能跨執行保留的 memory `Session` 來自己建。

## 交付 (Ship It)

`outputs/skill-reflexion-buffer.md` 建立並維護 episodic buffer，包含反思擷取、TTL 和去重。給定任務類型和一次失敗，它會產生真正對下一次 trial 有幫助的反思（而不是籠統的「要更小心」）。

## 練習 (Exercises)

1. 從二元 evaluator 換成回傳距離指標（離目標多遠）的 scalar evaluator。收斂會更快嗎？
2. 為反思加上 10 次 trial 的 TTL。過了那個點之後，較舊的反思是有害還是有益？
3. 實作 heuristic evaluator：如果相同 action 重複，就標記該 trial 為卡住。它如何與 Self-Reflector 互動？
4. 用一個忽略反思的對抗式 Actor 執行 Reflexion。最少要做多少反思 prompt engineering，才能迫使 Actor 注意到它們？
5. 讀 Reflexion 論文第 4 節關於 AlfWorld 的部分。從概念上重現 130% 成功率提升：相對於原版 ReAct，關鍵差異是什麼？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Reflexion | 「自我修正」 | Shinn et al. 2023：Actor、Evaluator、Self-Reflector 加上 episodic memory |
| Verbal reinforcement | 「不用 gradient 的學習」 | 把自然語言反思放在下一次 trial 的 prompt 前面 |
| Episodic memory | 「每個任務的反思」 | 針對單一任務類型、儲存先前反思的有界 buffer |
| Scalar evaluator | 「二元成功訊號」 | 來自 ground truth 的通過/失敗或數值分數 |
| Heuristic evaluator | 「依模式的偵測器」 | 預先定義的失敗特徵（例如卡在 loop、步數過多） |
| Self-evaluator | 「對自己 trace 做 LLM-as-judge」 | 沒有 ground truth 時較低訊號的備案；搭配 tool-grounded verification |
| Memory rot | 「過時的反思」 | Episodic buffer 塞滿過時條目；用壓縮/TTL 修復 |
| Sleep-time reflection | 「非同步自我反思」 | 在主要路徑之外執行 Self-Reflector，讓主 agent 維持快速 |

## 延伸閱讀 (Further Reading)

- [Shinn et al., Reflexion: Language Agents with Verbal Reinforcement Learning (arXiv:2303.11366)](https://arxiv.org/abs/2303.11366)：經典論文
- [Letta, Sleep-time Compute](https://www.letta.com/blog/sleep-time-compute)：正式環境中的非同步反思
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents)：把 episodic buffer 當作 context 的一部分來管理
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：reflection 節點模式
