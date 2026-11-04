# Society of Mind 與 Multi-Agent Debate (Society of Mind and Multi-Agent Debate)

> Minsky 在 1986 年提出的前提：智慧就是一個由專家組成的社會。這個想法每十年就被重新發現一次。2023 年，Du 等人把它變成具體的演算法：多個 LLM 實例各自提出答案、讀彼此的答案、互相批評，然後更新。經過 N 輪，它們收斂到一個 consensus，在六項推理與事實性任務上勝過 zero-shot CoT 和 reflection。有兩個發現很重要：**多個 agent** 和 **多輪** 各自獨立地都有貢獻。這個社會勝過 single-agent 的獨白；多輪交換勝過一次性投票。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 04 (Primitive Model)
**Time:** ~60 分鐘

## 問題 (Problem)

Self-consistency，也就是對同一個模型取樣很多次再取多數答案，是你能外掛上去最便宜的推理改進。它有效，但很快就飽和。你把樣本數加倍，也不會再看到明顯的提升。

Debate 打破這個飽和。它不是從同一個模型取 N 個獨立樣本，而是讓 N 個 agent 讀彼此的推理並修正。樣本之間的相關性下降（它們不再是 i.i.d.），而收斂點往往在 i.i.d. 投票信心十足地答錯的地方，反而是正確的。

## 核心概念 (Concept)

### Du et al. 2023 演算法 (The Du et al. 2023 algorithm)

出自 arXiv:2305.14325 (ICML 2024)：

1. N 個 agent 各自對問題產生初始答案。
2. 對於第 r = 2..R 輪：每個 agent 看到其他 agent 在第 r-1 輪的答案，並被要求「考慮這些之後，給出你更新後的答案」。
3. R 輪之後，對最終答案做多數決投票。

論文在 MMLU、GSM8K、傳記、MATH 和事實性基準上測試。Debate 一致地勝過 CoT 和 Self-Reflection。

### 兩個獨立的旋鈕 (Two independent knobs)

同一篇論文的 ablation：

- **只增加 agent 數量**（1 輪、N 個的多數決）在大多數任務上勝過 single-agent，但會進入高原。
- **只增加輪數**（1 個 agent 看自己先前的推理）幾乎沒有幫助，這是 reflection 眾所周知的弱點。
- **兩者一起** 才產生大幅躍進。多個 agent 之間的多輪交換驅動了增益。

### 為什麼有效 (Why it works)

兩個機制：

1. **接觸到分歧。** 當 agent 看到另一個 agent 得出不同結論的推理鏈，就必須辯護或更新。不論哪一種，第 r+1 輪的 context 都比第 r 輪更豐富。
2. **降低相關的錯誤。** 在 self-consistency 中，所有樣本都來自同一個模型，所以錯誤是相關的，平均之後會得到一個信心十足的錯誤答案。不同的模型或不同的 seed 能降低相關性。不同的 *被辯論的觀點* 則讓相關性降得更低。

### 異質 Debate (Heterogeneous debate)

A-HMAD 和相關的後續研究，讓不同的 agent 使用 *不同的基礎模型*。Llama + Claude + GPT 一起辯論，可以降低單一文化崩潰 (monoculture collapse)（Lesson 26），因為一個模型家族相關的錯誤，不會被其他家族共有。

缺點：參與辯論的弱模型，可能把 consensus 拖向它的錯誤答案（參見 "Should we be going MAD?", arXiv:2311.17371）。

### NLSOM：129 個 agent 的延伸 (NLSOM — the 129-agent extension)

Zhuge 等人（"Mindstorms in Natural Language-Based Societies of Mind," arXiv:2305.17066）把這個想法擴展到 129 個成員的社會。結果：專門化與自組織會隨規模湧現，系統在視覺問答等任務上勝過 single-agent。

### 失敗模式 (Failure modes)

- **阿諛串聯 (Sycophancy cascade)。** 所有 agent 都順從聽起來最有信心的那一個。辯論崩潰成最大聲的那個聲音。用對抗角色來 prompt（「有一個 agent 必須主張相反的立場」）會有幫助。
- **主題漂移 (Topic drift)。** 辯論輪數多了會偏離原本的問題。緩解：每一輪都重新注入問題。
- **運算量暴增 (Compute blowup)。** N 個 agent × R 輪 = N·R 次 LLM 呼叫，每次的 context 還在增長。5 個 agent、5 輪的辯論是 25 次 context 逐漸變大的呼叫。每個問題的成本可能超過單次 CoT 呼叫的 10×。

```figure
multi-agent-debate
```

## 動手做 (Build It)

`code/main.py` 對一道數學題執行 3 個 agent × 3 輪的辯論，每個 agent 一開始都有不同（可能是錯的）答案。Agent 是寫好腳本的，每個都用「依寫好的信心加權，平均鄰居的答案」來「更新」。收斂過程可以在逐輪的 log 中看到。

這個 demo 展示兩個關鍵效果：

- 一輪交換就讓 agent 更接近正確答案。
- 第 2 輪之後多加的輪數，報酬遞減（符合 Du et al. 的高原）。

執行：

```
python3 code/main.py
```

## 實際使用 (Use It)

`outputs/skill-debate-configurator.md` 為新任務設定 debate：agent 數量、輪數、異質性（同一模型 vs 混合）、角色分配（對稱 vs 有一個對抗者）。它也會在你執行之前估計 token 成本。

## 交付 (Ship It)

如果你要上線 debate：

- **把輪數上限設為 3。** Du et al. 顯示 3 輪就能取得大部分增益。再多是成本，不是品質。
- **把 agent 上限設為 5。** 超過 5 個之後，context 膨脹與成本佔主導。
- **預設使用異質。** Pool 中至少有兩種不同的基礎模型。
- **對抗者的位子。** 一個 agent 被 prompt 成無論如何都要持反對意見。打破阿諛。
- **記錄每一輪。** 隱藏中間輪次的 debate 系統無法除錯，也無法稽核。

## 練習 (Exercises)

1. 執行 `code/main.py`，然後把輪數設為 5，觀察報酬遞減。額外的收斂從哪一輪開始停止？
2. 加入第四個 agent，賦予對抗角色：永遠反對目前的多數。這會破壞還是改善收斂？
3. 繪製（印出）每一輪的一致性分數（站在多數答案上的 agent 比例）。它何時達到 1.0？這等同於「正確」嗎？
4. 閱讀 Du et al. 第 4 節的 ablation。用這份程式碼重現「只有 agent」、「只有輪數」與「兩者皆有」的結果。
5. 閱讀 "Should we be going MAD?" (arXiv:2311.17371)，列出 round-robin 之外的兩種 debate 變體，例如 judge-led、chain-of-debate、adversarial。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Society of Mind | 「Minsky 的想法」 | 把智慧視為互動的專家；1986 年的框架，現在透過 LLM debate 實際操作。 |
| Multi-agent debate | 「Agent 吵架」 | N 個 agent 提出、互相批評、在 R 輪中修正，最後多數決。 |
| Consensus | 「他們同意」 | 不是知識論意義上的真理，只是站在多數答案上的比例。可能信心十足地錯。 |
| Rounds | 「交換的步驟」 | 一輪 = 每個 agent 讀其他 agent 並更新一次。 |
| Heterogeneous debate | 「混用模型家族」 | 使用不同的基礎模型來降低錯誤的相關性。 |
| Sycophancy cascade | 「大家都同意大聲的那個」 | Debate 的失敗：agent 不管對錯，都順從最有信心的 agent。 |
| NLSOM | 「129 個 agent 的社會」 | Natural-language society of mind；Zhuge et al. 的擴大版本。 |
| Correlated error | 「同一個模型，同一個 bug」 | self-consistency 飽和的原因；跨不同觀點的 debate 能降低相關性。 |

## 延伸閱讀 (Further Reading)

- [Du et al. — Improving Factuality and Reasoning in Language Models through Multiagent Debate](https://arxiv.org/abs/2305.14325) — 參考論文，ICML 2024
- [Zhuge et al. — Mindstorms in Natural Language-Based Societies of Mind](https://arxiv.org/abs/2305.17066) — 129 個 agent 的 NLSOM
- [Should we be going MAD? A Look at Multi-Agent Debate Strategies for LLMs](https://arxiv.org/abs/2311.17371) — 對 debate 變體做基準測試
- [Debate project page](https://composable-models.github.io/llm_debate/) — Du et al. 的程式碼、demo 與 ablation 細節
