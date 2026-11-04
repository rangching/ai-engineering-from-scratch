# Tree of Thoughts 與 LATS：刻意的搜尋 (Tree of Thoughts and LATS: Deliberate Search)

> 單一條 chain-of-thought trajectory 沒有回頭的空間。ToT（Yao et al., 2023）把 reasoning 變成一棵樹，每個節點都有自我評估。LATS（Zhou et al., 2024）用 Monte Carlo Tree Search 把 ToT 與 ReAct、Reflexion 統一起來。Game of 24 從 4%（CoT）提高到 74%（ToT）；LATS 在 HumanEval 上達到 92.7% pass@1。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 03 (Reflexion)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 把 reasoning 看成搜尋：節點是「thought」，邊是「展開」，價值是「有多有希望」。
- 用 stdlib 實作 ToT 風格的 BFS 樹搜尋，並加上自我評估打分。
- 擴充成 toy LATS MCTS loop，包含 select / expand / simulate / backpropagate。
- 判斷何時搜尋值得付出 token 倍數的成本（Game of 24、程式碼生成），何時單一 trajectory 就夠了（簡單的問答）。

## 問題 (The Problem)

Chain-of-thought 是線性的行走。如果第一步錯了，之後每一步都建立在錯誤的前提上。在 Game of 24（用四個數字配合 + − × ÷ 湊出 24）上，GPT-4 CoT 的準確率只有 4%。模型太早選錯子運算式，就無法恢復。

Reasoning 需要的是：提出多個候選、評估它們、挑出有希望的，並在遇到死路時回溯。這就是搜尋。Tree of Thoughts 和 LATS 是兩種經典的表述方式。

## 核心概念 (The Concept)

### Tree of Thoughts (Yao et al., NeurIPS 2023)

每個節點是一個連貫的中間步驟（「一個 thought」）。每個節點可以展開成 K 個子 thought。LLM 用打分 prompt 對每個節點做自我評估。搜尋探索這棵樹，可以用 BFS、DFS 或 beam。

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

自我評估是承重的部分。論文展示了三種變體：`sure / likely / impossible` 分類、`1..10` 數值分數，以及在候選之間投票。三種在 Game of 24 上都大幅勝過 CoT（用 GPT-4 從 4% -> 74%）。

### LATS (Zhou et al., ICML 2024)

LATS 用 MCTS 把 ToT、ReAct 和 Reflexion 統一起來。LLM 扮演三個角色：

- **Policy**：提出候選的下一步 action（ReAct 風格）。
- **Value function**：為部分 trajectory 打分（ToT 風格的自我評估）。
- **Self-reflector**：失敗時寫下自然語言反思（Reflexion 風格），並用它重新播種之後的 rollout。

環境回饋（observation）會混入 value function，讓搜尋參考真實的 tool 結果，而不只是模型的意見。論文發表時的結果：用 GPT-4 在 HumanEval 達到 92.7% pass@1（SOTA），用 GPT-3.5 在 WebShop 平均 75.9（接近 gradient-based fine-tuning）。

### 最精簡的 MCTS (MCTS, minimally)

每次迭代有四個階段：

1. **Select**：用 UCT（upper confidence bound for trees）從 root 走到 leaf。
2. **Expand**：透過 policy 產生 K 個子節點。
3. **Simulate**：用 policy 從某個子節點 rollout，用 value function（或環境 reward）替 leaf 打分。
4. **Backpropagate**：沿路徑往上更新拜訪次數與價值估計。

UCT 公式：`Q(s, a) + c * sqrt(ln N(s) / N(s, a))`。第一項是 exploitation；第二項是 exploration。依任務調整 `c`。

### 成本的現實 (The cost reality)

搜尋會讓 token 暴增。ToT 在 Game of 24 上使用的 token 是 CoT 的 100 到 1000 倍。LATS 也類似。這不是免費的；搜尋只保留給：

- 單一 trajectory 明顯不夠的任務（Game of 24、複雜程式碼）。
- 正確性比 wall-clock 時間更重要的任務。
- 有便宜又可靠的 value function 的任務（程式碼用 unit test，數學用明確目標）。

如果你的任務只有單一正確答案，evaluator 又有雜訊，搜尋常常會讓事情更糟，因為它會找到一個「分數很高」的錯誤答案。

### 2026 年的定位 (2026 positioning)

大多數正式環境的 agent 不跑 LATS。它們跑的是帶有 tool-grounded verification 的 ReAct（CRITIC，Lesson 05）。搜尋出現在特定的利基：

- 把測試當作 value function 的 coding agent（HumanEval 風格）。
- 探索多條查詢路徑的 deep-research agent。
- LangGraph subgraph 裡以 planning 為主的 workflow。

AlphaEvolve（Lesson 11）是 2025 年的極端例子：對程式碼做演化式搜尋，fitness 可由機器檢查，取得前沿等級的進展（56 年來第一次改進 4x4 矩陣乘法）。

```figure
tree-of-thoughts
```

## 動手做 (Build It)

`code/main.py` 實作：

- 在一個風格化的「選擇算術運算」任務上，做一個小型 ToT BFS。
- 在同一任務上，做一個 toy LATS MCTS loop（Select / Expand / Simulate / Backpropagate），使用 UCT 選擇。
- 一個 value function，結合符號分數與 self-eval 分數。

執行：

```
python3 code/main.py
```

Trace 顯示 ToT 用 BFS 每個節點展開三個候選，對照 LATS 透過 MCTS 收斂到最佳 rollout。兩者都印出 token 數。

## 實際使用 (Use It)

LangGraph 以 subgraph 模式提供 ToT 風格的探索；LangChain 團隊關於 LATS 的文章（2024 年 5 月）是參考教學。LlamaIndex 提供 `TreeOfThoughts` agent。對大多數 2026 年的正式 agent 來說，這個模式藏在 `if task_complexity > threshold: use_search()` 的閘門後面，參見 Lesson 05 的 evaluator-optimizer 模式。

## 交付 (Ship It)

`outputs/skill-search-policy.md` 依任務形態、預算和 evaluator 精確度，在線性 ReAct、ToT、LATS 和演化式搜尋之間做選擇。

## 練習 (Exercises)

1. 用 UCT c=0.1 與 c=2.0 執行 toy LATS。Trace 有什麼變化？
2. 把 value function 換成雜訊更大的打分器（加入隨機抖動）。MCTS 還找得到最佳 leaf 嗎？它能容忍的最低訊噪比是多少？
3. 實作 beam-search ToT（每一層保留 top-k），並與 BFS 比較。在 token 預算很緊時哪個比較好？
4. 讀 LATS 第 5.1 節。重現 HumanEval 的 trajectory 數量：要多少次 rollout 才能達到論文回報的 pass@1？
5. 讀 LATS 論文中討論「LATS 何時幫助較小」的部分。寫一段話的決策規則，把任務形態對應到搜尋策略。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Tree of Thoughts | 「分支的 CoT」 | Yao et al.：由 thought 節點組成、帶自我評估的樹 |
| LATS | 「給 LLM 用的 MCTS」 | Zhou et al.：用 MCTS 統一 ToT + ReAct + Reflexion |
| UCT | 「Upper confidence bound」 | 平衡 exploitation (Q) 與 exploration (ln N / n) 的選擇公式 |
| Value function | 「這個狀態有多好」 | 以 prompt 取得的 LLM 分數或環境 reward；供 backprop 使用 |
| Policy | 「Action 提案者」 | ReAct 風格的產生器；輸出候選的下一個 thought/action |
| Rollout | 「模擬的 trajectory」 | 用 policy 從某節點走到 leaf，再用 value 打分 |
| Backpropagate | 「更新祖先節點」 | 把 leaf 的 reward 沿路徑往上推，更新拜訪次數與 Q |
| Search cost | 「Token 爆炸」 | 在 Game of 24 上是 CoT 的 100-1000 倍；採用前先編預算 |

## 延伸閱讀 (Further Reading)

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)：經典論文
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406)：帶有 Reflexion 回饋的 MCTS
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview)：搜尋用的 subgraph 模式
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)：搭配程式化 evaluator 的演化式搜尋
