# 用 HTN 與演化式搜尋做 Planning (Planning with HTN and Evolutionary Search)

> 符號式 planning 處理計畫可被證明正確的情況。演化式程式碼搜尋處理 fitness function 可由機器檢查的情況。ChatHTN（2025）和 AlphaEvolve（2025）展示了各自搭配 LLM 後能解鎖什麼。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 02 (ReWOO and Plan-and-Execute)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋 Hierarchical Task Network：task、method、operator、precondition、effect。
- 描述 ChatHTN 的混合 loop：符號式搜尋，加上 LLM 備援分解。
- 解釋 AlphaEvolve 的演化 loop，以及為什麼它只在有程式化 evaluator 時才有效。
- 用 stdlib 實作 toy HTN planner 和 toy 演化式搜尋。

## 問題 (The Problem)

ReWOO（Lesson 02）、Plan-and-Execute 和 ReAct 涵蓋了大部分 agent planning。有兩種情況它們處理得不好：

1. **需要可證明正確的計畫。** 排程、航線規劃、合規 workflow：計畫在建構上就必須是健全的。一份流暢但偶爾 hallucinate 出某個步驟的 LLM 計畫是不可接受的。
2. **有機器可檢查 fitness function 的最佳化。** 矩陣乘法、排程啟發式、編譯器 pass：目標不是「一份正確的計畫」，而是「最好的計畫」。

HTN planning 和 AlphaEvolve 解決這兩個不同的問題。兩者都把 LLM 當作放大器，而不是取代品。

## 核心概念 (The Concept)

### Hierarchical Task Network

HTN 包含：

- **Task**：複合的（待分解）和原始的（可直接執行）。
- **Method**：把複合 task 分解成子 task 的方式，附帶 precondition。
- **Operator**：帶有 precondition 與 effect 的原始 action。
- **State**：一組事實。

Planning：給定一個目標 task 和初始狀態，找出一個分解成原始 operator 的方式，且這些 operator 的 precondition 依序都能被滿足。

HTN 比 LLM 還早出現，至今仍是可證明正確計畫的參考標準。

### ChatHTN (Gopalakrishnan et al., 2025)

ChatHTN（arXiv:2505.11814）把符號式 HTN 與 LLM 查詢交錯進行：

1. 嘗試用既有的 method 分解目前的複合 task。
2. 如果沒有 method 適用，就問 LLM：「在狀態 `s` 下你會如何分解 `task`？」
3. 把 LLM 的回應轉成候選子 task。
4. 對照 operator schema 驗證；拒絕無效的分解。
5. 遞迴。

論文的核心主張：產生的每一份計畫都可證明是健全的，因為 LLM 的建議只會以候選分解的形式進入，絕不直接修改計畫。符號層負責正確性；LLM 負責擴充 method library。

線上 method 學習（OpenReview `gwYEDY9j2x`，2025 年後續工作）加入了一個 learner，用迴歸將 LLM 產生的分解一般化，把 LLM 查詢頻率最多降低 75%。

### AlphaEvolve (Novikov et al., 2025)

AlphaEvolve（arXiv:2506.13131，DeepMind，2025 年 6 月）是另一種東西：由 Gemini 2.0 Flash/Pro 組合所編排的演化式程式碼搜尋。

Loop：

1. 從一個 seed 程式 + 一個程式化 evaluator（回傳 fitness 分數）開始。
2. 一組 LLM 提出 mutation。
3. 把 mutation 送進 evaluator。
4. 保留最好的；再次 mutate。

已發表的成果：

- 56 年來第一次改進 Strassen 的 4x4 複數矩陣乘法（48 次純量乘法）。
- 透過 Borg 排程啟發式，回收了 0.7% 的 Google 運算資源。
- 在前沿工作負載上，FlashAttention 加速 32%。

硬性限制：fitness function 必須可由機器檢查。對散文式答案做演化式搜尋不會收斂。

### 何時用哪個 (When to use which)

| 問題類型 | 使用 | 原因 |
|---------------|-----|-----|
| 有硬性限制的排程 | HTN + ChatHTN | 可證明的健全性 |
| 編譯器最佳化 | AlphaEvolve | 機器可檢查的 fitness |
| 多步驟任務執行 | ReAct / ReWOO | LLM 在 loop 中，沒有形式化保證 |
| 有測試的程式碼改進 | AlphaEvolve | 測試就是 evaluator |
| 受政策約束的自動化 | HTN | Precondition 編碼了政策 |

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **HTN 沒有 operator。** 沒有 precondition/effect schema，健全性的主張就崩塌。ChatHTN 的「LLM 建議分解」需要 schema 來拒絕無效的步驟。
- **AlphaEvolve 沒有真正的 evaluator。** 「問 LLM 這段程式碼有沒有比較好」不是 fitness function。Evaluator 必須是確定性的而且很快。
- **過度工程 (Over-engineering)。** 大多數 agent 任務兩者都不需要。先試 ReAct 或 ReWOO。

```figure
htn-tree-expand
```

## 動手做 (Build It)

`code/main.py` 實作兩個 toy：

- 一個 stdlib HTN planner，包含 operator、method、precondition、effect，以及在沒有 method 符合複合 task 時啟動的 `LLMFallback`。這個「LLM」是腳本化的分解器，讓 planner 可以離線執行。
- 一個對算術程式做的 stdlib 演化式搜尋：成長運算式，使其輸出在一組測試集上最小化 `|f(x) - target|`。Evaluator 是確定性的。

執行：

```
python3 code/main.py
```

Trace 顯示 HTN planner 分解一個複合 task（途中有 LLM 備援），以及演化 loop 收斂到目標運算式。

## 實際使用 (Use It)

- **HTN planner**：`pyhop`、`SHOP3`，或為特定領域的政策強制執行自己打造。
- **ChatHTN**：研究程式碼；這個模式（符號式 + LLM 備援）可以順利移植到任何 HTN planner。
- **AlphaEvolve**：DeepMind 論文；這個模式（組合 + evaluator）是可重現的。OpenEvolve 等開源 fork 正在出現。
- **Agent framework**：目前沒有任何一個原生提供 HTN 或 AlphaEvolve。把它建成 subagent 或背景 worker。

## 交付 (Ship It)

`outputs/skill-hybrid-planner.md` 產生混合 planner 骨架（HTN 或演化式），並明確界定 LLM 的角色範圍。

## 練習 (Exercises)

1. 為 HTN planner 加上回溯：當 operator 的 postcondition 在執行時失敗，就回滾並嘗試下一個 method。
2. 為 ChatHTN 加一個 LLM-method 快取：當 LLM 在狀態模式 `P` 下分解 task `T` 時，把結果存起來。下一次呼叫時先檢查 method library。
3. 把演化式搜尋的 evaluator 換成真正的測試套件。演化出一個能通過 20 個測試案例的 sort 函式；回報收斂所需的世代數。
4. 讀 AlphaEvolve 的 evaluator 設計筆記。為你在意的領域（SQL 查詢最佳化、測試套件精簡、部署 YAML）設計一個 evaluator。
5. 結合：用 HTN 把複合 task 分解成子 task，再對每個子 task 的原始 operator 做演化式搜尋。它在哪裡表現出色，又在哪裡過度工程？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| HTN | 「階層式 planner」 | 帶有 operator、precondition、effect 的 task 分解 |
| Method | 「分解規則」 | 把複合 task 拆成子 task 的方式 |
| Operator | 「原始 action」 | 帶有 precondition 與 effect 的具體步驟 |
| ChatHTN | 「LLM + HTN」 | 沒有 method 符合時，符號式 planner 去問 LLM |
| AlphaEvolve | 「演化式程式碼搜尋」 | 一組 LLM mutate 程式碼；由確定性 evaluator 選擇 |
| Fitness function | 「Evaluator」 | 對輸出的確定性、機器可檢查分數 |
| Online method learning | 「快取的 LLM 分解」 | 儲存並一般化 LLM 的計畫，降低查詢成本 |

## 延伸閱讀 (Further Reading)

- [Gopalakrishnan et al., ChatHTN (arXiv:2505.11814)](https://arxiv.org/abs/2505.11814)：符號式 + LLM 混合 planner
- [Novikov et al., AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131)：以 LLM mutation 進行的演化式程式碼搜尋
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：何時用 planner，何時用簡單的 loop
