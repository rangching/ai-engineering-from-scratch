# ReWOO 與 Plan-and-Execute：解耦的 Planning (ReWOO and Plan-and-Execute: Decoupled Planning)

> ReAct 把思考和行動交錯在同一個串流裡。ReWOO 把它們分開：先做一份完整的計畫，再執行。token 少 5 倍，HotpotQA 準確率 +4%，而且可以把 planner 蒸餾到 7B 模型。Plan-and-Execute 把它一般化；Plan-and-Act 把它擴展到 web navigation。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼 ReWOO 的 Planner / Worker / Solver 分工，比 ReAct 的交錯 loop 更省 token、更穩健。
- 實作 plan DAG、依相依順序執行的 executor，以及組合 worker 輸出的 solver，全部只用 stdlib。
- 用 2026 年的「五種 workflow 模式」(Anthropic) 框架，判斷任務該用 plan-then-execute 還是交錯式 ReAct。
- 辨識長時程的 web 或 mobile 任務何時需要 Plan-and-Act 的合成 plan 資料。

## 問題 (The Problem)

ReAct 交錯的 thought-action-observation loop 簡單又有彈性，但每次 tool call 都必須帶著完整的先前 context，包括每一個先前的 thought。token 用量會隨深度呈二次方成長。更糟的是：當 tool 在 loop 中途失敗，模型得從錯誤 observation 重新推導整份計畫。

ReWOO（Xu et al., arXiv:2305.18323, May 2023）注意到這點，並下了一個賭注：一開始就規劃好全部，平行取得證據，最後再組合答案。一次 LLM call 做計畫，N 次 tool call 取得證據（可以平行），一次 LLM call 做 solve。取捨是犧牲彈性（計畫是靜態的），換來好得多的 token 效率和更清楚的失敗模式。

## 核心概念 (The Concept)

### 三個角色 (The three roles)

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner 產生一個 DAG。每個節點指明一個 tool、它的參數，以及依賴哪些先前節點（用 `#E1`、`#E2` 這樣的參照）。Worker 依拓撲順序執行節點。Solver 把一切組合起來。

### 為什麼少 5 倍 token (Why 5x fewer tokens)

ReAct 的 prompt 長度會隨步數線性增加。到第 10 步時，prompt 包含 thought 1 加 action 1 加 observation 1，再加 thought 2 加 action 2 加 observation 2，依此類推。每個中間步驟還會重複帶著原始 prompt。

ReWOO 只付一次 planner prompt（大），N 個小的 worker prompt（每個只有 tool call，沒有鏈），和一次 solver prompt。在 HotpotQA 上，論文測得 token 少約 5 倍，同時絕對準確率 +4。

### 為什麼更穩健 (Why it is more robust)

在 ReAct 中，如果 worker 3 失敗，loop 必須在串流途中從錯誤裡推理出路。在 ReWOO 中，worker 3 回傳一個錯誤字串；solver 在 context 裡連同原始計畫一起看到它，可以優雅地降級。失敗定位是以節點為單位，不是以步驟為單位。

### Planner 蒸餾 (Planner distillation)

論文的第二個結果：因為 planner 看不到 observation，你可以用 175B 老師模型的 planner 輸出來 fine-tune 一個 7B 模型。小模型負責 planning；推論時不需要大模型。這現在已是標準做法，許多 2026 年的正式 agent 會用小 planner 配大 executor，或反過來。

### Plan-and-Execute (2023)

LangChain 團隊在 2023 年 8 月的文章把 ReWOO 一般化，取了個模式名稱：Plan-and-Execute。前置的 planner 輸出步驟清單，executor 執行每一步，選用的 replanner 可以在觀察結果後修訂計畫。這比 ReWOO 更接近 ReAct（replanner 把 observation 帶回 planning），但保留了 token 節省。

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act 把這個模式擴展到長時程的 web 與 mobile agent。關鍵貢獻是合成 plan 資料：一個帶標籤的 trajectory 產生器，產出 plan 明確呈現的訓練資料。用來 fine-tune planner 模型，讓它們在類似 WebArena 的任務上，即使超過 30 到 50 步也能持續運作，而單一 ReAct trajectory 在那種長度會失去連貫性。

### 何時選哪個 (When to pick which)

| 模式 | 適用時機 |
|---------|------|
| ReAct | 短任務、環境未知、需要反應式的例外處理 |
| ReWOO | 結構化任務、tool 已知、對 token 敏感、證據可平行取得 |
| Plan-and-Execute | 類似 ReWOO，但在部分執行後可 replanning |
| Plan-and-Act | 長時程（>30 步）、web/mobile/computer-use |
| Tree of Thoughts | 搜尋值得付出成本時（Lesson 04） |

Anthropic 2024 年 12 月的建議：從最簡單的開始。如果任務是一次 tool call 加一個摘要，就不要建 ReWOO。如果任務是 40 步的研究作業，就不要只用 ReAct。

```figure
rewoo-plan
```

## 動手做 (Build It)

`code/main.py` 實作一個 toy ReWOO：

- `Planner`：腳本化的 policy，從 prompt 輸出 plan DAG。
- `Worker`：透過 registry 分派每個節點的 tool call。
- `Solver`：腳本化的組合器，讀取證據並產生最終答案。
- 相依解析：像 `#E1` 這樣的參照，會被替換成先前 worker 的輸出。

Demo 用兩步計畫回答「法國首都的人口是多少，四捨五入到百萬？」：(1) 查首都，(2) 查人口，然後 solve。

執行：

```
python3 code/main.py
```

Trace 先顯示完整計畫，接著是 worker 結果，最後是 solver 的組合。把 token 數（我們印出粗略的字元數）與 ReAct 風格的交錯執行比較，這類結構化任務是 ReWOO 勝出。

## 實際使用 (Use It)

LangGraph 把 Plan-and-Execute 當作 recipe 提供（ReAct 用 `create_react_agent`，plan-execute 用自訂 graph）。CrewAI 的 Flows 直接編碼了這個模式：你先定義任務，Flow DAG 負責執行。Plan-and-Act 的合成資料做法大多仍屬研究；執行期模式（明確的 plan DAG）已經透過 LangGraph 和 CrewAI Flows 進入正式環境。

## 交付 (Ship It)

`outputs/skill-rewoo-planner.md` 會根據 tool catalog，從使用者請求產生 ReWOO plan DAG。在交給 executor 之前，它會驗證計畫（無環、每個參照都已解析、每個 tool 都存在）。

## 練習 (Exercises)

1. 對互相獨立的 plan 節點平行執行 worker。在有 2 個平行群組的 6 節點 DAG 上，這能帶來什麼？
2. 加一個 replanner 節點，在任何 worker 回傳錯誤時觸發。要讓 ReWOO 變成 Plan-and-Execute，最小的改動是什麼？
3. 把 `Planner` 換成小模型（7B 等級），`Solver` 留在前沿模型。比較端到端品質，這個分工在哪裡失效？
4. 讀 ReWOO 論文第 4 節關於 planner 蒸餾的部分。從概念上重現 175B -> 7B 的結果：你需要什麼訓練資料，又如何評分 plan 品質？
5. 把 toy 移植成 Plan-and-Act 的 trajectory 形狀：plan 是序列，不是 DAG。哪些取捨會改變？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| ReWOO | 「Reasoning without observations」 | 先規劃，再平行取得證據，最後 solve；planning prompt 中不含 observation |
| Plan-and-Execute | 「LangChain 的 plan-execute 模式」 | 在執行後多一個選用 replanner 節點的 ReWOO |
| Plan-and-Act | 「規模化的 plan-execute」 | 明確的 planner/executor 分工，搭配長時程任務的合成 plan 訓練資料 |
| Evidence reference | 「#E1, #E2, ...」 | Plan 節點的佔位符，在分派時替換成先前 worker 的輸出 |
| Planner distillation | 「小 planner，大 executor」 | 用大型老師模型的 planner trace 去 fine-tune 小模型 |
| Token efficiency | 「更少的來回」 | 論文中在 HotpotQA 上比 ReAct 少 5 倍 token |
| DAG executor | 「拓撲分派器」 | 依相依順序執行 plan 節點；每一層內可平行 |

## 延伸閱讀 (Further Reading)

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323)：經典論文
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572)：用合成 plan 擴展的 planner-executor
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview)：framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：選擇可行的最簡單模式
