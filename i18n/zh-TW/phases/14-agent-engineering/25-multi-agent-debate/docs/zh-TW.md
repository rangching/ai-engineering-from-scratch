# Multi-Agent Debate 與協作 (Multi-Agent Debate and Collaboration)

> Du et al.（ICML 2024，"Society of Minds"）讓 N 個模型實例獨立提出答案，然後在 R 輪中反覆互相批評以達成收斂。能改善事實性、規則遵循和推理。稀疏拓撲在 token 成本上勝過 full mesh。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 12 (Workflow Patterns), Phase 14 · 05 (Self-Refine and CRITIC)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋 debate 協定：N 個提案者、R 輪、收斂到共同的答案。
- 描述為什麼 debate 能改善事實性、規則遵循和推理。
- 解釋稀疏拓撲：不是每個辯論者都需要看到其他所有人。
- 在腳本化的 LLM 上用 stdlib 實作 debate，包含 full-mesh 與稀疏兩種變體；量測 token 成本對準確率。

## 問題 (The Problem)

Self-Refine（Lesson 05）是單一模型批評自己，有 groupthink 的風險。CRITIC（Lesson 05）以外部 tool 為批評打底，但不一定可用。Debate 引入第三種模式：多個實例、交叉批評，透過意見不合達成收斂。

## 核心概念 (The Concept)

### Society of Minds (Du et al., ICML 2024)

- N 個模型實例對同一個問題獨立提出答案。
- 在 R 輪中，每個模型讀取其他模型的提案並加以批評。
- 模型依據批評更新自己的答案。
- R 輪之後，回傳收斂的答案。

原始實驗因成本考量使用 N=3、R=2。在困難問題（MMLU、GSM8K、Chess Move Validity、傳記生成）上，更多 agent 與更多輪次能提升準確率。

跨模型組合勝過單一模型的 debate：ChatGPT + Bard 一起用優於任何一個單獨使用。

### 稀疏拓撲 (Sparse topology)

"Improving Multi-Agent Debate with Sparse Communication Topology"（arXiv:2406.11776，2024-2025）顯示 full-mesh debate 不一定是最佳的。稀疏拓撲（star、ring、hub-and-spoke）能以較低的 token 成本達到相當的準確率。每個辯論者只看到部分對等者。

意涵：

- Full mesh N=5, R=3 = 5 × 3 = 15 個提案，每個讀取 4 個對等者 = 60 次批評操作。
- Star N=5, R=3（一個 hub + 4 個 spoke）= 15 個提案，spoke 只讀 hub = 12 次批評操作。

### Debate 何時有幫助 (When debate helps)

- **事實性。** N 個獨立提案，交叉檢查可減少 hallucination。
- **規則遵循。** Chess move validity：一個模型漏掉某條規則，其他模型能抓到。
- **開放式推理。** 多種框架逐步收斂到正確答案。

### Debate 何時有害 (When debate hurts)

- **對延遲敏感的 UX。** N × R 個串行輪次是你可能承擔不起的延遲。
- **對成本敏感的規模。** 每個問題 N × R 的 token。
- **簡單的事實查詢。** 一次查詢比五次辯論便宜。

### 2026 年的實際實例 (2026 practical instantiations)

- **Anthropic orchestrator-workers**（Lesson 12）：debate 的一種變體，附帶綜合步驟。
- **LangGraph supervisor**（Lesson 13）：中央 router + 專家 agent 可以把 debate 實作成一個節點。
- **OpenAI Agents SDK**（Lesson 16）：agent 來回 handoff 做迭代批評。
- **Multi-agent eval**：搭配 debate 與 evaluator-optimizer 以取得 eval 訊號。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **收斂崩塌 (Convergence collapse)。** 所有 agent 收斂到第一個錯誤的答案。用強制意見不合的輪次來緩解。
- **Hub 失敗 (Hub failure)。** 在 star 拓撲中，一個壞的 hub 會污染所有人。輪替或使用多個 hub。
- **Prompt 同質化 (Prompt homogenization)。** 所有 agent 使用相同的 prompt；它們會產生相同的答案。使用多樣的 prompt 和/或模型。

```figure
debate-converge
```

## 動手做 (Build It)

`code/main.py` 實作 stdlib 的 debate：

- `Debater` 類別（腳本化的 LLM，每個辯論者有各自的意見漂移）。
- `FullMeshDebate` 與 `SparseDebate` 執行器。
- 三個問題：一個事實性、一個規則型、一個推理型。
- 指標：收斂的答案、收斂所需輪次、批評操作總數。

執行：

```
python3 code/main.py
```

輸出：每種協定的準確率與成本；稀疏拓撲在 3 題中的 2 題以較低成本達到與 full mesh 相同的結果。

## 實際使用 (Use It)

- **Anthropic orchestrator-workers** 用於簡單的 2-3 個 worker 的 debate。
- **LangGraph** 用於帶 checkpoint 的有狀態多輪 debate。
- **自行打造** 用於研究或特殊的正確性保證。

## 交付 (Ship It)

`outputs/skill-debate.md` 建立可設定拓撲、N、R 和收斂規則的 multi-agent debate 骨架。

## 練習 (Exercises)

1. 實作「強制意見不合」規則：在第 1 輪，每個辯論者都必須產生不同的提案。量測它對收斂速度的影響。
2. 加入依信心加權的彙整：辯論者回傳 (answer, confidence)；彙整器依信心加權。有幫助嗎？
3. 把其中一個「agent」換成意見不同的另一個腳本化 LLM。異質性能改善準確率嗎？
4. 在你的 3 個問題上量測 full mesh 與稀疏的 token 成本。繪製成本對準確率的圖。
5. 讀 Society of Minds 論文。把你的 toy 移植到 N=5、R=3。什麼會壞掉？什麼會變好？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Debate | 「Multi-agent 批評」 | N 個提案者、R 輪交叉批評、收斂 |
| Full mesh | 「每個人都讀每個人」 | 每一輪每個辯論者都讀取每個對等者 |
| Sparse topology | 「有限的對等者視野」 | 辯論者只讀取部分對等者 |
| Hub-and-spoke | 「Star 拓撲」 | 一個中央辯論者，N-1 個 spoke 只讀 hub |
| Convergence | 「達成共識」 | 辯論者收斂到共同的答案 |
| Society of Minds | 「Du et al. 的 debate 論文」 | ICML 2024 的 multi-agent debate 方法 |

## 延伸閱讀 (Further Reading)

- [Du et al., Improving Factuality and Reasoning in Language Models through Multiagent Debate (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325)：經典的 multi-agent debate
- [Sparse Communication Topology (arXiv:2406.11776)](https://arxiv.org/abs/2406.11776)：稀疏拓撲的結果
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents)：把 orchestrator-workers 視為 debate 的變體
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：單一模型自我批評的對應方式
