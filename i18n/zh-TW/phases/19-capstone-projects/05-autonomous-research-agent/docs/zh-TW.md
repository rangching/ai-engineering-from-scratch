# Capstone 05 — 自主研究 Agent (Autonomous Research Agent，AI-Scientist 等級)

> Sakana 的 AI-Scientist-v2 發表了完整論文。Agent Laboratory 跑了實驗。Allen AI 分享了 trace。2026 年的樣貌是：對實驗做 plan-execute-verify 的樹搜尋、有預算的成本、sandbox 內執行程式碼、具視覺回饋的 LaTeX 寫手，以及自動化的 NeurIPS 風格審稿者組合。這個 capstone 要你做一個，在每篇論文 $30 內端到端跑完，並通過 Sakana 所記錄的 sandbox 逃逸 red team。

**Type:** Capstone
**Languages:** Python (agent + sandbox), LaTeX (output)
**Prerequisites:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Phases exercised:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Time:** 40 小時

## 問題 (Problem)

自主研究 agent 在 2026 年跨過了一道門檻。Sakana AI 的 AI-Scientist-v2 登上 Nature，其產生的論文通過了 workshop 的同儕審查。ShinkaEvolve (ICLR 2026) 把這條路線延伸到演化式假說。AMD 的 Agent Laboratory 釋出了可重現的 trace。這些 agent 並不神奇，它們是在一棵候選實驗樹上跑 plan-execute-verify 迴圈，搭配成本上限、固定 seed 的 sandbox 與自動審查。功夫在迴圈、預算與安全機制。

你要靠針對窄領域的 seed idea 實作一個，來學會這個迴圈 (例如在 100M 參數的 transformer 上做 attention 稀疏性的 ablation)。價值不在第一次執行就發現新東西，而在基礎建設：樹搜尋、實驗 sandbox、寫手與審稿者迴圈、red-team 報告。Sakana 團隊記錄過 sandbox 逃逸的失敗案例；你的 agent 必須通過同樣的 red team。

## 核心概念 (Concept)

這個 agent 是一個 best-first 樹搜尋。節點是實驗規格：(hypothesis, config, code, expected outcome)。Expand 步驟用小幅修改提出子節點 (換 optimizer、調整 batch size、移除某個元件做 ablation)。每個子節點在全新的 sandbox 中執行，並有硬性資源上限。結果回饋到評分函式，依 (novelty × quality × 剩餘預算) 對節點排序。樹會一直成長到預算耗盡，然後把最佳分支寫成論文。

寫手是多模態的。它產生 LaTeX 草稿、編譯、渲染圖表，再把渲染好的 PDF 送回 Claude Opus 4.7 的 vision 模式，針對版面、圖表可讀性、主張與證據的對應做批評。由五位 LLM 評審組成的審稿者組合，輸出 NeurIPS 風格的分數 (novelty、rigor、clarity、reproducibility、impact)；平均低於閾值時，論文會帶著批評回到寫手手上。

安全是承重牆。每個實驗都在 E2B 或 Daytona sandbox 中執行，沒有網路對外連線、牆鐘時間有限，資源上限固定。Agent 的程式碼產生步驟會通過政策層，阻擋會逃出 sandbox 的 syscall。Red-team 報告重現 Sakana 所記錄的攻擊面 (fork bomb、檔案系統逃逸、LLM 寫出的網路呼叫)。

## 架構 (Architecture)

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## 技術堆疊 (Stack)

- 編排：具備 checkpointing 與人工核准關卡的 LangGraph
- 樹搜尋：針對實驗節點自訂的 best-first (沿用 Sakana v2 的 AB-MCTS 風格)
- Sandbox：每個實驗一個 E2B，Docker-in-Docker 作為備援；以 cgroups 設定資源上限
- 文獻：Semantic Scholar Graph API + OpenAlex + 本機 FAISS 摘要快取
- 寫手：LaTeX 範本 + Claude Opus 4.7 (vision 模式) 做圖表批評與版面
- 審稿者：5 位評審組成的組合 (Opus 4.7、GPT-5.4、Gemini 3 Pro、DeepSeek R1、Qwen3-Max)，加權彙整
- 實驗框架：實體實驗用 PyTorch 2.5，記錄用 W&B
- 可觀測性：agent trace 用 Langfuse，每篇論文 $30 硬性預算

```figure
ce-experiment-tree
```

## 動手做 (Build It)

1. **Seed 與領域範圍。** 取一個 seed idea (例如「研究 1B 以下 transformer 的 attention map 稀疏模式」)。定義搜尋空間：模型、資料集、運算預算。

2. **文獻檢閱。** 向 Semantic Scholar + OpenAlex 查詢 50 篇被引用最多的相關論文；在本機快取摘要；產生一頁的領域摘要。

3. **樹的骨架。** 以 seed 假說初始化根節點。實作 `expand(node) -> children`，用小幅修改提案 (每個子節點只改一項 config)。把 `score(node)` 實作為 novelty × quality × budget 的加權項。

4. **Sandbox 包裝。** 每個實驗執行 `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` (或等效的 E2B 政策)。Seed 寫入 sandbox；輸出以唯讀方式掛載回外部。

5. **Plan-execute-verify 迴圈。** `plan` 提出子節點。`execute` 執行 sandbox，擷取 log 與指標。`verify` 對指標做單元檢查 (loss 有下降嗎？ablation 有隔離出效果嗎？)。失敗的節點會把失敗原因存在樹上。

6. **寫手。** 預算用完後，選出最佳分支。用 matplotlib 渲染圖表。以分支 trace 為 context，透過 Claude Opus 4.7 產生 LaTeX 草稿。編譯。把編譯好的 PDF 送回 Opus 4.7 vision 做批評。反覆迭代。

7. **審稿者組合。** 五位評審依 NeurIPS 風格的評分標準，對草稿的 (novelty、rigor、clarity、reproducibility、impact) 評分。平均 < 4.0/5 時，帶著批評回到寫手。最多重寫 3 次後強制停止。

8. **Red team。** 建立或整合一組針對 sandbox 的對抗任務：fork bomb、網路外洩嘗試、檔案系統逃逸、LLM 寫出的 shell 特殊字元。確認全部被阻擋。寫下發現。

9. **可重現性。** 每篇論文都附帶樹搜尋 trace JSON、seed、W&B run 連結、sandbox 設定，以及一份能端到端重現的 README。

## 實際使用 (Use It)

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## 交付 (Ship It)

`outputs/skill-ai-scientist.md` 是交付成果。給定 seed idea + 領域 + $30 預算，它會執行完整管線，並輸出可供審查的論文與可重現性 bundle。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 論文品質 | 對照已發表 workshop 論文的盲審評分標準 |
| 20 | 實驗嚴謹度 | 基準線、seed、ablation；每個主張都有結果表中的儲存格支持 |
| 20 | 成本與運算紀律 | 強制執行每篇 $30 上限，並以 Langfuse 追蹤 |
| 20 | 安全性 | Sandbox red team 通過；驗證網路政策與 kill-switch |
| 15 | 可重現性 | 以相同 seed 一行指令重跑，即可重現論文 |
| **100** | | |

## 練習 (Exercises)

1. 對同一領域的三個不同 seed idea 執行管線。比較樹搜尋有哪些部分重疊，找出重複浪費的運算。

2. 對預估超過 $5 的節點，在實驗執行前加入 human-in-the-loop 關卡。衡量總成本下降多少。

3. 把審稿者組合換成單一評審。在一組已知品質差的論文 held-out 集合上，衡量誤接受率。

4. 加入網路外洩的 red team 測試：agent 寫出嘗試 `curl` 外部位址的程式碼。確認 `--network=none` 政策會擋下，並記錄這次嘗試。

5. 把你的樹搜尋與平面隨機基準線 (相同預算、無展開策略) 比較。回報 novelty × quality 的增益。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Tree search | 「AB-MCTS 式展開」 | 以 novelty×quality×budget 分數，對實驗節點做 best-first 探索 |
| Sandbox | 「實驗隔離」 | 沒有網路、CPU/記憶體有上限、seed 固定、輸入唯讀的容器 |
| Vision critique | 「先渲染再閱讀」 | 把論文編譯成 PDF，送回 VLM 做版面與主張證據的批評 |
| Reviewer ensemble | 「自動同儕審查」 | 多位 LLM 評審以 NeurIPS 評分標準為論文評分；加權彙整後作為管線的關卡 |
| Novelty score | 「這是新的嗎？」 | 啟發式分數，離 50 篇文獻快取越近，扣分越多 |
| Cost ceiling | 「$ 預算」 | 每篇論文總花費的硬性上限；Langfuse 計數器 + 執行前估算 |
| Red team | 「Sandbox 逃逸稽核」 | 若政策有誤就會逃出 sandbox 的對抗任務 |

## 延伸閱讀 (Further Reading)

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — 參考用的正式研究 agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — 原始方法論
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — 演化式延伸
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — 多角色研究實驗室框架
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — 參考用的編排層
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — 文獻搜尋
- [E2B sandboxes](https://e2b.dev) — 參考用的實驗隔離
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/ReviewerGuidelines) — 審稿者組合所編碼的評分標準
