# Voting、Self-Consistency 與 Debate Topology (Voting, Self-Consistency, and Debate Topology)

> 最便宜的聚合方式：取樣 N 個獨立的 agent，做多數決。Wang et al. 2022 的 self-consistency 用的是同一個模型取樣 N 次。Multi-agent 把它延伸到 **異質 (heterogeneous)** 的 agent，以逃離單一文化：不同的模型、不同的 prompt、不同的 temperature、不同的 context。除了多數決之外，debate topology 也很重要：MultiAgentBench (arXiv:2503.01935, ACL 2025) 評估了 star / chain / tree / graph 的協調方式，發現 **graph 最適合研究型任務**，而超過約 4 個 agent 後會出現「協調稅 (coordination tax)」。AgentVerse (ICLR 2024) 記錄了兩種湧現的模式：志願行為 (volunteer) 與從眾行為 (conformity)，而從眾既是特性（找出 consensus），也是風險（groupthink，Lesson 24）。這堂課描繪 topology 的空間，建出每個變體，並測量協調稅。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 分鐘

## 問題 (Problem)

Debate 可以提升準確度 (Du et al., arXiv:2305.14325)。它也可能降低準確度。Debate 有沒有幫助，取決於四個結構性的選擇：

1. 誰跟誰說話（topology）。
2. 幾輪（Du 2023：輪數和 agent 數各自獨立都重要）。
3. Agent 是否異質（不同的基礎模型能打破單一文化）。
4. 是否有對抗性的聲音（steel-manning 對 straw-manning）。

把「跑 5 個 agent 然後投票」硬塞進任務的團隊，常常比單一 agent 還差。這些失敗不是隨機的，它們跟 topology 和異質性有關。這堂課就是 topology 的地圖。

## 核心概念 (Concept)

### Self-consistency，單一模型的基準 (Self-consistency, the single-model baseline)

Wang et al. 2022（"Self-Consistency Improves Chain of Thought Reasoning"）在 temperature > 0 下對同一個模型取樣 N 次，並對推理路徑的答案做多數決。在 GSM8K 上的結果：N=40 個樣本比單次 greedy decode 有顯著的提升。Self-consistency 是 multi-agent 投票的 single-agent 前身。

限制：self-consistency 只用一個基礎模型。錯誤在結構上就是相關的。如果模型有系統性偏誤，N 個樣本都會共享它。

### Multi-agent 投票，異質的延伸 (Multi-agent vote, the heterogeneous extension)

把 N 個樣本換成 N 個 *不同的* agent。不同的基礎模型（Claude、GPT、Llama）、不同的 prompt、不同的 tool 存取權。好處：錯誤不相關。代價：不同的 agent 成本不同；協調它們會增加額外負擔。

2026 年異質 debate 的標準名稱是 **A-HMAD**，也就是 Adversarial Heterogeneous Multi-Agent Debate。未被普遍採用，但論文用這個詞來指「不同的模型辯論，降低來自單一文化崩潰的相關錯誤」。

### 四種 Topology (The four topologies)

```
star                chain               tree                graph

    ┌─A─┐           A─B─C─D         ┌──A──┐              A───B
    │   │                           │     │              │ × │
    B   C                           B     C              D───C
    │   │                          / \   / \
    D   E                         D   E F   G           (fully connected)
```

Star：一個 hub，其他所有 agent 只與 hub 對話。等同於沒有回傳通道的 supervisor-worker。
Chain：線性，每個 agent 看到前一個的輸出。類似 pipeline。
Tree：階層式，用於 hierarchical agent 系統（Lesson 06）。
Graph：任意對任意。包含全連接的 clique 與任意 DAG。

### 協調稅（MultiAgentBench）(The coordination tax (MultiAgentBench))

MultiAgentBench（MARBLE，ACL 2025，arXiv:2503.01935）在包含研究、程式開發與規劃的任務組合上，對 star、chain、tree、graph 做基準測試。主要的測量結果：

- **Graph** topology 在研究型任務上勝出。資訊任意對任意流動；agent 可以互相批評。
- **Star** 在快速回答的事實型任務上勝出。Hub 負責過濾與整合。
- **Chain** 在逐步的 pipeline（分階段精煉）上勝出。
- **協調稅** 在 graph topology 中超過約 4 個 agent 後出現。實際耗時與 token 成本的成長速度快過品質。

4 個 agent 的天花板是經驗上的，不是根本性的。它反映了 2026 年 LLM 的 context 容量：每個 agent 的 context 被 peer 的輸出塞滿，一旦每個人都看得到所有人，加入第 N+1 個 agent 的邊際價值就下降了。

### Multi-Agent Debate 策略（"Should we be going MAD?"）(Multi-Agent Debate Strategies ("Should we be going MAD?"))

arXiv:2311.17371 是 2023 年對 MAD 策略的綜述。其他人也重現的主要發現：在 *結構上與* self-consistency *相似* 的 MAD 變體（獨立取樣 + 聚合），在相同預算下往往表現不如 self-consistency。當 agent 真正異質、且 debate 具有對抗性結構（有一個 agent 持反對論點）時，MAD 的幫助最大。

### AgentVerse 的湧現模式 (AgentVerse emergent patterns)

AgentVerse（ICLR 2024，https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf）記錄了兩種即使沒有明確設計，也會從 multi-agent debate 中湧現的行為：

- **Volunteer（志願）。** Agent 在沒有被要求的情況下主動提供協助（「下一步我可以接手」）。有用：它把工作分配給最有能力處理某個子任務的 agent。
- **Conformity（從眾）。** Agent 調整自己的立場去配合批評者，即使批評者是錯的。這是 debate 版本的阿諛（Lesson 14）。

從眾就是為什麼「辯論到達成一致為止」會獎勵霸凌者。限制輪數並搭配獨立的裁判可以緩解。

### 異質性：真正能推動準確度的旋鈕 (Heterogeneity: the actual knob that moves accuracy)

2024-2026 年實務文獻中的一個 pattern：把你 N 個 agent 當中的一個換成不同的基礎模型，帶來的準確度提升，比把 N 增加 1 還大。直覺是單一文化：每多一個獨立的錯誤來源，價值高過多一個相關的樣本。

極端來說，異質性勝過數量。在大多數有明確 ground truth 的任務上，三個不同的模型勝過同一個模型的五個副本。

### Jury 方法 (Jury methods)

Sibyl 框架（在 Minsky-LLM 文獻中被引用）將「jury」形式化：一小群專門的 agent，在每個階段透過投票來精煉答案。與單純的多數決不同，jury 有角色：一個 agent 負責交叉詰問，一個提供 context，一個為合理性打分。Jury 方法介於單純投票（便宜、容易單一文化）與完整 MAD（昂貴、容易從眾）之間。

### 投票加 debate 勝出的時候 (When vote-with-debate dominates)

- 問題有 ground truth（事實、數學、程式行為）。投票的收斂才有意義。
- Agent 可以存取不同的來源或 tool（有異質性可用）。
- 輪數有限制（通常 2-3），並有獨立的裁判或 verifier。
- 預算允許 3-5 個 agent。在 graph topology 上超過 5-7 個之後，協調稅佔主導。

### 投票加 debate 有害的時候 (When vote-with-debate hurts)

- 問題是意見型的。Agent 會收斂到看起來最有信心的答案，而不是最正確的。
- 所有 agent 共用一個基礎模型。單一文化使 consensus 失去意義。
- 輪數沒有限制。從眾每次都會贏。
- 任務很簡單。單一 agent 用 N=5 的 self-consistency 更便宜，準確度也一樣。

```figure
sw-debate-topology
```

## 動手做 (Build It)

`code/main.py` 實作：

- `run_star(agents, hub, question)` — hub 輪詢每個 worker 並聚合。
- `run_chain(agents, question)` — 循序精煉。
- `run_tree(root, children, question)` — 階層式，使用深度為 2 的聚合。
- `run_graph(agents, question, rounds)` — 全員對全員的 debate，輪數有上限。
- 寫好腳本的異質性刻度：每個 agent 有一個 `error_bias`，表示它的系統性錯誤傾向。
- 一套測量工具，在 N=3、5、7 時執行每種 topology，並回報 (accuracy, total_tokens, wallclock_simulated)。

執行：

```
python3 code/main.py
```

預期輸出：一張 topology × N → (accuracy, tokens, latency) 的表。Graph 在 N=3-5 的研究型任務上勝出；star 在快速事實型任務上勝出；N=7 的 graph 顯示出協調稅（延遲膨脹得比準確度快）。

## 實際使用 (Use It)

`outputs/skill-topology-picker.md` 是一個 skill，會讀取任務描述並推薦 topology（star / chain / tree / graph）、N（agent 數量）、異質性設定（要使用的基礎模型），以及輪數上限。

## 交付 (Ship It)

對任何 ensemble：

- 從使用單一強力基礎模型、**N=5 的 self-consistency** 開始。這是便宜的基準。
- 如果準確度很重要，升級到 **N=3 的異質投票**。測量差距。
- 只有當任務有結構（研究、多步驟）、且有限制輪數的可行性時，才升級到 **debate topology**。
- 一律記錄少數叢集。當少數持續是對的，你就得到一個多樣性的訊號。
- 準確度之外，也要對實際耗時和 token 做基準測試。「準確度更好但成本 10 倍」是一個商業決策。

## 練習 (Exercises)

1. 執行 `code/main.py`。繪製 graph topology 的協調稅曲線：準確度對 N、token 對 N。曲線在 N 為多少時出現轉折？
2. 實作 A-HMAD：三個偏誤刻意不同的 agent。在 Lesson 14 的 monoculture 攻擊下，全部偏誤相同的基準與 A-HMAD 相比如何？
3. 在 graph topology 加入一個不投票、只為最終 consensus 打分的「judge」角色。這會改變湧現的從眾行為嗎？
4. 閱讀 AgentVerse 論文（ICLR 2024）。找出你的實作最強烈展現的是哪種湧現行為。你能透過修改 prompt 引出相反的行為嗎？
5. 閱讀 MultiAgentBench (arXiv:2503.01935) 第 4 節（topology 實驗）。用你的測量工具，在論文中的一個任務上重現「graph 勝出研究型任務」的結果。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Self-consistency | 「取樣 N 次，投票」 | Wang 2022。單一模型，N 個 temperature>0 的樣本，對推理路徑做多數決。 |
| Heterogeneity | 「不同的模型」 | 由不同基礎模型或 prompt 家族組成的 ensemble。打破單一文化。 |
| MAD | 「Multi-agent debate」 | agent 在多輪中交換批評的通用名稱。見 Du 2023。 |
| A-HMAD | 「對抗式異質 MAD」 | 強調不同模型 + 對抗性結構的 MAD 變體。 |
| Topology | 「誰跟誰說話」 | Star、chain、tree、graph。決定資訊流。 |
| Coordination tax | 「報酬遞減」 | graph 上超過約 4 個 agent 後，成本成長得比品質快。 |
| Volunteer behavior | 「主動協助」 | AgentVerse 的湧現模式：agent 主動提出要接手一個步驟。 |
| Conformity behavior | 「在壓力下附和」 | AgentVerse 的湧現模式：agent 向批評者靠攏。 |
| Jury | 「小型專門小組」 | Sibyl 風格、帶角色（詰問者、context、評分者）的 ensemble。 |

## 延伸閱讀 (Further Reading)

- [Wang et al. — Self-Consistency Improves Chain of Thought Reasoning](https://arxiv.org/abs/2203.11171) — 單一模型的基準
- [Du et al. — Improving Factuality and Reasoning via Multiagent Debate](https://arxiv.org/abs/2305.14325) — agent 數和輪數各自獨立都重要
- [MultiAgentBench / MARBLE](https://arxiv.org/abs/2503.01935) — topology 基準，顯示 graph 最適合研究、chain 最適合 pipeline
- [Should we be going MAD?](https://arxiv.org/abs/2311.17371) — MAD 策略綜述；發現在相同預算下 MAD 常輸給 self-consistency
- [AgentVerse (ICLR 2024)](https://proceedings.iclr.cc/paper_files/paper/2024/file/578e65cdee35d00c708d4c64bce32971-Paper-Conference.pdf) — volunteer 與 conformity 的湧現模式
- [MARBLE repo](https://github.com/ulab-uiuc/MARBLE) — 參考的基準實作
