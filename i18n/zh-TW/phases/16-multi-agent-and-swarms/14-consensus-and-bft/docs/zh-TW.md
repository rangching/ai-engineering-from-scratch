# Agent 的 Consensus 與 Byzantine Fault Tolerance (Consensus and Byzantine Fault Tolerance for Agents)

> 經典分散式系統的 BFT，遇上隨機性的 LLM。2025-2026 年出現三個研究方向：**CP-WBFT** (arXiv:2511.10400) 用 confidence probe 為每一票加權；**DecentLLMs** (arXiv:2507.14928) 採用無領導者設計，讓 worker 平行提案，再用 geometric median 聚合；**WBFT** (arXiv:2505.05103) 把加權投票與 Hierarchical Structure Clustering 結合，分出 Core 與 Edge 節點。"Can AI Agents Agree?" (arXiv:2603.01213) 的誠實實證結果是：即使只是純量的一致性，今天也很脆弱，單一個有欺騙性的 agent 就能攻破一個 Mixture-of-Agents。BFT 是必要的，但不充分。這堂課建一個最精簡的 BFT protocol，注入三種針對 agent 的攻擊（byzantine 說謊、阿諛式從眾、相關錯誤的單一文化），並測量各種 consensus 變體如何應對。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 07 (Society of Mind and Debate), Phase 16 · 13 (Shared Memory)
**Time:** ~75 分鐘

## 問題 (Problem)

你有 N 個 LLM agent，各自產生一個答案。它們意見不一。多數決挑到錯的那個，因為有兩個 agent 是相關的（同一個基礎模型、同樣的訓練資料、同樣的失敗模式）。第三個 agent 剛好以新穎的方式出錯，所以這個多數是假的多數。

現在再加一個有欺騙性的 agent：它故意說謊。或加一個阿諛的 agent：它同意最後發言的人。在經典 BFT 中，假設 Byzantine 節點占比 `f < n/3`，且行為任意。2026 年的現實是，LLM 節點即使誠實也是隨機的、跨模型彼此相關，並且受到彼此輸出的影響。你不能把它們當成獨立的 Bernoulli 投票者。

經典 BFT（PBFT，1999）沒有錯，只是不完整。它處理任意的位元翻轉。它不處理「三個誠實的 agent 因為共用訓練資料而共享同一個幻想」。這堂課從 PBFT 的基礎出發，再疊上三種 2025-2026 年的調整。

## 核心概念 (Concept)

### 經典 BFT 給了你什麼 (What classical BFT gives you)

Practical Byzantine Fault Tolerance（Castro & Liskov, OSDI 1999）能容忍 `f < n/3` 個 Byzantine 節點。這個 protocol 有三個階段（pre-prepare、prepare、commit）和兩個原語（簽章訊息、quorum certificate）。在 `n >= 3f + 1` 個誠實或惡意的節點之間，對單一值達成共識。

保證很強，但有幾個假設：

1. **獨立的錯誤。** Byzantine 節點之間不會協調。
2. **誠實節點真的誠實。** 誠實輸出的正確性不是問題；protocol 只負責對齊分歧。
3. **問題有 ground-truth 答案。** 對錯誤事實達成共識，仍然是共識。

LLM agent 違反這三項。跑同一個基礎模型的兩個 agent 共享錯誤。「誠實」的 LLM 仍然會幻想。而在模稜兩可的問題上，「真相」就是 agent 們決定的東西，沒有外部的 oracle。

### 三種針對 LLM 的攻擊 (The three LLM-specific attacks)

**Byzantine 說謊。** 一個 agent 故意輸出錯誤答案。若 `f < n/3`，經典 BFT 可以處理。

**阿諛式從眾 (Sycophantic conformity)。** 一個 agent 在投票前讀了別人的答案，然後跟最後發言的人對齊。不是惡意，但會與最大聲的聲音相關。經典 BFT 防不了這個，因為這個 agent 通過每一項簽章檢查。

**相關錯誤的單一文化 (Correlated-error monoculture)。** 三個 agent 共用同一個基礎模型。它們幻想出同一個錯誤答案。多數是錯的。經典 BFT 沒有幫助，因為三個都「誠實地」同意。

### 2025-2026 年的回應 (The 2025-2026 responses)

**CP-WBFT** (arXiv:2511.10400) — Confidence-Probed Weighted BFT。每個投票者在答案上附加 confidence probe（自我回報的機率，或獨立校準模型的預測）。投票權重隨 confidence 縮放。回報在完全圖上 BFT 改善 +85.71%。緩解對象：阿諛式從眾（從眾的 agent 對它順勢採取的立場，往往 confidence 較低）。

**DecentLLMs** (arXiv:2507.14928) — 無領導者。Worker agent 平行提案，evaluator agent 為提案打分，最終答案是評分後各立場的 geometric median。在 `f < n/2` 時仍然穩健。緩解對象：Byzantine 說謊與相關錯誤（geometric median 對離群值穩健，會被拉向密集的叢集，而不是被模型偏見影響的平均值）。

**WBFT** (arXiv:2505.05103) — 結合 Hierarchical Structure Clustering 的 Weighted BFT。投票權重依回應品質，加上從歷史學來的信任分數來指派。把 agent 分成 Core 和 Edge；Core agent 必須先達成共識，Edge agent 跟隨。緩解對象：可擴展性（Core 共識小而快），以及部分的單一文化問題（Core 可以為了多樣性而挑選）。

### 實證："Can AI Agents Agree?" (arXiv:2603.01213)

這篇論文測量多個前沿模型上的純量一致性（LLM agent 對單一數值達成一致）。發現令人不舒服：

- 即使沒有對手，LLM agent 在許多基準上對純量問題的分歧率也超過 30%。
- 單一個採用欺騙性 persona 的 agent，就能把 Mixture-of-Agents 的 consensus 拉離誠實基準 40 個百分點以上。
- 分歧率與模型多樣性相關：異質的 ensemble 比同質的分歧更多（好：錯誤不相關），但漂移得也更慢（壞：達成一致所需時間更長）。

重點：BFT 給你的是對齊輸出的機制，但它不會告訴你對齊後的輸出對不對。要與驗證（Phase 16 · 08 角色專門化）、多樣性（Phase 16 · 15 debate 變體）和 evaluator agent（Phase 16 · 24 基準）結合。

### 核心 protocol，精簡版 (The core protocol, stripped down)

針對 LLM agent 的最小 BFT 輪次：

```
1. task arrives; each agent i produces answer a_i
2. each agent attaches confidence probe c_i in [0, 1]
3. aggregator collects (a_i, c_i) from all n agents
4. aggregator groups by semantic cluster (equivalent answers)
5. aggregator computes weight for each cluster C:
     w(C) = sum_{i in C} c_i
6. winner = cluster with max weight, if max > threshold * sum(c_i)
   else: retry or escalate
7. minority clusters logged with provenance for post-hoc audit
```

Semantic clustering 這一步是 LLM 特有的轉折。「研究報告了 4.2%」和「4.2% 的提升」兩個答案是同一叢集。天真的字串相等檢查會漏掉這點。正式環境中請使用便宜的 embedding 模型或明確的正規化。

### 閾值調整 (Threshold tuning)

`threshold` 參數決定何時接受、何時重試。太低：你會接受薄弱的多數。太高：你什麼都不會接受。經驗範圍：`n=5-7` 個 agent 時為 0.5-0.67，`n` 越小越高。低於閾值時，升級給人類，或換一組不同的 agent ensemble。

### Consensus 沒有幫助的地方 (Where consensus does not help)

- **模稜兩可的問題。** 如果問題沒有 ground truth，consensus 就只是一種意見。就叫它意見。
- **複合問題。** 「寫程式並解釋」有兩個答案。分別獨立投票。
- **對抗性的多輪。** 如果 agent 能觀察先前的輪次並模仿（Du 2023 debate），它們會開始彼此同意，無論真相為何。限制輪數（通常 2-3 輪）。

```figure
swarm-consensus-wave
```

## 動手做 (Build It)

`code/main.py` 實作：

- `AgentVoter` — 寫好腳本的 policy，輸出 (answer, confidence)。
- `MajorityVote` — 經典的 plurality。
- `CPWBFT` — 帶 semantic clustering 的 confidence 加權投票。
- `DecentLLMs` — 對評分後的提案做 geometric-median 聚合。
- `Scenario` — 在三種攻擊 pattern 下執行每個聚合器。

實作的攻擊 pattern：

1. `byzantine`：一個 agent 以高 confidence 說謊。
2. `sycophancy`：一個 agent 複製它看到的第一個答案，並帶上相符的 confidence。
3. `monoculture`：三個 agent 以中等 confidence 共享一個錯誤答案（相關錯誤）。

執行：

```
python3 code/main.py
```

預期輸出：一張 (攻擊, 聚合器) -> 最終答案的表，並標出正確答案。Plurality 在 monoculture 的情況下失敗。CPWBFT 的 confidence 加權能緩解 sycophancy。當 monoculture 不到整體人口的一半時，DecentLLMs 的 geometric median 會拉向誠實的叢集。

## 實際使用 (Use It)

`outputs/skill-consensus-designer.md` 為 multi-agent ensemble 設計 consensus protocol：clustering 方法、加權、閾值，以及低於閾值輪次的升級政策。

## 交付 (Ship It)

在上線任何 consensus 機制之前：

- **至少用上述三種 pattern 做攻擊測試。** 你的 protocol 應該以可預測的方式失敗，而不是默默失敗。
- **記錄每個少數叢集** 並附上 provenance。少數叢集是你針對相關錯誤的早期預警系統。
- **強制限制輪數。** 不要「一直辯論到達成一致」，那是在獎勵阿諛。
- **把一致性與正確性分開。** Consensus 的輸出送到 verifier；verifier 獨立於 ensemble。
- **監控一致率。** 急遽上升代表從眾偏誤；急遽下降代表模型漂移。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 plurality 在 monoculture 攻擊下失敗，但當 monoculture 的 confidence 低於 0.7 時，CPWBFT 能部分緩解。
2. 加入第四種攻擊 pattern：**沉默棄權 (silent abstention)**，一個 agent 拒絕回答（「我不知道」）。每個聚合器該如何對待棄權？實作你的選擇。
3. 把 semantic clustering 從字串正規化換成 embedding 相似度（使用任何開源 embedding 模型）。Sycophancy 攻擊會發生什麼事？
4. 閱讀 CP-WBFT (arXiv:2511.10400)。實作 confidence-probe 的校準步驟（由獨立的校準模型檢查每個 agent 自我回報的 confidence）。測量在 monoculture 情境下的準確度增益。
5. 閱讀 "Can AI Agents Agree?" (arXiv:2603.01213)。重現一個簡化的純量一致性實驗：三個 agent、一個純量問題、欺騙性 persona 的 prompt。CPWBFT 或 DecentLLMs 抓得到嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| BFT | 「Byzantine fault tolerance」 | Castro-Liskov 1999 年的 protocol，在 `f < n/3` 個任意錯誤下達成共識。 |
| Byzantine | 「任何壞行為」 | 可以說謊、丟掉訊息、默默失效的節點，除了安全地當機之外什麼都做得出來。 |
| Confidence probe | 「你有多確定？」 | 附加在投票上、自我回報或由校準器預測的機率。 |
| Semantic clustering | 「同樣的答案，不同的措辭」 | 在計票之前，把等價的答案分組。 |
| Geometric median | 「穩健的中心」 | 使與樣本點距離總和最小的點。與平均值不同，它對離群值穩健。 |
| Monoculture | 「同一個模型，同樣的失敗」 | agent 共用訓練資料或基礎模型時，產生的相關錯誤。 |
| Sycophantic conformity | 「附和大聲的人」 | agent 的投票偏向最先／最大聲發言的人。 |
| Core/Edge | 「階層式 BFT」 | WBFT 的分法：小型 Core 先達成共識，Edge 節點跟隨。限制延遲。 |

## 延伸閱讀 (Further Reading)

- [Castro & Liskov — Practical Byzantine Fault Tolerance (OSDI 1999)](https://www.usenix.org/conference/osdi-99/practical-byzantine-fault-tolerance) — 基礎
- [CP-WBFT — Confidence-Probe Weighted BFT](https://arxiv.org/abs/2511.10400) — 依 confidence 為投票加權
- [DecentLLMs — leaderless multi-agent consensus](https://arxiv.org/abs/2507.14928) — geometric-median 聚合
- [WBFT — Weighted BFT with Hierarchical Structure Clustering](https://arxiv.org/abs/2505.05103) — 用 Core/Edge 分法限制延遲
- [Can AI Agents Agree?](https://arxiv.org/abs/2603.01213) — 純量一致性的脆弱性與欺騙性 persona 攻擊
