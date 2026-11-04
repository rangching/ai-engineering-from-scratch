# 給 LLM 用的 Swarm Optimization (PSO, ACO) (Swarm Optimization for LLMs (PSO, ACO))

> 受生物啟發的最佳化正在 LLM 上捲土重來。**LMPSO** (arXiv:2504.09247) 使用 PSO，其中每個 particle 的速度是一個 prompt，由 LLM 產生下一個候選；在結構化序列輸出（數學運算式、程式）上效果很好。**Model Swarms** (arXiv:2410.11163) 把每個 LLM 專家視為模型權重流形上的一個 PSO particle，回報在 9 個資料集上、只用 200 個實例，就比 12 個基準平均提升 **13.3%**。**SwarmPrompt** (ICAART 2025) 把 PSO + Grey Wolf 混合起來做 prompt 最佳化。**AMRO-S** (arXiv:2603.12933) 是受 ACO 啟發、用於 multi-agent LLM 路由的 pheromone 專家，**加速 4.7x**，路由證據可解釋，並以品質把關的非同步更新，把推論與學習解耦。這堂課在 prompt 參數空間上實作 PSO，在 agent 路由上實作 ACO，測量為什麼這些經典演算法適合 LLM 時代，以及什麼時候不適合。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 14 (Consensus and BFT)
**Time:** ~75 分鐘

## 問題 (Problem)

你有一個 prompt，在你的任務評測上得分 62%。你想改進它。天真的做法是不用梯度的手動微調，擴展性很差。強化學習需要獎勵訊號和足夠的 rollout 才能訓練。對 prompt 做 backprop 實際上不可行，因為 prompt 是離散的字串，不是可微分的參數。

經典的受生物啟發的最佳化，也就是適用連續搜尋空間的 PSO、適用路徑選擇的 ACO，正是為這種情況設計的：不用梯度、以群體為基礎、每次評估的成本很低。把它們與 LLM 配對，由 LLM 負責不用梯度的搜尋步驟，你就得到一個出乎意料實用的最佳化器。

同樣的 pattern 也適用於 multi-agent 系統中的 agent *路由*。ACO 風格的 pheromone 軌跡記錄哪個 agent 在哪種任務類型上表現最好，讓 router 利用這條軌跡，並讓 pheromone 衰減，使路線可以被重新發現。

## 核心概念 (Concept)

### PSO 複習 (Kennedy & Eberhart 1995) (PSO refresher)

Particle Swarm Optimization：連續搜尋空間中的 particle 群體。每個 particle 有位置 `x_i` 和速度 `v_i`。每次迭代：

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

其中 `p_best` 是 particle 自己的最佳值，`g_best` 是群體的最佳值，`w, c1, c2` 是慣性 + 認知 + 社會的權重，`r1, r2` 是隨機因子。

### 用在 LLM 輸出上的 PSO：LMPSO (PSO on LLM outputs — LMPSO)

arXiv:2504.09247 把 PSO 調整成用於 LLM 產生的結構化輸出（數學運算式、程式）。每個 particle 是一個候選輸出。速度是一個 *prompt*，描述如何把目前的輸出朝個人／全域最佳值修改。LLM 根據這個速度 prompt 產生新的輸出。速度的「慣性」是像「做小幅度的漸進式修改」這樣的 prompt。

在以下情況效果很好：
- 輸出是結構化的（可解析、可評估）。
- Fitness 是自動的（測試執行、算術運算評估）。
- 群體很小（~10-30 個 particle），讓 LLM 呼叫的總次數保持在可管理的範圍。

當 fitness 需要人類審查時效果不好，因為每次迭代的成本會高到難以負擔。

### Model Swarms

arXiv:2410.11163 把 PSO 從輸出層拿到 *模型* 層。每個「particle」是一個專家 LLM（參數）。群體透過不用梯度的更新，把參數朝集體最佳值移動。回報：在 9 個資料集上比 12 個基準平均提升 13.3%，每次迭代只用 200 個實例。

關鍵的洞見是，LLM 專家模型在共享的參數流形（adapter 權重、LoRA delta）中本來就彼此接近。在這個低維子空間上做 PSO，既便宜又有效。

### ACO 複習 (Dorigo 1992) (ACO refresher)

Ant Colony Optimization：螞蟻走過一張圖；每條路徑都有 pheromone 軌跡。螞蟻的移動機率以 pheromone 強度加權。完成任務的螞蟻依解的品質成比例地沉積 pheromone。Pheromone 隨時間衰減。

### AMRO-S：用於 agent 路由的 ACO (AMRO-S — ACO for agent routing)

arXiv:2603.12933 把 ACO 用於 multi-agent 路由。每種任務類型是一個「目的地」；每個 agent 是一條可能的路線。Pheromone 強化能產生好輸出的路線。主要貢獻：

- **可解釋的路由證據。** Pheromone 強度是人類可讀的訊號。
- **以品質把關的非同步更新。** Pheromone 只在品質檢查通過後才更新，把推論與學習解耦。
- 在 multi-agent 路由基準上 **加速 4.7x**。

品質把關很重要：沒有它，快但錯的 agent 會累積 pheromone，系統就會鎖定在糟糕的路線上。

### 何時對 LLM 使用 PSO / ACO (When to use PSO / ACO for LLMs)

**在以下情況使用 PSO：**
- 搜尋空間是連續的，或對應到連續的參數（prompt embedding、LoRA 權重、數值型的生成參數）。
- Fitness 便宜且自動。
- 群體可以很小（10-30）。

**在以下情況使用 ACO：**
- 你有路由或路徑選擇的問題。
- 決策會隨時間強化（同樣的任務類型會再回來）。
- 你需要路由決策的可解釋證據。

**在以下情況兩者都不要用：**
- Fitness 需要人類審查（每次迭代太昂貴）。
- 搜尋空間是離散的、屬於 PSO 無法涵蓋的組合型（改用 genetic algorithm）。
- 即時決策需要嚴格的延遲（PSO/ACO 相對於單次的啟發式方法收斂較慢）。

### 為什麼受生物啟發的方法仍然會贏 (Why bio-inspired still wins)

以梯度為基礎的方法需要可微分的訊號。LLM 的輸出和路由決策並不容易微分。偽梯度方法（強化學習訓練的 router、DPO 風格的 prompt 調整器）可行，但需要昂貴的訓練。

PSO 和 ACO 只需要一個 *評估器* 函式。只要你能為候選輸出或路由決策評分，就能在這個空間上最佳化。這讓適用門檻低很多。

### 實務上的限制 (Practical limits)

- **群體預算。** N 個 particle × T 次迭代 × 每次評估的成本。對於約 $0.02 / 次呼叫的 LLM 評估，20 個 particle 的 PSO 跑 50 次迭代，成本約 $20。請據此規劃。
- **探索與利用。** Pheromone 衰減率和 PSO 慣性之間需要取捨；衰減太快 → 忘記解；太慢 → 卡在早期的局部最佳解。
- **災難性漂移。** 如果 fitness 地形改變（新的資料分布），兩種演算法都可能先收斂、再發散。監控最佳 fitness 的穩定性。

```figure
swarm-stigmergy
```

## 動手做 (Build It)

`code/main.py` 實作：

- `LMPSO` — 對數值型 prompt 參數（temperature、top_k 權重）做 PSO。每個 particle 的「LLM 生成」被模擬為一個寫好腳本的 fitness 函式。執行演算法 30 次迭代，並顯示 g_best 的收斂。
- `AMRO_S` — ACO 風格的路由。3 個 agent、4 種任務類型、pheromone 矩陣、100 個被路由的任務。印出 (任務類型 → agent 選擇) 隨時間的分布，以顯示軌跡的形成。
- 比較：在相同的任務流上，隨機路由 vs ACO 路由。測量品質與延遲。

執行：

```
python3 code/main.py
```

預期輸出：
- LMPSO：g_best 的 fitness 在 30 次迭代中從隨機值進步到接近最佳。
- AMRO-S：pheromone 表在每種任務類型上穩定於正確的 agent；ACO 路由在品質上比隨機高約 30-40%，並降低延遲（更少的重試）。

## 實際使用 (Use It)

`outputs/skill-swarm-optimizer.md` 協助你在 LLM／agent 最佳化問題中，於 PSO、ACO、genetic algorithm 和以梯度為基礎的最佳化器之間做選擇。

## 交付 (Ship It)

- **從小處開始。** 10-20 個 particle，20-50 次迭代。只有在收斂曲線顯示明顯增益時才擴大。
- **每次迭代記錄 pheromone 或 g_best。** 沒有軌跡就除錯 swarm 最佳化器很痛苦。
- **以品質把關更新。** 尤其是 ACO 路由：快但錯的 agent 不能累積 pheromone。
- **在分布改變時重設衰減。** 當你的評測分布改變，老化的 pheromone 就過時了；重設，或暫時把衰減率加倍。
- **限制每次迭代的成本。** 輸出每次迭代成本的指標。每次迭代花 $500、只獲得 0.5% 增益的 PSO 不值得上線。

## 練習 (Exercises)

1. 執行 `code/main.py`。觀察 LMPSO 的收斂。把群體大小改成 5、10、20、50。收斂所需時間在多大的群體時趨於飽和？
2. 實作「災難性漂移」實驗：第 30 次迭代之後，改變 fitness 函式。PSO 適應得多快？重設 `p_best` 有幫助嗎？
3. 在 AMRO-S 加入品質把關：只有評測分數 > 0.7 的執行才沉積 pheromone。這和沒有把關的版本相比，收斂有什麼變化？
4. 閱讀 LMPSO (arXiv:2504.09247)。把論文中的「速度即 prompt」對應回你的數值速度。模擬中遺失了什麼，又保留了什麼？
5. 閱讀 AMRO-S (arXiv:2603.12933)。實作搭配非同步 pheromone 更新、解耦的「推論快速路徑」。在持續負載下，這如何改變系統延遲？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| PSO | 「Particle Swarm Optimization」 | Kennedy-Eberhart 1995。以群體為基礎、不用梯度的最佳化器。 |
| ACO | 「Ant Colony Optimization」 | Dorigo 1992。透過 pheromone 軌跡做路徑／路線最佳化。 |
| LMPSO | 「用 LLM 生成的 PSO」 | arXiv:2504.09247。速度是 prompt；LLM 產生候選。 |
| Model Swarms | 「對專家權重做 PSO」 | arXiv:2410.11163。在模型參數子空間上做不用梯度的更新。 |
| AMRO-S | 「用於 agent 路由的 ACO」 | arXiv:2603.12933。任務類型 × agent 的 pheromone 矩陣。 |
| p_best / g_best | 「個人／全域最佳」 | 每個 particle 與整個群體至今找到的最佳解。 |
| Pheromone | 「路由記憶」 | 邊上的強度；隨時間衰減；依品質沉積。 |
| Quality-gated update | 「只從好的執行中學習」 | 以品質檢查為條件的 pheromone 沉積。 |
| Catastrophic drift | 「分布偏移」 | Fitness 地形改變；舊的 p_best 與 pheromone 過時。 |

## 延伸閱讀 (Further Reading)

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — 1995 年的 PSO 論文
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — 1992 年 ACO 的基礎
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — 用於結構化 LLM 輸出的 PSO
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — 對模型權重子空間做 PSO
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — 帶品質把關、由 pheromone 驅動的路由
