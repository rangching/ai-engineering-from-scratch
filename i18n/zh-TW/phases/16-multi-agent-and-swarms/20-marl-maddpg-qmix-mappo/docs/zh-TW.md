# MARL：MADDPG、QMIX、MAPPO (MARL — MADDPG, QMIX, MAPPO)

> Multi-agent 協調的強化學習淵源，到了 2026 年仍然影響著 LLM-agent 系統。**MADDPG** (Lowe et al., NeurIPS 2017, arXiv:1706.02275) 引入了 Centralized Training, Decentralized Execution (CTDE)：訓練時每個 critic 看得到所有 agent 的狀態與行動；測試時只有本地的 actor 在執行。適用於合作、競爭和混合場景。**QMIX** (Rashid et al., ICML 2018, arXiv:1803.11485) 是帶單調混合網路的 value decomposition；每個 agent 的 Q 組合成聯合 Q，使 `argmax` 能乾淨地分散，在 StarCraft Multi-Agent Challenge (SMAC) 上佔主導地位。**MAPPO** (Yu et al., NeurIPS 2022, arXiv:2103.01955) 是帶集中式 value function 的 PPO；在 particle-world、SMAC、Google Research Football、Hanabi 上，只需極少調校就「出奇地有效」。這些支撐了必須分散式行動的 agent 團隊的 policy 訓練。MAPPO 是 **2026 年合作式 MARL 的預設基準**。這堂課用一個小型 grid-world 玩具建出每一個，在接觸 LLM-agent 訓練之前，把這三個想法變成肌肉記憶。

**Type:** Learn
**Languages:** Python (stdlib, small NumPy-free implementations)
**Prerequisites:** Phase 09 (Reinforcement Learning), Phase 16 · 09 (Parallel Swarm Networks)
**Time:** ~90 分鐘

## 問題 (Problem)

LLM-agent 系統越來越常為 agent 之間的協調訓練 policy：何時該讓步、何時該行動、該呼叫哪個 peer。告訴你如何訓練這類 policy 的文獻，是 Multi-Agent Reinforcement Learning (MARL)，它早於 LLM 浪潮，並且有一小組佔主導地位的演算法。

沒有 pattern 詞彙就讀 MARL 論文很痛苦。Centralized training with decentralized execution (CTDE)、value decomposition 和 centralized critic 不是流行語，而是針對特定問題的特定答案：

- 獨立 RL（每個 agent 自己學）從每個 agent 的角度看是非平穩的。不好。
- 集中式 RL（一個 agent 控制全部）無法擴展，也違反執行時的限制。
- CTDE 兩全其美：用全域資訊訓練，用本地 policy 部署。

## 核心概念 (Concept)

### 論文使用的三種環境 (Three environments the papers use)

- **Particle World（multi-agent particle env）。** 簡單的 2D 物理，有合作／競爭任務。MADDPG 最初的測試平台。
- **StarCraft Multi-Agent Challenge (SMAC)。** 合作式的微操，部分可觀察。QMIX 的測試平台。離散行動、連續狀態。
- **Google Research Football、Hanabi、MPE。** MAPPO 的基準。

不同的環境有不同的行動／觀察型別。演算法據此選擇。

### MADDPG (2017)：CTDE pattern

每個 agent `i` 有一個 actor `mu_i(o_i)`，把它自己的觀察對應到行動。每個 agent 也有一個 critic `Q_i(x, a_1, ..., a_n)`，在訓練時看得到所有觀察與所有行動。Actor 依據 critic 的評估，以 policy gradient 更新。

```
actor update:    grad_theta_i J = E[grad_theta mu_i(o_i) * grad_a_i Q_i(x, a_1..n) at a_i=mu_i(o_i)]
critic update:   TD on Q_i(x, a_1..n) given next-state joint estimate
```

為什麼用 CTDE：訓練時我們知道每個人的行動；我們用這個來降低每個 critic 的變異。部署時，每個 agent 只看到 `o_i` 並呼叫 `mu_i(o_i)`。

失敗模式：critic 隨 N 個 agent 而成長（輸入包含所有行動）。不做近似的話，擴展不到約 10 個 agent 以上。

### QMIX (2018)：Value decomposition

只適用於合作。聯合行動價值 `Q_tot` 是各 agent Q 值的單調混合函式：

```math
Q_{\text{tot}}(\tau, a) = f\big(Q_1(\tau_1, a_1), \ldots, Q_n(\tau_n, a_n)\big), \quad \frac{\partial f}{\partial Q_i} \ge 0
```

單調性保證 `argmax_a Q_tot` 可以由每個 agent 獨立選擇 `argmax_{a_i} Q_i` 來計算。這正是你需要的 **分散式執行特性**。訓練時，混合網路從各 agent 的 Q 產生 `Q_tot`。

為什麼 QMIX 在 SMAC 上勝出：合作式的 StarCraft 微操有同質的 agent、本地觀察、全域獎勵，非常適合 value decomposition。

失敗模式：單調性的限制很嚴苛；有些任務的獎勵結構無法做單調分解（一個 agent 為團隊犧牲）。擴展版本（QTRAN、QPLEX）放寬了這一點。

### MAPPO (2022)：被忽略的預設 (MAPPO — the overlooked default)

Multi-Agent PPO：帶集中式 value function 的 PPO。每個 agent 有自己的 policy；所有 agent 共用（或各自擁有）能看到完整狀態的 value function。Yu et al. 2022 在五個基準上把 MAPPO 對比 MADDPG、QMIX 及其擴展版本，發現：

- MAPPO 在 particle-world、SMAC、Google Research Football、Hanabi、MPE 上持平或勝過 off-policy 的 MARL 方法。
- 只需極少的超參數調校。
- 訓練穩定；跨 seed 可重現。

在這篇論文之前，社群低估了 on-policy MARL。2026 年，MAPPO 是合作式 MARL 的預設基準；任何新方法都必須打敗它。

### 為什麼 LLM-agent 工程師應該在意 (Why LLM-agent engineers should care)

三種直接的用途：

1. **Router 訓練。** 一個 meta-agent 選擇哪個 sub-agent 處理任務。這是有 N 個分散式 sub-agent 和一個集中式 router 的 MARL 問題。MAPPO 適合。
2. **角色湧現。** 在 generative-agent 模擬中，訓練 agent 隨時間採取互補的角色，其實是偽裝起來的 MARL 問題。QMIX 風格的 value decomposition 在結構上就強制了互補性。
3. **Multi-agent tool 使用。** 當 agent 共用 tool 並競爭預算時，透過 CTDE 訓練它們，能產出尊重資源限制、可部署的本地 policy。

實務上的但書：2026 年，大多數正式的 LLM-agent 系統是用 prompt 來設定 policy，而不是訓練。當你 (a) 有大量互動資料、(b) 有明確的獎勵訊號、(c) 願意投資訓練基礎設施時，才會用到 MARL。

### 作為 RL 之外設計 pattern 的 CTDE (CTDE as a design pattern beyond RL)

即使不做訓練，CTDE 也是一個有用的架構 pattern：

- *設計* 時，假設團隊完全可見。
- *執行期*，強制分散式執行：每個 agent 只看到 `o_i`。

這個 pattern 逼你把每個 agent 的狀態明確化，並事先思考部分可觀察性。許多正式的 multi-agent 系統默默地到處假設共享狀態，CTDE 的紀律可以防止這件事。

### 非平穩性問題 (The non-stationarity problem)

當多個 agent 同時學習時，每個 agent 的環境（包含其他 agent 的 policy）是非平穩的。經典的單 agent RL 證明就此失效。本課的 MARL 演算法都在處理這個問題：

- MADDPG：全域 critic 看得到所有行動，所以它的價值估計是平穩的。
- QMIX：value decomposition 把學習移到最佳性定義明確的聯合 Q 空間。
- MAPPO：集中式 value function 抑制來自其他 policy 變化的變異。

在 LLM-agent 系統中，非平穩性表現為「我的 agent 上個月還能用，現在上游那個 agent 改了，我的就怪怪的」。用 CTDE 訓練 MARL 是有原則的解法；prompt 層級的修法比較快，但比較不持久。

### 這堂課不涵蓋的部分 (What this lesson does NOT cover)

訓練真正的網路是 Phase 09 的主題。這堂課建立的是寫好腳本 policy 的版本，在不做梯度更新的情況下展示 CTDE、value-decomposition 和 centralized-value 這幾種 pattern。目標是在你拿起完整的 MARL 函式庫（PyMARL、MARLlib、RLlib multi-agent）之前，先內化這些 pattern。

```figure
sw-ctde
```

## 動手做 (Build It)

`code/main.py` 實作三種 pattern 的示範，全都在一個小型的 2 agent 合作式 grid-world 上：

- 環境：4x4 網格上的 2 個 agent，一個獎勵顆粒。只要任一 agent 抵達顆粒，獎勵 = 1；任務結束。
- `IndependentAgents` — 每個 agent 把其他 agent 當作環境。基準。
- `MADDPGStyle` — 集中式 critic 計算聯合價值；actor policy 依它更新。寫好腳本的 policy 改進。
- `QMIXStyle` — 帶單調 mixer 的 value decomposition。
- `MAPPOStyle` — 集中式 value function；policy 對照共享的 baseline 更新。

四者執行相同的 episode，並回報平均抵達目標所需步數。CTDE 的變體收斂到比獨立基準更短的路徑。

執行：

```
python3 code/main.py
```

預期輸出：獨立的 agent 平均約需 ~6 步；CTDE 的變體收斂到約 ~3.5 步（4x4 網格的最佳值是 3）。儘管 policy 是寫好腳本的，pattern 的差異仍然顯現出來。

## 實際使用 (Use It)

`outputs/skill-marl-picker.md` 是一個 skill，為指定的 multi-agent 任務挑選 MARL 演算法：合作 vs 競爭、同質 vs 異質、行動空間型別、規模、獎勵訊號。

## 交付 (Ship It)

MARL 在正式環境中很少見。當你真的用到時：

- **從 MAPPO 開始。** 2022 年的論文把它確立為基準；先重現它，可以省下追逐更花俏方法的好幾週。
- **記錄每個 agent 的觀察與行動串流。** 沒有每個 agent 的追蹤，就無法除錯 MARL。
- **把訓練程式碼與執行程式碼分開。** CTDE 是一種紀律；讓執行路徑真的只看得到 `o_i`。
- **獎勵塑形的警告。** MARL 對獎勵設計極度敏感。塑形裡只要有一個協調上的 bug，agent 就會學著利用它。要執行對抗性測試。
- **對 LLM agent**，先考慮 prompt 層級的 policy。只有在互動資料 + 獎勵訊號 + 基礎設施都具備時，才投資 MARL 訓練。

## 練習 (Exercises)

1. 執行 `code/main.py`。測量獨立 agent 和 MAPPO 風格 agent 之間抵達目標步數的差距。在 6x6 網格上，差距是擴大還是縮小？
2. 實作競爭的變體：兩個 agent、一個顆粒，只有第一個抵達的獲得獎勵。哪個 pattern 能乾淨地處理競爭？歷史上是 MADDPG。
3. 閱讀 MADDPG (arXiv:1706.02275) 第 3 節。用你自己的話，在 pseudocode 中符號化地實作精確的 critic 更新規則。
4. 閱讀 MAPPO (arXiv:2103.01955)。作者為什麼主張集中式 value + PPO 在他們的基準上勝過 off-policy 的 MARL？列出三個最有力的主張。
5. 把 CTDE 當作設計 pattern，套用到一個假想的 LLM-agent 系統（例如 research agent + summarizer + coder）。設計時可得、但執行期無法取得的聯合資訊是什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MARL | 「Multi-Agent RL」 | 用於 multi-agent 系統的強化學習。 |
| CTDE | 「Centralized Training, Decentralized Execution」 | 用全域資訊訓練；用本地 policy 部署。 |
| MADDPG | 「Multi-Agent DDPG」 | 每個 agent 的 critic 看得到所有觀察 + 行動的 CTDE。 |
| QMIX | 「Value decomposition」 | 對各 agent 的 Q 做單調混合。僅適用合作。 |
| MAPPO | 「Multi-Agent PPO」 | 帶集中式 value function 的 PPO。2026 年的預設基準。 |
| Value decomposition | 「各別 Q 的總和」 | 把聯合 Q 表示成各 agent Q 的單調函式。 |
| Non-stationarity | 「移動的目標」 | 隨著其他 agent 學習，每個 agent 的環境也在變。MARL 的核心問題。 |
| On-policy / off-policy | 「從目前的／重播的資料學習」 | PPO 是 on-policy（MAPPO）；DDPG 和 Q-learning 是 off-policy。 |
| SMAC | 「StarCraft Multi-Agent Challenge」 | 合作式微操基準；QMIX 的主場。 |

## 延伸閱讀 (Further Reading)

- [Lowe et al. — Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments](https://arxiv.org/abs/1706.02275) — MADDPG；NeurIPS 2017
- [Rashid et al. — QMIX: Monotonic Value Function Factorisation for Deep Multi-Agent Reinforcement Learning](https://arxiv.org/abs/1803.11485) — QMIX；ICML 2018
- [Yu et al. — The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games](https://arxiv.org/abs/2103.01955) — MAPPO；NeurIPS 2022
- [BAIR blog post on MAPPO](https://bair.berkeley.edu/blog/2021/07/14/mappo/) — 對 MAPPO 結果易讀的說明
- [SMAC repository](https://github.com/oxwhirl/smac) — StarCraft Multi-Agent Challenge
