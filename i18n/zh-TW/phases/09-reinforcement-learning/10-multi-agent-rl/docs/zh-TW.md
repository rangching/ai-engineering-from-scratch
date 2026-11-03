# Multi-Agent RL

> Single-agent RL 假設 environment 是 stationary。把兩個正在學習的 agent 放進同一個世界，這個假設就破功了：每個 agent 都是對方 environment 的一部分，而且兩者都在改變。Multi-agent RL 就是一套技巧，用來在 Markov 假設不再成立時，仍讓學習收斂。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 分鐘

## 問題 (The Problem)

學習在房間裡導航的機器人，是 single-agent RL 問題。足球隊不是。AlphaStar 對上 StarCraft 對手不是。競標 agent 的市場不是。兩輛車在四向停車路口協商不是。許多現實世界中多對多的問題都不是。

在每個 multi-agent 的情境中，從任何一個 agent 的角度來看，其他 agents *就是* environment 的一部分。當它們學習並改變行為，environment 就變成 non-stationary。Markov property（「下一個 state 只取決於目前的 state 和我的 action」）被破壞了，因為下一個 state 也取決於 *其他* agents 選了什麼，而它們的 policy 是會移動的目標。

這破壞了 tabular 的收斂證明（Q-learning 的保證假設 environment 是 stationary）。它也破壞天真的 deep RL：agents 在迴圈裡互相追逐，永遠不會收斂到穩定的 policy。你需要 multi-agent 專用的技術：centralized training / decentralized execution、counterfactual baselines、league play、self-play。

2026 年的應用：機器人群體、交通路由、自動駕駛車隊、市場模擬器、multi-agent LLM 系統（Phase 16），以及任何有一個以上智慧玩家的遊戲。

## 核心概念 (The Concept)

![Four MARL regimes: indep, centralized critic, self-play, league](../assets/marl.svg)

**形式化：Markov Game。** MDP 的推廣：states $S$、聯合 action $a = (a_1, \ldots, a_n)$、transition $P(s' \mid s, a)$，以及每個 agent 各自的 rewards $R_i(s, a, s')$。每個 agent $i$ 在自己的 policy $\pi_i$ 下最大化自己的 return。如果 rewards 都相同，就是 **fully cooperative**。如果是零和，就是 **adversarial**。如果是混合，就是 **general-sum**。

**核心挑戰：**

- **Non-stationarity。** 從 agent $i$ 的角度看，$P(s' \mid s, a_i)$ 取決於 $\pi_{-i}$，而它正在改變。
- **Credit assignment。** 共享 reward 時，是哪個 agent 造成的？
- **Exploration 協調。** Agents 必須探索互補的策略，而不是重複探索同一個 state。
- **可擴展性。** 聯合 action 空間隨 $n$ 呈指數成長。
- **Partial observability。** 每個 agent 只看得到自己的觀測；全域 state 是隱藏的。

**四種主要的體制：**

**1. Independent Q-learning / independent PPO (IQL, IPPO)。** 每個 agent 學自己的 Q 或 policy，把其他 agents 視為 environment 的一部分。簡單，有時有效（尤其是 experience replay 扮演平滑化的 agent-modeling 技巧時）。理論上的收斂：沒有。實務上：對鬆散耦合的任務還可以，對緊密耦合的任務很差。

**2. Centralized training, decentralized execution (CTDE)。** 最常見的現代範式。每個 agent 有自己的 *policy* $\pi_i$，以局部觀測 $o_i$ 為條件，也就是部署時標準的 decentralized execution。在 *訓練* 期間，centralized critic $Q(s, a_1, \ldots, a_n)$ 以完整的全域 state 與聯合 action 為條件。例子：
- **MADDPG**（Lowe et al. 2017）：每個 agent 搭配 centralized critic 的 DDPG。
- **COMA**（Foerster et al. 2017）：counterfactual baseline，也就是問「如果我當時採取 action $a'$，我的 reward 會是多少？」，藉此分離出我的貢獻。
- **MAPPO** / 搭配共享 critic 的 **IPPO**（Yu et al. 2022）：搭配 centralized value function 的 PPO。2026 年在 cooperative MARL 中佔主導。
- **QMIX**（Rashid et al. 2018）：value decomposition，$Q_{\text{tot}}(s, a) = f(Q_1(s, a_1), \ldots, Q_n(s, a_n))$，搭配單調 mixing。

**3. Self-play。** 同一個 agent 的兩份副本互相對打。對手的 policy *就是* 我過去某個快照的 policy。AlphaGo / AlphaZero / MuZero。OpenAI Five。在零和遊戲中效果最好；訓練訊號是對稱的。

**4. League play。** Self-play 在 general-sum / adversarial environments 的延伸：維護一個由過去與目前 policies 組成的族群，從 league 中取樣對手，對它們訓練。加入 exploiters（專門擊敗目前最強者）與 main exploiters（專門擊敗 exploiters）。AlphaStar（StarCraft II）。當遊戲存在「剪刀石頭布」式的策略循環時就需要它。

**Communication。** 允許 agents 彼此傳送學出來的訊息 $m_i$。在 cooperative 情境中有效。Foerster et al. (2016) 證明可微分的 agent 間通訊可以端到端訓練。今天以 LLM 為基礎的 multi-agent 系統（Phase 16）基本上就是用自然語言溝通。

```figure
f3-marl-orbit
```

## 動手做 (Build It)

這堂課使用有兩個 cooperative agents 的 6×6 GridWorld。它們從相對的角落出發，必須抵達共同的目標。共享 reward：任一 agent 還在移動時每步 `-1`，兩者都抵達時 `+10`。見 `code/main.py`。

### 步驟 1：multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*聯合* action 空間是 `|A|² = 16`。全域 state 是兩個位置。

### 步驟 2：independent Q-learning

每個 agent 執行自己的 Q-table，以聯合 state 為 key。每一步：兩者都選 ε-greedy actions，收集聯合 transition，各自用共享 reward 更新自己的 Q。

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        done = False
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            if done:
                Q1[s][a1] += alpha * (r - Q1[s][a1])
                Q2[s][a2] += alpha * (r - Q2[s][a2])
                break
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

在這個任務上有效，因為 rewards 密集且方向一致。在緊密耦合的任務上會失敗（例如其中一個 agent 必須 *等待* 另一個）。

### 步驟 3：搭配 decomposed-value 更新的 centralized Q

對聯合 actions 使用單一 Q：$Q(s, a_1, a_2)$。用共享 reward 更新。在執行時透過邊緣化來去中心化：$\pi_i(s) = \arg\max_{a_i} \max_{a_{-i}} Q(s, a_1, a_2)$。用指數級的聯合 action 空間，換取 *正確的* 全域視角。

### 步驟 4：簡單的 self-play（adversarial 2-agent）

同一個 agent，兩個角色。讓 agent A 對 agent B 訓練；每 $K$ 個 episodes 之後，把 A 的權重複製到 B。對稱訓練，進步穩定。縮小版的 AlphaZero 做法。

## 常見陷阱 (Pitfalls)

- **Non-stationary 的 replay。** 搭配獨立 agents 的 experience replay 比 single-agent 更糟，因為舊的 transitions 是由現在已經過時的對手產生的。修正：重新標記，或依新舊程度加權。
- **Credit assignment 含糊。** 長 episode 之後才有共享 reward；沒有明確的方法說是哪個 agent 貢獻的。修正：counterfactual baselines（COMA），或針對每個 agent 做 reward shaping。
- **Policy 漂移 / 互相追逐。** 每個 agent 的最佳回應會隨著彼此的更新而改變。修正：centralized critic、較慢的 learning rate，或一次只凍結一個。
- **透過協調的 reward hacking。** Agents 找到設計者沒預期到的協調式漏洞。拍賣 agents 收斂到出價為零。修正：謹慎的 reward 設計、行為限制。
- **Exploration 重複。** 兩個 agents 探索相同的 state-action 配對。修正：每個 agent 各自的 entropy bonus，或以角色為條件。
- **League 循環。** 純 self-play 可能卡在支配循環中。修正：搭配多樣對手的 league play。
- **樣本爆炸。** `n` 個 agents × state 空間 × 聯合 actions。用 function approximation 近似；使用分解的 action 空間（每個 agent 一個 policy 輸出 head）。

## 實際使用 (Use It)

2026 年的 MARL 應用地圖：

| 領域 | 方法 | 備註 |
|------|------|------|
| Cooperative 導航 / 操作 | MAPPO / QMIX | CTDE；共享 critic + decentralized actors。 |
| 雙人遊戲（西洋棋、圍棋、撲克） | 搭配 MCTS 的 Self-play (AlphaZero) | 零和；對稱訓練。 |
| 複雜多人遊戲（Dota、StarCraft） | League play + imitation 預訓練 | OpenAI Five、AlphaStar。 |
| 自動駕駛車隊 | 搭配 attention 的 CTDE MAPPO / PPO | Partial obs；團隊大小可變。 |
| 拍賣市場 | Game-theoretic 均衡 + RL | `n` → ∞ 時使用 Mean-field RL。 |
| LLM multi-agent 系統（Phase 16） | 自然語言通訊 + 角色條件 | RL 迴圈位於 agent 規劃層。 |

2026 年，MARL 成長最大的領域是以 LLM 為基礎的：由語言模型 agents 組成的群體進行協商、辯論、建構軟體。RL 以對 *軌跡層級* 輸出（而非 token 層級）的偏好最佳化形式出現（Phase 16 · 03）。

## 交付 (Ship It)

存成 `outputs/skill-marl-architect.md`：

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## 練習 (Exercises)

1. **簡單。** 在 2-agent cooperative GridWorld 上訓練 independent Q-learning。需要多少個 episodes，平均 return 才會 > 0？畫出聯合 learning curve。
2. **中等。** 加入一個「協調」任務：只有當兩個 agents 在同一回合踏上目標時，才算抵達。Independent Q 還會收斂嗎？什麼地方壞掉了？
3. **困難。** 為 MAPPO 風格的訓練實作 centralized critic，並在協調任務上與 independent PPO 比較收斂速度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Markov game | 「Multi-agent MDP」 | $(S, A_1, \ldots, A_n, P, R_1, \ldots, R_n)$；每個 agent 有自己的 reward。 |
| CTDE | 「Centralized training, decentralized execution」 | 訓練時使用聯合 critic；每個 agent 的 policy 只使用局部 obs。 |
| IPPO | 「Independent PPO」 | 每個 agent 各自執行 PPO。簡單的 baseline；常被低估。 |
| MAPPO | 「Multi-agent PPO」 | 搭配以全域 state 為條件的 centralized value function 的 PPO。 |
| QMIX | 「單調的 value decomposition」 | $Q_{\text{tot}} = f_{\text{monotone}}(Q_1, \ldots, Q_n)$ 允許 decentralized argmax。 |
| COMA | 「Counterfactual multi-agent」 | Advantage = 我的 Q 減去對我的 action 邊緣化後的期望 Q。 |
| Self-play | 「Agent 對上過去的自己」 | 單一 agent、兩個角色；零和遊戲的標準做法。 |
| League play | 「族群訓練」 | 快取過去的 policies，從池中取樣對手；能處理策略循環。 |

## 延伸閱讀 (Further Reading)

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — 搭配 centralized critic 的 CTDE。
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — 用於 credit assignment 的 counterfactual baselines。
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — 具單調性的 value decomposition。
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO 在 MARL 上出乎意料地強。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — 大規模的 league play。
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — 零和遊戲中的純 self-play。
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — 包含教科書對 multi-agent 情境的簡短處理，以及 CTDE 旨在解決的 non-stationarity 問題。
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — 涵蓋 cooperative、competitive 與混合 MARL 及其收斂結果的綜述。
