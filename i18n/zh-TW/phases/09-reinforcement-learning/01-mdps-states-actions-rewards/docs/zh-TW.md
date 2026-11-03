# MDPs、States、Actions 與 Rewards (MDPs, States, Actions & Rewards)

> Markov Decision Process 由五個東西組成：states、actions、transitions、rewards、discount。RL 裡的一切，包括 Q-learning、PPO、DPO、GRPO，都是在這個結構上做最佳化。學會一次，其餘的 reinforcement learning 就能順著讀懂。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 分鐘

## 問題 (The Problem)

你正在寫一個西洋棋 bot。或是庫存規劃器。或是交易 agent。或是訓練推理模型用的 PPO 迴圈。四個不同領域，卻有一個令人意外的事實：這四個最後都化簡成同一個數學物件。

Supervised learning 給你 `(x, y)` 配對，要你擬合一個函式。Reinforcement learning 沒有任何標籤，只有一連串的 states、你採取的 actions，以及一個純量 reward。這步棋有贏下比賽嗎？補貨決策有省到錢嗎？這筆交易有獲利嗎？LLM 剛產生的那個 token，有讓評審給出更高的 reward 嗎？

在把這串資料正式化之前，你無法從中學習。「我看到什麼」、「我做了什麼」、「接下來發生什麼」、「結果有多好」，每一項都得變成可以推理的物件。這種正式化就是 Markov Decision Process。本階段的每個 RL 演算法，包括最後的 RLHF 與 GRPO 迴圈，都是在這個結構上做最佳化。

## 核心概念 (The Concept)

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**五個物件。**

- **States** $S$。agent 做決定所需的一切資訊。在 GridWorld 是格子；在西洋棋是棋盤；在 LLM 是 context window 加上任何記憶。
- **Actions** $A$。可以做的選擇。上下左右移動、下一步棋、產生一個 token。
- **Transitions** $P(s' \mid s, a)$。給定 state $s$ 與 action $a$，下一個 state 的分布。西洋棋是 deterministic，庫存是 stochastic，LLM decoding 則幾乎是 deterministic。
- **Rewards** $R(s, a, s')$。純量訊號。贏 = +1，輸 = -1。營收減成本。GRPO 裡的 log-likelihood ratio 項。
- **Discount** $\gamma \in [0, 1)$。未來的 reward 相對於當下的權重有多大。$\gamma = 0.99$ 約對應 100 步的視野；$\gamma = 0.9$ 約對應 10 步。

**Markov property** $P(s_{t+1} \mid s_t, a_t) = P(s_{t+1} \mid s_0, a_0, \ldots, s_t, a_t)$。未來只取決於目前的 state。如果不是這樣，代表 state 的表示不完整。這不是方法的失敗，而是 state 的失敗。

**Policies 與 returns。** Policy $\pi(a \mid s)$ 把 states 對應到 actions 的分布。Return $G_t = r_t + \gamma r_{t+1} + \gamma^2 r_{t+2} + \ldots$ 是未來 rewards 的折扣總和。Value $V^\pi(s) = \mathbb{E}[G_t \mid s_t = s]$ 是在 policy $\pi$ 下，從 $s$ 出發的期望 return。Q-value $Q^\pi(s, a) = \mathbb{E}[G_t \mid s_t = s, a_t = a]$ 是從 $s$ 出發、且第一步指定 action 的期望 return。每個 RL 演算法都是估計這兩者之一，再據此改進 $\pi$。

**Bellman equations。** 本階段所有內容都會用到的 fixed-point 方程式：

```math
\begin{aligned}
V^\pi(s) &= \sum_a \pi(a \mid s) \sum_{s', r} P(s', r \mid s, a) [r + \gamma V^\pi(s')] \\
Q^\pi(s, a) &= \sum_{s', r} P(s', r \mid s, a) [r + \gamma \sum_{a'} \pi(a' \mid s') Q^\pi(s', a')]
\end{aligned}
```

這兩條式子把期望 return 拆成「這一步的 reward」加上「落腳處的折扣 value」。它是遞迴的。Phase 9 的每個演算法，不是把這個方程式迭代到收斂（dynamic programming），就是從中取樣（Monte Carlo），不然就是往前 bootstrap 一步（temporal difference）。

```figure
discount-horizon
```

## 動手做 (Build It)

### 步驟 1：一個小型 deterministic MDP

一個 4×4 的 GridWorld。agent 從左上角出發，終點在右下角，每走一步 reward 為 -1，actions 為 `{up, down, left, right}`。見 `code/main.py`。

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

五行程式，這就是整個 environment。Deterministic transitions、固定的步數懲罰、吸收型的終止 state。

### 步驟 2：執行一個 policy

Policy 是從 state 到 action 分布的函式。最簡單的是均勻隨機。

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

把 random policy 跑 1000 次。在這個 4×4 棋盤上，平均 return 約為 -60 到 -80。最佳 return 是 -6（沿直線往右下走）。縮小這個差距，就是 Phase 9 的全部內容。

### 步驟 3：用 Bellman equation 精確計算 `V^π`

對小型 MDP 來說，Bellman equation 是一個線性系統。列舉所有 states、套用期望值，迭代到數值不再變動為止。

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

這就是 iterative policy evaluation。它是 Sutton & Barto 裡的第一個演算法，也是後續所有 RL 方法的理論基礎。

### 步驟 4：`γ` 是有實際意義的 hyperparameter

有效視野大約是 $1 / (1 - \gamma)$。$\gamma = 0.9$ → 10 步。$\gamma = 0.99$ → 100 步。$\gamma = 0.999$ → 1000 步。

太低，agent 會短視近利。太高，credit assignment 會變得雜訊很大，因為許多早期步驟都要分擔遠期 reward 的責任。LLM 的 RLHF 通常用 $\gamma = 1$，因為 episode 短而且有界。控制任務用 `0.95–0.99`。長視野的策略遊戲用 `0.999`。

## 常見陷阱 (Pitfalls)

- **非 Markov 的 state。** 如果你需要最近三次觀測才能做決定，那「state」就不只是目前的觀測。解法：堆疊 frames（Atari 上的 DQN 堆 4 個），或使用 recurrent state（對觀測序列用 LSTM/GRU）。
- **Sparse rewards。** 只在贏的時候給 reward，在大型 state space 裡幾乎無法學習。可以做 reward shaping（加入中間訊號），或用 imitation 來 bootstrap（Phase 9 · 09）。
- **Reward hacking。** 最佳化代理 reward 常會產生病態行為。OpenAI 的賽船 agent 為了反覆收集加速道具而原地繞圈，根本不去完賽。永遠要從目標結果定義 reward，而不是從代理指標。
- **Discount 設定錯誤。** 在無限視野的任務上用 $\gamma = 1$，會讓每個 value 都變成無限大。一定要用有限視野或 $\gamma < 1$ 來設上限。
- **Reward 尺度。** Rewards 為 {+100, -100} 與 {+1, -1} 會得到相同的最佳 policy，但 gradient 大小差很多。丟進 PPO/DQN 之前，先正規化到 `[-1, 1]` 左右。

## 實際使用 (Use It)

2026 年的做法，是在動手寫程式前，先把每條 RL pipeline 化簡成 MDP：

| 情境 | State | Action | Reward | γ |
|------|-------|--------|--------|---|
| 控制（行走、操作） | 關節角度 + 速度 | 連續的力矩 | 依任務設計的 shaped reward | 0.99 |
| 遊戲（西洋棋、圍棋、撲克） | 棋盤 + 歷史 | 合法的一步 | 贏=+1 / 輸=-1 | 1.0（有限） |
| 庫存 / 定價 | 庫存 + 需求 | 訂購數量 | 營收 - 成本 | 0.95 |
| LLM 的 RLHF | Context tokens | 下一個 token | 結尾的 reward-model 分數 | 1.0（episode 約 200 tokens） |
| 推理用的 GRPO | Prompt + 部分回應 | 下一個 token | 結尾的 verifier 0/1 | 1.0 |

在寫任何訓練迴圈之前，先寫下這五元組。大多數「RL 不 work」的 bug 回報，追根究柢都是 MDP 在紙上就沒定義對。

## 交付 (Ship It)

存成 `outputs/skill-mdp-modeler.md`：

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## 練習 (Exercises)

1. **簡單。** 在 `code/main.py` 實作 4×4 GridWorld 與 random-policy rollout。跑 10,000 個 episodes。回報 return 的平均與標準差，並與最佳 return（-6）比較。
2. **中等。** 對 uniform-random policy，用 $\gamma \in \lbrace 0.5, 0.9, 0.99 \rbrace$ 執行 `policy_evaluation`。每個 $\gamma$ 都把 `V` 印成 4×4 格子。解釋為什麼 $\gamma$ 越大，靠近終點的 state values 成長得越快。
3. **困難。** 把 GridWorld 改成 stochastic：每個 action 以機率 $p = 0.1$ 滑向相鄰的方向。重新評估 uniform policy。`V[start]` 變好還是變差？為什麼？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| MDP | 「Reinforcement learning 的設定」 | 滿足 Markov property 的元組 $(S, A, P, R, \gamma)$。 |
| State | 「agent 看到的東西」 | 在所選 policy 類別下，對未來動態而言的充分統計量。 |
| Policy | 「agent 的行為」 | 條件分布 $\pi(a \mid s)$，或 deterministic 映射 $s \to a$。 |
| Return | 「總 reward」 | 從目前這一步開始的折扣總和 $\sum \gamma^t r_t$。 |
| Value | 「state 有多好」 | 在 $\pi$ 下，從 $s$ 出發的期望 return。 |
| Q-value | 「action 有多好」 | 在 $\pi$ 下，從 $s$ 出發且第一個 action 為 $a$ 的期望 return。 |
| Bellman equation | 「Dynamic programming 的遞迴」 | 把 value / Q 分解為一步 reward 加上折扣後的後繼 value 的 fixed-point 式子。 |
| Discount $\gamma$ | 「未來與現在的取捨」 | 對遠期 reward 的幾何權重；有效視野 $\approx 1/(1-\gamma)$。 |

## 延伸閱讀 (Further Reading)

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書。第 3 章講 MDPs 與 Bellman equations；第 1 章說明 reward hypothesis 的動機，後面每一堂課都建立在它之上。
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — Bellman equation 的起源。
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — 從 deep RL 角度出發的簡潔 MDP 入門。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — 作業研究領域中，關於 MDPs 與精確求解方法的參考書。
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://cs.brown.edu/media/filer_public/d1/a6/d1a6f66a-289a-4b81-9596-417114843489/littman.pdf) — 把 MDPs 當成 dynamic programming 特例，推導得最清楚的一份。
