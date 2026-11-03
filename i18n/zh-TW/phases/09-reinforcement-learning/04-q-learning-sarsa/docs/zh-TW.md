# Temporal Difference：Q-Learning 與 SARSA (Temporal Difference — Q-Learning & SARSA)

> Monte Carlo 要等到 episode 結束。TD 則在每一步之後，藉由 bootstrapping 下一個 value 的估計值來更新。Q-learning 是 off-policy 而且樂觀；SARSA 是 on-policy 而且謹慎。兩者都只要一行程式。兩者也是本階段每個 deep-RL 方法的基礎。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming), Phase 9 · 03 (Monte Carlo)
**Time:** ~75 分鐘

## 問題 (The Problem)

Monte Carlo 能用，但有兩個代價高昂的要求。它需要會結束的 episodes，而且要等最終 return 出來才更新。如果你的 episode 有 1,000 步，MC 要等 1,000 步才能更新任何東西。它變異數高、bias 低，實務上很慢。

Dynamic programming 的特性正好相反：bootstrapped backup 的變異數為零，但需要已知的 model。

Temporal difference (TD) learning 取兩者之間。從單一 transition $(s, a, r, s')$ 形成一步的目標 $r + \gamma V(s')$，把 $V(s)$ 往它推一點。不需要 model，也不需要完整的 episodes。因為右側用了近似的 $V$，所以有 bias，但變異數比 MC 低很多，而且從第一步就能 online 更新。

這是整個現代 RL 的樞紐，包括 DQN、A2C、PPO、SAC。Phase 9 的其餘內容，都是建立在你這堂課要寫的一步 TD 更新之上，一層層疊上的 function approximation 與技巧。

## 核心概念 (The Concept)

![Q-learning vs SARSA: off-policy max vs on-policy Q(s', a')](../assets/td.svg)

**V 的 TD(0) 更新：**

```math
V(s) \leftarrow V(s) + \alpha [r + \gamma V(s') - V(s)]
```

括號裡的量是 TD error $\delta = r + \gamma V(s') - V(s)$。它是 MC 裡 $G_t - V(s_t)$ 的 online 版本。收斂需要 $\alpha$ 滿足 Robbins-Monro（$\sum \alpha = \infty$，$\sum \alpha^2 < \infty$），而且所有 states 都被造訪無限多次。

**Q-learning。** 用於控制的 off-policy TD 方法：

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \max_{a'} Q(s', a') - Q(s, a)]
```

`max` 假設從 $s'$ 之後會遵循 *greedy* policy，不管 agent 實際上採取什麼 action。這種解耦讓 Q-learning 在 agent 以 ε-greedy 做 exploration 的同時，仍然學到 $Q^{\ast}$。Mnih et al. (2015) 把它變成 Atari 上的 deep Q-learning（Lesson 05）。

**SARSA。** On-policy 的 TD 方法：

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma Q(s', a') - Q(s, a)]
```

名字來自元組 $(s, a, r, s', a')$。SARSA 使用 agent *實際* 接下來採取的 action $a'$，而不是 greedy 的 `argmax`。它會收斂到目前執行中的 ε-greedy $\pi$ 的 $Q^\pi$，在 $\epsilon \to 0$ 的極限下就變成 $Q^{\ast}$。

**Cliff-walking 的差異。** 在經典的 cliff-walking 任務中（掉下懸崖 = reward -100），Q-learning 學到沿著懸崖邊緣的最佳路徑，但在 exploration 期間偶爾會吃到懲罰。SARSA 則學到離懸崖一步遠的較安全路徑，因為它把 exploration 的雜訊算進了 Q-value。經過訓練，兩者在 $\epsilon \to 0$ 時都會達到最佳。實務上這個差異很重要：當部署時 exploration 真的在發生，SARSA 的行為比較保守。

**Expected SARSA。** 把 $Q(s', a')$ 換成它在 $\pi$ 下的期望值：

```math
Q(s, a) \leftarrow Q(s, a) + \alpha [r + \gamma \sum_{a'} \pi(a' \mid s') Q(s', a') - Q(s, a)]
```

變異數比 SARSA 低（不用取樣 $a'$），on-policy 的目標則相同。常是現代教科書裡的預設做法。

**n-step TD 與 TD(λ)。** 等待 $n$ 步之後才 bootstrap，在 TD(0) 與 MC 之間做內插。$n=1$ 是 TD，$n=\infty$ 是 MC。TD(λ) 用幾何權重 $(1-\lambda)\lambda^{n-1}$ 對所有 $n$ 取平均。大多數 deep-RL 使用的 $n$ 介於 3 到 20。

```figure
qlearning-gridworld
```

## 動手做 (Build It)

### 步驟 1：ε-greedy policy 上的 SARSA

```python
def sarsa(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})

    def choose(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        s = env.reset()
        a = choose(s)
        while True:
            s_next, r, done = env.step(s, a)
            a_next = choose(s_next) if not done else None
            target = r + (gamma * Q[s_next][a_next] if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s, a = s_next, a_next
    return Q
```

八行。與 Q-learning 的 *唯一* 差別就在目標那一行。

### 步驟 2：Q-learning

```python
def q_learning(env, episodes, alpha=0.1, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    for _ in range(episodes):
        s = env.reset()
        while True:
            a = choose(s, Q, epsilon)
            s_next, r, done = env.step(s, a)
            target = r + (gamma * max(Q[s_next].values()) if not done else 0.0)
            Q[s][a] += alpha * (target - Q[s][a])
            if done:
                break
            s = s_next
    return Q
```

`max` 讓目標與 behavior 解耦。這一個符號，就是 on-policy 與 off-policy 的差別。

### 步驟 3：learning curves

每 100 個 episodes 追蹤一次平均 return。在簡單的 deterministic GridWorld 上，Q-learning 收斂得比較快；在 cliff-walking 上，SARSA 比較保守。在 `code/main.py` 的 4×4 GridWorld 上，使用 $\alpha=0.1, \epsilon=0.1$，兩者大約在 ~2,000 個 episodes 後都接近最佳。

### 步驟 4：與 DP 的真值比較

執行 value iteration（Lesson 02）取得 $Q^{\ast}$。檢查 $\max_{s,a} \lvert Q_{\text{learned}}(s,a) - Q^{\ast}(s,a) \rvert$。一個健康的 tabular TD agent，在 4×4 GridWorld 上跑 10,000 個 episodes 後，誤差會落在 `~0.5` 以內。

## 常見陷阱 (Pitfalls)

- **初始 Q 值很重要。** 樂觀初始化（負 reward 任務中設 $Q = 0$）會鼓勵 exploration。悲觀初始化則可能讓 greedy policy 永遠卡住。
- **α 的排程。** 固定的 $\alpha$ 適合 non-stationary 的問題。遞減的 $\alpha_n = 1/n$ 理論上保證收斂，但實務上太慢，把 $\alpha$ 固定在 `[0.05, 0.3]` 並監看 learning curve。
- **ε 的排程。** 從高處開始（$\epsilon=1.0$），遞減到 $\epsilon=0.05$。「GLIE」（greedy in the limit with infinite exploration）是收斂條件。
- **Q-learning 的 max bias。** 當 $Q$ 有雜訊時，`max` 運算子會往上偏。導致高估：Hasselt 的 Double Q-learning（Lesson 05 的 DDQN 使用）用兩張 Q 表來修正。
- **不會結束的 episodes。** TD 可以在沒有終點的情況下學習，但你需要設定步數上限，或在上限處正確處理 bootstrap。標準做法：把上限視為非終止，繼續 bootstrap。
- **State hashing。** 如果 states 是 tuple/tensor，請使用可 hash 的 key（用 tuple，不要用 list；用四捨五入後的 float tuple，不要用原始值）。

## 實際使用 (Use It)

2026 年的 TD 版圖：

| 任務 | 方法 | 原因 |
|------|------|------|
| 小型 tabular environments | Q-learning | 直接學到最佳 policy。 |
| On-policy、安全至上 | SARSA / Expected SARSA | Exploration 期間保守。 |
| 高維度 state | DQN (Phase 9 · 05) | 搭配 replay 與 target net 的神經網路 Q-function。 |
| 連續 actions | SAC / TD3 (Phase 9 · 07) | 對 Q-network 做 TD 更新；policy net 輸出 actions。 |
| LLM RL（以 reward-model 為基礎） | PPO / GRPO (Phase 9 · 08, 12) | 透過 GAE 取得 TD 風格 advantage 的 actor-critic。 |
| Offline RL | CQL / IQL (Phase 9 · 08) | 帶有保守式 regularization 的 Q-learning。 |

你在 2026 年論文裡讀到的「RL」，有九成是 Q-learning 或 SARSA 的某種延伸。在深入閱讀之前，先把 tabular 更新練到身體記得。

## 交付 (Ship It)

存成 `outputs/skill-td-agent.md`：

```markdown
---
name: td-agent
description: Pick between Q-learning, SARSA, Expected SARSA for a tabular or small-feature RL task.
version: 1.0.0
phase: 9
lesson: 4
tags: [rl, td-learning, q-learning, sarsa]
---

Given a tabular or small-feature environment, output:

1. Algorithm. Q-learning / SARSA / Expected SARSA / n-step variant. One-sentence reason tied to on-policy vs off-policy and variance.
2. Hyperparameters. α, γ, ε, decay schedule.
3. Initialization. Q_0 value (optimistic vs zero) and justification.
4. Convergence diagnostic. Target learning curve, `|Q - Q*|` check if DP is possible.
5. Deployment caveat. How will exploration behave at inference? Is SARSA's conservatism needed?

Refuse to apply tabular TD to state spaces > 10⁶. Refuse to ship a Q-learning agent without a max-bias caveat. Flag any agent trained with ε held at 1.0 throughout (no exploitation phase).
```

## 練習 (Exercises)

1. **簡單。** 在 4×4 GridWorld 上實作 Q-learning 與 SARSA。畫出 2,000 個 episodes 的 learning curves（每 100 個 episodes 的平均 return）。誰收斂得比較快？
2. **中等。** 建立一個 cliff-walking environment（4×12，最後一列是懸崖，reward 為 -100 並重置回起點）。比較 Q-learning 與 SARSA 的最終 policy。擷取各自走的路徑。哪一個更靠近懸崖？
3. **困難。** 實作 Double Q-learning。在有雜訊 reward 的 GridWorld 上（對每步 reward 加入 Gaussian 雜訊 $\sigma=5$），證明 Q-learning 會明顯高估 $V^{\ast}(0,0)$，而 Double Q-learning 不會。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| TD error | 「更新訊號」 | $\delta = r + \gamma V(s') - V(s)$，bootstrapped 的殘差。 |
| TD(0) | 「一步 TD」 | 每個 transition 之後更新，只用到下一個 state 的估計值。 |
| Q-learning | 「Off-policy RL 入門」 | 對下一個 state 的 actions 取 `max` 的 TD 更新；不論 behavior policy 為何，都會學到 $Q^{\ast}$。 |
| SARSA | 「On-policy 的 Q-learning」 | 使用實際的下一個 action 的 TD 更新；學到目前 ε-greedy π 的 $Q^\pi$。 |
| Expected SARSA | 「低變異數的 SARSA」 | 把取樣的 $a'$ 換成它在 π 下的期望值。 |
| GLIE | 「正確的 exploration 排程」 | Greedy in the Limit with Infinite Exploration；Q-learning 收斂所需。 |
| Bootstrapping | 「在目標中使用目前的估計值」 | TD 與 MC 的區別所在。bias 的來源，但能大幅降低變異數。 |
| Maximization bias | 「Q-learning 會高估」 | 對有雜訊的估計值取 `max` 會往上偏；用 Double Q-learning 修正。 |

## 延伸閱讀 (Further Reading)

- [Watkins & Dayan (1992). Q-learning](https://link.springer.com/article/10.1007/BF00992698) — 原始論文與收斂證明。
- [Sutton & Barto (2018). Ch. 6 — Temporal-Difference Learning](http://incompleteideas.net/book/RLbook2020.pdf) — TD(0)、SARSA、Q-learning、Expected SARSA。
- [Hasselt (2010). Double Q-learning](https://papers.nips.cc/paper_files/paper/2010/hash/091d584fced301b442654dd8c23b3fc9-Abstract.html) — 修正 maximization bias。
- [Seijen, Hasselt, Whiteson, Wiering (2009). A Theoretical and Empirical Analysis of Expected SARSA](https://ieeexplore.ieee.org/document/4927542) — expected SARSA 的動機。
- [Rummery & Niranjan (1994). On-line Q-learning using connectionist systems](https://www.researchgate.net/publication/2500611_On-Line_Q-Learning_Using_Connectionist_Systems) — 創造出 SARSA 這個名稱的論文（當時稱為「modified connectionist Q-learning」）。
- [Sutton & Barto (2018). Ch. 7 — n-step Bootstrapping](http://incompleteideas.net/book/RLbook2020.pdf) — 把 TD(0) 推廣為 TD(n)，是從 Q-learning 通往 eligibility traces，以及後來 PPO 中 GAE 的路徑。
