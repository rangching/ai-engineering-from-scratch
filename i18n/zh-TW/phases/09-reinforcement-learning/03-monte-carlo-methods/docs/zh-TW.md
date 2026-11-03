# Monte Carlo Methods：從完整 Episodes 學習 (Monte Carlo Methods — Learning from Complete Episodes)

> Dynamic programming 需要 model。Monte Carlo 只需要 episodes。執行 policy，觀察 returns，取平均。這是 RL 裡最簡單的想法，也是解鎖後面一切的關鍵。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 分鐘

## 問題 (The Problem)

Dynamic programming 很優雅，但它假設你能對每個 state 與 action 查詢 $P(s' \mid s, a)$。現實世界幾乎沒有什麼是這樣運作的。機器人無法用解析方式算出關節施加力矩後，相機像素的分布。定價演算法無法對每一種可能的顧客反應做積分。LLM 無法列舉出一個 token 之後所有可能的接續。

你需要一種只要能從 environment *取樣* 就能運作的方法。執行 policy，得到一條軌跡 `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`，用它來估計 values。這就是 Monte Carlo。

從 DP 轉到 MC，在觀念上很重要：我們從 *已知 model + 精確 backup*，轉成 *取樣 rollouts + 平均 return*。變異數會暴增，但適用範圍大幅擴張。這堂課之後的每個 RL 演算法，包括 TD、Q-learning、REINFORCE、PPO、GRPO，本質上都是 Monte Carlo 估計器，有時再疊上 bootstrapping。

## 核心概念 (The Concept)

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**一句話說完核心想法：** $V^\pi(s) = \mathbb E_\pi[G_t \mid s_t = s] \approx (1/N) \sum_i G^{(i)}(s)$，其中 $G^{(i)}(s)$ 是在 policy $\pi$ 下，每次造訪 $s$ 之後觀察到的 returns。

**First-visit 與 every-visit MC。** 假設一個 episode 多次造訪 state $s$：first-visit MC 只計入第一次造訪的 return；every-visit MC 則計入所有造訪。First-visit MC 是 unbiased。Every-visit MC 在有限個 episodes 之後可能有些微 bias，因為它每個 episode 平均的 returns 數量是隨機的，而這個數量可能和 returns 本身相關，但兩者在造訪次數趨近無限時，都會收斂到 $V^\pi(s)$。First-visit 比較容易分析（樣本是 iid）。Every-visit 每個 episode 用到更多資料，實務上通常收斂得更快。

**Incremental mean。** 不必儲存所有 returns，直接更新滾動平均：

```math
V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]
```

整理一下：$V_{\text{new}} = V_{\text{old}} + \alpha \cdot (\text{target} - V_{\text{old}})$，其中 $\alpha = 1/n$。把 $1/n$ 換成固定的 step-size $\alpha \in (0, 1)$，你就得到一個能追蹤 $\pi$ 變化的 non-stationary MC 估計器。這一步，就是從 MC 跳到 TD、再到每個現代 RL 演算法的整段跳躍。

**Exploration 現在成了問題。** DP 靠列舉走遍每個 state。MC 只看得到 policy 造訪過的 states。如果 $\pi$ 是 deterministic，state space 的整片區域永遠不會被取樣到，它們的 value 估計會永遠停在零。依歷史順序有三種解法：

1. **Exploring starts。** 每個 episode 都從隨機的 (s, a) 配對開始。能保證覆蓋範圍；但實務上不切實際（你無法把機器人「重置」到任意 state）。
2. **ε-greedy。** 對目前的 Q 取 greedy，但以機率 $\epsilon$ 隨機挑一個 action。所有 state-action 配對在漸近意義下都會被取樣到。
3. **Off-policy MC。** 在 behavior policy $\mu$ 下收集資料，再透過 importance sampling 學習 target policy $\pi$。變異數很高，但這是通往 DQN 這類 replay-buffer 方法的橋樑。

**Monte Carlo Control。** Evaluate → improve → evaluate，就像 policy iteration，只是 evaluation 是以取樣為基礎：

1. 執行 $\pi$，得到一個 episode。
2. 用觀察到的 returns 更新 $Q(s, a)$。
3. 讓 $\pi$ 對 $Q$ 取 ε-greedy。
4. 重複。

在溫和的條件下（每個配對都被造訪無限多次、$\alpha$ 滿足 Robbins-Monro），以機率 1 收斂到 $Q^{\ast}$ 與 $\pi^{\ast}$。

```figure
epsilon-greedy
```

## 動手做 (Build It)

### 步驟 1：rollout → (s, a, r) 的 list

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

沒有 model，只有 `env.reset()` 與 `env.step(s, a)`。介面和 gym environment 相同，但被精簡過。

### 步驟 2：計算 returns（反向 sweep）

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

只跑一趟，$O(T)$。反向遞迴 $G_t = r_{t+1} + \gamma G_{t+1}$ 避免重新加總。

### 步驟 3：first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

三行就完成主要工作：第一次造訪時標記 state、累加計數、更新滾動平均。

### 步驟 4：ε-greedy MC control（on-policy）

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### 步驟 5：與 DP 黃金標準比較

隨著 episodes → ∞，你的 MC 對 $V^\pi$ 的估計應該會和 Lesson 02 的 DP 結果一致。實務上：在 4×4 GridWorld 跑 50,000 個 episodes，可以逼近 DP 答案的 `~0.1` 以內。

## 常見陷阱 (Pitfalls)

- **無限長的 episodes。** MC 要求 episodes 必須 *結束*。如果你的 policy 可能永遠循環，就設定 `max_steps` 上限，並把達到上限視為隱含的失敗。GridWorld 搭配 random policy 經常會超時，這是正常的，只要確定你有正確計算它。
- **變異數。** MC 使用完整的 returns。在長 episodes 上變異數很大：結尾一個倒楣的 reward，會讓 $V(s_0)$ 位移同樣的量。TD 方法（Lesson 04）透過 bootstrapping 降低這個問題。
- **State 覆蓋。** 對全新的 Q 做 greedy MC，遇到平手時永遠只會嘗試同一個 action。你 *必須* 做 exploration（ε-greedy、exploring starts、UCB）。
- **Non-stationary policies。** 如果 $\pi$ 會改變（像 MC control 那樣），舊的 returns 就來自不同的 policy。固定 α 的 MC 能處理這點；sample-average MC 不行。
- **Off-policy importance sampling。** 權重 $\pi(a \mid s)/\mu(a \mid s)$ 會沿整條軌跡連乘。變異數會隨視野爆炸。可以用 per-decision weighted IS 來限制，或改用 TD。

## 實際使用 (Use It)

Monte Carlo 方法在 2026 年扮演的角色：

| 使用情境 | 為什麼用 MC |
|----------|-------------|
| 短視野遊戲（blackjack、撲克） | Episodes 自然會結束；returns 很乾淨。 |
| 離線評估已記錄的 policy | 對儲存的軌跡取折扣 returns 的平均。 |
| Monte Carlo Tree Search（AlphaZero） | 從樹葉出發的 MC rollouts 引導選擇。 |
| LLM 的 RL 評估 | 對某個 policy 的取樣 completions 計算平均 reward。 |
| PPO 的 baseline 估計 | Advantage 目標 $A_t = G_t - V(s_t)$ 用的是 MC 的 $G_t$。 |
| 教學 RL | 真正能運作的最簡單演算法：拿掉 bootstrapping，就能看見核心。 |

現代的 deep-RL 演算法（PPO、SAC）透過 $n$-step returns 或 GAE，在純 MC（完整 returns）與純 TD（一步 bootstrap）之間做內插。兩個端點都是同一個估計器的實例。

## 交付 (Ship It)

存成 `outputs/skill-mc-evaluator.md`：

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## 練習 (Exercises)

1. **簡單。** 實作 4×4 GridWorld 上 uniform-random policy 的 first-visit MC evaluation。跑 10,000 個 episodes。畫出 $V(0,0)$ 隨 episode 數量變化的曲線，並與 DP 的答案對照。
2. **中等。** 用 $\epsilon \in \lbrace 0.01, 0.1, 0.3 \rbrace$ 實作 ε-greedy MC control。比較 20,000 個 episodes 之後的平均 return。曲線長什麼樣子？bias-variance 的取捨在哪裡？
3. **困難。** 用 importance sampling 實作 *off-policy* MC：在 uniform-random policy $\mu$ 下收集資料，估計 deterministic 最佳 policy $\pi$ 的 $V^\pi$。比較一般 IS、per-decision IS 與 weighted IS。哪一個變異數最低？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Monte Carlo | 「隨機取樣」 | 對分布的 iid 樣本取平均來估計期望值。 |
| Return $G_t$ | 「未來的 reward」 | 從第 $t$ 步到 episode 結束的折扣 rewards 總和：$\sum_{k\ge 0} \gamma^k r_{t+k+1}$。 |
| First-visit MC | 「每個 state 只算一次」 | 一個 episode 中只有第一次造訪會貢獻到 value 估計。 |
| Every-visit MC | 「用上所有造訪」 | 每次造訪都有貢獻；有限樣本下略有 bias，極限下具一致性，樣本效率也更高。 |
| ε-greedy | 「Exploration 的雜訊」 | 以機率 $1-\epsilon$ 選 greedy action；以機率 $\epsilon$ 選隨機 action。 |
| Importance sampling | 「修正取樣來自錯誤分布的問題」 | 用 $\pi(a \mid s)/\mu(a \mid s)$ 的乘積重新加權 returns，以便從 $\mu$ 的資料估計 $V^\pi$。 |
| On-policy | 「從我自己的資料學習」 | Target policy = behavior policy。例如 vanilla MC、PPO、SARSA。 |
| Off-policy | 「從別人的資料學習」 | Target policy ≠ behavior policy。例如 importance-sampled MC、Q-learning、DQN。 |

## 延伸閱讀 (Further Reading)

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 經典講解。
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — first-visit 與 every-visit 的分析。
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC 與變異數控制。
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — 現代的低變異數 IS 估計器。
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — 第一個大規模實證，展示 MC/TD self-play 能收斂到超越人類的棋力；也是本階段後半每一堂課在概念上的先驅。
