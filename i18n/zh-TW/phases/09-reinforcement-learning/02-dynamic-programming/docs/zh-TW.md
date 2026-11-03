# Dynamic Programming：Policy Iteration 與 Value Iteration (Dynamic Programming — Policy Iteration & Value Iteration)

> Dynamic programming 是作弊版的 RL。你早就知道 transition 與 reward 函式，只要把 Bellman equation 迭代到 $V$ 或 $\pi$ 不再變動就好。它是所有以取樣為基礎的方法都想逼近的基準。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs)
**Time:** ~75 分鐘

## 問題 (The Problem)

你有一個已知模型的 MDP：對任何 state-action 配對，你都能查詢 $P(s' \mid s, a)$ 與 $R(s, a, s')$。庫存經理知道需求分布。棋盤遊戲的 transitions 是 deterministic。GridWorld 只要四行 Python。你有一個 *model*。

Model-free RL（Q-learning、PPO、REINFORCE）是為了沒有 model 的情況而發明的，你只能從 environment 取樣。但當你有 model 時，有更快、更好的方法：dynamic programming。Bellman 在 1957 年就設計出來了。它們至今仍定義了什麼叫正確：當人們說「這個 MDP 的最佳 policy」，指的就是 DP 會回傳的那個 policy。

你在 2026 年還需要它們，有三個理由。第一，RL 研究中的每個 tabular environment（GridWorld、FrozenLake、CliffWalking），都是用 DP 求出黃金標準的 policy。第二，精確的 values 讓你能 *debug* 取樣方法：如果 Q-learning 對 $V^{\ast}(s_0)$ 的估計和 DP 的答案差了 30%，你的 Q-learning 就有 bug。第三，現代的 offline RL 與 planning 方法（MCTS、AlphaZero 的搜尋、Phase 9 · 10 的 model-based RL），都是在學到或給定的 model 上迭代 Bellman backup。

## 核心概念 (The Concept)

![Policy iteration and value iteration, side by side](../assets/dp.svg)

**兩個演算法，都是對 Bellman 做 fixed-point 迭代。**

**Policy iteration。** 交替執行兩個步驟，直到 policy 不再改變。

1. *Evaluation：* 給定 policy $\pi$，反覆套用 $V(s) \leftarrow \sum_a \pi(a \mid s) \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]$ 直到收斂，算出 $V^\pi$。
2. *Improvement：* 給定 $V^\pi$，讓 $\pi$ 對 $V^\pi$ 取 greedy：$\pi(s) \leftarrow \arg\max_a \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]$。

收斂是有保證的，因為 (a) 每個 improvement 步驟要嘛讓 $\pi$ 保持不變，要嘛讓某些 state 的 $V^\pi$ 嚴格增加，(b) deterministic policies 的空間是有限的。即使 state space 很大，通常也只需要約 5–20 次外層迭代就收斂。

**Value iteration。** 把 evaluation 與 improvement 合併成一次 sweep。套用 Bellman *optimality* equation：

```math
V(s) \leftarrow \max_a \sum_{s',r} P(s',r \mid s,a) [r + \gamma V(s')]
```

重複直到 $\max_s \lvert V_{\text{new}}(s) - V(s) \rvert < \epsilon$。最後取 greedy action 來抽出 policy。每次迭代嚴格來說更快（沒有內層 evaluation 迴圈），但通常需要更多次迭代才會收斂。

**Generalized policy iteration (GPI)。** 統一的觀點。Value function 與 policy 被鎖在一個雙向改進的迴圈裡；任何能讓兩者趨向彼此一致的方法（async value iteration、modified policy iteration、Q-learning、actor-critic、PPO）都是 GPI 的一個實例。

**為什麼 $\gamma < 1$ 很重要。** Bellman operator 在 sup-norm 下是 $\gamma$-contraction：$\lVert T V - T V' \rVert_\infty \le \gamma \lVert V - V' \rVert_\infty$。Contraction 代表有唯一的 fixed point，而且是幾何速度收斂。拿掉 $\gamma < 1$，保證就沒了，你就需要有限視野或吸收型的終止 state。

```figure
value-iteration-gamma
```

## 動手做 (Build It)

### 步驟 1：建立 GridWorld MDP model

使用和 Lesson 01 相同的 4×4 GridWorld。我們加入一個 stochastic 變體：以機率 `0.1`，agent 會滑向隨機的垂直方向。

```python
SLIP = 0.1

def transitions(state, action):
    if state == TERMINAL:
        return [(state, 0.0, 1.0)]
    outcomes = []
    for direction, prob in action_probs(action):
        outcomes.append((apply_move(state, direction), -1.0, prob))
    return outcomes
```

`transitions(s, a)` 回傳一個 `(s', r, p)` 的 list。這就是整個 model。

### 步驟 2：policy evaluation

給定 policy `π(s) = {action: prob}`，迭代 Bellman equation，直到 $V$ 不再變動：

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = sum(pi_a * sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a))
                   for a, pi_a in policy(s).items())
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

### 步驟 3：policy improvement

把 $\pi$ 換成對 $V$ 取 greedy 的 policy。如果 $\pi$ 沒有改變就回傳，代表已經到達最佳解。

```python
def policy_improvement(V, gamma=0.99):
    new_policy = {}
    for s in states():
        best_a = max(
            ACTIONS,
            key=lambda a: sum(p * (r + gamma * V[s_prime])
                              for s_prime, r, p in transitions(s, a)),
        )
        new_policy[s] = best_a
    return new_policy
```

### 步驟 4：把它們串起來

```python
def policy_iteration(gamma=0.99):
    policy = {s: "up" for s in states()}   # arbitrary start
    for _ in range(100):
        V = policy_evaluation(lambda s: {policy[s]: 1.0}, gamma)
        new_policy = policy_improvement(V, gamma)
        if new_policy == policy:
            return V, policy
        policy = new_policy
```

在 4×4 上的典型收斂情況：4–6 次外層迭代。輸出 $V^{\ast}(0,0) \approx -6$，以及一個嚴格減少步數的 policy。

### 步驟 5：value iteration（單一迴圈版本）

```python
def value_iteration(gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in states()}
    while True:
        delta = 0.0
        for s in states():
            v = max(sum(p * (r + gamma * V[s_prime])
                       for s_prime, r, p in transitions(s, a))
                   for a in ACTIONS)
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            break
    policy = policy_improvement(V, gamma)
    return V, policy
```

同樣的 fixed point，程式碼更少。

## 常見陷阱 (Pitfalls)

- **忘了處理終止 state。** 如果對吸收型 state 套用 Bellman，它仍會挑出一個什麼都不會改變的「最佳 action」。用 `if s == terminal: V[s] = 0` 來防範。
- **Sup-norm 與 L2 收斂。** 用 `max |V_new - V|`，不要用平均。理論保證是建立在 sup-norm 上。
- **In-place 與 synchronous 更新。** 就地更新 `V[s]`（Gauss-Seidel）比另外開一個 `V_new` dict（Jacobi）收斂得更快。實務程式碼用 in-place。
- **Policy 平手。** 如果兩個 actions 的 Q-value 相等，`argmax` 每次迭代可能用不同方式打破平手，造成「policy 穩定」的檢查來回震盪。請用穩定的平手規則（固定順序中的第一個 action）。
- **State space 爆炸。** DP 每次 sweep 是 $O(\lvert S \rvert \cdot \lvert A \rvert)$。可以處理到約 10⁷ 個 states。超過就需要 function approximation（Phase 9 · 05 之後）。

## 實際使用 (Use It)

2026 年，DP 是正確性的基準，也是 planners 的內層迴圈：

| 使用情境 | 方法 |
|----------|------|
| 精確求解小型 tabular MDP | Value iteration（較簡單）或 policy iteration（外層步數較少） |
| 驗證 Q-learning / PPO 的實作 | 在 toy environment 上與 DP 最佳的 $V^{\ast}$ 比較 |
| Model-based RL（Phase 9 · 10） | 在學到的 transition model 上做 Bellman backup |
| AlphaZero / MuZero 的 planning | Monte Carlo Tree Search = async Bellman backup |
| Offline RL（CQL、IQL） | Conservative Q-iteration：對 OOD actions 加上懲罰的 DP |

每次有人說「最佳 value function」，他們指的是「DP 的 fixed point」。當你在論文裡看到 $V^{\ast}$ 或 $Q^{\ast}$，腦中就想像這個迴圈。

## 交付 (Ship It)

存成 `outputs/skill-dp-solver.md`：

```markdown
---
name: dp-solver
description: Solve a small tabular MDP exactly via policy iteration or value iteration. Report convergence behavior.
version: 1.0.0
phase: 9
lesson: 2
tags: [rl, dynamic-programming, bellman]
---

Given an MDP with a known model, output:

1. Choice. Policy iteration vs value iteration. Reason tied to |S|, |A|, γ.
2. Initialization. V_0, starting policy. Convergence sensitivity.
3. Stopping. Sup-norm tolerance ε. Expected number of sweeps.
4. Verification. V*(s_0) computed exactly. Greedy policy extracted.
5. Use. How this baseline will be used to debug/evaluate sampling-based methods.

Refuse to run DP on state spaces > 10⁷. Refuse to claim convergence without a sup-norm check. Flag any γ ≥ 1 on an infinite-horizon task as a guarantee violation.
```

## 練習 (Exercises)

1. **簡單。** 在 4×4 GridWorld 上，用 $\gamma \in \lbrace 0.9, 0.99 \rbrace$ 執行 value iteration。需要幾次 sweeps，才會讓 $\max \lvert \Delta V \rvert < 10^{-6}$？把 $V^{\ast}$ 印成 4×4 格子。
2. **中等。** 在 *stochastic* GridWorld（slip 機率 `0.1`）上比較 policy iteration 與 value iteration。計算：sweeps 次數、實際耗時、最終的 $V^{\ast}(0,0)$。哪一個在迭代次數上收斂得比較快？在實際耗時上呢？
3. **困難。** 實作 modified policy iteration：在 evaluation 步驟中，只跑 $k$ 次 sweeps，而不是跑到收斂。對 $k \in \lbrace 1, 2, 5, 10, 50 \rbrace$ 畫出 $V^{\ast}(0,0)$ 的誤差對 $k$ 的曲線。這條曲線告訴你 evaluation / improvement 之間有什麼取捨？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Policy iteration | 「DP 演算法」 | 交替進行 evaluation（$V^\pi$）與 improvement（對 $V^\pi$ 取 greedy 的 $\pi$），直到 policy 不再改變。 |
| Value iteration | 「比較快的 DP」 | 一次 sweep 就套用 Bellman optimality backup；以幾何速度收斂到 $V^{\ast}$。 |
| Bellman operator | 「那個遞迴」 | $(T V)(s) = \max_a \sum P (r + \gamma V(s'))$；在 sup-norm 下是 $\gamma$-contraction。 |
| Contraction | 「DP 為什麼會收斂」 | 任何滿足 $\lVert T x - T y \rVert \le \gamma \lVert x - y \rVert$ 的 operator $T$ 都有唯一的 fixed point。 |
| GPI | 「一切都是 DP」 | Generalized Policy Iteration：任何讓 $V$ 與 $\pi$ 趨向彼此一致的方法。 |
| Synchronous update | 「Jacobi 風格」 | 整個 sweep 都用舊的 $V$；容易分析但比較慢。 |
| In-place update | 「Gauss-Seidel 風格」 | 邊更新邊使用 $V$；實務上收斂得更快。 |

## 延伸閱讀 (Further Reading)

- [Sutton & Barto (2018). Ch. 4 — Dynamic Programming](http://incompleteideas.net/book/RLbook2020.pdf) — policy iteration 與 value iteration 的經典講解。
- [Bertsekas (2019). Reinforcement Learning and Optimal Control](http://www.athenasc.com/rlbook_athena.html) — 對 contraction-mapping 論證有嚴謹的處理。
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — modified policy iteration 及其收斂分析。
- [Howard (1960). Dynamic Programming and Markov Processes](https://mitpress.mit.edu/9780262582300/dynamic-programming-and-markov-processes/) — 最早的 policy iteration 論文。
- [Bertsekas & Tsitsiklis (1996). Neuro-Dynamic Programming](http://www.athenasc.com/ndpbook.html) — 從 DP 通往 approximate-DP / deep RL 的橋樑，後面每一堂課都會用到。
