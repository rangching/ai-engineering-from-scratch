# Policy Gradient：從零實作 REINFORCE (Policy Gradient — REINFORCE from Scratch)

> 別再估計 value 了。直接把 policy 參數化，計算期望 return 的 gradient，往上坡走一步。Williams (1992) 用一個定理就寫完了。這就是 PPO、GRPO，以及每個 LLM RL 迴圈存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 分鐘

## 問題 (The Problem)

Q-learning 與 DQN 把 *value* function 參數化。你用 `argmax Q` 來挑選 actions。這對離散 actions 與離散 states 沒問題。但當 actions 是連續的就不行了（要對 10 維的力矩做哪一種 `argmax`？），或是當你想要 stochastic policy 時也不行（`argmax` 本質上是 deterministic）。

Policy gradients 改為把 *policy* 參數化。$\pi_\theta(a \mid s)$ 是一個輸出 actions 分布的神經網路。從中取樣就能行動。計算期望 return 對 $\theta$ 的 gradient。往上坡走一步。沒有 `argmax`。沒有 Bellman 遞迴。只是對 $J(\theta) = \mathbb E_{\pi_\theta}[G]$ 做 gradient ascent。

REINFORCE 定理 (Williams 1992) 告訴你這個 gradient 是可以計算的：$\nabla J(\theta) = \mathbb E_\pi[ G \cdot \nabla_\theta \log \pi_\theta(a \mid s) ]$。執行一個 episode。計算 return。在每一步乘上 $\nabla \log \pi_\theta(a \mid s)$。取平均。做 gradient ascent。完成。

2026 年的每個 LLM-RL 演算法，包括 PPO、DPO、GRPO，都是 REINFORCE 的改良。把它練到身體記得，是學習本階段其餘內容，以及 Phase 10 · 07（RLHF 實作）與 Phase 10 · 08（DPO）的前提。

## 核心概念 (The Concept)

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**Policy gradient 定理。** 對任何由 $\theta$ 參數化的 policy $\pi_\theta$：

```math
\nabla J(\theta) = \mathbb{E}_{\tau \sim \pi_\theta}[ \sum_{t=0}^{T} G_t \cdot \nabla_\theta \log \pi_\theta(a_t \mid s_t) ]
```

其中 $G_t = \sum_{k=t}^{T} \gamma^{k-t} r_{k+1}$ 是從第 $t$ 步開始的折扣 return。期望值是對從 $\pi_\theta$ 取樣出的完整軌跡 $\tau$ 取的。

**證明很短。** 在期望值之下對 $J(\theta) = \sum_\tau P(\tau; \theta) G(\tau)$ 微分。使用 $\nabla P(\tau; \theta) = P(\tau; \theta) \nabla \log P(\tau; \theta)$（log-derivative trick）。把 $\log P(\tau; \theta) = \sum \log \pi_\theta(a_t \mid s_t) + \text{不依賴 } \theta \text{ 的 environment 項}$ 拆開。Environment 項會消失。兩行代數就得到這個定理。

**降低變異數的技巧。** 單純的 REINFORCE 變異數大得驚人：returns 有雜訊，$\nabla \log \pi$ 有雜訊，兩者相乘雜訊更大。兩個標準的修正：

1. **Baseline 相減。** 把 $G_t$ 換成 $G_t - b(s_t)$，其中 baseline $b(s_t)$ 不依賴 $a_t$。它是 unbiased，因為 $\mathbb{E}[b(s_t) \cdot \nabla \log \pi(a_t \mid s_t)] = 0$。典型的選擇：$b(s_t) = \hat{V}(s_t)$，由 critic 學出來 → actor-critic（Lesson 07）。
2. **Reward-to-go。** 把 $\sum_t G_t \cdot \nabla \log \pi_\theta(a_t \mid s_t)$ 換成 $\sum_t G_t^{\text{from } t} \cdot \nabla \log \pi_\theta(a_t \mid s_t)$。對某個 action 來說，只有未來的 returns 重要，過去的 rewards 只會貢獻零均值的雜訊。

合起來，你得到：

```math
\nabla J \approx (1/N) \sum_{i=1}^{N} \sum_{t=0}^{T_i} [ G_t^{(i)} - \hat{V}(s_t^{(i)}) ] \cdot \nabla_\theta \log \pi_\theta(a_t^{(i)} \mid s_t^{(i)})
```

這就是帶 baseline 的 REINFORCE，也是 A2C（Lesson 07）與 PPO（Lesson 08）的直系祖先。

**Softmax policy 參數化。** 對離散 actions，標準的選擇：

```math
\pi_\theta(a \mid s) = \exp(f_\theta(s, a)) / \sum_{a'} \exp(f_\theta(s, a'))
```

其中 $f_\theta$ 是任何對每個 action 輸出一個分數的神經網路。Gradient 有簡潔的形式：

```math
\nabla_\theta \log \pi_\theta(a \mid s) = \nabla_\theta f_\theta(s, a) - \sum_{a'} \pi_\theta(a' \mid s) \nabla_\theta f_\theta(s, a')
```

也就是，被採取的 action 的分數，減去它在 policy 下的期望值。

**連續 actions 用 Gaussian policy。** $\pi_\theta(a \mid s) = N(\mu_\theta(s), \sigma_\theta(s))$。$\nabla \log N(a; \mu, \sigma)$ 有閉合形式。這就是 Phase 9 · 07 的 SAC 所需要的全部。

```figure
policy-gradient-landscape
```

## 動手做 (Build It)

### 步驟 1：softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

對 tabular env 使用 linear policy（每個 action 一個權重向量）。對 Atari，換成 CNN 並保留 softmax head。

### 步驟 2：取樣與 log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### 步驟 3：記錄 log-probs 的 rollout

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### 步驟 4：REINFORCE 更新

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient $\nabla \log \pi(a \mid s) = e_a - \pi(\cdot \mid s)$（$a$ 的 one-hot 減去機率）是 softmax policy gradients 的核心。把它刻進肌肉記憶裡。

### 步驟 5：baselines

對最近幾個 episodes 的 $G$ 取滾動平均，這樣的降變異數程度就足以讓 4×4 GridWorld 跑起來；大約需要 ~500 個 episodes 收斂。把 baseline 升級為學出來的 $\hat{V}(s)$，你就得到 actor-critic。

## 常見陷阱 (Pitfalls)

- **Gradient 爆炸。** Returns 可能非常大。在乘上 $\nabla \log \pi$ 之前，一律把整個 batch 的 $G$ 正規化成 `~N(0, 1)`。
- **Entropy 崩塌。** Policy 太早收斂到近乎 deterministic 的 action，停止 exploration，然後卡住。修正：在目標中加入 entropy bonus $\beta \cdot H(\pi(\cdot \mid s))$。
- **高變異數。** 單純的 REINFORCE 需要幾千個 episodes。加入 critic baseline（Lesson 07），或 TRPO/PPO 的 trust region（Lesson 08）是標準修正。
- **樣本效率低。** On-policy 代表每個 transition 在一次更新後就丟掉。透過 importance sampling 做 off-policy 修正可以把資料找回來，代價是變異數（PPO 的 ratio 就是被 clip 過的 IS 權重）。
- **Non-stationary 的 gradients。** 100 個 episodes 之前的同一個 gradient 用的是舊的 $\pi$。On-policy 方法因此每隔幾次 rollout 就更新一次。
- **Credit assignment。** 沒有 reward-to-go，過去的 rewards 會貢獻雜訊。一律使用 reward-to-go。

## 實際使用 (Use It)

2026 年，REINFORCE 很少被直接執行，但它的 gradient 公式無所不在：

| 使用情境 | 衍生方法 |
|----------|----------|
| 連續控制 | 搭配 Gaussian policy 的 PPO / SAC |
| LLM RLHF | 帶 KL 懲罰、在 token 層級 policy 上執行的 PPO |
| LLM 推理 (DeepSeek) | GRPO：帶 group-relative baseline、沒有 critic 的 REINFORCE |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| 離散 action 機器人 | A2C、A3C、PPO |
| 只有偏好的設定 | DPO：把 REINFORCE 改寫成偏好似然 loss，不需取樣 |

當你在 2026 年的訓練腳本裡讀到 `loss = -advantage * log_prob`，那就是帶 baseline 的 REINFORCE。整篇整篇的論文（DPO、GRPO、RLOO）都是在這一行之上做降變異數的技巧。

## 交付 (Ship It)

存成 `outputs/skill-policy-gradient-trainer.md`：

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## 練習 (Exercises)

1. **簡單。** 在 4×4 GridWorld 上，用 linear softmax policy 實作 REINFORCE。不使用 baseline，訓練 1,000 個 episodes。畫出 learning curve；測量變異數（returns 的標準差）。
2. **中等。** 加入滾動平均 baseline。再訓練一次。和單純版本比較樣本效率與變異數。baseline 讓收斂所需的步數減少了多少？
3. **困難。** 加入 entropy bonus $\beta \cdot H(\pi)$。掃描 $\beta \in \lbrace 0, 0.01, 0.1, 1.0 \rbrace$。畫出最終 return 與 policy entropy。這個任務的最佳點在哪裡？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Policy gradient | 「直接訓練 policy」 | $\nabla J(\theta) = \mathbb{E}[G \cdot \nabla \log \pi_\theta(a \mid s)]$；由 log-derivative trick 推導。 |
| REINFORCE | 「最早的 PG 演算法」 | Williams (1992)；Monte Carlo returns 乘上 log-policy gradient。 |
| Log-derivative trick | 「Score function 估計器」 | $\nabla P(\tau;\theta) = P(\tau;\theta) \cdot \nabla \log P(\tau;\theta)$；讓期望值的 gradient 變得可計算。 |
| Baseline | 「降低變異數」 | 從 $G$ 中減去的任何 $b(s)$；因為 $\mathbb{E}[b \cdot \nabla \log \pi] = 0$ 所以是 unbiased。 |
| Reward-to-go | 「只算未來的 returns」 | 用 $G_t^{\text{from } t}$ 取代完整的 $G_0$；正確且變異數較低。 |
| Entropy bonus | 「鼓勵 exploration」 | $+\beta \cdot H(\pi(\cdot \mid s))$ 項，避免 policy 崩塌。 |
| On-policy | 「用剛看到的資料訓練」 | Gradient 的期望值是對目前的 policy 取的，因此無法直接重複使用舊資料。 |
| Advantage | 「比平均好多少」 | $A(s, a) = G(s, a) - V(s)$；帶 baseline 的 REINFORCE 所乘的那個有正負號的量。 |

## 延伸閱讀 (Further Reading)

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — REINFORCE 的原始論文。
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — 搭配 function approximation 的現代 policy-gradient 定理。
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書講解。
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — 清楚的教學說明，附 PyTorch 程式碼。
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — 降低變異數，以及把 REINFORCE 連到 trust-region 家族（TRPO、PPO）的 natural-gradient 觀點。
