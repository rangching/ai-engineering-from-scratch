# Actor-Critic：A2C 與 A3C (Actor-Critic — A2C and A3C)

> REINFORCE 的雜訊很大。加入一個學習 $\hat{V}(s)$ 的 critic，從 return 中減掉它，你就得到期望值相同、但變異數低很多的 advantage。這就是 actor-critic。A2C 以同步方式執行；A3C 則跨多個 thread 執行。兩者都是每個現代 deep-RL 方法的思考模型。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (TD Learning), Phase 9 · 06 (REINFORCE)
**Time:** ~75 分鐘

## 問題 (The Problem)

單純的 REINFORCE 能用，但它的變異數很糟。Monte Carlo returns $G_t$ 在不同 episodes 之間可能差到 10 倍。把這些雜訊乘上 $\nabla \log \pi$ 再取平均，得到的 gradient 估計器要花上幾千個 episodes，才能把 policy 移動到 DQN 用少得多的更新就能到達的距離。

變異數來自使用原始的 returns。如果你減去一個 baseline $b(s_t)$（任何 state 的函式，包括學出來的 value），期望值不變，變異數卻下降。最好又可行的 baseline 是 $\hat{V}(s_t)$。現在乘上 $\nabla \log \pi$ 的量就是 *advantage*：

```math
A(s, a) = G - \hat{V}(s)
```

如果 action 產生了高於平均的 return，它就是好的；低於平均就是壞的。帶有學出來的 critic 的 REINFORCE 就是 *actor-critic*。Critic 給 actor 一位低變異數的老師。這是 2015 年之後每個 deep-policy 方法的基礎（A2C、A3C、PPO、SAC、IMPALA）。

## 核心概念 (The Concept)

![Actor-critic: policy net plus value net, TD residual as advantage](../assets/actor-critic.svg)

**兩個網路，一個共用的 loss：**

- **Actor** $\pi_\theta(a \mid s)$：policy。取樣來行動。用 policy gradient 訓練。
- **Critic** $V_\phi(s)$：估計從 state 出發的期望 return。訓練目標是最小化 $(V_\phi(s) - \text{target})^2$。

**Advantage。** 兩種標準形式：

- *MC advantage：* $A_t = G_t - V_\phi(s_t)$。Unbiased，變異數較高。
- *TD advantage：* $A_t = r_{t+1} + \gamma V_\phi(s_{t+1}) - V_\phi(s_t)$。有 bias（用了 $V_\phi$），但變異數低很多。也稱為 *TD residual* $\delta_t$。

**n-step advantage。** 在兩者之間做內插：

```math
A_t^{(n)} = r_{t+1} + \gamma r_{t+2} + \ldots + \gamma^{n-1} r_{t+n} + \gamma^n V_\phi(s_{t+n}) - V_\phi(s_t)
```

$n = 1$ 是純 TD。$n = \infty$ 是 MC。大多數實作在 Atari 上用 $n = 5$，在 MuJoCo 上的 PPO 用 $n = 2048$。

**Generalized Advantage Estimation (GAE)。** Schulman et al. (2016) 提出對所有 n-step advantages 做指數加權平均：

```math
A_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma \lambda)^l \delta_{t+l}
```

其中 $\lambda \in [0, 1]$。$\lambda = 0$ 是 TD（低變異數、高 bias）。$\lambda = 1$ 是 MC（高變異數、unbiased）。$\lambda = 0.95$ 是 2026 年的預設值，調到 bias/variance 旋鈕在你想要的位置為止。

**A2C：同步的 advantage actor-critic。** 在 $N$ 個平行 environments 上各收集 $T$ 步。為每一步計算 advantages。在合併的 batch 上更新 actor 與 critic。重複。這是 A3C 比較簡單、也更容易擴展的兄弟。

**A3C：非同步的 advantage actor-critic。** Mnih et al. (2016)。產生 $N$ 個 worker thread，每個執行一個 env。每個 worker 在自己的 rollout 上於本地計算 gradients，再以非同步方式把它們套用到共用的 parameter server。不需要 replay buffer，worker 靠執行不同的軌跡來去相關。A3C 證明了你可以在 CPU 上大規模訓練。到了 2026 年，以 GPU 為基礎的 A2C（批次平行 envs）佔主導，因為 GPU 想要大 batch。

**合併的 loss。**

```math
L(\theta, \phi) = -\mathbb{E}[ A_t \cdot \log \pi_\theta(a_t \mid s_t) ] + c_v \cdot \mathbb{E}[(V_\phi(s_t) - G_t)^2] - c_e \cdot \mathbb{E}[H(\pi_\theta(\cdot \mid s_t))]
```

三項：policy-gradient loss、value 回歸、entropy bonus。`c_v ~ 0.5`、`c_e ~ 0.01` 是標準的起點。

```figure
actor-critic
```

## 動手做 (Build It)

### 步驟 1：critic

用 MSE 更新的線性 critic $V_\phi(s) = w \cdot \text{features}(s)$：

```python
def critic_update(w, x, target, lr):
    v_hat = dot(w, x)
    err = target - v_hat
    for j in range(len(w)):
        w[j] += lr * err * x[j]
    return v_hat
```

在 tabular env 上，critic 幾百個 episodes 內就會收斂。在 Atari 上，把線性 critic 換成共用的 CNN trunk + value head。

### 步驟 2：n-step advantage

給定長度為 $T$ 的 rollout，以及 bootstrap 的最終 $V(s_T)$：

```python
def compute_advantages(rewards, values, gamma=0.99, lam=0.95, last_value=0.0):
    advantages = [0.0] * len(rewards)
    gae = 0.0
    for t in reversed(range(len(rewards))):
        next_v = values[t + 1] if t + 1 < len(values) else last_value
        delta = rewards[t] + gamma * next_v - values[t]
        gae = delta + gamma * lam * gae
        advantages[t] = gae
    returns = [a + v for a, v in zip(advantages, values)]
    return advantages, returns
```

`returns` 是 critic 的目標。`advantages` 是要乘上 $\nabla \log \pi$ 的量。

### 步驟 3：合併更新

```python
for step_i, (x, a, _r, probs) in enumerate(traj):
    adv = advantages[step_i]
    target_v = returns[step_i]

    # critic
    critic_update(w, x, target_v, lr_v)

    # actor
    for i in range(N_ACTIONS):
        grad_logpi = (1.0 if i == a else 0.0) - probs[i]
        for j in range(N_FEAT):
            theta[i][j] += lr_a * adv * grad_logpi * x[j]
```

On-policy，每次更新一個 rollout，actor 與 critic 各用不同的 learning rate。

### 步驟 4：平行化（A3C 對 A2C）

- **A3C：** 啟動 $N$ 個 threads。每個執行自己的 env 與自己的 forward pass。定期把 gradient 更新推送到共用的 master。master 上不加鎖，race 沒關係，只會增加一些雜訊。
- **A2C：** 在單一行程中執行 $N$ 個 env 實例，把觀測堆成 `[N, obs_dim]` 的 batch，做批次 forward pass 與批次 backward pass。GPU 使用率更高、具決定性、也更容易推理。2026 年的預設。

為了清楚起見，我們的 toy 程式是單一 thread；改寫成批次的 A2C 只要三行 numpy。

## 常見陷阱 (Pitfalls)

- **在 actor gradient 之前 critic 就有 bias。** 如果 critic 是隨機的，它的 baseline 沒有資訊量，你是在純雜訊上訓練。在開啟 policy gradient 之前先暖機 critic 幾百步，或使用較慢的 actor learning rate。
- **Advantage 正規化。** 對每個 batch 把 advantages 正規化成零均值/單位標準差。幾乎零成本就能大幅穩定訓練。
- **共用的 trunk。** 對影像輸入，讓 actor 與 critic 共用特徵擷取器，heads 則分開。共用的特徵會同時從兩個 loss 受益。
- **On-policy 的約定。** A2C 只重複使用資料做恰好一次更新。再多，你的 gradient 就有 bias（PPO 加入的就是 importance-sampling 修正）。
- **Entropy 崩塌。** 沒有 $c_e > 0$，policy 幾百次更新內就會變得近乎 deterministic，停止 exploration。
- **Reward 尺度。** Advantage 的大小取決於 reward 的尺度。正規化 rewards（例如除以滾動標準差），讓不同任務的 gradient 大小一致。

## 實際使用 (Use It)

A2C/A3C 在 2026 年很少是最終選擇，但它們是之後一切方法所精進的架構：

| 方法 | 與 A2C 的關係 |
|------|---------------|
| PPO | A2C + 為了多 epoch 更新而 clip 過的 importance ratio |
| IMPALA | A3C + V-trace off-policy 修正 |
| SAC (Phase 9 · 07) | 帶 soft-value critic 的 off-policy A2C（下一課） |
| GRPO (Phase 9 · 12) | 沒有 critic 的 A2C：group-relative advantage |
| DPO | 壓縮成偏好排序 loss 的 A2C，不需取樣 |
| AlphaStar / OpenAI Five | 搭配 league training + imitation 預訓練的 A2C |

如果你在 2026 年的論文看到「advantage」，就想到 actor-critic。

## 交付 (Ship It)

存成 `outputs/skill-actor-critic-trainer.md`：

```markdown
---
name: actor-critic-trainer
description: Produce an A2C / A3C / GAE configuration for a given environment, with advantage estimation and loss weights specified.
version: 1.0.0
phase: 9
lesson: 7
tags: [rl, actor-critic, gae]
---

Given an environment and compute budget, output:

1. Parallelism. A2C (GPU batched) vs A3C (CPU async) and the number of workers.
2. Rollout length T. Steps per env per update.
3. Advantage estimator. n-step or GAE(λ); specify λ.
4. Loss weights. `c_v` (value), `c_e` (entropy), gradient clip.
5. Learning rates. Actor and critic (separate if using).

Refuse single-worker A2C on environments with horizon > 1000 (too on-policy, too slow). Refuse to ship without advantage normalization. Flag any run with `c_e = 0` and observed entropy < 0.1 as entropy-collapsed.
```

## 練習 (Exercises)

1. **簡單。** 在 4×4 GridWorld 上，用 MC advantage $(G_t - V(s_t))$ 訓練 actor-critic。和 Lesson 06 帶滾動平均 baseline 的 REINFORCE 比較樣本效率。
2. **中等。** 改用 TD-residual advantage $(r + \gamma V(s') - V(s))$。測量 advantage batches 的變異數。降低了多少？
3. **困難。** 實作 GAE(λ)。掃描 $\lambda \in \lbrace 0, 0.5, 0.9, 0.95, 1.0 \rbrace$。畫出最終 return 對樣本效率。這個任務的 bias/variance 最佳點在哪裡？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Actor | 「Policy net」 | $\pi_\theta(a \mid s)$，由 policy gradient 更新。 |
| Critic | 「Value net」 | $V_\phi(s)$，透過對 returns / TD 目標做 MSE 回歸來更新。 |
| Advantage | 「比平均好多少」 | $A(s, a) = Q(s, a) - V(s)$ 或它的估計器。$\nabla \log \pi$ 的乘數。 |
| TD residual | 「δ」 | $\delta_t = r + \gamma V(s') - V(s)$；一步的 advantage 估計。 |
| GAE | 「內插旋鈕」 | n-step advantages 的指數加權和，由 $\lambda$ 參數化。 |
| A2C | 「同步 actor-critic」 | 跨 envs 批次處理；每個 rollout 做一次 gradient 步驟。 |
| A3C | 「非同步 actor-critic」 | Worker threads 把 gradients 推送到共用的 param server。原始論文；2026 年較少見。 |
| Bootstrap | 「在視野終點使用 V」 | 截斷 rollout，加上 $\gamma^n V(s_{t+n})$ 來補完這個總和。 |

## 延伸閱讀 (Further Reading)

- [Mnih et al. (2016). Asynchronous Methods for Deep Reinforcement Learning](https://arxiv.org/abs/1602.01783) — A3C，最早的非同步 actor-critic 論文。
- [Schulman et al. (2016). High-Dimensional Continuous Control Using Generalized Advantage Estimation](https://arxiv.org/abs/1506.02438) — GAE。
- [Sutton & Barto (2018). Ch. 13 — Actor-Critic Methods](http://incompleteideas.net/book/RLbook2020.pdf) — 基礎；當 critic 是神經網路時，搭配第 9 章的 function approximation 一起讀。
- [Espeholt et al. (2018). IMPALA](https://arxiv.org/abs/1802.01561) — 可擴展的分散式 actor-critic，搭配 V-trace off-policy 修正。
- [OpenAI Baselines / Stable-Baselines3](https://stable-baselines3.readthedocs.io/) — 值得一讀的正式 A2C/PPO 實作。
- [Konda & Tsitsiklis (2000). Actor-Critic Algorithms](https://papers.nips.cc/paper/1786-actor-critic-algorithms) — 雙時間尺度 actor-critic 分解的基礎收斂結果。
