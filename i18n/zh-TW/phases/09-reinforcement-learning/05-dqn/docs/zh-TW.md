# Deep Q-Networks (DQN)

> 2013 年：Mnih 用原始像素訓練出一個 Q-learning network，在七款 Atari 遊戲上擊敗所有傳統 RL agent。2015 年：擴展到 49 款遊戲，發表在 Nature，開啟了 deep-RL 時代。DQN 就是 Q-learning 加上三個讓 function approximation 穩定下來的技巧。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 分鐘

## 問題 (The Problem)

Tabular Q-learning 需要為每個 (state, action) 配對各存一個 Q-value。西洋棋盤約有 ~10⁴³ 個 states。一個 Atari frame 有 210×160×3 = 100,800 個特徵。Tabular RL 在幾千個 states 就撐不住了，更別說數十億個。

事後看來，解法很明顯：用神經網路 $Q(s, a; \theta)$ 取代 Q-table。但這個「事後看來很明顯」花了幾十年。在「deadly triad」之下，用 Q-learning 做天真的 function approximation 會發散，也就是 function approximation + bootstrapping + off-policy learning。Mnih et al. (2013, 2015) 找出三個讓學習穩定的工程技巧：

1. **Experience replay** 讓 transitions 之間去相關。
2. **Target network** 凍結 bootstrap 的目標。
3. **Reward clipping** 讓 gradient 的大小正規化。

Atari 上的 DQN 是第一次用單一架構、單一組 hyperparameter，從原始像素解決數十個控制問題。此後所有的「deep-RL」，包括 DDQN、Rainbow、Dueling、Distributional、R2D2、Agent57，都是疊在這個三招基礎之上。

## 核心概念 (The Concept)

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**目標。** DQN 對神經網路 Q-function 最小化一步 TD loss：

```math
L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}[ (r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2 ]
```

$\theta$ = online network，每一步用 gradient descent 更新。$\theta^-$ = target network，定期從 $\theta$ 複製（約每 ~10,000 步一次）。$D$ = 過去 transitions 的 replay buffer。

**三個技巧，依重要性排序：**

**Experience replay。** 一個容量約 `~10⁶` 個 transitions 的環狀 buffer。每個訓練步驟都從中均勻隨機取樣一個 minibatch。這會打破時間上的相關性（連續的 frames 幾乎一模一樣），讓網路能多次從罕見的有 reward 的 transitions 學習，並讓連續的 gradient 更新去相關。沒有它，on-policy TD 搭配神經網路在 Atari 上會發散。

**Target network。** 在 Bellman equation 兩側使用同一個網路 $Q(\cdot; \theta)$，會讓目標在每次更新時都跟著移動，也就是「追自己的尾巴」。解法：另外保留一個權重凍結的網路 $Q(\cdot; \theta^-)$。每 $C$ 步，複製 $\theta \to \theta^-$。這能讓回歸目標在數千次 gradient 步驟中保持穩定。Soft updates $\theta^- \leftarrow \tau \theta + (1-\tau) \theta^-$（用於 DDPG、SAC）是比較平滑的變體。

**Reward clipping。** Atari 的 reward 大小從 1 到 1000 以上都有。把它 clip 到 $\lbrace-1, 0, +1 \rbrace$，可以避免任何單一遊戲主導 gradient。當 reward 的大小有意義時是錯的；對只有正負號重要的 Atari 則沒問題。

**Double DQN。** Hasselt (2016) 修正 maximization bias：用 online net 來 *選擇* action，用 target net 來 *評估* 它。

```math
\text{target} = r + \gamma Q(s', \arg\max_{a'} Q(s', a'; \theta); \theta^-)
```

可直接替換，而且表現一致地更好。預設就用它。

**其他改進（Rainbow, 2017）：** prioritized replay（更常取樣 TD error 高的 transitions）、dueling 架構（分開的 $V(s)$ 與 advantage heads）、noisy networks（學出來的 exploration）、n-step returns、distributional Q（C51/QR-DQN）、multi-step bootstrapping。每一項大約增加幾個百分點；增益大致可以相加。

```figure
f3-dqn-stability
```

## 動手做 (Build It)

這裡的程式碼只用標準函式庫、不用 numpy：我們在一個很小的連續 GridWorld 上，使用手刻的單隱藏層 MLP，所以每個訓練步驟只要幾微秒。演算法和大規模的 Atari DQN 完全相同。

### 步驟 1：replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

Atari 的容量是 ~10⁶，和原始 DQN 一樣（它在開始學習前會先收集 50,000 個 transitions）；我們的 toy env 用 5,000 就夠了。

### 步驟 2：一個很小的 Q-network（手刻 MLP）

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass：linear → ReLU → linear。整個網路就這樣。

### 步驟 3：DQN 更新

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

整體形狀就是 Lesson 04 的 Q-learning，只有兩點不同：(a) 我們對可微分的 $Q(\cdot; \theta)$ 做 backprop，而不是索引表格，(b) 目標用的是 $Q(\cdot; \theta^-)$。

### 步驟 4：外層迴圈

每個 episode：對 $Q(\cdot; \theta)$ 以 ε-greedy 行動，把 transitions 推進 buffer，取樣一個 minibatch，做一次 gradient 步驟，定期同步 $\theta^- \leftarrow \theta$。模式如下：

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

在我們這個 16 維 one-hot state 的小 GridWorld 上，agent 大約在 ~500 個 episodes 內學到接近最佳的 policy。在 Atari 上，要放大到 2 億個 frames，並加上 CNN 特徵擷取器。

## 常見陷阱 (Pitfalls)

- **Deadly triad。** Function approximation + off-policy + bootstrapping 可能發散。DQN 用 target net + replay 來緩解；兩者都不要拿掉。
- **Exploration。** ε 必須遞減，通常在訓練的前 ~10% 從 1.0 降到 0.01。早期 exploration 不足，Q-net 會收斂到局部的盆地。
- **高估。** 對有雜訊的 Q 取 `max` 會往上偏。正式環境一律使用 Double DQN。
- **Reward 尺度。** 對 reward 做 clip 或正規化；gradient 的大小與 reward 的大小成正比。
- **Replay buffer 冷啟動。** 在 buffer 裝有幾千個 transitions 之前不要訓練。早期對 ~20 個樣本做 gradient 會 overfit。
- **Target 同步頻率。** 太頻繁 ≈ 沒有 target net；太少 ≈ target 過時。Atari DQN 用 10,000 個 env 步。經驗法則：每約 1/100 的訓練視野同步一次。
- **觀測的前處理。** Atari DQN 堆疊 4 個 frames，讓 state 具有 Markov 性質。任何帶有速度資訊的 env 都需要 frame-stacking 或 recurrent state。

## 實際使用 (Use It)

2026 年，DQN 很少是 state-of-the-art，但仍是 off-policy 演算法的參考標準：

| 任務 | 首選方法 | 為什麼不用 DQN？ |
|------|----------|------------------|
| 離散 action、類 Atari | Rainbow DQN 或 Muesli | 同一個框架，更多技巧。 |
| 連續控制 | SAC / TD3 (Phase 9 · 07) | DQN 沒有 policy network。 |
| On-policy / 高吞吐量 | PPO (Phase 9 · 08) | 沒有 replay buffer；比較容易擴展。 |
| Offline RL | CQL / IQL / Decision Transformer | 保守的 Q 目標，不會有 bootstrapping 爆炸。 |
| 大型離散 action 空間（推薦系統） | 搭配 action embedding 的 DQN，或 IMPALA | 可行；細節設計很重要。 |
| LLM RL | PPO / GRPO | 是序列層級，不是步驟層級；loss 不同。 |

這些課題仍然通用。Replay 與 target networks 出現在 SAC、TD3、DDPG、SAC-X、AlphaZero 的 self-play buffer，以及每個 offline RL 方法中。Reward clipping 則以 PPO 中 advantage 正規化的形式延續下來。這個架構就是藍圖。

## 交付 (Ship It)

存成 `outputs/skill-dqn-trainer.md`：

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。畫出每個 episode 的 return 曲線。需要多少個 episodes，滾動平均才會超過 -10？
2. **中等。** 停用 target network（Bellman 目標兩側都用 online net）。測量訓練的不穩定程度：return 是震盪還是發散？
3. **困難。** 加入 Double DQN：用 online net 挑選 `argmax a'`，用 target net 評估。在有雜訊 reward 的 GridWorld 上，比較有無 Double DQN 時，1,000 個 episodes 後 `Q(s_0, best_a)` 相對於真實 $V^{\ast}(s_0)$ 的 bias。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| DQN | 「Deep Q-learning」 | 使用神經網路 Q-function、replay buffer 與 target network 的 Q-learning。 |
| Experience replay | 「打亂的 transitions」 | 每個 gradient 步驟都均勻取樣的環狀 buffer；讓資料去相關。 |
| Target network | 「凍結的 bootstrap」 | 用於 Bellman 目標的 Q 的定期副本；讓訓練穩定。 |
| Deadly triad | 「RL 為什麼會發散」 | Function approximation + bootstrapping + off-policy = 沒有收斂保證。 |
| Double DQN | 「修正 maximization bias」 | Online net 選擇 action，target net 評估它。 |
| Dueling DQN | 「V 與 A 的 heads」 | 把 Q 分解為 V + A - mean(A)；輸出相同，gradient 流動更好。 |
| Rainbow | 「所有技巧」 | 把 DDQN + PER + dueling + n-step + noisy + distributional 合為一體。 |
| PER | 「Prioritized Replay」 | 依 TD-error 大小成比例地取樣 transitions。 |

## 延伸閱讀 (Further Reading)

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 年 NeurIPS workshop 論文，開啟了 deep RL。
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature 論文，49 款遊戲的 DQN。
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN。
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN。
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — 疊加各種技巧的論文。
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書對「deadly triad」（function approximation + bootstrapping + off-policy）的處理，DQN 的 target network 與 replay buffer 就是為了馴服它而設計的。
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — 在 ablation 研究中使用的單檔 DQN 參考實作；適合和本課從零開始的版本對照閱讀。
