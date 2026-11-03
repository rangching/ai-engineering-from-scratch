# Proximal Policy Optimization (PPO)

> A2C 在一次更新後就把每個 rollout 丟掉。PPO 把 policy gradient 包在 clip 過的 importance ratio 裡，讓你能對同一批資料做 10 個以上的 epochs，而 policy 不會爆掉。Schulman et al. (2017)。到 2026 年仍是預設的 policy-gradient 演算法。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 分鐘

## 問題 (The Problem)

A2C（Lesson 07）是 on-policy：gradient $\mathbb E_{\pi_\theta}[A \cdot \nabla \log \pi_\theta]$ 需要從 *目前的* $\pi_\theta$ 取樣的資料。更新一次，$\pi_\theta$ 就變了；你用過的資料現在變成 off-policy。重複使用，你的 gradient 就有 bias。

Rollouts 很昂貴。在 Atari 上，跨 8 個 envs × 128 步的一次 rollout = 1024 個 transitions，以及十幾秒的 environment 時間。只做一次 gradient 步驟就丟掉太浪費。

Trust Region Policy Optimization（TRPO, Schulman 2015）是第一個解法：限制每次更新，讓新舊 policy 之間的 KL divergence 保持在 $\delta$ 以下。理論上很乾淨，但每次更新都需要解一次 conjugate-gradient。2026 年沒有人在跑 TRPO。

PPO（Schulman et al. 2017）用一個簡單的 clipped objective 取代硬性的 trust-region 限制。多一行程式碼。每個 rollout 十個 epochs。不需要 conjugate gradients。理論保證夠用。九年後，從 MuJoCo 到 RLHF，它仍是預設的 policy-gradient 演算法。

## 核心概念 (The Concept)

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**Importance ratio。**

```math
r_t(\theta) = \frac{\pi_\theta(a_t \mid s_t)}{\pi_{\theta_{\text{old}}}(a_t \mid s_t)}
```

這是新 policy 對收集資料的那個 policy 的 likelihood ratio。$r_t = 1$ 表示沒有變化。$r_t = 2$ 表示新 policy 採取 $a_t$ 的可能性是舊的兩倍。

**Clipped surrogate。**

```math
L^{\text{CLIP}}(\theta) = \mathbb{E}_t \left[ \min\left( r_t(\theta) A_t, \mathop{\mathrm{clip}}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
```

兩項：

- 如果 advantage $A_t > 0$ 而 ratio 試圖增長超過 $1 + \epsilon$，clip 會讓 gradient 變平：不要把好的 action 推到比舊機率高出超過 $+\epsilon$。
- 如果 advantage $A_t < 0$ 而 ratio 試圖降到 $1 - \epsilon$ 以下（意思是我們會讓壞的 action 的可能性比舊 policy 低很多），clip 會讓 gradient 變平：不要把壞的 action 推到比舊機率低超過 $-\epsilon$。

`min` 處理另一個方向：如果 ratio 已經朝 *有利的* 方向移動，你仍然會得到 gradient（對會傷害你的那一側不做 clip）。

典型的 $\epsilon = 0.2$。把 objective 畫成 $r_t$ 的函式：一個分段線性函數，在「好的那一側」有平坦的屋頂，在「壞的那一側」有平坦的地板。

**完整的 PPO loss。**

```math
L(\theta, \phi) = L^{\text{CLIP}}(\theta) - c_v \cdot (V_\phi(s_t) - V_t^{\text{target}})^2 + c_e \cdot H(\pi_\theta(\cdot \mid s_t))
```

和 A2C 相同的 actor-critic 結構。三個係數，通常是 $c_v = 0.5$、$c_e = 0.01$、$\epsilon = 0.2$。

**訓練迴圈。**

1. 在 $N$ 個平行 envs 上各收集 $T$ 步，共 $N \times T$ 個 transitions。
2. 計算 advantages（GAE），把它們凍結成常數。
3. 把 $\pi_{\theta_{\text{old}}}$ 凍結成目前 $\pi_\theta$ 的快照。
4. 進行 $K$ 個 epochs，對每個 `(s, a, A, V_target, log π_old(a|s))` 的 minibatch：
   - 計算 $r_t(\theta) = \exp(\log \pi_\theta(a \mid s) - \log \pi_{\text{old}}(a \mid s))$。
   - 套用 $L^{\text{CLIP}}$ + value loss + entropy。
   - 做一次 gradient 步驟。
5. 丟掉這個 rollout。回到步驟 1。

$K = 10$ 搭配 64 的 minibatches 是一組標準的 hyperparameter。PPO 很穩健：確切的數字在 ±50% 以內很少有影響。

**KL-penalty 變體。** 原始論文提出另一種使用自適應 KL 懲罰的替代做法：$L = L^{\text{PG}} - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{old}})$，其中 $\beta$ 依觀察到的 KL 調整。Clipping 版本成為主流；KL 變體則在 RLHF 中延續下來（在那裡，對 reference policy 的 KL 本來就是你一直想要的另一個獨立限制）。

```figure
ppo-clip
```

## 動手做 (Build It)

### 步驟 1：在 rollout 時記錄 `log π_old(a | s)`

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

快照只在 rollout 時取一次。更新的 epochs 期間不會改變。

### 步驟 2：計算 GAE advantages（Lesson 07）

和 A2C 相同。對整個 batch 做正規化。

### 步驟 3：clipped surrogate 更新

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

「clip → gradient 為零」這個模式是 PPO 的核心。如果新 policy 已經在有利的方向上漂得太遠，更新就會停止。

### 步驟 4：value 與 entropy

和 A2C 一樣，加入對 critic 目標的標準 MSE，以及 actor 上的 entropy bonus。

### 步驟 5：診斷

每次更新都要觀察三件事：

- **平均 KL** $\mathbb{E}[\log \pi_{\text{old}} - \log \pi_\theta]$。應維持在 `[0, 0.02]`。如果衝過 `0.1`，就降低 `K_EPOCHS` 或 `LR`。
- **Clip fraction**：ratio 落在 $[1-\epsilon, 1+\epsilon]$ 之外的樣本比例。應為 `~0.1-0.3`。如果 `~0`，clip 從不觸發 → 提高 `LR` 或 `K_EPOCHS`。如果 `~0.5+`，代表你對 rollout 過度擬合 → 降低它們。
- **Explained variance** $1 - \text{Var}(V_{\text{target}} - V_{\text{pred}}) / \text{Var}(V_{\text{target}})$。Critic 品質的指標。隨著 critic 學習，應朝 1 攀升。

## 常見陷阱 (Pitfalls)

- **Clip 係數調得不對。** $\epsilon = 0.2$ 是事實上的標準。降到 `0.1` 會讓更新太膽怯；`0.3+` 則容易不穩定。
- **Epochs 太多。** $K > 20$ 常會造成不穩定，因為 policy 漂離 $\pi_{\text{old}}$ 太遠。為 epochs 設上限，尤其是大型網路。
- **沒有做 reward 正規化。** 大的 reward 尺度會吃掉 clip 的範圍。計算 advantages 之前先正規化 rewards（滾動標準差）。
- **忘了 advantage 正規化。** 對每個 batch 做零均值/單位標準差的正規化是標準做法。跳過它會讓 PPO 在多數 benchmark 上壞掉。
- **Learning rate 沒有衰減。** PPO 受益於線性衰減到零的 LR。固定 LR 通常比較差。
- **Importance ratio 的數學錯誤。** 為了數值穩定，一律用 `exp(log_new - log_old)`，不要用 `new / old`。
- **Gradient 的正負號錯了。** 最大化 surrogate = *最小化* $-L^{\text{CLIP}}$。正負號弄反是最常見的 PPO bug。

## 實際使用 (Use It)

PPO 是 2026 年在出人意料多的領域中預設的 RL 演算法：

| 使用情境 | PPO 變體 |
|----------|----------|
| MuJoCo / 機器人控制 | 搭配 Gaussian policy、GAE(0.95) 的 PPO |
| Atari / 離散遊戲 | 搭配 categorical policy、滾動 128 步 rollouts 的 PPO |
| LLM 的 RLHF | 對 reference model 帶 KL 懲罰、在回應結尾由 RM 給 reward 的 PPO |
| 大規模遊戲 agent | IMPALA + PPO (AlphaStar, OpenAI Five) |
| 推理型 LLM | GRPO（Lesson 12）：沒有 critic 的 PPO 變體 |
| 只有偏好的資料 | DPO：把 PPO+KL 壓縮成閉合形式，不需 online 取樣 |

PPO 的 *loss 形狀*，也就是 clipped surrogate + value + entropy，是 DPO、GRPO，以及幾乎每條 RLHF pipeline 的骨架。

## 交付 (Ship It)

存成 `outputs/skill-ppo-trainer.md`：

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## 練習 (Exercises)

1. **簡單。** 在 4×4 GridWorld 上以 $\epsilon=0.2, K=4$ 執行 PPO。在相同的 env 步數下，和 A2C（每個 rollout 一個 epoch）比較樣本效率。
2. **中等。** 掃描 $K \in \lbrace 1, 4, 10, 30 \rbrace$。畫出 return 對 env 步數，並追蹤每次更新的平均 KL。在這個任務上，$K$ 多大時 KL 會爆炸？
3. **困難。** 把 clipped surrogate 換成自適應 KL 懲罰（若 $\mathrm{KL} > 2 \cdot \text{target}$ 就把 $\beta$ 加倍，若 $\mathrm{KL} < \text{target}/2$ 就減半）。比較最終 return、穩定性，以及不需 clip 的程度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Importance ratio | 「r_t(θ)」 | $\pi_\theta(a \mid s) / \pi_{\text{old}}(a \mid s)$；與收集資料的 policy 之間的偏離。 |
| Clipped surrogate | 「PPO 的主要技巧」 | $\min(r \cdot A, \mathop{\mathrm{clip}}(r, 1-\epsilon, 1+\epsilon) \cdot A)$；在有利的一側超過 clip 後 gradient 變平。 |
| Trust region | 「TRPO / PPO 的用意」 | 限制每次更新的 KL，以保證單調改進。 |
| KL penalty | 「軟性的 trust region」 | 另一種 PPO：$L - \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{old}})$。自適應 $\beta$。 |
| Clip fraction | 「clipping 多常被觸發」 | 診斷指標，應為 0.1-0.3；超出範圍代表調得不對。 |
| Multi-epoch training | 「資料重複使用」 | 每個 rollout 做 K 個 epochs；用變異數的代價換取樣本效率。 |
| On-policy-ish | 「大致上 on-policy」 | PPO 名義上是 on-policy，但 K>1 的 epochs 能安全地使用略微 off-policy 的資料。 |
| PPO-KL | 「另一個 PPO」 | KL 懲罰變體；用於 RLHF，那裡對 reference 的 KL 本來就是限制。 |

## 延伸閱讀 (Further Reading)

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — 原始論文。
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO，PPO 的前身。
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — 對每個 PPO hyperparameter 都做了 ablation。
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT；RLHF 中 PPO 的做法。
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — 搭配 PyTorch 的清晰現代講解。
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — 許多論文使用的單檔 PPO 參考實作。
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — 在語言模型上跑 PPO 的正式做法；和 Lesson 09（RLHF）一起讀。
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — 「37 個程式碼層級最佳化」論文；哪些 PPO 技巧是真正承重的，哪些只是民間傳說。
