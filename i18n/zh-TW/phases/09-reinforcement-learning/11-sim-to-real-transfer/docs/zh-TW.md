# Sim-to-Real Transfer

> 在模擬器裡訓練、卻在硬體上失敗的 policy，就是背下了模擬器的 policy。Domain randomization、domain adaptation 與 system identification 是讓學出來的控制器跨越 reality gap 的三項工具。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 9 · 08 (PPO), Phase 2 · 10 (Bias/Variance)
**Time:** ~45 分鐘

## 問題 (The Problem)

訓練真實的機器人既慢、危險又昂貴。雙足機器人要花數百萬個訓練 episodes 才學會走路；真實的雙足機器人只要跌倒一次就可能損壞硬體。模擬給你無限次的重置、可決定性重現、平行 environments，而且不會造成實體損壞。

但模擬器是錯的。軸承的摩擦比 MuJoCo 建模的更大。相機有模擬器沒有納入的鏡頭畸變。馬達有延遲、背隙與飽和，99% 的模擬模型都略過了。風、灰塵與多變的光線，會破壞在乾淨渲染上訓練的 policy。**Reality gap**，也就是模擬分布與真實分布之間的系統性差異，是機器人領域實際部署 RL 的核心問題。

你需要一個 *對 sim-to-real 分布位移具有穩健性* 的 policy。歷史上有三種方法：隨機化模擬器（domain randomization）、用少量真實資料調整 policy（domain adaptation / fine-tuning），或辨識真實系統的參數並加以匹配（system identification）。2026 年主流的做法是把三者與大規模平行模擬（Isaac Sim、Isaac Lab、GPU 上的 Mujoco MJX）結合。

## 核心概念 (The Concept)

![Three sim-to-real regimes: domain randomization, adaptation, system identification](../assets/sim-to-real.svg)

**Domain Randomization (DR)。** Tobin et al. 2017、Peng et al. 2018。訓練期間，隨機化每個在真實機器人上可能不同的模擬參數：質量、摩擦係數、馬達 PD 增益、感測器雜訊、相機位置、光線、紋理、接觸模型。Policy 學到一個關於「今天身處哪一個模擬」的條件分布，並在整個範圍內泛化。如果真實機器人落在訓練的範圍內，policy 就能運作。

- **優點：** 不需要真實資料。一種做法，適用許多機器人。
- **缺點：** 過度隨機化的訓練會產生「通用」但過於謹慎的 policy。太多雜訊 ≈ 太多 regularization。

**System Identification (SI)。** 在訓練之前，把模擬器的參數擬合到真實世界的資料。如果你能在真實機器人上量到手臂關節的摩擦，就把它填進模擬。然後訓練一個預期這些數值的 policy。需要能接觸到真實系統，但能直接縮小 reality gap。

- **優點：** 精確、低雜訊的訓練目標。
- **缺點：** Policy 看不到殘留的模型誤差；小的未辨識效應（例如馬達死區）仍會讓部署失敗。

**Domain Adaptation。** 在模擬中訓練，再用少量真實資料 fine-tune。有兩種風格：

- **Real2Sim2Real：** 用真實 rollouts 學一個殘差模擬器 $f(s, a, z) - f_{\text{sim}}(s, a)$，在修正後的模擬中訓練。不需要太多真實資料就能縮小差距。
- **Observation adaptation：** 訓練一個 policy，透過學出來的特徵擷取器（例如 GAN pixel-to-pixel）把真實 obs 對應到類似模擬的 obs。控制器留在模擬中。

**Privileged learning / teacher-student。** Miki et al. 2022（ANYmal 四足機器人）。在模擬中訓練一個可以取得特權資訊（真實的摩擦、地形高度、IMU 漂移）的 *teacher*。蒸餾出一個只看得到真實感測器觀測的 *student*。Student 學會從歷史推斷特權特徵，對各種物理參數都穩健。

**大規模平行模擬。** 2024–2026。Isaac Lab、Mujoco MJX、Brax 都能在單一 GPU 上執行數千個平行機器人。用 4,096 個平行人形機器人跑 PPO，幾小時就能收集到數年的經驗。隨著訓練分布變寬，「reality gap」縮小；當這 4,096 個 envs 各有不同的隨機化參數時，DR 幾乎是免費的。

**2026 年的真實世界做法（以四足行走為例）：**

1. 大規模平行模擬，對重力、摩擦、馬達增益、負載做 domain randomization。
2. 用特權資訊（地形圖、身體速度真值）訓練 teacher policy。
3. 只用本體感覺（腿部關節編碼器）從 teacher 蒸餾出 student policy。
4. 可選：在真實 IMU 上透過 autoencoder 做 observation adaptation。
5. 部署。在 10 個以上的環境中 zero-shot。如果失敗，就用受安全限制的 PPO 做幾分鐘的真實世界 fine-tune。

```figure
f3-reality-gap
```

## 動手做 (Build It)

這堂課的程式碼是在帶有 *雜訊* transitions 的 GridWorld 上，對 domain randomization 的小型示範。我們訓練一個 policy，讓它在「sim」中經歷隨機化的 slip 機率，並在「real」中以訓練期間從未見過的 slip 程度來評估。這個形狀直接對應 MuJoCo 到硬體的遷移。

### 步驟 1：參數化的 sim

```python
def step(state, action, slip):
    if rng.random() < slip:
        action = random_perpendicular(action)
    ...
```

`slip` 是模擬器公開的參數。在真實機器人上，它可以是摩擦、質量、馬達增益，任何在 sim 與 real 之間會偏移的東西。

### 步驟 2：用 DR 訓練

在每個 episode 開始時，取樣 `slip ~ Uniform[0.0, 0.4]`。訓練 PPO / Q-learning / 任何方法。做很多個 episodes。

### 步驟 3：在「real」slip 上 zero-shot 評估

在 `slip ∈ {0.0, 0.1, 0.2, 0.3, 0.5, 0.7}` 上評估。前四個在訓練的支撐範圍內；`0.5` 與 `0.7` 在範圍之外。用 DR 訓練的 policy 在範圍內應該保持接近最佳，在範圍外則平緩地退化。用固定 slip 訓練的 policy 在訓練 slip 之外會很脆弱。

### 步驟 4：與窄範圍訓練比較

只用 `slip = 0.0` 訓練第二個 policy。在同樣的 `slip` 掃描上評估。你應該會看到，只要真實 slip > 0，表現就災難性地下滑。

## 常見陷阱 (Pitfalls)

- **隨機化太多。** 用 `slip ∈ [0, 0.9]` 訓練，你的 policy 會變得極度趨避風險，永遠不嘗試最佳路徑。要匹配 *預期的* 真實世界分布，而不是「什麼都可能發生」。
- **隨機化太少。** 只用一小片範圍訓練，policy 完全無法泛化。使用自適應 curriculum（Automatic Domain Randomization），隨著 policy 進步而拓寬分布。
- **參數空間辨識錯誤。** 隨機化了錯的東西（真實差距是馬達延遲，你卻隨機化相機色調），DR 就沒有幫助。先分析真實機器人。
- **特權資訊洩漏。** Teacher 如果在 actions 中使用全域 state，而不只是觀測，可能產生 student 追不上的結果。確保 teacher 的 policy 在給定觀測歷史的情況下，student 是能實現的。
- **Sim-to-sim 遷移失敗。** 如果你的 policy 對更困難的 sim 變體不穩健，它對真實世界也不會穩健。部署之前，一律先在保留的 sim 變體上測試。
- **沒有真實世界的安全範圍。** 在 sim 中有效、並「在 real 中有效」的 policy，如果沒有低階的安全防護，仍可能損壞硬體。在非學習式控制器中加入速率限制、力矩限制、關節限制。

## 實際使用 (Use It)

2026 年的 sim-to-real 技術堆疊：

| 領域 | 技術堆疊 |
|------|----------|
| 腿式運動（ANYmal、Spot、人形） | Isaac Lab + DR + 特權 teacher / student |
| 操作（靈巧手、取放） | Isaac Lab + DR + 視覺用的 DR-GAN |
| 自動駕駛 | CARLA / NVIDIA DRIVE Sim + DR + 真實 fine-tune |
| 無人機競速 | RotorS / Flightmare + DR + online 調整 |
| 手指/手內操作 | OpenAI Dactyl（前所未有規模的 DR） |
| 工業手臂 | MuJoCo-Warp + SI + 少量真實 fine-tune |

在各種規模的控制上，工作流程是一致的：盡你所能擬合 sim，隨機化你擬合不了的部分，訓練巨大的 policies，蒸餾，再搭配安全防護部署。

## 交付 (Ship It)

存成 `outputs/skill-sim2real-planner.md`：

```markdown
---
name: sim2real-planner
description: Plan a sim-to-real transfer pipeline for a given robot + task, covering DR, SI, and safety.
version: 1.0.0
phase: 9
lesson: 11
tags: [rl, sim2real, robotics, domain-randomization]
---

Given a robot platform, a task, and access to real hardware time, output:

1. Reality gap inventory. Suspected sources ranked by expected impact (contact, sensing, actuation delay, vision).
2. DR parameters. Exact list, ranges, distribution. Justify each range against real measurements.
3. SI steps. Which parameters to measure; measurement method.
4. Teacher/student split. What privileged info the teacher uses; what obs the student uses.
5. Safety envelope. Low-level limits, emergency stops, backup controller.

Refuse to deploy without (a) a zero-shot sim-variant test, (b) a safety shield, (c) a rollback plan. Flag any DR range wider than 3× measured real variability as likely over-randomized.
```

## 練習 (Exercises)

1. **簡單。** 在固定 slip 的 GridWorld（slip=0.0）上訓練 Q-learning agent。在 $\text{slip} \in \lbrace 0.0, 0.1, 0.3, 0.5 \rbrace$ 上評估。畫出 return 對 slip 的圖。
2. **中等。** 訓練一個取樣 `slip ~ Uniform[0, 0.3]` 的 DR Q-learning agent。在同樣的掃描上評估。DR 在 slip=0.5（分布外）時帶來多少好處？
3. **困難。** 實作一個 curriculum：從 slip=0.0 開始，每當 policy 達到最佳的 90% 就拓寬 DR 範圍。測量 zero-shot 達到 slip=0.3 所需的總 environment 步數，並與固定 DR 的 baseline 比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| Reality gap | 「Sim 與 real 的差異」 | 訓練與部署時物理/感測之間的分布位移。 |
| Domain randomization (DR) | 「在隨機的 sim 之間訓練」 | 訓練期間隨機化 sim 參數，讓 policy 能泛化。 |
| System identification (SI) | 「量測真實並擬合 sim」 | 估計真實的物理參數；把 sim 設成與之匹配。 |
| Domain adaptation | 「用真實資料 fine-tune」 | sim 訓練之後，用少量真實世界資料 fine-tune；可調整 obs 或動力學。 |
| Privileged info | 「Teacher 的真值」 | 只有 sim 才有的資訊；student 必須從 obs 歷史推斷。 |
| Teacher/student | 「把特權 -> 可觀測 蒸餾」 | Teacher 用捷徑訓練；student 學會在沒有捷徑的情況下模仿。 |
| ADR | 「Automatic Domain Randomization」 | 隨著 policy 進步而拓寬 DR 範圍的 curriculum。 |
| Real2Sim | 「用真實資料縮小差距」 | 學習一個殘差，讓 sim 模仿真實 rollouts。 |

## 延伸閱讀 (Further Reading)

- [Tobin et al. (2017). Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World](https://arxiv.org/abs/1703.06907) — 最早的 DR 論文（機器人視覺）。
- [Peng et al. (2018). Sim-to-Real Transfer of Robotic Control with Dynamics Randomization](https://arxiv.org/abs/1710.06537) — 動力學的 DR，四足運動。
- [OpenAI et al. (2019). Solving Rubik's Cube with a Robot Hand](https://arxiv.org/abs/1910.07113) — Dactyl，大規模的 ADR。
- [Miki et al. (2022). Learning robust perceptive locomotion for quadrupedal robots in the wild](https://www.science.org/doi/10.1126/scirobotics.abk2822) — ANYmal 的 teacher-student。
- [Makoviychuk et al. (2021). Isaac Gym: High Performance GPU Based Physics Simulation for Robot Learning](https://arxiv.org/abs/2108.10470) — 驅動 2025–2026 年部署的大規模平行 sim。
- [Akkaya et al. (2019). Automatic Domain Randomization](https://arxiv.org/abs/1910.07113) — ADR curriculum 方法。
- [Sutton & Barto (2018). Ch. 8 — Planning and Learning with Tabular Methods](http://incompleteideas.net/book/RLbook2020.pdf) — Dyna 的框架（用 model 做 planning + rollouts），是現代 sim-to-real pipelines 的基礎。
- [Zhao, Queralta & Westerlund (2020). Sim-to-Real Transfer in Deep Reinforcement Learning for Robotics: a Survey](https://arxiv.org/abs/2009.13303) — 附 benchmark 結果的 sim-to-real 方法分類。
