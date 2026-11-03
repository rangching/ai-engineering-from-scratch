# 遊戲中的 RL：AlphaZero、MuZero 與 LLM 推理時代 (RL for Games — AlphaZero, MuZero, and the LLM-Reasoning Era)

> 1992：TD-Gammon 純靠 TD 擊敗了西洋雙陸棋的人類冠軍。2016：AlphaGo 擊敗李世乭。2017：AlphaZero 從零開始稱霸西洋棋、將棋與圍棋。2025：DeepSeek-R1 證明了同樣的做法，把 PPO 換成 GRPO，在推理上也行得通。遊戲是推動本階段每一次突破的 benchmark。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 分鐘

## 問題 (The Problem)

遊戲擁有 RL 想要的一切。乾淨的 reward（勝／負）。無限的 episodes（self-play 重置）。完美的模擬（遊戲 *本身就是* 模擬器）。離散或小型連續的 action 空間。強迫產生對抗穩健性的 multi-agent 結構。

而且遊戲是每一次重大 RL 突破的測試場。TD-Gammon（西洋雙陸棋，1992）。Atari-DQN（2013）。AlphaGo（2016）。AlphaZero（2017）。OpenAI Five（Dota 2，2019）。AlphaStar（StarCraft II，2019）。MuZero（學出來的 model，2019）。AlphaTensor（矩陣乘法，2022）。AlphaDev（排序演算法，2023）。DeepSeek-R1（數學推理，2025），這是遊戲 RL 技術能用在文字上的最新證明。

這個 capstone 用一個統一的視角，概覽三個里程碑式的架構：AlphaZero、MuZero 與 GRPO：**self-play + search + policy improvement**。每一個都推廣了前一個；尤其 GRPO 就是把 AlphaZero 的做法套用到 LLM 推理上，以 tokens 為 actions，以數學驗證作為勝利訊號。

## 核心概念 (The Concept)

![AlphaZero ↔ MuZero ↔ GRPO: same loop, different environments](../assets/rl-games.svg)

**統一的迴圈。**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero（2017）。** Silver et al. 給定一個規則已知的遊戲（西洋棋、將棋、圍棋）：

- Policy-value network：一座塔 $f_\theta(s) \to (p, v)$。$p$ 是合法走法的 prior。$v$ 是期望的遊戲結果。
- Monte Carlo Tree Search (MCTS)：每一步棋，展開一棵可能後續的樹。使用 $(p, v)$ 作為 prior + bootstrap。以 UCB（PUCT）選擇節點：$a^{\ast} = \arg\max_a Q(s, a) + c \cdot p(a \mid s) \cdot \sqrt{N(s)} / (1 + N(s, a))$。
- Self-play：agent 對 agent 下棋。在第 $t$ 步，MCTS 的造訪分布 $\pi_t$ 成為 policy 的訓練目標。
- Loss：$L = (v - z)^2 - \pi \cdot \log p + c \cdot \lVert \theta \rVert^2$。$z$ 是遊戲結果（+1 / 0 / -1）。

零人類知識。零手工啟發式。一套做法，各自在幾千萬場 self-play 對局後，精通了西洋棋、將棋與圍棋。

**MuZero（2019）。** Schrittwieser et al. 移除了規則必須已知的要求。

- 不使用固定的 environment，而是學一個 *latent dynamics model* $(h, g, f)$：
  - $h(s)$：把觀測編碼成 latent state。
  - $g(s_{\text{latent}}, a)$：預測下一個 latent state + reward。
  - $f(s_{\text{latent}})$：預測 policy prior + value。
- MCTS 在 *學出來的 latent 空間* 中執行。相同的搜尋，相同的訓練迴圈。
- 適用於圍棋、西洋棋、將棋 *以及* Atari：一個演算法，不需要規則知識。

**Stochastic MuZero（2022）。** 加入 stochastic dynamics 與 chance nodes；延伸到西洋雙陸棋這類遊戲。

**Muesli、Gumbel MuZero（2022-2024）。** 改進樣本效率與 deterministic 搜尋。

**GRPO（2024-2025）。** DeepSeek-R1 的做法。相同的 AlphaZero 形狀的迴圈，套用到語言模型推理：

- 「遊戲」：回答一道數學 / 程式 / 推理題。「贏」= verifier（測試案例通過、數值答案吻合）回傳 1。
- Policy：LLM。Actions：tokens。State：prompt + 目前為止的回應。
- 沒有 critic（PPO 風格的 $V_\phi$）。取而代之的是，對每個 prompt，從 policy 取樣 $G$ 個 completions。為每一個計算 reward。使用 **group-relative advantage** $A_i = (r_i - \text{mean}_r) / \text{std}_r$ 作為 REINFORCE 風格更新的訊號。
- 對 reference policy 的 KL 懲罰，以防止漂移（像 RLHF 一樣）。
- 完整的 loss：

  $L_{\text{GRPO}}(\theta) = -\mathbb E_{q, \lbrace o_i \rbrace} \left[ \frac{1}{G} \sum_i A_i \cdot \log \pi_\theta(o_i \mid q) \right] + \beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$

沒有學習式的 reward model，沒有 critic，沒有 MCTS。由規則式 verifier 提供 reward，group-relative baseline 取代 critic，搜尋步驟被省略。在推理 benchmark 上，以少很多的運算量達到或超越 PPO-RLHF 的品質。

**完整的 R1 做法。** DeepSeek-R1（DeepSeek 2025）在一篇論文裡有兩個模型：

- **R1-Zero。** 從 DeepSeek-V3 base model 開始。不做 SFT。直接套用 GRPO，搭配兩個 reward 成分：*accuracy reward*（規則式：最終答案是否解析為正確數字／程式是否通過 unit tests）與 *format reward*（completion 是否用 `<think>…</think>` 標籤包住它的 chain-of-thought）。經過數千步，平均回應長度從 ~100 成長到 ~10,000 tokens，數學 benchmark 分數攀升到接近 o1-preview 的水準。模型從零學會推理。缺點：它的 chains of thought 常常難以閱讀、混用語言，也缺乏文風上的打磨。
- **R1。** 用四階段 pipeline 修正 R1-Zero 的可讀性問題：
  1. **Cold-start SFT。** 收集幾千筆格式乾淨的長 CoT 示範。用它們對 base model 做 supervised-finetune。這提供了一個可讀的起點。
  2. **以推理為導向的 GRPO。** 套用 GRPO，搭配 accuracy+format rewards，再加上 *language-consistency* reward 以避免語碼轉換。
  3. **Rejection sampling + SFT 第二輪。** 從 RL checkpoint 取樣 ~600K 條推理軌跡，只保留最終答案正確且 CoT 可讀的，再與 ~200K 筆非推理的 SFT 範例（寫作、QA、自我認知）結合。再次 fine-tune base。
  4. **全面性的 GRPO。** 再做一輪 RL，同時涵蓋推理（規則式 rewards）與一般對齊（基於有幫助／無害偏好的 rewards）。

結果在 AIME 與 MATH-500 上以開放權重追平 o1，而且夠小，可以蒸餾。同一篇論文也釋出六個蒸餾的 dense 模型（Qwen-1.5B 到 Llama-70B），是透過對 R1 的推理軌跡做 SFT 得到的，student 端沒有 RL。對強大的 RL teacher 做蒸餾，在 student 的規模上一貫勝過從零開始做 RL。

**為什麼推理用 GRPO 而不是 PPO。** DeepSeekMath 論文（2024 年 2 月）中有三個理由：(1) 不用訓練 value network，記憶體減半；(2) group baseline 自然地處理推理任務產生的、稀疏的軌跡結尾 reward；(3) 每個 prompt 各自的正規化，讓難度差異極大的問題之間，advantages 也能相互比較，這是 PPO 單一 critic 做不到的。

**無搜尋 vs 以搜尋為基礎。** 遊戲已經分岔：

- *長視野的完全資訊遊戲*（圍棋、西洋棋）：仍以搜尋為基礎。AlphaZero / MuZero 佔主導。
- *LLM 推理*：生產環境中還沒有 MCTS；對完整 rollouts 做 GRPO，推論運算用 best-of-N。Process reward models（PRMs）暗示著步驟層級的搜尋可能會被重新加回來。

```figure
f3-selfplay-ladder
```

## 動手做 (Build It)

`code/main.py` 中的程式碼實作了 **縮小版的 GRPO**：一個有多組樣本群的 bandit。演算法和在 LLM 上相同；只是 policy 與 environment 比較簡單。它教的是 *loss* 與 *group-relative advantage*，這是 DeepSeekMath 在 2024 年的創新。

### 步驟 1：一個很小的 verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

在真正的 GRPO 中，verifier 會執行 unit tests 或檢查數學上是否相等。

### 步驟 2：policy：每個 prompt 對 K 個答案 tokens 做 softmax

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

等同於以 prompt 為條件的 LLM 最後一層輸出。

### 步驟 3：group 取樣與 group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage 是 2024 年 DeepSeek 的技巧。不需要 critic。「baseline」是 group 的平均，正規化使用 group 的標準差。

### 步驟 4：與 REINFORCE baseline 比較（無 value）

相同的設定，相同的運算量，用一般的 REINFORCE。GRPO 收斂得更快、更穩定。

### 步驟 5：觀察 entropy 與 KL

和 RLHF 相同的診斷：對 reference 的平均 KL、policy entropy、reward 隨時間的變化。一旦這些穩定下來，訓練就完成了。

## 常見陷阱 (Pitfalls)

- **透過 verifier 鑽漏洞的 reward hacking。** GRPO 繼承了 RLHF 的風險：如果 verifier 是錯的或可被利用，LLM 就會找到漏洞。穩健的 verifiers（多個測試案例、形式化證明）很重要。
- **Group 太小。** Group-mean baseline 的變異數約為 $1/G$（標準誤差約為 $1/\sqrt{G}$）。低於 $G = 4$，advantage 訊號就有雜訊；標準選擇是 $G = 8$ 到 64。
- **長度 bias。** 不同長度的 LLM completions 有不同的 log-probabilities。以 token 數正規化，或使用序列層級的 log-prob，或截斷到最大長度。
- **純 self-play 循環。** AlphaZero 風格的訓練在 general-sum 遊戲中可能卡在支配循環。用多樣的對手池（league play，Lesson 10）來緩解。
- **Search 與 policy 不匹配。** AlphaZero 訓練 policy 去模仿搜尋的輸出。如果 policy net 太小，無法表示搜尋的分布，訓練就會停滯。
- **運算量下限。** MuZero / AlphaZero 需要大量運算。單一次 ablation 往往就要數百 GPU 小時。有縮小版的示範可供學習（例如 Connect Four 上的 AlphaZero）。
- **Verifier 的覆蓋範圍。** 對有 bug 的解答也能通過的 unit tests，會強化那個 bug。設計能抓到邊界情況的 verifiers。

## 實際使用 (Use It)

2026 年遊戲 RL 的版圖，依領域：

| 領域 | 主流方法 |
|------|----------|
| 雙人零和棋盤遊戲（圍棋、西洋棋、將棋） | AlphaZero / MuZero / KataGo |
| 不完全資訊的牌類遊戲（撲克） | CFR + deep learning（DeepStack、Libratus、Pluribus） |
| Atari / 像素遊戲 | Muesli / MuZero / IMPALA-PPO |
| 大型多人策略遊戲（Dota、StarCraft） | PPO + self-play + league（OpenAI Five、AlphaStar） |
| LLM 數學/程式推理 | GRPO（DeepSeek-R1、Qwen-RL、開放的重現實作） |
| LLM 對齊 | DPO / RLHF-PPO（不用 GRPO；verifier 是偏好而非可驗證的） |
| 機器人 | PPO + DR（不是遊戲 RL，但使用相同的 policy-gradient 工具） |
| 組合問題 | AlphaZero 變體（AlphaTensor、AlphaDev） |

這個 *做法*，也就是 self-play、搜尋強化的改進、policy 蒸餾，橫跨文字、像素與實體控制。GRPO 是最年輕的實例；還有更多即將出現。

## 交付 (Ship It)

存成 `outputs/skill-game-rl-designer.md`：

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## 練習 (Exercises)

1. **簡單。** 實作 `code/main.py` 中的 GRPO bandit。在 2 個 prompts × 各 4 個答案 tokens 上訓練。以 $G=8$ 在 < 1,000 次更新內收斂。
2. **中等。** 換成 PPO（clipped）與一般的 REINFORCE。在同一個 bandit 上，與 GRPO 比較樣本效率與 reward 變異數。
3. **困難。** 延伸成長度為 2 的「推理鏈」：agent 輸出兩個 tokens，verifier 對這一對給 reward。測量 GRPO 如何處理跨兩步序列的 credit assignment。（提示：對每個 *完整序列* 計算 group advantage，再傳播到兩個 token 位置。）

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| MCTS | 「搭配學習網路的樹搜尋」 | Monte Carlo Tree Search；以學出來的 $(p, v)$ priors 做 UCB1/PUCT 選擇。 |
| AlphaZero | 「Self-play + MCTS」 | 訓練 policy-value net 去匹配 MCTS 造訪次數與遊戲結果。 |
| MuZero | 「學習 model 的 AlphaZero」 | 相同的迴圈，但透過學出來的 dynamics 在 latent 空間中進行。 |
| GRPO | 「沒有 critic 的 PPO」 | Group Relative Policy Optimization；帶 group-mean baseline + KL 的 REINFORCE。 |
| PUCT | 「AlphaZero 的 UCB」 | $Q + c \cdot p \cdot \sqrt{N} / (1 + N_a)$，在 value 估計與 prior 之間取得平衡。 |
| Self-play | 「Agent 對上過去的自己」 | 零和遊戲的標準做法；對稱的訓練訊號。 |
| League play | 「以族群為基礎的 self-play」 | 過去 + 目前 + exploiters 都被取樣當作對手。 |
| Verifier reward | 「可驗證的 RL」 | Reward 來自決定性的檢查器（測試通過、答案吻合）。 |
| Process reward | 「PRM」 | 對每個推理步驟打分數，而不只是最終答案。 |

## 延伸閱讀 (Further Reading)

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270)。
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404)。
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4)。
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z)。
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — 提出 GRPO 與 group-relative baseline 的論文。
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — 完整的四階段 R1 做法，加上 R1-Zero 的 ablation。
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — 大規模的 CFR + deep learning。
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — 一切的起點。
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — 搭配自訂 reward functions 套用 GRPO 的正式參考。
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — 在多種規模上開放重現 R1 的做法。
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — 教科書對 self-play、搜尋與「設計的 reward」的框架，R1 在 LLM 規模上實現了它們。
