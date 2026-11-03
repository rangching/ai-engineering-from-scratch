# Reward Modeling 與 RLHF (Reward Modeling & RLHF)

> 人類無法為「好的助理回應」寫出 reward function，但可以比較兩個回應並挑出比較好的那個。用這些比較結果擬合一個 reward model，再針對它對語言模型做 RL。Christiano 2017。InstructGPT 2022。這個做法把 GPT-3 變成了 ChatGPT。到 2026 年，它大多被 DPO 取代，但這個思考模型仍然適用。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 分鐘

## 問題 (The Problem)

你用 next-token-prediction 目標訓練了一個語言模型。它寫得出合乎文法的英文。它也會說謊、囉嗦，而且該拒絕的時候不拒絕。你無法靠更多 pretraining 解決這個問題，網路文字正是問題所在，而不是解藥。

你想要一個 *純量 reward*，用來表示「對於指令 X，回應 A 比回應 B 好」。手寫這個 reward function 是不可能的。「有幫助」不是 tokens 上的閉合形式運算式。但人類可以比較兩個輸出並標出偏好。這在大規模下收集起來很便宜。

RLHF（Christiano et al. 2017；Ouyang et al. 2022）把偏好轉成 reward model，再透過 PPO 針對該 reward 最佳化 LM。分三步：SFT → RM → PPO。這個做法讓 ChatGPT、Claude、Gemini，以及 2023–2025 年間每一個對齊過的 LLM 得以問世。

2026 年，PPO 這一步大多被 DPO（Phase 10 · 08）取代，因為它更便宜，對對齊調整來說效果也幾乎一樣好。但 *reward model* 這一塊仍然是每個 Best-of-N sampler、每條 RL-from-verifiable-rewards pipeline，以及每個使用 process reward model 的推理模型的底層基礎。理解 RLHF，你就理解了整個對齊堆疊。

## 核心概念 (The Concept)

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**階段 1：Supervised Fine-Tuning (SFT)。** 從預訓練的 base model 開始。用人類寫的目標行為示範（遵循指令的回應、有幫助的回覆等）做 fine-tune。結果：模型 $\pi_{\text{SFT}}$ *偏向好的行為*，但 action 空間仍然沒有邊界。

**階段 2：Reward Model 訓練。**

- 針對 prompts $x$ 收集成對的回應 $(y_+, y_-)$，由人類標註「y_+ 優於 y_-」。
- 訓練 reward model $R_\phi(x, y)$，讓它給 $y_+$ 較高的分數。
- Loss：**Bradley-Terry pairwise logistic**：

  $L(\phi) = -\mathbb{E}[ \log \sigma(R_\phi(x, y_+) - R_\phi(x, y_-)) ]$

  $\sigma$ 是 sigmoid。reward 的差值對應到偏好的 log-odds。BT 自 1952 年（Bradley-Terry）以來就是標準，也是現代 RLHF 的主流選擇。

- $R_\phi$ 通常從 SFT model 初始化，頂端接一個純量 head。同樣的 transformer 主幹；用單一 linear layer 輸出 reward。

**階段 3：對 RM 做帶 KL 懲罰的 PPO。**

- 從 $\pi_{\text{SFT}}$ 初始化可訓練的 policy $\pi_\theta$。保留一個凍結的 *reference* $\pi_{\text{ref}} = \pi_{\text{SFT}}$。
- 回應 $y$ 結尾的 reward：

  $r_{\text{total}}(x, y) = R_\phi(x, y) - \beta \cdot \mathrm{KL}(\pi_\theta(\cdot \mid x) \Vert \pi_{\text{ref}}(\cdot \mid x))$

  KL 懲罰防止 $\pi_\theta$ 任意偏離 $\pi_{\text{SFT}}$：它是 *regularizer*，不是硬性的 trust region。$\beta$ 通常是 0.01-0.05。
- 用這個 reward 執行 PPO（Lesson 08）。Advantages 是在 token 層級的軌跡上計算，但 RM 只對完整回應打分數。

**為什麼要 KL？** 沒有它，PPO 會很樂意找到 reward-hacking 的策略：RM 只在分布內的 completions 上訓練過。分布外的回應可能得到比任何人類寫的回應都高的分數。KL 讓 $\pi_\theta$ 留在 RM 訓練過的流形附近。它是 RLHF 中最重要的單一旋鈕。

**2026 年的現況：**

- **DPO**（Rafailov 2023）：閉合形式的代數把階段 2+3 壓縮成對偏好資料的單一 supervised loss。沒有 RM，沒有 PPO。在對齊 benchmark 上品質相同，運算量卻少很多。見 Phase 10 · 08。
- **GRPO**（DeepSeek 2024–2025）：用 group-relative baseline 取代 critic 的 PPO，reward 來自 *verifier*（程式能執行 / 數學答案吻合），而不是人類訓練的 RM。在推理模型中佔主導。見 Phase 9 · 12。
- **Process reward models (PRMs)：** 對部分解答（每個推理步驟）打分數，用於推理的 RLHF 與 GRPO 變體。
- **Constitutional AI / RLAIF：** 用已對齊的 LLM 產生偏好，取代人類。可擴大偏好資料的預算。

```figure
reward-model
```

## 動手做 (Build It)

這堂課使用以字串表示的小型合成「prompts」與「responses」。RM 是對 bag-of-tokens 表示的線性評分器。沒有真正的 LLM，重要的是 pipeline 的 *形狀*，而不是規模。見 `code/main.py`。

### 步驟 1：合成偏好資料

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

在真正的 RLHF 中，這會換成人類標註者。形狀 `(prompt, preferred_response, rejected_response)` 完全相同。

### 步驟 2：Bradley-Terry reward model

線性分數：`R(x, y) = w · bag(y)`。訓練以最小化 BT pairwise log-loss：

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

幾百次更新之後，`w` 會給好的字詞 tokens 正權重，給壞的負權重。

### 步驟 3：在 RM 之上的類 PPO policy

我們的 toy policy 從詞彙表中產生單一 token。我們在 RM 下對 token 打分數，計算 $\log \pi_\theta(\text{token} \mid \text{prompt})$，加上對 reference 的 KL 懲罰，再套用 clipped PPO surrogate。

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### 步驟 4：監控 KL

每次更新都追蹤平均 $\mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$。如果它悄悄超過 `~5-10`（每個回應的 KL，以 nats 為單位，對 tokens 加總，這是 TRL 記錄的單位），代表 policy 已經離 $\pi_{\text{SFT}}$ 很遠：不是 $\beta$ 太低（提高它把 policy 拉回來），就是 reward hacking 開始了。這是真正的 RLHF 中最重要的診斷指標。

### 步驟 5：用 TRL 的正式做法

一旦你理解了 toy pipeline，以下就是真正的函式庫使用者會寫的同一個迴圈。Hugging Face 的 [TRL](https://huggingface.co/docs/trl) 是參考實作：階段 2 用 `RewardTrainer`，階段 3 用 `PPOTrainer`（內建對 reference 的 KL）。

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

函式庫替你做了三件事。`adap_kl_ctrl=True` 實作了自適應 β 排程：若觀察到的 KL 超過 `target_kl`，β 加倍；若低於一半，β 減半。Reference model 依慣例是凍結的，你絕不能不小心和 `policy` 共用參數。而 value head 與 policy 位於同一個主幹上（`AutoModelForCausalLMWithValueHead` 附加了一個純量 MLP head），這也是 TRL 把 `policy/kl` 與 `value/loss` 分開回報的原因。

## 常見陷阱 (Pitfalls)

- **過度最佳化 / reward hacking。** RM 並不完美；$\pi_\theta$ 會找到得分高但其實很糟的對抗性 completions。症狀：reward 無止境地攀升，而人類評估分數持平或下降。修正：提早停止、提高 $\beta$、擴大 RM 的訓練資料。
- **Length hacking。** 以有幫助的回應訓練的 RM，常會隱含地獎勵長度。Policy 學會灌水。補救：長度正規化的 reward，或搭配具長度感知 RM 的 RLAIF。
- **RM 太小。** RM 至少要和 policy 一樣大。很小的 RM 無法忠實地為 policy 的輸出打分數。
- **KL 調校。** β 太低 → 漂移與 reward hacking。β 太高 → policy 幾乎沒變。標準技巧是使用 *自適應* β，目標是每步固定的 KL。
- **偏好資料的雜訊。** 約 30% 的人類標註有雜訊或含糊不清。可以用一致性過濾後的資料訓練 RM，或對 BT 使用 temperature 來校準。
- **Off-policy 問題。** PPO 資料在第一個 epoch 之後會略微 off-policy。如 Lesson 08 所述，監控 clip fraction。

## 實際使用 (Use It)

2026 年的 RLHF 是分層的：

| 層 | 目標 | 方法 |
|----|------|------|
| 遵循指令、有幫助、無害 | Alignment | 偏好用 DPO（Phase 10 · 08），而非 RLHF-PPO。 |
| 推理正確性（數學、程式） | 能力 | 搭配 verifier reward 的 GRPO（Phase 9 · 12）。 |
| 長視野的多步驟任務 | Agentic | 搭配步驟層級 process reward models 的 PPO / GRPO。 |
| 安全 / 拒絕行為 | 安全 | 搭配獨立安全 RM 的 RLHF-PPO，或 Constitutional AI。 |
| 推論時的 Best-of-N | 快速對齊 | 在 decode 時使用 RM；不需要訓練 policy。 |
| Reward distillation | 推論運算量 | 在凍結的 LM 上訓練一個小型「reward head」。 |

RLHF 在 2022–2024 年是 *那個* 方法。到了 2026 年，正式的對齊 pipeline 以 DPO 優先，PPO 只用在重度依賴 RM 或安全至上的步驟。

## 交付 (Ship It)

存成 `outputs/skill-rlhf-architect.md`：

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## 練習 (Exercises)

1. **簡單。** 在 500 組合成偏好配對上訓練 `code/main.py` 的 Bradley-Terry reward model。在保留的 100 組上測量 pairwise 準確率。應該超過 90%。
2. **中等。** 以 $\beta \in \lbrace 0.0, 0.1, 1.0 \rbrace$ 執行 toy PPO-RLHF 迴圈。對每個 $\beta$，畫出 RM 分數對 KL-to-reference 隨更新次數的變化。哪一個會 reward-hack？
3. **困難。** 在同一份偏好資料上實作 DPO（閉合形式的偏好似然 loss），並在使用的運算量與最終達到的 RM 分數上，與 RLHF-PPO pipeline 比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------|----------|
| RLHF | 「Alignment RL」 | 三階段的 SFT + RM + PPO pipeline（Christiano 2017、Ouyang 2022）。 |
| Reward Model (RM) | 「評分網路」 | 透過 Bradley-Terry 對 pairwise 偏好擬合出來的純量函式。 |
| Bradley-Terry | 「Pairwise logistic loss」 | $P(y_+ \succ y_-) = \sigma(R(y_+) - R(y_-))$；標準的 RM 目標。 |
| KL penalty | 「留在 reference 附近」 | reward 中的 $\beta \cdot \mathrm{KL}(\pi_\theta \Vert \pi_{\text{ref}})$；防止 reward hacking 的 regularizer。 |
| Reward hacking | 「Goodhart 定律」 | Policy 利用 RM 的缺陷；症狀：reward 上升，人類評估持平。 |
| RLAIF | 「AI 標註的偏好」 | 標籤來自另一個 LM 而非人類的 RLHF。 |
| PRM | 「Process Reward Model」 | 對部分推理步驟打分數；用於推理 pipeline。 |
| Constitutional AI | 「Anthropic 的方法」 | 由明確規則引導、由 AI 產生的偏好。 |

## 延伸閱讀 (Further Reading)

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — 開啟 RLHF 的論文。
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — ChatGPT 背後的做法。
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — 較早期、用於摘要的 RLHF。
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO；2026 年 RLHF 之後的預設。
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF 與自我批判迴圈。
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH 論文。
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — 正式的 `RewardTrainer` 與 `PPOTrainer`。閱讀 trainer 原始碼，了解自適應 KL 與 value-head 的細節。
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf)，作者 Lambert、Castricato、von Werra、Havrilla — 附圖解、對三階段 pipeline 的經典導覽。
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — 這個函式庫；`examples/` 內有針對 Llama、Mistral 與 Qwen 的端到端 RLHF 腳本。
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward hypothesis 的觀點；思考 reward hacking 的必要前置知識。
