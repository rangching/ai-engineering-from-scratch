# 把遵循指令當作 Alignment 訊號 (Instruction-Following as Alignment Signal)

> 後面每一個對 RLHF 的批評，都是針對這條 pipeline 而來。在研究最佳化壓力如何扭曲 proxy 之前，你得先看清楚這個 proxy。InstructGPT (Ouyang et al., 2022) 定義了參考架構：用指令與回應配對做 supervised fine-tuning，用成對偏好排序訓練 reward model，再用 PPO 對 reward model 最佳化，並加上對 SFT policy 的 KL penalty。1.3B 的 InstructGPT 勝過 175B 的 GPT-3，被人類更偏好。就是這個結果，讓 2026 年每一家 frontier 實驗室至今仍然採用 RLHF 形式的 post-training pipeline。

**Type:** Learn
**Languages:** Python (stdlib, toy three-stage pipeline)
**Prerequisites:** Phase 10 · 06 (SFT), Phase 10 · 07 (RLHF), Phase 10 · 08 (DPO)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 說出 InstructGPT pipeline 的三個階段，以及每個階段使用的 loss。
- 解釋為什麼 1.3B 的 instruction-tuned 模型，在人類偏好評估上贏過原始的 175B GPT-3。
- 說明第 3 階段的 KL penalty 在防什麼，以及移除它為什麼會退化成 mode-seeking 行為。
- 描述 alignment tax，以及 Ouyang et al. 用來對付它的 PPO-ptx 做法。

## 問題 (The Problem)

預訓練的語言模型只會接續文字，不會回答問題。你問 GPT-3「寫一個反轉 list 的 Python 函式」，常常得到另一個 prompt，因為大部分訓練分佈是網頁文字，而網頁文字後面接的還是網頁文字。模型做的是它該做的事，只是這個「該做的事」不對。

每家認真的實驗室用來解決這個問題的 proxy，都是人類偏好。兩個 completion 交給評分員，評分員選出比較好的，reward model 去學這個評分員。接著用 RL 迴圈把 policy 推向 reward model 給高分的輸出。這就是 InstructGPT 的完整主張，三句話講完。論文其餘部分都是工程。

## 核心概念 (The Concept)

### 第 1 階段：supervised fine-tuning (SFT)

收集 prompt 與回應的配對，回應是一個善意的人會寫出的內容。Ouyang et al. 使用了 13k 個來自標註員與 OpenAI API 的 prompt。用標準的 cross-entropy loss 在這些資料上 fine-tune 基礎模型。

SFT 給你的：模型現在會回答問題，而不是接續問題。它沒給你的：當有多個答案都合理時，評分員偏好哪一個的任何訊號。

### 第 2 階段：reward model (RM)

對每個 prompt，從 SFT 模型取樣 K 個 completion，由標註員排序。訓練一個 reward model，讓它對任何 prompt 與回應的配對打分，使得當 `y_w` 比 `y_l` 更受偏好時：

```math
L_{\text{RM}} = -\log \sigma\big(r(x, y_w) - r(x, y_l)\big)
```

這是 Bradley-Terry 成對偏好 loss。RM 通常從 SFT 模型初始化，把 LM head 換成純量 head。

Reward model 很小：6B 就足以支撐 175B 的 InstructGPT。它們也很脆弱，論文第 5 節大多在談小規模就已出現的 reward-hacking 行為。

### 第 3 階段：帶 KL penalty 的 PPO

定義目標函式：

```math
J(\pi) = \mathbb{E}_{x \sim D,\, y \sim \pi(\cdot \mid x)} \big[ r(x, y) \big] - \beta \, \mathrm{KL}\big(\pi(\cdot \mid x) \,\|\, \pi_{\text{SFT}}(\cdot \mid x)\big)
```

用 PPO 最大化它。KL 項讓 `pi` 不會離 SFT policy 太遠。沒有它，最佳化器會找到 adversarial example，也就是 RM 給高分的字串，原因是 RM 從沒看過它們，而不是人類真的偏好它們。

KL 係數 `beta` 是 RLHF 最重要的單一超參數。太低：reward hacking。太高：相較 SFT 毫無進步。

### Alignment tax

RLHF 之後，模型更受人類偏好，但在標準 benchmark（SQuAD、HellaSwag、DROP）上退步。Ouyang et al. 把這稱為 alignment tax，並用 PPO-ptx 修正：把預訓練的 gradient 混進 RL 目標，讓模型不會忘記那些從未被獎勵過的下游任務該怎麼做。

```math
J_{\text{ptx}}(\pi) = J(\pi) + \gamma \, \mathbb{E}_{x \sim D_{\text{pretrain}}} \big[ \log \pi(x) \big]
```

PPO-ptx 成為標準做法。Anthropic、DeepMind 和 Meta 都使用某種變體。

### 結果

1.3B 的 InstructGPT (SFT + RM + PPO-ptx) 約有 70% 的時間被標註員評為優於 175B 的基礎 GPT-3。在來自正式流量的 hidden-test prompt 上，差距更大。從這個數字可以讀出兩件事：

1. Alignment 是和能力不同的另一條軸。175B 模型能力較強；1.3B 模型 alignment 較好；標註員偏好 aligned 的那個。
2. 能力的下限由基礎模型決定。你無法用 RLHF 讓基礎模型知道它從沒看過的事實。

### 為什麼這是 Phase 18 的參考基準

後面課程的每一個批評，包含 reward hacking（第 2 課）、DPO（第 3 課）、sycophancy（第 4 課）、CAI（第 5 課）、sleeper agents（第 7 課）、alignment faking（第 9 課），都是在反對這條 pipeline 的某個部分。Reward hacking 攻擊第 2 階段。DPO 把第 2、3 階段合併。CAI 取代人類標註員。Sycophancy 顯示標註員是帶偏差的訊號。Alignment faking 顯示 policy 可以完全繞過第 3 階段。如果腦中沒有這條 pipeline，你無法跟上任何一個批評。

```figure
al-instruct-pipeline
```

## 實際使用 (Use It)

`code/main.py` 用玩具偏好資料模擬這三個階段。基礎「policy」是在動作 {A, B, C} 上有偏差的硬幣。第 1 階段 SFT 在 200 個 prompt 上模仿標註員的動作。第 2 階段從 500 筆成對排序擬合出 Bradley-Terry reward model。第 3 階段執行簡化的 PPO 更新，並加上對 SFT policy 的 KL penalty。你可以看著 reward 上升、KL divergence 增大、policy 漂移，也可以關掉 KL 項，看 reward hacking 在 50 步更新內出現。

要觀察的重點：

- `beta = 0.1` 與 `beta = 0.0` 的 reward 軌跡。
- 訓練步數上的 $\mathrm{KL}(\pi \Vert \pi_{\text{SFT}})$。
- 最終的動作分佈與標註員偏好的比較。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-instructgpt-explainer.md`。給它一份 RLHF pipeline 描述或論文摘要，它會指出被修改的是三個階段中的哪一個、每個階段用什麼 loss，以及是否存在 KL penalty 或等效的 regularizer。

## 練習 (Exercises)

1. 執行 `code/main.py`。設定 `beta = 0.0`，回報 200 步 PPO 後的動作分佈。用一段話解釋 mode-seeking 行為。

2. 修改 reward model，讓動作 B 有 +0.5 的偏差（模擬 reward bug）。以 `beta = 0.1` 執行 PPO。KL penalty 能防止 policy 利用這個偏差嗎？`beta` 到多少時，利用行為會變得明顯？

3. 閱讀 Ouyang et al. (arXiv:2203.02155) 的 Figure 1。分別執行 1、5、20、100 步 PPO，並測量對 SFT 模型的偏好，重現標註員偏好曲線。

4. 論文 Section 4.3 指出 1.3B 的 InstructGPT 約有 70% 的時間勝過 175B 的 GPT-3。為什麼在 hidden 的正式 prompt 上這個比例會比在標註員自己的 prompt 上更高？

5. 在相同的偏好資料上，把 PPO loss 換成 DPO (Phase 10 · 08)。比較最終 policy 漂移（對 SFT 的 KL）與最終 reward。在 reward 相同時，哪個方法漂得更遠？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| SFT | 「instruction tuning」 | 第 1 階段：在 prompt 與回應配對上做 cross-entropy fine-tune |
| Reward model | 「the RM」 | 對 (prompt, response) 輸出純量的回歸器，用成對標籤以 Bradley-Terry 訓練 |
| Bradley-Terry | 「pairwise preference loss」 | $-\log \sigma(r_w - r_l)$；把成對排序化簡為二元分類 |
| KL penalty | 「the regularizer」 | `beta * KL(pi \|\| pi_SFT)`，讓 RL policy 停留在 SFT 錨點附近 |
| PPO-ptx | 「PPO with pretraining mix」 | 在 PPO 目標中加入一部分預訓練 log-likelihood，抵銷 alignment tax |
| Alignment tax | 「the RLHF regression」 | RLHF 後在 RLHF 沒有針對的標準 benchmark 上的下降 |
| Labeler preference | 「the ground truth」 | 人類排序的樣本；RM 是它的統計 proxy，而不是「人類價值」的 proxy |

## 延伸閱讀 (Further Reading)

- [Ouyang et al. — Training language models to follow instructions with human feedback (arXiv:2203.02155)](https://arxiv.org/abs/2203.02155)：InstructGPT 論文，後續每一條 RLHF pipeline 的基礎
- [Stiennon et al. — Learning to summarize from human feedback (arXiv:2009.01325)](https://arxiv.org/abs/2009.01325)：用於摘要的 RLHF 前作
- [Christiano et al. — Deep reinforcement learning from human preferences (arXiv:1706.03741)](https://arxiv.org/abs/1706.03741)：最早的偏好式 RL 公式
- [Bai et al. — Training a Helpful and Harmless Assistant with RLHF (arXiv:2204.05862)](https://arxiv.org/abs/2204.05862)：Anthropic 對 InstructGPT pipeline 的 HH 延伸
