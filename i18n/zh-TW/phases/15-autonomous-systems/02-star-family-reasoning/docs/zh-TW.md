# STaR、V-STaR、Quiet-STaR：自我教學推理 (Self-Taught Reasoning)

> 最小的自我改進 loop 就藏在 rationale 裡。模型產生一段 chain of thought，保留那些得到正確答案的，然後拿它們來 fine-tune。這就是 STaR。V-STaR 加上 verifier，讓推論時的挑選更好。Quiet-STaR 則把 rationale 推進到每一個 token。三者都有效，但沒有一個是魔法：這個 loop 會保留任何碰巧得到正確答案的捷徑。

**Type:** Learn
**Languages:** Python (stdlib, bootstrap-loop simulator)
**Prerequisites:** Phase 13 · 01-03 (Reasoning and CoT), Phase 15 · 01 (long-horizon framing)
**Time:** ~60 分鐘

## 問題 (The Problem)

教模型推理最直接的方法，是收集人類寫的推理軌跡。這很貴、很慢，而且受限於人類願意寫多少高品質的 chain-of-thought。

STaR (Self-Taught Reasoner, Zelikman et al., 2022) 問的是：如果模型自己寫 rationale，再拿已知答案來評分呢？這個 loop 是：

1. 取樣一條推理軌跡加上答案。
2. 如果最終答案正確，保留這條軌跡。
3. 用保留下來的軌跡做 fine-tune。
4. 重複。

它有效。GSM8K 和 CommonsenseQA 都在沒有新的人工標註下進步了。但這個 loop 有內建的偏差：任何產生正確答案的 rationale 都會被保留，不管推理本身是否合理。V-STaR (Hosseini et al., 2024) 用學出來的 verifier 修補這點；Quiet-STaR (Zelikman et al., 2024) 則把這個想法推廣到每個 token 的內部 rationale。

## 核心概念 (The Concept)

### STaR：從有效的做法 bootstrap

從一個推理能力偏弱的 base model 開始。對每個訓練問題，取樣一段 rationale 加答案。如果答案符合標籤，就保留這組 (problem, rationale, answer)。用保留下來的集合 fine-tune 模型。重複。

有一個細節很重要。如果模型永遠答不對某個問題，這個 loop 就無法從中學習。STaR 加入 **rationalization**：對模型答錯的問題，把正確答案當作提示放進去，再要求模型產生一段通向該答案的 rationale。這些 rationalize 出來的 rationale 也會加進訓練集。

原始論文 (Zelikman et al., 2022) 的結果：GPT-J base model 在 GSM8K 上，透過帶有 rationalization 的多輪 STaR，從 5.8% 進步到 10.7%，絕對值約 5 個百分點。在 CommonsenseQA 上，經 STaR 訓練的 GPT-J 6B 達到 72.5%，與 fine-tune 過的 GPT-3 175B（約 73%）相當，後者的規模大約是前者的 30 倍，而且是用人工標註的 rationale 訓練的。

### V-STaR：用 DPO 訓練 verifier

STaR 把錯誤的 rationale 丟掉。Hosseini et al. (2024) 觀察到，這些也是資料：每一組 (rationale, 「這是否正確」) 都可以拿來訓練 verifier。他們對正確與錯誤的解答都使用 Direct Preference Optimization，建立一個 ranker。推論時，取樣 N 條 rationale，挑 verifier 評分最高的那個。

回報的進步幅度：在 GSM8K 和 MATH 上，比先前的自我改進 baseline 高 +4 到 +17 個百分點，其中大部分的收益來自用 verifier 做推論時的挑選，而不是用它做額外的 generator fine-tuning。

### Quiet-STaR：每個 token 的內部 rationale

Zelikman et al. (2024) 問：如果模型學會在每個 token 位置都產生一小段內部 rationale，而不只是在問題與答案之間，會怎樣？Quiet-STaR 訓練模型在預測每個 token 之前，先輸出一段隱藏的「想法」，再透過學出來的權重，把「有想法」的預測與 baseline 預測混合。

結果：Mistral 7B 在沒有針對特定任務 fine-tune 的情況下，zero-shot 的絕對進步為：GSM8K 從 5.9% 到 10.9%，CommonsenseQA 從 36.3% 到 47.2%。模型學會了「何時該思考」：困難的 token 會得到較長的內部 rationale，簡單的幾乎沒有。

### 為什麼三者都有同一個安全疑慮

三種方法都用最終答案當作 gradient 訊號。一段透過有缺陷的推理（利用捷徑、猜測，或使用無法泛化的模式）得到正確答案的 rationale，會被正向強化。在 in-distribution 的問題上，捷徑有效。在 out-of-distribution 的問題上，它會悄悄失效。

V-STaR 的 verifier 透過學習對 rationale 排序來減輕這個問題，但 verifier 是用同一組標籤訓練的。它可能學會偏好格式漂亮的錯誤推理，而不是誠實的不確定。比較安全的設計，是把 STaR 風格的資料結合 (a) process-supervised reward model（獎勵中間步驟，而不只是答案），以及 (b) 能打破簡單捷徑的 held-out OOD 評估。

### 比較

| 方法 | 訓練訊號 | 推論成本 | 資料浪費 | 已知失敗模式 |
|---|---|---|---|---|
| STaR | 答案正確就保留 (rationale, answer) | 1x | 丟掉所有錯誤的 rationale | 捷徑 rationale |
| STaR + rationalization | 以上，加上提示正確答案的重試 | 1x | 較少 | rationalize 出來的 rationale 可能不合理 |
| V-STaR | STaR + 用兩類資料訓練的 DPO verifier | Nx (best-of-N) | 極少 | verifier 可能強化有自信的錯誤 |
| Quiet-STaR | 每個 token 的 rationale + 混合權重 | 1.5-3x | 極少 | gradient 仍然以答案為條件 |

### 它在 2026 技術棧中的位置

STaR 很老了。但這個模式在 2025-2026 年到處重現。在可驗證的數學問題上做 RL（DeepSeek-R1、Kimi-k1.5、o1），就是放大版的 STaR 以答案為條件的 gradient 訊號。Process reward model (Lightman et al., 2023；OpenAI 的 "Let's verify step by step") 則是 process-supervised 的替代方案。AlphaEvolve（第 3 課）是針對程式碼的 STaR，用程式評估器取代標籤。Darwin Godel Machine（第 4 課）則是針對 agent scaffolding 本身的 STaR。

理解 STaR，這些東西就都通了。它是最小可行的自我改進 loop。

```figure
reflection-loop
```

## 實際使用 (Use It)

`code/main.py` 在一個玩具算術任務上執行模擬的 STaR loop。你可以觀察：

- 準確率如何隨著 bootstrap 輪數上升。
- 捷徑如何混進來：simulator 包含一個「偷懶」的 rationale 類別，它有 40% 的機率得到正確答案，但泛化很差。看看 STaR 會不會保留它們。
- Verifier（V-STaR 風格）如何在推論時幫上忙，但無法完全清除訓練時引入的捷徑。

## 交付 (Ship It)

`outputs/skill-star-loop-reviewer.md` 幫你在訓練之前，稽核一條提議中的 self-taught-reasoning pipeline。

## 練習 (Exercises)

1. 執行 simulator。把捷徑頻率設成零，再設成 0.4。即使兩次在訓練分布上都達到 90% 以上，最終準確率的差距有多大？

2. 在 simulator 中加入 held-out OOD 測試。從不同的分布抽取問題，並同時在 in-distribution 與 OOD 資料集上評估 bootstrap 出來的模型。量化兩者的差距。

3. 閱讀 Quiet-STaR 論文 (arXiv:2403.09629) 的第 3 節。各用三句話解釋「end-of-thought」token 和 mixing-weight head。

4. 比較 STaR 的「答對才保留」過濾，和獨立獎勵每個 rationale 步驟的 process-supervised 替代方案。找出標註成本的差異，以及可能的品質差異。

5. 設計一個能在已部署模型中抓到捷徑 rationale 的評估。它不必完美，只要能打破 STaR loop 會強化的那些最簡單的捷徑。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| STaR | 「Self-Taught Reasoner」 | 用模型產生、且得到正確答案的 rationale 做 fine-tune；重複 |
| Rationalization | 「帶提示的重試」 | 對 base model 答錯的問題，注入正確答案，再要求它產生 rationale |
| V-STaR | 「Verifier STaR」 | 用正確與錯誤的 rationale 以 DPO 訓練 verifier，用於推論時的挑選 |
| Quiet-STaR | 「每個 token 的 rationale」 | 在每個 token 位置產生隱藏的想法；與 baseline 預測混合 |
| Answer-conditioned gradient | 「以結果為基礎的訊號」 | 訓練 loop 獎勵最終答案，而不是推理步驟 |
| Process reward model | 「步驟層級的 verifier」 | 用每一步的正確性而非結果訓練的 reward model，與 STaR 形成對比 |
| Shortcut rationale | 「答案對，推理錯」 | 透過無法泛化的模式得到標籤的 rationale；STaR 會保留這些 |

## 延伸閱讀 (Further Reading)

- [Zelikman et al. (2022). STaR: Bootstrapping Reasoning With Reasoning](https://arxiv.org/abs/2203.14465) — 原始論文。
- [Hosseini et al. (2024). V-STaR: Training Verifiers for Self-Taught Reasoners](https://arxiv.org/abs/2402.06457) — 加入 DPO verifier 做推論時的挑選。
- [Zelikman et al. (2024). Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking](https://arxiv.org/abs/2403.09629) — 每個 token 的內部 rationale。
- [Lightman et al. (2023). Let's Verify Step by Step](https://arxiv.org/abs/2305.20050) — process reward model，另一種 gradient 訊號。
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — 在可驗證任務上做 RL，把 STaR 放大到 frontier 訓練。
