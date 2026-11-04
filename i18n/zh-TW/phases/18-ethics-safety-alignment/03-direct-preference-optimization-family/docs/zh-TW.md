# Direct Preference Optimization 家族 (The Direct Preference Optimization Family)

> Rafailov et al. (2023) 證明 RLHF 的最佳解可以用偏好資料寫成封閉形式，所以你可以跳過顯式的 reward model，直接最佳化 policy。這個洞見衍生出一整個家族：IPO、KTO、SimPO、ORPO、BPO，各自修補 DPO 的一種失敗模式。2026 年，direct alignment algorithm 所跑的 frontier post-training 比 PPO 還多。但第 2 課的 over-optimization curve 依然適用：DAA 並沒有逃過 Goodhart，只是改變了它咬人的位置。

**Type:** Learn
**Languages:** Python (stdlib, six-variant preference-loss comparator)
**Prerequisites:** Phase 18 · 01 (InstructGPT), Phase 18 · 02 (Reward hacking), Phase 10 · 08 (DPO basics)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 從帶 KL 的 RLHF 最佳解推導出 DPO 的封閉形式。
- 說出 IPO、KTO、SimPO、ORPO、BPO 各自修補 DPO 的哪一種失敗模式。
- 區分「implicit reward gap」與「偏好強度」，並解釋為什麼 IPO 的 identity mapping 很重要。
- 解釋為什麼 Rafailov et al. (NeurIPS 2024) 證明 DAA 即使沒有顯式 RM 仍然會 over-optimize。

## 問題 (The Problem)

RLHF 的目標函式（第 1 課）：

```math
\max_{\pi} \; \mathbb{E}_{x, y \sim \pi} \big[ r(x, y) \big] - \beta \, \mathrm{KL}(\pi \,\|\, \pi_{\text{ref}})
```

有已知的最佳解：

```math
\pi^*(y \mid x) = \frac{1}{Z(x)} \, \pi_{\text{ref}}(y \mid x) \exp\!\left(\frac{r(x, y)}{\beta}\right)
```

所以 reward 是由最佳 policy 與參考 policy 的比值隱含定義的：

```math
r(x, y) = \beta \log \frac{\pi^*(y \mid x)}{\pi_{\text{ref}}(y \mid x)} + \beta \log Z(x)
```

把它代入 Bradley-Terry 偏好 likelihood，partition function `Z(x)` 會被消去，因為它只依賴 `x`。剩下的是一個只含 policy 參數的 loss，不需要 reward model。這就是 DPO。

麻煩在於：這個推導假設最佳解是可以達到的、偏好資料屬於同分佈、而且參考 policy 是真正的 mode 錨點。這些條件沒有一個完全成立。家族中的每個成員，修補的是不同被違反的假設。

## 核心概念 (The Concept)

### DPO (Rafailov et al., 2023)

```math
L_{\text{DPO}} = -\log \sigma\!\left(
  \beta \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)}
  - \beta \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)}
\right)
```

可能出錯的地方：

- Implicit reward gap `beta * (log(pi/pi_ref)_w - log(pi/pi_ref)_l)` 沒有上界。再小的偏好都可能產生任意大的差距。
- 這個 loss 會把 chosen 與 rejected 的 log-prob 往相反方向推。只要 rejected 掉得更快，它就可以把 chosen 的絕對 log-prob 往下推。這就是 Degraded Chosen Response 現象。
- 分佈外的偏好（兩個回應在參考 policy 下都很罕見的配對），會產生任意的 implicit reward。

### IPO (Azar et al., 2024)

Identity Preference Optimization 把 log-sigmoid 換成偏好機率上的 identity mapping。loss 變成對有界目標的平方誤差：

```math
L_{\text{IPO}} = \left( \log \frac{\pi(y_w \mid x)}{\pi_{\text{ref}}(y_w \mid x)} - \log \frac{\pi(y_l \mid x)}{\pi_{\text{ref}}(y_l \mid x)} - \frac{1}{2\beta} \right)^2
```

margin 的上界是 `1/(2 beta)`。偏好強度與 implicit-reward 差距成正比。不會爆炸。

### KTO (Ethayarajh et al., 2024)

Kahneman-Tversky Optimization 完全放棄成對結構。給定單一個有標註的輸出，以及一個二元的「desirable」或「undesirable」訊號，它把它對應到 prospect-theory 的效用：

```math
v(x, y) = \sigma\!\left(\beta \log \frac{\pi(y \mid x)}{\pi_{\text{ref}}(y \mid x)} - z_{\text{ref}}\right)
```

對收益與損失使用不同權重（loss aversion）。好處：你可以使用未配對的資料，而這種資料多得多。

### SimPO (Meng et al., 2024)

Simple Preference Optimization 讓訓練訊號與生成對齊。完全移除參考 policy，並以長度對 log-likelihood 做正規化：

```math
L_{\text{SimPO}} = -\log \sigma\!\left(
  \frac{\beta}{|y_w|} \log \pi(y_w \mid x)
  - \frac{\beta}{|y_l|} \log \pi(y_l \mid x)
  - \gamma
\right)
```

加上 margin `gamma` 來穩定。長度正規化消除了利用 DPO 長度偏誤這種失敗模式的誘因（依構造，較長的 `y_w` 會有較大的 log-prob 差距）。

### ORPO (Hong et al., 2024)

Odds-Ratio Preference Optimization 在標準 SFT 的 negative log-likelihood 上加一個偏好項：

```math
\begin{aligned}
L_{\text{ORPO}} &= L_{\text{NLL}}(y_w) + \lambda L_{\text{OR}} \\
L_{\text{OR}} &= -\log \sigma\!\left(\log \frac{\text{odds}(y_w)}{\text{odds}(y_l)}\right)
\end{aligned}
```

沒有參考 policy，SFT 項就是 regularizer。從基礎模型到 aligned 模型，一個階段訓練完成。不需要獨立的 SFT checkpoint。

### BPO (ICLR 2026 submission, OpenReview id=b97EwMUWu7)

指出 Degraded Chosen Responses 問題：DPO 保留了排序 `y_w > y_l`，但 `y_w` 的絕對 log-prob 可能下降。BPO 加上一行修正，懲罰 chosen 回應往下的移動。在 Llama-3.1-8B-Instruct 的數學推理上，回報比 DPO 高 +10.1% 的準確率。

### 普遍的結果：DAA 仍然會 over-optimize

Rafailov et al.「Scaling Laws for Reward Model Overoptimization in Direct Alignment Algorithms」(NeurIPS 2024) 在多個資料集、多種 KL 預算下，用 DPO、IPO、SLiC 訓練 policy。gold-reward 對 KL 的曲線，呈現與 Gao et al. 相同的先峰值後崩潰形狀。Implicit reward 在訓練期間會查詢分佈外的樣本；KL regularization 無法穩定這一點。

DAA 沒有逃過 Goodhart。它們把咬人的位置，從「reward model 被 over-optimize」改成「參考 policy 比值被 over-optimize」。通用的解法，如更好的資料、ensemble、提早停止，對兩者都適用。

### 如何選擇（2026）

- 如果你有大量成對偏好資料：用保守 beta 的 DPO；若長度偏誤明顯，用 SimPO。
- 如果你有未配對的二元回饋：KTO。
- 如果你想要從基礎模型出發的單階段 pipeline：ORPO。
- 如果你在 DPO 日誌中看到 chosen log-prob 下降：BPO。
- 如果偏好強度差異很大，而且 DPO 已飽和：IPO。

每個實驗室都在一組測試上跑這五種，再依任務挑出贏家。沒有理由認為數學推理和安全的最佳解是同一個。

```figure
dpo-margin
```

## 實際使用 (Use It)

`code/main.py` 在一個玩具偏好資料集上比較六種 loss（DPO、IPO、KTO、SimPO、ORPO、BPO），資料集中每一對的真實偏好強度各不相同。每種 loss 都用同樣的 500 對樣本，在一個小型 softmax policy 上最佳化。針對每個方法畫出最終勝率、chosen-log-prob 漂移，以及 implicit-reward 的分散程度。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-preference-loss-selector.md`。給它資料集統計（成對或未配對、偏好強度可變或一致、長度分佈）和目標（單階段或先 SFT 再偏好），它會推薦一種偏好 loss，並回報它所防範的失敗模式。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報 DPO 與 BPO 最終 chosen-log-prob 的下降量。BPO 應該保留較高的 chosen 絕對機率，請驗證這點。

2. 修改偏好資料，讓所有配對的強度相等。六種方法中哪一個最穩健？哪一個會退化？解釋 IPO 在這裡的優勢。

3. 讓 rejected 回應平均比 chosen 長 2 倍。其他都不改，用數值展示 DPO 的長度利用，以及 SimPO 的修正。

4. Rafailov et al. (NeurIPS 2024) 主張 DAA 會 over-optimize。重現一個單點版本：畫出 chosen 減 rejected 的 KL divergence，並在大 beta 下觀察 DPO 的 over-optimization。

5. 閱讀 BPO 論文摘要 (OpenReview b97EwMUWu7)。寫下 BPO 加到 DPO 上的那一行修正，並與 `code/main.py` 中的實作對照確認。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| DPO | 「RLHF without a reward model」 | 由 RLHF 封閉形式最佳解推導出的 loss；只含 policy 參數 |
| Implicit reward | 「the log-ratio」 | `beta * log(pi(y\|x) / pi_ref(y\|x))`，DPO 所隱含的 reward |
| IPO | 「bounded DPO」 | 把 log-sigmoid 換成 identity；implicit reward 差距上限為 `1/(2 beta)` |
| KTO | 「unpaired DPO」 | 對單一標籤套用帶 loss aversion 的 prospect-theory 效用 |
| SimPO | 「reference-free DPO」 | 長度正規化的 log-likelihood 加 margin；沒有參考 policy |
| ORPO | 「one-stage DPO」 | NLL 加 odds-ratio 偏好項；從基礎模型一次訓練完成 |
| BPO | 「chosen-preserving DPO」 | DPO 加上對 chosen 回應絕對 log-prob 下降的懲罰 |
| Degraded Chosen | 「chosen goes down」 | 只要 rejected 掉得更快，DPO 就會降低 chosen 的 log-prob |
| DAA | 「direct alignment algorithm」 | 任何跳過顯式 RM 的偏好 loss 方法 |

## 延伸閱讀 (Further Reading)

- [Rafailov et al. — Direct Preference Optimization (NeurIPS 2023, arXiv:2305.18290)](https://arxiv.org/abs/2305.18290)
- [Azar et al. — A General Theoretical Paradigm to Understand Learning from Human Preferences (AISTATS 2024, arXiv:2310.12036)](https://arxiv.org/abs/2310.12036)：IPO
- [Ethayarajh et al. — KTO: Model Alignment as Prospect Theoretic Optimization (arXiv:2402.01306)](https://arxiv.org/abs/2402.01306)
- [Meng, Xia, Chen — SimPO (NeurIPS 2024, arXiv:2405.14734)](https://arxiv.org/abs/2405.14734)
- [Hong, Lee, Thorne — ORPO (EMNLP 2024, arXiv:2403.07691)](https://arxiv.org/abs/2403.07691)
- [BPO — Behavior Preservation Optimization (ICLR 2026 OpenReview b97EwMUWu7)](https://openreview.net/forum?id=b97EwMUWu7)
- [Rafailov et al. — Scaling Laws for RM Overoptimization in DAAs (NeurIPS 2024, arXiv:2406.02900)](https://arxiv.org/abs/2406.02900)
