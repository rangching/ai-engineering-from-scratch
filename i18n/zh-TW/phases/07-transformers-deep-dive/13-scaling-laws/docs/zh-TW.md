# Scaling Laws

> 2020 年 Kaplan 的論文說：模型越大，loss 越低。2022 年 Hoffmann 的論文說：你之前訓練不足。算力分成兩個桶子，參數與 token，而怎麼分並不明顯。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 分鐘

## 問題 (The Problem)

當你有 C FLOPs 的訓練算力，想得到最好的模型時，你面對兩個旋鈕：

1. **多少參數 (N)？** 模型越大，容量越高。
2. **多少訓練 token (D)？** 資料越多，越能善用容量。

FLOPs 大約以 `6 × N × D` 成長。你可以把 N 調高、D 調低，或把 D 調高、N 調低。哪個比較好？

2022 年以前，答案是「大力推高 N」。GPT-3（2020）有 175B 參數，在約 300B 個 token 上訓練。比例約為每個參數 1.7 個 token。Kaplan scaling law 支持這個做法。

Hoffmann et al. (2022) 訓練了一小組名為 Chinchilla 的模型，發現了不同的結果：最佳比例更接近**每個參數 20 個 token**。GPT-3 訓練不足了 10 倍。Chinchilla（70B 參數、1.4T 個 token）在每個基準上都勝過 GPT-3（175B、300B 個 token），而推論成本少了 2.5 倍。

2026 年是 Chinchilla 的世界，但有一個重要的轉折。Llama 3 8B 用 15 兆個 token 訓練，比例是每個參數 1,875 個 token。超過 Chinchilla 最佳值 94 倍。對於會被大規模使用的模型，推論成本比訓練成本更重要，所以為了更小、可部署的體積而 over-train（超過 Chinchilla），是 2026 年的預設。

## 核心概念 (The Concept)

![Chinchilla curves: loss vs compute at various N/D ratios](../assets/scaling-laws.svg)

### Hoffmann 定律 (The Hoffmann law)

根據 Chinchilla 論文，loss 遵循：

```math
L(N, D) = \frac{A}{N^\alpha} + \frac{B}{D^\beta} + E
```

- $N$ = 參數量（不含 embedding）。
- $D$ = 訓練 token 數。
- $\alpha \approx 0.34$、$\beta \approx 0.28$（大致對稱）。
- $E \approx 1.69$，不可降低的 loss 上限。
- $A \approx 406$、$B \approx 411$。

隨著規模擴大，兩項互相拉扯。在固定算力（$C = 6ND$）下，對 $N$ 取導數並求解。以上述的 $\alpha$ 和 $\beta$，$N$ 的指數約為 0.45，$D$ 的指數約為 0.55，所以精確的最佳比例會隨預算緩慢漂移；`code/main.py` 用 grid search 找出它。論文的前兩種擬合方法直接測量最佳值，兩個指數都約為 0.5。這就給出固定比例的經驗法則。在 $C = 6ND$ 中令 $D = 20N$，經驗法則就是：

```math
\begin{aligned}
N_{\text{opt}} &\approx 0.22 \times (C/6)^{0.5} \\
D_{\text{opt}} &\approx 4.5 \times (C/6)^{0.5} \\
D_{\text{opt}} / N_{\text{opt}} &\approx 20
\end{aligned}
```

算力最佳：每個參數 20 個 token。

### 為什麼還是要 over-train (Why over-training anyway)

Chinchilla-optimal 最小化的是每單位訓練 FLOP 的訓練 loss。但訓練成本你只付一次；推論成本卻是永遠要付。

對一個每月服務一兆個 token 的聊天機器人來說，推論主導了總成本。Llama 的做法：訓練更小的模型，訓練更久。8B 搭配 15T 個 token，是深度針對推論最佳化的：

- 能放進消費級 GPU。
- 延遲只是 70B Chinchilla-optimal 的一小部分。
- 品質對大多數任務來說夠接近了。

DeepMind 2024 年的論文（"Over-training is the new optimal"）將此形式化。對以推論為主的工作負載，依服務量而定，正確的比例更接近每個參數 100 到 500 個 token。

### Emergence 對上平滑性 (Emergence vs smoothness)

主張：某些能力（算術、多步驟推理、遵循 chain-of-thought）會在某個規模「突然浮現」。

Schaeffer et al. (2023) 主張這是測量上的假象：emergent 的指標使用不連續的計分（exact match、在門檻處的準確率），掩蓋了底層 logit 的平滑進步。連續指標（cross-entropy）呈現平滑曲線。

2026 年的共識是：透過連續 loss 做預測是可靠的。基準分數的跳升常是計分器造成的假象。請依據連續指標來規劃預算。

### 2026 年的樣貌 (The 2026 picture)

Scaling law 依然有效，但：

| 因素 | 改變的方式 |
|--------|-------------|
| 資料品質 | 精選「好的」token（Phi 風格）會讓曲線移動超過 2 倍的有效算力 |
| MoE | 總參數與啟動 FLOPs 解耦；scaling law 改以每啟動 FLOP 計算 |
| Post-training | 有些能力（遵循指令、程式碼）受 SFT+RLHF 的影響比預訓練更大 |
| 多模態 | 影像 + 文字 token 一起 scale；每個模態有各自的曲線 |
| 合成資料 | 模型自己生成訓練資料；有效算力可以複利成長 |

Muon optimizer（Kimi Moonlight，2024）在相同資料下，展現出比 AdamW 約 2 倍的有效算力增益。2026 年有些訓練預設使用 Muon。它改變 scaling law 的絕對常數，而不是它的形狀。

```figure
scaling-laws
```

## 動手做 (Build It)

請看 `code/main.py`。我們實作 Chinchilla 的 loss 方程式，並在數個算力預算下求解算力最佳的 $(N, D)$。

### 步驟 1：Chinchilla loss (Chinchilla loss)

```python
def chinchilla_loss(N, D, A=406.4, B=410.7, alpha=0.34, beta=0.28, E=1.69):
    return A / N ** alpha + B / D ** beta + E
```

在固定 $C = 6ND$ 下，把 $L$ 畫成 $(N, D)$ 上的等高線。找出最小值。

### 步驟 2：算力最佳的前緣 (compute-optimal frontier)

對從 `1e17` 到 `1e25` FLOPs 的算力預算，找出在 $6ND = C$ 限制下最小化 loss 的 $(N, D)$。驗證比例 $D/N \approx 20$。

### 步驟 3：Over-training 的成本 (over-training cost)

計算訓練一個小 10 倍的模型（最佳 N 的 1/10、最佳 D 的 10 倍）要多付出的額外 loss。同時回報換來的推論 FLOP 節省（與 N 成正比）。

### 步驟 4：與真實模型比較 (compare to real models)

代入 GPT-3、Chinchilla、Llama 3 8B、DeepSeek-V3（啟動參數）已知的 `(N, D)` 配對，比較預測與回報的 loss。

## 實際使用 (Use It)

你不太可能自己訓練前沿模型。但 scaling law 能告訴你：

1. **你的 fine-tune 資料是否足夠。** 如果你任務專用的資料低於基礎模型每個參數 20 個 token，預期會在某個 loss 下限飽和。
2. **是否該選更大的基礎模型。** 如果你把全部預算都花在推論上，請偏好更小、訓練更久的模型。
3. **報酬在哪裡遞減。** 超過 Chinchilla-optimal 的 1000 倍後，log-loss 的變化就成了雜訊。

**2026 年的研究軌跡：**

- **資料受限的情境。** 網路上高品質 token 的數量是有限的（過濾後的英文約 5 到 10 兆）。前沿預訓練正在逼近這個上限。合成資料、多語言、多模態，以及用 RLHF 擴大規模的 fine-tuning 是下一批槓桿。
- **算力倍增的技巧。** Muon optimizer、MoE、更好的資料策展，每一個都改變絕對常數，而不是漸近線。
- **RL 的 scaling law。** 開放問題。早期證據顯示 RL 樣本上存在 power-law，但指數與預訓練差異很大。

## 交付 (Ship It)

請看 `outputs/skill-training-budget-estimator.md`。這個 skill 會根據算力預算、部署限制與目標 loss，為新的訓練執行挑選 `(N, D, hours, GPU)`。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。印出算力預算 `1e20`、`1e22`、`1e24` 下 Chinchilla-optimal 的 $(N, D)$。與真實模型表比較。
2. **中等。** 實作 Hoffmann 的「loss 作為算力函式」曲線。畫出算力最佳前緣上 loss 對 `log10(C)` 的圖。找出定律預測我們何時需要 $>10^{28}$ FLOPs，才能讓 cross-entropy 再降 0.1。
3. **困難。** 在 5 個用同一資料集訓練的小模型（100K 到 10M 參數）上，擬合你自己的 scaling law。估計 $\alpha$ 與 $E$。你的指數與已發表的指數吻合到什麼程度？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Parameters (N) | 「模型大小」 | 不含 embedding 的權重數量；決定容量。 |
| Tokens (D) | 「訓練資料」 | 看過的訓練 token 數量；決定參數被善用的程度。 |
| Compute (C) | 「花掉的 FLOPs」 | 標準 transformer 大約是 `6 × N × D`。 |
| Chinchilla-optimal | 「D/N ≈ 20」 | 讓每單位預訓練 FLOP 的 loss 最小的比例。 |
| Over-training | 「超過 Chinchilla」 | 多花訓練 FLOPs 來節省推論 FLOPs；D/N >> 20。 |
| Irreducible loss | 「下限」 | Scaling law 中的 $E$ 項；資料本身的 entropy。 |
| Emergent capability | 「規模擴大時的突然跳升」 | 常是計分器造成的假象；連續 loss 是平滑的。 |
| Effective compute | 「訓練效率的倍增器」 | 更好的資料 / optimizer / 架構，會讓一個 FLOP 走得更遠。 |

## 延伸閱讀 (Further Reading)

- [Kaplan et al. (2020). Scaling Laws for Neural Language Models](https://arxiv.org/abs/2001.08361) — 第一篇 scaling law 論文；訓練不足。
- [Hoffmann et al. (2022). Training Compute-Optimal Large Language Models](https://arxiv.org/abs/2203.15556) — Chinchilla。
- [Schaeffer et al. (2023). Are Emergent Abilities of Large Language Models a Mirage?](https://arxiv.org/abs/2304.15004) — 把 emergence 視為測量假象。
- [Sardana, Frankle (2024). Beyond Chinchilla-Optimal: Accounting for Inference in Language Model Scaling Laws](https://arxiv.org/abs/2401.00448) — 為什麼 Llama 的 over-training 對它的工作負載是正確的。
- [Jordan et al. (2024). Muon: An optimizer for hidden layers in neural networks](https://kellerjordan.github.io/posts/muon/) — 2 倍算力倍增器。
