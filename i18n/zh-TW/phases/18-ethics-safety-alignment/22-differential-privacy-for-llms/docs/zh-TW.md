# LLM 的 Differential Privacy (Differential Privacy for LLMs)

> DP-SGD 仍是標準做法：注入雜訊的 gradient 更新提供形式化的 (epsilon, delta) 保證。運算、記憶體與效用上的開銷相當可觀；參數高效的 DP fine-tuning (LoRA + DP-SGD) 是 2025 年的常見配置 (ACM 2025)。兩批證據彼此拉扯：以 canary 為基礎的 membership inference (Duan et al., 2024) 回報對語言模型的成功有限；訓練資料萃取 (Carlini et al., 2021; Nasr et al., 2025) 則能恢復大量逐字記憶。調和之道 (arXiv:2503.06808, March 2025)：差距在於測量的對象不同，是被植入的 canary，還是「最容易被萃取」的資料。新的 canary 設計讓不需要 shadow model 的 loss-based MIA 成為可能，並對一個以真實資料訓練、具備實際 DP 保證的 LLM，做出第一次非平凡的 DP 稽核。替代方案：PMixED (arXiv:2403.15638)，推論時透過 next-token 分佈上的 mixture of experts 做 private prediction；DP 合成資料生成 (Google Research 2024)。新興攻擊：Differential Privacy Reversal via LLM Feedback，也就是信心分數外洩。

**Type:** Build
**Languages:** Python (stdlib, DP-SGD noise-injection and ε-δ accountant demonstration)
**Prerequisites:** Phase 01 · 09 (information theory), Phase 10 · 01 (large-model training)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 定義 (epsilon, delta)-differential privacy，並說明 DP-SGD 的做法。
- 解釋 2024-2025 年的拉扯：canary MIA 與訓練資料萃取給出不同的圖像。
- 描述 PMixED，以及為什麼推論時的 private prediction 是 DP 訓練的替代方案。
- 描述 Differential Privacy Reversal via LLM Feedback 攻擊。

## 問題 (The Problem)

LLM 會記憶。Carlini et al. 2021 顯示，量產的語言模型能依要求逐字重現訓練文字。DP 是形式化的防禦：訓練成讓輸出在可證明的意義上對任何單一訓練樣本不敏感。2024-2025 年的證據顯示，DP-SGD 是必要的，但已部署的 ε 值可能不符合威脅模型。

## 核心概念 (The Concept)

### (ε, δ)-differential privacy

對於任意兩個僅相差一個樣本的資料集，以及任何事件 $S$，若隨機化演算法 $M$ 滿足下式，則它是 $(\varepsilon, \delta)$-DP：

```math
P(M(D) \in S) \le e^{\varepsilon} \cdot P(M(D') \in S) + \delta.
```

解讀：輸出分佈足夠接近（由 ε 參數化），使得任何單一個人的貢獻都無法被可靠地推斷出來，除非發生機率為 δ 的例外。

### DP-SGD

Abadi et al. 2016。標準做法：
1. 抽樣一個 mini-batch。
2. 計算每個樣本的 gradient。
3. 把每個樣本的 gradient 裁剪到門檻 C。
4. 加總裁剪後的 gradient，並加入標準差為 $\sigma \cdot C$ 的高斯雜訊。
5. 用含雜訊的總和更新參數。

隱私成本由 accountant（Moments Accountant、Rényi DP accountant）追蹤。LLM 文獻中回報的 ε 值，依威脅模型、資料敏感度與效用目標差異很大；並沒有普遍「安全」的預設 ε。已發表的例子在某些 LLM 訓練設定中大約涵蓋 ε ≈ 1 到 10，但這些只是示範，不是建議的預設值。較低的 ε 通常需要更多雜訊，並可能增加效用損失。

### LoRA + DP-SGD

對 frontier 模型做完整的 DP-SGD 成本過高。LoRA (Hu et al. 2022) 把 gradient 更新限制在小的 adapter 上，減少每個樣本 gradient 的儲存量。LoRA + DP-SGD 是 2025 年的常見配置。DP 保證適用於 adapter；基礎模型保持固定。

### 2024-2025 年的拉扯

兩條證據線：

- **Canary MIA (Duan et al. 2024)。** 把獨特的 canary 插入訓練資料，測量 membership-inference 攻擊者能否辨認它們。回報對語言模型的成功有限。暗示 MIA 很難。
- **訓練資料萃取 (Carlini 2021, Nasr et al. 2025)。** 用前綴提示模型；測量它是否能還原訓練中的逐字文字。回報大量記憶。暗示在相關意義上 MIA 很容易。

2025 年 3 月的調和 (arXiv:2503.06808)：兩者測量的是不同的東西。MIA 問的是「樣本 e 在 D 裡面嗎？」，對象是被植入的 canary。萃取問的是「我能還原 D 的什麼？」。「最容易被萃取」的樣本才是隱私上重要的；canary 低估了這點，因為它們並未被最佳化成容易被萃取。

新的 canary 設計。不需要 shadow model 的 loss-based MIA。對一個以真實資料訓練、具備實際 DP 保證的 LLM，做出第一次非平凡的 DP 稽核。

### DP 訓練的替代方案

- **PMixED (arXiv:2403.15638)。** 推論時的 private prediction。在 next-token 分佈上做 mixture of experts；每個 expert 只看到訓練資料的一個 shard；聚合時加入雜訊以達成 DP。完全避開 DP 訓練。
- **DP 合成資料生成 (Google Research 2024)。** 用 DP-SGD 對 LoRA 做 fine-tune，取樣合成資料，再用合成資料訓練下游分類器。

兩者都避開了完整 DP 訓練的效用成本，代價是採用不同的威脅模型。

### Differential Privacy Reversal via LLM Feedback

2025 年新興的攻擊。把經 DP 訓練的模型的信心分數當作 oracle，來重新識別個人。即使輸出本身沒有外洩，信心分佈也可能外洩。

防禦：不要暴露信心分數，或在暴露前先截斷 / 量化它們。這是 (ε, δ)-DP 訓練之外的額外要求。

### 這在 Phase 18 中的位置

第 20 到 21 課是 bias/fairness。第 22 課是隱私。第 23 課是透過 watermarking 達成的來源追溯。第 27 課涵蓋監管上的資料來源層。

```figure
an-dp-clip-noise
```

## 實際使用 (Use It)

`code/main.py` 在一個玩具二元分類資料集上模擬 DP-SGD。你可以掃描雜訊乘數 σ 與裁剪範數 C，並追蹤 (ε, δ) 預算與準確率成本。一個「canary 攻擊」會插入獨特的訓練樣本，並測量 log-loss 測試能否在 DP 之前與之後偵測到它。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-dp-audit.md`。給它一個語言模型部署上的 DP 主張，它會稽核：(ε, δ) 值、所用的 accountant、MIA 評估協定，以及是否評估過信心暴露向量。

## 練習 (Exercises)

1. 執行 `code/main.py`。掃描 $\sigma \in \lbrace 0.5, 1.0, 2.0 \rbrace$，並回報 (ε, δ) 與準確率的取捨。找出效用崩潰的那一點。

2. 實作 canary 插入與 log-loss 測試。測量在 $\sigma = 1.0$ 時，DP-SGD 前後的偵測率。

3. 閱讀 Nasr et al. 2025 關於訓練資料萃取的研究。為什麼萃取的成功在中等 ε 下沒有崩潰？這對把 MIA 當作評估有什麼含意？

4. 設計一個完全在推論時運作、使用 PMixED (arXiv:2403.15638) 的部署。PMixED 處理了 DP-SGD 沒有處理的什麼威脅模型？

5. 勾勒 DP Reversal via LLM Feedback 攻擊。設計一個限制信心分數外洩的對策，並估計其部署成本。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| DP | 「(ε, δ)-differential privacy」 | 形式化的隱私：在相鄰資料集變動下，輸出分佈保持接近 |
| DP-SGD | 「noise-injected SGD」 | Gradient 裁剪加上高斯雜訊；標準的 DP 訓練 |
| LoRA + DP-SGD | 「efficient private fine-tune」 | 在低秩 adapter 上做 DP-SGD；2025 年的標準配置 |
| MIA | 「membership inference」 | 判斷某個樣本是否曾在訓練資料中的攻擊 |
| Canary | 「inserted watermark example」 | 用來測量 DP 外洩的獨特訓練樣本 |
| PMixED | 「private inference mixture」 | 透過 next-token 分佈上的 mixture-of-experts，在推論時達成 DP |
| DP Reversal | 「confidence leakage attack」 | 把模型的信心當作 re-identification oracle 的攻擊 |

## 延伸閱讀 (Further Reading)

- [Abadi et al. — DP-SGD (arXiv:1607.00133)](https://arxiv.org/abs/1607.00133)：標準的 DP 訓練演算法
- [Carlini et al. — Extracting Training Data (arXiv:2012.07805)](https://arxiv.org/abs/2012.07805)：經典的萃取論文
- [Duan et al. — Canary MIA on LLMs (arXiv:2402.07841, 2024)](https://arxiv.org/abs/2402.07841)：成功有限的 MIA
- [Kowalczyk et al. — Auditing DP for LLMs (arXiv:2503.06808, March 2025)](https://arxiv.org/abs/2503.06808)：拉扯的調和
- [PMixED (arXiv:2403.15638)](https://arxiv.org/abs/2403.15638)：推論時的 private prediction
