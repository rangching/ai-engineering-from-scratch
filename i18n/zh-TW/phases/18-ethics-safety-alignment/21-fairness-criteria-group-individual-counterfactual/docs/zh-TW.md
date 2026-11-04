# Fairness 準則：群體、個體、反事實 (Fairness Criteria — Group, Individual, Counterfactual)

> 三個家族構成了 fairness 文獻的骨架。群體公平 (Group fairness)：demographic parity、equalized odds、conditional use accuracy equality，也就是受保護群體之間平均而言有相同的比率。個體公平 (Individual fairness, Dwork et al. 2012)：相似的個體得到相似的決策；對決策映射施加 Lipschitz 條件。反事實公平 (Counterfactual fairness, Kusner et al. 2017)：若敏感屬性在反事實下被改變，決策不變，則該決策對個人是公平的。2024 年的理論結果 (NeurIPS 2024)：CF 與準確率之間存在固有的取捨；一個與模型無關的方法，可以把最佳但不公平的預測器轉換成 CF 預測器，且準確率損失有上界。Backtracking counterfactuals (arXiv:2401.13935, January 2024)：一種新典範，避免需要對受法律保護的屬性做介入。哲學上的調和 (ICLR Blogposts 2024)：有了因果圖，滿足特定的群體公平度量就蘊含了反事實公平。

**Type:** Learn
**Languages:** Python (stdlib, three-criteria comparison)
**Prerequisites:** Phase 18 · 20 (bias), Phase 02 (classical ML)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出三個群體公平準則（demographic parity、equalized odds、conditional use accuracy equality）與一個不可能性結果。
- 透過 Dwork et al. 2012 的 Lipschitz 公式化描述個體公平。
- 描述反事實公平及其對因果圖的依賴。
- 解釋 backtracking counterfactuals，以及它為什麼能避開對受保護屬性做介入的問題。

## 問題 (The Problem)

第 20 課談的是測量 bias。第 21 課談的是定義這個測量應該服務的 fairness 標準。這三個家族給出結構上不同的標準：模型可以是群體公平卻個體不公平，也可以是反事實公平卻群體不公平。選擇標準是政策決定；沒有任何標準是普遍最佳的。

## 核心概念 (The Concept)

### 群體公平 (Group fairness)

- **Demographic parity。** 對所有群體，P(Y=1 | A=a) = P(Y=1 | A=a')。接受率相等。
- **Equalized odds。** P(Y=1 | Y*=y, A=a) = P(Y=1 | Y*=y, A=a')。各群體的 TPR 與 FPR 相等。
- **Conditional use accuracy equality。** P(Y*=y | Y=y, A=a) = P(Y*=y | Y=y, A=a')。各群體的預測值相等。

不可能性 (Chouldechova, Kleinberg-Mullainathan-Raghavan 2017)：在基礎比率不相等時，這三者無法同時滿足。

### 個體公平 (Individual fairness)

Dwork et al. 2012。對於某個任務專屬的相似度度量 d，如果存在某個 Lipschitz 常數 L，使得 |f(x) - f(x')| <= L * d(x, x')，則決策映射 f 在個體層面是公平的。相似的個體得到相似的決策。

需要定義 d。這是政策問題，不是統計問題。

### 反事實公平 (Counterfactual fairness)

Kusner et al. 2017。如果在族群的因果模型下，個體 i 的敏感屬性在反事實下被改變時決策不變，則該決策對個體 i 是反事實公平的。

需要一個因果 DAG。這個 DAG 是建模上的選擇。反事實公平的正當性，只和 DAG 一樣強。

### CF 與準確率的取捨

NeurIPS 2024 的理論結果：反事實公平與預測準確率之間存在固有的取捨。一個與模型無關的方法，可以把最佳但不公平的預測器轉換成 CF 預測器，且準確率成本有上界。準確率成本取決於最佳不公平預測器中敏感屬性係數的大小。

### Backtracking counterfactuals

arXiv:2401.13935 (January 2024)。傳統反事實需要對敏感屬性做介入，也就是「如果這個人是不同性別，決策會改變嗎」。在法律上，這有問題：分類法規中，受保護屬性不能被介入。

Backtracking counterfactuals 把方向反過來：不對屬性做介入，而是問這個人實際特徵的哪種組合，會產生反事實的結果。這避開了法律上的反對意見。

### 哲學上的調和

ICLR Blogposts 2024。有了因果圖，滿足特定的群體公平度量就蘊含了反事實公平。這三個家族並不是正交的；它們是同一個底層因果結構的不同面向。

這並未解決不可能性定理（基礎比率不相等時，仍然無法同時達成群體公平）。但它顯示，「群體」與「個體 / 反事實」之間表面上的對立，有一部分是因為沒有明確說出因果模型而造成的假象。

### 這在 Phase 18 中的位置

第 20 課是 bias 測量。第 21 課是 fairness 定義。第 22 課是隱私（differential privacy）。第 23 課是 watermarking。這些是與分配相關的課程，補充與欺騙相關的第 7 到 11 課。

```figure
an-fairness-trilemma
```

## 實際使用 (Use It)

`code/main.py` 建立一個玩具二元分類資料集，含有敏感屬性與不相等的基礎比率。在一個簡單的分類器上計算 demographic parity、equalized odds 與 conditional use accuracy equality。觀察這三個指標彼此不一致。為 demographic parity 套用重新加權，並觀察它對另外兩個指標的代價。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-fairness-criterion.md`。給它一個 fairness 主張或政策，它會找出所主張的是哪個準則、在所宣稱的不相等基礎比率下，模型是否還能滿足其餘準則，以及該主張依賴什麼因果 DAG。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報預設資料上的三個群體指標。套用以 demographic parity 為目標的重新加權，並重新回報。

2. 用非敏感特徵上的 L2 實作 Dwork et al. 2012 的個體公平度量。回報有多少配對在 L=1 時違反 Lipschitz。

3. 閱讀 Kusner et al. 2017。為履歷評分建構一個簡單的兩特徵因果 DAG，並找出它所蘊含的反事實公平條件。

4. 2024 年的 backtracking-counterfactuals 論文避免對受保護屬性做介入。描述一個這點對法規遵循很重要的情境。

5. ICLR 2024 的調和主張群體公平與反事實公平是同一結構的不同面向。從 `code/main.py` 的三個準則中挑兩個，並說明能讓它們等價的因果假設。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Demographic parity | 「equal rates」 | P(Y=1 | A=a) 在各群體間相等 |
| Equalized odds | 「equal TPR/FPR」 | 各群體間的 true-positive 與 false-positive 比率相等 |
| Conditional use accuracy | 「equal PPV/NPV」 | 各群體間的預測值相等 |
| Individual fairness | 「Lipschitz condition」 | 相似的個體得到相似的決策 |
| Counterfactual fairness | 「causal alteration invariance」 | 在反事實屬性改變下決策不變 |
| Backtracking counterfactual | 「explain via actuals」 | 從結果往回推理的反事實，而不是從屬性往前推理 |
| Impossibility theorem | 「the three conflict」 | Chouldechova / KMR 2017：基礎比率不相等時，群體準則互斥 |

## 延伸閱讀 (Further Reading)

- [Dwork et al. — Fairness through Awareness (arXiv:1104.3913)](https://arxiv.org/abs/1104.3913)：個體公平
- [Kusner, Loftus, Russell, Silva — Counterfactual Fairness (arXiv:1703.06856)](https://arxiv.org/abs/1703.06856)：反事實公平
- [Chouldechova — Fair prediction with disparate impact (arXiv:1703.00056)](https://arxiv.org/abs/1703.00056)：不可能性
- [Backtracking Counterfactuals (arXiv:2401.13935)](https://arxiv.org/abs/2401.13935)：處理受保護屬性介入的新典範
