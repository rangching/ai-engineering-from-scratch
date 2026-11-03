# 機器學習的統計學 (Statistics for Machine Learning)

> 統計學讓你知道，你的模型是真的有效，還是只是運氣好。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06 (Probability and Distributions), 07 (Bayes' Theorem)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 從零計算描述性統計、Pearson/Spearman correlation 與 covariance matrix
- 執行假設檢定（t-test、chi-squared），並正確解讀 p-value 與 confidence interval
- 用 bootstrap resampling，在不做任何分佈假設的情況下，為任何指標建構 confidence interval
- 用 effect size 衡量，區分統計顯著性與實務顯著性

## 問題 (The Problem)

你訓練了兩個模型。Model A 在測試集上得分 0.87。Model B 得分 0.89。你部署了 Model B。三週後，正式環境的指標比以前更差。發生了什麼事？

Model B 其實並沒有勝過 Model A。那 0.02 的差距只是雜訊。你的測試集太小，或 variance 太高，或兩者皆是。你交付了偽裝成進步的隨機性。

這種事一直發生。Kaggle 排行榜大洗牌。無法重現的論文。只根據幾百個樣本就宣布勝者的 A/B test。根本原因永遠相同：有人跳過了統計學。

統計學給你工具，讓你分辨訊號與雜訊。它告訴你一個差異何時是真的、你該有多大的信心，以及在你能信任一個結果之前需要多少資料。每個 ML pipeline、每次模型比較、每個實驗，都需要統計學。沒有它，你就是在猜。

## 核心概念 (The Concept)

### 描述性統計：總結你的資料 (Descriptive Statistics: Summarizing Your Data)

在你建模任何東西之前，你需要知道資料長什麼樣子。描述性統計把資料集壓縮成幾個能捕捉其形狀的數字。

**集中趨勢的度量** 回答「中間在哪裡？」

```
Mean:   sum of all values / count
        mu = (1/n) * sum(x_i)

Median: middle value when sorted
        Robust to outliers. If you have [1, 2, 3, 4, 1000], the mean is 202
        but the median is 3.

Mode:   most frequent value
        Useful for categorical data. For continuous data, rarely informative.
```

Mean 是平衡點。Median 是中間點。當兩者分歧時，你的分佈是偏斜的。收入分佈的 $\text{mean} \gg \text{median}$（因為億萬富翁造成右偏）。訓練期間的 loss 分佈常常是 $\text{mean} \ll \text{median}$（因為簡單樣本造成左偏）。

**離散程度的度量** 回答「資料有多分散？」

```
Variance:   average squared deviation from the mean
            sigma^2 = (1/n) * sum((x_i - mu)^2)

Standard deviation:  square root of variance
                     sigma = sqrt(sigma^2)
                     Same units as the data, so more interpretable.

Range:      max - min
            Sensitive to outliers. Almost never useful alone.

IQR:        Q3 - Q1 (interquartile range)
            The range of the middle 50% of the data.
            Robust to outliers. Used for box plots and outlier detection.
```

**Percentile** 把排序後的資料分成 100 等份。第 25 百分位數 (Q1) 代表有 25% 的值落在這一點以下。第 50 百分位數是 median。第 75 百分位數是 Q3。

```
For latency monitoring:
  P50 = median latency        (typical user experience)
  P95 = 95th percentile       (bad but not worst case)
  P99 = 99th percentile       (tail latency, often 10x the median)
```

在 ML 裡，你會在推論延遲、預測信心分佈，以及理解誤差分佈時關心 percentile。一個平均誤差很低、但 P99 誤差很糟糕的模型，可能對安全攸關的應用毫無用處。

**樣本與母體統計量。** 從樣本計算 variance 時，要除以 $(n-1)$ 而不是 $n$。這稱為 Bessel's correction。它補償了你的樣本 mean 並非真正的母體 mean 這個事實。分母用 $n$，你會系統性地低估真實的 variance。用 $(n-1)$，估計就是 unbiased 的。

```math
\begin{aligned}
\text{Population variance:} \quad \sigma^2 &= \frac{1}{N} \sum (x_i - \mu)^2 \\
\text{Sample variance:} \quad s^2 &= \frac{1}{n-1} \sum (x_i - \bar{x})^2
\end{aligned}
```

實務上：如果 $n$ 很大（數千個樣本），差異可以忽略。如果 $n$ 很小（幾十個樣本），就很重要。

### Correlation：變數如何一起變動 (Correlation: How Variables Move Together)

Correlation 衡量兩個變數之間線性關係的強度與方向。

**Pearson correlation coefficient** 衡量線性關聯：

```
r = sum((x_i - x_bar)(y_i - y_bar)) / (n * s_x * s_y)

r = +1:  perfect positive linear relationship
r = -1:  perfect negative linear relationship
r =  0:  no linear relationship (but there might be a nonlinear one!)

Range: [-1, 1]
```

Pearson 假設關係是線性的，而且兩個變數大致呈 normal 分佈。它對 outlier 很敏感。單一個極端點就能把 $r$ 從 0.1 拉到 0.9。

**Spearman rank correlation** 衡量單調 (monotonic) 關聯：

```
1. Replace each value with its rank (1, 2, 3, ...)
2. Compute Pearson correlation on the ranks

Spearman catches any monotonic relationship, not just linear.
If y = x^3, Pearson gives r < 1 but Spearman gives rho = 1.
```

**何時使用哪一個：**

```
Pearson:    Both variables are continuous and roughly normal.
            You care about the linear relationship specifically.
            No extreme outliers.

Spearman:   Ordinal data (rankings, ratings).
            Data is not normally distributed.
            You suspect a monotonic but not linear relationship.
            Outliers are present.
```

**黃金法則：** correlation 不代表 causation。冰淇淋銷量與溺水死亡人數是相關的，因為兩者都在夏天增加。你的模型 accuracy 與參數數量是相關的，但增加參數並不會自動提升 accuracy（見：overfitting）。

### Covariance Matrix (Covariance Matrix)

兩個變數之間的 covariance 衡量它們如何一起變動：

```
Cov(X, Y) = (1/n) * sum((x_i - x_bar)(y_i - y_bar))

Cov(X, Y) > 0:  X and Y tend to increase together
Cov(X, Y) < 0:  when X increases, Y tends to decrease
Cov(X, Y) = 0:  no linear co-movement
```

對 $d$ 個 feature，covariance matrix $C$ 是一個 $d \times d$ 的 matrix，其中 $C[i][j] = \mathop{\mathrm{Cov}}(\text{feature}_i, \text{feature}_j)$。對角線元素 $C[i][i]$ 是每個 feature 的 variance。

```
C = | Var(x1)      Cov(x1,x2)  Cov(x1,x3) |
    | Cov(x2,x1)  Var(x2)      Cov(x2,x3) |
    | Cov(x3,x1)  Cov(x3,x2)  Var(x3)     |

Properties:
  - Symmetric: C[i][j] = C[j][i]
  - Positive semi-definite: all eigenvalues >= 0
  - Diagonal = variances
  - Off-diagonal = covariances
```

**與 PCA 的關聯。** PCA 對 covariance matrix 做 eigendecomposition。Eigenvector 是 principal component（variance 最大的方向）。Eigenvalue 告訴你每個 component 涵蓋多少 variance。這正是 Lesson 10 涵蓋的內容，但現在你看到了為什麼 covariance matrix 是該拿來分解的對象：它編碼了資料中所有成對的線性關係。

**與 correlation 的關聯。** Correlation matrix 是標準化變數（每個都除以其標準差）的 covariance matrix。Correlation 把 covariance normalize，讓所有值都落在 $[-1, 1]$。

### 假設檢定 (Hypothesis Testing)

假設檢定是在不確定性下做決策的框架。你從一個主張開始，收集資料，然後判斷資料是否與該主張一致。

**設定：**

```
Null hypothesis (H0):        the default assumption, usually "no effect"
Alternative hypothesis (H1): what you are trying to show

Example:
  H0: Model A and Model B have the same accuracy
  H1: Model B has higher accuracy than Model A
```

**p-value** 是在假設 H0 為真的情況下，看到與你觀察到的資料一樣極端的資料的機率。它不是 H0 為真的機率。這是統計學中最常見的單一誤解。

```
p-value = P(data this extreme | H0 is true)

If p-value < alpha (typically 0.05):
    Reject H0. The result is "statistically significant."
If p-value >= alpha:
    Fail to reject H0. You do not have enough evidence.
    This does NOT mean H0 is true.
```

**Confidence interval** 給出參數的一個合理數值範圍：

```
95% confidence interval for the mean:
    x_bar +/- z * (s / sqrt(n))

where z = 1.96 for 95% confidence

Interpretation: if you repeated this experiment many times, 95% of the
computed intervals would contain the true mean. It does NOT mean there
is a 95% probability the true mean is in this specific interval.
```

Confidence interval 的寬度告訴你精確度。寬的區間代表高度的不確定性。窄的區間代表你的估計很精確（但如果你的資料有偏差，則不一定準確）。

### t-test (The t-test)

t-test 比較 mean。有幾種變體。

**One-sample t-test：** 母體 mean 是否不同於某個假設值？

```math
\begin{aligned}
t &= \frac{\bar{x} - \mu_0}{s / \sqrt{n}} \\[4pt]
\text{degrees of freedom} &= n - 1
\end{aligned}
```

**Two-sample t-test（獨立）：** 兩組的 mean 是否不同？

```
t = (x_bar_1 - x_bar_2) / sqrt(s1^2/n1 + s2^2/n2)

This is Welch's t-test, which does not assume equal variances.
Always use Welch's unless you have a specific reason for equal variances.
```

**Paired t-test：** 當測量成對出現時（同一個模型在相同的資料切分上評估）：

```
Compute d_i = x_i - y_i for each pair
Then run a one-sample t-test on the d_i values against mu_0 = 0
```

在 ML 裡，paired t-test 很常見：你在相同的 10 個 cross-validation fold 上執行兩個模型，並成對比較它們的分數。

### Chi-squared Test (Chi-squared Test)

Chi-squared test 檢查觀察到的頻率是否符合預期的頻率。適用於類別型資料。

```
chi^2 = sum((observed - expected)^2 / expected)

Example: does a language model's output distribution match the
training distribution across categories?

Category    Observed   Expected
Positive       120        100
Negative        80        100
chi^2 = (120-100)^2/100 + (80-100)^2/100 = 4 + 4 = 8

With 1 degree of freedom, chi^2 = 8 gives p < 0.005.
The difference is significant.
```

### ML 模型的 A/B Testing (A/B Testing for ML Models)

ML 中的 A/B testing 與網頁的 A/B testing 不同。模型比較有其特定的挑戰：

```
1. Same test set:    Both models must be evaluated on identical data.
                     Different test sets make comparison meaningless.

2. Multiple metrics: Accuracy alone is not enough. You need precision,
                     recall, F1, latency, and fairness metrics.

3. Variance:         Use cross-validation or bootstrap to estimate
                     the variance of each metric, not just point estimates.

4. Data leakage:     If the test set was used during model selection,
                     your comparison is biased. Hold out a final test set.
```

**流程：**

```
1. Define your metric and significance level (alpha = 0.05)
2. Run both models on the same k-fold cross-validation splits
3. Collect paired scores: [(a1, b1), (a2, b2), ..., (ak, bk)]
4. Compute differences: d_i = b_i - a_i
5. Run a paired t-test on the differences
6. Check: is the mean difference significantly different from 0?
7. Compute a confidence interval for the mean difference
8. Compute effect size (Cohen's d) to judge practical significance
```

### 統計顯著性與實務顯著性 (Statistical Significance vs Practical Significance)

一個結果可能在統計上顯著，但在實務上毫無意義。只要資料夠多，即使是微不足道的差異也會變得統計上顯著。

```
Example:
  Model A accuracy: 0.9234
  Model B accuracy: 0.9237
  n = 1,000,000 test samples
  p-value = 0.001

Statistically significant? Yes.
Practically significant? A 0.03% improvement is not worth the
engineering cost of deploying a new model.
```

**Effect size** 量化差異有多大，與樣本大小無關：

```math
\begin{aligned}
\text{Cohen's } d &= \frac{\text{mean}_1 - \text{mean}_2}{\text{pooled\_std}} \\[4pt]
d &= 0.2: \quad \text{small effect} \\
d &= 0.5: \quad \text{medium effect} \\
d &= 0.8: \quad \text{large effect}
\end{aligned}
```

永遠同時報告 p-value 和 effect size。p-value 告訴你差異是不是真的。Effect size 告訴你它重不重要。

### 多重比較問題 (Multiple Comparison Problem)

當你檢定很多個假設時，有些會因為偶然而「顯著」。如果你在 $\alpha = 0.05$ 下檢定 20 件事，即使什麼都不是真的，你也預期會有 1 個偽陽性。

```
P(at least one false positive) = 1 - (1 - alpha)^m

m = 20 tests, alpha = 0.05:
P(false positive) = 1 - 0.95^20 = 0.64

You have a 64% chance of at least one false positive.
```

**Bonferroni correction：** 把 $\alpha$ 除以檢定的次數。

```
Adjusted alpha = alpha / m = 0.05 / 20 = 0.0025

Only reject H0 if p-value < 0.0025.
Conservative but simple. Works when tests are independent.
```

在 ML 裡，當你跨多個指標比較一個模型、檢定許多 hyperparameter 設定，或在多個資料集上評估時，這就很重要。

### Bootstrap 方法 (Bootstrap Methods)

Bootstrap 透過有放回地對資料重新抽樣，來估計統計量的抽樣分佈。不需要對底層分佈做任何假設。

**演算法：**

```
1. You have n data points
2. Draw n samples WITH replacement (some points appear multiple times,
   some not at all)
3. Compute your statistic on this bootstrap sample
4. Repeat B times (typically B = 1000 to 10000)
5. The distribution of bootstrap statistics approximates the
   sampling distribution
```

**Bootstrap confidence interval（percentile 方法）：**

```
Sort the B bootstrap statistics
95% CI = [2.5th percentile, 97.5th percentile]
```

**為什麼 bootstrap 對 ML 很重要：**

```
- Test set accuracy is a point estimate. Bootstrap gives you
  confidence intervals.
- You cannot assume metric distributions are normal (especially
  for AUC, F1, precision at k).
- Bootstrap works for ANY statistic: median, ratio of two means,
  difference in AUC between two models.
- No closed-form formula needed.
```

**用 bootstrap 做模型比較：**

```
1. You have predictions from Model A and Model B on the same test set
2. For each bootstrap iteration:
   a. Resample test indices with replacement
   b. Compute metric_A and metric_B on the resampled set
   c. Store diff = metric_B - metric_A
3. 95% CI for the difference:
   [2.5th percentile of diffs, 97.5th percentile of diffs]
4. If the CI does not contain 0, the difference is significant
```

這比 paired t-test 更穩健，因為它不做任何分佈假設。

### Parametric 與 Non-parametric 檢定 (Parametric vs Non-parametric Tests)

**Parametric 檢定** 假設特定的分佈（通常是 normal）：

```
t-test:         assumes normally distributed data (or large n by CLT)
ANOVA:          assumes normality and equal variances
Pearson r:      assumes bivariate normality
```

**Non-parametric 檢定** 不做任何分佈假設：

```
Mann-Whitney U:     compares two groups (replaces independent t-test)
Wilcoxon signed-rank: compares paired data (replaces paired t-test)
Spearman rho:       correlation on ranks (replaces Pearson)
Kruskal-Wallis:     compares multiple groups (replaces ANOVA)
```

**何時使用 non-parametric：**

```
- Small sample size (n < 30) and data is clearly non-normal
- Ordinal data (ratings, rankings)
- Heavy outliers you cannot remove
- Skewed distributions
```

**何時使用 parametric：**

```
- Large sample size (CLT makes the test statistic approximately normal)
- Data is roughly symmetric without extreme outliers
- More statistical power (better at detecting real differences)
```

在 ML 實驗中，你通常 $n$ 很小（5 或 10 個 cross-validation fold），所以像 Wilcoxon signed-rank 這類 non-parametric 檢定，往往比 t-test 更合適。

### Central Limit Theorem：實務上的意涵 (Central Limit Theorem: Practical Implications)

CLT 說明，不論底層母體的分佈為何，當 $n$ 增大時，樣本 mean 的分佈會趨近 normal 分佈。

```
If X_1, X_2, ..., X_n are iid with mean mu and variance sigma^2:

    X_bar ~ Normal(mu, sigma^2 / n)    as n -> infinity

Works for n >= 30 in most cases.
For highly skewed distributions, you might need n >= 100.
```

**為什麼這對 ML 很重要：**

```
1. Justifies confidence intervals and t-tests on aggregated metrics
2. Explains why averaging over cross-validation folds gives stable
   estimates even when individual folds vary wildly
3. Mini-batch gradient descent works because the average gradient
   over a batch approximates the true gradient (CLT in action)
4. Ensemble methods: averaging predictions from many models gives
   more stable output than any single model
```

**CLT 不做的事：**

```
- Does NOT make your data normal. It makes the MEAN of samples normal.
- Does NOT work for heavy-tailed distributions with infinite variance
  (Cauchy distribution).
- Does NOT apply to dependent data (time series without correction).
```

### ML 論文中常見的統計錯誤 (Common Statistical Mistakes in ML Papers)

1. **在訓練集上測試。** 保證會 overfitting。一律要保留模型在訓練期間從未見過的資料。

2. **沒有 confidence interval。** 只報告單一個 accuracy 數字而不附帶不確定性，會讓結果無法重現、無法驗證。

3. **忽略多重比較。** 檢定 50 種設定，然後不經修正就只報告最好的那個，會拉高偽陽性率。

4. **混淆統計顯著性與實務顯著性。** 對 0.01% 的 accuracy 提升給出 p-value 0.001，並沒有意義。

5. **在不平衡資料上使用 accuracy。** 在 99% 為負類別的資料集上達到 99% accuracy，代表模型什麼都沒學到。請使用 precision、recall、F1 或 AUC。

6. **挑選指標 (cherry-picking)。** 只報告你的模型獲勝的那個指標。誠實的評估會報告所有相關的指標。

7. **在 train/test 切分之間洩漏資訊。** 在切分之前做 normalization，或用未來的資料預測過去。

8. **測試集很小，又沒有 variance 估計。** 在 100 個樣本上評估，就宣稱有 2% 的進步，那是雜訊，不是訊號。

9. **在資料並非獨立時假設獨立。** 同一位病人的醫療影像、同一份文件中的多個句子。同一組內的觀察值是相關的。

10. **P-hacking。** 嘗試不同的檢定、子集或排除準則，直到得到 $p < 0.05$。這樣的結果只是搜尋過程的產物。

## 動手實作 (Building It)

你將實作：

1. **從零寫描述性統計**（mean、median、mode、標準差、percentile、IQR）
2. **Correlation 函式**（Pearson 與 Spearman，加上 covariance matrix）
3. **假設檢定**（one-sample t-test、two-sample t-test、chi-squared test）
4. **Bootstrap confidence interval**（適用於任何統計量，不需要任何假設）
5. **A/B test 模擬器**（產生資料、檢定、檢查 Type I 與 Type II 錯誤）
6. **統計顯著性與實務顯著性的示範**（展示大的 $n$ 會讓一切都「顯著」）

全部從零開始，只用 `math` 和 `random`。不用 numpy，不用 scipy。

```figure
f3-bootstrap-resample
```

## 關鍵術語 (Key Terms)

| 術語 | 定義 |
|---|---|
| Mean | 數值總和除以個數。對 outlier 敏感。 |
| Median | 排序後資料的中間值。對 outlier 穩健。 |
| Standard deviation | Variance 的平方根。以原始單位衡量離散程度。 |
| Percentile | 有給定百分比的資料落在其下的值。 |
| IQR | Interquartile range。Q3 減 Q1。中間 50% 的離散程度。 |
| Pearson correlation | 衡量兩個變數之間的線性關聯。範圍 $[-1, 1]$。 |
| Spearman correlation | 使用排名衡量單調關聯。 |
| Covariance matrix | 所有 feature 之間成對 covariance 組成的 matrix。 |
| Null hypothesis | 預設假設：沒有效果或沒有差異。 |
| p-value | 在虛無假設為真的情況下，資料如此極端的機率。 |
| Confidence interval | 在特定信心水準下，參數的合理數值範圍。 |
| t-test | 檢定 mean 是否有顯著差異。使用 t-distribution。 |
| Chi-squared test | 檢定觀察到的頻率是否不同於預期的頻率。 |
| Effect size | 差異的大小，與樣本大小無關。常用 Cohen's d。 |
| Bonferroni correction | 把顯著性門檻除以檢定次數，以控制偽陽性。 |
| Bootstrap | 有放回地重新抽樣，以估計抽樣分佈。 |
| Type I error | 偽陽性。在 H0 為真時拒絕 H0。 |
| Type II error | 偽陰性。在 H0 為假時未能拒絕 H0。 |
| Statistical power | 正確拒絕錯誤 H0 的機率。Power = 1 減去 Type II 錯誤率。 |
| Central limit theorem | 隨著樣本大小增加，樣本 mean 會收斂到 normal 分佈。 |
| Parametric test | 假設資料有特定的分佈（通常是 normal）。 |
| Non-parametric test | 不做分佈假設。作用在排名或正負號上。 |
