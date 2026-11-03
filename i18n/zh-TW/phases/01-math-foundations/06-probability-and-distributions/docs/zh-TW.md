# 機率與分佈 (Probability and Distributions)

> 機率是 AI 用來表達不確定性的語言。

**Type:** Learn
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 從零實作 Bernoulli、categorical、Poisson、uniform 與 normal 分佈的 PMF 與 PDF
- 計算 expected value、variance，並用 Central Limit Theorem 解釋為什麼 Gaussian 無所不在
- 用數值穩定技巧（減去最大的 logit）打造 softmax 與 log-softmax 函式
- 從 logits 計算 cross-entropy loss，並把它連結到 negative log-likelihood

## 問題 (The Problem)

分類器輸出 `[0.03, 0.91, 0.06]`。語言模型從 50,000 個候選詞中挑出下一個詞。Diffusion model 從學到的分佈中取樣來生成圖片。這些全都是機率的實際應用。

模型做的每個預測都是一個機率分佈。每個 loss function 衡量的是預測分佈離真實分佈有多遠。每一步訓練，都是在調整參數，讓一個分佈更像另一個分佈。不懂機率，你連一篇 ML 論文都讀不了、一個模型都除不了錯，也搞不懂為什麼你的訓練 loss 是 NaN。

## 核心概念 (The Concept)

### 事件、樣本空間與機率 (Events, Sample Spaces, and Probability)

樣本空間 (sample space) $S$ 是所有可能結果的集合。事件 (event) 是樣本空間的子集合。機率把事件對應到 0 到 1 之間的數字。

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

三條公理定義了整個機率論：
1. 對任何事件 $A$，$P(A) \ge 0$
2. $P(S) = 1$（總有事情會發生）
3. 當 $A$ 與 $B$ 不可能同時發生時，$P(A \text{ or } B) = P(A) + P(B)$

其他一切（Bayes' theorem、期望值、分佈）都從這三條規則推導出來。

### 條件機率與獨立性 (Conditional Probability and Independence)

$P(A \mid B)$ 是在 $B$ 已經發生的條件下，$A$ 發生的機率。

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

當知道其中一個事件完全不會告訴你另一個事件的資訊時，兩個事件是獨立的：

```math
\begin{aligned}
\text{Independent:} \quad & P(A \mid B) = P(A) \\
\text{Equivalent to:} \quad & P(A \text{ and } B) = P(A) \cdot P(B)
\end{aligned}
```

擲硬幣是獨立的。不放回地抽牌則不是。

### Probability Mass Function 與 Probability Density Function (Probability Mass Functions vs Probability Density Functions)

離散隨機變數有 probability mass function (PMF)。每個結果都有一個明確的機率，可以直接讀出來。

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

連續隨機變數有 probability density function (PDF)。單一點上的密度不是機率。機率來自對一個區間積分密度。

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

這個差別在 ML 裡很重要。分類的輸出是 PMF（離散的選擇）。VAE 的 latent space 用 PDF（連續的）。

### 常見的分佈 (Common Distributions)

**Bernoulli：** 一次試驗，兩種結果。用來建模二元分類。

```math
\begin{aligned}
P(X = 1) &= p \\
P(X = 0) &= 1 - p \\
\text{Mean} = p, &\quad \text{Variance} = p(1-p)
\end{aligned}
```

**Categorical：** 一次試驗，$k$ 種結果。用來建模多類別分類（softmax 輸出）。

```math
\begin{aligned}
& P(X = i) = p_i, \quad \text{where } \textstyle\sum_i p_i = 1 \\
& \text{Example: } P(\text{cat}) = 0.7, \quad P(\text{dog}) = 0.2, \quad P(\text{bird}) = 0.1
\end{aligned}
```

**Uniform：** 所有結果的可能性相同。用於隨機初始化。

```math
\begin{aligned}
\text{Discrete:} \quad & P(X = k) = 1/n \text{ for } k \in \lbrace 1, \ldots, n \rbrace \\
\text{Continuous:} \quad & f(x) = 1/(b-a) \text{ for } x \in [a, b]
\end{aligned}
```

**Normal (Gaussian)：** 鐘形曲線。由 mean ($\mu$) 與 variance ($\sigma^2$) 決定參數。

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson：** 固定區間內罕見事件的次數。用來建模事件發生率。

```math
\begin{aligned}
P(X = k) &= \frac{\lambda^k e^{-\lambda}}{k!} \\
\text{Mean} = \lambda, &\quad \text{Variance} = \lambda
\end{aligned}
```

### Expected Value 與 Variance (Expected Value and Variance)

Expected value 是加權平均的結果。

```math
\begin{aligned}
\text{Discrete:} \quad & \mathbb{E}[X] = \sum_i x_i \, P(X = x_i) \\
\text{Continuous:} \quad & \mathbb{E}[X] = \int x \, f(x) \, dx
\end{aligned}
```

Variance 衡量圍繞 mean 的離散程度。

```math
\begin{aligned}
\mathrm{Var}(X) &= \mathbb{E}[(X - \mathbb{E}[X])^2] = \mathbb{E}[X^2] - (\mathbb{E}[X])^2 \\
\text{Standard deviation} &= \sqrt{\mathrm{Var}(X)}
\end{aligned}
```

在 ML 裡，expected value 以 loss function 的形式出現（資料分佈上的平均 loss）。Variance 告訴你模型的穩定性。Gradient 的 variance 高，代表訓練有雜訊。

### Joint 與 Marginal 分佈 (Joint and Marginal Distributions)

Joint distribution $P(X, Y)$ 同時描述兩個隨機變數。

Joint PMF 範例（$X$ = 天氣，$Y$ = 雨傘）：

| | $Y=0$（沒帶傘） | $Y=1$（帶傘） | Marginal $P(X)$ |
|---|---|---|---|
| $X=0$（晴天） | 0.40 | 0.10 | $P(X=0) = 0.50$ |
| $X=1$（下雨） | 0.05 | 0.45 | $P(X=1) = 0.50$ |
| **Marginal $P(Y)$** | $P(Y=0) = 0.45$ | $P(Y=1) = 0.55$ | 1.00 |

Marginal distribution 把另一個變數加總掉：

```math
P(X = x) = \sum_{y} P(X = x, Y = y)
```

上表中的列總和與欄總和就是 marginal。

### 為什麼 Normal 分佈無所不在 (Why the Normal Distribution Shows Up Everywhere)

Central Limit Theorem：來自同一個分佈的許多獨立隨機變數，其平均值在經過置中並以 $\sqrt{n}$ 縮放之後，會趨近 normal 分佈，不論原本分佈的形狀為何，只要它的 variance 是有限的。對置中並縮放後的總和也是如此。

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution with finite variance.
```

這就是為什麼：
- 測量誤差大致呈 normal（許多微小的獨立來源）
- Neural network 的 weight 初始化使用 normal 分佈
- SGD 中的 gradient 雜訊大致呈 normal（許多樣本 gradient 的總和）
- 在給定 mean 與 variance 的情況下，normal 分佈是 maximum entropy 的分佈

### Log Probability (Log Probabilities)

原始的機率會造成數值問題。把許多很小的機率相乘，很快就會 underflow 變成零。

```math
\begin{aligned}
P(\text{sentence}) &= P(\text{word}_1) \cdot P(\text{word}_2) \cdot \ldots \cdot P(\text{word}_n) \\
&= 0.01 \cdot 0.003 \cdot 0.02 \cdot \ldots \\
&\to 0.0 \quad \text{(underflow after } \sim 30 \text{ terms)}
\end{aligned}
```

Log probability 解決了這個問題。乘法變成加法。

```math
\begin{aligned}
\log P(\text{sentence}) &= \log P(\text{word}_1) + \log P(\text{word}_2) + \ldots + \log P(\text{word}_n) \\
&= -4.6 + -5.8 + -3.9 + \ldots \\
&\to \text{finite number (no underflow)}
\end{aligned}
```

規則：
- $\log(a \cdot b) = \log(a) + \log(b)$
- log probability 永遠 $\le 0$（因為 $0 < P \le 1$）
- 越負 = 越不可能
- Cross-entropy loss 就是正確類別的 negative log probability

### 把 Softmax 當作機率分佈 (Softmax as a Probability Distribution)

Neural network 輸出原始分數（logits）。Softmax 把它們轉成有效的機率分佈。

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Softmax 技巧：先減去最大的 logit 再取指數，以防止 overflow。

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax 把 softmax 和 log 結合起來，以達到數值穩定。PyTorch 在內部計算 cross-entropy loss 時就是用它。

### 取樣 (Sampling)

取樣是指從一個分佈中抽出隨機值。在 ML 裡：
- Dropout 隨機取樣要把哪些 neuron 歸零
- Data augmentation 隨機取樣轉換方式
- 語言模型從預測的分佈中取樣下一個 token
- Diffusion model 取樣雜訊，再逐步去噪

從任意分佈取樣，需要像 inverse transform sampling、rejection sampling 或 reparameterization trick（用於 VAE）這類技術。

```figure
gaussian-pdf
```

## 動手做 (Build It)

### 步驟 1：機率基礎

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### 步驟 2：從零寫 PMF 與 PDF

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### 步驟 3：Expected value 與 variance

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### 步驟 4：從分佈中取樣

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### 步驟 5：Softmax 與 log probability

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### 步驟 6：Central Limit Theorem 示範

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### 步驟 7：視覺化

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

包含所有視覺化的完整實作在 `code/probability.py`。

## 實際使用 (Use It)

有了 NumPy 和 SciPy，上面的一切都只要一行：

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

這些你都從零做過了。現在你知道函式庫的呼叫背後在做什麼。

## 練習 (Exercises)

1. 為 exponential 分佈實作 inverse transform sampling。取樣 10,000 個值，並把直方圖與真實的 PDF 比較來驗證。

2. 為兩顆灌鉛骰子建立 joint distribution 表。計算 marginal 分佈，並檢查這兩顆骰子是否獨立。

3. 對一個 5 類別分類器計算 cross-entropy loss，它輸出的 logits 是 `[2.0, 0.5, -1.0, 3.0, 0.1]`，正確類別是 index 3。然後用 PyTorch 的 `nn.CrossEntropyLoss` 驗證你的答案。

4. 寫一個函式，接收一串 log probability，回傳最可能的序列、總 log probability，以及對應的原始機率。用一個 50 個詞、每個詞機率為 0.01 的句子來測試。

### 提示 (Hints)

**練習 1。** 從 uniform 分佈抽出 `u`，回傳 $-\ln(1 - u) / \lambda$。檢查樣本平均值是否接近 $1/\lambda$，以及直方圖的各個 bin 是否符合 $\lambda e^{-\lambda x}$。

**練習 2。** 填滿一張 6x6 的表，其中每個格子都是機率，且所有格子的總和是 1，然後呼叫 `joint_to_marginals`。使用 `check_independence`：只有當每個格子都等於它兩個 marginal 的乘積時，這兩顆骰子才是獨立的。

**練習 3。** 對 logits 套用 `softmax`，再對 index 3 的機率取 $-\ln$。結果必須接近 0.421，這是 `nn.CrossEntropyLoss` 對相同 logits 和 target 給出的值。

**練習 4。** 把 log probability 相加；不要把原始機率相乘。對 50 個機率為 0.01 的詞，總和必須是 $50 \ln(0.01) \approx -230.26$，而原始機率是 $10^{-100}$。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意思 |
|------|----------------|----------------------|
| Sample space | 「所有的可能性」 | 一個實驗所有可能結果的集合 $S$ |
| PMF | 「機率函數」 | 給出每個離散結果確切機率的函數，總和為 1 |
| PDF | 「機率曲線」 | 連續變數的密度函數。對一個區間積分才會得到機率 |
| Conditional probability | 「在某個條件下的機率」 | $P(A \mid B) = P(A \text{ and } B) / P(B)$。是 Bayesian 思維與 Bayes' theorem 的基礎 |
| Independence | 「它們互不影響」 | $P(A \text{ and } B) = P(A) \cdot P(B)$。知道其中一個事件，完全不會告訴你另一個事件的資訊 |
| Expected value | 「平均值」 | 所有結果依機率加權的總和。Loss function 就是一個 expected value |
| Variance | 「有多分散」 | 與 mean 的平方偏差的期望值。Variance 高 = 估計有雜訊、不穩定 |
| Normal distribution | 「鐘形曲線」 | $f(x) = (1/\sqrt{2\pi\sigma^2}) \cdot \exp(-(x-\mu)^2/(2\sigma^2))$。因為 CLT 而無所不在 |
| Central Limit Theorem | 「平均值會變成 normal」 | 許多獨立樣本經置中與縮放後的平均值，對任何 variance 有限的來源，都會收斂到 normal 分佈 |
| Joint distribution | 「兩個變數放在一起」 | $P(X, Y)$ 描述 $X$ 與 $Y$ 每一種結果組合的機率 |
| Marginal distribution | 「把另一個變數加總掉」 | $P(X) = \sum_y P(X, Y)$。從 joint 還原出單一變數的分佈 |
| Log probability | 「機率的 log」 | $\log P(x)$。把乘積變成加總，避免長序列的數值 underflow |
| Softmax | 「把分數變成機率」 | $\text{softmax}(z_i) = \exp(z_i) / \sum_j \exp(z_j)$。把實數值的 logits 映射成有效的機率分佈 |
| Cross-entropy | 「那個 loss function」 | $-\sum p_{\text{true}} \cdot \log(p_{\text{predicted}})$。衡量兩個分佈有多不同。越低越好 |
| Logits | 「模型的原始輸出」 | 經過 softmax 之前、未正規化的分數。名稱來自 logistic function |
| Sampling | 「抽出隨機值」 | 依照機率分佈產生數值。模型產生輸出的方式 |

## 延伸閱讀 (Further Reading)

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - 用視覺化證明為什麼平均值會變成 normal
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - 涵蓋這裡所有內容還更多的精簡參考資料
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - 為什麼數值穩定很重要，以及如何做到
