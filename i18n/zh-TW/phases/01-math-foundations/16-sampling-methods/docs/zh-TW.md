# 取樣方法 (Sampling Methods)

> 取樣是 AI 探索各種可能性空間的方式。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 06-07 (Probability, Bayes' Theorem)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 只用 uniform 亂數，從零實作 inverse CDF、rejection 與 importance sampling
- 為語言模型的 token 生成打造 temperature、top-k 與 top-p (nucleus) sampling
- 說明 reparameterization trick，以及它為什麼能讓 VAE 對取樣過程做 backpropagation
- 執行 Metropolis-Hastings MCMC，從未 normalize 的目標分佈中取樣

## 問題 (The Problem)

語言模型處理完你的 prompt 後，會產生一個含有 50,000 個 logits 的 vector，詞彙表中的每個 token 各一個。現在它得挑一個。怎麼挑？

如果它永遠挑機率最高的 token，每個回應都會一模一樣。決定論式的，很無聊。如果它均勻隨機挑選，輸出就是胡言亂語。答案落在這兩個極端之間的某處，而那個「某處」是由取樣控制的。

取樣不只限於文字生成。強化學習透過取樣軌跡來估計 policy gradient。VAE 透過從學到的分佈中取樣，並對隨機性做 backpropagation，來學習 latent representation。Diffusion model 透過取樣雜訊並反覆去噪來生成影像。Monte Carlo 方法估計沒有封閉形式解的積分。MCMC 演算法探索無法逐一列舉的高維 posterior 分佈。

每個生成式 AI 系統都是一個取樣系統。取樣策略決定了輸出的品質、多樣性與可控性。這堂課從零打造每個主要的取樣方法，從 uniform 亂數開始，一路到驅動現代 LLM 與生成模型的技術。

## 核心概念 (The Concept)

### 為什麼取樣很重要 (Why Sampling Matters)

取樣在 AI 與機器學習中扮演四種基本角色：

**生成。** 語言模型、diffusion model 與 GAN 都是透過取樣來產生輸出。取樣演算法直接控制創意、連貫性與多樣性。Temperature、top-k 與 nucleus sampling 是工程師每天轉動的旋鈕。

**訓練。** Stochastic gradient descent 對 mini-batch 取樣。Dropout 對要停用的 neuron 取樣。Data augmentation 對隨機轉換取樣。Importance sampling 重新加權樣本，以降低強化學習（PPO、TRPO）中 gradient 的 variance。

**估計。** ML 中許多量沒有封閉形式的解。資料分佈上的期望 loss、energy-based model 的 partition function、Bayesian inference 中的 evidence。Monte Carlo 估計透過對樣本取平均，來近似所有這些量。

**探索。** MCMC 演算法探索 Bayesian inference 中的 posterior 分佈。Evolutionary strategy 對參數擾動取樣。Thompson sampling 在 bandit 中平衡探索與利用。

核心挑戰：你只能直接從簡單的分佈（uniform、normal）取樣。對其他一切，你需要一種方法，把簡單的樣本轉換成來自你目標分佈的樣本。

### Uniform 隨機取樣 (Uniform Random Sampling)

每個取樣方法都從這裡開始。Uniform 亂數產生器產生 $[0, 1)$ 中的值，其中每個長度相等的子區間都有相等的機率。

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

要從 $n$ 個項目的離散集合中均勻取樣，產生 $U$ 並回傳 $\lfloor n U \rfloor$。要從連續範圍 $[a, b]$ 取樣，計算 $a + (b - a) U$。

關鍵洞見：單一個 uniform 亂數，剛好包含了產生任何分佈的一個樣本所需的隨機性。訣竅是找到正確的轉換。

### Inverse CDF 方法（Inverse Transform Sampling） (Inverse CDF Method (Inverse Transform Sampling))

累積分佈函數 (CDF) 把數值映射到機率：

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF 把機率映射回數值。如果 $U \sim \text{Uniform}(0, 1)$，那麼 $X = F^{-1}(U)$ 就服從目標分佈。

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Exponential 分佈範例：**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

當你能以封閉形式寫出 $F^{-1}$ 時，這個方法完美有效。對 normal 分佈，沒有封閉形式的 inverse CDF，所以我們改用其他方法（Box-Muller，或數值近似）。

**離散版本：** 對離散分佈，把 CDF 建成累積和，產生 $U$，然後找出累積和第一次超過 $U$ 的索引。這就是 Lesson 06 中 `sample_categorical` 的運作方式。

### Rejection Sampling (Rejection Sampling)

當你無法對 CDF 求反函數，但能計算（至多差一個常數的）目標 PDF 時，rejection sampling 就派得上用場。

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = Z/M, where Z is the integral of p(x)
                (Z = 1, so the rate is 1/M, when p is normalized)
```

Bound $M$ 越緊，接受率越高。在低維度（1-3）時，rejection sampling 運作良好。在高維度時，接受率會呈指數下降，因為 proposal 的大部分體積都被拒絕。這就是 rejection sampling 的維度詛咒。

**範例：從 truncated normal 取樣。** 在截斷的範圍上使用 uniform proposal。包絡線 $M$ 是 normal PDF 在該範圍內的最大值。

**範例：從半圓取樣。** 在外接矩形內均勻提出候選點。如果該點落在半圓內就接受。Monte Carlo 就是這樣計算 $\pi$ 的：接受率等於面積比 $\pi/4$。

### Importance Sampling (Importance Sampling)

有時你不需要來自目標分佈 $p(x)$ 的樣本。你需要的是估計 $p(x)$ 下的期望，而你手上有的是來自另一個分佈 $q(x)$ 的樣本。

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

這在強化學習中至關重要。在 PPO (Proximal Policy Optimization) 中，你在舊 policy $\pi_{\text{old}}$ 下收集軌跡，卻想要最佳化新 policy $\pi_{\text{new}}$。Importance weight 是 $\pi_{\text{new}}(a \mid s) / \pi_{\text{old}}(a \mid s)$。PPO 會 clip 這些 weight，防止新 policy 偏離舊 policy 太遠。

Importance sampling 估計量的 variance，取決於 $q$ 與 $p$ 有多相似。如果 $q$ 與 $p$ 差異很大，少數幾個樣本會得到極大的 weight，並主導估計。Self-normalized importance sampling 除以 weight 的總和，來減輕這個問題：

```math
\mathbb{E}_p[f(x)] \approx \frac{\sum_i w_i \, f(x_i)}{\sum_i w_i}
```

### Monte Carlo 估計 (Monte Carlo Estimation)

Monte Carlo 估計透過對隨機樣本取平均來近似積分。大數法則保證收斂。

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

誤差率與維度無關。這就是 Monte Carlo 方法在高維度中佔主導地位的原因，因為在高維度中以網格為基礎的積分是不可能的。

**估計 pi：**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**估計期望值：**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC)：Metropolis-Hastings (Markov Chain Monte Carlo (MCMC): Metropolis-Hastings)

MCMC 建構一條 Markov chain，其 stationary distribution 就是目標分佈 $p(x)$。經過足夠多步之後，來自這條 chain 的樣本（近似地）就是來自 $p(x)$ 的樣本。

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

對於對稱的 proposal $(q(x' \mid x) = q(x \mid x'))$，比值簡化為 $p(x')/p(x)$。這就是原始的 Metropolis 演算法。

**為什麼有效。** 接受規則確保了 detailed balance：位於 $x$ 並移動到 $x'$ 的機率，等於位於 $x'$ 並移動到 $x$ 的機率。Detailed balance 蘊含 $p(x)$ 是這條 chain 的 stationary distribution。

**實務上的考量：**
- Burn-in：在 chain 達到平衡之前，丟棄早期的樣本
- Thinning：每 $k$ 個樣本保留一個，以降低 autocorrelation
- Proposal 的尺度：太小，chain 移動緩慢（高接受率、探索慢）；太大，大部分的 proposal 都被拒絕（低接受率、卡在原地）
- 在高維度中，Gaussian proposal 的最佳接受率大約是 0.234

### Gibbs Sampling (Gibbs Sampling)

Gibbs sampling 是 MCMC 針對多變量分佈的特例。它不是一次在所有維度上提出移動，而是每次從條件分佈中更新一個變數。

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling 要求你能從每個條件分佈 $p(x_i \mid x_{-i})$ 取樣。對許多模型來說，這很直接：
- Bayesian network：條件分佈可從圖結構得出
- Gaussian mixture：條件分佈是 Gaussian
- Ising model：每個 spin 的條件分佈只取決於它的鄰居

接受率永遠是 1（每個 proposal 都被接受），因為從精確的條件分佈取樣，自動滿足 detailed balance。

**限制。** 當變數高度相關時，Gibbs sampling 混合得很慢，因為一次只更新一個變數，無法在分佈中做出大幅度的對角線移動。

### Temperature Sampling（用於 LLM） (Temperature Sampling (Used in LLMs))

語言模型為詞彙表中的每個 token 輸出 logits $z_1, \ldots, z_V$。Softmax 把這些轉成機率。Temperature 在 softmax 之前重新縮放 logits：

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**為什麼有效。** 把 logits 除以 $T < 1$ 會放大 logits 之間的差異。如果 $z_1 = 2$ 且 $z_2 = 1$，除以 $T = 0.5$ 得到 $z_1/T = 4$ 和 $z_2/T = 2$，讓差距變大。經過 softmax 後，logit 最高的 token 會得到大得多的份額。

**實務上：**
- $T = 0.0$：greedy decoding，最適合事實型 Q&A
- $T = 0.3\text{-}0.7$：略帶創意，適合程式碼生成
- $T = 0.7\text{-}1.0$：均衡，適合一般對話
- $T = 1.0\text{-}1.5$：創意寫作、腦力激盪
- $T > 1.5$：越來越隨機，很少有用

Temperature 不會改變哪些 token 是可能的。它改變的是分配給每個 token 的機率質量。

### Top-k Sampling (Top-k Sampling)

Top-k sampling 把候選集合限制在機率最高的 $k$ 個 token，然後重新 normalize 並從這個受限集合中取樣。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k 防止模型選中詞彙分佈長尾中極不可能的 token（錯字、無意義的內容）。問題在於：不論上下文如何，$k$ 都是固定的。當模型很有把握（一個 token 有 95% 的機率）時，$k = 40$ 仍然允許 39 個替代選項。當模型不確定（機率分散在 1000 個 token 上）時，$k = 40$ 會截掉合理的選項。

### Top-p (Nucleus) Sampling (Top-p (Nucleus) Sampling)

Top-p sampling 動態調整候選集合的大小。它不是保留固定數量的 token，而是保留累積機率超過 $p$ 的最小 token 集合。

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

當模型有把握時，nucleus sampling 保留的 token 很少（也許 2-3 個）。當模型不確定時，它保留很多（也許 200 個）。這種自適應的行為，就是 nucleus sampling 通常比 top-k 產生更好文字的原因。

**常見的組合：**
- Temperature 0.7 + top-p 0.9：良好的通用設定
- Temperature 0.0 (greedy)：最適合決定性任務
- Temperature 1.0 + top-k 50：Fan et al. (2018) 原始論文的設定

Top-k 與 top-p 可以結合。先套用 top-k，再對剩下的集合套用 top-p。

### Reparameterization Trick（用於 VAE） (Reparameterization Trick (Used in VAEs))

Variational autoencoder (VAE) 的學習方式是：把輸入編碼成 latent 空間中的分佈，從該分佈取樣，再把樣本解碼回來。問題是：你無法對取樣運算做 backpropagation。

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick 把隨機性與參數分開：

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

這之所以成立，是因為 $\mathcal{N}(\mu, \sigma^2)$ 與 $\mu + \sigma \cdot \mathcal{N}(0, 1)$ 具有相同的分佈。關鍵洞見：把隨機性移到一個與參數無關的來源 ($\epsilon$)，然後把樣本表示成參數的可微分轉換。

**在 VAE 訓練迴圈中：**
1. Encoder 為每個輸入輸出 $\mu$ 和 $\log(\sigma^2)$
2. 取樣 $\epsilon \sim \mathcal{N}(0, 1)$
3. 計算 $z = \mu + \sigma \epsilon$
4. Decode $z$ 以重建輸入
5. 對步驟 4、3、2、1 做 backpropagation（可行，因為步驟 3 是可微分的）

沒有 reparameterization trick，VAE 就無法用標準的 backpropagation 訓練。這個單一的洞見讓 VAE 變得實用。

### Gumbel-Softmax（可微分的類別取樣） (Gumbel-Softmax (Differentiable Categorical Sampling))

Reparameterization trick 適用於連續分佈（Gaussian）。對離散的 categorical 分佈，我們需要不同的做法。Gumbel-Softmax 提供了對 categorical 取樣的可微分近似。

**Gumbel-Max 技巧（不可微分）：**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax（可微分近似）：**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax 產生離散樣本的連續鬆弛 (relaxation)。輸出是機率 vector（soft one-hot），而不是 hard one-hot。Gradient 流過 softmax。在訓練的 forward pass 中，你可以使用「straight-through」estimator：forward pass 使用 hard argmax，但 backward pass 使用 soft Gumbel-Softmax 的 gradient。

**應用：**
- VAE 中的離散 latent 變數
- Neural architecture search（選擇離散的運算）
- Hard attention 機制
- 具有離散動作的強化學習

### Stratified Sampling (Stratified Sampling)

標準的 Monte Carlo 取樣可能因為偶然而在樣本空間中留下空隙。Stratified sampling 將空間分成多個階層 (strata)，並從每一層取樣，藉此強制涵蓋均勻。

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

與標準的 Monte Carlo 相比，stratified sampling 的 variance 永遠較低或相等：

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**應用：**
- 數值積分（quasi-Monte Carlo）
- 訓練資料切分（確保每個 fold 中類別平衡）
- 搭配分層的 importance sampling（結合兩種技術）
- NeRF (Neural Radiance Fields) 沿著相機光線使用 stratified sampling

### 與 Diffusion Model 的關聯 (Connection to Diffusion Models)

Diffusion model 透過取樣過程生成影像。Forward process 在 $T$ 步內對影像加入 Gaussian 雜訊，直到它變成純雜訊。Reverse process 學習去噪，逐步還原原始影像。

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

與這堂課的方法的關聯：
- 每個去噪步驟都使用 reparameterization trick（取樣雜訊，再套用決定性的轉換）
- 雜訊排程 $\lbrace \alpha_t \rbrace$ 控制著一種 temperature annealing
- 訓練使用 Monte Carlo 估計來近似 ELBO (evidence lower bound)
- Diffusion model 中的 ancestral sampling 是一條 Markov chain（每一步只取決於當前狀態）

整個影像生成過程就是迭代取樣：從雜訊開始，並在每一步，以學到的去噪模型為條件，取樣一個稍微不那麼嘈雜的版本。

```figure
monte-carlo-pi
```

## 動手做 (Build It)

### 步驟 1：Uniform 與 inverse CDF 取樣

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

產生 10,000 個 exponential 樣本，並驗證 mean 是 $1/\lambda$。

### 步驟 2：Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

用 rejection sampling 從 truncated normal 分佈抽樣。透過對樣本畫直方圖來驗證形狀。

### 步驟 3：Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

用 uniform proposal 估計 normal 分佈下的 $\mathbb{E}[X^2]$。與已知答案 ($\mu^2 + \sigma^2$) 比較。

### 步驟 4：Monte Carlo 估計 pi

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### 步驟 5：Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

從雙峰分佈（兩個 Gaussian 的混合）取樣。視覺化 chain 的軌跡。

### 步驟 6：Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### 步驟 7：Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

展示 temperature 如何改變一組 token logits 的輸出分佈。

### 步驟 8：Top-k 與 top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### 步驟 9：Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

示範 gradient 會流過 reparameterized 的樣本，卻不會流過直接取樣的樣本。

### 步驟 10：Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

展示降低 temperature 如何讓輸出趨近 one-hot vector。

包含所有視覺化的完整實作在 `code/sampling.py`。

## 實際使用 (Use It)

用 NumPy 和 SciPy，正式環境用的版本：

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

大規模的 MCMC，請使用專用的函式庫：
- PyMC：使用 NUTS（自適應 HMC）的完整 Bayesian 建模
- emcee：ensemble MCMC 取樣器
- NumPyro/JAX：GPU 加速的 MCMC

這些你都從零做過了。現在你知道函式庫的呼叫背後在做什麼。

## 練習 (Exercises)

1. 為 Cauchy 分佈實作 inverse CDF 取樣。CDF 是 $F(x) = 0.5 + \arctan(x)/\pi$。產生 10,000 個樣本，並將直方圖與真實的 PDF 一起畫出。注意重尾（遠離中心的極端值）。

2. 用 rejection sampling，以 Uniform(0, 1) proposal 產生 Beta(2, 5) 分佈的樣本。把被接受的樣本與真實的 Beta PDF 一起畫出。理論上的接受率是多少？

3. 用 1,000、10,000 和 100,000 個樣本的 Monte Carlo，估計 $\sin(x)$ 從 0 到 $\pi$ 的積分。比較每個層級的誤差。驗證誤差按 $O(1/\sqrt{N})$ 縮放。

4. 實作 Metropolis-Hastings，從與 $\exp(-(x^2 y^2 + x^2 + y^2 - 8x - 8y) / 2)$ 成正比的 2D 分佈 $p(x, y)$ 取樣。畫出樣本與 chain 的軌跡。試驗不同的 proposal 標準差。

5. 打造一個完整的文字生成示範：給定一個有 10 個詞、附 logits 的詞彙表，用 (a) greedy、(b) temperature=0.7、(c) top-k=3、(d) top-p=0.9 生成 20 個 token 的序列。比較 5 次執行中輸出的多樣性。

### 提示 (Hints)

**練習 1。** 對 $x$ 解 $u = F(x)$，得到 $x = \tan(\pi(u - 0.5))$。檢查樣本 median 接近 0，且約 6% 的樣本滿足 $\lvert x \rvert > 10$，這等於 $1 - (2/\pi)\arctan(10)$。

**練習 2。** Beta(2, 5) 的密度是 $30x(1-x)^4$。在 mode $x = 1/5$ 找出它的最大值，並把該值當作 `rejection_sample` 中的 $M$。理論上的接受率是 $1/M$，所以把它與你實際接受的 proposal 比例做比較。

**練習 3。** 精確值是 2。對每個 $n$ 呼叫 `monte_carlo_integral(math.sin, 0, math.pi, n)`，並用多個 seed 重複每次執行。當 $n$ 增加 10 倍時，平均誤差應該下降約 $\sqrt{10} \approx 3.2$ 倍。

**練習 4。** 寫出目標的 log，即 $-(x^2 y^2 + x^2 + y^2 - 8x - 8y)/2$，並傳給 `metropolis_hastings_2d`。記錄每個 proposal 標準差的接受率：非常小的值幾乎接受每一步但移動緩慢，非常大的值則幾乎拒絕每一步。

**練習 5。** Greedy decoding 在每一步取最大 logit 的索引。其他三種方法使用 `temperature_sample`、`top_k_sample` 和 `top_p_sample`。Greedy 在全部 5 次執行中必須給出相同的序列，所以計算每種方法不同序列的數量來衡量多樣性。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意思 |
|------|----------------|----------------------|
| Sampling | 「抽出隨機值」 | 依照機率分佈產生數值。是所有生成式 AI 背後的機制 |
| Uniform distribution | 「全部同樣可能」 | $[a, b]$ 中的每個值都有相等的機率密度 $1/(b-a)$。所有取樣方法的起點 |
| Inverse CDF | 「機率轉換」 | $F^{-1}(U)$ 把 uniform 樣本轉換成任何已知 CDF 之分佈的樣本。精確且有效率 |
| Rejection sampling | 「提出候選再接受/拒絕」 | 從簡單的 proposal 產生樣本，以正比於目標/proposal 比值的機率接受。精確，但會浪費樣本 |
| Importance sampling | 「重新加權樣本」 | 用來自 $q(x)$ 的樣本，以 $p(x)/q(x)$ 為每個樣本加權，來估計 $p(x)$ 下的期望。是 RL 中 PPO 的核心 |
| Monte Carlo | 「對隨機樣本取平均」 | 把積分近似為樣本平均。誤差 $O(1/\sqrt{N})$，與維度無關 |
| MCMC | 「會收斂的隨機漫步」 | 建構一條 stationary distribution 就是目標分佈的 Markov chain。Metropolis-Hastings 是基礎演算法 |
| Metropolis-Hastings | 「爬坡一定接受，下坡有時接受」 | 提出移動，根據密度比值決定是否接受。Detailed balance 確保收斂到目標分佈 |
| Gibbs sampling | 「一次一個變數」 | 固定其他變數，從條件分佈更新每個變數。100% 接受率 |
| Temperature | 「信心旋鈕」 | 在 softmax 之前把 logits 除以 $T$。$T<1$ 使分佈變尖（更有信心），$T>1$ 使分佈變平（更多樣） |
| Top-k sampling | 「保留最好的 k 個」 | 將機率最高的 $k$ 個 token 以外的全部歸零，重新 normalize，再取樣。候選集合大小固定 |
| Nucleus sampling (top-p) | 「保留機率高的」 | 保留累積機率超過 $p$ 的最小 token 集合。候選集合大小是自適應的 |
| Reparameterization trick | 「把隨機性移到外面」 | 寫成 $z = \mu + \sigma \epsilon$，其中 $\epsilon \sim \mathcal{N}(0,1)$。讓取樣變得可微分。對 VAE 訓練至關重要 |
| Gumbel-Softmax | 「Soft 的類別取樣」 | 使用 Gumbel 雜訊加上帶 temperature 的 softmax，對 categorical 取樣的可微分近似 |
| Stratified sampling | 「強制涵蓋」 | 將樣本空間分成多個階層，從每一層取樣。variance 永遠比單純的 Monte Carlo 低 |
| Burn-in | 「暖身期」 | 在 chain 達到 stationary distribution 之前被丟棄的早期 MCMC 樣本 |
| Detailed balance | 「可逆條件」 | $p(x) T(x \to y) = p(y) T(y \to x)$。使 $p$ 成為 Markov chain 之 stationary distribution 的充分條件 |
| Diffusion sampling | 「迭代去噪」 | 從雜訊開始，套用學到的去噪步驟來生成資料。每一步都是條件取樣運算 |

## 延伸閱讀 (Further Reading)

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - MCMC 基礎的詳細教學
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - 原始的 Gumbel-Softmax 論文
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - nucleus (top-p) sampling 論文
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - 提出 reparameterization trick 的 VAE 論文
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM 把取樣與影像生成連結起來
