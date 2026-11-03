# 數值穩定性 (Numerical Stability)

> 浮點數是一個會漏水的抽象層。它會在訓練時咬你一口，而且你完全不會預料到。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01-04
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 用減去最大值的技巧，實作數值穩定的 softmax 與 log-sum-exp
- 辨識浮點運算中的 overflow、underflow 與 catastrophic cancellation
- 用 centered finite difference，對照數值 gradient 驗證解析 gradient
- 說明為什麼訓練時 bfloat16 比 float16 更受青睞，以及 loss scaling 如何防止 gradient underflow

## 問題 (The Problem)

你的模型訓練了三小時，然後 loss 變成 NaN。你加了一行 print。在第 9,000 步時 logits 都還正常。到了第 9,001 步，它們變成 `inf`。到第 9,002 步，每個 gradient 都是 `nan`，訓練就死了。

或者：你的模型順利訓練完成，但 accuracy 比論文宣稱的低了 2%。你檢查了所有東西。Architecture 吻合。Hyperparameter 吻合。資料也吻合。問題在於論文用的是 float32，而你用了 float16，卻沒有做正確的縮放。累積的捨入誤差默默吃掉了你的 accuracy。

又或者：你從零實作 cross-entropy loss。在小的 logits 上沒問題。當 logits 超過 100，它回傳 `inf`。Softmax overflow 了，因為 `exp(100)` 比 float32 能表示的數字還大。每個 ML 框架都用一個兩行的技巧處理這件事。你不知道有這個技巧。

數值穩定性不是理論上的顧慮。它是一次成功的訓練，與一次悄悄失敗的訓練之間的差別。你日後會除錯的每個嚴重的 ML bug，最終都會歸結到浮點數。

## 核心概念 (The Concept)

### IEEE 754：電腦如何儲存實數 (IEEE 754: How Computers Store Real Numbers)

電腦依照 IEEE 754 標準，把實數儲存為浮點數值。一個 float 有三個部分：符號位元、exponent 與 mantissa（significand）。

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Mantissa 決定精度（有幾位有效數字）。Exponent 決定範圍（數字可以多大或多小）。

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 約有 7 位十進位的精度。這代表它能分辨 1.0000001 和 1.0000002，但無法分辨 1.00000001 和 1.00000002。超過 7 位之後，一切都是捨入雜訊。

float16 約有 3 位精度。它能表示的最大數字是 65,504。對 ML 而言，這小得令人不安，因為 logits、gradient 與 activation 經常超過這個數字。

bfloat16 是 Google 對 float16 範圍問題的解答。它擁有與 float32 相同的 8 位元 exponent（相同範圍，最大可到 3.4e38），但只有 7 個 mantissa 位元（精度比 float16 低）。訓練 neural network 時，範圍比精度更重要，所以 bfloat16 通常勝出。

### 為什麼 0.1 + 0.2 != 0.3 (Why 0.1 + 0.2 != 0.3)

數字 0.1 無法在二進位浮點數中被精確表示。在 base 2 中，它是一個無限循環的小數：

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 把它截斷為 23 位元的 mantissa。儲存的值約為 0.100000001490116。同樣地，0.2 儲存為約 0.200000002980232。它們的和是 0.300000004470348，不是 0.3。

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

這對 ML 很重要，因為：

1. 像 `if loss < threshold` 這樣的 loss 比較可能給出錯誤的答案
2. 累加許多很小的值（數千步的 gradient 更新）會偏離真實的總和
3. 如果你用 `==` 比較 float，checksum 與可重現性測試會失敗

解法：永遠不要用 `==` 比較 float。請用 `abs(a - b) < epsilon` 或 `math.isclose()`。

### Catastrophic Cancellation (Catastrophic Cancellation)

當你把兩個幾乎相等的浮點數相減，有效數字會互相抵銷，剩下的是被提升到最前面幾位的捨入雜訊。

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

單一次減法就造成 19% 的相對誤差。在 ML 裡，這會發生在你：

- 計算 mean 很大的資料的 variance：當 $\mathbb{E}[x]$ 很大時的 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$
- 把幾乎相等的 log-probability 相減
- 用太小的 epsilon 計算 finite-difference gradient

解法：重新整理公式，避免把大的、幾乎相等的數字相減。對 variance，使用 Welford 演算法，或先把資料置中。對 log-probability，全程在 log-space 中運算。

### Overflow 與 Underflow (Overflow and Underflow)

Overflow 發生在結果太大而無法表示時。Underflow 發生在結果太小時（比最小可表示的正數更接近零）。

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

`exp()` 函數是 ML 中 overflow 的主要來源：

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

`log()` 函數則是朝另一個方向出問題：

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

在 ML 裡，`exp()` 出現在 softmax、sigmoid 與機率計算中。`log()` 出現在 cross-entropy、log-likelihood 與 KL divergence 中。沒有正確的技巧，`log(exp(x))` 這個組合就是個地雷區。

### Log-Sum-Exp 技巧 (The Log-Sum-Exp Trick)

直接計算 `log(sum(exp(x_i)))` 在數值上很危險。如果任何一個 `x_i` 很大，`exp(x_i)` 就會 overflow。如果所有 `x_i` 都很負，每個 `exp(x_i)` 都會 underflow 成零，而 `log(0)` 是 `-inf`。

技巧：在取指數之前，先減去最大值。

```math
\log\left(\sum_i \exp(x_i)\right) = \max(x) + \log\left(\sum_i \exp(x_i - \max(x))\right)
```

為什麼有效：減去 `max(x)` 之後，最大的指數是 `exp(0) = 1`。不可能 overflow。總和中至少有一項是 1，所以總和至少是 1，而 `log(1) = 0`。不可能 underflow 成 `-inf`。

證明：

```math
\begin{aligned}
& \log\left(\sum_i \exp(x_i)\right) \\
&= \log\left(\sum_i \exp(x_i - c + c)\right) && \text{(add and subtract } c\text{)} \\
&= \log\left(\sum_i \exp(x_i - c) \cdot \exp(c)\right) && (\exp(a+b) = \exp(a) \cdot \exp(b)) \\
&= \log\left(\exp(c) \cdot \sum_i \exp(x_i - c)\right) && \text{(factor out } \exp(c)\text{)} \\
&= c + \log\left(\sum_i \exp(x_i - c)\right) && (\log(a \cdot b) = \log(a) + \log(b))
\end{aligned}
```

令 `c = max(x)`，overflow 就消除了。

這個技巧在 ML 中無所不在：
- Softmax normalization
- Cross-entropy loss 的計算
- 序列模型中的 log-probability 加總
- Mixture of Gaussians
- Variational inference

### 為什麼 Softmax 需要減去最大值的技巧 (Why Softmax Needs the Max-Subtraction Trick)

Softmax 把 logits 轉成機率：

```math
\mathop{\mathrm{softmax}}(x_i) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
```

沒有這個技巧，logits [100, 101, 102] 會造成 overflow：

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

2.69e43 > 3.4e38, the float32 maximum, so every term overflows.
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

有了這個技巧，減去 max(x) = 102：

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

機率完全相同。計算是安全的。這不是一種最佳化，而是正確性的必要條件。

### NaN 與 Inf：偵測與預防 (NaN and Inf: Detection and Prevention)

`nan`（Not a Number）和 `inf`（無限大）會像病毒一樣在運算中傳播。Gradient 更新中的一個 `nan` 會讓 weight 變成 `nan`，進而讓之後的每個輸出都變成 `nan`。訓練在一步之內就死了。

`inf` 如何出現：
- 對很大的正數取 `exp()`
- 除以零：`1.0 / 0.0`
- 累加中的 `float32` overflow

`nan` 如何出現：
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- 對負數取 `sqrt()`
- 對負數取 `log()`
- 任何涉及既有 `nan` 的算術

偵測：

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

預防策略：

1. 限制 `exp()` 的輸入範圍：`exp(clamp(x, -80, 80))`
2. 在分母加上 epsilon：`x / (y + 1e-8)`
3. 在 `log()` 內加上 epsilon：`log(x + 1e-8)`
4. 使用穩定的實作（log-sum-exp、stable softmax）
5. 使用 gradient clipping 防止 weight 爆炸
6. 除錯期間，在每次 forward pass 後檢查 `nan`/`inf`

### 數值 Gradient 檢查 (Numerical Gradient Checking)

解析 gradient（來自 backpropagation）可能有 bug。數值 gradient 檢查透過 finite difference 計算 gradient 來驗證它們。

Centered difference 公式：

```math
\frac{df}{dx} \approx \frac{f(x + h) - f(x - h)}{2h}
```

它的精度是 $O(h^2)$，遠比只有 $O(h)$ 的 forward difference $(f(x+h) - f(x)) / h$ 好。

選擇 $h$：太大，近似就不準。太小，catastrophic cancellation 會毀掉答案。典型的值是 `h = 1e-5` 到 `1e-7`。

檢查方式：計算解析 gradient 與數值 gradient 之間的相對差異。

```math
\text{relative\_error} = \frac{\lvert \text{grad}_{\text{analytical}} - \text{grad}_{\text{numerical}} \rvert}{\max(\lvert \text{grad}_{\text{analytical}} \rvert, \lvert \text{grad}_{\text{numerical}} \rvert, 10^{-8})}
```

經驗法則：
- relative_error < 1e-7：完美，gradient 正確
- relative_error < 1e-5：可接受，大概是正確的
- relative_error > 1e-3：有地方出錯了
- relative_error > 1：gradient 完全錯誤

實作新的 layer 或 loss function 時，一律要檢查 gradient。PyTorch 為此提供了 `torch.autograd.gradcheck()`。

### Mixed Precision 訓練 (Mixed Precision Training)

現代 GPU 有專用硬體（Tensor Core），計算 float16 matrix multiplication 比 float32 快 2-8 倍。Mixed precision 訓練就是利用這一點：

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

純 float16 訓練的問題：gradient 往往非常小（1e-8 或更小）。Float16 會把低於約 6e-8 的任何值 underflow 成零。你的模型停止學習，因為所有的 gradient 更新都是零。

解法是 loss scaling：

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling 會自動調整縮放係數。從一個大的值（65536）開始。如果 gradient overflow 成 `inf`，就減半。如果經過 $N$ 步都沒有 overflow，就加倍。

### bfloat16 與 float16：為什麼訓練時 bfloat16 勝出 (bfloat16 vs float16: Why bfloat16 Wins for Training)

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 的精度較高（10 個 mantissa 位元 vs 7 個），但範圍有限（最大約 65,504）。bfloat16 的精度較低，但範圍與 float32 相同（最大約 3.4e38）。

對於訓練 neural network：

- 在訓練出現尖峰時，activation 與 logits 經常超過 65,504。float16 會 overflow；bfloat16 能處理。
- float16 需要 loss scaling，但 bfloat16 通常不需要，因為它的範圍涵蓋了 gradient 大小的整個頻譜。
- bfloat16 是 float32 的單純截斷：丟掉 mantissa 的低 16 位元。轉換很簡單，而且 exponent 沒有損失。

float16 適合推論，因為數值有界、精度更重要。bfloat16 適合訓練，因為範圍更重要。這就是為什麼 TPU 與現代 NVIDIA GPU（A100、H100）原生支援 bfloat16。

### Gradient Clipping (Gradient Clipping)

當 gradient 在許多層中呈指數增長時，就會發生 exploding gradient（常見於 RNN、深層網路與 transformer）。單一個很大的 gradient 就能在一步之內破壞所有的 weight。

兩種 clipping：

**依值 clip：** 獨立地限制每個 gradient 元素。

```
grad = clamp(grad, -max_val, max_val)
```

簡單，但可能改變 gradient vector 的方向。

**依 norm clip：** 縮放整個 gradient vector，讓它的 norm 不超過門檻。

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

保留 gradient 的方向。這就是 `torch.nn.utils.clip_grad_norm_()` 做的事，也是標準的選擇。

典型的值：transformer 用 `max_norm=1.0`，RL 用 `max_norm=0.5`，較簡單的網路用 `max_norm=5.0`。

Gradient clipping 不是 hack，而是一種安全機制。沒有它，單一個離群的 batch 就可能產生大到足以毀掉數週訓練的 gradient。

### 把 Normalization Layer 當作數值穩定器 (Normalization Layers as Numerical Stabilizers)

Batch normalization、layer normalization 和 RMS normalization 通常被介紹為幫助訓練收斂的 regularizer。它們同時也是數值穩定器。

沒有 normalization，activation 會在各層中呈指數增長或縮小：

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization 在每一層重新置中並重新縮放 activation：

```math
\mathop{\mathrm{LayerNorm}}(x) = \frac{x - \mathop{\mathrm{mean}}(x)}{\mathop{\mathrm{std}}(x) + \epsilon} \cdot \gamma + \beta
```

`epsilon`（通常是 1e-5）在所有 activation 都相同時，防止除以零。學到的參數 `gamma` 和 `beta` 讓網路能還原它所需的任何尺度。

這讓整個網路中的數值維持在數值安全的範圍內，同時防止 forward pass 的 overflow 與 backward pass 的 gradient 爆炸。

### 常見的 ML 數值 Bug (Common ML Numerical Bugs)

**Bug：幾個 epoch 之後 loss 變成 NaN。**
原因：logits 變得太大，softmax overflow。或者 learning rate 太高，weight 發散了。
解法：使用穩定的 softmax（減去最大值）、降低 learning rate、加入 gradient clipping。

**Bug：Loss 卡在 log(num_classes)。**
原因：模型輸出接近 uniform 的機率。通常代表 gradient 正在消失，或模型完全沒在學習。
解法：檢查資料標籤是否正確、驗證 loss function、檢查是否有 dead ReLU。

**Bug：Validation accuracy 比預期低了 1-3%。**
原因：使用了 mixed precision，卻沒有適當的 loss scaling。Gradient underflow 默默地把小的更新歸零。
解法：啟用 dynamic loss scaling，或改用 bfloat16。

**Bug：某些 layer 的 gradient norm 是 0.0。**
原因：dead ReLU neuron（所有輸入都是負的），或 float16 underflow。
解法：使用 LeakyReLU 或 GELU、使用 gradient scaling、檢查 weight 初始化。

**Bug：模型在一張 GPU 上正常，在另一張上卻給出不同結果。**
原因：浮點數累加順序不具決定性。GPU 的平行 reduction 在不同硬體上以不同順序加總，而浮點數加法不滿足結合律。
解法：接受小的差異（1e-6），或設定 `torch.use_deterministic_algorithms(True)` 並接受速度上的代價。

**Bug：`exp()` 在 loss 計算中回傳 `inf`。**
原因：原始 logits 直接傳進 `exp()`，沒有減去最大值的技巧。
解法：使用 `torch.nn.functional.log_softmax()`，它在內部實作了 log-sum-exp。

**Bug：從 float32 換成 float16 後，訓練發散。**
原因：float16 無法表示小於 6e-8 的 gradient 大小，也無法表示大於 65,504 的 activation。
解法：使用帶 loss scaling 的 mixed precision（AMP），或改用 bfloat16。

```figure
logsumexp-stability
```

## 動手做 (Build It)

### 步驟 1：示範浮點數的精度極限

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### 步驟 2：實作 naive 與 stable 的 softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### 步驟 3：實作 stable 的 log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### 步驟 4：實作 stable 的 cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### 步驟 5：Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## 實際使用 (Use It)

### Mixed precision 模擬

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### NaN/Inf 偵測

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

完整的實作與所有邊界情況的示範，請見 `code/numerical.py`。

## 交付 (Ship It)

這堂課會產出：
- `code/numerical.py`，包含 stable softmax、log-sum-exp、cross-entropy、gradient checking 與 mixed precision 模擬
- `outputs/prompt-numerical-debugger.md`，用於診斷訓練中的 NaN/Inf 與數值問題

這些穩定的實作會在 Phase 3 打造訓練迴圈時，以及 Phase 4 實作 attention 機制時再次出現。

## 練習 (Exercises)

1. **Catastrophic cancellation。** 在 float32 中，用 naive 公式 $\mathbb{E}[x^2] - \mathbb{E}[x]^2$ 計算 [1000000.0, 1000001.0, 1000002.0] 的 variance。再用 Welford 的 online 演算法計算。把誤差與真實的 variance (0.6667) 比較。

2. **找出精度極限。** 在 Python 中，找出最小的正 float32 值 `x`，使得 `1.0 + x == 1.0`。這就是 machine epsilon。驗證它與 `numpy.finfo(numpy.float32).eps` 相符。

3. **Log-sum-exp 的邊界情況。** 用以下情況測試你的 `logsumexp_stable` 函式：(a) 所有值相等，(b) 一個值遠大於其餘的，(c) 所有值都非常負 (-1000)。驗證它在 naive 版本失敗的地方給出正確的結果。

4. **對 neural network layer 做 gradient checking。** 實作單一 linear layer `y = Wx + b` 及其解析的 backward pass。用 `numerical_gradient` 驗證一個 3x2 weight matrix 的正確性。

5. **Loss scaling 實驗。** 模擬 float16 訓練：建立範圍在 [1e-9, 1e-3] 的隨機 gradient，轉成 float16，並衡量有多大比例變成零。接著套用 loss scaling（乘以 1024），轉成 float16，再縮放回來，並再次衡量變成零的比例。

### 提示 (Hints)

**練習 1。** 在平方之前，先用 `np.float32` 轉換這三個值，然後呼叫 `variance_naive` 和 `welford_variance`。float32 中接近 $10^{12}$ 的值，其間距遠大於 1，所以預期 naive 的結果會遠離 0.6667，而 Welford 的結果會很接近。

**練習 2。** 從 `x = np.float32(1.0)` 開始，當 `np.float32(1.0) + x != np.float32(1.0)` 時，持續把它減半。所有值都保持為 float32，然後把最後一個仍能改變總和的值與 `numpy.finfo(numpy.float32).eps` 比較，後者是 $2^{-23}$。

**練習 3。** 先預測每個答案。對 $n$ 個相等的值 $c$，結果是 $c + \ln n$。對一個大的值，結果接近那個值。對接近 -1000 的值，結果接近 $-1000 + \ln n$，而 `logsumexp_naive` 會回傳 `-inf`，因為每個指數都 underflow 成 0。

**練習 4。** 對於 $y = Wx + b$ 與 loss $L$，$W$ 的 gradient 是上游 gradient 與 $x$ 的 outer product。把 loss 包成扁平化 $W$ 的函式，呼叫 `numerical_gradient`，並把兩個結果都傳給 `check_gradient`。

**練習 5。** 用 `10 ** np.random.uniform(-9, -3, n)` 在 log 尺度上取樣大小，並用 `np.float16` 轉換。Float16 無法保留低於約 $6 \times 10^{-8}$ 的值，所以預期縮放前大約四分之一的值會變成零，而乘以 1024 之後幾乎沒有。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意思 |
|------|----------------|----------------------|
| IEEE 754 | 「浮點數標準」 | 定義二進位浮點格式、捨入規則與特殊值（inf、nan）的國際標準。每個現代 CPU 和 GPU 都實作了它。 |
| Machine epsilon | 「精度極限」 | 在特定浮點格式中，使 $1.0 + e \ne 1.0$ 的最小值 $e$。對 float32 約為 1.19e-7。 |
| Catastrophic cancellation | 「減法造成的精度損失」 | 把幾乎相等的浮點數相減時，有效數字互相抵銷，捨入雜訊主導了結果。 |
| Overflow | 「數字太大」 | 結果超過最大可表示的值，變成 inf。exp(89) 會讓 float32 overflow。 |
| Underflow | 「數字太小」 | 結果比最小可表示的正數更接近零，變成 0.0。exp(-104) 會讓 float32 underflow。 |
| Log-sum-exp trick | 「先減去最大值」 | 提出 $\exp(\max(x))$ 來計算 $\log(\sum \exp(x))$，以防止 overflow 與 underflow。用於 softmax、cross-entropy 與 log-probability 的運算。 |
| Stable softmax | 「不會爆炸的 softmax」 | 在取指數之前減去 $\max(\text{logits})$。數值上結果相同，且不可能 overflow。 |
| Gradient checking | 「驗證你的 backprop」 | 把 backpropagation 的解析 gradient 與 finite difference 的數值 gradient 比較，以抓出實作 bug。 |
| Mixed precision | 「Float16 forward、float32 backward」 | 對速度關鍵的運算使用較低精度的 float，對數值敏感的運算使用較高精度的 float。典型的加速是 2-3 倍。 |
| Loss scaling | 「防止 gradient underflow」 | 在 backprop 之前把 loss 乘上一個大的常數，讓 gradient 維持在 float16 可表示的範圍內，然後在更新 weight 之前除以同一個常數。 |
| bfloat16 | 「Brain floating point」 | Google 的 16 位元格式，有 8 個 exponent 位元（與 float32 範圍相同）和 7 個 mantissa 位元（精度比 float16 低）。訓練時的首選。 |
| Gradient clipping | 「限制 gradient norm」 | 縮放 gradient vector，讓它的 norm 不超過門檻。防止 exploding gradient 毀掉 weight。 |
| NaN | 「Not a Number」 | 由未定義運算（0/0、inf-inf、sqrt(-1)）產生的特殊浮點值。會傳播到所有後續的算術中。 |
| Inf | 「無限大」 | 由 overflow 或除以零產生的特殊浮點值。可以組合產生 NaN（inf - inf、inf * 0）。 |
| Numerical gradient | 「暴力求 derivative」 | 計算 $f(x+h)$ 與 $f(x-h)$ 再除以 $2h$ 來近似 derivative。很慢，但用於驗證很可靠。 |

## 延伸閱讀 (Further Reading)

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- 決定性的參考資料，內容稠密但完整
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- 為 float16 訓練引入 loss scaling 的 NVIDIA 論文
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- PyTorch 中 mixed precision 的實用指南
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- Google 為什麼為 TPU 選擇這個格式
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- 降低浮點數加總捨入誤差的演算法
