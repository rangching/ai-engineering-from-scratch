# JAX 入門 (Introduction to JAX)

> PyTorch 會修改 (mutate) tensor。TensorFlow 會建圖。JAX 則編譯純函數。最後這一種，會改變你思考 deep learning 的方式。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 03 Lessons 01-10, basic NumPy
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 使用 JAX 的 functional API（jax.numpy、jax.grad、jax.jit、jax.vmap）撰寫純函數式的神經網路程式
- 說明 PyTorch 的 eager mutation 與 JAX 的 functional 編譯模型之間的關鍵設計差異
- 套用 jit 編譯與 vmap 向量化，加速訓練迴圈，並與單純的 Python 做比較
- 在 JAX 中訓練一個簡單的網路，並對比明確的 state 管理與 PyTorch 物件導向的做法

## 問題 (The Problem)

你知道怎麼在 PyTorch 裡建構神經網路。你定義一個 `nn.Module`，呼叫 `.backward()`，執行 optimizer 的 step。它能運作。有數百萬人在用。

但 PyTorch 的 DNA 裡內建了一個限制：它在 Python 裡 eager 地追蹤運算，一次一個。每一個 `tensor + tensor` 都是一次獨立的 kernel 啟動。每個訓練步驟都會重新直譯同一段 Python 程式。這在你需要跨 2,048 顆 TPU 訓練一個 5400 億參數的模型之前，都還行得通。到了那時候，額外開銷就會要你的命。

Google DeepMind 用 JAX 訓練 Gemini。Anthropic 用 JAX 訓練 Claude。這些不是小規模的運算：它們是地球上最大的神經網路訓練任務。他們選擇 JAX，是因為它把你的訓練迴圈視為一個可編譯的程式，而不是一連串的 Python 呼叫。

JAX 是擁有三種超能力的 NumPy：自動微分、JIT 編譯到 XLA，以及自動向量化。你寫一個處理單一範例的函數。JAX 給你一個能處理整個 batch、計算 gradient、編譯成機器碼、並跨多個 device 執行的函數。而且完全不需要更動原本的函數。

## 核心概念 (The Concept)

### JAX 的哲學 (The JAX Philosophy)

JAX 是一個 functional framework。沒有 class，沒有可變 state，沒有 `.backward()` 方法。取而代之的是：

| PyTorch | JAX |
|---------|-----|
| 帶有 state 的 `nn.Module` class | 純函數：`f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager execution | 透過 XLA 做 JIT 編譯 |
| `for x in batch:` 手動迴圈 | `jax.vmap(f)` 自動向量化 |
| `DataParallel` / `FSDP` | `jax.pmap(f)` 自動平行化 |
| 可變的 `model.parameters()` | 不可變的 array pytree |

這不是風格偏好，而是編譯器的限制。JIT 編譯需要純函數：相同的輸入永遠產生相同的輸出，沒有副作用。正是這個限制，讓 100 倍的加速成為可能。

### jax.numpy：熟悉的外表 (jax.numpy: The Familiar Surface)

JAX 在加速器上重新實作了 NumPy API：

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

相同的函數名稱。相同的 broadcasting 規則。相同的切片語意。但 array 位於 GPU/TPU 上，而且每個運算都可以被編譯器追蹤。

一個關鍵的差異：JAX array 是不可變的。不能寫 `a[0] = 5`。要改成：`a = a.at[0].set(5)`。這會讓你彆扭一個星期，然後就豁然開朗：不可變性，正是讓 `grad`、`jit` 和 `vmap` 這類轉換可以組合的原因。

### jax.grad：Functional 自動微分 (jax.grad: Functional Autodiff)

PyTorch 把 gradient 附加在 tensor 上（`.grad`）。JAX 則把 gradient 附加在函數上。

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` 接收一個函數，並回傳一個計算 gradient 的新函數。沒有 `.backward()` 呼叫。沒有儲存在 tensor 上的 computation graph。gradient 只是另一個你可以呼叫、組合或 JIT 編譯的函數。

這可以任意組合：

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

二階導數。三階導數。Jacobian。Hessian。全都是透過組合 `grad` 做到的。PyTorch 也做得到（`torch.autograd.functional.hessian`），但那是後來硬加上去的。在 JAX 裡，這是基礎。

限制是：`grad` 只對純函數有效。裡面不能有 print 敘述（它們在 tracing 期間執行，而不是在實際執行時）。不能修改外部 state。沒有明確的 key 管理就不能產生亂數。

### jit：編譯到 XLA (jit: Compile to XLA)

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

第一次呼叫時，JAX 會追蹤 (trace) 這個函數：記錄發生了哪些運算，但不實際執行。然後它把這個 trace 交給 XLA（Accelerated Linear Algebra），也就是 Google 為 TPU 和 GPU 打造的編譯器。XLA 會融合運算、消除多餘的記憶體複製，並產生最佳化的機器碼。

之後的呼叫會完全跳過 Python。編譯後的程式碼在加速器上以 C++ 的速度執行。

JIT 有幫助的時候：
- 訓練步驟（同樣的運算重複幾千次）
- 推論（相同的模型，不同的輸入）
- 任何以相似 shape 的輸入被呼叫超過一次的函數

JIT 有害的時候：
- 帶有依賴數值的 Python 控制流程的函數（`if x > 0`，其中 x 是被追蹤的 array）
- 只執行一次的運算（編譯的額外開銷超過執行時間）
- 除錯（tracing 會隱藏實際的執行過程）

控制流程的限制是真實存在的。`jax.lax.cond` 取代 `if/else`。`jax.lax.scan` 取代 `for` 迴圈。這些不是可選的，而是編譯的代價。

### vmap：自動向量化 (vmap: Automatic Vectorization)

你寫一個處理單一範例的函數：

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` 把它提升成能處理一個 batch：

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` 的意思是：不對 `params` 做 batch（共用），對 `x` 的 axis 0 做 batch。沒有手動的 `for` 迴圈。不用 reshape。不用在各處傳遞 batch 維度。JAX 會找出 batch 維度，並把整個運算向量化。

這不是語法糖。`vmap` 產生融合的向量化程式碼，執行速度比 Python 迴圈快 10-100 倍。而且它能與 `jit` 和 `grad` 組合：

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

逐範例的 gradient。一行搞定。這在 PyTorch 裡不用一些 hack 幾乎做不到。

### pmap：跨 Device 的 Data Parallelism (pmap: Data Parallelism Across Devices)

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` 把函數複製到所有可用的 device（GPU/TPU）上，並切分 batch。在函數內部，`jax.lax.pmean` 和 `jax.lax.psum` 負責跨 device 同步 gradient。

Google 使用 `pmap`（以及它的後繼者 `shard_map`），在數千顆 TPU v5e 晶片上訓練 Gemini。程式設計模型是：寫出單一 device 的版本，用 `pmap` 包起來，完成。

### Pytree：通用的資料結構 (Pytrees: The Universal Data Structure)

JAX 操作的對象是「pytree」：list、tuple、dict 和 array 的巢狀組合。你的模型參數就是一個 pytree：

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

每一個 JAX 轉換（`grad`、`jit`、`vmap`）都知道如何走訪 pytree。`jax.tree.map(f, tree)` 會把 `f` 套用到每一個 leaf。optimizer 就是這樣一次更新所有參數的：

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

沒有 `.parameters()` 方法。沒有參數註冊。樹狀結構本身就是模型。

### Functional vs 物件導向 (Functional vs Object-Oriented)

PyTorch 把 state 存放在物件內部：

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX 使用帶有明確 state 的純函數：

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

params 是從外面傳進來的。什麼都沒有被儲存。什麼都沒有被修改。這讓每個函數都可以測試、可以組合、可以編譯。這也意味著你得自己管理 params，或是使用像 Flax 或 Equinox 這樣的函式庫。

### JAX 生態系 (The JAX Ecosystem)

JAX 給你基本元件。函式庫給你好用的介面：

| 函式庫 | 角色 | 風格 |
|---------|------|-------|
| **Flax** (Google) | 神經網路 layer | 帶有明確 state 的 `nn.Module` |
| **Equinox** (Patrick Kidger) | 神經網路 layer | 以 pytree 為基礎，Pythonic |
| **Optax** (DeepMind) | Optimizer + LR schedule | 可組合的 gradient 轉換 |
| **Orbax** (Google) | Checkpointing | 儲存/還原 pytree |
| **CLU** (Google) | 指標 + 記錄 | 訓練迴圈工具 |

Optax 是標準的 optimizer 函式庫。它把 gradient 轉換（Adam、SGD、clipping）與參數更新分開，讓組合變得輕而易舉：

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### 何時用 JAX，何時用 PyTorch (When to Use JAX vs PyTorch)

| 因素 | JAX | PyTorch |
|--------|-----|---------|
| TPU 支援 | 一等公民（Google 兩者都是自己做的） | 由社群維護（torch_xla） |
| GPU 支援 | 不錯（透過 XLA 的 CUDA） | 最佳（原生 CUDA） |
| 除錯 | 困難（tracing + 編譯） | 容易（eager，逐行） |
| 生態系 | 以研究為主（Flax、Equinox） | 龐大（HuggingFace、torchvision 等） |
| 招募 | 小眾（Google/DeepMind/Anthropic） | 主流（到處都是） |
| 大規模訓練 | 較優（XLA、pmap、mesh） | 不錯（FSDP、DeepSpeed） |
| 原型開發速度 | 較慢（functional 的額外負擔） | 較快（改了就走） |
| 正式環境推論 | TensorFlow Serving、Vertex AI | TorchServe、Triton、ONNX |
| 誰在用 | DeepMind (Gemini)、Anthropic (Claude) | Meta (Llama)、OpenAI (GPT)、Stability AI |

老實說：除非你有特定理由使用 JAX，否則就用 PyTorch。那些理由包括：能取得 TPU、需要逐範例的 gradient、大規模的多 device 訓練，或是在 Google/DeepMind/Anthropic 工作。

### JAX 中的亂數 (Random Numbers in JAX)

JAX 沒有全域的亂數 state。每個隨機運算都需要明確的 PRNG key：

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

一開始這很煩人。但它保證了跨 device 與跨編譯的可重現性，而 PyTorch 的 `torch.manual_seed` 在多 GPU 的情況下做不到這點。

```figure
batchnorm-effect
```

## 動手做 (Build It)

### 步驟 1：設定與資料 (Setup and Data)

我們將使用 JAX 和 Optax，在 MNIST 上訓練一個 3 層 MLP。784 個輸入，兩個 hidden layer 分別有 256 和 128 個 neuron，10 個輸出類別。

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### 步驟 2：初始化參數 (Initialize Parameters)

沒有 class。只是一個回傳 pytree 的函數：

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He 初始化，手動完成。從一個 seed 切出三個 PRNG key。每個 weight 都是巢狀 dict 裡的不可變 array。

### 步驟 3：Forward Pass

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

純函數。傳入 params，傳出預測。沒有 `self`，沒有儲存的 state。`loss_fn` 從零開始計算 cross-entropy：softmax、log、取負平均。

### 步驟 4：JIT 編譯的訓練步驟 (JIT-Compiled Training Step)

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` 在一次 pass 中同時回傳 loss 值和 gradient。`@jax.jit` decorator 把兩個函數都編譯到 XLA。第一次呼叫之後，每個訓練步驟都不需要碰到 Python。

### 步驟 5：訓練迴圈 (Training Loop)

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 個 epoch。測試 accuracy 約 97%。第一個 epoch 很慢（JIT 編譯）。第 2-10 個 epoch 很快。

注意少了什麼：沒有 `.zero_grad()`，沒有 `.backward()`，沒有 `.step()`。整個更新就是一次組合起來的函數呼叫。gradient 被算出、經過 Adam 轉換、再套用到參數上，全都在 `train_step` 裡面完成。

## 實際使用 (Use It)

### Flax：Google 的標準 (Flax: The Google Standard)

Flax 是最常見的 JAX 神經網路函式庫。它把 `nn.Module` 帶了回來，但搭配明確的 state 管理：

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

結構和 PyTorch 相同，但 `params` 與模型是分開的。`model.init()` 建立 params。`model.apply(params, x)` 執行 forward pass。模型物件本身沒有 state。

### Equinox：Pythonic 的替代方案 (Equinox: The Pythonic Alternative)

Equinox（作者 Patrick Kidger）把模型表示成 pytree：

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

模型本身就是一個 pytree。不需要 `.apply()`。參數只是模型的 leaf。這更貼近 JAX 的思考方式。

### Optax：可組合的 Optimizer (Optax: Composable Optimizers)

Optax 把 gradient 轉換與更新解耦：

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping、learning rate warmup、weight decay，全都組合成一串轉換。每個轉換看到 gradient、修改它，再傳給下一個。沒有單一龐大的 optimizer class。

## 交付 (Ship It)

**安裝：**

```bash
pip install jax jaxlib optax flax
```

GPU 支援：

```bash
pip install jax[cuda12]
```

TPU（Google Cloud）：

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**效能上的陷阱：**

- 第一次 JIT 呼叫很慢（編譯）。在 benchmark 之前先暖機。
- 避免在 JIT 內對 JAX array 使用 Python 迴圈。改用 `jax.lax.scan` 或 `jax.lax.fori_loop`。
- `jax.debug.print()` 可以在 JIT 內運作。一般的 `print()` 不行。
- 用 `jax.profiler` 或 TensorBoard 做效能分析。XLA 編譯可能會隱藏瓶頸。
- JAX 預設會預先配置 75% 的 GPU 記憶體。設定 `XLA_PYTHON_CLIENT_PREALLOCATE=false` 可以停用。

**Checkpointing：**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**這堂課產出：**
- `outputs/prompt-jax-optimizer.md` -- 一個 prompt，用來選擇正確的 JAX optimizer 設定
- `outputs/skill-jax-patterns.md` -- 一個 skill，涵蓋 JAX 中的 functional 模式

## 練習 (Exercises)

1. 為 MLP 加入 dropout。在 JAX 中，dropout 需要 PRNG key：把 key 穿過 forward pass，並為每個 dropout layer 切分出新的 key。比較有無 dropout 的測試 accuracy。

2. 用 `jax.vmap` 為 32 張 MNIST 影像的 batch 計算逐範例的 gradient。計算每個範例的 gradient norm。哪些範例的 gradient 最大？為什麼？

3. 把手動寫的 forward 函數換成通用的 `mlp_forward(params, x)`，可以處理任意層數。使用 `jax.tree.leaves` 自動判斷深度。

4. 分別在有和沒有 `@jax.jit` 的情況下，對訓練步驟做 benchmark。各計時 100 步。在你的硬體上加速幅度有多大？第一次呼叫的編譯額外開銷是多少？

5. 透過組合 `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))` 來實作 gradient clipping。分別在有和沒有 clipping 的情況下訓練。畫出訓練過程中的 gradient norm，觀察其效果。

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|----------------------|
| XLA | 「讓 JAX 變快的東西」 | Accelerated Linear Algebra：一種編譯器，會融合運算，並從 computation graph 產生最佳化的 GPU/TPU kernel |
| JIT | 「Just-in-time 編譯」 | JAX 在第一次呼叫時追蹤函數，編譯到 XLA，之後的呼叫就執行編譯好的版本 |
| Pure function | 「沒有副作用」 | 輸出只取決於輸入的函數：沒有全域 state、沒有 mutation、沒有明確 key 就沒有隨機性 |
| vmap | 「自動 batching」 | 把處理單一範例的函數轉換成處理 batch 的函數，不需要重寫 |
| pmap | 「自動平行化」 | 把函數複製到多個 device 上，並切分輸入 batch |
| Pytree | 「巢狀的 array dict」 | list、tuple、dict 和 array 的任何巢狀結構，JAX 都能走訪並轉換 |
| Tracing | 「記錄運算」 | JAX 用抽象值執行函數以建立 computation graph，而不計算真正的結果 |
| Functional autodiff | 「函數的 grad」 | 透過轉換函數來計算導數，而不是在 tensor 上附加 gradient 儲存空間 |
| Optax | 「JAX 的 optimizer 函式庫」 | 一個可組合的 gradient 轉換函式庫（Adam、SGD、clipping、scheduling），可以串接在一起 |
| Flax | 「JAX 的 nn.Module」 | Google 為 JAX 打造的神經網路函式庫，加入 layer 抽象，同時讓 state 保持明確 |

## 延伸閱讀 (Further Reading)

- JAX documentation: https://jax.readthedocs.io/ -- 官方文件，有關於 grad、jit 和 vmap 的優秀教學
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) -- 解釋設計哲學的原始論文
- Flax documentation: https://flax.readthedocs.io/ -- Google 為 JAX 打造的神經網路函式庫
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) -- Flax 之外 Pythonic 的替代方案
- DeepMind, "Optax: composable gradient transformation and optimisation" -- 標準的 optimizer 函式庫
- "You Don't Know JAX" (Colin Raffel, 2020) -- 出自 T5 作者之一之手、關於 JAX 陷阱與模式的實用指南
