# Vector、Matrix 與運算 (Vectors, Matrices & Operations)

> 每個 neural network 都只是多了幾個步驟的 matrix multiplication。

**Type:** Build
**Languages:** Python, Julia
**Prerequisites:** Phase 1, Lesson 01 (Linear Algebra Intuition)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 打造一個 Matrix class，支援 element-wise 運算、matrix multiplication、transpose、determinant 與 inverse
- 區分 element-wise multiplication 與 matrix multiplication，並說明各自適用的時機
- 只用自己從零寫的 Matrix class，實作單一 dense neural network layer（`relu(W @ x + b)`）
- 說明 broadcasting 的規則，以及 neural network 框架如何做 bias 加法

## 問題 (The Problem)

你想打造一個 neural network。你讀程式碼，看到這一行：

```
output = activation(weights @ input + bias)
```

那個 `@` 就是 matrix multiplication。`weights` 是 matrix，`input` 是 vector。如果你不知道這些運算在做什麼，這行就像魔法。如果你知道，它就是一層 layer 的完整 forward pass，只用三個運算。

模型處理的每張圖片都是由像素值組成的 matrix。每個 word embedding 都是 vector。每個 neural network 的每一層都是 matrix transformation。你若不熟 matrix 運算，就無法打造 AI 系統，就像不懂變數就寫不了程式。

這堂課從零開始，幫你練到熟。

## 核心概念 (The Concept)

### Vector：有順序的數字清單 (Vectors: ordered lists of numbers)

Vector 是一串數字，有方向也有 magnitude。在 AI 裡，vector 代表資料點、feature 或參數。

```
v = [3, 4]        -- a 2D vector
w = [1, 0, -2]    -- a 3D vector
```

2D vector `[3, 4]` 指向平面上的座標 (3, 4)。它的長度（magnitude）是 5（3-4-5 三角形）。

### Matrix：數字組成的格子 (Matrices: grids of numbers)

Matrix 是 2D 的格子，有列和欄。一個 $m \times n$ 的 matrix 有 $m$ 列、$n$ 欄。

```math
A = \begin{bmatrix} 1 & 2 & 3 \\ 4 & 5 & 6 \end{bmatrix} \quad \text{2x3 matrix (2 rows, 3 columns)}
```

在 neural network 裡，weight matrix 把輸入 vector 轉成輸出 vector。一個有 784 個輸入、128 個輸出的 layer，會用一個 128x784 的 weight matrix。

### 為什麼 shape 很重要 (Why shapes matter)

Matrix multiplication 有一條嚴格的規則：`(m x n) @ (n x p) = (m x p)`。內側的維度必須相符。

```
(128 x 784) @ (784 x 1) = (128 x 1)
  weights       input       output

Inner dimensions: 784 = 784  -- valid
```

如果你在 PyTorch 看到 shape mismatch 錯誤，原因就在這裡。

### 運算總覽 (The operations map)

| 運算 | 做什麼 | 在 neural network 的用途 |
|-----------|-------------|-------------------|
| Addition | 逐元素合併 | 把 bias 加到輸出 |
| Scalar multiply | 縮放每個元素 | Learning rate * gradients |
| Matrix multiply | 轉換 vector | Layer 的 forward pass |
| Transpose | 互換列與欄 | Backpropagation |
| Determinant | 用單一數字概括 | 檢查是否可逆 |
| Inverse | 還原一個 transformation | 解 linear system |
| Identity | 什麼都不做的 matrix | 初始化、residual connection |

### Element-wise 與 matrix multiplication 的差別 (Element-wise vs matrix multiplication)

初學者常常在這裡搞混。

Element-wise：相同位置相乘。兩個 matrix 的 shape 必須一樣。

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} * \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 5 & 12 \\ 21 & 32 \end{bmatrix}
```

Matrix multiplication：列與欄的 dot product。內側維度必須相符。

```math
\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} @ \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} = \begin{bmatrix} 1 \cdot 5 + 2 \cdot 7 & 1 \cdot 6 + 2 \cdot 8 \\ 3 \cdot 5 + 4 \cdot 7 & 3 \cdot 6 + 4 \cdot 8 \end{bmatrix} = \begin{bmatrix} 19 & 22 \\ 43 & 50 \end{bmatrix}
```

運算不同，結果不同，規則也不同。

### Broadcasting (Broadcasting)

當你把 bias vector 加到一個輸出 matrix 上時，兩者的 shape 並不相同。Broadcasting 會把較小的 array 拉伸到合適的大小。

```
| 1  2  3 |   +   [10, 20, 30]
| 4  5  6 |

Broadcasting stretches the vector across rows:

| 1  2  3 |   | 10  20  30 |   | 11  22  33 |
| 4  5  6 | + | 10  20  30 | = | 14  25  36 |
```

現代框架都會自動這麼做。理解它，可以避免「shape 看起來不對，程式卻能跑」的困惑。

```figure
vector-projection
```

## 動手做 (Build It)

### 步驟 1：Vector class

```python
class Vector:
    def __init__(self, data):
        self.data = list(data)
        self.size = len(self.data)

    def __repr__(self):
        return f"Vector({self.data})"

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.data, other.data)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.data, other.data)])

    def __mul__(self, scalar):
        return Vector([x * scalar for x in self.data])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.data, other.data))

    def magnitude(self):
        return sum(x ** 2 for x in self.data) ** 0.5
```

### 步驟 2：具備核心運算的 Matrix class

```python
class Matrix:
    def __init__(self, data):
        self.data = [list(row) for row in data]
        self.rows = len(self.data)
        self.cols = len(self.data[0])
        self.shape = (self.rows, self.cols)

    def __repr__(self):
        rows_str = "\n  ".join(str(row) for row in self.data)
        return f"Matrix({self.shape}):\n  {rows_str}"

    def __add__(self, other):
        return Matrix([
            [self.data[i][j] + other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def __sub__(self, other):
        return Matrix([
            [self.data[i][j] - other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def scalar_multiply(self, scalar):
        return Matrix([
            [self.data[i][j] * scalar for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def element_wise_multiply(self, other):
        return Matrix([
            [self.data[i][j] * other.data[i][j] for j in range(self.cols)]
            for i in range(self.rows)
        ])

    def matmul(self, other):
        return Matrix([
            [
                sum(self.data[i][k] * other.data[k][j] for k in range(self.cols))
                for j in range(other.cols)
            ]
            for i in range(self.rows)
        ])

    def transpose(self):
        return Matrix([
            [self.data[j][i] for j in range(self.rows)]
            for i in range(self.cols)
        ])

    def determinant(self):
        if self.shape == (1, 1):
            return self.data[0][0]
        if self.shape == (2, 2):
            return self.data[0][0] * self.data[1][1] - self.data[0][1] * self.data[1][0]
        det = 0
        for j in range(self.cols):
            minor = Matrix([
                [self.data[i][k] for k in range(self.cols) if k != j]
                for i in range(1, self.rows)
            ])
            det += ((-1) ** j) * self.data[0][j] * minor.determinant()
        return det

    def inverse_2x2(self):
        det = self.determinant()
        if det == 0:
            raise ValueError("Matrix is singular, no inverse exists")
        return Matrix([
            [self.data[1][1] / det, -self.data[0][1] / det],
            [-self.data[1][0] / det, self.data[0][0] / det]
        ])

    @staticmethod
    def identity(n):
        return Matrix([
            [1 if i == j else 0 for j in range(n)]
            for i in range(n)
        ])
```

### 步驟 3：看它實際運作

```python
A = Matrix([[1, 2], [3, 4]])
B = Matrix([[5, 6], [7, 8]])

print("A + B =", (A + B).data)
print("A @ B =", A.matmul(B).data)
print("A^T =", A.transpose().data)
print("det(A) =", A.determinant())
print("A^-1 =", A.inverse_2x2().data)

I = Matrix.identity(2)
print("A @ A^-1 =", A.matmul(A.inverse_2x2()).data)
```

### 步驟 4：連結到 neural network

```python
import random

inputs = Matrix([[0.5], [0.8], [0.2]])
weights = Matrix([
    [random.uniform(-1, 1) for _ in range(3)]
    for _ in range(2)
])
bias = Matrix([[0.1], [0.1]])

def relu_matrix(m):
    return Matrix([[max(0, val) for val in row] for row in m.data])

pre_activation = weights.matmul(inputs) + bias
output = relu_matrix(pre_activation)

print(f"Input shape: {inputs.shape}")
print(f"Weight shape: {weights.shape}")
print(f"Output shape: {output.shape}")
print(f"Output: {output.data}")
```

這就是單一 dense layer：`output = relu(W @ x + b)`。每個 neural network 裡的每個 dense layer，做的都是這件事。

## 實際使用 (Use It)

NumPy 用更少的程式碼就能完成上面所有事，而且快好幾個數量級。

```python
import numpy as np

A = np.array([[1, 2], [3, 4]])
B = np.array([[5, 6], [7, 8]])

print("A + B =\n", A + B)
print("A * B (element-wise) =\n", A * B)
print("A @ B (matrix multiply) =\n", A @ B)
print("A^T =\n", A.T)
print("det(A) =", np.linalg.det(A))
print("A^-1 =\n", np.linalg.inv(A))
print("I =\n", np.eye(2))

inputs = np.random.randn(3, 1)
weights = np.random.randn(2, 3)
bias = np.array([[0.1], [0.1]])
output = np.maximum(0, weights @ inputs + bias)

print(f"\nNeural network layer: {weights.shape} @ {inputs.shape} = {output.shape}")
print(f"Output:\n{output}")
```

Python 的 `@` 運算子會呼叫 `__matmul__`。NumPy 用 C 與 Fortran 寫成的最佳化 BLAS 常式來實作它。數學一樣，速度快 100 倍。

NumPy 中的 broadcasting：

```python
matrix = np.array([[1, 2, 3], [4, 5, 6]])
bias = np.array([10, 20, 30])
print(matrix + bias)
```

NumPy 會自動把 1D 的 bias 廣播到兩列上。每個 neural network 框架的 bias 加法都是這樣運作的。

## 交付 (Ship It)

這堂課會產出一份 prompt，透過幾何直覺來教 matrix 運算。請見 `outputs/prompt-matrix-operations.md`。

這裡建好的 Matrix class，是我們在 Phase 3, Lesson 10 要打造的迷你 neural network 框架的基礎。

## 練習 (Exercises)

1. **驗證 inverse。** 計算 `A @ A.inverse_2x2()`，確認得到 identity matrix。用三個不同的 2x2 matrix 試試。當 determinant 為零時會發生什麼事？

2. **實作 3x3 inverse。** 擴充 Matrix class，用 adjugate 方法計算 3x3 matrix 的 inverse。拿 NumPy 的 `np.linalg.inv` 來比對測試。

3. **打造兩層網路。** 只用你的 Matrix class（不用 NumPy），建立兩層 neural network：input (3) -> hidden (4) -> output (2)。初始化隨機 weight，跑一次 forward pass，並驗證所有 shape 都正確。

### 提示 (Hints)

**練習 1。** 對三個 matrix 計算 `A @ A.inverse_2x2()`，並把每一格與 identity 比較，容許誤差為 `1e-9`。對於 determinant 為 0 的 matrix，例如 [[1, 2], [2, 4]]，`inverse_2x2` 會丟出 `ValueError`，這正是預期的結果。

**練習 2。** 計算 3x3 的 cofactor matrix，轉置後得到 adjugate，再把每一格除以 `determinant()`。檢查 `A @ A_inv` 會得到 identity，且每一格都與 `np.linalg.inv(A)` 在 `1e-9` 內相符。

**練習 3。** 相乘之前，先寫下每個 matrix 的 shape：像 `demo_neural_network_layer` 一樣，`W1` 是 4x3、`W2` 是 2x4。3x1 的輸入必須得到 4x1 的 hidden vector 和 2x1 的輸出。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意思 |
|------|----------------|----------------------|
| Vector | 「一支箭頭」 | 有順序的數字清單。在 AI 裡：高維空間中的一個點。 |
| Matrix | 「一張數字表」 | 一個 linear transformation。把 vector 從一個空間映射到另一個空間。 |
| Matrix multiply | 「就是把數字相乘」 | 第一個 matrix 的每一列與第二個 matrix 的每一欄做 dot product。順序很重要。 |
| Transpose | 「翻過來」 | 互換列與欄。把 $m \times n$ 的 matrix 變成 $n \times m$。在 backpropagation 中很關鍵。 |
| Determinant | 「從 matrix 算出來的某個數字」 | 衡量 matrix 把面積（2D）或體積（3D）縮放多少。為零代表這個 transformation 壓扁了一個維度。 |
| Inverse | 「把 matrix 還原」 | 能反轉 transformation 的 matrix。只有在 determinant 不為零時才存在。 |
| Identity matrix | 「無聊的 matrix」 | 相當於乘以 1 的 matrix。用在 residual connection (ResNets)。 |
| Broadcasting | 「自動修正 shape 的魔法」 | 沿著缺少的維度重複，把較小的 array 拉伸到與較大的相符。 |
| Element-wise | 「一般的乘法」 | 相同位置相乘。兩個 array 必須有相同的 shape（或可以 broadcast）。 |

## 延伸閱讀 (Further Reading)

- [3Blue1Brown: Essence of Linear Algebra](https://www.3blue1brown.com/topics/linear-algebra) - 這裡涵蓋的每個運算的視覺化直覺
- [NumPy documentation on broadcasting](https://numpy.org/doc/stable/user/basics.broadcasting.html) - NumPy 遵循的確切規則
- [Stanford CS229 Linear Algebra Review](http://cs229.stanford.edu/section/cs229-linalg.pdf) - 針對 ML 的線性代數精簡參考資料
