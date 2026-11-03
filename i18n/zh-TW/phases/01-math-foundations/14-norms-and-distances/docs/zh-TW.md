# Norm 與距離 (Norms and Distances)

> 你的距離函數定義了「相似」的意思。選錯了，下游的一切都會壞掉。

**Type:** Build
**Language:** Python
**Prerequisites:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors, Matrices & Operations)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 從零實作 L1、L2、cosine、Mahalanobis、Jaccard 與 edit distance 函式
- 為特定的 ML 任務選擇合適的距離度量，並說明其他選項為何行不通
- 把 L1 與 L2 norm 連結到 LASSO 與 Ridge regularization，以及它們的幾何限制區域
- 示範同一份資料集在不同度量下，如何產生不同的最近鄰

## 問題 (The Problem)

你有兩個 vector。也許是 word embedding，也許是使用者 profile，也許是像素陣列。你需要知道：它們有多接近？

答案完全取決於你選哪個距離函數。兩個資料點在某個度量下可能是最近鄰，在另一個度量下卻相距甚遠。你的 KNN 分類器、推薦引擎、vector database、clustering 演算法、loss function，全都取決於這個選擇。選錯了，你的模型就是在最佳化錯誤的東西。

沒有放諸四海皆準的最佳距離。L2 適用於空間資料。Cosine similarity 主宰 NLP。Jaccard 處理集合。Edit distance 處理字串。Mahalanobis 考慮相關性。Wasserstein 搬移機率質量。每一個都編碼了對「相似」意義的不同假設。

這堂課從零打造每個主要的距離函數，告訴你各自何時是合適的工具，並示範同一份資料如何因為你使用的度量不同，而產生完全不同的最近鄰。

## 核心概念 (The Concept)

### Norm：衡量 vector 的大小 (Norms: measuring vector magnitude)

Norm 衡量一個 vector 的「大小」。兩個 vector 之間的每個距離函數，都可以寫成它們差的 norm：$d(a, b) = \lVert a - b \rVert$。所以理解 norm，就是理解距離。

### L1 Norm（Manhattan distance） (L1 Norm (Manhattan distance))

L1 norm 加總所有分量的絕對值。

```math
\lVert x \rVert_1 = |x_1| + |x_2| + \cdots + |x_n|
```

之所以稱為 Manhattan distance，是因為它衡量的是你在城市街區格網上走了多遠，而你只能沿著軸移動，不能走對角線。

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

何時使用 L1：
- 高維稀疏資料（文字 feature、one-hot encoding）
- 當你想要對 outlier 有穩健性時（單一巨大的差異不會主導結果）
- Feature selection 問題（L1 regularization 促進稀疏性）

與 L1 regularization (Lasso) 的關聯：把 $\lVert w \rVert_1$ 加到 loss function，會懲罰 weight 絕對值的總和。這會把小的 weight 推到恰好為零，自動執行 feature selection。L1 懲罰在 weight 空間中產生菱形的限制區域，而菱形的角落位於軸上，也就是某些 weight 為零的地方。

與 loss function 的關聯：Mean Absolute Error (MAE) 是預測與目標之間平均的 L1 距離。它對所有誤差施以線性懲罰，因此相較於 MSE，對 outlier 更穩健。

### L2 Norm（Euclidean distance） (L2 Norm (Euclidean distance))

L2 norm 是直線距離。各分量平方和的平方根。

```math
\lVert x \rVert_2 = \sqrt{x_1^2 + x_2^2 + \cdots + x_n^2}
```

這就是你在幾何課學到的距離。$n$ 維中的畢氏定理。

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

何時使用 L2：
- 低到中維度的連續資料
- 當 feature 的尺度相近時
- 實體距離（空間資料、感測器讀數）
- 像素層級的影像相似度

與 L2 regularization (Ridge) 的關聯：把 $\lVert w \rVert_2^2$ 加到 loss function，會懲罰大的 weight。和 L1 不同，它不會把 weight 推到零。它按比例把所有 weight 縮向零。L2 懲罰產生圓形的限制區域，所以軸上沒有角落。Weight 會變小，但很少恰好為零。

與 loss function 的關聯：Mean Squared Error (MSE) 是 L2 距離平方的平均。平方對大誤差的懲罰比對小誤差更重。

```math
\begin{aligned}
\text{MAE (L1 loss):} \quad & |y - \hat{y}| && \text{Linear penalty. Robust to outliers.} \\
\text{MSE (L2 loss):} \quad & (y - \hat{y})^2 && \text{Quadratic penalty. Sensitive to outliers.}
\end{aligned}
```

### Lp Norm：一般化的家族 (Lp Norms: the general family)

L1 和 L2 是 Lp norm 的特例：

```math
\lVert x \rVert_p = (|x_1|^p + |x_2|^p + \cdots + |x_n|^p)^{1/p}
```

不同的 $p$ 值會產生不同形狀的「單位球」（與原點距離至多為 1 的所有點的集合；下面的形狀是它們的邊界）：

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity Norm（Chebyshev distance） (L-infinity Norm (Chebyshev distance))

當 $p$ 趨近無限大時，Lp norm 收斂到絕對值最大的分量。

```math
\lVert x \rVert_\infty = \max(|x_1|, |x_2|, \ldots, |x_n|)
```

兩點之間的距離，由它們差異最大的那個單一維度決定。其他所有維度都被忽略。

```math
\begin{aligned}
\text{Point } A &= (1, 1) \\
\text{Point } B &= (4, 5) \\[6pt]
L_\infty \text{ distance} &= \max(|4-1|, |5-1|) = \max(3, 4) = 4
\end{aligned}
```

何時使用 L-infinity：
- 當任何單一維度的最壞情況偏差很重要時
- 棋盤遊戲（西洋棋的國王走的是 L-infinity：往任何方向一步的成本都是 1）
- 製造公差（每個維度都必須在規格內）

### Cosine Similarity 與 Cosine Distance (Cosine Similarity and Cosine Distance)

Cosine similarity 衡量兩個 vector 之間的夾角，忽略它們的 magnitude。

```math
\text{cos\_sim}(a, b) = \frac{a \cdot b}{\lVert a \rVert_2 \, \lVert b \rVert_2}
```

它的範圍從 -1（方向相反）到 +1（方向相同）。互相垂直的 vector，cosine similarity 為 0。

Cosine distance 把它轉換成距離：cosine_distance = 1 - cosine_similarity。範圍從 0（方向相同）到 2（方向相反）。

```math
\begin{aligned}
& a = (1, 0) \qquad b = (1, 1) \\[6pt]
& \text{cos\_sim} = \frac{1 \cdot 1 + 0 \cdot 1}{1 \cdot \sqrt{2}} = \frac{1}{\sqrt{2}} = 0.707 \\
& \text{cos\_dist} = 1 - 0.707 = 0.293
\end{aligned}
```

為什麼 cosine 主宰 NLP 與 embedding：在文字中，文件長度不應該影響相似度。一份關於貓的文件，長度是另一份關於貓的文件的兩倍，仍然應該算「相似」。Cosine similarity 忽略 magnitude（長度），只關心方向。兩份詞彙分佈相同但長度不同的文件，會指向同一個方向，得到 cosine similarity 1.0。

何時使用 cosine similarity：
- 文字相似度（TF-IDF vector、word embedding、sentence embedding）
- 任何 magnitude 是雜訊、方向才是訊號的領域
- 推薦系統（使用者偏好 vector）
- Embedding 搜尋（vector database 幾乎都使用 cosine 或 dot product）

### Dot Product 相似度與 Cosine 相似度 (Dot Product Similarity vs Cosine Similarity)

兩個 vector 的 dot product 是：

```math
\begin{aligned}
a \cdot b &= a_1 b_1 + a_2 b_2 + \cdots + a_n b_n \\
&= \lVert a \rVert \, \lVert b \rVert \cos(\text{angle})
\end{aligned}
```

Cosine similarity 是 dot product 除以兩者 magnitude 的 normalization。當兩個 vector 都已經是單位 normalize（magnitude = 1）時，dot product 與 cosine similarity 完全相同。

```math
\text{If } \lVert a \rVert = 1 \text{ and } \lVert b \rVert = 1: \quad a \cdot b = \cos(\text{angle between } a \text{ and } b)
```

它們不同的時候：dot product 包含 magnitude 資訊。Magnitude 較大的 vector 會得到較高的 dot product 分數。這在某些檢索系統中很重要，你想讓「熱門」的項目排名較前。Magnitude 就充當隱含的品質或重要性訊號。

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

實務上：
- 想要純粹的方向相似度時，用 cosine similarity
- 當 magnitude 帶有意義的資訊時，用 dot product
- 許多 vector database（Pinecone、Weaviate、Qdrant）讓你在兩者之間選擇
- 如果你的 embedding 已做過 L2-normalize，選哪個都沒差

### Mahalanobis Distance (Mahalanobis Distance)

Euclidean distance 把所有維度一視同仁。但如果你的 feature 彼此相關，或尺度不同，L2 會給出誤導的結果。

Mahalanobis distance 考慮了資料的 covariance 結構。

```math
d_M(x, y) = \sqrt{(x - y)^T S^{-1} (x - y)}
```

其中 $S$ 是資料的 covariance matrix。

直覺上：Mahalanobis distance 先把資料去相關並 normalize（whitening），然後在轉換後的空間中計算 L2 距離。如果 $S$ 是 identity matrix（feature 不相關且 variance 為 1），Mahalanobis distance 就簡化成 Euclidean distance。

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

何時使用 Mahalanobis distance：
- Outlier 偵測（與 mean 的 Mahalanobis distance 很大的點就是 outlier）
- 當 feature 有不同尺度與相關性時的分類
- 當你有足夠的資料能估計出可靠的 covariance matrix 時
- 製造業的品質管制（多變量製程監控）

### Jaccard Similarity（用於集合） (Jaccard Similarity (for sets))

Jaccard similarity 衡量兩個集合之間的重疊程度。

```math
J(A, B) = \frac{|A \cap B|}{|A \cup B|}
```

範圍從 0（沒有重疊）到 1（相同的集合）。Jaccard distance = 1 - Jaccard similarity。

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

何時使用 Jaccard：
- 比較標籤、類別或 feature 的集合
- 根據詞彙是否出現（而非頻率）的文件相似度
- 近似重複偵測（Jaccard 的 MinHash 近似）
- 比較二元 feature vector（有/無資料）
- 評估 segmentation 模型（Intersection over Union = Jaccard）

### Edit Distance（Levenshtein distance） (Edit Distance (Levenshtein Distance))

Edit distance 計算把一個字串轉成另一個字串所需的最少單一字元運算次數。運算有：插入、刪除或替換。

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

用 dynamic programming 計算。填一個 matrix，其中第 $(i, j)$ 項是字串 A 前 $i$ 個字元與字串 B 前 $j$ 個字元之間的 edit distance。

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

何時使用 edit distance：
- 拼字檢查與更正
- DNA 序列比對（搭配加權的運算）
- 模糊字串比對
- 雜亂文字資料的去重

### KL Divergence（不是距離，但被當作距離使用） (KL Divergence (not a distance, but used like one))

KL divergence 衡量一個機率分佈與另一個有多不同。已在 Lesson 09 介紹過，但它屬於這個討論，因為人們把它當作「距離」使用，儘管它不是。

```math
D_{KL}(P \,\|\, Q) = \sum_x p(x) \log\frac{p(x)}{q(x)}
```

關鍵性質：KL divergence 不是對稱的。

```math
D_{KL}(P \,\|\, Q) \ne D_{KL}(Q \,\|\, P)
```

這代表它不滿足距離度量的基本要求。它也不滿足三角不等式。它是 divergence，不是距離。

Forward KL $(D_{KL}(P \Vert Q))$ 是「追求 mean」：$Q$ 試圖涵蓋 $P$ 的所有 mode。
Reverse KL $(D_{KL}(Q \Vert P))$ 是「追求 mode」：$Q$ 專注於 $P$ 的單一 mode。

你會在這些地方看到 KL divergence：
- VAE（ELBO 中的 KL 項把 latent 分佈推向 prior）
- Knowledge distillation（student 試圖匹配 teacher 的分佈）
- RLHF（KL 懲罰讓微調後的模型接近 base model）
- Policy gradient 方法（限制 policy 的更新幅度）

### Wasserstein Distance（Earth Mover's Distance） (Wasserstein Distance (Earth Mover's Distance))

Wasserstein distance 衡量把一個機率分佈轉換成另一個所需的最小「功」。可以這樣想：如果一個分佈是一堆土，另一個是一個坑，你得搬多少土、搬多遠？

```math
W(P, Q) = \inf_{\gamma} \mathbb{E}[d(x, y)] \quad \text{(over all transport plans } \gamma\text{)}
```

對於 1D 分佈，它簡化為累積分佈函數 (CDF) 絕對差的積分：

```math
W_1(P, Q) = \int |\text{CDF}_P(x) - \text{CDF}_Q(x)| \, dx
```

為什麼 Wasserstein 很重要：
- 它是真正的度量（對稱，滿足三角不等式）
- 即使分佈沒有重疊，它也能提供 gradient（KL divergence 會變成無限大）
- 這個性質讓它成為 Wasserstein GAN (WGAN) 的核心，WGAN 解決了原始 GAN 訓練不穩定的問題

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

何時使用 Wasserstein：
- GAN 訓練（WGAN、WGAN-GP）
- 比較可能沒有重疊的分佈
- Optimal transport 問題
- 影像檢索（比較顏色 histogram）

### 為什麼不同任務需要不同的距離 (Why Different Tasks Need Different Distances)

| 任務 | 最佳距離 | 原因 |
|------|--------------|-----|
| 文字相似度 | Cosine | Magnitude 是雜訊，方向才是意義 |
| 影像像素比較 | L2 | 空間關係很重要，feature 的尺度相近 |
| 稀疏高維 feature | L1 | 穩健，不會放大罕見的大差異 |
| 集合重疊（標籤、類別） | Jaccard | 資料天生是集合值，不是 vector |
| 字串比對 | Edit distance | 運算對應人類編輯的直覺 |
| Outlier 偵測 | Mahalanobis | 考慮 feature 的相關性與尺度 |
| 比較分佈 | KL divergence | 衡量用 $Q$ 取代 $P$ 所損失的資訊 |
| GAN 訓練 | Wasserstein | 即使分佈沒有重疊也能提供 gradient |
| Embedding（vector DB） | Cosine 或 dot product | Embedding 被訓練成用方向編碼意義 |
| 推薦 | Dot product | Magnitude 可以編碼熱門程度或信心 |
| DNA 序列 | 加權 edit distance | 替換成本依核苷酸配對而異 |
| 製造品管 | L-infinity | 任何維度的最壞情況偏差都很重要 |

### 與 Loss Function 的關聯 (Connection to Loss Functions)

Loss function 就是套用在預測與目標上的距離函數。

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### 與 Regularization 的關聯 (Connection to Regularization)

Regularization 把對 weight 的 norm 懲罰加到 loss function 上。

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

為什麼 L1 會產生稀疏性而 L2 不會：想像 2D weight 空間中的限制區域。L1 是菱形，L2 是圓形。Loss function 的等高線（橢圓）最有可能在菱形的角落接觸它，也就是其中一個 weight 為零的地方。它們在圓形上接觸的是平滑的點，此處兩個 weight 都不為零。

### 最近鄰搜尋 (Nearest Neighbor Search)

每個距離函數都隱含了一個最近鄰搜尋問題：給定一個查詢點，在資料集中找出最接近的點。

在一個有 $n$ 個點、$d$ 個維度的資料集中，精確的最近鄰搜尋每次查詢是 $O(n \cdot d)$。對大型資料集來說，這太慢了。

Approximate Nearest Neighbor (ANN) 演算法以少量的準確度，換取大幅的速度提升：

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) 是現代 vector database 中的主流演算法。它建立一個多層的圖，其中每個節點連接到它的近似最近鄰。搜尋從頂層（稀疏、長跳躍）開始，往下到底層（密集、短跳躍）。

```figure
norm-unit-balls
```

## 動手做 (Build It)

### 步驟 1：所有 norm 與距離函數

完整的實作請見 `code/distances.py`。每個函數都只用基本的 Python 數學從零打造。

### 步驟 2：同樣的資料、不同的距離、不同的鄰居

`distances.py` 裡的示範建立一個資料集、挑一個查詢點，並展示最近鄰如何依距離度量而改變。在 L1 下「最近」的點，在 L2 或 cosine 下未必最近。

### 步驟 3：Embedding 相似度搜尋

程式碼包含一個模擬的 embedding 相似度搜尋，用 cosine similarity 與 L2 distance 找出與查詢最相似的「文件」，展示排名可能不同。

## 實際使用 (Use It)

最常見的實務用途：在 vector database 中尋找相似的項目。

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

當你呼叫 `model.encode(text)` 然後搜尋 vector database 時，底層發生的就是這件事。Embedding 模型把文字對應成 vector。Vector database 計算你的查詢 vector 與每個已儲存 vector 之間的 cosine similarity（或 dot product），並用 ANN 演算法避免逐一檢查所有 vector。

## 練習 (Exercises)

1. 計算 (1, 2, 3) 與 (4, 0, 6) 之間的 L1、L2 與 L-infinity 距離。驗證對任何一對點，$L_\infty \le L_2 \le L_1$ 永遠成立。證明為什麼這個順序是必然的。

2. 建立兩個 vector，其 cosine similarity 很高（> 0.9），但 L2 距離很大（> 10）。從幾何上解釋發生了什麼事。接著建立兩個 vector，其 cosine similarity 很低（< 0.3），但 L2 距離很小（< 0.5）。

3. 實作一個函式，接收一個資料集與一個查詢點，回傳在 L1、L2、cosine 與 Mahalanobis 距離下的最近鄰。找出一個資料集，讓這四種度量對哪個點最近的看法都不一致。

4. 用 CDF 方法手算 [0.5, 0.5, 0, 0] 與 [0, 0, 0.5, 0.5] 之間的 Wasserstein distance。接著計算 [0.25, 0.25, 0.25, 0.25] 與 [0, 0, 0.5, 0.5] 之間的。哪個比較大？為什麼？

5. 實作 MinHash 來近似 Jaccard similarity。產生 100 個隨機集合，計算所有配對的精確 Jaccard，並與使用 50、100 和 200 個 hash 函數的 MinHash 近似值比較。畫出近似誤差。

### 提示 (Hints)

**練習 1。** 差向量是 (-3, 2, -3)，所以用 `l1_norm`、`l2_norm` 和 `linf_norm` 計算這個向量的三個 norm。證明時，比較最大的 $\lvert d_i \rvert$ 與平方和，並比較平方和與絕對值總和的平方。

**練習 2。** Cosine similarity 忽略長度，所以從兩個指向幾乎相同方向的 vector 開始，並讓其中一個長得多。對第二種情況，使用原點附近兩個指向不同方向的短 vector，然後用 `cosine_similarity` 和 `l2_distance` 檢查兩個條件。

**練習 3。** 對 `l1_distance`、`l2_distance`、`cosine_distance` 和 `mahalanobis_distance` 各呼叫一次 `find_nearest_neighbor(query, dataset, distance_fn)`。`demo_different_neighbors` 展示了一個各度量意見不一的資料集，所以先讀它，再調整這些點，直到四個答案都不同。

**練習 4。** 寫出每個分佈的累積和，然後把兩個累積和的絕對差加總起來。用 `wasserstein_1d(p, q)` 檢查兩個手算結果，接著用機率質量必須移動多遠來解釋較大的那個值。

**練習 5。** 對每個 hash 函數，集合的 MinHash signature 是其元素上的最小 hash 值，而兩個集合在多少比例的 hash 函數上得到相同的最小值，就估計出它們的 Jaccard similarity。把每個估計值與 `jaccard_similarity` 比較；對 $k$ 個 hash 函數，誤差應大致按 $1/\sqrt{k}$ 下降。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意思 |
|------|----------------|----------------------|
| Norm | 「vector 的大小」 | 把 vector 映射到非負純量的函數，滿足三角不等式、絕對齊次性，且只有零 vector 才為零 |
| L1 norm | 「Manhattan distance」 | 分量絕對值的總和。在最佳化中產生稀疏性。對 outlier 穩健 |
| L2 norm | 「Euclidean distance」 | 分量平方和的平方根。Euclidean 空間中的直線距離 |
| Lp norm | 「一般化的 norm」 | 分量絕對值的 $p$ 次方總和的 $p$ 次方根。L1 和 L2 是特例 |
| L-infinity norm | 「Max norm」或「Chebyshev distance」 | 絕對值最大的分量。當 $p$ 趨近無限大時 Lp 的極限 |
| Cosine similarity | 「vector 之間的夾角」 | 以兩者 magnitude normalize 的 dot product。範圍從 -1 到 +1。忽略 vector 長度 |
| Cosine distance | 「1 減 cosine similarity」 | 把 cosine similarity 轉換成距離。範圍從 0 到 2 |
| Dot product | 「沒有 normalize 的 cosine」 | 對應分量乘積的總和。等於 cosine similarity 乘上兩者的 magnitude |
| Mahalanobis distance | 「考慮相關性的距離」 | 在用資料 covariance matrix 做過 whitening（去相關並 normalize）的空間中的 L2 距離 |
| Jaccard similarity | 「集合重疊」 | 交集大小除以聯集大小。用於集合，而非 vector |
| Edit distance | 「Levenshtein distance」 | 把一個字串轉成另一個所需的最少插入、刪除與替換次數 |
| KL divergence | 「分佈之間的距離」 | 不是真正的距離（不對稱）。衡量用 $Q$ 編碼 $P$ 所多出的 bit |
| Wasserstein distance | 「Earth mover's distance」 | 把質量從一個分佈運送到另一個分佈所需的最小功。是真正的度量 |
| Approximate nearest neighbor | 「ANN 搜尋」 | 比精確搜尋快得多、找出近似最近點的演算法（HNSW、LSH、IVF） |
| HNSW | 「vector DB 用的演算法」 | Hierarchical Navigable Small World 圖。用於快速近似最近鄰搜尋的多層圖 |
| L1 regularization | 「Lasso」 | 把 weight 的 L1 norm 加到 loss 上。把 weight 推向零（稀疏性） |
| L2 regularization | 「Ridge」或「weight decay」 | 把 weight 的 L2 norm 平方加到 loss 上。把 weight 縮向零，但不產生稀疏性 |
| Elastic Net | 「L1 + L2」 | 結合 L1 與 L2 regularization。比單獨使用任一種更能處理相關的 feature 群組 |

## 延伸閱讀 (Further Reading)

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - Meta 為十億規模 ANN 搜尋打造的函式庫
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - 把 Earth Mover's distance 引入 GAN 的論文
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - 基礎的 ANN 演算法
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec，cosine similarity 成為 embedding 預設選擇的起點
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - scikit-learn 中距離度量與鄰居演算法的實用指南
