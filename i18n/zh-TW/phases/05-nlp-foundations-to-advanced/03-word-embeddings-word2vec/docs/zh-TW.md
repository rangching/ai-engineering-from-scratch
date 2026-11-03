# Word Embeddings — 從零開始實作 Word2Vec

> 一個字的意義，來自它身邊的夥伴。用這個想法訓練一個淺層網路，幾何結構就自己跑出來了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 3 · 03 (Backpropagation from Scratch)
**Time:** ~75 分鐘

## 問題 (The Problem)

TF-IDF 知道 `dog` 和 `puppy` 是不同的字，卻不知道它們的意思幾乎一樣。用 `dog` 訓練出來的分類器，無法推廣到關於 `puppy` 的評論。你可以列出同義詞來遮掩這個問題，但遇到罕見詞、領域術語，以及所有你沒預料到的語言，這招就失效了。

你想要一種表示法，讓 `dog` 和 `puppy` 在空間中靠得很近。讓 `king - man + woman` 落在 `queen` 附近。讓用 `dog` 訓練的模型，能免費把一些訊號轉移到 `puppy`。

Word2Vec 給了我們這樣的空間。兩層神經網路，兆級 token 的訓練，2013 年發表。它的架構簡單到幾乎讓人不好意思，成果卻重塑了 NLP 十年。

## 核心概念 (The Concept)

**Distributional hypothesis**（Firth, 1957）：「看一個字身邊的夥伴，就能認識這個字。」如果兩個字出現在相似的上下文中，它們的意思大概也相似。

Word2Vec 有兩種版本，都利用了這個想法。

- **Skip-gram。** 給定中心字，預測周圍的字。window size 為 2 時，`cat -> (the, sat, on)`。
- **CBOW (continuous bag of words)。** 給定周圍的字，預測中心字。`(the, sat, on) -> cat`。

Skip-gram 訓練較慢，但處理罕見字比較好，所以成了預設選擇。

這個網路只有一個沒有非線性的 hidden layer。輸入是詞彙表上的 one-hot 向量，輸出是詞彙表上的 softmax。訓練完成後，把輸出層丟掉，hidden layer 的權重就是 embedding。

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

關鍵技巧：對 10 萬個字做 softmax，成本高到無法接受。Word2Vec 用 **negative sampling** 把它變成二元分類任務：預測「這個 context 字有沒有出現在這個中心字附近，是或否」。每個訓練 pair 只取樣少數幾個負樣本（沒有共同出現的字），而不是對整個詞彙表計算 softmax。

```figure
word-vector-arithmetic
```

## 動手做 (Build It)

### 步驟 1：從語料庫產生訓練 pair

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

window 內每一組 (center, context) pair，都是一個正向訓練範例。

### 步驟 2：embedding 表

兩個矩陣。`W` 是中心字的 embedding 表（你要保留的那個）。`W'` 是 context 字的表（常被丟棄，有時會與 `W` 取平均）。

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

用小的隨機值初始化。詞彙表大小 1 萬、維度 100 是實際的設定；教學用 50 個字、16 維，就足以看到幾何結構。

### 步驟 3：negative sampling 目標函數

對每個正向 pair `(center, context)`，從詞彙表隨機取樣 `k` 個字當負樣本。訓練模型，讓正樣本的 dot product `W[center] · W'[context]` 變高，負樣本的變低。

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

神奇的公式：正向 pair 的 logistic loss（希望 sigmoid 接近 1），加上負向 pair 的 logistic loss（希望 sigmoid 接近 0）。梯度會流向兩張表。完整推導在原始論文裡，想記牢的話，拿紙筆走一遍。

### 步驟 4：在玩具語料庫上訓練

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

在大型語料庫上訓練足夠的 epoch 後，共享 context 的字會有相似的中心 embedding。在玩具語料庫上，你只能隱約看到效果。在數十億 token 上，效果就非常驚人。

### 步驟 5：類比技巧 (analogy trick)

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

在預訓練的 300 維 Google News 向量上：

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`。不是因為模型知道什麼是王室，而是因為向量 `(king - man)` 捕捉到類似「王室」的東西，把它加到 `woman` 上，就落在王室女性的區域附近。

## 實際使用 (Use It)

從零寫 Word2Vec 是為了教學。正式的 NLP 用 `gensim`。

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

實際工作中，你幾乎不會自己訓練 Word2Vec，而是下載預訓練向量。

- **GloVe** — Stanford 的 co-occurrence matrix 分解方法。有 50d、100d、200d、300d 的 checkpoint。一般涵蓋範圍很好。第 04 課會專門介紹 GloVe。
- **fastText** — Facebook 對 Word2Vec 的擴充，會對 character n-gram 做 embedding。透過組合 subword 來處理 out-of-vocabulary 的字。第 04 課。
- **Google News 上預訓練的 Word2Vec** — 300 維，300 萬字的詞彙表，2013 年發布，至今仍每天被下載。

### 2026 年 Word2Vec 依然勝出的時候

- 輕量的特定領域檢索。在筆電上用一小時訓練醫學摘要，就能得到通用模型捕捉不到的專用向量。
- 類比式的 feature engineering。`gender_vector = mean(man - woman pairs)`。從其他字扣掉它，就得到性別中立的軸。公平性研究仍在使用。
- 可解釋性。100 維小到能用 PCA 或 t-SNE 畫出來，真的看到分群形成。
- 任何必須在裝置端、沒有 GPU 的地方做推論。Word2Vec 的查詢只是取出一列。

### Word2Vec 失敗的地方

一詞多義 (polysemy) 這道牆。`bank` 只有一個向量。`river bank` 和 `financial bank` 共用它。`table`（試算表 vs. 家具）也共用。下游分類器無法從向量分辨這些詞義。

Contextual embedding（ELMo、BERT，以及之後所有 transformer）解決了這個問題，做法是依周圍上下文，為每次出現的字產生不同的向量。這就是從 Word2Vec 到 BERT 的跳躍：從靜態到 contextual。Phase 7 會涵蓋 transformer 的部分。

另一個失敗是 out-of-vocabulary 問題。如果 `Zoomer-approved` 不在訓練資料裡，Word2Vec 就從沒見過它，而且沒有 fallback。fastText 用 subword 組合解決這點（第 04 課）。

## 交付 (Ship It)

存成 `outputs/skill-embedding-probe.md`：

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## 練習 (Exercises)

1. **簡單。** 在一個很小的語料庫（20 個關於貓和狗的句子）上跑訓練迴圈。200 個 epoch 後，驗證 `nearest(vocab, W, W[vocab["cat"]])` 的前 3 名包含 `dog`。如果沒有，增加 epoch 或詞彙量。
2. **中等。** 加入高頻字的 subsampling。頻率高於 `10^-5` 的字，會以與其頻率成正比的機率，從訓練 pair 中被丟掉。測量這對罕見字相似度的影響。
3. **困難。** 在 20 Newsgroups 語料庫上訓練一個模型。計算兩個偏見軸：`he - she` 和 `doctor - nurse`。把職業相關的字投影到兩個軸上。回報哪些職業的偏見差距最大。這正是公平性研究者使用的探測方式。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Word embedding | 把字變成向量 | 從上下文學到的 dense、低維（通常 100 到 300）表示。 |
| Skip-gram | Word2Vec 的招式 | 從中心字預測 context 字。比 CBOW 慢，但對罕見字比較好。 |
| Negative sampling | 訓練捷徑 | 把對完整詞彙表的 softmax，換成對 `k` 個隨機字的二元分類。 |
| Static embedding | 一個字一個向量 | 不論上下文，向量都相同。遇到一詞多義就失敗。 |
| Contextual embedding | 隨上下文改變的向量 | 依周圍的字，每次出現都有不同向量。transformer 產生的就是這種。 |
| OOV | Out of vocabulary | 訓練中沒見過的字。Word2Vec 無法為它們產生向量。 |

## 延伸閱讀 (Further Reading)

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — negative-sampling 論文，短而好讀。
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — 如果覺得原論文的數學太密，這篇是最清楚的梯度推導。
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — 真正有效的正式訓練設定。
