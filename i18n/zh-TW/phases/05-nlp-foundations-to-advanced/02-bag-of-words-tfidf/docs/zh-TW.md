# Bag of Words、TF-IDF 與文字表示 (Text Representation)

> 先數數看，再想。到了 2026 年，在定義明確的任務上，TF-IDF 依然打得贏 embedding。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 02 (Linear Regression from Scratch)
**Time:** ~75 分鐘

## 問題 (The Problem)

模型需要數字，你手上卻是字串。

每個 NLP pipeline 都得回答同一個問題：怎麼把長度不一的 token 串流，變成分類器能吃的固定大小向量？這個領域最早的答案，是最笨但有效的那一個：數字，做成向量。

這種向量撐起的正式環境 NLP，比任何 embedding 模型都多。垃圾郵件過濾、主題分類、log 異常偵測、搜尋排序（BM25 之前）、第一波情感分析、學術 NLP benchmark 的第一個十年。2026 年的實務工作者，面對範圍狹窄的分類任務，還是會先拿它來用。它快、可解釋，而且在字詞是否出現才是重點的任務上，常常和 4 億參數的 embedding 模型分不出高下。

這堂課會從零建立 bag of words，再建立 TF-IDF。接著示範 scikit-learn 三行就做完同樣的事。最後點出會讓你改用 embedding 的失敗模式。

## 核心概念 (The Concept)

**Bag of Words (BoW)** 丟掉順序。對每份文件，數出詞彙表中每個字出現幾次。向量長度就是詞彙表大小，第 `i` 個位置是第 `i` 個字的次數。

**TF-IDF** 重新加權 BoW。出現在每份文件的字沒有資訊量，所以把權重調低。在整個語料庫很少見、但在單一文件頻繁出現的字是訊號，所以把權重調高。

```math
\begin{aligned}
\text{TF-IDF}(w, d) &= \text{TF}(w, d) \cdot \text{IDF}(w) \\
&= \frac{\mathop{\mathrm{count}}(w \text{ in } d)}{|d|} \cdot \log\frac{N}{\mathop{\mathrm{df}}(w)}
\end{aligned}
```

其中 `TF` 是字在文件中的 term frequency，`df` 是 document frequency（有多少份文件包含這個字），`N` 是文件總數。`log` 讓無所不在的字，權重保持在有界範圍內。

關鍵特性：兩者都產生座標軸可解釋的 sparse vector。你可以看訓練好的分類器權重，讀出哪些字把文件推向各個類別。768 維的 BERT embedding 就辦不到。

```figure
bow-tfidf
```

## 動手做 (Build It)

### 步驟 1：建立詞彙表

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

輸入：已 tokenize 的文件清單（任何 word-level tokenizer 都行，這堂課的 `code/main.py` 用的是簡化的小寫版本）。輸出：`{word: index}` 字典。穩定的插入順序，代表 index 0 是第一份文件裡看到的第一個字。慣例各有不同，scikit-learn 是依字母順序排序。

### 步驟 2：bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

每一列是一份文件，每一欄是詞彙表索引。`[i][j]` 這一格，是「字 `j` 在文件 `i` 出現幾次」。文件 1 的 `cat` 是 2，因為它真的出現兩次。文件 0 的 `ran` 是 0，因為它沒出現。

### 步驟 3：term frequency 與 document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

有兩個值得點名的平滑技巧。`(n+1)/(d+1)` 避免出現 `log(x/0)`。尾端的 `+1` 確保出現在每份文件的字，IDF 仍然是 1（而不是 0），這和 scikit-learn 的預設一致。其他實作用的是原始的 `log(N/df)`。兩種都行，平滑版比較友善。

### 步驟 4：TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

三份文件，五個詞彙（`the`、`cat`、`sat`、`dog`、`ran`）。`the` 出現在三份文件，所以 IDF 很低。`dog` 只出現在一份，所以 IDF 很高。真實的詞彙表有幾千個字，這些向量會是 sparse 的（大多數項目是零）。即使是這個玩具範例，有辨別力的字也會突顯出來。

### 步驟 5：對每一列做 L2 normalize

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

沒有 normalization 的話，較長的文件會得到較大的向量，並在相似度分數中佔主導。L2 normalization 把每份文件放到單位超球面上。這時列與列之間的 cosine similarity，就只是一次 dot product。

## 實際使用 (Use It)

scikit-learn 提供了正式版本。

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` 一次呼叫就完成 tokenization、詞彙表和 BoW。`TfidfVectorizer` 再加上 IDF 加權和 L2 normalization。兩者都回傳 sparse matrix。對 10 萬份文件，dense 版本放不進記憶體，所以要保持 sparse，直到分類器要求 dense 才轉換。

幾個會改變一切的參數：

| 參數 | 效果 |
|-----|--------|
| `ngram_range=(1, 2)` | 納入 bigram。通常能提升分類表現。 |
| `min_df=2` | 丟掉出現在少於 2 份文件的字。在雜訊多的資料上縮減詞彙表。 |
| `max_df=0.95` | 丟掉出現在超過 95% 文件的字。不用寫死清單，就能近似移除 stopword。 |
| `stop_words="english"` | scikit-learn 內建的 stopword 清單。依任務而定，情感分析*不該*丟掉否定詞。 |
| `sublinear_tf=True` | 用 `1 + log(tf)` 取代原始的 `tf`。當某個詞在單一文件重複很多次時有幫助。 |

### TF-IDF 依然勝出的時候（截至 2026 年）

- 垃圾郵件偵測、主題標記、log 異常標記。重點是字詞有沒有出現，語意細微差異不重要。
- 低資料量情境（幾百筆標註範例）。TF-IDF 加 logistic regression 沒有 pretraining 成本。
- 任何在意延遲的地方。TF-IDF 加線性模型在微秒內就能回答。用 transformer 對一份文件做 embedding 要 10 到 100 毫秒。
- 必須解釋預測結果的系統。檢查分類器的係數，權重最高的正向字詞就是原因。

### TF-IDF 失敗的時候

語意盲點這個失敗模式。看這兩份文件：

- "The movie was not good at all."
- "The movie was excellent."

一個是負評，一個是正評。它們的 TF-IDF 重疊剛好是 `{the, movie, was}`。bag-of-words 分類器得記住：`not` 出現在 `good` 附近會翻轉標籤。資料夠多時它學得會，但永遠不如理解語法的模型那樣優雅。

另一個失敗：推論時遇到 out-of-vocabulary 的字。用 IMDb 評論訓練的 BoW 模型，如果 `Zoomer-approved` 從沒在訓練中出現過，它就完全不知道怎麼處理。Subword embedding（第 04 課）可以處理，TF-IDF 不行。

### 混合做法：TF-IDF 加權的 embedding

2026 年中等資料量分類的務實預設做法：把 TF-IDF 權重當成 word embedding 上的 attention。

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

你同時得到 embedding 的語意能力，以及 TF-IDF 對罕見字的強調。分類器在 pooled 向量上訓練。在標註資料少於約 5 萬筆時，對情感、主題和意圖分類，這個做法的表現優於單獨使用任何一邊。

## 交付 (Ship It)

存成 `outputs/prompt-vectorization-picker.md`：

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## 練習 (Exercises)

1. **簡單。** 在 L2-normalized 的 TF-IDF 輸出上實作 `cosine_similarity(doc_vec_a, doc_vec_b)`。驗證相同文件得到 1.0，詞彙完全不重疊的文件得到 0.0。
2. **中等。** 為 `bag_of_words` 加上 `n-gram` 支援。參數 `n` 產生 `n`-gram 的計數。測試 `["the", "cat", "sat"]` 在 `n=2` 時，會產生 `["the cat", "cat sat"]` 的 bigram 計數。
3. **困難。** 用 GloVe 100d 向量（下載一次後快取）建立上面的 TF-IDF 加權 embedding 混合做法。在 20 Newsgroups 資料集上，比較它與純 TF-IDF、純 mean-pooled embedding 的分類準確率。回報各自在哪些情況勝出。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| BoW | 字頻向量 | 一份文件中詞彙表各字的次數。丟掉順序。 |
| TF | Term frequency | 一個字在文件中的次數，可選擇依文件長度做 normalize。 |
| DF | Document frequency | 至少包含該字一次的文件數量。 |
| IDF | Inverse document frequency | 平滑後的 `log(N / df)`。壓低到處都出現的字的權重。 |
| Sparse vector | 大多是零 | 詞彙表通常有 1 萬到 10 萬個字，任何一份文件裡大多數都不會出現。 |
| Cosine similarity | 向量夾角 | L2-normalized 向量的 dot product。1 代表相同，0 代表正交。 |

## 延伸閱讀 (Further Reading)

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — 標準的 API 參考，並附上每個參數的說明。
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — 讓 TF-IDF 成為十年預設做法的論文。
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — 2026 年對舊方法何時勝出、為什麼勝出的看法。
