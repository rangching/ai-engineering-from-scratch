# 情感分析 (Sentiment Analysis)

> 最經典的 NLP 任務。傳統文字分類你需要知道的大部分內容，都會在這裡出現。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 2 · 14 (Naive Bayes)
**Time:** ~75 分鐘

## 問題 (The Problem)

「The food was not great.」是正面還是負面？

情感聽起來很簡單：評論者說他們喜歡或不喜歡某樣東西，幫句子貼標籤就好。它會成為經典 NLP 任務的原因，是每個看起來簡單的案例背後，都藏著一個困難的案例。否定會翻轉意思，諷刺會顛倒意思。"Not bad at all" 是正面的，儘管它有兩個帶負面色彩的字。表情符號攜帶的訊號比周圍的文字還多。領域詞彙也很重要（音樂評論裡的 `tight` 和時尚評論裡的 `tight`）。

情感分析是傳統 NLP 的實驗室。只要你理解為什麼每個天真的 baseline 都有特定的失敗模式，你就理解為什麼會發明每個更豐富的模型。這堂課會從零建立 Naive Bayes baseline，再加上 logistic regression，並點出那些讓正式環境的情感分析變成合規等級問題的陷阱。

## 核心概念 (The Concept)

傳統情感分析是兩步驟的做法。

1. **表示 (Represent)。** 把文字變成特徵向量。BoW、TF-IDF 或 n-gram。
2. **分類 (Classify)。** 在有標註的範例上擬合線性模型（Naive Bayes、logistic regression、SVM）。

Naive Bayes 是最笨但有效的模型。假設在給定標籤的情況下，每個特徵彼此獨立。從計數估計 `P(word | positive)` 和 `P(word | negative)`。推論時把機率相乘。這個「naive」的獨立性假設錯得可笑，結果卻出奇地強。原因是：在 sparse 的文字特徵和中等資料量下，分類器在意的是每個字偏向哪一邊，而不是偏多少。

Logistic regression 修正了獨立性假設。它為每個特徵學一個權重，包括負權重。bigram 特徵 `not good` 會得到負權重。Naive Bayes 對沒標註過的 bigram 做不到這點。

```figure
sentiment-logits
```

## 動手做 (Build It)

### 步驟 1：一個真實的迷你資料集

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

刻意做得很小。真正的工作會用上萬筆範例（IMDb、SST-2、Yelp polarity），數學完全相同。

### 步驟 2：從零實作 multinomial Naive Bayes

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing（alpha=1.0）就是 Laplace smoothing。沒有它的話，在某個類別中沒出現過的字機率為零，log 就會爆掉。實務上常用 `alpha=0.01`，`alpha=1.0` 是教學的預設值。

### 步驟 3：從零實作 logistic regression

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization 在這裡很重要。文字特徵是 sparse 的，沒有 L2 的話模型會背下訓練範例。從 `0.01` 開始再調整。

### 步驟 4：處理否定（失敗模式）

看看 "not good" 和 "not bad"。BoW 分類器看到的是 `{not, good}` 和 `{not, bad}`，並從訓練中哪個出現比較多去學。bigram 分類器看到 `not_good` 和 `not_bad`，並把它們學成不同的特徵。這通常就夠了。

沒有 bigram 時有個比較粗糙、但有用的修法：**negation scoping**。在否定詞後面，一直到下一個標點符號為止，給每個 token 加上 `NOT_` 前綴。

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

現在 `good` 和 `NOT_good` 是不同的特徵，分類器可以給它們相反的權重。三行預處理，就讓情感 benchmark 的準確率有可測量的提升。

### 步驟 5：真正重要的評估指標

如果類別不平衡，單看 accuracy 會誤導人。真實的情感語料庫通常是 70 到 80% 正面或 70 到 80% 負面，一個永遠猜多數類別的分類器能拿到 80% accuracy，卻毫無用處。請把下面每一項都回報：

- **每個類別的 precision 與 recall。** 每個類別一組。對它們取 macro-average，得到一個尊重類別平衡的單一數字。
- **Macro-F1（不平衡資料的主要指標）。** 各類別 F1 分數的平均，權重相等。類別不平衡時用它取代 accuracy。
- **Weighted-F1（替代指標）。** 與 macro 相同，但依類別頻率加權。當不平衡本身有商業意義時，與 macro-F1 一起回報。
- **Confusion matrix。** 原始計數。在相信任何單一數值指標之前，一定要先檢查，它會揭露模型把哪兩個類別搞混。
- **每個類別的錯誤樣本。** 每個類別抓 5 個預測錯誤的例子，把它們讀完。沒有什麼能取代實際閱讀錯誤。

對嚴重不平衡的資料（比例超過 95 比 5），請回報 **AUROC** 和 **AUPRC** 而不是 accuracy。AUPRC 對少數類別更敏感，而那通常正是你在意的（垃圾郵件、詐欺、罕見的情感）。

**要避免的常見錯誤。** 在不平衡資料上回報 micro-F1 而不是 macro-F1，會得到一個看起來很高的數字，因為它被多數類別主導。Macro-F1 逼你看到少數類別的表現。

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## 實際使用 (Use It)

scikit-learn 六行就能正確完成。

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

注意三件事。`stop_words=None` 保留了否定詞。`ngram_range=(1, 2)` 加入 bigram，讓 `not good` 成為特徵（scikit-learn 用空格連接兩個字）。`sublinear_tf=True` 抑制重複字詞的影響。這三個參數，就是 SST-2 上 75% 準確率 baseline 和 85% 準確率 baseline 的差別。

### 什麼時候該改用 transformer

- 諷刺偵測。傳統模型在這裡就是不行，沒得商量。
- 情感在文件中途轉變的長篇評論。
- Aspect-based sentiment。"Camera was great but battery was terrible." 你需要把情感歸屬到各個面向，只有 transformer 或結構化輸出模型做得到。
- 非英語、低資源語言。Multilingual BERT 免費給你一個 zero-shot baseline。

如果你需要上述任何一項，請跳到 phase 7（transformer 深入探討）。否則，TF-IDF 加 bigram 加否定處理的 Naive Bayes 或 logistic regression，就是你 2026 年的正式環境 baseline。

### 可重現性陷阱（又來了）

重新訓練情感模型是例行公事，重新評估它們卻不是。論文中回報的 accuracy 數字，使用特定的切分、特定的預處理、特定的 tokenizer。如果你沒用相同的 pipeline，就拿新模型和 baseline 比較，會得到誤導的差異。請永遠在你自己的 pipeline 上重新產生 baseline，而不是照抄論文的數字。

## 交付 (Ship It)

存成 `outputs/prompt-sentiment-baseline.md`：

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## 練習 (Exercises)

1. **簡單。** 把 `apply_negation` 當成預處理步驟加進 scikit-learn pipeline，並在小型情感資料集上測量 F1 的變化。
2. **中等。** 實作加上類別權重的 logistic regression（在 scikit-learn 傳入 `class_weight="balanced"`，或自己推導梯度）。在合成的 90 比 10 類別不平衡資料上測量效果。
3. **困難。** 在情感模型的殘差上訓練第二個分類器，建立諷刺偵測器。記錄你的實驗設定。當你的準確率低於隨機水準時，要提醒讀者（2 類諷刺偵測的隨機水準約 50%，多數第一次嘗試都落在這裡）。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Polarity | 正面或負面 | 二元標籤；有時擴充成中性或更細的粒度（5 星）。 |
| Aspect-based sentiment | 各面向的情感極性 | 把情感歸屬到文字中提到的特定實體或屬性。 |
| Negation scoping | 反轉附近的 token | 在 "not" 後面的 token 加上 `NOT_` 前綴，直到標點符號為止。 |
| Laplace smoothing | 計數加 1 | 防止 Naive Bayes 出現機率為零的特徵。 |
| L2 regularization | 縮小權重 | 在 loss 加上 `lambda * sum(w^2)`。對 sparse 文字特徵不可或缺。 |

## 延伸閱讀 (Further Reading)

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — 奠基性的綜述。很長，但前四節涵蓋了所有傳統方法。
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — 證明 bigram 加 Naive Bayes 在短文本上很難被超越的論文。
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — `CountVectorizer`、`TfidfVectorizer` 以及你會調整的每個參數的參考文件。
