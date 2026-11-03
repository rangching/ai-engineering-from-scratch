# Transformer 之前的文字生成 — N-gram Language Model

> 如果一個字讓人意外，模型就是不好。Perplexity 把意外變成一個數字。Smoothing 讓它保持有限。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分鐘

## 問題 (The Problem)

在 transformer、RNN、word embedding 出現之前，language model 靠數數來預測下一個字：它跟在前 `n-1` 個字後面出現過幾次。數 "the cat" → "sat" 47 次，"the cat" → "jumped" 12 次，"the cat" → "refrigerator" 0 次。正規化後，就得到機率分布。

這就是 n-gram language model。從 1980 年到 2015 年，每個語音辨識器、每個拼字檢查器、每個 phrase-based 機器翻譯系統都靠它運作。需要便宜的裝置端 language modeling 時，它依然在運作。

有趣的問題是，遇到沒見過的 n-gram 該怎麼辦。純計數的模型會對任何沒見過的東西指派零機率，這是災難性的，因為句子很長，幾乎每個長句子都至少含有一個沒見過的序列。五十年的 smoothing 研究解決了這個問題。結果就是 Kneser-Ney smoothing，而現代深度學習繼承了它的經驗主義傳統。

## 核心概念 (The Concept)

![N-gram model: count, smooth, generate](../assets/ngram.svg)

### 預測遊戲 (The prediction game)

在這些機制出現之前，有一個實驗定義了什麼是 language model。蓋住英文句子的下一個字母。請人猜，一次猜一個，直到猜對為止。記下猜測的次數。對幾百個字母重複這件事。

猜測次數不是瑣事。它們是文字的無損重新編碼：把次數序列交給第二個一模一樣的猜測者，他就能重建每個字母，因為在每個位置，他都確切知道哪些猜測排在前面。一則可以用更少符號重新編碼的訊息，每個符號攜帶的資訊較少，所以猜測次數的統計，為英文的 entropy 設下了上限。

Shannon 在 1951 年做了這個實驗，得到的數字至今仍支配著這個領域。27 個符號的字母表（26 個字母加空白）每個字母可以攜帶 `log2(27) ≈ 4.75` bits。有 100 個字母 context 的人類猜測者，落在每個字母 0.6 到 1.3 bits 之間。英文大約有四分之三是被迫的走法。模型必須學習的結構，在任何模型能夠學習它之前就已經被量測出來了。

此後的每個 language model，都是這個遊戲的機械式玩家，而這堂課裡的每個評估數字，都是這個遊戲的計分：

- **Cross-entropy loss** 是模型每個符號平均需要的 bits 數。訓練 LM，字面上就是在最小化它在猜測遊戲中的得分。
- **Perplexity** 是 `2^bits`（或 `e^nats`）：模型猜完之後，仍然面對的分支因子。對 27 個符號均勻猜測，perplexity 是 27；每個字母 1 bit 的玩家，perplexity 是 2。
- **Context 長度就是玩家的記憶。** Trigram 模型以兩個 token 的記憶來玩。Transformer 以 10 萬個 token 玩同一個遊戲。規則從未改變；是玩家變強了。

要留意一個單位的切換：這個遊戲以每個字母的 bits (`log2`) 計分，而下面的 n-gram 公式以每個字詞 token 的 nats（自然對數）計分。由於以 nats 計的 perplexity `e^H` 等於以 bits 計的 `2^H`，這兩種觀點是同一個量測、不同的單位。

```figure
prediction-game
```

**N-gram 機率：** `P(w_i | w_{i-n+1}, ..., w_{i-1})`。固定 `n`（通常 trigram 用 3，4-gram 用 4）。從計數計算：

```math
P(w \mid \text{context}) = \frac{\mathop{\mathrm{count}}(\text{context}, w)}{\mathop{\mathrm{count}}(\text{context})}
```

**零計數問題。** 訓練中沒見過的任何 n-gram，機率都是零。一項 2007 年在 Brown 語料庫上的研究發現，即使是 4-gram 模型，保留集中也有 30% 的 4-gram 在訓練中沒見過。沒有 smoothing，你無法在任何真實文本上評估。

**Smoothing 方法，依複雜度排序：**

1. **Laplace（加一）。** 每個計數加 1。簡單，對罕見事件很糟。
2. **Good-Turing。** 根據頻率的頻率，把機率質量從高頻事件重新分配給沒見過的事件。
3. **Interpolation。** 以可調整的權重，結合 n-gram、(n-1)-gram 等的估計。
4. **Backoff。** 如果 n-gram 計數為零，就退回 (n-1)-gram。Katz backoff 將此正規化。
5. **Absolute discounting。** 從所有計數中減去固定的折扣 `D`，重新分配給沒見過的事件。
6. **Kneser-Ney。** Absolute discounting 加上對低階模型的巧妙選擇：使用 *continuation probability*（一個字出現在多少種 context 中），而不是原始頻率。

Kneser-Ney 的洞察很深。"San Francisco" 是常見的 bigram。Unigram "Francisco" 大多出現在 "San" 之後。天真的 absolute discounting 會給 "Francisco" 很高的 unigram 機率（因為計數很高）。Kneser-Ney 注意到 "Francisco" 只出現在一種 context 中，因此相應地降低它的 continuation probability。結果：一個以 "Francisco" 結尾的全新 bigram，會得到適當的低機率。

**評估：perplexity。** 保留測試集上，每個字平均負對數似然的指數。越低越好。Perplexity 為 100，代表模型的困惑程度，就像在 100 個字之間均勻選擇一樣。

```math
\text{perplexity} = \exp\left(-\frac{1}{N} \sum_i \log P(w_i \mid \text{context}_i)\right)
```

```figure
ngram-backoff
```

## 動手做 (Build It)

### 步驟 1：Trigram 計數

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

輸入是已 tokenize 的句子清單。輸出是 n-gram 計數和 context 計數。`<s>` 和 `</s>` 是句子邊界。

### 步驟 2：Laplace smoothing

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

每個計數加 1。它能平滑，但把過多質量分配給沒見過的事件，連帶傷害了罕見但已知的事件。

### 步驟 3：Kneser-Ney（bigram、interpolated）

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

三個運作的部分。`continuation_prob` 捕捉「這個字出現在多少種不同的 context 中？」（Kneser-Ney 的創新）。`lambda_prev` 是折扣釋放出的質量，用來加權 backoff。最終機率是折扣後的主項，加上加權的 continuation 項。

### 步驟 4：用取樣生成文字

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

依機率比例取樣。每個 seed 總是給出不同的輸出。若想要類似 beam search 的輸出，就在每一步挑 argmax（greedy），再加上一個小小的隨機性旋鈕（temperature）。

### 步驟 5：Perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

越低越好。在 Brown 語料庫上，調得好的 4-gram KN 模型 perplexity 約為 140。Transformer LM 在同一個測試集上是 15 到 30。差距約 10 倍。這個差距就是這個領域繼續往前走的原因。

## 實際使用 (Use It)

- **傳統 NLP 教學。** 你能得到的、對 smoothing、MLE 和 perplexity 最清楚的接觸。
- **KenLM。** 正式環境的 n-gram 函式庫。在重視低延遲的語音和 MT 系統中，被用作 rescorer。
- **裝置端自動完成。** 鍵盤中的 trigram 模型，至今仍是如此。
- **Baseline。** 在宣稱你的神經網路 LM 很好之前，一定要先計算 n-gram LM 的 perplexity。如果你的 transformer 沒有大幅勝過 KN，就是哪裡出錯了。

## 交付 (Ship It)

存成 `outputs/prompt-lm-baseline.md`：

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## 練習 (Exercises)

1. **簡單。** 在 1,000 句的莎士比亞語料庫上訓練 trigram LM。生成 20 個句子。它們在局部上看似合理，整體卻不連貫。這是標準的示範。
2. **中等。** 在保留的莎士比亞切分上，為你的 KN 模型實作 perplexity。與 Laplace 比較。你應該會看到 KN 的 perplexity 低 30 到 50%。
3. **困難。** 建立 trigram 拼字修正器：給定拼錯的字和它的 context，產生修正選項，並依 LM 下的 context 機率排序。在 Birkbeck 拼字語料庫（公開）上評估。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| N-gram | 字的序列 | `n` 個連續 token 的序列。 |
| Smoothing | 避免零 | 重新分配機率質量，讓沒見過的事件得到非零的機率。 |
| Perplexity | LM 品質指標 | 保留資料上的 `exp(-average log-prob)`。越低越好。 |
| Backoff | 退回較短的 context | 如果 trigram 計數為零，就用 bigram。Katz backoff 將此形式化。 |
| Kneser-Ney | n-gram 最好的 smoothing | Absolute discounting 加上低階模型的 continuation probability。 |
| Continuation probability | KN 特有 | 依 `w` 出現的 context 數量加權的 `P(w)`，而不是依原始計數。 |
| Entropy of text | 每個符號的資訊量 | 給定 context，編碼下一個符號平均需要的 bits。Shannon 在 1951 年對有最多 100 個字母 context 的印刷英文所做的估計：每個字母 0.6 到 1.3 bits，在任何模型存在之前就量測出來了。 |

## 延伸閱讀 (Further Reading)

- [Shannon (1951). Prediction and Entropy of Printed English](https://www.princeton.edu/~wbialek/rome/refs/shannon_51.pdf) — 猜測遊戲實驗，定義了每個 language model 至今仍在優化的目標。
- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — n-gram LM 與 smoothing 的標準論述。
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — 確立 Kneser-Ney 為最佳 n-gram smoother 的論文。
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — 原始的 KN 論文。
- [KenLM](https://kheafield.com/code/kenlm/) — 快速的正式環境 n-gram LM，2026 年仍用於對延遲敏感的應用。
