# GloVe、FastText 與 Subword Embedding

> Word2Vec 為每個字訓練一個 embedding。GloVe 分解 co-occurrence matrix。FastText 對字的片段做 embedding。BPE 則銜接到 transformer。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec from Scratch)
**Time:** ~45 分鐘

## 問題 (The Problem)

Word2Vec 留下兩個未解的問題。

第一，當時有另一條平行的研究路線，直接分解 co-occurrence matrix（LSA、HAL），而不是做線上的 skip-gram 更新。Word2Vec 的迭代方法真的比較好嗎？還是差異只是兩種方法處理計數方式的產物？**GloVe** 回答了這點：用精心選擇的 loss 做矩陣分解，表現能追平甚至超越 Word2Vec，訓練成本還更低。

第二，兩種方法對沒見過的字都沒有辦法。`Zoomer-approved`、`dogecoin`、上週才創造的專有名詞、罕見詞根的每個屈折變化形式。**FastText** 透過對 character n-gram 做 embedding 解決了這點：一個字是它各部分的總和，包括詞素 (morpheme)，所以連 out-of-vocabulary 的字也能得到合理的向量。

第三，transformer 出現之後，問題又換了。word-level 詞彙表最多到約一百萬個項目，真實語言比這更開放。**Byte-pair encoding (BPE)** 與其相關方法，透過學習一個涵蓋一切的常見 subword 單元詞彙表來解決。每個現代 LLM 的現代 tokenizer，都是 subword tokenizer。

這堂課會把三者都走一遍，再說明什麼時候該用哪一個。

## 核心概念 (The Concept)

**GloVe (Global Vectors)。** 建立 word-word co-occurrence matrix `X`，其中 `X[i][j]` 是字 `j` 出現在字 `i` 的上下文中的頻率。訓練向量，使得 `v_i · v_j + b_i + b_j ≈ log(X[i][j])`。對 loss 加權，讓高頻 pair 不會佔主導。就這樣。

**FastText。** 一個字是它的 character n-gram 加上字本身的總和。`where` 變成 `<wh, whe, her, ere, re>, <where>`。字向量是這些組成向量的總和。照 Word2Vec 的方式訓練。好處：沒見過的字（`whereupon`）可以由已知的 n-gram 組合出來。

**BPE (Byte-Pair Encoding)。** 從個別 byte（或字元）的詞彙表開始。數出語料庫中每一組相鄰的 pair，把最頻繁的 pair 合併成新 token。重複 `k` 次。結果：詞彙表包含基本符號加上 `k` 次合併（從 byte 開始時是 `k + 256` 個 token），其中常見序列（`ing`、`tion`、`the`）是單一 token，罕見字則被拆成熟悉的片段。每個句子都能被 tokenize。

```figure
n5-subword-merge
```

## 動手做 (Build It)

### GloVe：分解 co-occurrence matrix

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

有兩個值得點名的環節。加權函數 `f(x) = (x/x_max)^alpha` 會壓低非常頻繁的 pair（像 `(the, and)`），避免它們主導 loss。最終的 embedding 是 `W`（中心）與 `W_tilde`（context）兩張表的總和。把兩者相加是已發表的技巧，通常比只用其中一張表更好。

### FastText：具備 subword 感知的 embedding

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

每個字由它的 n-gram 集合表示（通常是 3 到 6 個字元）。字的 embedding 是它所有 n-gram embedding 的總和。做 skip-gram 訓練時，把這個放在 Word2Vec 原本使用單一向量的位置即可。

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

對沒見過的字，只要它有一些 n-gram 是已知的，你仍然能得到向量。`whereupon` 與 `where` 共享 `<wh`、`her`、`ere` 和 `<where`，所以兩者會落在彼此附近。

### BPE：學習而來的 subword 詞彙表

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

第一次迭代合併最常見的相鄰 pair。經過足夠的迭代後，常見的子字串（`low`、`est`、`tion`）變成單一 token，罕見字則被乾淨地拆開。

真正的 GPT / BERT / T5 tokenizer 會學 3 萬到 10 萬次合併。結果：任何文字都能 tokenize 成長度有界的已知 ID 序列，永遠不會有 OOV。

## 實際使用 (Use It)

實務上，你很少自己訓練這些東西，而是載入預訓練的 checkpoint。

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

在 transformer 時代的 BPE 風格 subword tokenization：

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

`Ġ` 前綴標記字的邊界（GPT-2 的慣例）。每個現代 tokenizer 都是 BPE 的變體、WordPiece (BERT) 或 SentencePiece (T5, LLaMA)。

### 該選哪一個

| 情境 | 選擇 |
|-----------|------|
| 預訓練的通用 word vector，不需要容忍 OOV | GloVe 300d |
| 預訓練的通用 word vector，必須處理拼字錯誤 / 新創詞 / 詞形變化豐富的語言 | FastText |
| 任何要送進 transformer 的東西（訓練或推論） | 用模型附帶的 tokenizer，絕對不要換。 |
| 從零訓練你自己的 language model | 先在你的語料庫上訓練 BPE 或 SentencePiece tokenizer |
| 用線性模型做正式的文字分類 | 仍然是 TF-IDF。第 02 課。 |

## 交付 (Ship It)

存成 `outputs/skill-embeddings-picker.md`：

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## 練習 (Exercises)

1. **簡單。** 執行 `char_ngrams("playing")` 和 `char_ngrams("played")`。計算兩個 n-gram 集合的 Jaccard 重疊。你應該會看到相當多共享的片段（`pla`、`lay`、`play`），這就是 FastText 能在詞形變化之間良好轉移的原因。
2. **中等。** 擴充 `learn_bpe` 來追蹤詞彙表的成長。把每個語料庫字元對應的 token 數，畫成合併次數的函數。你應該會看到一開始壓縮很快，之後漸近到每個 token 約 2 到 3 個字元。
3. **困難。** 在莎士比亞全集上訓練一個 1k 次合併的 BPE。比較常見字與罕見專有名詞的 tokenization。測量前後每個字的平均 token 數。寫下讓你意外的地方。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Co-occurrence matrix | 字與字的頻率表 | `X[i][j]` = 字 `j` 出現在字 `i` 周圍 window 的頻率。 |
| Subword | 字的片段 | character n-gram (FastText) 或學習而來的 token (BPE/WordPiece/SentencePiece)。 |
| BPE | Byte-pair encoding | 反覆合併最頻繁的相鄰 pair，直到詞彙表達到目標大小。 |
| OOV | Out of vocabulary | 模型從沒見過的字。Word2Vec/GloVe 會失敗，FastText 和 BPE 能處理。 |
| Byte-level BPE | 在原始 byte 上做 BPE | GPT-2 的做法。詞彙表從 256 個 byte 開始，所以永遠不會有 OOV。 |

## 延伸閱讀 (Further Reading)

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — GloVe 論文，七頁，至今仍是 loss 最好的推導。
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText。
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — 把 BPE 引入現代 NLP 的論文。
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — BPE、WordPiece 和 SentencePiece 在實務上到底差在哪裡。
