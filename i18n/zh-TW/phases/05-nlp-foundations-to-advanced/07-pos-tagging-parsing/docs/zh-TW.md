# 詞性標註與句法分析 (POS Tagging and Syntactic Parsing)

> 文法曾經退流行過一陣子。後來每個 LLM pipeline 都需要驗證結構化抽取，它就回來了。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分鐘

## 問題 (The Problem)

第 01 課承諾過，lemmatization 需要詞性標籤 (part-of-speech tag)。不知道 `running` 是動詞，lemmatizer 就無法把它還原成 `run`。不知道 `better` 是形容詞，就無法還原成 `good`。

這個承諾背後藏著一整個子領域。詞性標註 (POS tagging) 指派文法類別。句法分析 (syntactic parsing) 還原句子的樹狀結構：哪個字修飾哪個字，哪個動詞支配哪些論元。傳統 NLP 花了二十年精進這兩件事。後來深度學習把它們壓縮成預訓練 transformer 之上的 token-classification 任務，研究社群就轉向別處了。

但應用端沒有。每個結構化抽取 pipeline 底層仍然使用 POS 和 dependency tree。LLM 產生的 JSON 要對照文法限制來驗證。Question-answering 系統用 dependency parse 拆解查詢。Machine translation 的品質評估器會檢查 parse tree 的對齊。

值得知道。這堂課會介紹 tagset、baseline，以及你該停止從零實作、直接呼叫 spaCy 的那個時間點。

## 核心概念 (The Concept)

**POS tagging** 為每個 token 標上文法類別。**Penn Treebank (PTB)** tagset 是英文的預設，有 36 個標籤，其中的區分在一般讀者看來很繁瑣：`NN` 單數名詞、`NNS` 複數名詞、`NNP` 單數專有名詞、`VBD` 過去式動詞、`VBZ` 第三人稱單數現在式動詞，依此類推。**Universal Dependencies (UD)** tagset 比較粗（17 個標籤），與語言無關，成為跨語言工作的預設。

```
The/DET cats/NOUN were/AUX running/VERB at/ADP 3pm/NOUN ./PUNCT
```

**句法分析 (Syntactic parsing)** 產生一棵樹。兩種主要風格：

- **Constituency parsing。** 名詞片語、動詞片語、介系詞片語彼此巢狀。輸出是非終端類別（NP、VP、PP）組成的樹，字是葉節點。
- **Dependency parsing。** 每個字都有一個它所依附的中心詞 (head)，並標上文法關係。輸出是一棵樹，每條邊都是 (head, dependent, relation) 三元組。

Dependency parsing 在 2010 年代勝出，因為它能乾淨地跨語言推廣，特別是詞序自由的語言。

```
running is ROOT
cats is nsubj of running
were is aux of running
at is prep of running
3pm is pobj of at
```

```figure
pos-tagger
```

```figure
dependency-arcs
```

## 動手做 (Build It)

### 步驟 1：最常見標籤 baseline

最笨但有效的 POS tagger。對每個字，預測它在訓練資料中最常出現的標籤。

```python
from collections import Counter, defaultdict


def train_mft(train_examples):
    word_tag_counts = defaultdict(Counter)
    all_tags = Counter()
    for tokens, tags in train_examples:
        for token, tag in zip(tokens, tags):
            word_tag_counts[token.lower()][tag] += 1
            all_tags[tag] += 1
    word_best = {w: c.most_common(1)[0][0] for w, c in word_tag_counts.items()}
    default_tag = all_tags.most_common(1)[0][0]
    return word_best, default_tag


def predict_mft(tokens, word_best, default_tag):
    return [word_best.get(t.lower(), default_tag) for t in tokens]
```

在 Brown 語料庫上，這個 baseline 約有 85% 準確率。不算好，卻是任何正經模型都不該低於的底線。

### 步驟 2：bigram HMM tagger

對序列的聯合機率建模：

```math
P(\text{tags}, \text{words}) = \prod_i P(\text{tag}_i \mid \text{tag}_{i-1}) \cdot P(\text{word}_i \mid \text{tag}_i)
```

兩張表：transition 機率（給定前一個 tag 的 tag）、emission 機率（給定 tag 的字）。兩者都從計數估計，並使用 Laplace smoothing。用 Viterbi 解碼（在 tag lattice 上做動態規劃）。

```python
import math


def train_hmm(train_examples, alpha=0.01):
    transitions = defaultdict(Counter)
    emissions = defaultdict(Counter)
    tags = set()
    vocab = set()

    for tokens, ts in train_examples:
        prev = "<BOS>"
        for token, tag in zip(tokens, ts):
            transitions[prev][tag] += 1
            emissions[tag][token.lower()] += 1
            tags.add(tag)
            vocab.add(token.lower())
            prev = tag
        transitions[prev]["<EOS>"] += 1

    return transitions, emissions, tags, vocab


def log_prob(table, given, key, smooth_denom, alpha):
    return math.log((table[given].get(key, 0) + alpha) / smooth_denom)


def viterbi(tokens, transitions, emissions, tags, vocab, alpha=0.01):
    tags_list = list(tags)
    n = len(tokens)
    V = [[0.0] * len(tags_list) for _ in range(n)]
    back = [[0] * len(tags_list) for _ in range(n)]

    for j, tag in enumerate(tags_list):
        em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
        tr_denom = sum(transitions["<BOS>"].values()) + alpha * (len(tags_list) + 1)
        tr = log_prob(transitions, "<BOS>", tag, tr_denom, alpha)
        em = log_prob(emissions, tag, tokens[0].lower(), em_denom, alpha)
        V[0][j] = tr + em
        back[0][j] = 0

    for i in range(1, n):
        for j, tag in enumerate(tags_list):
            em_denom = sum(emissions[tag].values()) + alpha * (len(vocab) + 1)
            em = log_prob(emissions, tag, tokens[i].lower(), em_denom, alpha)
            best_prev = 0
            best_score = -1e30
            for k, prev_tag in enumerate(tags_list):
                tr_denom = sum(transitions[prev_tag].values()) + alpha * (len(tags_list) + 1)
                tr = log_prob(transitions, prev_tag, tag, tr_denom, alpha)
                score = V[i - 1][k] + tr + em
                if score > best_score:
                    best_score = score
                    best_prev = k
            V[i][j] = best_score
            back[i][j] = best_prev

    last_best = max(range(len(tags_list)), key=lambda j: V[n - 1][j])
    path = [last_best]
    for i in range(n - 1, 0, -1):
        path.append(back[i][path[-1]])
    return [tags_list[j] for j in reversed(path)]
```

Bigram HMM 在 Brown 上約有 93% 準確率。從 85% 跳到 93%，主要來自 transition 機率，模型學到 `DET NOUN` 很常見，而 `NOUN DET` 很罕見。

### 步驟 3：為什麼現代 tagger 能勝過它

Transition 加 emission 機率都是局部的。它們無法捕捉 `saw` 在 "I bought a saw" 裡是名詞，在 "I saw the movie." 裡卻是動詞。使用任意特徵（字尾、字形、前後的字、字本身）的 CRF 可達約 97%。BiLSTM-CRF 或 transformer 可達 98% 以上。

這個任務的天花板，是由標註者之間的不一致決定的。人類標註者在 Penn Treebank 上約有 97% 的時間意見一致。超過 98% 的模型，很可能是對測試集 overfitting。

### 步驟 4：dependency parsing 草圖

從零實作完整的 dependency parsing 超出範圍，標準的教科書處理方式在 Jurafsky 和 Martin 裡。要知道兩個傳統家族：

- **Transition-based** parser（arc-eager、arc-standard）像 shift-reduce parser 那樣運作：讀取 token，把它們 shift 到 stack 上，並套用 reduce 動作來建立弧 (arc)。貪婪解碼很快。經典實作是 MaltParser，現代神經網路版本是 Chen 和 Manning 的 transition-based parser。
- **Graph-based** parser（Eisner 演算法、Dozat-Manning biaffine）為每條可能的 head-dependent 邊打分，並挑出最大生成樹。比較慢，但比較準。

大多數應用工作，直接呼叫 spaCy：

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running at 3pm.")
for token in doc:
    print(f"{token.text:10s} tag={token.tag_:5s} pos={token.pos_:6s} dep={token.dep_:10s} head={token.head.text}")
```

```
The        tag=DT    pos=DET    dep=det        head=cats
cats       tag=NNS   pos=NOUN   dep=nsubj      head=running
were       tag=VBD   pos=AUX    dep=aux        head=running
running    tag=VBG   pos=VERB   dep=ROOT       head=running
at         tag=IN    pos=ADP    dep=prep       head=running
3pm        tag=NN    pos=NOUN   dep=pobj       head=at
.          tag=.     pos=PUNCT  dep=punct      head=running
```

從下往上讀 `dep` 這一欄，句子的文法結構就浮現出來了。

## 實際使用 (Use It)

每個正式的 NLP 函式庫，都把 POS 和 dependency parser 當作標準 pipeline 的一部分提供。

- **spaCy** (`en_core_web_sm` / `md` / `lg` / `trf`)。快、準，與 tokenization、NER、lemmatization 整合。`token.tag_` (Penn)、`token.pos_` (UD)、`token.dep_` (dependency relation)。
- **Stanford NLP (stanza)**。Stanford 對 CoreNLP 的後繼者。在 60 多種語言上達到 state-of-the-art。
- **trankit**。Transformer-based，UD 準確率很好。
- **NLTK**。`pos_tag`。可用，但慢、較舊。適合教學。

### 2026 年這仍然重要的地方

- **Lemmatization。** 第 01 課需要 POS 才能正確 lemmatize，永遠如此。
- **從 LLM 輸出做結構化抽取。** 驗證產生的句子是否遵守文法限制（例如主詞動詞一致、必要的修飾語）。
- **Aspect-based sentiment。** Dependency parse 會告訴你哪個形容詞修飾哪個名詞。
- **查詢理解。** "movies directed by Wes Anderson starring Bill Murray" 透過 parse 拆解成結構化的條件。
- **跨語言轉移。** UD tag 和 dependency relation 與語言無關，讓新語言可以做 zero-shot 的結構化分析。
- **低運算量 pipeline。** 如果你無法部署 transformer，POS 加 dependency parse 加 gazetteer 能帶你走得出乎意料地遠。

## 交付 (Ship It)

存成 `outputs/skill-grammar-pipeline.md`：

```markdown
---
name: grammar-pipeline
description: Design a classical POS + dependency pipeline for a downstream NLP task.
version: 1.0.0
phase: 5
lesson: 07
tags: [nlp, pos, parsing]
---

Given a downstream task (information extraction, rewrite validation, query decomposition, lemmatization), you output:

1. Tagset to use. Penn Treebank for English-only legacy pipelines, Universal Dependencies for multilingual or cross-lingual.
2. Library. spaCy for most production, stanza for academic-grade multilingual, trankit for highest UD accuracy. Name the specific model ID.
3. Integration pattern. Show the 3-5 lines that call the library and consume the needed attributes (`.pos_`, `.dep_`, `.head`).
4. Failure mode to test. Noun-verb ambiguity (`saw`, `book`, `can`) and PP-attachment ambiguity are the classical traps. Sample 20 outputs and eyeball.

Refuse to recommend rolling your own parser. Building parsers from scratch is a research project, not an application task. Flag any pipeline that consumes POS tags without handling lowercase/uppercase variants as fragile.
```

## 練習 (Exercises)

1. **簡單。** 在小型有標註語料庫（例如 NLTK 的 Brown 子集）上使用最常見標籤 baseline，在保留的句子上測量準確率。驗證約 85% 的結果。
2. **中等。** 訓練上面的 bigram HMM，並回報每個 tag 的 precision/recall。HMM 最容易搞混哪些 tag？
3. **困難。** 用 spaCy 的 dependency parse，從 1000 個句子的樣本中抽取主詞-動詞-受詞 (subject-verb-object) 三元組。在 50 個人工標註的三元組上評估。記錄抽取失敗的地方（常見於被動語態、並列結構和省略的主詞）。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| POS tag | 字的類型 | 文法類別。PTB 有 36 個，UD 有 17 個。 |
| Penn Treebank | 標準 tagset | 英文專用。動詞時態與名詞單複數區分得很細。 |
| Universal Dependencies | 多語言 tagset | 比 PTB 粗；與語言無關；跨語言工作的預設。 |
| Dependency parse | 句子的樹 | 每個字有一個 head，每條邊有一個文法關係。 |
| Viterbi | 動態規劃 | 在給定 emission 和 transition 的情況下，找出機率最高的 tag 序列。 |

## 延伸閱讀 (Further Reading)

- [Jurafsky and Martin — Speech and Language Processing, chapters 8 and 18](https://web.stanford.edu/~jurafsky/slp3/) — POS 與 parsing 的標準教科書處理。
- [Universal Dependencies project](https://universaldependencies.org/) — 每個多語言 parser 都使用的跨語言 tagset 與 treebank 集合。
- [spaCy linguistic features guide](https://spacy.io/usage/linguistic-features) — `Token` 上每個屬性的實用參考。
- [Chen and Manning (2014). A Fast and Accurate Dependency Parser using Neural Networks](https://nlp.stanford.edu/pubs/emnlp2014-depparser.pdf) — 把神經網路 parser 帶入主流的論文。
