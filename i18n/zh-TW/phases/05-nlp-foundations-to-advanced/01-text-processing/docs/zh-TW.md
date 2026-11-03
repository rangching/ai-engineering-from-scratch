# 文字處理 (Text Processing) — Tokenization、Stemming、Lemmatization

> 語言是連續的，模型是離散的。預處理就是中間的橋樑。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2 · 14 (Naive Bayes)
**Time:** ~45 分鐘

## 問題 (The Problem)

模型讀不懂「The cats were running.」，它只讀得懂整數。

每個 NLP 系統一開始都要回答同樣三個問題。一個字從哪裡開始？這個字的詞根是什麼？怎樣讓 "run"、"running"、"ran" 在有幫助的時候被當成同一件事，沒幫助的時候又被當成不同的事？

tokenization 做錯，模型就會從垃圾資料學習。如果你的 tokenizer 把 `don't` 當成一個 token，卻把 `do n't` 當成兩個，訓練分布就會分裂。如果你的 stemmer 把 `organization` 和 `organ` 縮成同一個 stem，topic modeling 就毀了。如果你的 lemmatizer 需要詞性 (part-of-speech) 上下文，你卻沒有提供，動詞就會被當成名詞。

這堂課會從零開始建立這三個預處理步驟，再示範 NLTK 和 spaCy 怎麼做同樣的事，讓你看到取捨。

## 核心概念 (The Concept)

三種操作，各有各的工作，也各有各的失敗模式。

**Tokenization** 把字串切成 token。「Token」刻意定義得很模糊，因為合適的粒度取決於任務。傳統 NLP 用 word-level，transformer 用 subword，沒有空白分隔的語言用 character。

**Stemming** 用規則砍掉字尾。快、激進、很笨。`running -> run`、`organization -> organ`。第二個就是失敗模式。

**Lemmatization** 用文法知識把字還原成字典形式。比較慢、比較準，需要查詢表或 morphological analyzer。`ran -> run`（要知道 "ran" 是 "run" 的過去式）。`better -> good`（要知道比較級的形式）。

經驗法則：速度重要、又能容忍雜訊時用 stem（搜尋索引、粗略分類）。語意重要時用 lemmatize（question answering、semantic search、任何使用者會讀到的東西）。

```figure
edit-distance
```

## 動手做 (Build It)

### 步驟 1：regex word tokenizer

最簡單又有用的 tokenizer，是在非英數字元處切開，同時把標點符號保留成獨立的 token。不完美，也不是最終版，但一行就能跑。

```python
import re

def tokenize(text):
    return re.findall(r"[A-Za-z]+(?:'[A-Za-z]+)?|[0-9]+|[^\sA-Za-z0-9]", text)
```

三個 pattern，依優先順序排列：可以含一個內部單引號的字（`don't`、`it's`）、純數字、以及任何單一的非空白非英數字元（標點符號）當成獨立 token。

```python
>>> tokenize("The cats weren't running at 3pm.")
['The', 'cats', "weren't", 'running', 'at', '3', 'pm', '.']
```

要注意的失敗模式：`3pm` 會被切成 `['3', 'pm']`，因為我們在字母連續段與數字連續段之間交替。對多數任務已經夠用。URL、email、hashtag 都會壞掉。要上線的話，在通用 pattern 前面加上專用 pattern。

### 步驟 2：Porter stemmer（只做 step 1a）

完整的 Porter 演算法有五個階段的規則。光是 step 1a，就涵蓋了最常見的英文字尾，也足以教會你這個模式。

```python
def stem_step_1a(word):
    if word.endswith("sses"):
        return word[:-2]
    if word.endswith("ies"):
        return word[:-2]
    if word.endswith("ss"):
        return word
    if word.endswith("s") and len(word) > 1:
        return word[:-1]
    return word
```

```python
>>> [stem_step_1a(w) for w in ["caresses", "ponies", "caress", "cats"]]
['caress', 'poni', 'caress', 'cat']
```

規則由上往下讀。`ies -> i` 這條規則就是 `ponies -> poni` 而不是 `pony` 的原因。真正的 Porter 有 step 1b 可以修正。規則會互相競爭，前面的規則優先。順序比任何單一規則都重要。

### 步驟 3：以查詢表為基礎的 lemmatizer

真正的 lemmatization 需要 morphology。一個容易教學的版本，是用一張小型 lemma 表加上 fallback。

```python
LEMMA_TABLE = {
    ("running", "VERB"): "run",
    ("ran", "VERB"): "run",
    ("runs", "VERB"): "run",
    ("better", "ADJ"): "good",
    ("best", "ADJ"): "good",
    ("cats", "NOUN"): "cat",
    ("cat", "NOUN"): "cat",
    ("were", "VERB"): "be",
    ("was", "VERB"): "be",
    ("is", "VERB"): "be",
}

def lemmatize(word, pos):
    key = (word.lower(), pos)
    if key in LEMMA_TABLE:
        return LEMMA_TABLE[key]
    if pos == "VERB" and word.endswith("ing"):
        return word[:-3]
    if pos == "NOUN" and word.endswith("s"):
        return word[:-1]
    return word.lower()
```

```python
>>> lemmatize("running", "VERB")
'run'
>>> lemmatize("cats", "NOUN")
'cat'
>>> lemmatize("better", "ADJ")
'good'
>>> lemmatize("watched", "VERB")
'watched'
```

最後一個案例是關鍵的教學時刻。`watched` 不在我們的表裡，而我們的 fallback 只處理 `ing`。真正的 lemmatization 要涵蓋 `ed`、不規則動詞、比較級形容詞，以及帶有音變的複數（`children -> child`）。這就是為什麼正式系統會使用 WordNet、spaCy 的 morphologizer，或完整的 morphological analyzer。

### 步驟 4：把它們接起來

```python
def preprocess(text, pos_tagger=None):
    tokens = tokenize(text)
    stems = [stem_step_1a(t.lower()) for t in tokens]
    tags = pos_tagger(tokens) if pos_tagger else [(t, "NOUN") for t in tokens]
    lemmas = [lemmatize(word, pos) for word, pos in tags]
    return {"tokens": tokens, "stems": stems, "lemmas": lemmas}
```

缺少的一塊是 POS tagger。Phase 5 · 07 (POS Tagging) 會建一個。現在先把所有東西預設為 `NOUN`，並承認這個限制。

## 實際使用 (Use It)

NLTK 和 spaCy 提供了正式版本，各只要幾行。

### NLTK

```python
import nltk
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("averaged_perceptron_tagger_eng")

from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk import pos_tag

text = "The cats were running."
tokens = word_tokenize(text)
stems = [PorterStemmer().stem(t) for t in tokens]
lemmatizer = WordNetLemmatizer()
tagged = pos_tag(tokens)


def nltk_pos_to_wordnet(tag):
    if tag.startswith("V"):
        return "v"
    if tag.startswith("J"):
        return "a"
    if tag.startswith("R"):
        return "r"
    return "n"


lemmas = [lemmatizer.lemmatize(t, nltk_pos_to_wordnet(tag)) for t, tag in tagged]
```

`word_tokenize` 能處理縮寫、Unicode，以及你的 regex 漏掉的邊界情況。`PorterStemmer` 會跑完全部五個階段。`WordNetLemmatizer` 需要把 POS tag 從 NLTK 的 Penn Treebank 體系轉成 WordNet 的縮寫集合。上面這段轉換的接線，正是大多數教學跳過的部分。

### spaCy

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("The cats were running.")

for token in doc:
    print(token.text, token.lemma_, token.pos_)
```

```
The      the     DET
cats     cat     NOUN
were     be      AUX
running  run     VERB
.        .       PUNCT
```

spaCy 把整條 pipeline 藏在 `nlp(text)` 後面。Tokenization、POS tagging、lemmatization 全部會執行。大規模時比 NLTK 快，開箱即用也比較準。代價是你不容易替換個別元件。

### 該選哪一個

| 情境 | 選擇 |
|-----------|------|
| 教學、研究、需要替換元件 | NLTK |
| 正式環境、多語言、速度重要 | spaCy |
| Transformer pipeline（反正你會用模型自己的 tokenizer） | 使用 `tokenizers` / `transformers`，跳過傳統預處理 |

### 沒有人警告過你的兩種失敗模式

大多數教學只教演算法就停了。有兩件事會咬到真正的預處理 pipeline，而且幾乎沒人談。

**可重現性漂移 (Reproducibility drift)。** NLTK 和 spaCy 在不同版本之間，會改變 tokenization 和 lemmatizer 的行為。spaCy 2.x 產生 `['do', "n't"]` 的地方，3.x 可能產生 `["don't"]`。你的模型是用一種分布訓練的，推論卻跑在另一種分布上。準確率悄悄下降，沒有人知道為什麼。請在 `requirements.txt` 裡固定函式庫版本。寫一個預處理 regression test，把 20 個範例句子的預期 tokenization 凍結起來，每次升級都跑一次。

**訓練與推論不一致 (Training / inference mismatch)。** 訓練時用激進的預處理（轉小寫、移除 stopword、stemming），部署時卻吃原始的使用者輸入，然後看著效能崩盤。這是正式環境 NLP 最常見的單一失敗。如果訓練時有預處理，推論時就必須執行完全相同的函式。把預處理當成模型套件裡的函式交付，而不是讓 serving 團隊重寫的 notebook cell。

## 交付 (Ship It)

一個可重複使用的 prompt，幫工程師不用讀三本教科書就能挑出預處理策略。

存成 `outputs/prompt-preprocessing-advisor.md`：

```markdown
---
name: preprocessing-advisor
description: Recommends a tokenization, stemming, and lemmatization setup for an NLP task.
phase: 5
lesson: 01
---

You advise on classical NLP preprocessing. Given a task description, you output:

1. Tokenization choice (regex, NLTK word_tokenize, spaCy, or transformer tokenizer). Explain why.
2. Whether to stem, lemmatize, both, or neither. Explain why.
3. Specific library calls. Name the functions. Quote the POS-tag translation if NLTK is involved.
4. One failure mode the user should test for.

Refuse to recommend stemming for user-visible text. Refuse to recommend lemmatization without POS tags. Flag non-English input as needing a different pipeline.
```

## 練習 (Exercises)

1. **簡單。** 擴充 `tokenize`，讓 URL 保持為單一 token。測試：`tokenize("Visit https://example.com today.")` 應該產生一個 URL token。
2. **中等。** 實作 Porter step 1b。如果一個字含有母音且以 `ed` 或 `ing` 結尾，就移除它。要處理雙子音規則（`hopping -> hop`，而不是 `hopp`）。
3. **困難。** 建立一個 lemmatizer，以 WordNet 當查詢表，WordNet 沒有該詞時退回你的 Porter stemmer。在有標註的語料庫上，對照純 WordNet 與純 Porter 測量準確率。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Token | 一個字 | 模型消化的任何單位。可以是 word、subword、character 或 byte。 |
| Stem | 字的詞根 | 以規則剝除字尾的結果。不一定是真正的字。 |
| Lemma | 字典形式 | 你會去查的那個形式。需要文法上下文才能正確算出。 |
| POS tag | 詞性 | 像 NOUN、VERB、ADJ 這樣的類別。要準確 lemmatize 就需要它。 |
| Morphology | 字形規則 | 字如何依時態、單複數、格而改變形式。Lemmatization 依賴它。 |

## 延伸閱讀 (Further Reading)

- [Porter, M. F. (1980). An algorithm for suffix stripping](https://tartarus.org/martin/PorterStemmer/def.txt) — 原始論文，五頁，至今仍是最清楚的解說。
- [spaCy 101 — linguistic features](https://spacy.io/usage/linguistic-features) — 真實的 pipeline 是怎麼接起來的。
- [NLTK book, chapter 3](https://www.nltk.org/book/ch03.html) — 你還沒想到的 tokenization 邊界情況。
