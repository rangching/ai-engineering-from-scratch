# 命名實體辨識 (Named Entity Recognition)

> 把名字抽出來。聽起來很簡單，直到你遇到模糊的邊界、巢狀的實體和領域術語。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word Embeddings)
**Time:** ~75 分鐘

## 問題 (The Problem)

「Apple sued Google over its iPhone search deal in the US.」有五個實體：Apple (ORG)、Google (ORG)、iPhone (PRODUCT)、search deal（也許）、US (GPE)。好的 NER 系統會抽出全部，並給出正確的類型。差的系統會漏掉 iPhone，把水果 Apple 和公司 Apple 搞混，還把 "US" 標成 PERSON。

NER 是每個結構化抽取 pipeline 底下的主力。履歷解析、合規 log 掃描、病歷匿名化、搜尋查詢理解、聊天機器人回應的 grounding、法律合約抽取。你幾乎看不到它，卻總是依賴它。

這堂課會走過傳統路線（rule-based、HMM、CRF），一路走到現代路線（BiLSTM-CRF，然後是 transformer）。每一步都解決前一步的特定限制。這個模式本身就是這堂課的重點。

## 核心概念 (The Concept)

**BIO tagging**（或 BILOU）把實體抽取變成序列標註問題。每個 token 標上 `B-TYPE`（實體開頭）、`I-TYPE`（實體內部）或 `O`（不在任何實體內）。

```
Apple    B-ORG
sued     O
Google   B-ORG
over     O
its      O
iPhone   B-PRODUCT
search   O
deal     O
in       O
the      O
US       B-GPE
.        O
```

多 token 的實體會串起來：`New B-GPE`、`York I-GPE`、`City I-GPE`。理解 BIO 的模型可以抽出任意範圍的 span。

架構的演進：

- **Rule-based。** Regex 加 gazetteer 查詢。對已知實體的 precision 很高，對新實體完全沒有涵蓋。
- **HMM。** Hidden Markov Model。給定 tag 時 token 的 emission 機率，加上 tag 到 tag 的 transition 機率。用 Viterbi 解碼，在有標註的資料上訓練。
- **CRF。** Conditional Random Field。類似 HMM 但屬於 discriminative，所以你可以混用任意特徵（字形、大小寫、鄰近的字）。2026 年在低資源部署中，仍是傳統的正式環境主力。
- **BiLSTM-CRF。** 用神經網路特徵取代手工特徵。LSTM 從兩個方向讀句子，上面的 CRF 層強制 tag 序列保持一致。
- **Transformer-based。** 用 token-classification head 對 BERT 做 fine-tuning。準確率最好，運算量也最大。

```figure
ner-bio-tagging
```

## 動手做 (Build It)

### 步驟 1：BIO tagging 輔助函式

```python
def spans_to_bio(tokens, spans):
    labels = ["O"] * len(tokens)
    for start, end, label in spans:
        labels[start] = f"B-{label}"
        for i in range(start + 1, end):
            labels[i] = f"I-{label}"
    return labels


def bio_to_spans(tokens, labels):
    spans = []
    current = None
    for i, label in enumerate(labels):
        if label.startswith("B-"):
            if current:
                spans.append(current)
            current = (i, i + 1, label[2:])
        elif label.startswith("I-") and current and current[2] == label[2:]:
            current = (current[0], i + 1, current[2])
        else:
            if current:
                spans.append(current)
                current = None
    if current:
        spans.append(current)
    return spans
```

```python
>>> tokens = ["Apple", "sued", "Google", "over", "iPhone", "sales", "."]
>>> labels = ["B-ORG", "O", "B-ORG", "O", "B-PRODUCT", "O", "O"]
>>> bio_to_spans(tokens, labels)
[(0, 1, 'ORG'), (2, 3, 'ORG'), (4, 5, 'PRODUCT')]
```

### 步驟 2：手工特徵

對傳統（非神經網路）NER 來說，特徵就是關鍵。有用的特徵有：

```python
def token_features(token, prev_token, next_token):
    return {
        "lower": token.lower(),
        "is_upper": token.isupper(),
        "is_title": token.istitle(),
        "has_digit": any(c.isdigit() for c in token),
        "suffix_3": token[-3:].lower(),
        "shape": word_shape(token),
        "prev_lower": prev_token.lower() if prev_token else "<BOS>",
        "next_lower": next_token.lower() if next_token else "<EOS>",
    }


def word_shape(word):
    out = []
    for c in word:
        if c.isupper():
            out.append("X")
        elif c.islower():
            out.append("x")
        elif c.isdigit():
            out.append("d")
        else:
            out.append(c)
    return "".join(out)
```

`word_shape("iPhone")` 回傳 `xXxxxx`。`word_shape("USA-2024")` 回傳 `XXX-dddd`。大小寫模式對專有名詞是高訊號。

### 步驟 3：簡單的 rule-based 加字典 baseline

```python
ORG_GAZETTEER = {"Apple", "Google", "Microsoft", "OpenAI", "Meta", "Amazon", "Netflix"}
GPE_GAZETTEER = {"US", "USA", "UK", "India", "Germany", "France"}
PRODUCT_GAZETTEER = {"iPhone", "Android", "Windows", "ChatGPT", "Claude"}


def rule_based_ner(tokens):
    labels = []
    for token in tokens:
        if token in ORG_GAZETTEER:
            labels.append("B-ORG")
        elif token in GPE_GAZETTEER:
            labels.append("B-GPE")
        elif token in PRODUCT_GAZETTEER:
            labels.append("B-PRODUCT")
        else:
            labels.append("O")
    return labels
```

正式的 gazetteer 有從 Wikipedia 和 DBpedia 爬來的數百萬筆項目。涵蓋率很好，消歧（`Apple` 是公司還是水果）卻很糟。這就是統計模型勝出的原因。

### 步驟 4：CRF 步驟（草圖，不是完整實作）

沒有機率論的基礎，用 50 行從零寫完整的 CRF 並不能讓人理解什麼。改用 `sklearn-crfsuite`：

```python
import sklearn_crfsuite

def to_features(tokens):
    out = []
    for i, tok in enumerate(tokens):
        prev = tokens[i - 1] if i > 0 else ""
        nxt = tokens[i + 1] if i + 1 < len(tokens) else ""
        out.append({
            "word.lower()": tok.lower(),
            "word.isupper()": tok.isupper(),
            "word.istitle()": tok.istitle(),
            "word.isdigit()": tok.isdigit(),
            "word.suffix3": tok[-3:].lower(),
            "word.shape": word_shape(tok),
            "prev.word.lower()": prev.lower(),
            "next.word.lower()": nxt.lower(),
            "BOS": i == 0,
            "EOS": i == len(tokens) - 1,
        })
    return out


crf = sklearn_crfsuite.CRF(algorithm="lbfgs", c1=0.1, c2=0.1, max_iterations=100, all_possible_transitions=True)
X_train = [to_features(s) for s in sentences_tokenized]
crf.fit(X_train, bio_labels_train)
```

`c1` 和 `c2` 是 L1 與 L2 regularization。`all_possible_transitions=True` 讓模型學到非法序列（例如 `O` 後面接 `I-ORG`）的機率很低，這就是 CRF 在你不必自己寫限制的情況下，強制 BIO 一致性的方式。

### 步驟 5：BiLSTM-CRF 多了什麼

特徵變成學出來的。輸入：token embedding（GloVe 或 fastText）。LSTM 從左到右、從右到左各讀一次。串接後的 hidden state 經過 CRF 輸出層。CRF 仍然強制 tag 序列一致，LSTM 則用學出來的特徵取代手工特徵。

```python
import torch
import torch.nn as nn


class BiLSTM_CRF_Head(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_labels):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, bidirectional=True, batch_first=True)
        self.fc = nn.Linear(hidden_dim * 2, n_labels)

    def forward(self, token_ids):
        e = self.embed(token_ids)
        h, _ = self.lstm(e)
        emissions = self.fc(h)
        return emissions
```

CRF 層請用 `torchcrf.CRF`（pip install pytorch-crf）。除非你有數萬句有標註的句子，否則它相對於手工特徵 CRF 的提升雖然可以測量，但比你預期的小。

## 實際使用 (Use It)

spaCy 開箱即附帶正式等級的 NER。

```python
import spacy

nlp = spacy.load("en_core_web_sm")
doc = nlp("Apple sued Google over its iPhone search deal in the US.")
for ent in doc.ents:
    print(f"{ent.text:20s} {ent.label_}")
```

```
Apple                ORG
Google               ORG
iPhone               ORG
US                   GPE
```

注意 `iPhone` 被標成 `ORG` 而不是 `PRODUCT`，spaCy 的 small 模型對產品實體的涵蓋很弱。large 模型 (`en_core_web_lg`) 比較好，transformer 模型 (`en_core_web_trf`) 又更好。

用 Hugging Face 做 BERT-based NER：

```python
from transformers import pipeline

ner = pipeline("ner", model="dslim/bert-base-NER", aggregation_strategy="simple")
print(ner("Apple sued Google over its iPhone in the US."))
```

```
[{'entity_group': 'ORG', 'word': 'Apple', ...},
 {'entity_group': 'ORG', 'word': 'Google', ...},
 {'entity_group': 'MISC', 'word': 'iPhone', ...},
 {'entity_group': 'LOC', 'word': 'US', ...}]
```

`aggregation_strategy="simple"` 會把連續的 B-X、I-X token 合併成一個 span。沒有它的話，你拿到的是 token 層級的標籤，得自己合併。

### 以 LLM 為基礎的 NER（2026 年的選項）

Zero-shot 與 few-shot 的 LLM NER，現在在許多領域已經能與 fine-tuned 模型競爭，在標註資料稀少時更是好得多。

- **Zero-shot prompting。** 給 LLM 一份實體類型清單和範例 schema，要求 JSON 輸出。開箱即用，在新領域的準確率中等。
- **ZeroTuneBio 風格的 prompting。** 把任務拆解成候選抽取 → 意義說明 → 判斷 → 重新檢查。多階段的 prompt（不是 one-shot）能大幅提升生醫 NER 的準確率。同樣的模式也適用於法律、金融與科學領域。
- **結合 RAG 的動態 prompting。** 每次推論時，從小型標註種子集中檢索最相似的標註範例，即時建構 few-shot prompt。在 2026 年的 benchmark 中，這讓 GPT-4 的生醫 NER F1 比靜態 prompting 高出 11 到 12%。
- **依實體類型拆解。** 對長文件，一次呼叫抽取所有實體類型，recall 會隨長度增加而下降。改成每種實體類型各跑一次抽取。推論成本較高，但準確率明顯更高。這是臨床筆記和法律合約的標準做法。

截至 2026 年的正式環境建議：在蒐集訓練資料之前，先用 LLM zero-shot 建立 baseline。F1 常常已經夠好，你根本不需要 fine-tune。

### 傳統 NER 依然勝出的地方

即使有 LLM 可用，在下列情況傳統 NER 仍然勝出：

- 延遲預算低於 50 毫秒。
- 你有數千筆標註範例，並且需要 98% 以上的 F1。
- 領域有穩定的 ontology，預訓練的 CRF 或 BiLSTM 能良好轉移。
- 法規限制要求使用地端、非生成式的模型。

### 它會失靈的地方

- **領域轉移 (Domain shift)。** 用 CoNLL 訓練的 NER 用在法律合約上，表現比 gazetteer 還差。請在你的領域上 fine-tune。
- **巢狀實體 (Nested entities)。** "Bank of America Tower" 同時是 ORG 和 FACILITY。標準 BIO 無法表示重疊的 span。你需要 nested NER（多輪或 span-based 模型）。
- **長實體。** "United States Federal Deposit Insurance Corporation." token 層級的模型有時會把它切開。請使用 `aggregation_strategy` 或做後處理。
- **稀疏類型。** 醫學 NER 的標籤，如 DRUG_BRAND、ADVERSE_EVENT、DOSE。通用模型完全不懂。Scispacy 和 BioBERT 是那裡的起點。

## 交付 (Ship It)

存成 `outputs/skill-ner-picker.md`：

```markdown
---
name: ner-picker
description: Pick the right NER approach for a given extraction task.
version: 1.0.0
phase: 5
lesson: 06
tags: [nlp, ner, extraction]
---

Given a task description (domain, label set, language, latency, data volume), output:

1. Approach. Rule-based + gazetteer, CRF, BiLSTM-CRF, or transformer fine-tune.
2. Starting model. Name it (spaCy model ID, Hugging Face checkpoint ID, or "custom, trained from scratch").
3. Labeling strategy. BIO, BILOU, or span-based. Justify in one sentence.
4. Evaluation. Use `seqeval`. Always report entity-level F1 (not token-level).

Refuse to recommend fine-tuning a transformer for under 500 labeled examples unless the user already has a pretrained domain model. Flag nested entities as needing span-based or multi-pass models. Require a gazetteer audit if the user mentions "production scale" and labels are unchanged from CoNLL-2003.
```

## 練習 (Exercises)

1. **簡單。** 實作 `bio_to_spans`（`spans_to_bio` 的反向操作），並在 10 個句子上驗證來回轉換的一致性。
2. **中等。** 在 CoNLL-2003 英文 NER 資料集上訓練上面的 sklearn-crfsuite CRF。用 `seqeval` 回報各實體的 F1。典型結果：約 84 F1。
3. **困難。** 在特定領域的 NER 資料集（醫學、法律或金融）上 fine-tune `distilbert-base-cased`。與 spaCy small 模型比較。記錄資料洩漏檢查，並寫下讓你意外的地方。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| NER | 抽出名字 | 為 token span 標上類型（PERSON、ORG、GPE、DATE 等）。 |
| BIO | 標註方案 | `B-X` 開始，`I-X` 延續，`O` 在外面。 |
| BILOU | 更好的 BIO | 加上 `L-X`（最後）、`U-X`（單一），讓邊界更乾淨。 |
| CRF | 結構化分類器 | 對標籤之間的 transition 建模，而不只是 emission。強制產生有效序列。 |
| Nested NER | 重疊的實體 | 一個 span 與它的子 span 是不同的實體。BIO 無法表達。 |
| Entity-level F1 | 正確的 NER 指標 | 預測的 span 必須與真實 span 完全相符。token 層級的 F1 會高估準確率。 |

## 延伸閱讀 (Further Reading)

- [Lample et al. (2016). Neural Architectures for Named Entity Recognition](https://arxiv.org/abs/1603.01360) — BiLSTM-CRF 論文，經典。
- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers](https://arxiv.org/abs/1810.04805) — 引入了後來成為標準的 token-classification 模式。
- [spaCy linguistic features — named entities](https://spacy.io/usage/linguistic-features#named-entities) — `Doc.ents` 和 `Span` 上每個屬性的實用參考。
- [seqeval](https://github.com/chakki-works/seqeval) — 正確的指標函式庫，請永遠使用它。
