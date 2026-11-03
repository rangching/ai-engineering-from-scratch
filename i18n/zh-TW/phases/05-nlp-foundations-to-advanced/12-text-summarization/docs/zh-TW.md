# 文字摘要 (Text Summarization)

> Extractive 系統告訴你文件說了什麼。Abstractive 系統告訴你作者的意思是什麼。任務不同，陷阱也不同。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Time:** ~75 分鐘

## 問題 (The Problem)

一篇 2,000 字的新聞出現在你的動態中。你需要 120 個字抓住重點。你可以從文章中挑出最重要的三個句子 (extractive)，也可以用自己的話重寫內容 (abstractive)。兩者都叫摘要，卻是完全不同的問題。

Extractive 摘要是排序問題。為每個句子打分，回傳前 `k` 名。輸出一定合乎文法，因為是原封不動取出來的。風險是會漏掉分散在文章各處的內容。

Abstractive 摘要是生成問題。Transformer 以輸入為條件產生新文字。輸出流暢又精簡，但可能捏造來源裡沒有的事實。風險是信心滿滿地虛構。

這堂課會兩種都建立，並帶出各自擁有的失敗模式。

## 核心概念 (The Concept)

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive。** 把文章當成一張圖，節點是句子，邊是相似度。在圖上跑 PageRank（或類似的演算法），依句子與其他所有句子的連結程度打分。分數最高的句子就是摘要。標準的實作是 **TextRank**（Mihalcea 和 Tarau，2004）。

**Abstractive。** 在文件-摘要配對上 fine-tune transformer encoder-decoder（BART、T5、Pegasus）。推論時，模型讀文件，並透過 cross-attention 逐 token 產生摘要。尤其是 Pegasus，它使用 gap-sentence 預訓練目標，讓它不必多做 fine-tuning 就很擅長摘要。

用 **ROUGE**（Recall-Oriented Understudy for Gisting Evaluation）評估。ROUGE-1 和 ROUGE-2 為 unigram 和 bigram 重疊打分。ROUGE-L 為最長共同子序列 (longest common subsequence) 打分。越高越好，但 40 ROUGE-L 算「好」，50 算「卓越」。每篇論文都會回報三者。請使用 `rouge-score` 套件。

```figure
summarize-collapse
```

## 動手做 (Build It)

### 步驟 1：TextRank (extractive)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

有兩件事值得點名。相似度函數使用 log 正規化的字詞重疊，這是原始的 TextRank 變體。TF-IDF 向量的 cosine 也可以用。阻尼係數 0.85 與迭代次數是 PageRank 的預設值。

### 步驟 2：用 BART 做 abstractive

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN 是在 CNN/DailyMail 語料庫上 fine-tune 的。它開箱即用就能產生新聞風格的摘要。對其他領域（科學論文、對話、法律），請使用對應的 Pegasus checkpoint，或在你的目標資料上 fine-tune。

### 步驟 3：ROUGE 評估

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

請永遠使用 stemming。沒有它的話，"running" 和 "run" 會被當成不同的字，ROUGE 就會少算。

### 超越 ROUGE（2026 年的摘要評估）

ROUGE 二十年來一直是主流的摘要指標，在 2026 年單靠它已經不夠。一份針對 NLG 論文的大規模後設分析顯示：

- **BERTScore**（contextual embedding 相似度）在 2023 年之前逐漸取得地位，現在大多數摘要論文都會與 ROUGE 一起回報。
- **BARTScore** 把評估當成生成：依預訓練 BART 在給定來源的情況下，賦予該摘要的可能性來打分。
- **MoverScore**（對 contextual embedding 的 Earth Mover's Distance）在 2025 年的摘要 benchmark 中登上榜首，因為它比 ROUGE 更能捕捉語意重疊。
- **FactCC** 與 **QA-based faithfulness** 在 2021 到 2023 年很常見，現在常被 **G-Eval** 取代（一個 GPT-4 的 prompt 鏈，用 chain-of-thought 推理為連貫性、一致性、流暢度、相關性打分）。
- 當評分標準設計得好時，**G-Eval** 與類似的 LLM-judge 方法與人類判斷的吻合率約 80%。

正式環境的建議：回報 ROUGE-L 以便與舊結果比較，回報 BERTScore 看語意重疊，回報 G-Eval 看連貫性與事實性。用 50 到 100 筆人工標註的摘要校準。

### 步驟 4：事實性問題

Abstractive 摘要容易產生幻覺。Extractive 摘要的幻覺風險低得多，因為輸出是原封不動從來源取出的，不過如果來源句子被脫離上下文、已經過時，或引用順序顛倒，它們仍可能誤導讀者。這是正式系統在與合規相關的內容上，仍然偏好 extractive 方法的單一最大原因。

要點名的幻覺類型：

- **Entity swap（實體替換）。** 來源寫 "John Smith."，摘要寫 "John Brown."。
- **Number drift（數字漂移）。** 來源寫 "25,000."，摘要寫 "25 million."。
- **Polarity flip（極性翻轉）。** 來源寫 "rejected the offer."，摘要寫 "accepted the offer."。
- **Fact invention（事實捏造）。** 來源沒有提到 CEO，摘要卻說 CEO 核准了。

有效的評估方法：

- **FactCC。** 針對來源句子與摘要句子之間的 entailment 訓練的二元分類器。預測事實 / 非事實。
- **QA-based factuality。** 向 QA 模型提出答案在來源中的問題。如果摘要支持不同的答案，就標記出來。
- **Entity-level F1。** 比較來源與摘要中的命名實體。只出現在摘要裡的實體是可疑的。

對任何面向使用者、事實性重要的內容（新聞、醫療、法律、金融），extractive 是比較安全的預設。Abstractive 需要在流程中加入事實性檢查。

## 實際使用 (Use It)

2026 年的技術棧：

| 使用情境 | 建議 |
|---------|-------------|
| 新聞、3 到 5 句摘要、英文 | `facebook/bart-large-cnn` |
| 科學論文 | `google/pegasus-pubmed` 或調整過的 T5 |
| 多文件、長篇 | 任何 context 32k 以上、經過提示的 LLM |
| 對話摘要 | `philschmid/bart-large-cnn-samsum` |
| Extractive，設計上幻覺風險低 | TextRank 或 `sumy` 的 LSA / LexRank |

在運算量不受限的情況下，2026 年具備長 context 的 LLM 常常勝過專用模型。代價是成本與可重現性；專用模型提供比較一致的輸出。

## 交付 (Ship It)

存成 `outputs/skill-summary-picker.md`：

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## 練習 (Exercises)

1. **簡單。** 對 5 篇新聞跑 TextRank。把前 3 名句子與參考摘要比較。測量 ROUGE-L。在 CNN/DailyMail 風格的文章上，你應該會看到 30 到 45 的 ROUGE-L。
2. **中等。** 實作 entity-level 的事實性檢查：用 spaCy 從來源與摘要抽取命名實體，計算來源實體在摘要中的 recall，以及摘要實體對照來源的 precision。高 precision 加低 recall 代表安全但簡短；低 precision 代表有捏造的實體。
3. **困難。** 在 50 篇 CNN/DailyMail 文章上，比較 BART-large-CNN 與 LLM（Claude 或 GPT-4）。回報 ROUGE-L、事實性（以 entity F1 衡量），以及每份摘要的成本。記錄各自在哪裡勝出。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Extractive | 挑句子 | 原封不動回傳來源中的句子。永遠不會產生幻覺。 |
| Abstractive | 重寫 | 以來源為條件產生新文字。可能產生幻覺。 |
| ROUGE | 摘要指標 | 系統輸出與參考摘要之間的 n-gram / LCS 重疊。 |
| TextRank | 圖為基礎的 extractive | 在句子相似度圖上跑 PageRank。 |
| Factuality | 對不對 | 摘要的主張是否有來源支持。 |
| Hallucination | 捏造的內容 | 摘要中來源無法支持的內容。 |

## 延伸閱讀 (Further Reading)

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — extractive 的經典論文。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — BART 論文。
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus 與 gap-sentence 目標。
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — ROUGE 論文。
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — 事實性全景的論文。
