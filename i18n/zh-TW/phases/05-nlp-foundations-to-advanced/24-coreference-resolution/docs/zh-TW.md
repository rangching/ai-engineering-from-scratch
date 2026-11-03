# 共指消解 (Coreference Resolution)

> 「She called him. He did not answer. The doctor was at lunch.」兩個人被指涉了三次，卻沒有人被點名。Coreference resolution 要弄清楚誰是誰。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 07 (POS & Parsing)
**Time:** ~60 分鐘

## 問題 (The Problem)

要從一篇 300 字的文章中抽出所有提到 Apple Inc. 的地方。當文章寫 "Apple" 時很容易。當它寫 "the company"、"they"、"Cupertino's technology giant" 或 "Jobs's firm" 時就很難。如果不把這些提及 (mention) 解析到同一個實體，你的 NER pipeline 會漏掉 60 到 80% 的提及。

Coreference resolution 把所有指涉同一個真實世界實體的表達連結成一個群集 (cluster)。它是表層 NLP（NER、parsing）和下游語意（IE、QA、摘要、KG）之間的黏著劑。

為什麼它在 2026 年很重要：

- 摘要："The CEO announced..." vs "Tim Cook announced..."，摘要應該寫出 CEO 的名字。
- 問答："Who did she call?" 需要解析 "she"。
- 資訊抽取：如果知識圖譜裡有 "PER1 founded Apple" 和 "Jobs founded Apple" 兩筆分開的項目，那就是錯的。
- 多文件 IE：把關於同一事件的多篇文章中的提及合併，就是 cross-document coreference。

## 核心概念 (The Concept)

![Coreference clustering: mentions → entities](../assets/coref.svg)

**任務。** 輸入：一份文件。輸出：對提及 (span) 的分群，每個群集指涉一個實體。

**提及的類型。**

- **命名實體 (Named entity)。** "Tim Cook"
- **名詞性 (Nominal)。** "the CEO"、"the company"
- **代名詞性 (Pronominal)。** "he"、"she"、"they"、"it"
- **同位語 (Appositive)。** "Tim Cook, Apple's CEO,"

**架構。**

1. **Rule-based（Hobbs, 1978）。** 用文法規則、以句法樹為基礎的代名詞解析。很好的 baseline，在代名詞上出奇地難被超越。
2. **Mention-pair 分類器。** 對每一對提及 $(m_i, m_j)$，預測它們是否共指。再用遞移閉包 (transitive closure) 分群。2016 年之前的標準做法。
3. **Mention-ranking。** 對每個提及，為候選先行詞（包含「沒有先行詞」）排序，挑最高的。
4. **Span-based 端到端（Lee et al., 2017）。** Transformer encoder。列舉所有不超過長度上限的候選 span，預測提及分數，為每個 span 預測先行詞機率，再貪婪地分群。現代的預設。
5. **生成式（2024 起）。** 提示 LLM：「List every pronoun in this text and its antecedent.」在簡單案例上效果好，在長文件和罕見指涉對象上則吃力。

**評估指標。** 五個標準指標（MUC、B³、CEAF、BLANC、LEA），因為沒有單一指標能捕捉分群品質。回報前三者的平均，稱為 CoNLL F1。2026 年在 CoNLL-2012 上的 state-of-the-art 約為 83 F1。

**已知的困難案例。**

- 指涉好幾頁前就出現的實體的定指描述。
- 橋接回指 (Bridging anaphora)（"the wheels" → 先前提到的一輛車）。
- 中文和日文等語言中的零回指 (zero anaphora)。
- 後指 (Cataphora)（代名詞出現在指涉對象之前）："When **she** walked in, Mary smiled."

```figure
coref-links
```

## 動手做 (Build It)

### 步驟 1：預訓練的神經網路 coreference（AllenNLP / spaCy-experimental）

```python
import spacy
nlp = spacy.load("en_coreference_web_trf")   # experimental model
doc = nlp("Apple announced new products. The company said it would ship them soon.")
for cluster in doc._.coref_clusters:
    print(cluster, "->", [m.text for m in cluster])
```

在較長的文件上，你會得到類似這樣的結果：
- Cluster 1: [Apple, The company, it]
- Cluster 2: [new products, them]

### 步驟 2：rule-based 代名詞解析器（教學）

僅用標準函式庫的實作見 `code/main.py`：

1. 抽取提及：命名實體（首字母大寫的 span）、代名詞（字典查詢）、定指描述 ("the X")。
2. 對每個代名詞，檢視前 K 個提及，並依以下項目打分：
   - 性別 / 數的一致性（啟發法）
   - 近因性 (recency)（越近越好）
   - 句法角色（偏好主詞）
3. 連結分數最高的先行詞。

與神經網路模型沒得比，但它展示了搜尋空間，以及端到端模型必須做出的決定。

### 步驟 3：用 LLM 做 coreference

```python
prompt = f"""Text: {text}

List every pronoun and noun phrase that refers to a person or company.
Cluster them by what they refer to. Output JSON:
[{{"entity": "Apple", "mentions": ["Apple", "the company", "it"]}}, ...]
"""
```

要留意兩種失敗模式。第一，LLM 會過度合併（把指涉兩個不同人的 "him" 和 "her" 合在一起）。第二，LLM 在長文件中會無聲地丟掉提及。一定要用 span 偏移量檢查來驗證。

### 步驟 4：評估

標準的 conll-2012 腳本會計算 MUC、B³、CEAF-φ4 並回報平均。對內部評估，先從你標註的測試集上 span 層級的 precision 和 recall 開始，再加上 mention-linking F1。

## 陷阱

- **Singleton 爆炸。** 有些系統把每個提及都回報為自己的群集。B³ 對此寬容，MUC 則會懲罰。一定要檢查三個指標。
- **長 context 中的代名詞。** 在超過 2,000 token 的文件上，效能下降約 15 F1。小心地 chunk。
- **性別假設。** 寫死的性別規則，遇到非二元的指涉對象、組織、動物就會壞掉。請使用學習而來的模型或中性的打分。
- **LLM 在長文件上的漂移。** 單次 API 呼叫無法可靠地對 50 個以上段落的提及分群。請使用滑動視窗加合併。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| 英文、單一文件 | `en_coreference_web_trf`（spaCy-experimental）或 AllenNLP 神經網路 coref |
| 多語言 | 在 OntoNotes 或 Multilingual CoNLL 上訓練的 SpanBERT / XLM-R |
| 跨文件事件 coref | 專門的端到端模型（2025 到 2026 年 SOTA） |
| 快速的 LLM baseline | 搭配結構化輸出 coref prompt 的 GPT-4o / Claude |
| 正式的對話系統 | rule-based fallback 加神經網路為主，關鍵 slot 由人工審查 |

2026 年實際出貨的整合模式：先跑 NER，再跑 coref，把 coref 群集合併進 NER 實體。下游任務看到的是每個群集一個實體，而不是每個提及一個實體。

## 交付 (Ship It)

存成 `outputs/skill-coref-picker.md`：

```markdown
---
name: coref-picker
description: Pick a coreference approach, evaluation plan, and integration strategy.
version: 1.0.0
phase: 5
lesson: 24
tags: [nlp, coref, information-extraction]
---

Given a use case (single-doc / multi-doc, domain, language), output:

1. Approach. Rule-based / neural span-based / LLM-prompted / hybrid. One-sentence reason.
2. Model. Named checkpoint if neural.
3. Integration. Order of operations: tokenize → NER → coref → downstream task.
4. Evaluation. CoNLL F1 (MUC + B³ + CEAF-φ4 average) on held-out set + manual cluster review on 20 documents.

Refuse LLM-only coref for documents over 2,000 tokens without sliding-window merge. Refuse any pipeline that runs coref without a mention-level precision-recall report. Flag gender-heuristic systems deployed in demographically diverse text.
```

## 練習 (Exercises)

1. **簡單。** 在 5 段手寫的段落上執行 `code/main.py` 中的 rule-based 解析器。對照標準答案，測量 mention-link 準確率。
2. **中等。** 在一篇新聞上使用預訓練的神經網路 coref 模型。將群集與你自己的人工標註比較。它在哪裡失敗？
3. **困難。** 建立強化 coref 的 NER pipeline：先 NER，再透過 coref 群集合併。在 100 篇文章上，測量相對於僅 NER 的實體涵蓋率改善。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Mention | 一個指涉 | 指涉某個實體的文字 span（名字、代名詞、名詞片語）。 |
| Antecedent | "it" 指的是什麼 | 後面的提及所共指的較早提及。 |
| Cluster | 實體的提及 | 全部指涉同一個真實世界實體的提及集合。 |
| Anaphora | 回指 | 較後的提及指涉較早的（"he" → "John"）。 |
| Cataphora | 後指 | 較早的提及指涉較後的（"When he arrived, John..."）。 |
| Bridging | 隱含指涉 | "I bought a car. The wheels were bad."（是那輛車的輪子。） |
| CoNLL F1 | 排行榜上的數字 | MUC、B³、CEAF-φ4 F1 分數的平均。 |

## 延伸閱讀 (Further Reading)

- [Jurafsky & Martin, SLP3 Ch. 26 — Coreference Resolution and Entity Linking](https://web.stanford.edu/~jurafsky/slp3/26.pdf) — 標準的教科書章節。
- [Lee et al. (2017). End-to-end Neural Coreference Resolution](https://arxiv.org/abs/1707.07045) — span-based 端到端。
- [Joshi et al. (2020). SpanBERT](https://arxiv.org/abs/1907.10529) — 能改善 coref 的預訓練。
- [Pradhan et al. (2012). CoNLL-2012 Shared Task](https://aclanthology.org/W12-4501/) — 這個 benchmark。
- [Hobbs (1978). Resolving Pronoun References](https://www.sciencedirect.com/science/article/pii/0024384178900064) — rule-based 的經典。
