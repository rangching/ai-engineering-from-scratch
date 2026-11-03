# 實體連結與消歧 (Entity Linking & Disambiguation)

> NER 找到了 "Paris"。Entity linking 要決定：是法國的巴黎？Paris Hilton？德州的 Paris？還是 Paris（特洛伊王子）？沒有 linking，你的知識圖譜仍然是模糊的。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 24 (Coreference Resolution)
**Time:** ~60 分鐘

## 問題 (The Problem)

一個句子寫著：「Jordan beat the press.」你的 NER 把 "Jordan" 標成 PERSON。很好。但是*哪一個* Jordan？

- Michael Jordan（籃球）？
- Michael B. Jordan（演員）？
- Michael I. Jordan（Berkeley 的 ML 教授，沒錯，這種混淆在 ML 論文中確實存在）？
- Jordan（約旦這個國家）？
- Jordan（希伯來文的名字）？

Entity linking (EL) 把每個提及解析到知識庫 (knowledge base) 中的唯一項目：Wikidata、Wikipedia、DBpedia，或你的領域 KB。有兩個子任務：

1. **候選生成 (Candidate generation)。** 給定 "Jordan"，哪些 KB 項目是合理的？
2. **消歧 (Disambiguation)。** 給定 context，哪個候選才是對的？

兩個步驟都是可學習的，也都有 benchmark。合併後的 pipeline 十年來都很穩定，變化的是消歧器的品質。

## 核心概念 (The Concept)

![Entity linking pipeline: mention → candidates → disambiguated entity](../assets/entity-linking.svg)

**候選生成。** 給定提及的表面形式 ("Jordan")，在別名索引 (alias index) 中查詢候選。Wikipedia 的別名字典涵蓋了大多數命名實體："JFK" → John F. Kennedy、Jacqueline Kennedy、JFK 機場、JFK（電影）。典型的索引每個提及會回傳 10 到 30 個候選。

**消歧：三種方法。**

1. **Prior 加 context（Milne & Witten, 2008）。** `P(entity | mention) × context-similarity(entity, text)`。效果好、速度快、不需訓練。
2. **以 embedding 為基礎（ESS / REL / Blink）。** 編碼提及加 context。編碼每個候選的描述。挑 cosine 最大的。2020 到 2024 年的預設。
3. **生成式（GENRE, 2021；以 LLM 為基礎，2023 起）。** 逐 token 解碼實體的標準名稱。限制在有效實體名稱的 trie 上，所以輸出保證是有效的 KB id。

**端到端 vs pipeline。** 現代模型（ELQ、BLINK、ExtEnD、GENRE）在一次運算中完成 NER、候選生成與消歧。Pipeline 系統在正式環境中仍占主導，因為你可以替換元件。

### 兩種量測

- **Mention recall（候選生成）。** 標準答案提及中，正確的 KB 項目出現在候選清單內的比例。是整個 pipeline 的下限。
- **消歧準確率 / F1。** 在候選正確的情況下，top-1 正確的頻率。

兩者都要回報。一個系統在 80% 的候選 recall 下有 99% 的消歧準確率，那就是一個 80% 的 pipeline。

```figure
gx-entity-linking
```

## 動手做 (Build It)

### 步驟 1：從 Wikipedia redirect 建立別名索引

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Wikipedia 別名資料：約 1,800 萬組（別名, 實體）配對。從 Wikidata dump 下載。儲存成 inverted index。

### 步驟 2：以 context 為基礎的消歧

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard 重疊只是玩具。請換成 embedding 上的 cosine similarity（transformer 版本見 `code/main.py` 的 step-2）。

### 步驟 3：以 embedding 為基礎（BLINK 風格）

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

建立索引時，對每個 KB 實體做一次 embedding。查詢時，對提及加 context 做一次 embedding，與候選池做 dot product，挑最大的。

### 步驟 4：生成式 entity linking（概念）

GENRE 逐字元解碼實體的 Wikipedia 標題。Constrained decoding（見第 20 課）確保只能輸出有效的標題。與以 KB 為後盾的 trie 緊密整合。現代的後代是 REL-GEN，以及搭配結構化輸出、由 LLM 提示的 EL。

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

搭配白名單（Outlines 的 `choice`），這是 2026 年最容易出貨的 EL pipeline。

### 步驟 5：在 AIDA-CoNLL 上評估

AIDA-CoNLL 是標準的 EL benchmark：1,393 篇 Reuters 文章、3.4 萬個提及、Wikipedia 實體。回報 KB 內準確率 (`P@1`)，以及 KB 外的 NIL 偵測率。

## 陷阱

- **NIL 處理。** 有些提及不在 KB 中（新興實體、冷門人物）。系統必須預測 NIL，而不是猜一個錯的實體。要分開量測。
- **提及邊界錯誤。** 上游 NER 漏掉部分 span（"Bank of America" 只被標成 "Bank"），EL 的 recall 就會下降。
- **熱門度偏誤。** 訓練出來的系統會過度預測頻繁出現的實體。ML 論文中的 "Michael I. Jordan" 提及，常常被連到籃球的 Jordan。
- **跨語言 EL。** 把中文文字中的提及對應到英文 Wikipedia 實體。需要多語言 encoder 或一個翻譯步驟。
- **KB 過時。** 新公司、事件、人物不在去年的 Wikipedia dump 裡。正式 pipeline 需要更新迴圈。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| 通用英文加 Wikipedia | BLINK 或 REL |
| 跨語言，KB = Wikipedia | mGENRE |
| 對 LLM 友善，每天少量提及 | 用候選清單加受限 JSON 提示 Claude/GPT-4 |
| 特定領域 KB（醫療、法律） | 具 KB 感知檢索的自訂 BERT，並在領域的 AIDA 風格資料集上 fine-tune |
| 極低延遲 | 僅用完全比對的 prior（Milne-Witten baseline） |
| 研究 SOTA | GENRE / ExtEnD / 生成式 LLM-EL |

2026 年實際出貨的正式模式：NER → coref → 對每個提及做 EL → 把群集收斂成每個群集一個標準實體。輸出：文件中每個實體一個 KB id，而不是每個提及一個。

## 交付 (Ship It)

存成 `outputs/skill-entity-linker.md`：

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## 練習 (Exercises)

1. **簡單。** 在 10 個有歧義的提及（Paris、Jordan、Apple）上實作 `code/main.py` 中的 prior 加 context 消歧器。手動標註正確的實體。測量準確率。
2. **中等。** 用 sentence transformer 編碼 50 個有歧義的提及。對每個候選的描述做 embedding。比較以 embedding 為基礎的消歧與 Jaccard context 重疊。
3. **困難。** 建立一個 1k 實體的領域 KB（例如你公司的員工加產品）。端到端實作 NER 加 EL。在 100 個保留句子上測量 precision 和 recall。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Entity linking (EL) | 連到 Wikipedia | 把提及對應到唯一的 KB 項目。 |
| Candidate generation | 可能是誰？ | 為提及回傳合理 KB 項目的候選短清單。 |
| Disambiguation | 挑對的那個 | 用 context 為候選打分，挑出贏家。 |
| Alias index | 查詢表 | 從表面形式 → 候選實體的對應。 |
| NIL | 不在 KB 中 | 明確預測沒有任何 KB 項目符合。 |
| KB | 知識庫 | Wikidata、Wikipedia、DBpedia，或你的領域 KB。 |
| AIDA-CoNLL | 那個 benchmark | 1,393 篇附有標準實體連結的 Reuters 文章。 |

## 延伸閱讀 (Further Reading)

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://researchcommons.waikato.ac.nz/entities/publication/b9a0b520-abc5-47c5-a86a-da6c579893ab) — 奠基性的 prior 加 context 方法。
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — 以 embedding 為基礎的主力。
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — 搭配 constrained decoding 的生成式 EL。
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — benchmark 論文。
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — 開放的正式環境技術棧。
