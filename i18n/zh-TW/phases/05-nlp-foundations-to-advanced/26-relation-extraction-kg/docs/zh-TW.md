# 關係抽取與知識圖譜建構 (Relation Extraction & Knowledge Graph Construction)

> NER 找到了實體。Entity linking 把它們錨定。Relation extraction 找出它們之間的邊。知識圖譜就是節點、邊，以及它們來源出處 (provenance) 的總和。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 06 (NER), Phase 5 · 25 (Entity Linking)
**Time:** ~60 分鐘

## 問題 (The Problem)

分析師讀到：「Tim Cook became CEO of Apple in 2011.」這裡有四個事實：

- `(Tim Cook, role, CEO)`
- `(Tim Cook, employer, Apple)`
- `(Tim Cook, start_date, 2011)`
- `(Apple, type, Organization)`

Relation Extraction (RE) 把自由文字變成結構化的三元組 `(subject, relation, object)`。在整個語料庫上彙整，你就有了知識圖譜。彙整後再查詢，你就有了 RAG、分析或合規稽核的推理基底。

2026 年的問題：LLM 抽取關係非常熱心，太熱心了。它們會幻覺出來源文字不支持的三元組。沒有出處，你就分不出真正的三元組和看似合理的虛構。2026 年的答案是 AEVS 風格的 anchor-and-verify pipeline。

## 核心概念 (The Concept)

![Text → triples → knowledge graph](../assets/relation-extraction.svg)

**三元組的形式。** `(subject_entity, relation_type, object_entity)`。關係來自封閉的 ontology（Wikidata properties、FIBO、UMLS）或開放集合（OpenIE 風格，什麼都行）。

**三種抽取方法。**

1. **Rule / pattern-based。** Hearst pattern："X such as Y" → `(Y, isA, X)`。再加上手工撰寫的 regex。脆弱、精確、可解釋。
2. **Supervised 分類器。** 給定句子中的兩個實體提及，從固定集合中預測關係。在 TACRED、ACE、KBP 上訓練。2015 到 2022 年的標準做法。
3. **Generative LLM。** 提示模型輸出三元組。開箱即用。需要出處，否則會幻覺出看似合理的垃圾。

**AEVS（Anchor-Extraction-Verification-Supplement，2026）。** 目前的幻覺緩解框架：

- **Anchor。** 以精確位置辨識每個實體 span 和關係片語 span。
- **Extract。** 產生連結到 anchor span 的三元組。
- **Verify。** 將每個三元組元素比對回來源文字；拒絕任何沒有支持的內容。
- **Supplement。** 做一次涵蓋率檢查，確保沒有任何 anchor 的 span 被丟掉。

幻覺會大幅減少。需要更多運算，但可稽核。

**開放 vs 封閉的取捨。**

- **封閉 ontology。** 固定的屬性清單（例如 Wikidata 的 11,000 多個屬性）。可預測、可查詢，但難以新創。
- **Open IE。** 任何動詞片語都變成關係。recall 高，precision 低，查詢時很亂。

正式的 KG 通常混合使用：用 open IE 做探索，再先把關係正規化 (canonicalize) 到封閉 ontology 上，才合併進主圖。

```figure
relation-triples
```

## 動手做 (Build It)

### 步驟 1：以 pattern 為基礎的抽取

```python
PATTERNS = [
    (r"(?P<s>[A-Z]\w+) (?:is|was) (?:a|an|the) (?P<o>[A-Z]?\w+)", "isA"),
    (r"(?P<s>[A-Z]\w+) (?:is|was) born in (?P<o>\w+)", "bornIn"),
    (r"(?P<s>[A-Z]\w+) works? (?:at|for) (?P<o>[A-Z]\w+)", "worksAt"),
    (r"(?P<s>[A-Z]\w+) founded (?P<o>[A-Z]\w+)", "founded"),
]
```

完整的玩具抽取器見 `code/main.py`。Hearst pattern 仍會出現在特定領域的 pipeline 中，因為它們好除錯。

### 步驟 2：supervised seq2seq 關係抽取

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("Babelscape/rebel-large")
model = AutoModelForSeq2SeqLM.from_pretrained("Babelscape/rebel-large")

text = "Tim Cook was born in Alabama. He later became CEO of Apple."
encoded = tok(text, return_tensors="pt", truncation=True)
output = model.generate(**encoded, max_length=200)
decoded = tok.batch_decode(output, skip_special_tokens=False)
```

REBEL 是以 BART 為基礎的 seq2seq 關係抽取器：它產生三元組，而不是對實體配對做分類。輸入文字，輸出三元組，關係名稱取自 Wikidata 的屬性。解碼出來的字串仍帶有 REBEL 的 `<triplet>`、`<subj>` 和 `<obj>` 標記，所以要依它們切開（如 model card 的 parser 所做），才能得到 `(subject, relation, object)` 記錄。在 distant-supervision 資料上 fine-tune。標準的 open-weights baseline。

### 步驟 3：搭配 anchoring 的 LLM 提示抽取

```python
prompt = f"""Extract (subject, relation, object) triples from the text.
For each triple, include the exact character span in the source text.

Text: {text}

Output JSON:
[{{"subject": {{"text": "...", "span": [start, end]}},
   "relation": "...",
   "object": {{"text": "...", "span": [start, end]}}}}, ...]

Only include triples fully supported by the text. No inference beyond what is stated.
"""
```

對照來源驗證每個回傳的 span。拒絕任何 `text[start:end] != triple_entity` 的結果。這是 AEVS「verify」步驟的最小形式。

### 步驟 4：正規化到封閉 ontology

```python
RELATION_MAP = {
    "is the CEO of": "P169",       # "chief executive officer"
    "was born in":   "P19",         # "place of birth"
    "founded":        "P112",       # "founded by" (inverted subject/object)
    "works at":       "P108",       # "employer"
}


def canonicalize(relation):
    rel_low = relation.lower().strip()
    if rel_low in RELATION_MAP:
        return RELATION_MAP[rel_low]
    return None   # drop unmapped open relations or route to manual review
```

正規化常常占了 60 到 80% 的工程工作。要預留預算。

### 步驟 5：建立小型圖並查詢

```python
triples = extract(text)
graph = {}
for s, r, o in triples:
    graph.setdefault(s, []).append((r, o))


def neighbors(node, relation=None):
    return [(r, o) for r, o in graph.get(node, []) if relation is None or r == relation]


print(neighbors("Tim Cook", relation="P108"))    # -> [(P108, Apple)]
```

這是每個 RAG-over-KG 系統的原子。可用 RDF triple store（Blazegraph、Virtuoso）、property graph（Neo4j），或向量增強的圖儲存來擴展。

## 陷阱

- **RE 之前先做 coreference。** "He founded Apple"，RE 需要知道 "he" 是誰。先跑 coref（第 24 課）。
- **實體正規化。** "Apple Inc" 和 "Apple" 必須解析到同一個節點。先做 entity linking（第 25 課）。
- **幻覺的三元組。** LLM 輸出文字不支持的三元組。要強制 span 驗證。
- **關係正規化漂移。** Open IE 的關係不一致（"was born in"、"came from"、"is a native of"）。要收斂成標準 id，否則圖無法查詢。
- **時間錯誤。** "Tim Cook is CEO of Apple"，現在是真的，2005 年是假的。許多關係是有時間界線的。請使用 qualifier（Wikidata 中的 `P580` 起始時間、`P582` 結束時間）。
- **領域不匹配。** REBEL 是在 Wikipedia 上訓練的。法律、醫療和科學文本常常需要特定領域 fine-tune 的 RE 模型。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| 快速的正式環境、通用領域 | 搭配 Wikidata 正規化的 REBEL 或 LlamaPred |
| 特定領域（生醫、法律） | SciREX 風格的領域 fine-tune 加自訂 ontology |
| LLM 提示、輸出可稽核 | AEVS pipeline：anchor → extract → verify → supplement |
| 大量新聞 IE | Pattern-based 加 supervised 的混合 |
| 從零建立 KG | Open IE 加人工正規化這一輪 |
| 時間性 KG | 帶 qualifier（起迄時間、時間點）抽取 |

整合模式：NER → coref → entity linking → relation extraction → ontology mapping → 載入圖。每個階段都是潛在的品質關卡。

## 交付 (Ship It)

存成 `outputs/skill-re-designer.md`：

```markdown
---
name: re-designer
description: Design a relation extraction pipeline with provenance and canonicalization.
version: 1.0.0
phase: 5
lesson: 26
tags: [nlp, relation-extraction, knowledge-graph]
---

Given a corpus (domain, language, volume) and downstream use (KG-RAG, analytics, compliance), output:

1. Extractor. Pattern-based / supervised / LLM / AEVS hybrid. Reason tied to precision vs recall target.
2. Ontology. Closed property list (Wikidata / domain) or open IE with canonicalization pass.
3. Provenance. Every triple carries source char-span + doc id. Non-negotiable for audit.
4. Merge strategy. Canonical entity id + relation id + temporal qualifiers; dedup policy.
5. Evaluation. Precision / recall on 200 hand-labelled triples + hallucination-rate on LLM-extracted sample.

Refuse any LLM-based RE pipeline without span verification (source provenance). Refuse open-IE output flowing into a production graph without canonicalization. Flag pipelines with no temporal qualifier on time-bounded relations (employer, spouse, position).
```

## 練習 (Exercises)

1. **簡單。** 在 5 個新聞文章句子上執行 `code/main.py` 中的 pattern 抽取器。人工檢查 precision。
2. **中等。** 對同樣的句子使用 REBEL（或小型 LLM）。比較三元組。哪個抽取器的 precision 較高？recall 較高？
3. **困難。** 建立 AEVS pipeline：用 LLM 抽取，再對照來源驗證 span。在 50 個 Wikipedia 風格的句子上，測量 verify 步驟前後的幻覺率。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Triple | 主詞-關係-受詞 | `(s, r, o)` 元組，是 KG 的原子單位。 |
| Open IE | 什麼都抽 | 開放詞彙的關係片語；recall 高，precision 低。 |
| Closed ontology | 固定的 schema | 有界的關係類型集合（Wikidata、UMLS、FIBO）。 |
| Canonicalization | 全部正規化 | 把表面名稱 / 關係對應到標準 id。 |
| AEVS | 有根據的抽取 | Anchor-Extraction-Verification-Supplement pipeline（2026）。 |
| Provenance | 真實來源的連結 | 每個三元組都帶有指向來源的文件 id 加字元 span。 |
| Distant supervision | 便宜的標籤 | 把文字與現有 KG 對齊，以建立訓練資料。 |

## 延伸閱讀 (Further Reading)

- [Mintz et al. (2009). Distant supervision for relation extraction without labeled data](https://www.aclweb.org/anthology/P09-1113.pdf) — distant-supervision 論文。
- [Huguet Cabot, Navigli (2021). REBEL: Relation Extraction By End-to-end Language generation](https://aclanthology.org/2021.findings-emnlp.204.pdf) — seq2seq RE 的主力。
- [Wadden et al. (2019). Entity, Relation, and Event Extraction with Contextualized Span Representations (DyGIE++)](https://arxiv.org/abs/1909.03546) — 聯合 IE。
- [AEVS — Anchor-Extraction-Verification-Supplement framework](https://www.mdpi.com/2073-431X/15/3/178) — 2026 年的幻覺緩解設計。
- [Wikidata SPARQL tutorial](https://www.wikidata.org/wiki/Wikidata:SPARQL_tutorial) — 標準的圖查詢。
