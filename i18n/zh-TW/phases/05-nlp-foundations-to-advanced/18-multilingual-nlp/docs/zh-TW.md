# 多語言 NLP (Multilingual NLP)

> 一個模型，100 多種語言，大多數語言完全沒有訓練資料。Cross-lingual transfer 是 2020 年代實用的奇蹟。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 04 (GloVe, FastText, Subword), Phase 5 · 11 (Machine Translation)
**Time:** ~45 分鐘

## 問題 (The Problem)

英文有數十億筆標註範例，烏爾都語 (Urdu) 只有幾千筆，邁蒂利語 (Maithili) 幾乎沒有。任何服務全球使用者的實用 NLP 系統，都必須能在長尾語言上運作，而這些語言沒有特定任務的訓練資料。

多語言模型的解法，是同時用多種語言訓練同一個模型。共享的表示讓模型把在高資源語言學到的技能，轉移到低資源語言。在英文情感分析上 fine-tune 模型，它在烏爾都語上開箱即用，就能產生出奇地好的情感預測。這就是 zero-shot cross-lingual transfer，它重塑了 NLP 推向全世界的方式。

這堂課會點出取捨、標準模型，以及讓剛接觸多語言工作的團隊栽跟頭的那個決定：為轉移選擇來源語言。

## 核心概念 (The Concept)

![Cross-lingual transfer via shared multilingual embedding space](../assets/multilingual.svg)

**共享詞彙表。** 多語言模型使用在所有目標語言文字上訓練的 SentencePiece 或 WordPiece tokenizer。詞彙表是共享的：同一個 subword 單元，在相近的語言之間代表同一個詞素。英文和義大利文的 `anti-` 得到同一個 token。

**共享表示。** 在多種語言上以 masked language modeling 預訓練的 transformer，學到不同語言中語意相似的句子會產生相似的 hidden state。mBERT、XLM-R 和 NLLB 都有這個現象。英文 "cat" 的 embedding 會靠近法文的 "chat" 和西班牙文的 "gato"，整句的 embedding 也是如此。

**Zero-shot transfer。** 在一種語言（通常是英文）的標註資料上 fine-tune 模型。推論時，在模型支援的任何其他語言上執行。不需要目標語言的標籤。對類型學上相近的語言結果很強，對距離遠的語言則較弱。

**Few-shot fine-tuning。** 加入 100 到 500 筆目標語言的標註範例。在分類任務上，準確率會跳升到英文 baseline 的 95 到 98%。這是多語言 NLP 中成本效益最高的單一槓桿。

## 模型

| 模型 | 年份 | 涵蓋範圍 | 備註 |
|-------|------|----------|-------|
| mBERT | 2018 | 104 種語言 | 在 Wikipedia 上訓練。第一個實用的多語言 LM。對低資源語言較弱。 |
| XLM-R | 2019 | 100 種語言 | 在 CommonCrawl 上訓練（比 Wikipedia 大得多）。立下 cross-lingual 的 baseline。Base 270M，Large 550M。 |
| XLM-V | 2023 | 100 種語言 | 詞彙表有 100 萬個 token 的 XLM-R（原本是 25 萬）。對低資源語言更好。 |
| mT5 | 2020 | 101 種語言 | 用於多語言生成的 T5 架構。 |
| NLLB-200 | 2022 | 200 種語言 | Meta 的翻譯模型；包含 55 種低資源語言。 |
| BLOOM | 2022 | 46 種語言加 13 種程式語言 | 以多語言方式訓練的開放 176B LLM。 |
| Aya-23 | 2024 | 23 種語言 | Cohere 的多語言 LLM。在阿拉伯文、印地文、斯瓦希里文上表現強。 |

依使用情境挑選。分類任務以 XLM-R-base 為合理的預設。生成任務依翻譯或開放生成，選擇 mT5 或 NLLB。LLM 風格的工作則搭配 Aya-23 或 Claude，並使用明確的多語言 prompt。

## 來源語言的決定（2026 年研究）

多數團隊預設以英文作為 fine-tuning 的來源。2026 年的最新研究顯示，這常常是錯的。

語言相似度比原始語料庫大小更能預測轉移品質。對斯拉夫語系的目標，德文或俄文常勝過英文。對印度語系的目標，印地文常勝過英文。**qWALS** 相似度指標（2026，基於 World Atlas of Language Structures 的特徵）將此量化。**LANGRANK**（Lin et al., ACL 2019）是另一個較早的方法，它綜合語言相似度、語料庫大小與親緣關係，為候選來源語言排序。

實用規則：如果你的目標語言有類型學上相近的高資源親屬語言，先試著在那個語言上 fine-tune，再與英文 fine-tune 比較。

```figure
n5-crosslingual-bridge
```

## 動手做 (Build It)

### 步驟 1：zero-shot cross-lingual 分類

```python
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch

tok = AutoTokenizer.from_pretrained("joeddav/xlm-roberta-large-xnli")
model = AutoModelForSequenceClassification.from_pretrained("joeddav/xlm-roberta-large-xnli")


def classify(text, candidate_labels, hypothesis_template="This text is about {}."):
    scores = {}
    for label in candidate_labels:
        hypothesis = hypothesis_template.format(label)
        inputs = tok(text, hypothesis, return_tensors="pt", truncation=True)
        with torch.no_grad():
            logits = model(**inputs).logits[0]
        entail_score = torch.softmax(logits, dim=-1)[2].item()
        scores[label] = entail_score
    return dict(sorted(scores.items(), key=lambda x: -x[1]))


print(classify("I love this product!", ["positive", "negative", "neutral"]))
print(classify("मुझे यह उत्पाद पसंद है!", ["positive", "negative", "neutral"]))
print(classify("J'adore ce produit !", ["positive", "negative", "neutral"]))
```

一個模型，三種語言，同一個 API。在 NLI 資料上訓練的 XLM-R，透過 entailment 技巧能良好地轉移到分類任務。

### 步驟 2：多語言 embedding 空間

```python
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")

pairs = [
    ("The cat is sleeping.", "Le chat dort."),
    ("The cat is sleeping.", "El gato está durmiendo."),
    ("The cat is sleeping.", "Die Katze schläft."),
    ("The cat is sleeping.", "The dog is barking."),
]

for eng, other in pairs:
    emb_eng = model.encode([eng], normalize_embeddings=True)[0]
    emb_other = model.encode([other], normalize_embeddings=True)[0]
    sim = float(np.dot(emb_eng, emb_other))
    print(f"  {eng!r} <-> {other!r}: cos={sim:.3f}")
```

翻譯後的句子在 embedding 空間中靠得很近，不同的英文句子則離得比較遠。這就是 cross-lingual 檢索、分群和相似度之所以可行的原因。

### 步驟 3：few-shot fine-tuning 策略

```python
from transformers import TrainingArguments, Trainer
from datasets import Dataset


def few_shot_finetune(base_model, base_tokenizer, examples):
    ds = Dataset.from_list(examples)

    def tokenize_fn(ex):
        out = base_tokenizer(ex["text"], truncation=True, max_length=128)
        out["labels"] = ex["label"]
        return out

    ds = ds.map(tokenize_fn)
    args = TrainingArguments(
        output_dir="out",
        per_device_train_batch_size=8,
        num_train_epochs=5,
        learning_rate=2e-5,
        save_strategy="no",
    )
    trainer = Trainer(model=base_model, args=args, train_dataset=ds)
    trainer.train()
    return base_model
```

對 100 到 500 筆目標語言範例，`num_train_epochs=5` 和 `learning_rate=2e-5` 是安全的預設值。較高的 learning rate 會讓多語言對齊崩潰，你就得到一個只會英文的模型。

## 真正有效的評估

- **在保留集上的各語言準確率。** 不要彙總。彙總會隱藏長尾。
- **與單語言 baseline 比較。** 對資料夠多的語言，從零訓練的單語言模型有時勝過多語言模型。要測試。
- **實體層級的測試。** 目標語言中的命名實體。多語言模型對與拉丁文字差異大的文字系統，tokenization 常常很弱。
- **Cross-lingual 一致性。** 兩種語言中相同的意思，應該產生相同的預測。測量差距。

## 實際使用 (Use It)

2026 年的技術棧：

| 任務 | 建議 |
|-----|-------------|
| 分類，100 種語言 | fine-tune 過的 XLM-R-base (~270M) |
| Zero-shot 文字分類 | `joeddav/xlm-roberta-large-xnli` |
| 多語言 sentence embedding | `sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2` |
| 翻譯，200 種語言 | `facebook/nllb-200-distilled-600M`（見第 11 課） |
| 多語言生成 | Claude、GPT-4、Aya-23、mT5-XXL |
| 低資源語言 NLP | XLM-V，或在相關高資源語言上做特定領域的 fine-tune |

如果效能重要，一定要預留在目標語言上 fine-tune 的預算。Zero-shot 是起點，不是最終答案。

### Tokenization 稅（低資源語言會出什麼問題）

多語言模型在它所有的語言之間共用一個 tokenizer。那份詞彙表是在以英文、法文、西班牙文、中文、德文為主的語料庫上訓練的。對於主要語言以外的任何語言，三種稅會悄悄地疊加：

- **Fertility 稅。** 低資源語言的文字，每個字被 tokenize 成的 token 遠多於英文。一個印地文句子可能需要等義英文句子 3 到 5 倍的 token。這 3 到 5 倍會吃掉你的 context window、訓練效率和延遲。
- **Variant recovery 稅。** 每個打字錯誤、變音符號變體、Unicode 正規化不一致或大小寫變化，在 embedding 空間中都變成冷啟動、互不相關的序列。模型學不到母語使用者視為理所當然的拼寫對應。
- **Capacity spillover 稅。** 第 1 和第 2 種稅消耗了 context 位置、層深度和 embedding 維度。留給實際推理的部分，系統性地小於高資源語言從同一個模型得到的部分。

實務上的症狀：你的模型在印地文上正常訓練，loss 曲線看起來正確，評估 perplexity 看起來合理，但正式環境的輸出卻有些微錯誤。詞形變化在句子中途崩壞，罕見的屈折變化無法恢復。**你無法靠擴大資料量來彌補壞掉的 tokenizer。**

緩解方式：挑一個對你的目標語言涵蓋良好的 tokenizer（XLM-V 的 100 萬 token 詞彙表是直接的解法）；訓練前在保留的目標文字上驗證 tokenization 的 fertility；對真正長尾的文字系統使用 byte-level fallback（SentencePiece 的 `byte_fallback=True`、GPT-2 風格的 byte-level BPE），讓任何東西都不會是 OOV。

## 交付 (Ship It)

存成 `outputs/skill-multilingual-picker.md`：

```markdown
---
name: multilingual-picker
description: Pick source language, target model, and evaluation plan for a multilingual NLP task.
version: 1.0.0
phase: 5
lesson: 18
tags: [nlp, multilingual, cross-lingual]
---

Given requirements (target languages, task type, available labeled data per language), output:

1. Source language for fine-tuning. Default English; check LANGRANK or qWALS if target language has a typologically close high-resource language.
2. Base model. XLM-R (classification), mT5 (generation), NLLB (translation), Aya-23 (generative LLM).
3. Few-shot budget. Start with 100-500 target-language examples if available. Zero-shot only if labeling is infeasible.
4. Evaluation plan. Per-language accuracy (not aggregate), cross-lingual consistency, entity-level F1 on non-Latin scripts.

Refuse to ship a multilingual model without per-language evaluation — aggregate metrics hide long-tail failures. Flag scripts with low tokenization coverage (Amharic, Tigrinya, many African languages) as needing a model with byte-fallback (SentencePiece with byte_fallback=True, or byte-level tokenizer like GPT-2).
```

## 練習 (Exercises)

1. **簡單。** 對英文、法文、印地文和阿拉伯文，每種語言 10 個句子執行 zero-shot 分類 pipeline。回報各語言的準確率。你應該會看到法文很強、印地文不錯、阿拉伯文不穩定。
2. **中等。** 用 `paraphrase-multilingual-MiniLM-L12-v2` 在一個小型混合語言語料庫上建立 cross-lingual retriever。用英文查詢，檢索任何語言的文件。測量 recall@5。
3. **困難。** 在印地文分類任務上，比較以英文為來源與以印地文為來源的 fine-tuning。在兩種設定下都使用 500 筆目標語言範例做 few-shot fine-tuning。回報哪個來源產生較好的印地文準確率，以及好多少。這就是縮小版的 LANGRANK 論點。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Multilingual model | 一個模型，多種語言 | 跨語言共享詞彙表與參數。 |
| Cross-lingual transfer | 在一種語言訓練，在另一種語言執行 | 在來源語言上 fine-tune，在沒有目標語言標籤的情況下於目標語言評估。 |
| Zero-shot | 沒有目標語言標籤 | 不在目標語言上 fine-tune 的轉移。 |
| Few-shot | 少量目標語言標籤 | 用 100 到 500 筆目標語言範例進行 fine-tuning。 |
| mBERT | 第一個多語言 LM | 在 Wikipedia 上預訓練的 104 種語言 BERT。 |
| XLM-R | 標準的 cross-lingual baseline | 在 CommonCrawl 上預訓練的 100 種語言 RoBERTa。 |
| NLLB | Meta 的 200 種語言 MT | No Language Left Behind。包含 55 種低資源語言。 |

## 延伸閱讀 (Further Reading)

- [Conneau et al. (2019). Unsupervised Cross-lingual Representation Learning at Scale](https://arxiv.org/abs/1911.02116) — XLM-R 論文。
- [Pires, Schlinger, Garrette (2019). How Multilingual is Multilingual BERT?](https://arxiv.org/abs/1906.01502) — 開啟 cross-lingual transfer 研究路線的分析論文。
- [Costa-jussà et al. (2022). No Language Left Behind](https://arxiv.org/abs/2207.04672) — NLLB-200 論文。
- [Üstün et al. (2024). Aya Model: An Instruction Finetuned Open-Access Multilingual Language Model](https://arxiv.org/abs/2402.07827) — Aya，Cohere 的多語言 LLM。
- [Language Similarity Predicts Cross-Lingual Transfer Learning Performance (2026)](https://www.mdpi.com/2504-4990/8/3/65) — qWALS / LANGRANK 來源語言論文。
