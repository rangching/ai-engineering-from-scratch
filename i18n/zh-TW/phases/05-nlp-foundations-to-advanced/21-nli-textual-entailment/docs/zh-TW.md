# 自然語言推論 (Natural Language Inference) — Textual Entailment

> 「t entails h」的意思是，讀了 t 的人會推斷 h 為真。NLI 的任務就是預測 entailment / contradiction / neutral。表面上很無聊，在正式環境中卻是承重的關鍵。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment Analysis), Phase 5 · 13 (Question Answering)
**Time:** ~60 分鐘

## 問題 (The Problem)

你建了一個摘要器，它產生了摘要。你怎麼知道摘要裡沒有幻覺？

你建了一個聊天機器人，它回答「yes」。你怎麼知道這個答案有被檢索到的段落支持？

你需要把 1 萬篇新聞依主題分類，卻沒有訓練標籤。你能重複利用某個模型嗎？

這三個問題都可以歸結為自然語言推論 (Natural Language Inference)。NLI 問的是：給定前提 `t` 和假設 `h`，`h` 是被 `t` 蘊含 (entailed)、被反駁 (contradicted)，還是中性 (neutral，無關)？

- **幻覺檢查：** `t` = 來源文件，`h` = 摘要主張。不是 entailment 就是幻覺。
- **有根據的 QA：** `t` = 檢索到的段落，`h` = 生成的答案。不是 entailment 就是捏造。
- **Zero-shot 分類：** `t` = 文件，`h` = 把標籤口語化（"This is about sports"）。Entailment 就是預測的標籤。

一個任務，三種正式用途。這就是為什麼每個 RAG 評估框架底層都附帶一個 NLI 模型。

## 核心概念 (The Concept)

![NLI: three-way classification, premise vs hypothesis](../assets/nli.svg)

**三個標籤。**

- **Entailment。** `t` → `h`。"The cat is on the mat" 蘊含 "There is a cat."
- **Contradiction。** `t` → ¬`h`。"The cat is on the mat" 與 "There is no cat." 矛盾。
- **Neutral。** 兩個方向都推不出什麼。"The cat is on the mat" 對 "The cat is hungry." 是中性的。

**不是邏輯上的 entailment。** NLI 是*自然*語言推論，是一般人類讀者會推斷的內容，而不是嚴格的邏輯。在 NLI 中 "John walked his dog" 蘊含 "John has a dog"，但嚴格的一階邏輯只有在你把「擁有」公理化之後，才會接受它。

**資料集。**

- **SNLI**（2015）。57 萬組人工標註的配對，以圖片說明作為前提。領域狹窄。
- **MultiNLI**（2017）。43.3 萬組配對，涵蓋 10 種文體。2026 年標準的訓練語料庫。
- **ANLI**（2019）。Adversarial NLI。人類專門寫出設計來打破現有模型的範例。更難。
- **DocNLI, ConTRoL**（2020 到 2021）。文件長度的前提。測試多跳和長距離推論。

**架構。** Transformer encoder（BERT、RoBERTa、DeBERTa）讀入 `[CLS] premise [SEP] hypothesis [SEP]`。`[CLS]` 的表示送進 3 類 softmax。在 MNLI 上訓練，在保留的 benchmark 上評估，對分布內的配對可得到 90% 以上的準確率。

**透過 NLI 做 zero-shot。** 給定一份文件和候選標籤，把每個標籤變成假設（"This text is about sports"）。為每個計算 entailment 機率，挑最大的。這就是 Hugging Face 的 `zero-shot-classification` pipeline 背後的機制。

```figure
nli-router
```

## 動手做 (Build It)

### 步驟 1：執行預訓練的 NLI 模型

```python
from transformers import pipeline

nli = pipeline("text-classification",
               model="facebook/bart-large-mnli",
               top_k=None)  # return all labels; replaces deprecated return_all_scores=True

premise = "The cat is sleeping on the couch."
hypothesis = "There is a cat in the room."

result = nli({"text": premise, "text_pair": hypothesis})[0]
print(result)
# [{'label': 'entailment', 'score': 0.97},
#  {'label': 'neutral', 'score': 0.02},
#  {'label': 'contradiction', 'score': 0.01}]
```

對正式環境的 NLI，`facebook/bart-large-mnli` 和 `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` 是開放的預設選擇。DeBERTa-v3 稱霸排行榜。

### 步驟 2：zero-shot 分類

```python
zs = pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

text = "The stock market rallied after the central bank cut interest rates."
labels = ["finance", "sports", "politics", "technology"]

result = zs(text, candidate_labels=labels)
print(result)
# {'labels': ['finance', 'politics', 'technology', 'sports'],
#  'scores': [0.92, 0.05, 0.02, 0.01]}
```

模板預設是 "This example is about {label}."。可用 `hypothesis_template` 自訂。不需要訓練資料，不需要 fine-tuning，開箱即用。

### 步驟 3：RAG 的 faithfulness 檢查

```python
def is_faithful(answer, context, threshold=0.5):
    result = nli({"text": context, "text_pair": answer})[0]
    entail = next(s for s in result if s["label"] == "entailment")
    return entail["score"] > threshold
```

這是 RAGAS faithfulness 的核心。把生成的答案拆成原子主張 (atomic claims)。對照檢索到的 context 檢查每個主張。回報被蘊含的比例。

### 步驟 4：手寫的 NLI 分類器（概念性）

標準函式庫版本的玩具實作見 `code/main.py`：前提和假設透過詞彙重疊加否定偵測來比較。與 transformer 模型沒得比，但它展示了這個任務的形狀：輸入兩段文字，輸出 3 類標籤，loss = 在 `{entail, contradict, neutral}` 上的 cross-entropy。

## 陷阱

- **只看假設的捷徑。** 模型光從假設就能在 SNLI 上以約 60% 預測標籤，因為 "not"、"nobody"、"never" 與 contradiction 相關。這是偵測標籤洩漏的強力 baseline。
- **詞彙重疊啟發法。** 子序列啟發法（「每個子序列都被蘊含」）能通過 SNLI，卻在 HANS/ANLI 失敗。請使用對抗式 benchmark。
- **文件長度的退化。** 單句 NLI 模型在文件長度的前提上，F1 會掉 20 以上。長 context 請使用在 DocNLI 上訓練的模型。
- **Zero-shot 模板敏感度。** "This example is about {label}" vs "{label}" vs "The topic is {label}" 可以讓準確率擺盪 10 分以上。請調整模板。
- **領域不匹配。** MNLI 是用一般英文訓練的。法律、醫療和科學文本需要特定領域的 NLI 模型（例如 SciNLI、MedNLI）。

## 實際使用 (Use It)

2026 年的技術棧：

| 使用情境 | 模型 |
|---------|-------|
| 通用 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 快速 / 邊緣 | `cross-encoder/nli-deberta-v3-base` |
| Zero-shot 分類（輕量） | `facebook/bart-large-mnli` |
| 文件層級 NLI | `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` |
| 多語言 | `MoritzLaurer/multilingual-MiniLMv2-L6-mnli-xnli` |
| RAG 中的幻覺偵測 | RAGAS / DeepEval 內的 NLI 層 |

2026 年的元模式：NLI 是文本理解的萬用膠帶。每當你需要「A 支持 B 嗎？」或「A 與 B 矛盾嗎？」，先想到 NLI，再去想另一次 LLM 呼叫。

## 交付 (Ship It)

存成 `outputs/skill-nli-picker.md`：

```markdown
---
name: nli-picker
description: Pick an NLI model, label template, and evaluation setup for a classification / faithfulness / zero-shot task.
version: 1.0.0
phase: 5
lesson: 21
tags: [nlp, nli, zero-shot]
---

Given a use case (faithfulness check, zero-shot classification, document-level inference), output:

1. Model. Named NLI checkpoint. Reason tied to domain, length, language.
2. Template (if zero-shot). Verbalization pattern. Example.
3. Threshold. Entailment cutoff for the decision rule. Reason based on calibration.
4. Evaluation. Accuracy on held-out labeled set, hypothesis-only baseline, adversarial subset.

Refuse to ship zero-shot classification without a 100-example labeled sanity check. Refuse to use a sentence-level NLI model on document-length premises. Flag any claim that NLI solves hallucination — it reduces it; it does not eliminate it.
```

## 練習 (Exercises)

1. **簡單。** 對涵蓋三個類別的 20 組手寫 (premise, hypothesis, label) 三元組執行 `facebook/bart-large-mnli`。測量準確率。加入對抗式的「子序列啟發法」陷阱（"I did not eat the cake" vs "I ate the cake"），看它會不會出錯。
2. **中等。** 在 100 則 AG News 標題上，比較 zero-shot 模板 `"This text is about {label}"`、`"The topic is {label}"` 與 `"{label}"`。回報準確率的擺盪幅度。
3. **困難。** 建立 RAG faithfulness 檢查器：原子主張拆解加上每個主張的 NLI。在 50 個附有標準 context 的 RAG 生成答案上評估。對照人工標籤，測量 false-positive 與 false-negative 率。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| NLI | Natural Language Inference | 前提與假設關係的 3 類分類。 |
| RTE | Recognizing Textual Entailment | NLI 的舊名；同一個任務。 |
| Entailment | 「t 蘊含 h」 | 一般讀者在給定 t 的情況下，會推斷 h 為真。 |
| Contradiction | 「t 排除 h」 | 一般讀者在給定 t 的情況下，會推斷 h 為假。 |
| Neutral | 「無法決定」 | 從 t 到 h 兩個方向都推不出什麼。 |
| Zero-shot classification | 把 NLI 當分類器 | 把標籤口語化成假設，挑 entailment 最大的。 |
| Faithfulness | 答案有被支持嗎？ | 對（檢索到的 context，生成的答案）做 NLI。 |

## 延伸閱讀 (Further Reading)

- [Bowman et al. (2015). A large annotated corpus for learning natural language inference](https://arxiv.org/abs/1508.05326) — SNLI。
- [Williams, Nangia, Bowman (2017). A Broad-Coverage Challenge Corpus for Sentence Understanding through Inference](https://arxiv.org/abs/1704.05426) — MultiNLI。
- [Nie et al. (2019). Adversarial NLI](https://arxiv.org/abs/1910.14599) — ANLI benchmark。
- [Yin, Hay, Roth (2019). Benchmarking Zero-shot Text Classification](https://arxiv.org/abs/1909.00161) — 把 NLI 當分類器。
- [He et al. (2021). DeBERTa: Decoding-enhanced BERT with Disentangled Attention](https://arxiv.org/abs/2006.03654) — 2026 年 NLI 的主力。
