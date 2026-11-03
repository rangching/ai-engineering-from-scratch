# 問答系統 (Question Answering Systems)

> 三種系統塑造了現代的 QA。Extractive 找出 span。Retrieval-augmented 把答案紮根在文件上。Generative 產生答案。每個現代 AI 助理都是這三者的混合。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Time:** ~75 分鐘

## 問題 (The Problem)

使用者輸入「When did the first iPhone launch?」，期待得到「June 29, 2007.」。不是「Apple's history is long and varied.」，也不是孤零零一個沒有句子的「2007」。要的是直接、有根據、正確的答案。

過去十年，有三種架構主導了 QA。

- **Extractive QA。** 給定問題和一段已知包含答案的段落，找出答案 span 在段落中的起點和終點索引。SQuAD 是標準 benchmark。
- **Open-domain QA。** 不提供段落。先檢索相關段落，再抽取或生成答案。這是今天每個 RAG pipeline 的基石。
- **Generative / Closed-book QA。** 大型語言模型從它的參數記憶中回答。沒有檢索。推論最快，但在事實上最不可靠。

2026 年的趨勢是混合式：檢索最好的幾個段落，再提示生成式模型根據這些段落來回答。這就是 RAG，第 14 課會深入介紹檢索的那一半。這堂課建立 QA 的那一半。

## 核心概念 (The Concept)

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive。** 用 transformer（BERT 家族）將問題和段落一起編碼。訓練兩個 head，預測答案的起點和終點 token 索引。Loss 是在有效位置上的 cross-entropy。輸出是段落中的一個 span。（依設計）永遠不會產生幻覺，（依設計）也永遠處理不了段落回答不了的問題。

**Retrieval-augmented (RAG)。** 兩個階段。第一，retriever 從語料庫找出前 `k` 個段落。第二，reader（extractive 或 generative）用這些段落產生答案。retriever 與 reader 分開，讓兩者可以各自獨立訓練和評估。現代 RAG 常在兩者之間加上 reranker。

**Generative。** Decoder-only LLM（GPT、Claude、Llama）從學到的權重回答。沒有檢索步驟。對常識很出色，對罕見或最新的事實則是災難。幻覺率與該事實在預訓練資料中的頻率成反比。

```figure
qa-span
```

## 動手做 (Build It)

### 步驟 1：用預訓練模型做 extractive QA

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` 是在 SQuAD 2.0 上訓練的，其中包含無法回答的問題。預設情況下，即使模型的 null 分數勝出，`question-answering` pipeline 仍會回傳分數最高的 span，它*不會*自動回傳空答案。要得到明確的「沒有答案」行為，請在 pipeline 呼叫中傳入 `handle_impossible_answer=True`：這樣 pipeline 只有在 null 分數超過每個 span 的分數時，才會回傳空答案。無論如何，都要檢查 `score` 欄位。

### 步驟 2：Retrieval-augmented pipeline（草圖）

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

兩階段 pipeline。Dense retriever (Sentence-BERT) 依語意相似度找出相關段落。Extractive reader (RoBERTa-SQuAD) 從合併後的前幾名段落中抽出答案 span。適用於小型語料庫。對百萬份文件的語料庫，請使用 FAISS 或向量資料庫。

### 步驟 3：搭配 RAG 的 generative

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt 的模式很重要。明確告訴模型要根據 context 回答，並在 context 不足時回傳 "I don't know"，比起天真的 prompt，能把幻覺率降低 40 到 60%。更精細的模式還會加上引用、信心分數和結構化抽取。

### 步驟 4：反映真實世界的評估

SQuAD 使用 **Exact Match (EM)** 和 **token-level F1**。EM 是正規化（轉小寫、去除標點、移除冠詞）後的嚴格比對，預測要嘛完全相符，要嘛得 0 分。F1 則根據預測與參考答案之間的 token 重疊計算，並給予部分分數。兩者都會低估改寫：「June 29, 2007」對上「June 29th, 2007」通常 EM 為 0（序數詞破壞了正規化），但憑藉重疊的 token 仍能拿到相當高的 F1。

對正式環境的 QA：

- **答案準確率**（由 LLM 或人類判斷，因為指標無法捕捉語意等價）。
- **引用準確率。** 被引用的段落真的支持答案嗎？用生成的引用和檢索到的段落做字串比對，就能輕鬆自動檢查。
- **拒答校準 (Refusal calibration)。** 當答案不在檢索到的段落中，系統是否正確地說「我不知道」？測量虛假自信率。
- **檢索 recall。** 在評估 reader 之前，先測量 retriever 是否把正確的段落放進前 `k` 名。reader 無法補救缺少的段落。

### RAGAS：2026 年正式環境的評估框架

`RAGAS` 是專為 RAG 系統打造的，也是 2026 年的出貨預設。它在不需要標準答案參考的情況下，為四個面向打分：

- **Faithfulness。** 答案中的每個主張都來自檢索到的 context 嗎？以 NLI-based entailment 衡量。是你主要的幻覺指標。
- **Answer relevance。** 答案有回應問題嗎？透過從答案產生假設性問題，再與真實問題比較來衡量。
- **Context precision。** 在檢索到的 chunk 中，有多少比例真的相關？低 precision 代表 prompt 裡有雜訊。
- **Context recall。** 檢索到的集合包含所有需要的資訊嗎？低 recall 代表 reader 不可能成功。

不需參考答案的評分，讓你可以在即時的正式流量上評估，而不用整理標準答案。對於 exact-match 指標派不上用場的開放式問題，再疊加 LLM-as-judge。

`pip install ragas`。接上你的 retriever 加 reader。每個查詢得到四個純量。對退步發出警報。

## 實際使用 (Use It)

2026 年的技術棧。

| 使用情境 | 建議 |
|---------|-------------|
| 給定段落，找出答案 span | `deepset/roberta-base-squad2` |
| 針對固定語料庫，不能接受 closed-book | RAG：dense retriever 加 LLM reader |
| 針對文件庫即時查詢 | 搭配 hybrid (BM25 + dense) retriever 加 reranker 的 RAG（第 14 課） |
| 對話式 QA（後續追問） | 帶有對話歷史的 LLM，每一輪都搭配 RAG |
| 高度事實性、受法規管制的領域 | 在權威語料庫上做 extractive；絕不單獨使用 generative |

Extractive QA 在 2026 年不流行了，因為搭配 LLM 的 RAG 能處理更多情況。但在需要逐字引用的場合，它仍然會出貨：法律研究、法規遵循、稽核工具。

## 交付 (Ship It)

存成 `outputs/skill-qa-architect.md`：

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## 練習 (Exercises)

1. **簡單。** 在 10 段 Wikipedia 段落上，設定上面的 SQuAD extractive pipeline。手寫 10 個問題。測量答案正確的頻率。如果段落和問題都乾淨，你應該會看到 7 到 9 個正確。
2. **中等。** 加入拒答分類器。當最高的檢索分數低於門檻（例如 0.3 cosine）時，回傳 "I don't know"，而不是呼叫 reader。在保留集上調整門檻。
3. **困難。** 在你選擇的 10,000 份文件語料庫上建立 RAG pipeline。實作帶有 RRF fusion 的 hybrid 檢索 (BM25 + dense)（見第 14 課）。測量有無 hybrid 步驟時的答案準確率。記錄哪些問題類型受益最多。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Extractive QA | 找出答案 span | 預測答案在給定段落中的起點和終點索引。 |
| Open-domain QA | 針對語料庫的 QA | 沒有給定段落；必須先檢索再回答。 |
| RAG | 先檢索再生成 | Retrieval-augmented generation。Retriever 加 reader 的 pipeline。 |
| SQuAD | 標準 benchmark | Stanford Question Answering Dataset。使用 EM 與 F1 指標。 |
| Hallucination | 捏造的答案 | reader 的輸出無法被檢索到的 context 支持。 |
| Refusal calibration | 知道何時該閉嘴 | 系統無法回答時，能正確地說「我不知道」。 |

## 延伸閱讀 (Further Reading)

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — benchmark 論文。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，QA 的標準 dense retriever。
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — 命名 RAG 的論文。
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — 完整的 RAG 綜述。
