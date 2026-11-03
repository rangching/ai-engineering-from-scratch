# 資訊檢索與搜尋 (Information Retrieval and Search)

> BM25 精確但脆弱。Dense 撒下大網，卻會漏掉關鍵字。Hybrid 是 2026 年的預設。其他都是調校。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 分鐘

## 問題 (The Problem)

使用者輸入「what happens if someone lies to get money」，期待找到真正涵蓋這件事的法條：「Section 420 IPC.」。關鍵字搜尋完全找不到（沒有共同詞彙）。如果 embedding 沒有在法律文本上訓練過，語意搜尋也找不到。真正的搜尋兩者都要能處理。

IR 是每個 RAG 系統、每個搜尋列、每個文件網站的模糊查詢底下的 pipeline。2026 年在正式環境中有效的架構，不是單一方法，而是一串互補的方法，每一個都接住前一個的失敗。

這堂課會建立每一個環節，並指出各自接住了哪些失敗。

## 核心概念 (The Concept)

![Hybrid retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

四層。挑你需要的。

1. **Sparse retrieval (BM25)。** 快速，對完全比對很精確，對語意很糟。在 inverted index 上執行。對數百萬份文件，每次查詢在 10 毫秒以內。能把法條引用、產品代碼、錯誤訊息、命名實體做對。
2. **Dense retrieval。** 把查詢和文件編碼成向量，做最近鄰搜尋。能捕捉改寫和語意相似度。會漏掉只差一個字元的完全關鍵字比對。使用 FAISS 或向量資料庫，每次查詢 50 到 200 毫秒。
3. **Fusion。** 合併 sparse 和 dense 的排序清單。Reciprocal Rank Fusion (RRF) 是簡單的預設選擇，因為它忽略原始分數（它們的尺度不同），只使用排名位置。當你知道某個訊號在你的領域占主導時，也可以選擇加權融合。
4. **Cross-encoder rerank。** 取 fusion 的前 30 名。執行 cross-encoder（把查詢和文件一起輸入，為每一對打分），保留前 5 名。Cross-encoder 每一對都比 bi-encoder 慢，但準確得多。只對前 30 名執行，就能攤平成本。

三路檢索（BM25 加 dense 加 SPLADE 這類 learned-sparse）在 2026 年的 benchmark 中勝過兩路，但需要 learned-sparse 索引的基礎設施。對多數團隊來說，兩路加 cross-encoder rerank 是最佳平衡點。

```figure
gx-hybrid-retrieval
```

## 動手做 (Build It)

### 步驟 1：從零實作 BM25

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

有兩個值得知道的參數。`k1=1.5` 控制 term-frequency 的飽和度；越高代表越重視詞的重複。`b=0.75` 控制長度正規化；0 忽略文件長度，1 完全正規化。預設值是 Robertson 在原始論文中的建議，很少需要調整。

### 步驟 2：用 bi-encoder 做 dense 檢索

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

對 embedding 做 L2 normalize，讓 dot product 等於 cosine。`all-MiniLM-L6-v2` 是 384 維，速度快，對大多數英文檢索已經夠強。做多語言工作，請使用 `paraphrase-multilingual-MiniLM-L12-v2`。要最高準確率，用 `bge-large-en-v1.5` 或 `e5-large-v2`。

### 步驟 3：Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

`k=60` 這個常數來自原始的 RRF 論文。較高的 `k` 會讓名次差異的貢獻變平緩；較低的 `k` 讓前幾名占主導。60 是已發表的預設值，很少需要調整。

### 步驟 4：hybrid 搜尋加 rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

三個階段組合起來。BM25 找出字面匹配。Dense 找出語意匹配。RRF 合併兩個排序，不需要分數校準。Cross-encoder 把查詢與文件配對後，為前 30 名重新打分，捕捉 bi-encoder 漏掉的細緻相關性。保留前 5 名。

### 步驟 5：評估

| 指標 | 意義 |
|--------|---------|
| Recall@k | 在正確文件存在的查詢中，它出現在前 k 名的頻率有多高？ |
| MRR (Mean Reciprocal Rank) | 第一份相關文件的 1/rank 的平均值。 |
| nDCG@k | 考慮相關性的程度差異，而不只是二元的相關 / 不相關。 |

專就 RAG 而言，retriever 的 **Recall@k** 是最重要的數字。如果正確的段落不在檢索到的集合中，你的 reader 就沒辦法回答。

除錯提示：對失敗的查詢，比對 sparse 與 dense 的排序差異。如果其中一個找到了正確文件而另一個沒有，你遇到的是詞彙不匹配（修法：補上缺少的那一半）或語意歧義（修法：更好的 embedding 或 reranker）。

## 實際使用 (Use It)

2026 年的技術棧：

| 規模 | 技術棧 |
|-------|-------|
| 1k 到 10 萬份文件 | 記憶體內的 BM25 加 `all-MiniLM-L6-v2` embedding 加 RRF。不需要獨立資料庫。 |
| 10 萬到 1,000 萬份文件 | dense 用 FAISS 或 pgvector，BM25 用 Elasticsearch / OpenSearch。平行執行。 |
| 1,000 萬份以上 | 支援 hybrid 的 Qdrant / Weaviate / Vespa / Milvus。在前 30 名上做 cross-encoder rerank。 |
| 最佳品質的前沿 | 三路 (BM25 + dense + SPLADE) 加 ColBERT late-interaction rerank |

不論你選哪一個，都要預留評估的預算。在評估端到端 RAG 準確率之前，先 benchmark 檢索的 recall。reader 無法補救 retriever 漏掉的東西。

### 2026 年正式環境 RAG 辛苦得來的教訓

- **80% 的 RAG 失敗源自資料匯入 (ingestion) 與 chunking，而不是模型。** 團隊花好幾週更換 LLM、調整 prompt，檢索卻默默地每三次查詢就回傳一次錯誤的 context。先修 chunking。
- **Chunking 策略比 chunk 大小更重要。** 固定大小的切分會切壞表格、程式碼和巢狀標題。Sentence-aware 是預設；semantic 或 LLM-based 的 chunking 對技術文件和產品手冊有回報。
- **Parent-doc 模式。** 為了精確度，檢索小的「子」chunk。當同一個父段落的多個子 chunk 都出現時，換成父區塊以保留 context。這能穩定提升答案品質，而且不用重新訓練。
- **k_rerank=3 通常是最佳值。** 超過這個數量的每個額外 chunk，都只會增加 token 成本和生成延遲，卻不會提升答案品質。如果 k=8 對你來說仍然比 k=3 好，代表 reranker 表現不佳。
- **HyDE / 查詢擴展。** 從查詢產生一個假設性答案，對它做 embedding，再檢索。能彌補短問題和長文件之間的措辭落差。不需訓練就能免費提升 precision。
- **Context 預算在 8K token 以內。** 如果持續撞到這個上限，代表 reranker 的門檻太鬆。
- **所有東西都要版本化。** Prompt、chunking 規則、embedding 模型、reranker。任何漂移都會無聲地破壞答案品質。以 faithfulness、context precision 和未回答問題比例作為 CI 關卡，能在使用者看到之前擋下退步。
- **三路檢索（BM25 加 dense 加 SPLADE 這類 learned-sparse）在 2026 年的 benchmark 勝過兩路**，尤其是專有名詞和語意混合的查詢。當基礎設施支援 SPLADE 索引時再出貨。

根據 2026 年的產業量測，適當的檢索設計能減少 70 到 90% 的幻覺。大多數 RAG 的效能提升來自更好的檢索，而不是模型 fine-tuning。

## 交付 (Ship It)

存成 `outputs/skill-retrieval-picker.md`：

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## 練習 (Exercises)

1. **簡單。** 在 500 份文件的語料庫上實作上面的 `hybrid_search`。測試 20 個查詢。比較僅 BM25、僅 dense 與 hybrid 在 recall@5 的差異。
2. **中等。** 加入 MRR 計算。對每個有已知正確文件的測試查詢，找出正確文件在 BM25、dense 和 hybrid 排序中的名次。回報各自的 MRR。
3. **困難。** 使用 MultipleNegativesRankingLoss (Sentence Transformers) 在你的領域上 fine-tune 一個 dense encoder。用 500 組查詢-文件配對建立訓練集。比較 fine-tune 前後的 recall。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| BM25 | 關鍵字搜尋 | Okapi BM25。依 term frequency、IDF 和長度為文件打分。 |
| Dense retrieval | 向量搜尋 | 把查詢與文件編碼成向量，找出最近鄰。 |
| Bi-encoder | Embedding 模型 | 分別獨立地編碼查詢與文件。查詢時很快。 |
| Cross-encoder | Reranker 模型 | 把查詢與文件一起編碼。慢但準確。 |
| RRF | 排名融合 | 加總 `1/(k + rank)` 來合併兩個排序。 |
| Recall@k | 檢索指標 | 相關文件出現在前 k 名的查詢所占的比例。 |

## 延伸閱讀 (Further Reading)

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — BM25 的權威論述。
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR，標準的 bi-encoder。
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — 縮小與 dense 差距的 learned-sparse retriever。
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — RRF 論文。
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — late-interaction 檢索。
