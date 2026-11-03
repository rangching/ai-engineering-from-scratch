# Embedding 模型 (Embedding Models) — 2026 深入探討

> Word2Vec 給你每個字一個向量。現代 embedding 模型給你每個段落一個向量，可跨語言，有 sparse、dense 和 multi-vector 的視角，大小可配合你的索引。選錯了，你的 RAG 就會檢索到錯的東西。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 03 (Word2Vec), Phase 5 · 14 (Information Retrieval)
**Time:** ~60 分鐘

## 問題 (The Problem)

你的 RAG 系統有 40% 的時候檢索到錯的段落。元凶很少是向量資料庫或 prompt，而是 embedding 模型。

2026 年選擇 embedding，意味著要在五個軸上做取捨：

1. **Dense vs sparse vs multi-vector。** 每個段落一個向量、每個 token 一個向量，或是 sparse 的加權詞袋。
2. **語言涵蓋範圍。** 單語言的英文模型在純英文任務上仍然勝出。語料庫混合語言時，多語言模型勝出。
3. **Context 長度。** 512 token vs 8,192 vs 32,768，而真正有效的容量通常只有宣稱最大值的 60 到 70%。
4. **維度預算。** 3,072 個全精度 float = 每個向量 12 KB。在 1 億個向量時，儲存成本是每月 1,300 美元。Matryoshka 截斷可以把它降低 4 倍。
5. **開放 vs 託管。** Open-weight 代表你掌控技術棧和資料。託管代表你用控制權換取永遠最新。

這堂課會點出取捨，讓你能憑證據做選擇，而不是憑上一季流行什麼。

## 核心概念 (The Concept)

![Dense, sparse, and multi-vector embeddings](../assets/embedding-modes.svg)

**Dense embedding。** 每個段落一個向量（通常 384 到 3,072 維）。用 cosine similarity 依語意接近度為段落排序。OpenAI `text-embedding-3-large`、BGE-M3 的 dense 模式、Voyage-3。預設選擇。

**Sparse embedding。** SPLADE 風格。Transformer 為每個詞彙表 token 預測一個權重，再把大部分歸零。結果是大小為 |vocab| 的 sparse 向量。捕捉詞彙匹配（像 BM25），但具有學習而來的詞權重。對關鍵字密集的查詢很強。

**Multi-vector（late interaction）。** ColBERTv2、Jina-ColBERT。每個 token 一個向量。用 MaxSim 計分：對每個查詢 token，找到最相似的文件 token，再把分數加總。儲存和計分的成本較高，但在長查詢和特定領域的語料庫上勝出。

**BGE-M3：三合一。** 單一模型同時輸出 dense、sparse 和 multi-vector 表示。每一種都可以獨立查詢；分數透過加權總和融合。當你想從單一 checkpoint 得到彈性時，這是 2026 年的預設。

**Matryoshka Representation Learning。** 訓練方式讓向量的前 N 個維度本身就能構成有用的獨立 embedding。把 1,536 維向量截斷到 256 維，只損失約 1% 準確率，卻省下 6 倍儲存空間。OpenAI text-3、Cohere v4、Voyage-4、Jina v5、Gemini Embedding 2、Nomic v1.5 以上都支援。

### MTEB 排行榜只說出一部分的故事

Massive Text Embedding Benchmark，發布時（2022）在 8 種任務類型上有 56 個任務，在 MTEB v2 擴展到 100 多個任務。2026 年初，Gemini Embedding 2 在檢索上居首（67.71 MTEB-R）。Cohere embed-v4 在整體上領先（65.2 MTEB）。BGE-M3 在 open-weight 多語言上領先（63.0）。排行榜是必要的，但不充分，永遠要在你自己的領域上做 benchmark。

### 三層模式

| 使用情境 | 模式 |
|----------|---------|
| 快速的第一輪 | Dense bi-encoder（BGE-M3、text-3-small） |
| 提升 recall | Sparse（SPLADE、BGE-M3 sparse）加 RRF 融合 |
| 前 50 名的 precision | Multi-vector（ColBERTv2）或 cross-encoder reranker |

多數正式的技術棧三者都用。

```figure
gx-matryoshka
```

## 動手做 (Build It)

### 步驟 1：baseline，用 Sentence-BERT 做 dense embedding

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` 讓 dot product 等於 cosine similarity。永遠要設定它。

### 步驟 2：Matryoshka 截斷

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

截斷後要重新 normalize。Nomic v1.5、OpenAI text-3 和 Voyage-4 的訓練方式，讓前幾個層級的截斷是無損的。非 Matryoshka 的模型（原始的 Sentence-BERT）在截斷時會急遽退化。

### 步驟 3：BGE-M3 的多功能性

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

三個索引，一次推論呼叫。分數融合：

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

在你的領域上調整權重。

### 步驟 4：在自訂任務上跑 MTEB 評估

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

在*具代表性*的子集上執行你的候選模型。不要只相信排行榜名次，你的領域很重要。

### 步驟 5：從零手寫 cosine

見 `code/main.py`。平均化的 Hashing Trick embedding（僅用標準函式庫）。與 transformer embedding 沒得比，但展示了形狀：tokenize → 向量 → normalize → dot product。

## 陷阱

- **查詢和文件用同一個模型。** 有些模型（Voyage、Jina-ColBERT）使用非對稱編碼，查詢和文件走不同的路徑。永遠要檢查 model card。
- **缺少前綴。** `bge-*` 模型需要在查詢前面加上 `"Represent this sentence for searching relevant passages: "`。忘記的話，recall 會差 3 到 5 分。
- **過度截斷 Matryoshka。** 1,536 → 256 通常安全。1,536 → 64 則不然。請在你的評估集上驗證。
- **Context 截斷。** 多數模型會無聲地截斷超過最大長度的輸入。長文件需要 chunking（見第 23 課）。
- **忽略延遲的長尾。** MTEB 分數隱藏了 p99 延遲。6 億參數的模型可能比 3.35 億的模型高 2 分，每次查詢的成本卻是 3 倍。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| 僅英文、快速、API | `text-embedding-3-large` 或 `voyage-3-large` |
| Open-weight、英文 | `BAAI/bge-large-en-v1.5` |
| Open-weight、多語言 | `BAAI/bge-m3` 或 `Qwen3-Embedding-8B` |
| 長 context（32k 以上） | Voyage-3-large、Cohere embed-v4、Qwen3-Embedding-8B |
| 僅 CPU 的部署 | Nomic Embed v2（1.37 億參數，MoE） |
| 儲存受限 | Matryoshka 截斷加 int8 量化 |
| 關鍵字密集的查詢 | 加入 SPLADE sparse，與 dense 做 RRF 融合 |

2026 年的模式：從 BGE-M3 或 text-3-large 開始，用 MTEB 在你的領域上評估，如果特定領域的模型勝出超過 3 分，就換掉。

## 交付 (Ship It)

存成 `outputs/skill-embedding-picker.md`：

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## 練習 (Exercises)

1. **簡單。** 用 `bge-small-en-v1.5` 以完整維度 (384) 編碼 100 個句子，再以 Matryoshka 128 維編碼。在 10 個查詢上測量 MRR 的下降。
2. **中等。** 在你領域的 500 個段落上，比較 BGE-M3 的 dense、sparse 與 colbert。哪個在 recall@10 勝出？RRF 融合勝過最好的單一模式嗎？
3. **困難。** 在你前兩個領域任務上，對三個候選模型執行 MTEB。回報 MTEB 分數、100 個查詢 batch 上的 p99 延遲，以及每百萬次查詢的成本 ($)。挑出 Pareto 最佳的那個。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Dense embedding | 那個向量 | 每段文字一個固定大小的向量。用 cosine similarity 排序。 |
| Sparse embedding | 學習而來的 BM25 | 每個詞彙表 token 一個權重；大多是零；端到端訓練。 |
| Multi-vector | ColBERT 風格 | 每個 token 一個向量；MaxSim 計分；索引較大，recall 較好。 |
| Matryoshka | 俄羅斯娃娃技巧 | 前 N 個維度本身就是有效的較小 embedding。 |
| MTEB | 那個 benchmark | Massive Text Embedding Benchmark，發布時 56 個任務，v2 有 100 多個。 |
| BEIR | 檢索 benchmark | 18 個 zero-shot 檢索任務；常被引用來說明跨領域的穩健性。 |
| Asymmetric encoding | 查詢 ≠ 文件的路徑 | 模型對查詢和文件使用不同的投影。 |

## 延伸閱讀 (Further Reading)

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — bi-encoder 論文。
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — 排行榜論文。
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — 統一的三模式模型。
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — 維度階梯的訓練目標。
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — 正式環境中的 late interaction。
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — 即時排名。
