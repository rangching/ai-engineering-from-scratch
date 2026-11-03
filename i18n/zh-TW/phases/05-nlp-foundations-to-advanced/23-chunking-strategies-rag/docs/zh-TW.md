# RAG 的 Chunking 策略 (Chunking Strategies for RAG)

> Chunking 設定對檢索品質的影響，與選擇 embedding 模型一樣大（Vectara NAACL 2025）。Chunking 做錯，再多的 reranking 也救不了你。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 14 (Information Retrieval), Phase 5 · 22 (Embedding Models)
**Time:** ~60 分鐘

## 問題 (The Problem)

你把一份 50 頁的合約放進 RAG 系統。使用者問：「What is the termination clause?」Retriever 卻回傳封面頁。為什麼？因為模型是用 512 token 的 chunk 訓練的，而終止條款位於第 20 頁，被分頁切成兩半，附近也沒有能把它和查詢連起來的關鍵字。

解法不是「買更好的 embedding 模型」，而是 chunking。要多大？要重疊嗎？在哪裡切？要帶周圍的 context 嗎？

2026 年 2 月的 benchmark 顯示了令人意外的結果：

- Vectara 2026 年的研究：recursive 512 token chunking 勝過 semantic chunking，準確率 69% 對 54%。
- 在 Natural Questions 上用 SPLADE 加 Mistral-8B：overlap 沒有帶來任何可測量的好處。
- Context 懸崖：回應品質在約 2,500 token 的 context 附近急遽下降。

「顯而易見」的答案（semantic chunking、20% overlap、1000 token）常常是錯的。這堂課為六種策略建立直覺，並告訴你何時該用哪一種。

## 核心概念 (The Concept)

![Six chunking strategies visualized on one passage](../assets/chunking.svg)

**Fixed chunking。** 每 N 個字元或 token 切一刀。最簡單的 baseline。會在句子中間切斷。壓縮好，連貫性差。

**Recursive。** LangChain 的 `RecursiveCharacterTextSplitter`。先試著在 `\n\n` 切，再試 `\n`，然後 `.`，最後是空白。乾淨地逐級退回。2026 年的預設。

**Semantic。** 對每個句子做 embedding。計算相鄰句子之間的 cosine similarity。在相似度低於門檻的地方切開。保留主題的連貫性。比較慢；有時會產生僅 40 token 的細小碎片，傷害檢索。

**Sentence。** 在句子邊界切開。每個 chunk 一個句子，或 N 個句子的視窗。在約 5k token 以內，能以一小部分成本追平 semantic chunking。

**Parent-document。** 儲存小的子 chunk 用於檢索，*同時*儲存較大的父 chunk 用於提供 context。以子 chunk 檢索；回傳父 chunk。優雅地退化：不好的子 chunk 仍然會回傳合理的父 chunk。

**Late chunking（2024）。** 先在 token 層級對整份文件做 embedding，再把 token embedding 彙整 (pool) 成 chunk embedding。保留跨 chunk 的 context。適用於長 context 的 embedder（BGE-M3、Jina v3）。運算量較高。

**Contextual retrieval（Anthropic, 2024）。** 在每個 chunk 前面加上 LLM 產生的、描述它在文件中位置的摘要（「This chunk is section 3.2 of the termination clauses...」）。在 Anthropic 自己的 benchmark 中，檢索提升 35 到 50%。建立索引的成本昂貴。

### 勝過每個預設值的規則

讓 chunk 大小配合查詢類型：

| 查詢類型 | Chunk 大小 |
|------------|-----------|
| 事實型 ("what is the CEO's name?") | 256-512 token |
| 分析型 / 多跳 | 512-1024 token |
| 整個章節的理解 | 1024-2048 token |

NVIDIA 2026 年的 benchmark。Chunk 要大到足以包含答案加上局部 context，又要小到讓 retriever 的 top-K 結果聚焦在答案，而不是 context 雜訊。

```figure
n5-chunk-cuts
```

## 動手做 (Build It)

### 步驟 1：fixed 與 recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### 步驟 2：semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

在你的領域上調整 `threshold`。太高 → 碎片。太低 → 一個巨大的 chunk。

### 步驟 3：parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

關鍵洞察：要對父 chunk 去重。多個子 chunk 可能對應到同一個父 chunk；全部回傳會浪費 context。

### 步驟 4：contextual retrieval（Anthropic 模式）

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

為加上 context 的 chunk 建立索引。查詢時，檢索會受益於額外的周邊訊號。

### 步驟 5：評估

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

一定要做 benchmark。對你的語料庫「最好」的策略，可能與任何部落格文章都不符。

## 陷阱

- **只用事實型查詢評估 chunking。** 多跳查詢會揭露截然不同的贏家。請使用依查詢類型分層的評估集。
- **Semantic chunking 沒有最小大小。** 會產生傷害檢索的 40 token 碎片。一定要強制 `min_tokens`。
- **把 overlap 當成貨物崇拜。** 2026 年的研究發現，overlap 常常沒有任何好處，卻讓索引成本加倍。要量測，不要假設。
- **沒有強制最小 / 最大值。** 5 token 或 5000 token 的 chunk 都會破壞檢索。要夾住範圍。
- **跨文件 chunking。** 絕不讓 chunk 跨越兩份文件。一定要逐份文件做 chunking，再合併。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 策略 |
|-----------|----------|
| 第一次建置，未知的語料庫 | Recursive、512 token、不重疊 |
| 事實型 QA | Recursive、256-512 token |
| 分析型 / 多跳 | Recursive、512-1024 token 加 parent-document |
| 大量交叉引用（合約、論文） | Late chunking 或 contextual retrieval |
| 對話 / 對話語料庫 | 輪次層級的 chunk 加說話者 metadata |
| 短句（推文、評論） | 一份文件 = 一個 chunk |

從 recursive 512 開始。在 50 個查詢的評估集上測量 recall@5。再從那裡調整。

## 交付 (Ship It)

存成 `outputs/skill-chunker.md`：

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## 練習 (Exercises)

1. **簡單。** 用 fixed(512, 0)、recursive(512, 0) 與 recursive(512, 100) 對一份 20 頁的文件做 chunking。比較 chunk 數量與邊界品質。
2. **中等。** 在 5 份文件上建立 30 個查詢的評估集。測量 recursive、semantic 與 parent-document 的 recall@5。哪個勝出？和部落格文章相符嗎？
3. **困難。** 實作 contextual retrieval。測量相對於 baseline recursive 的 MRR 提升。回報索引成本（LLM 呼叫次數）對準確率增益。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Chunk | 文件的一塊 | 會被 embed、建索引與檢索的子文件單位。 |
| Overlap | 安全邊際 | 相鄰 chunk 共享的 N 個 token；在 2026 年的 benchmark 中常常沒用。 |
| Semantic chunking | 聰明的 chunking | 在相鄰句子 embedding 相似度下降的地方切開。 |
| Parent-document | 兩層檢索 | 檢索小的子 chunk，回傳較大的父 chunk。 |
| Late chunking | embedding 之後才 chunk | 在 token 層級 embed 完整文件，再 pool 成 chunk 向量。 |
| Contextual retrieval | Anthropic 的招式 | 建立索引前，在每個 chunk 前面加上 LLM 產生的摘要。 |
| Context cliff | 2500 token 的牆 | RAG 中在約 2.5k context token 附近觀察到的品質下降（2026 年 1 月）。 |

## 延伸閱讀 (Further Reading)

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — 正式環境的預設。
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking 與 embedding 的選擇同樣重要。
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — late chunking 論文。
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — 用 LLM 產生的 context 前綴，讓檢索提升 35 到 50%。
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — 依查詢類型決定 chunk 大小。
