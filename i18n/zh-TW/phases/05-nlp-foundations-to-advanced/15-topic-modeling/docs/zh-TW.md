# Topic Modeling — LDA 與 BERTopic

> LDA：文件是多個主題的混合，主題是字詞上的分布。BERTopic：文件在 embedding 空間中聚成群，每一群就是主題。目標相同，拆解方式不同。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 03 (Word2Vec)
**Time:** ~45 分鐘

## 問題 (The Problem)

你有 1 萬張客服工單、5 萬篇新聞，或 20 萬則推文。你想在不逐篇閱讀的情況下，知道這批資料在講什麼。你沒有標註好的類別，甚至不知道有幾個類別。

Topic modeling 不需要監督就能回答這個問題。給它一個語料庫，它會回傳一小組連貫的主題，以及每份文件在這些主題上的分布。

有兩個演算法家族占主導。LDA（2003）把每份文件視為潛在主題的混合，每個主題是字詞上的分布。推論是貝氏式的。在需要 mixed-membership 主題指派和可解釋的字詞層級機率分布時，它仍然用於正式環境。

BERTopic（2020）用 BERT 編碼文件，用 UMAP 降維，用 HDBSCAN 分群，並透過 class-based TF-IDF 抽取主題字詞。它在短文本、社群媒體，以及語意相似度比字詞重疊更重要的場合勝出。一份文件只得到一個主題，這對長篇內容是限制。

這堂課會為兩者建立直覺，並指出對特定語料庫該選哪一個。

## 核心概念 (The Concept)

![LDA mixture model vs BERTopic clustering](../assets/topic-modeling.svg)

**LDA 的生成故事。** 每個主題是字詞上的分布。每份文件是主題的混合。要在文件中產生一個字，先從文件的混合中取樣一個主題，再從該主題的分布中取樣一個字。推論則反過來：給定觀察到的字，推斷每份文件的主題分布和每個主題的字詞分布。由 collapsed Gibbs sampling 或 variational Bayes 負責運算。

LDA 的關鍵輸出：

- `doc_topic`：矩陣 `(n_docs, n_topics)`，每一列總和為 1（文件的主題混合）。
- `topic_word`：矩陣 `(n_topics, vocab_size)`，每一列總和為 1（主題的字詞分布）。

**BERTopic pipeline。**

1. 用 sentence transformer（例如 `all-MiniLM-L6-v2`）編碼每份文件，得到 384 維向量。
2. 用 UMAP 降到約 5 維。BERT embedding 的維度對分群來說太高。
3. 用 HDBSCAN 分群。以密度為基礎，產生大小不一的群，以及一個「outlier」標籤。
4. 對每一群，在該群的文件上計算 class-based TF-IDF，抽出排名最高的字詞。

輸出是每份文件一個主題（加上 -1 的 outlier 標籤）。也可以選擇透過 HDBSCAN 的機率向量取得 soft membership。

```figure
topic-drift
```

## 動手做 (Build It)

### 步驟 1：用 scikit-learn 做 LDA

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import numpy as np


def fit_lda(documents, n_topics=5, max_features=1000):
    cv = CountVectorizer(
        max_features=max_features,
        stop_words="english",
        min_df=2,
        max_df=0.9,
    )
    X = cv.fit_transform(documents)
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=50,
        learning_method="online",
    )
    doc_topic = lda.fit_transform(X)
    feature_names = cv.get_feature_names_out()
    return lda, cv, doc_topic, feature_names


def print_top_words(lda, feature_names, n_top=10):
    for idx, topic in enumerate(lda.components_):
        top_idx = np.argsort(-topic)[:n_top]
        words = [feature_names[i] for i in top_idx]
        print(f"topic {idx}: {' '.join(words)}")
```

注意：移除了 stopword，用 min_df 和 max_df 過濾罕見與無所不在的詞，用 CountVectorizer（而不是 TfidfVectorizer），因為 LDA 期待的是原始計數。

### 步驟 2：BERTopic（正式環境）

```python
from bertopic import BERTopic

topic_model = BERTopic(
    embedding_model="sentence-transformers/all-MiniLM-L6-v2",
    min_topic_size=15,
    verbose=True,
)

topics, probs = topic_model.fit_transform(documents)
info = topic_model.get_topic_info()
print(info.head(20))
valid_topics = info[info["Topic"] != -1]["Topic"].tolist()
for topic_id in valid_topics[:5]:
    print(f"topic {topic_id}: {topic_model.get_topic(topic_id)[:10]}")
```

對 `Topic != -1` 的過濾，會丟掉 BERTopic 的 outlier 桶（HDBSCAN 無法分群的文件）。`min_topic_size` 控制 HDBSCAN 的最小群大小；BERTopic 函式庫的預設值是 10。這個範例為了符合這堂課的規模，明確把它設為 15。對超過 1 萬份文件的語料庫，請增加到 50 或 100。

### 步驟 3：評估

兩種方法都會輸出主題字詞。問題是這些字詞是否連貫。

- **Topic coherence (c_v)。** 結合滑動視窗 context 上，排名最高字詞對的 NPMI（normalized pointwise mutual information），把分數彙整成主題向量，再用 cosine similarity 比較這些向量。越高越好。使用 `gensim.models.CoherenceModel` 並設定 `coherence="c_v"`。
- **Topic diversity。** 所有主題排名最高的字詞中，不重複字詞的比例。越高越好（主題之間不重疊）。
- **質性檢查。** 讀每個主題排名最高的字詞。它們有指向真實的東西嗎？人類判斷仍然是最後一道防線。

## 該選哪一個

| 情境 | 選擇 |
|-----------|------|
| 短文本（推文、評論、標題） | BERTopic |
| 含有主題混合的長文件 | LDA |
| 沒有 GPU / 運算資源有限 | LDA 或 NMF |
| 需要文件層級的多主題分布 | LDA |
| 用 LLM 整合來標註主題 | BERTopic（直接支援） |
| 資源受限的邊緣部署 | LDA |
| 最大的語意連貫性 | BERTopic |

最大的實務考量是文件長度。BERT embedding 會截斷；LDA 的計數則適用於任何長度。對長於 embedding 模型 context 的文件，要嘛切 chunk 再彙整，要嘛使用 LDA。

## 實際使用 (Use It)

2026 年的技術棧：

- **BERTopic。** 短文本，以及語意重要的任何場合的預設選擇。
- **`gensim.models.LdaModel`。** 正式環境的傳統 LDA，成熟、久經考驗。
- **`sklearn.decomposition.LatentDirichletAllocation`。** 適合實驗用的簡易 LDA。
- **NMF。** Non-negative matrix factorization。LDA 的快速替代方案，在短文本上品質相當。
- **Top2Vec。** 設計類似 BERTopic。社群較小，但在某些 benchmark 上表現不錯。
- **FASTopic。** 較新，在非常大的語料庫上比 BERTopic 快。
- **以 LLM 為基礎的標註。** 先執行任何分群，再提示模型為每一群命名。

## 交付 (Ship It)

存成 `outputs/skill-topic-picker.md`：

```markdown
---
name: topic-picker
description: Pick LDA or BERTopic for a corpus. Specify library, knobs, evaluation.
version: 1.0.0
phase: 5
lesson: 15
tags: [nlp, topic-modeling]
---

Given a corpus description (document count, avg length, domain, language, compute budget), output:

1. Algorithm. LDA / NMF / BERTopic / Top2Vec / FASTopic. One-sentence reason.
2. Configuration. Number of topics: `recommended = max(5, round(sqrt(n_docs)))`, clamped to 200 for corpora under 40,000 docs; permit >200 only when the corpus is genuinely large (>40k) and note the increased compute cost. `min_df` / `max_df` filters and embedding model for neural approaches also belong here.
3. Evaluation. Topic coherence (c_v) via `gensim.models.CoherenceModel`, topic diversity, and a 20-sample human read.
4. Failure mode to probe. For LDA, "junk topics" absorbing stopwords and frequent terms. For BERTopic, the -1 outlier cluster swallowing ambiguous documents.

Refuse BERTopic on documents longer than the embedding model's context window without a chunking strategy. Refuse LDA on very short text (tweets, reviews under 10 tokens) as coherence collapses. Flag any n_topics choice below 5 as likely wrong; flag >200 on corpora under 40k docs as likely over-splitting.
```

## 練習 (Exercises)

1. **簡單。** 在 20 Newsgroups 資料集上擬合 5 個主題的 LDA。印出每個主題排名前 10 的字詞，手動為每個主題貼標籤。演算法找到真正的類別了嗎？
2. **中等。** 在同一個 20 Newsgroups 子集上擬合 BERTopic。與 LDA 比較找到的主題數量、排名最高的字詞，以及質性連貫性。哪一個更乾淨地呈現出真正的類別？
3. **困難。** 在你的語料庫上，為 LDA 和 BERTopic 都計算 c_v coherence。各以 5、10、20、50 個主題執行。畫出 coherence 對主題數的圖。回報哪個方法在不同主題數之間更穩定。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Topic | 語料庫在談的東西 | 字詞上的機率分布 (LDA)，或一群相似的文件 (BERTopic)。 |
| Mixed membership | 一份文件屬於多個主題 | LDA 為每份文件指派所有主題上的分布。 |
| UMAP | 降維 | 保留局部結構的 manifold learning；BERTopic 使用它。 |
| HDBSCAN | 密度分群 | 找出大小不一的群；對 outlier 產生「雜訊」標籤 (-1)。 |
| c_v coherence | 主題品質指標 | 滑動視窗上排名最高主題字詞的 NPMI，轉成 context 向量後以 cosine similarity 比較。 |

## 延伸閱讀 (Further Reading)

- [Blei, Ng, Jordan (2003). Latent Dirichlet Allocation](https://www.jmlr.org/papers/volume3/blei03a/blei03a.pdf) — LDA 論文。
- [Grootendorst (2022). BERTopic: Neural topic modeling with a class-based TF-IDF procedure](https://arxiv.org/abs/2203.05794) — BERTopic 論文。
- [Röder, Both, Hinneburg (2015). Exploring the Space of Topic Coherence Measures](https://svn.aksw.org/papers/2015/WSDM_Topic_Evaluation/public.pdf) — 引入 c_v 與相關指標的論文。
- [BERTopic documentation](https://maartengr.github.io/BERTopic/) — 正式環境的參考文件，範例很棒。
