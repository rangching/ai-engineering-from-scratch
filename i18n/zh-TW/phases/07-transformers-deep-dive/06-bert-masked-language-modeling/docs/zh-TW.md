# BERT：Masked Language Modeling (BERT — Masked Language Modeling)

> GPT 預測下一個字。BERT 預測缺掉的字。只差一句話，卻影響了接下來五年所有跟 embedding 有關的東西。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 5 · 02 (Text Representation)
**Time:** ~45 分鐘

## 問題 (The Problem)

2018 年，每個 NLP 任務，不管是情感分析、NER、QA 還是 entailment，都用自己的標註資料從頭訓練自己的模型。當時沒有可以 fine-tune 的、已預訓練好的「理解英文」checkpoint。ELMo（2018）證明了可以用雙向 LSTM 預訓練 contextual embedding；它有幫助，但泛化得不好。

BERT（Devlin et al. 2018）問了一個問題：如果我們拿一個 transformer encoder，用網路上的每個句子去訓練它，逼它根據兩側的 context 預測缺掉的字，會怎樣？然後你再針對下游任務 fine-tune 一個 head。參數效率簡直是一大啟示。

結果是：18 個月內，BERT 和它的變體（RoBERTa、ALBERT、ELECTRA）稱霸了當時存在的每一個 NLP 排行榜。到了 2020 年，地球上每個搜尋引擎、內容審核 pipeline 和語意搜尋系統，裡面都有一個 BERT。

2026 年，encoder-only 模型仍然是分類、檢索和結構化擷取的正確工具。它們每個 token 比 decoder 快 5 到 10 倍，而且它們的 embedding 是每個現代檢索堆疊的骨幹。ModernBERT（2024 年 12 月）用 Flash Attention + RoPE + GeGLU 把這個架構推進到 8K context。

## 核心概念 (The Concept)

![Masked language modeling: pick tokens, mask them, predict originals](../assets/bert-mlm.svg)

### 訓練訊號 (The training signal)

拿一個句子：`the quick brown fox jumps over the lazy dog`。

隨機遮住 15% 的 token：

```
input:  the [MASK] brown fox jumps [MASK] the lazy dog
target: the  quick brown fox jumps  over  the lazy dog
```

訓練模型在被遮住的位置預測原本的 token。因為 encoder 是雙向的，預測位置 1 的 `[MASK]` 可以用到位置 2 之後的 `brown fox jumps`。這是 GPT 做不到的事。

### BERT 的遮罩規則 (The BERT mask rules)

在被選中要預測的 15% token 之中：

- 80% 被換成 `[MASK]`。
- 10% 被換成隨機 token。
- 10% 維持不變。

為什麼不總是用 `[MASK]`？因為 `[MASK]` 在推論時從不出現。如果訓練模型在 100% 被遮住的位置都預期看到 `[MASK]`，預訓練和 fine-tuning 之間就會產生 distribution shift。10% 隨機加 10% 不變，能讓模型保持誠實。

### Next Sentence Prediction (NSP)，以及它為何被拿掉 (Next Sentence Prediction (NSP) — and why it was dropped)

最初的 BERT 也用 NSP 訓練：給定兩個句子 A 和 B，預測 B 是否接在 A 後面。RoBERTa（2019）對它做了 ablation，證明 NSP 有害無益。現代 encoder 都略過它。

### 2026 年有什麼改變：ModernBERT (What changed in 2026: ModernBERT)

2024 年的 ModernBERT 論文用 2026 年的基本元件重建了 block：

| 元件 | 原始 BERT (2018) | ModernBERT (2024) |
|-----------|----------------------|-------------------|
| 位置 | Learned absolute | RoPE |
| Activation | GELU | GeGLU |
| Normalization | LayerNorm | Pre-norm RMSNorm |
| Attention | Full dense | 交替的 local (128) + global |
| Context 長度 | 512 | 8192 |
| Tokenizer | WordPiece | BPE |

而且和 2018 年的堆疊不同，它原生支援 Flash Attention。在序列長度 8K 時，推論比 DeBERTa-v3 快 2 到 3 倍，GLUE 分數也更好。

### 2026 年仍然選 encoder 的使用情境 (Use cases that still pick an encoder in 2026)

| 任務 | 為什麼 encoder 勝過 decoder |
|------|---------------------------|
| 檢索 / 語意搜尋 embedding | 雙向 context = 每個 token 的 embedding 品質更好 |
| 分類（情感、意圖、毒性） | 一次 forward pass；沒有生成的額外開銷 |
| NER / token 標註 | 逐位置輸出，原生雙向 |
| Zero-shot entailment (NLI) | 在 encoder 上面接分類 head |
| RAG 的 reranker | Cross-encoder 評分，比 LLM reranker 快 10 倍 |

```figure
transformer-residual
```

## 動手做 (Build It)

### 步驟 1：遮罩邏輯 (masking logic)

請看 `code/main.py`。函式 `create_mlm_batch` 接收一串 token ID、vocab 大小，以及遮罩機率。回傳 input ID（已套用遮罩）和 label（只在被遮住的位置有值，其他位置是 -100，這是 PyTorch 的 ignore index 慣例）。

```python
def create_mlm_batch(tokens, vocab_size, mask_prob=0.15, rng=None):
    input_ids = list(tokens)
    labels = [-100] * len(tokens)
    for i, t in enumerate(tokens):
        if rng.random() < mask_prob:
            labels[i] = t
            r = rng.random()
            if r < 0.8:
                input_ids[i] = MASK_ID
            elif r < 0.9:
                input_ids[i] = rng.randrange(vocab_size)
            # else: keep original
    return input_ids, labels
```

### 步驟 2：在小型語料庫上執行 MLM 預測 (run MLM prediction on a tiny corpus)

在 20 個單字的詞彙、200 個句子上，訓練 2 層 encoder + MLM head。不做 gradient，我們只做 forward pass 的健全性檢查。完整訓練需要 PyTorch。

### 步驟 3：比較遮罩類型 (compare mask types)

展示三向規則如何讓模型在沒有 `[MASK]` 時也能使用。分別在未遮罩的句子和已遮罩的句子上預測。兩者都該產生合理的 token 分布，因為模型在訓練時兩種模式都看過。

### 步驟 4：Fine-tune head (fine-tune head)

在玩具情感資料集上，把 MLM head 換成分類 head。只訓練 head；encoder 是凍結的。每個 BERT 應用都遵循這個模式。

## 實際使用 (Use It)

```python
from transformers import AutoModel, AutoTokenizer

tok = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")
model = AutoModel.from_pretrained("answerdotai/ModernBERT-base")

text = "Attention is all you need."
inputs = tok(text, return_tensors="pt")
out = model(**inputs).last_hidden_state   # (1, N, 768)
```

**Embedding 模型就是 fine-tune 過的 BERT。** 像 `all-MiniLM-L6-v2` 這類 `sentence-transformers` 模型，是用 contrastive loss 訓練的 BERT。Encoder 一樣，改變的是 loss。

**Cross-encoder reranker 也是 fine-tune 過的 BERT。** 在 `[CLS] query [SEP] doc [SEP]` 上做 pair-classification。Query 和 doc 之間的雙向 attention，正是 cross-encoder 品質優於 biencoder 的原因。

**2026 年何時不該選 BERT。** 任何生成式任務。Encoder 沒有合理的方式 autoregressive 地產生 token。另外：任何 1B 參數以下、小型 decoder 能以更大彈性達到相同品質的情況（Phi-3-Mini、Qwen2-1.5B）。

## 交付 (Ship It)

請看 `outputs/skill-bert-finetuner.md`。這個 skill 會為新的分類或擷取任務，界定 BERT fine-tune 的範圍（backbone 選擇、head 規格、資料、評估、停止條件）。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`，並印出 10,000 個 token 上的遮罩分布。確認大約 15% 被選中，而其中約 80% 變成 `[MASK]`。
2. **中等。** 實作 whole-word masking：如果一個字被 tokenize 成多個 subword，就把所有 subword 一起遮住，或一個都不遮。測量這在 500 句語料庫上是否提升 MLM 準確率。
3. **困難。** 在公開資料集的 10,000 個句子上，訓練一個很小的（2 層、d=64）BERT。針對 SST-2 情感任務 fine-tune `[CLS]` token。與同參數量的 decoder-only 基準比較，誰贏？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| MLM | 「Masked language modeling」 | 訓練訊號：隨機把 15% 的 token 換成 `[MASK]`，預測原本的 token。 |
| Bidirectional | 「兩邊都看」 | Encoder attention 沒有 causal mask，每個位置都看得到其他所有位置。 |
| `[CLS]` | 「Pooler token」 | 加在每個序列最前面的特殊 token；它最後的 embedding 被當作句子層級的表示。 |
| `[SEP]` | 「分段分隔符」 | 分隔成對的序列（例如 query/doc、句子 A/B）。 |
| NSP | 「Next sentence prediction」 | BERT 的第二個預訓練任務；在 RoBERTa 中被證明無用，2019 年後被捨棄。 |
| Fine-tuning | 「適應一個任務」 | 讓 encoder 大多保持凍結；在上面訓練一個小 head 來做下游任務。 |
| Cross-encoder | 「Reranker」 | 同時接收 query 和 doc 作為輸入、輸出相關性分數的 BERT。 |
| ModernBERT | 「2024 年更新版」 | 用 RoPE、RMSNorm、GeGLU、交替 local/global attention、8K context 重建的 encoder。 |

## 延伸閱讀 (Further Reading)

- [Devlin et al. (2018). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding](https://arxiv.org/abs/1810.04805) — 原始論文。
- [Liu et al. (2019). RoBERTa: A Robustly Optimized BERT Pretraining Approach](https://arxiv.org/abs/1907.11692) — 如何正確訓練 BERT；終結了 NSP。
- [Clark et al. (2020). ELECTRA: Pre-training Text Encoders as Discriminators Rather Than Generators](https://arxiv.org/abs/2003.10555) — 在相同算力下，replaced-token detection 勝過 MLM。
- [Warner et al. (2024). Smarter, Better, Faster, Longer: A Modern Bidirectional Encoder](https://arxiv.org/abs/2412.13663) — ModernBERT 論文。
- [HuggingFace `modeling_bert.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/bert/modeling_bert.py) — 標準的 encoder 參考實作。
