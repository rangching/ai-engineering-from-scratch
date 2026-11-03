# Subword Tokenization — BPE、WordPiece、Unigram、SentencePiece

> Word tokenizer 遇到沒見過的字就卡住。Character tokenizer 讓序列長度暴增。Subword tokenizer 取其中間。每個現代 LLM 都建立在其中一種之上。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 5 · 04 (GloVe / FastText / Subword)
**Time:** ~60 分鐘

## 問題 (The Problem)

你的詞彙表有 5 萬個字。使用者輸入 "untokenizable"。你的 tokenizer 回傳 `[UNK]`。模型現在完全得不到這個字的訊號。更糟的是：語料庫中第 90 百分位的文件有 40 個罕見字，這代表每份文件有 40 bits 的資訊被丟掉。

Subword tokenization 解決了這個問題。常見字維持為單一 token。罕見字拆解成有意義的片段：`untokenizable` → `un`、`token`、`izable`。訓練資料涵蓋一切，因為任何字串最終都是一串 byte。

2026 年每個前沿 LLM，都建立在三種演算法之一（BPE、Unigram、WordPiece），並包裝在三種函式庫之一（tiktoken、SentencePiece、HF Tokenizers）裡。不選一個，你就無法交付 language model。

## 核心概念 (The Concept)

![BPE vs Unigram vs WordPiece, character-by-character](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding)。** 從字元層級的詞彙表開始。數出每一組相鄰的 pair，把最頻繁的 pair 合併成新 token。重複直到達到目標詞彙表大小。主流演算法：GPT-2/3/4、Llama、Gemma、Qwen2、Mistral。

**Byte-level BPE。** 同樣的演算法，但作用在原始 byte（256 個基本 token）而不是 Unicode 字元。保證沒有 `[UNK]` token，任何 byte 序列都能編碼。GPT-2 使用 50,257 個 token（256 個 byte 加 50,000 次合併加 1 個特殊 token）。

**Unigram。** 從一個巨大的詞彙表開始。為每個 token 指派 unigram 機率。反覆修剪那些移除後最不會增加語料庫 log-likelihood 的 token。推論時是機率性的：可以取樣 tokenization（透過 subword regularization 做資料增強時很有用）。T5、mBART、ALBERT、XLNet、Gemma 使用它。

**WordPiece。** 合併能最大化訓練語料庫 likelihood 的 pair，而不是原始頻率。BERT、DistilBERT、ELECTRA 使用它。

**SentencePiece vs tiktoken。** SentencePiece 是直接在原始 Unicode 文字上*訓練*詞彙表（BPE 或 Unigram）的函式庫，把空白編碼成 `▁`。tiktoken 是 OpenAI 針對預先建好詞彙表的快速*編碼器*；它不做訓練。

經驗法則：

- **訓練新的詞彙表：** SentencePiece（多語言，不需 pre-tokenization）或 HF Tokenizers。
- **對 GPT 詞彙表做快速推論：** tiktoken（cl100k_base、o200k_base）。
- **兩者皆要：** HF Tokenizers，一個函式庫同時負責訓練和服務。

```figure
bpe-merge
```

## 動手做 (Build It)

### 步驟 1：從零實作 BPE

見 `code/main.py`。迴圈如下：

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

演算法編碼了三個事實。`</w>` 標記字尾，讓 "low"（後綴）和 "lower"（前綴）保持不同。頻率加權讓高頻 pair 早早勝出。合併清單是有順序的，推論時依訓練的順序套用合併。

### 步驟 2：用學到的合併規則編碼

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

單純的做法是 $O(n \cdot \lvert \text{merges} \rvert)$。正式的實作（tiktoken、HF Tokenizers）使用帶有 priority queue 的合併排名查詢，以接近線性的時間執行。

### 步驟 3：SentencePiece 的實際使用

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

注意：不需要 pre-tokenization，空白被編碼成 `▁`，`character_coverage` 控制要多積極地保留罕見字元，而不是把它們對應到 `<unk>`。

### 步驟 4：用 tiktoken 處理與 OpenAI 相容的詞彙表

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

只能編碼。很快（Rust 後端）。與 GPT-4/5 的 tokenization 完全吻合，可用於 byte 計數、成本估算和 context window 預算。

## 2026 年仍會上線的陷阱

- **Tokenizer 漂移。** 用詞彙表 A 訓練，卻對詞彙表 B 部署。Token ID 不同，模型輸出亂碼。在 CI 中檢查 `tokenizer.json` 的 hash。
- **空白歧義。** BPE 的 "hello" 與 " hello" 會產生不同的 token。永遠明確指定 `add_special_tokens` 和 `add_prefix_space`。
- **多語言訓練不足。** 以英文為主的語料庫產生的詞彙表，會把非拉丁文字拆成多 5 到 10 倍的 token。在 GPT-3.5 上，同樣的 prompt 用日文 / 阿拉伯文要多花 5 到 10 倍成本。o200k_base 部分修正了這點。
- **Emoji 切分。** 單一個 emoji 可能佔 5 個 token。在預算 context 時要檢查 emoji 的處理。

## 實際使用 (Use It)

2026 年的技術棧：

| 情境 | 選擇 |
|-----------|------|
| 從零訓練單語言模型 | HF Tokenizers (BPE) |
| 訓練多語言模型 | SentencePiece (Unigram, `character_coverage=0.9995`) |
| 提供與 OpenAI 相容的 API | tiktoken（GPT-4 以上用 `o200k_base`） |
| 特定領域詞彙（程式碼、數學、蛋白質） | 在領域語料庫上訓練自訂 BPE，並與基礎詞彙表合併 |
| 邊緣推論、小模型 | Unigram（較小的詞彙表效果較好） |

詞彙表大小是 scaling 的決定，不是常數。粗略的經驗法則：小於 1B 參數用 32k，1 到 10B 用 50 到 100k，多語言 / 前沿模型用 200k 以上。

## 交付 (Ship It)

存成 `outputs/skill-bpe-vs-wordpiece.md`：

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## 練習 (Exercises)

1. **簡單。** 在 `code/main.py` 的小語料庫上訓練 500 次合併的 BPE。編碼三個保留的字。有多少個產生剛好 1 個 token，多少個產生超過 1 個 token？
2. **中等。** 在 100 個英文 Wikipedia 句子上，比較 `cl100k_base`、`o200k_base`，以及你用 vocab=32k 訓練的 SentencePiece BPE 的 token 數。回報各自的壓縮比。
3. **困難。** 用 BPE、Unigram 和 WordPiece 訓練同一個語料庫。測量在小型情感分類器上使用各自時的下游準確率。這個選擇會讓 F1 移動超過 1 分嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| BPE | Byte-Pair Encoding | 貪婪地合併最頻繁的字元 pair，直到達到目標詞彙表大小。 |
| Byte-level BPE | 永遠沒有未知 token | 在原始 256 個 byte 上做 BPE；GPT-2 / Llama 使用。 |
| Unigram | 機率性 tokenizer | 用 log-likelihood 從大的候選集修剪；T5、Gemma 使用。 |
| SentencePiece | 處理空白的那個 | 在原始文字上訓練 BPE/Unigram 的函式庫；空白編碼為 `▁`。 |
| tiktoken | 快的那個 | OpenAI 以 Rust 為後端、針對預建詞彙表的 BPE 編碼器。不做訓練。 |
| Merge list | 神奇數字 | `(a, b) → ab` 合併的有序清單；推論時依序套用。 |
| Character coverage | 多罕見算太罕見？ | tokenizer 必須涵蓋的訓練語料庫字元比例；通常約 0.9995。 |

## 延伸閱讀 (Further Reading)

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — BPE 論文。
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — Unigram 論文。
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — 這個函式庫。
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — 精簡的參考文件。
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook 加編碼清單。
