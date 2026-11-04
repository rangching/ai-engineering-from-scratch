# 從零打造 Tokenizer (Building a Tokenizer from Scratch)

> Lesson 01 給了你一個玩具。這堂課給你一把武器。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Time:** ~90 分鐘

## 學習目標 (Learning Objectives)

- 打造一個正式等級的 BPE tokenizer，能處理 Unicode、空白正規化與 special token
- 實作 byte-level fallback，讓 tokenizer 能編碼任何輸入（包含 emoji、CJK 與程式碼），而不會出現未知 token
- 加入 pre-tokenization 的 regex 規則，在套用 BPE merge 之前先在單字邊界切分文字
- 在語料上訓練自訂的 tokenizer，並針對多語言文字評估它相對於 tiktoken 的 compression ratio

## 問題 (The Problem)

Lesson 01 的 BPE tokenizer 在英文上運作良好。現在丟日文給它。或 emoji。或是混用 tab 和空格的 Python 程式碼。

它會壞掉。

不是因為 BPE 錯了，而是因為實作不完整。正式的 tokenizer 要能處理任何編碼的原始 byte，在切分前先正規化 Unicode，管理永遠不會被 merge 的 special token，把 pre-tokenization 與 subword 切分串起來，而且這一切都要夠快，才不會拖慢處理 15 兆個 token 的訓練管線。

GPT-2 的 tokenizer 有 50,257 個 token。Llama 3 有 128,256 個。GPT-4 大約有 100,000 個。這些都不是玩具等級的數字。那些 vocabulary 背後的 merge table，是用數百 GB 的文字訓練出來的。而周邊的機制，包括正規化、pre-tokenization、special token 注入、chat template 格式化，才是區分「只能處理 "hello world"」和「能處理整個網際網路」的關鍵。

你就要來打造這套機制。

## 核心概念 (The Concept)

### 完整的管線 (The Full Pipeline)

正式的 tokenizer 不是單一演算法，而是由五個階段組成的管線，每個階段解決不同的問題。

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

每個階段都有明確的工作：

| 階段 | 做什麼 | 為什麼重要 |
|-------|-------------|----------------|
| Normalize | NFKC Unicode、可選的轉小寫、可選的去除重音 | "fi" ligature (U+FB01) 會變成 "fi"（兩個字元）。沒有這一步，同一個字會得到不同的 token。 |
| Pre-Tokenize | 在 BPE 之前把文字切成多個區塊 | 避免 BPE 跨越單字邊界 merge。"the cat" 絕對不該產生 "e c" 這種 token。 |
| BPE Merge | 對 byte 序列套用學到的 merge 規則 | 核心的壓縮步驟，把原始 byte 變成 subword token。 |
| Special Tokens | 注入 [BOS]、[EOS]、[PAD]、chat template 標記 | 這些 token 有固定的 ID，絕不參與 BPE merge。模型需要它們來掌握結構。 |
| ID Mapping | 把 token 字串轉成整數 ID | 模型看到的是整數，不是字串。 |

### Byte-Level BPE (Byte-Level BPE)

Lesson 01 的 tokenizer 是對 UTF-8 byte 運作的。這個選擇是對的。但我們略過了一件重要的事：當這些 byte 不是合法的 UTF-8 時會怎樣？

Byte-level BPE 的解法，是把每個可能的 byte 值（0-255）都當成合法的 token。你的基礎 vocabulary 剛好是 256 個項目。任何檔案，不論是文字、二進位或損毀的，都能被 tokenize，而不會產生未知 token。

GPT-2 加了一個小技巧：把每個 byte 對應到一個可顯示的 Unicode 字元，讓 vocabulary 保持人類可讀。Byte 0x20（空格）在他們的對應中會變成字元 "G"。這純粹是外觀上的處理，演算法並不在意。

真正的威力在於：byte-level BPE 能處理地球上的每一種語言。中文字每個佔 3 個 UTF-8 byte。日文可能是 3 到 4 個 byte。阿拉伯文、天城文、emoji，全都只是 byte 序列。BPE 演算法在這些 byte 序列中找出模式的方式，和在英文 ASCII byte 中找模式完全一樣。

### Pre-Tokenization (Pre-Tokenization)

在 BPE 碰到你的文字之前，必須先把它切成區塊。這能避免 merge 演算法產生跨越單字邊界的 token。

GPT-2 用一個 regex 規則來切分文字：

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

這個規則會依縮寫（"don't" 變成 "don" + "'t"）、可帶前導空格的單字、數字、標點與空白來切分。前導空格會黏在單字上，所以 "the cat" 會變成 [" the", " cat"]，而不是 ["the", " ", "cat"]。

Llama 使用 SentencePiece，完全跳過 regex。它把原始 byte 流當成一條長序列，讓 BPE 演算法自己找出邊界。這比較簡單，但給了 BPE 更大的自由度去產生跨單字的 token。

這個選擇很重要。GPT-2 的 regex 讓 tokenizer 無法學到「一個字結尾的 "the" 和下一個字開頭的 "the" 應該合併」。SentencePiece 則允許這件事，有時能得到更有效率的壓縮，但 token 比較難解讀。

### Special Tokens (Special Tokens)

每個正式的 tokenizer 都會保留一些 token ID 給結構性標記：

| Token | 用途 | 使用者 |
|-------|---------|---------|
| `[BOS]` / `<s>` | 序列開頭 | Llama 3, GPT |
| `[EOS]` / `</s>` | 序列結尾 | 所有模型 |
| `[PAD]` | 為了 batch 對齊而補齊 | BERT, T5 |
| `[UNK]` | 未知 token（byte-level BPE 已消除這個需求） | BERT, WordPiece |
| `<\|im_start\|>` | 聊天訊息邊界開始 | ChatGPT, Qwen |
| `<\|im_end\|>` | 聊天訊息邊界結束 | ChatGPT, Qwen |
| `<\|user\|>` | 使用者回合標記 | Llama 3 |
| `<\|assistant\|>` | Assistant 回合標記 | Llama 3 |

Special token 絕不會被 BPE 切開。它們會在 merge 演算法執行前被精確比對，換成固定的 ID，周圍的文字再照一般方式 tokenize。

### Chat Template (Chat Templates)

這是大多數人會搞混、也是大多數實作會出錯的地方。

當你傳訊息給聊天模型時，API 接受的是一個訊息清單：

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

模型看不到 JSON，它看到的是一條扁平的 token 序列。Chat template 用 special token 把訊息轉成那條扁平序列。每個模型的做法都不一樣：

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

template 寫錯，模型就會輸出一堆亂碼。它是用某一種精確的格式訓練的。任何偏離，例如少了換行、換了 token、多了一個空格，都會讓輸入落到訓練分布之外。

### 速度 (Speed)

純 Python 對正式環境的 tokenization 來說太慢了。

tiktoken (OpenAI) 以 Rust 撰寫並提供 Python binding。HuggingFace tokenizers 也是 Rust。SentencePiece 是 C++。這些實作比純 Python 快 10 到 100 倍。

舉個例子：為 Llama 3 的 pre-training 對 15 兆個 token 做 tokenization，如果速度是每秒 100 萬個 token（快的 Python），要花 174 天。如果是每秒 1 億個 token（Rust），只要 1.7 天。

我們用 Python 來寫，是為了理解演算法。在正式環境中，你會使用編譯過的實作，只碰 Python 的包裝層。

```figure
weight-tying
```

## 動手做 (Build It)

### 步驟 1：Byte-Level 編碼 (Step 1: Byte-Level Encoding)

這是基礎。把任何字串轉成 byte 序列，把每個 byte 對應到一個可顯示的字元以便檢視，然後再反向還原。

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

用多語言文字測試，看看 byte 數：

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" 是 5 個 byte。"你好" 是 6 個 byte（每個字 3 個）。火焰 emoji 是 4 個 byte。Byte-level tokenizer 不在乎是哪種語言，byte 就是 byte。

### 步驟 2：用 Regex 寫 Pre-Tokenizer (Step 2: Pre-Tokenizer with Regex)

用 GPT-2 的 regex 規則把文字切成區塊。每個區塊由 BPE 獨立進行 tokenize。

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

`regex` 模組支援 Unicode property escape（`\p{L}` 代表字母，`\p{N}` 代表數字）。標準函式庫的 `re` 模組不支援，所以我們退回使用 ASCII 字元類別。若要做正式的多語言 tokenizer，請安裝 `regex`。

試試看：

```python
print(pre_tokenize("Hello, world! Don't stop."))
# ['Hello', ',', ' world', '!', ' Don', "'t", ' stop', '.']
```

前導空格會黏在它後面的單字上。第一個字前面沒有空格，所以它的區塊是 `Hello`。縮寫會在撇號處切開。標點符號自成一個區塊。BPE 永遠不會跨越這些邊界去 merge token。

### 步驟 3：對 Byte 序列做 BPE (Step 3: BPE on Byte Sequences)

這是 Lesson 01 的核心演算法，但現在對每個經過 pre-tokenization 的區塊獨立運作。

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### 步驟 4：處理 Special Token (Step 4: Special Token Handling)

Special token 需要精確比對與固定的 ID。它們完全繞過 BPE。

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### 步驟 5：完整的 Tokenizer Class (Step 5: Full Tokenizer Class)

把所有東西串起來：normalize、依 special token 切分、pre-tokenize、BPE merge、對應到 ID。

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### 步驟 6：多語言測試 (Step 6: Multilingual Test)

真正的考驗。把英文、中文、emoji 和程式碼丟給它。

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

中文字每個產生 3 個 byte。Emoji 產生 4 個 byte。這些都不會讓 tokenizer 當掉，也不會產生未知 token。這就是 byte-level BPE 的威力。

## 實際使用 (Use It)

### 比較真實的 Tokenizer (Comparing Real Tokenizers)

載入 Llama 3、GPT-4 與 Mistral 實際的 tokenizer，看看它們各自如何處理同一段多語言文字。

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

同一段文字，你會看到不同的 token 數。128K vocabulary 的 Llama 3 在合併常見模式上最積極。100K 的 GPT-4 居中。32K 的 Mistral 產生的 token 比較多，但 embedding 層比較小。

取捨永遠一樣：vocabulary 越大，序列越短，但參數越多。

## 交付 (Ship It)

這堂課會產出一個用來打造與除錯正式 tokenizer 的 prompt。請見 `outputs/prompt-tokenizer-builder.md`。

## 練習 (Exercises)

1. **簡單：** 加入 `get_token_bytes(id)` 方法，顯示任何 token ID 的原始 byte。用它檢查你最常見的 merge 後 token 實際代表什麼。
2. **中等：** 實作 Llama 風格的 pre-tokenizer，依空白與數字切分，但保留前導空格。在同一份語料上，比較它的 vocabulary 與 GPT-2 regex 做法的差異。
3. **困難：** 加入一個 chat template 方法，接收 `{"role": ..., "content": ...}` 訊息清單，並產生符合 Llama 3 聊天格式的正確 token 序列。用 HuggingFace 的實作來測試它。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|----------------------|
| Byte-level BPE | 「以 byte 為單位運作的 tokenizer」 | 基礎 vocabulary 為 256 個 byte 值的 BPE，能處理任何輸入而不產生未知 token |
| Pre-tokenization | 「BPE 之前的切分」 | 以 regex 或規則為基礎的切分，避免 BPE 跨越單字邊界 merge |
| NFKC normalization | 「Unicode 清理」 | 先做正規分解，再做相容性組合，"fi" ligature 會變成 "fi"，全形 "A" 會變成 "A" |
| Chat template | 「訊息如何變成 token」 | 把 role/content 訊息清單轉成扁平 token 序列的精確格式，每個模型各異，且必須與訓練格式一致 |
| Special tokens | 「控制用 token」 | 繞過 BPE 的保留 token ID，例如 [BOS]、[EOS]、[PAD]、聊天標記，在 merge 之前精確比對 |
| Fertility | 「每個字的 token 數」 | 輸出 token 數與輸入單字數的比值，GPT-4 的英文約 1.3，韓文是 2 到 3，越高代表越浪費 context |
| tiktoken | 「OpenAI 的 tokenizer」 | 附 Python binding 的 Rust BPE 實作，比純 Python 快 10 到 100 倍 |
| Merge table | 「vocabulary」 | 訓練中學到的 byte-pair merge 有序清單，它就是 tokenizer 學到的知識 |

## 延伸閱讀 (Further Reading)

- [OpenAI tiktoken source](https://github.com/openai/tiktoken)：GPT-3.5/4 使用的 Rust BPE 實作
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers)：支援 BPE、WordPiece、Unigram 的 Rust tokenizer 函式庫
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783)：128K vocabulary 與 tokenizer 訓練的細節
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226)：與語言無關的 tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py)：最初的 byte 對 Unicode 對應
