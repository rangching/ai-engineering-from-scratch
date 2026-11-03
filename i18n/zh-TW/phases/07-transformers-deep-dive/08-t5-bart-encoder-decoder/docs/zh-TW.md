# T5、BART：Encoder-Decoder 模型 (T5, BART — Encoder-Decoder Models)

> Encoder 負責理解。Decoder 負責生成。把它們拼回去，你就得到一個為「輸入 → 輸出」任務打造的模型：翻譯、摘要、改寫、轉錄。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT), Phase 7 · 07 (GPT)
**Time:** ~45 分鐘

## 問題 (The Problem)

Decoder-only 的 GPT 與 encoder-only 的 BERT，各自為了不同目標精簡了 2017 年的架構。但許多任務天生就是輸入對輸出：

- 翻譯：英文 → 法文。
- 摘要：5,000 個 token 的文章 → 200 個 token 的摘要。
- 語音辨識：音訊 token → 文字 token。
- 結構化擷取：散文 → JSON。

對這些任務，encoder-decoder 最合適。Encoder 產生來源的稠密表示。Decoder 生成輸出，並在每一步都 cross-attend 到那個表示。訓練在輸出端是位移一格。Loss 和 GPT 相同，只是以 encoder 的輸出為條件。

兩篇論文定義了現代的做法：

1. **T5**（Raffel et al. 2019）。「Text-to-Text Transfer Transformer」。把每個 NLP 任務重新表述成文字輸入、文字輸出。單一架構、單一詞彙、單一 loss。以 masked span prediction 預訓練（破壞輸入中的 span，在輸出中把它們解碼出來）。
2. **BART**（Lewis et al. 2019）。「Bidirectional and Auto-Regressive Transformer」。Denoising autoencoder：以多種方式破壞輸入（打亂、遮罩、刪除、旋轉），要求 decoder 重建原文。

2026 年，encoder-decoder 格式在輸入結構重要的地方依然存在：

- Whisper（語音 → 文字）。
- Google 的翻譯堆疊。
- 一些具有明確「上下文與編輯」結構的程式碼補全 / 修復模型。
- 用於結構化推理任務的 Flan-T5 及其變體。

Decoder-only 搶走了鎂光燈，但 encoder-decoder 從未消失。

## 核心概念 (The Concept)

![Encoder-decoder with cross-attention](../assets/encoder-decoder.svg)

### Forward 迴圈 (The forward loop)

```
source tokens ─▶ encoder ─▶ (N_src, d_model)  ──┐
                                                 │
target tokens ─▶ decoder block                   │
                 ├─▶ masked self-attention       │
                 ├─▶ cross-attention ◀───────────┘
                 └─▶ FFN
                ↓
              next-token logits
```

關鍵在於，encoder 對每個輸入只執行一次。Decoder 以 autoregressive 方式執行，但在每一步都 cross-attend 到*同一份* encoder 輸出。快取 encoder 輸出，對長輸入來說是免費的加速。

### T5 預訓練：Span corruption (T5 pretraining — span corruption)

挑選輸入中的隨機 span（平均長度 3 個 token，總共 15%）。把每個 span 換成獨一無二的 sentinel：`<extra_id_0>`、`<extra_id_1>`，依此類推。Decoder 只輸出被破壞的 span，並帶上它們的 sentinel 前綴：

```
source: The quick <extra_id_0> fox jumps <extra_id_1> dog
target: <extra_id_0> brown <extra_id_1> over the lazy
```

比預測整個序列更便宜的訊號。在 T5 論文的 ablation 中，與 MLM（BERT）和 prefix-LM（UniLM）不相上下。

### BART 預訓練：多重雜訊 denoising (BART pretraining — multi-noise denoising)

BART 嘗試五種加噪函式：

1. Token masking。
2. Token deletion。
3. Text infilling（遮住一個 span，由 decoder 插入正確長度的內容）。
4. Sentence permutation。
5. Document rotation。

結合 text infilling + sentence permutation，得到最好的下游數字。Decoder 總是重建原文。BART 的輸出是完整序列，而不只是被破壞的 span，所以預訓練的算力比 T5 高。

### 推論 (Inference)

與 GPT 相同的 autoregressive 生成。Greedy / beam / top-p 取樣都適用。Beam search（寬度 4 到 5）是翻譯和摘要的標準做法，因為輸出分布比聊天更窄。

### 2026 年何時選哪個變體 (When to pick each variant in 2026)

| 任務 | Encoder-decoder？ | 原因 |
|------|------------------|-----|
| 翻譯 | 通常是 | 來源序列明確；輸出分布固定；beam search 有效 |
| Speech-to-text | 是（Whisper） | 輸入模態與輸出不同；encoder 負責塑造音訊特徵 |
| 聊天 / 推理 | 否，用 decoder-only | 沒有持續存在的「輸入」，對話本身就是序列 |
| 程式碼補全 | 通常不是 | 長 context 的 decoder-only 勝出；像 Qwen 2.5 Coder 這類程式碼模型是 decoder-only |
| 摘要 | 兩者皆可 | BART、PEGASUS 勝過早期 decoder-only 基準；現代 decoder-only LLM 與它們不相上下 |
| 結構化擷取 | 兩者皆可 | T5 很乾淨，因為「text → text」能吸收任何輸出格式 |

自 2022 年左右開始的趨勢：decoder-only 接管了過去 encoder-decoder 擁有的任務，因為 (a) 經過 instruction-tuning 的 decoder-only LLM 透過 prompting 就能泛化到任何事，(b) 一種架構比兩種更容易 scale，(c) RLHF 假設的是 decoder。Encoder-decoder 在輸入模態不同（語音、影像）或 beam search 品質重要的地方持續存在。

```figure
encoder-decoder
```

## 動手做 (Build It)

請看 `code/main.py`。我們為一個玩具語料庫實作 T5 風格的 span corruption。這是這堂課中最有用的單一部分，因為它出現在之後每個 encoder-decoder 預訓練配方中。

### 步驟 1：Span corruption (span corruption)

```python
def corrupt_spans(tokens, mask_rate=0.15, mean_span=3.0, rng=None):
    """Pick spans summing to ~mask_rate of tokens. Return (corrupted_input, target)."""
    n = len(tokens)
    n_mask = max(1, int(n * mask_rate))
    n_spans = max(1, int(round(n_mask / mean_span)))
    ...
```

目標格式採用 T5 的慣例：`<sent0> span0 <sent1> span1 ...`。被破壞的輸入把未改變的 token 與位於 span 位置的 sentinel token 交錯排列。

### 步驟 2：驗證 round-trip (verify round-trip)

給定被破壞的輸入與目標，重建原始句子。如果你的破壞是可逆的，forward pass 就是明確定義的。這是健全性檢查，真正的訓練不會這樣做，但這個測試很便宜，能抓出 span 記帳中的 off-by-one 錯誤。

### 步驟 3：BART 加噪 (BART noising)

五個函式：`token_mask`、`token_delete`、`text_infill`、`sentence_permute`、`document_rotate`。組合其中兩個並展示結果。

## 實際使用 (Use It)

HuggingFace 參考範例：

```python
from transformers import T5ForConditionalGeneration, T5Tokenizer
tok = T5Tokenizer.from_pretrained("google/flan-t5-base")
model = T5ForConditionalGeneration.from_pretrained("google/flan-t5-base")

inputs = tok("translate English to French: Attention is all you need.", return_tensors="pt")
out = model.generate(**inputs, max_new_tokens=32)
print(tok.decode(out[0], skip_special_tokens=True))
```

T5 的招數：任務名稱放進輸入文字裡。同一個模型能處理幾十種任務，因為每個任務都是文字輸入、文字輸出。2026 年這個模式已被 instruction-tuned 的 decoder-only 模型推廣，但 T5 是第一個將它制度化的。

## 交付 (Ship It)

請看 `outputs/skill-seq2seq-picker.md`。這個 skill 會根據輸入輸出結構、延遲和品質目標，為新任務在 encoder-decoder 與 decoder-only 之間做選擇。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`，對一個 30 個 token 的句子套用 span corruption，驗證把非 sentinel 的來源 token 與解碼後的目標 span 串接起來，可以重現原句。
2. **中等。** 實作 BART 的 `text_infill` 雜訊：把隨機 span 換成單一 `<mask>` token，decoder 必須推斷出正確的 span 長度加上內容。展示一個範例。
3. **困難。** 在一個很小的英文 → pig-Latin 語料庫（200 對）上 fine-tune `flan-t5-small`。在保留的 50 對資料集上測量 BLEU。與用相同資料、相同算力 fine-tune `Llama-3.2-1B` 的結果比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Encoder-decoder | 「Seq2seq transformer」 | 兩個堆疊：處理輸入的雙向 encoder，以及帶有 cross-attention 的 causal decoder 負責輸出。 |
| Cross-attention | 「來源與目標對話的地方」 | Decoder 的 Q × encoder 的 K/V。Encoder 資訊進入 decoder 的唯一地方。 |
| Span corruption | 「T5 的預訓練招數」 | 把隨機 span 換成 sentinel token；decoder 輸出那些 span。 |
| Denoising objective | 「BART 的玩法」 | 對輸入套用雜訊函式，訓練 decoder 重建乾淨的序列。 |
| Sentinel token | 「`<extra_id_N>` 佔位符」 | 特殊 token，在來源中標記被破壞的 span，並在目標中重新標記它們。 |
| Flan | 「Instruction-tuned T5」 | 在超過 1,800 個任務上 fine-tune 的 T5；讓 encoder-decoder 在遵循指令上具有競爭力。 |
| Beam search | 「Decoding 策略」 | 每一步保留前 k 個部分序列；翻譯 / 摘要的標準做法。 |
| Teacher forcing | 「訓練時的輸入」 | 訓練時餵給 decoder 的是真實的前一個輸出 token，而不是取樣出來的。 |

## 延伸閱讀 (Further Reading)

- [Raffel et al. (2019). Exploring the Limits of Transfer Learning with a Unified Text-to-Text Transformer](https://arxiv.org/abs/1910.10683) — T5。
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension](https://arxiv.org/abs/1910.13461) — BART。
- [Chung et al. (2022). Scaling Instruction-Finetuned Language Models](https://arxiv.org/abs/2210.11416) — Flan-T5。
- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper，2026 年標準的 encoder-decoder。
- [HuggingFace `modeling_t5.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/t5/modeling_t5.py) — 參考實作。
