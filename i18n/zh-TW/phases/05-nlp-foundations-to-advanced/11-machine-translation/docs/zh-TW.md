# 機器翻譯 (Machine Translation)

> 翻譯是資助了 NLP 研究三十年的任務，現在仍持續資助著。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Time:** ~75 分鐘

## 問題 (The Problem)

模型讀入一種語言的句子，產生另一種語言的句子。長度會變，詞序會變。有些來源字對應到多個目標字，反之亦然。慣用語拒絕一對一的對應。"I miss you" 在法文是 "tu me manques"，字面上是「你對我來說是缺少的」。沒有任何 word-level 的對齊能撐過這種情況。

機器翻譯是迫使 NLP 發明 encoder-decoder、attention、transformer，最終發展出整個 LLM 典範的任務。每一步進展的出現，都是因為翻譯品質是可以衡量的，而人類與機器之間的差距又很頑固。

這堂課略過歷史課，直接教 2026 年可運作的 pipeline：預訓練的多語言 encoder-decoder（NLLB-200 或 mBART）、subword tokenization、beam search、BLEU 與 chrF 評估，以及少數仍會在沒被抓到的情況下上線的失敗模式。

## 核心概念 (The Concept)

![MT pipeline: tokenize → encode → decode with attention → detokenize](../assets/mt-pipeline.svg)

現代 MT 是在平行文本上訓練的 transformer encoder-decoder。Encoder 以來源語言的 tokenization 讀取來源。Decoder 一次一個 subword 地產生目標，並透過 cross-attention（第 10 課）使用 encoder 的輸出。解碼使用 beam search 來避開 greedy decoding 的陷阱。輸出會經過 detokenize、detruecase，並對照參考譯文計分。

三個營運上的選擇，決定了真實世界的 MT 品質。

- **Tokenizer。** 在混合語言語料庫上訓練的 SentencePiece BPE。跨語言共享詞彙表，是 NLLB 能做到 zero-shot 語言對的關鍵。
- **模型大小。** NLLB-200 distilled 600M 可以放進筆電。NLLB-200 3.3B 是已發布的正式環境預設。54.5B 是研究的上限。
- **解碼。** 一般內容使用 beam width 4 到 5。用長度懲罰 (length penalty) 避免輸出太短。需要術語一致時，使用 constrained decoding。

```figure
seq2seq-alignment
```

## 動手做 (Build It)

### 步驟 1：呼叫預訓練的 MT

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

這裡有三件事很重要。`src_lang` 告訴 tokenizer 要套用哪種文字系統和切分方式。`forced_bos_token_id` 告訴 decoder 要產生哪種語言。兩者都是 NLLB 專屬的技巧；mBART 和 M2M-100 有各自的慣例，彼此不能互換。

### 步驟 2：BLEU 與 chrF

BLEU 衡量輸出與參考譯文之間的 n-gram 重疊。四種參考 n-gram 大小 (1 到 4)，對 precision 取幾何平均，並對太短的輸出加上 brevity penalty。分數在 [0, 100] 之間。常被使用，卻讓人難以解讀：30 BLEU 是「堪用」，40 是「好」，50 是「卓越」，低於 1 BLEU 的差異只是雜訊。

chrF 衡量字元層級的 F-score。對詞形變化豐富的語言更敏感，這類語言的 BLEU 會少算匹配。常與 BLEU 一起回報。

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

請永遠使用 `sacrebleu`。它會正規化 tokenization，讓不同論文的分數可以比較。自己寫 BLEU 計算，就是誤導性 benchmark 的由來。

### 三層評估階層 (2026)

現代 MT 評估使用三個互補的指標家族。至少要用其中兩個才能上線。

- **啟發式 (Heuristic)**（BLEU、chrF）。快速、以參考譯文為基礎、可解釋，對改寫不敏感。用於與舊有結果比較和偵測退步。
- **學習式 (Learned)**（COMET、BLEURT、BERTScore）。以人類判斷訓練的神經網路模型，比較譯文與來源、參考譯文之間的語意相似度。COMET 自 2023 年起與 MT 研究的關聯性最高，在品質重要的場合是 2026 年的正式環境預設。
- **LLM-as-judge**（不需參考譯文）。提示大型模型就流暢度、充分性、語氣、文化適切性為譯文打分。當評分標準設計得好時，GPT-4-as-judge 與人類的一致率約 80%。用於沒有參考譯文的開放式內容。

2026 年務實的技術棧：用 `sacrebleu` 算 BLEU 和 chrF，用 `unbabel-comet` 算 COMET，再用經過提示的 LLM 作為最後面向人類的訊號。在相信任何指標用於正式資料之前，先用 50 到 100 筆人工標註的範例校準它。

不需參考譯文的指標（COMET-QE、BLEURT-QE、LLM-as-judge）讓你不用參考譯文就能評估翻譯，這對沒有參考譯文的長尾語言對很重要。

### 步驟 3：正式環境會壞在哪

上面可運作的 pipeline，有 80% 的時候會流暢地翻譯，另外 20% 則會無聲地失敗。有名字的失敗模式：

- **Hallucination（幻覺）。** 模型捏造來源裡沒有的內容。在不熟悉的領域詞彙上很常見。症狀：輸出很流暢，卻宣稱來源沒說過的事實。緩解方式：對領域術語使用 constrained decoding、對受法規管制的內容做人工審查、監控輸出遠長於輸入的情況。
- **Off-target generation（翻成錯誤語言）。** 模型翻成錯誤的語言。NLLB 在罕見語言對上出乎意料地容易發生。緩解方式：確認 `forced_bos_token_id`，並永遠對輸出做語言辨識 (language-ID) 模型檢查。
- **Terminology drift（術語漂移）。** "Sign up" 在文件 1 變成 "s'inscrire"，在文件 2 變成 "créer un compte"。對 UI 文字和面向使用者的字串，一致性比原始品質更重要。緩解方式：使用以詞彙表約束的解碼，或後編輯字典。
- **Formality mismatch（正式程度不符）。** 法文的 "tu" vs "vous"、日文的敬語層級。模型會選訓練中較常見的那種形式。對面向客戶的內容，這通常是錯的。緩解方式：若模型支援，在 prompt 前面加上正式程度 token，或用僅含正式用語的語料庫 fine-tune 小模型。
- **短輸入的長度爆炸。** 非常短的輸入句子常產生過長的譯文，因為在來源少於約 5 個 token 時，長度懲罰會急遽失效。緩解方式：依來源長度成比例設定硬性的最大長度上限。

### 步驟 4：針對領域做 fine-tuning

預訓練模型是通才。法律、醫療或遊戲對話翻譯，在領域平行資料上 fine-tune 後會有可測量的提升。做法並不特別：

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

幾千筆高品質的平行範例，勝過幾十萬筆雜訊多的網路爬取資料。訓練資料的品質，是正式環境中最大的單一槓桿。

## 實際使用 (Use It)

2026 年 MT 的正式環境技術棧：

| 使用情境 | 建議的起點 |
|---------|---------------------------|
| 任一對任一，200 種語言 | `facebook/nllb-200-distilled-600M`（筆電）或 `nllb-200-3.3B`（正式環境） |
| 以英文為中心、高品質、50 種語言 | `facebook/mbart-large-50-many-to-many-mmt` |
| 短量使用、便宜的推論、英法德西 | Helsinki-NLP / Marian 模型 |
| 對延遲要求嚴格的瀏覽器端 | ONNX 量化的 Marian（約 50 MB） |
| 最高品質、願意付費 | 搭配翻譯 prompt 的 GPT-4 / Claude / Gemini |

截至 2026 年，LLM 在數個語言對上已經超越專用的 MT 模型，尤其是慣用語內容和長 context。代價是每個 token 的成本和延遲。當 context 長度、風格一致性，或透過 prompt 做領域適應比吞吐量更重要時，就選 LLM。

## 交付 (Ship It)

存成 `outputs/skill-mt-evaluator.md`：

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## 練習 (Exercises)

1. **簡單。** 用 `nllb-200-distilled-600M` 把一段 5 個句子的英文段落翻成法文，再翻回英文。測量來回翻譯的結果與原文有多接近。你應該會看到語意被保留，但用字有漂移。
2. **中等。** 用 `fasttext lid.176` 或 `langdetect` 對翻譯輸出實作語言辨識檢查。整合到 MT 呼叫中，讓翻成錯誤語言的情況在回傳前就被抓到。
3. **困難。** 在你選擇的 5,000 組領域語料庫上 fine-tune `nllb-200-distilled-600M`。在保留集上測量 fine-tuning 前後的 BLEU。回報哪些類型的句子進步了、哪些退步了。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| BLEU | 翻譯分數 | 帶有 brevity penalty 的 n-gram precision。[0, 100]。 |
| chrF | 字元 F-score | 字元層級的 F-score。對詞形變化豐富的語言更敏感。 |
| NMT | 神經機器翻譯 | 在平行文本上訓練的 transformer encoder-decoder。2017 年後的預設。 |
| NLLB | No Language Left Behind | Meta 的 200 種語言 MT 模型家族。 |
| Constrained decoding | 受控輸出 | 強制特定 token 或 n-gram 出現 / 不出現在輸出中。 |
| Hallucination | 捏造的內容 | 來源無法支持的模型輸出。 |

## 延伸閱讀 (Further Reading)

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — NLLB 論文。
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — 為什麼 `sacrebleu` 是回報 BLEU 唯一正確的方式。
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — chrF 論文。
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — 實用的 fine-tuning 逐步講解。
