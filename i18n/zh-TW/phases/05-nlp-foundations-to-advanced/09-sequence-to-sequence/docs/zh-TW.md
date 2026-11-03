# Sequence-to-Sequence 模型

> 兩個假裝成翻譯員的 RNN。它們撞上的瓶頸，就是 attention 存在的原因。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Time:** ~75 分鐘

## 問題 (The Problem)

分類把變動長度的序列對應到單一標籤。翻譯則把變動長度的序列對應到另一個變動長度的序列。輸入和輸出使用不同的詞彙表，可能是不同的語言，也不保證長度一致。

seq2seq 架構（Sutskever、Vinyals、Le，2014）用一個刻意簡單的做法解決了這件事：兩個 RNN。一個讀來源句子，產生固定大小的 context vector。另一個讀取那個向量，逐個 token 產生目標句子。程式碼和你在第 08 課寫的一樣，只是接起來的方式不同。

這值得研究，有兩個原因。第一，context vector 的瓶頸是 NLP 中教學價值最高的失敗。它促成了 attention 和 transformer 擅長的一切。第二，訓練的做法（teacher forcing、scheduled sampling、推論時用 beam search）至今仍適用於每個現代的生成系統，包括 LLM。

## 核心概念 (The Concept)

**Encoder。** 讀取來源句子的 RNN。它最後的 hidden state 就是 **context vector**，整個輸入的固定大小摘要。據說，除了來源本身，什麼都不會丟失。

**Decoder。** 另一個以 context vector 初始化的 RNN。每一步它把先前產生的 token 當輸入，產生目標詞彙表上的分布。取樣或用 argmax 選出下一個 token，再餵回去。重複直到產生 `<EOS>` token 或達到最大長度。

**訓練：** 每個 decoder 步驟計算 cross-entropy loss，再對整個序列加總。對兩個網路做標準的 backprop through time。

**Teacher forcing。** 訓練時，decoder 在步驟 `t` 的輸入是位置 `t-1` 的*真實* token，而不是 decoder 自己先前的預測。這讓訓練穩定；沒有它，早期的錯誤會連鎖擴大，模型永遠學不會。推論時你必須使用模型自己的預測，所以訓練與推論之間永遠存在分布落差。這個落差稱為 **exposure bias**。

**瓶頸。** encoder 學到的關於來源的一切，都必須擠進那一個 context vector。長句子會損失細節，罕見字會變模糊。重新排序（chat noir vs. black cat）必須靠記憶，而不是計算。

Attention（第 10 課）藉由讓 decoder 看*每一個* encoder hidden state，而不只是最後一個，來修正這點。整個賣點就是這樣。

```figure
lstm-gates
```

## 動手做 (Build It)

### 步驟 1：encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` 的形狀是 `[batch, seq_len, hidden_dim]`，每個輸入位置一個 hidden state。`hidden` 的形狀是 `[1, batch, hidden_dim]`，是最後一步。第 08 課說過「做分類時對 outputs 做 pool」。這裡我們保留最後的 hidden state 當作 context vector，並忽略每一步的 outputs。

### 步驟 2：decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder 一次呼叫一步。輸入：一批單一 token 和目前的 hidden state。輸出：下一個 token 的詞彙表 logits 與更新後的 hidden state。

### 步驟 3：使用 teacher forcing 的訓練迴圈

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

有兩個值得點名的參數。`ignore_index=0` 會跳過 padding token 上的 loss。`teacher_forcing_ratio` 是每一步使用真實 token 而非模型預測的機率。從 1.0（完全 teacher forcing）開始，訓練過程中逐步退火到約 0.5，以縮小 exposure-bias 的落差。

### 步驟 4：推論迴圈（greedy）

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding 每一步都挑機率最高的 token。它可能會走偏：一旦你選定了某個 token，就收不回來了。**Beam search** 讓前 `k` 個局部序列保持存活，最後挑出分數最高的完整序列。Beam width 3 到 5 是標準做法。

### 步驟 5：示範瓶頸

在玩具複製任務上訓練模型：來源 `[a, b, c, d, e]`，目標 `[a, b, c, d, e]`。增加序列長度，觀察準確率。

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

單一的 GRU hidden state 無法無損地記住 40 個 token 的輸入。資訊在每個 encoder 步驟都存在，但 decoder 只看得到最後的 state。Attention 直接解決這點。

## 實際使用 (Use It)

PyTorch 有 `nn.Transformer` 和以 `nn.LSTM` 為基礎的 seq2seq 範本。Hugging Face 的 `transformers` 函式庫提供完整的 encoder-decoder 模型（BART、T5、mBART、NLLB），它們是在數十億個 token 上訓練的。

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

現代的 encoder-decoder 已經用 transformer 取代了 RNN。高層次的形狀（encoder、decoder、逐 token 生成）與 2014 年的 seq2seq 論文完全相同，只是每個區塊內部的機制不同。

### 什麼時候仍然使用 RNN-based seq2seq

新專案幾乎不會。特定的例外：

- 串流翻譯，你要在有界記憶體下一次消化一個 token 的輸入。
- 裝置端的文字生成，transformer 的記憶體成本高到無法負擔。
- 教學。理解 encoder-decoder 的瓶頸，是理解 transformer 為什麼勝出的最快路徑。

### Exposure bias 及其緩解方式

- **Scheduled sampling。** 訓練期間逐步退火 teacher forcing 的比例，讓模型學會從自己的錯誤中恢復。
- **Minimum risk training。** 用句子層級的 BLEU 分數取代 token 層級的 cross-entropy 來訓練，更接近你真正想要的。
- **Reinforcement learning fine-tuning。** 用某個指標獎勵序列生成器。現代 LLM 的 RLHF 就用到它。

這三種做法仍然適用於以 transformer 為基礎的生成。

## 交付 (Ship It)

存成 `outputs/prompt-seq2seq-design.md`：

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## 練習 (Exercises)

1. **簡單。** 實作玩具複製任務。在目標等於來源的輸入輸出 pair 上訓練 GRU seq2seq。測量長度 5、10、20 的準確率，重現瓶頸。
2. **中等。** 加入 beam width 為 3 的 beam search 解碼。在小型平行語料庫上測量 BLEU，並與 greedy 比較。記錄 beam search 在哪裡勝出（通常是最後幾個 token），以及在哪裡沒有差別。
3. **困難。** 在 1 萬組 paraphrase 資料集上 fine-tune `facebook/bart-base`。在保留的輸入上，比較 fine-tuned 模型的 beam-4 輸出與基礎模型的輸出。回報 BLEU，並挑 10 個質性的例子。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| Encoder | 輸入的 RNN | 讀取來源。產生每一步的 hidden state 與最後的 context vector。 |
| Decoder | 輸出的 RNN | 以 context vector 初始化。逐個產生目標 token。 |
| Context vector | 摘要 | encoder 最後的 hidden state。大小固定。就是 attention 要解決的瓶頸。 |
| Teacher forcing | 使用真實 token | 訓練時餵入真實的前一個 token。讓學習穩定。 |
| Exposure bias | 訓練與測試的落差 | 用真實 token 訓練的模型，從沒練習過從自己的錯誤中恢復。 |
| Beam search | 更好的解碼 | 每一步讓前 k 個局部序列保持存活，而不是貪婪地直接決定。 |

## 延伸閱讀 (Further Reading)

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — 原始的 seq2seq 論文，四頁。
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — 引入了 GRU 和 encoder-decoder 的框架。
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 論文，讀完這一課後立刻去讀。
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — 可以動手建立的 seq2seq 加 attention 程式碼。
