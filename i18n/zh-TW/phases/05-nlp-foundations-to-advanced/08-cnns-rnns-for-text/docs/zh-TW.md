# 用於文字的 CNN 與 RNN (CNNs and RNNs for Text)

> 卷積學習 n-gram，遞迴負責記憶。兩者都已被 attention 取代，但在受限的硬體上依然重要。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Time:** ~75 分鐘

## 問題 (The Problem)

TF-IDF 和 Word2Vec 產生的是忽略詞序的扁平向量。建立在它們之上的分類器，分不出 `dog bites man` 和 `man bites dog`。有時候詞序就帶有訊號。

在 transformer 出現之前，有兩個架構家族填補了這個缺口。

**用於文字的卷積網路 (TextCNN)。** 在 word embedding 序列上套用 1D 卷積。寬度 3 的 filter 是可學習的 trigram 偵測器：它跨越三個字並輸出一個分數。堆疊不同寬度（2、3、4、5）來偵測多尺度的模式。再做 max-pool，得到固定大小的表示。扁平、平行、快速。

**遞迴網路 (RNN、LSTM、GRU)。** 一次處理一個 token，維持一個把資訊往前帶的 hidden state。有順序性、帶有記憶、輸入長度彈性。從 2014 到 2017 年主導序列建模，然後 attention 出現了。

這堂課會建立兩者，再點出促成 attention 的失敗點。

## 核心概念 (The Concept)

**TextCNN**（Kim, 2014）。Token 先做 embedding。寬度 `k` 的 1D 卷積在連續的 `k`-gram embedding 上滑動 filter，產生 feature map。對該 map 做 global max-pooling，挑出最強的激活值。把多種 filter 寬度的 max-pool 輸出串接起來，送進分類器 head。

為什麼有效。Filter 就是可學習的 n-gram。Max-pooling 與位置無關，所以 "not good" 不論出現在評論的開頭或中間，都會觸發同一個特徵。三種 filter 寬度、每種 100 個 filter，就得到 300 個學出來的 n-gram 偵測器。訓練是平行的，沒有順序相依。

**RNN。** 在每個時間步 `t`，hidden state 為 `h_t = f(W * x_t + U * h_{t-1} + b)`。`W`、`U`、`b` 在時間上共享。時間 `T` 的 hidden state，是整個前綴的摘要。做分類時，對 `h_1 ... h_T` 做 pooling（max、mean 或 last）。

單純的 RNN 會遇到梯度消失。**LSTM** 加入了 gate，決定要忘記什麼、儲存什麼、輸出什麼，讓長序列中的梯度穩定。**GRU** 把 LSTM 簡化成兩個 gate，表現相近但參數較少。

**雙向 RNN (Bidirectional RNN)** 讓一個 RNN 往前跑、另一個往後跑，再把 hidden state 串接起來。每個 token 的表示都能看到左右兩邊的上下文。對標註任務不可或缺。

```figure
rnn-unroll
```

## 動手做 (Build It)

### 步驟 1：PyTorch 的 TextCNN

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` 把 `[batch, seq_len, embed_dim]` 重塑成 `[batch, embed_dim, seq_len]`，因為 `nn.Conv1d` 把中間的軸當作 channel。無論輸入長度如何，pooling 後的輸出大小都是固定的。

### 步驟 2：LSTM 分類器

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

對序列做 max-pool，而不是取最後一個 state 的 pool。做分類時，max-pooling 通常勝過取最後的 hidden state，因為長序列結尾的資訊往往會主導最後一個 state。

### 步驟 3：梯度消失示範（直覺）

沒有 gating 的單純 RNN 學不到長距離相依。看一個玩具任務：預測 token `A` 有沒有出現在序列的任何位置。如果 `A` 在第 1 個位置，序列長 100 個 token，來自 loss 的梯度得穿過 99 次遞迴權重的乘法才能傳回去。權重小於 1，梯度就消失；大於 1，就爆炸。

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTM 用 **cell state** 修正這點，它貫穿整個網路，只有加法的互動（forget gate 會以乘法縮放它，但梯度仍然沿著這條「高速公路」流動）。GRU 用更少的參數做類似的事。兩者都能讓你在超過 100 步的序列上穩定訓練。

### 步驟 4：為什麼這樣還是不夠

即使有 LSTM，仍有三個問題存在。

1. **順序瓶頸。** 在長度 1000 的序列上訓練 RNN，需要 1000 次串行的 forward/backward 步驟。無法在時間上平行化。
2. **encoder-decoder 設定中固定大小的 context vector。** Decoder 只看得到 encoder 最後的 hidden state，它是整個輸入壓縮後的結果。長輸入會損失細節。第 09 課會直接處理這點。
3. **遠距離相依的準確率天花板。** LSTM 勝過單純 RNN，但仍然很難把特定資訊傳遞超過 200 步。

Attention 解決了這三個問題。Transformer 完全拋棄了遞迴。第 10 課就是轉折點。

## 實際使用 (Use It)

PyTorch 的 `nn.LSTM`、`nn.GRU` 和 `nn.Conv1d` 都已經可以用於正式環境，訓練程式碼是標準寫法。

Hugging Face 提供預訓練的 embedding，你可以把它們接進來當輸入層：

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

符合限制時再使用的檢查清單。

- **邊緣 / 裝置端推論。** 搭配 GloVe embedding 的 TextCNN，比 transformer 小 10 到 100 倍。如果你的部署目標是手機，這就是你的技術棧。
- **串流 / 線上分類。** RNN 一次處理一個 token，transformer 需要完整序列。對即時進來的文字，LSTM 仍然勝出。
- **做 baseline 的小模型。** 在新任務上快速迭代。在 CPU 上 5 分鐘就能訓練出 TextCNN。
- **資料有限的序列標註。** 對 1k 到 10k 個標註句子，BiLSTM-CRF（第 06 課）仍然是正式等級的 NER 架構。

其他情況都交給 transformer。

## 交付 (Ship It)

存成 `outputs/prompt-text-encoder-picker.md`：

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## 練習 (Exercises)

1. **簡單。** 在 3 類的玩具資料集（資料由你自己編）上訓練 TextCNN。驗證多種 filter 寬度 (2, 3, 4) 在平均 F1 上優於單一寬度 (3)。
2. **中等。** 為 LSTM 分類器實作 max-pool、mean-pool 和 last-state pooling。在小型資料集上比較，記錄哪種 pooling 勝出，並推測原因。
3. **困難。** 建立 BiLSTM-CRF NER tagger（結合第 06 課與這一課）。在 CoNLL-2003 上訓練。與第 06 課只用 CRF 的 baseline，以及 BERT fine-tune 比較。回報訓練時間、記憶體與 F1。

## 關鍵術語 (Key Terms)

| 術語 | 一般人怎麼說 | 實際意義 |
|------|-----------------|-----------------------|
| TextCNN | 用於文字的 CNN | 在 word embedding 上堆疊 1D 卷積，並搭配 global max-pool。Kim (2014)。 |
| RNN | 遞迴網路 | 每個時間步更新 hidden state：`h_t = f(W x_t + U h_{t-1})`。 |
| LSTM | 有 gate 的 RNN | 加入 input / forget / output gate 與 cell state。在長序列上能穩定訓練。 |
| GRU | 較簡單的 LSTM | 兩個 gate 而不是三個。準確率相近，參數較少。 |
| Bidirectional | 雙向 | 前向加後向的 RNN 串接。每個 token 都能看到上下文的兩側。 |
| Vanishing gradient | 訓練訊號消失 | 單純 RNN 中重複乘上小於 1 的權重，讓早期步驟的梯度實際上變成零。 |

## 延伸閱讀 (Further Reading)

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — TextCNN 論文，八頁，好讀。
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — LSTM 論文，出乎意料地清楚。
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — 讓所有人都能看懂 LSTM 的那些圖。
