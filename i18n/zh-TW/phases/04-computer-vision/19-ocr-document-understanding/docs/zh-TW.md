# OCR 與文件理解 (OCR & Document Understanding)

> OCR 是三階段的 pipeline：偵測文字 box、辨識字元，然後排版。每個現代 OCR 系統都會重新排列或合併這些階段。

**Type:** Learn + Use
**Languages:** Python
**Prerequisites:** Phase 4 Lesson 06 (Detection), Phase 7 Lesson 02 (Self-Attention)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 追蹤古典的 OCR pipeline（detect -> recognise -> layout）以及現代的端到端替代方案（Donut、Qwen-VL-OCR）
- 為 sequence-to-sequence 的 OCR 訓練實作 CTC (Connectionist Temporal Classification) loss
- 不經訓練，直接使用 PaddleOCR 或 EasyOCR 做正式環境的文件解析
- 區分 OCR、版面解析 (layout parsing) 與文件理解，並為每個任務挑選正確的工具

## 問題 (The Problem)

滿是文字的影像到處都是：收據、發票、證件、掃描的書籍、表單、白板、招牌、螢幕截圖。從中萃取結構化資料，不只是字元，而是「這是總金額」，是價值最高的應用視覺問題之一。

這個領域分成三個技能層：

1. **OCR 本身**：把像素變成文字。
2. **版面解析 (Layout parsing)**：把 OCR 輸出分組成區域（標題、內文、表格、頁首）。
3. **文件理解 (Document understanding)**：從版面中萃取結構化欄位（"invoice_total = $42.50"）。

每一層都有古典與現代的做法，而「我想從影像拿到文字」與「我需要這張收據的總金額」之間的差距，比大多數團隊意識到的更大。

## 核心概念 (The Concept)

### 古典 pipeline (The classical pipeline)

```mermaid
flowchart LR
    IMG["Image"] --> DET["Text detection<br/>(DB, EAST, CRAFT)"]
    DET --> BOX["Word/line<br/>bounding boxes"]
    BOX --> CROP["Crop each region"]
    CROP --> REC["Recognition<br/>(CRNN + CTC)"]
    REC --> TXT["Text strings"]
    TXT --> LAY["Layout<br/>ordering"]
    LAY --> OUT["Reading-order text"]

    style DET fill:#dbeafe,stroke:#2563eb
    style REC fill:#fef3c7,stroke:#d97706
    style OUT fill:#dcfce7,stroke:#16a34a
```

- **Text detection（文字偵測）** 產生每一行或每個字的四邊形。
- **Recognition（辨識）** 把每個區域裁成固定高度，執行 CNN + BiLSTM + CTC 產生字元序列。
- **Layout（版面）** 重建閱讀順序（拉丁文是由上到下、由左到右；阿拉伯文、日文則不同）。

### 一段話講完 CTC (CTC in one paragraph)

OCR 辨識是從固定長度的 feature map 產生可變長度的序列。CTC（Graves et al., 2006）讓你不需要字元層級的對齊就能訓練。模型在每個時間步輸出 (詞彙 + blank) 上的分布；CTC loss 對所有對齊方式做邊際化，這些對齊方式在合併重複並移除 blank 之後，會化簡為目標文字。

```
raw output: "h h h _ _ e e l l _ l l o _ _"
after merge repeats and remove blanks: "hello"
```

CTC 是 CRNN 在 2015 年能夠運作的原因，到 2026 年仍然用來訓練大多數正式環境的 OCR 模型。

### 現代的端到端模型 (Modern end-to-end models)

- **Donut**（Kim et al., 2022）：ViT encoder + 文字 decoder；讀入影像，直接輸出 JSON。沒有文字 detector，也沒有版面模組。
- **TrOCR**：ViT + transformer decoder，用於行層級的 OCR。
- **Qwen-VL-OCR / InternVL**：針對 OCR 任務 fine-tune 的完整視覺-語言模型；2026 年在複雜文件上準確率最佳。
- **PaddleOCR**：古典的 DB + CRNN pipeline，包成成熟的正式環境套件；仍是開源界的主力。

端到端模型需要更多資料和算力，但跳過了多階段 pipeline 的誤差累積。

### 版面解析 (Layout parsing)

對結構化文件，執行版面模型（LayoutLMv3，或在 DocLayNet 資料集上訓練的 detector），為每個區域貼上標籤：Title、Paragraph、Figure、Table、Footnote。接著閱讀順序就變成「依版面順序走過各區域，並串接起來」。

對表單，使用 **Key-Value 萃取** 模型（視覺豐富的文件用 Donut，單純掃描用 LayoutLMv3）。它們接收影像 + 偵測到的文字 + 位置，並預測結構化的 key-value 配對。

### 評估指標 (Evaluation metrics)

- **Character Error Rate (CER)**：Levenshtein 距離 / 參考答案長度。越低越好。正式環境目標：乾淨掃描下 < 2%。
- **Word Error Rate (WER)**：同樣的東西，但在詞的層級。
- **結構化欄位的 F1**：用於 key-value 任務；衡量 `{invoice_total: 42.50}` 是否正確出現。
- **JSON 的 edit distance**：用於端到端的文件解析；Donut 論文引入了正規化的 tree edit distance。

```figure
cv3-ctc-collapse
```

## 動手做 (Build It)

### 步驟 1：CTC loss + greedy decoder (Step 1: CTC loss + greedy decoder)

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


def ctc_loss(log_probs, targets, input_lengths, target_lengths, blank=0):
    """
    log_probs:      (T, N, C) log-softmax over vocab including blank at index 0
    targets:        (N, S) int targets (no blanks)
    input_lengths:  (N,) per-sample time steps used
    target_lengths: (N,) per-sample target length
    """
    return F.ctc_loss(log_probs, targets, input_lengths, target_lengths,
                      blank=blank, reduction="mean", zero_infinity=True)


def greedy_ctc_decode(log_probs, blank=0):
    """
    log_probs: (T, N, C) log-softmax
    returns: list of index sequences (blanks removed, repeats merged)
    """
    preds = log_probs.argmax(dim=-1).transpose(0, 1).cpu().tolist()
    out = []
    for seq in preds:
        decoded = []
        prev = None
        for idx in seq:
            if idx != prev and idx != blank:
                decoded.append(idx)
            prev = idx
        out.append(decoded)
    return out
```

`F.ctc_loss` 在可用時會使用高效的 CuDNN 實作。Greedy decoder 比 beam search 簡單，通常與它的 CER 相差在 1% 以內。

### 步驟 2：小型 CRNN 辨識器 (Step 2: Tiny CRNN recogniser)

用於單行 OCR 的最小 CNN + BiLSTM。

```python
class TinyCRNN(nn.Module):
    def __init__(self, vocab_size=40, hidden=128, feat=32):
        super().__init__()
        self.cnn = nn.Sequential(
            nn.Conv2d(1, feat, 3, 1, 1), nn.BatchNorm2d(feat), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat, feat * 2, 3, 1, 1), nn.BatchNorm2d(feat * 2), nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(feat * 2, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
            nn.Conv2d(feat * 4, feat * 4, 3, 1, 1), nn.BatchNorm2d(feat * 4), nn.ReLU(inplace=True),
            nn.MaxPool2d((2, 1)),
        )
        self.rnn = nn.LSTM(feat * 4, hidden, bidirectional=True, batch_first=True)
        self.head = nn.Linear(hidden * 2, vocab_size)

    def forward(self, x):
        # x: (N, 1, H, W)
        f = self.cnn(x)                # (N, C, H', W')
        f = f.mean(dim=2).transpose(1, 2)  # (N, W', C)
        h, _ = self.rnn(f)
        return F.log_softmax(self.head(h).transpose(0, 1), dim=-1)  # (W', N, vocab)
```

固定高度的輸入（CNN 會把高度 max-pool 到 1）。寬度是 CTC 的時間維度。

### 步驟 3：合成 OCR (Step 3: Synthetic OCR)

產生白底黑字的字串，用於端到端的煙霧測試。每個字元變成一欄六根長條，用二進位拼出它在詞彙中的 index。影像必須帶有字元的身分：如果每個字元都是同一個空白方塊，模型就無法分辨 `a` 和 `x`，loss 就會停滯。

```python
import numpy as np

GLYPHS = "0123456789abcdefghijklmnopqrstuvwxyz"

def synthetic_line(text, height=32, char_width=16):
    W = char_width * len(text)
    img = np.ones((height, W), dtype=np.float32)
    for i, c in enumerate(text):
        x = i * char_width
        code = GLYPHS.find(c) + 1
        for bit in range(6):
            if code >> bit & 1:
                y = 4 + 4 * bit
                img[y:y + 3, x + 2:x + char_width - 2] = 0.0
    return img


def build_batch(strings, vocab):
    H = 32
    W = 16 * max(len(s) for s in strings)
    imgs = np.ones((len(strings), 1, H, W), dtype=np.float32)
    target_lengths = []
    targets = []
    for i, s in enumerate(strings):
        imgs[i, 0, :, :16 * len(s)] = synthetic_line(s)
        ids = [vocab.index(c) for c in s]
        targets.extend(ids)
        target_lengths.append(len(ids))
    return torch.from_numpy(imgs), torch.tensor(targets), torch.tensor(target_lengths)


vocab = ["_"] + list("0123456789abcdefghijklmnopqrstuvwxyz")
imgs, targets, lengths = build_batch(["hello", "world"], vocab)
print(f"images: {imgs.shape}   targets: {targets.shape}   lengths: {lengths.tolist()}")
```

真正的 OCR 資料集會加上字型、雜訊、旋轉、模糊和顏色。上面的 pipeline 完全相同。

### 步驟 4：訓練草稿 (Step 4: Training sketch)

```python
model = TinyCRNN(vocab_size=len(vocab))
opt = torch.optim.Adam(model.parameters(), lr=1e-3)

for step in range(400):
    strings = ["abc" + str(step % 10)] * 4 + ["xyz" + str((step + 1) % 10)] * 4
    imgs, targets, target_lens = build_batch(strings, vocab)
    log_probs = model(imgs)  # (W', 8, vocab)
    input_lens = torch.full((8,), log_probs.size(0), dtype=torch.long)
    loss = ctc_loss(log_probs, targets, input_lens, target_lens, blank=0)
    opt.zero_grad(); loss.backward(); opt.step()
```

在這個簡單的合成資料上，loss 應該在 400 個 step 內從兩位數降到約 0.3。`code/main.py` 用自己的字串集執行同樣的迴圈，然後解碼三個測試字串。

## 實際使用 (Use It)

三條正式環境的路徑：

- **PaddleOCR**：成熟、快速、支援多語言。一行用法：`paddleocr.PaddleOCR(lang="en").ocr(image_path)`。
- **EasyOCR**：Python 原生、多語言、PyTorch backbone。
- **Tesseract**：古典；當模型表現不佳時，對老舊的掃描文件仍然有用。

對端到端的文件解析，使用 Donut 或 VLM：

```python
from transformers import DonutProcessor, VisionEncoderDecoderModel

processor = DonutProcessor.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
model = VisionEncoderDecoderModel.from_pretrained("naver-clova-ix/donut-base-finetuned-cord-v2")
```

對收據、發票和有重複結構的表單，fine-tune Donut。對任意文件或需要推理的 OCR，像 Qwen-VL-OCR 這樣的 VLM 是目前的預設選擇。

## 交付 (Ship It)

這堂課會產出：

- `outputs/prompt-ocr-stack-picker.md`：一個 prompt，根據文件類型、語言與結構，挑選 Tesseract / PaddleOCR / Donut / VLM-OCR。
- `outputs/skill-ctc-decoder.md`：一個 skill，從零寫出 greedy 與 beam-search 的 CTC decoder，包含長度 normalisation。

## 練習 (Exercises)

1. **（簡單）** 在 5 位數隨機數字字串上訓練 TinyCRNN 500 個 step。回報在保留集上的 CER。
2. **（中等）** 把 greedy decoding 換成 beam search（beam_width=5）。回報 CER 的差異。Beam search 在哪些輸入上獲勝？
3. **（困難）** 在一組 20 張收據上使用 PaddleOCR，萃取品項明細，並針對 {item_name, price} 配對，對照手動標註的 ground truth 計算 F1。

## 關鍵術語 (Key Terms)

| 術語 | 大家常說的 | 實際意義 |
|------|----------------|----------------------|
| OCR | 「從像素取得文字」 | 把影像區域轉成字元序列 |
| CTC | 「不需對齊的 loss」 | 不需要逐時間步標籤就能訓練序列模型的 loss；對各種對齊方式做邊際化 |
| CRNN | 「經典 OCR 模型」 | Conv 特徵萃取器 + BiLSTM + CTC；2015 年的 baseline，至今仍在正式環境使用 |
| Donut | 「端到端 OCR」 | ViT encoder + 文字 decoder；直接從影像輸出 JSON |
| Layout parsing | 「找出區域」 | 偵測並標記文件中的 Title/Table/Figure/Paragraph 區域 |
| Reading order | 「文字順序」 | 把辨識出的區域排成句子的順序；對拉丁文很簡單，對混合版面並不簡單 |
| CER / WER | 「錯誤率」 | 以字元或詞為粒度的 Levenshtein 距離 / 參考長度 |
| VLM-OCR | 「會讀字的 LLM」 | 為 OCR 任務訓練或 prompt 的視覺-語言模型；目前在複雜文件上的 SOTA |

## 延伸閱讀 (Further Reading)

- [CRNN (Shi et al., 2015)](https://arxiv.org/abs/1507.05717)：原始的 CNN+RNN+CTC 架構
- [CTC (Graves et al., 2006)](https://www.cs.toronto.edu/~graves/icml_2006.pdf)：原始的 CTC 論文；密集地塞滿了演算法概念
- [Donut (Kim et al., 2022)](https://arxiv.org/abs/2111.15664)：不需要 OCR 的文件理解 transformer
- [PaddleOCR](https://github.com/PaddlePaddle/PaddleOCR)：開源的正式環境 OCR 技術堆疊
