# 音訊分類 (Audio Classification) — 從 MFCC 上的 k-NN 到 AST 和 BEATs

> 從「狗叫 vs 警報器」到「這是哪種語言」，都是音訊分類。特徵是 mels，架構每十年換一次，評估指標始終是 AUC、F1 和每個類別的 recall。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 3 · 06 (CNNs), Phase 5 · 08 (CNNs & RNNs for Text)
**Time:** ~75 分鐘

## 問題 (The Problem)

你拿到一段 10 秒的音訊，想知道：「這是什麼？」可能是都市聲音（警報器、鑽孔、狗）、語音指令（yes/no/stop）、語言辨識（en/es/ar）、說話者情緒（生氣/中性），或是環境聲音（室內/室外、人聲嘈雜）。這些全都是 *audio classification*，而在 2026 年，基準架構已經很成熟：log-mel → CNN 或 Transformer → softmax。

核心難題不在網路，而在資料。音訊資料集有嚴重的類別不平衡、強烈的 domain shift（乾淨 vs 吵雜），還有標籤雜訊（誰決定哪個算「urban babble」、哪個算「restaurant noise」？）。問題的八成在於資料整理、augmentation 和評估，而不是把 CNN 換成 Transformer。

## 核心概念 (The Concept)

![Audio classification ladder: k-NN on MFCCs to AST to BEATs](../assets/audio-classification.svg)

**MFCC 上的 k-NN（1990 年代的基準）。** 把每段音訊的 MFCC 攤平，與有標籤的資料庫計算 cosine similarity，回傳前 K 名的多數決。在乾淨的小資料集（Speech Commands、ESC-50）上出奇地強，而且不需要 GPU。

**Log-mel 上的 2D CNN（2015-2019）。** 把 `(T, n_mels)` 的 log-mel 當成圖片，套用 ResNet-18 或 VGG 風格的網路，沿時間軸做 global mean pool，再對類別做 softmax。到了 2026 年，這仍是多數 kaggle 競賽的基準。

**Audio Spectrogram Transformer，AST（2021-2024）。** 把 log-mel 切成 patch（例如 16×16），加上 position embeddings，餵給 ViT。在 AudioSet 上是監督式學習的 state of the art（mAP 0.485）。

**BEATs 與 WavLM-base（2024-2026）。** 在數百萬小時的資料上做 self-supervised pretraining。只需要你原本所需監督式資料的 1-10%，就能在你的任務上 fine-tune。2026 年，這是非語音音訊的預設起點。BEATs-iter3 在 AudioSet 上比 AST 高 1-2 mAP，運算量卻只有 1/4。

**把 Whisper-encoder 當作凍結的 backbone（2024）。** 取出 Whisper 的 encoder，丟掉 decoder，接上線性分類器。不用任何音訊 augmentation，就能在 language ID 和簡單事件分類上接近 SOTA。這是「免費午餐」基準。

### 類別不平衡才是真正的挑戰

ESC-50：50 個類別，每類 40 段，平衡又簡單。UrbanSound8K：10 個類別，不平衡程度 10:1。AudioSet：632 個類別，長尾達 100,000:1。有效的技巧：

- 訓練時使用平衡取樣（評估時不要）。
- Mixup：把兩段音訊（及其標籤）線性內插，作為 augmentation。
- SpecAugment：遮蔽隨機的時間與頻率帶。簡單，但非常關鍵。

### 評估 (Evaluation)

- 多類別互斥（Speech Commands）：top-1 accuracy、top-5 accuracy。
- 多類別多標籤（AudioSet、UrbanSound 風格）：mean average precision (mAP)。
- 嚴重不平衡：每類別 recall + macro F1。

你應該知道的 2026 年數字：

| Benchmark | Baseline | SOTA 2026 | 來源 |
|-----------|----------|-----------|--------|
| ESC-50 | 82% (AST) | 97.0% (BEATs-iter3) | BEATs paper (2024) |
| AudioSet mAP | 0.485 (AST) | 0.548 (BEATs-iter3) | HEAR leaderboard 2026 |
| Speech Commands v2 | 98% (CNN) | 99.0% (Audio-MAE) | HEAR v2 results |

```figure
mfcc-pipeline
```

## 動手做 (Build It)

### 步驟 1：特徵化

```python
def featurize_mfcc(signal, sr, n_mfcc=13, n_mels=40, frame_len=400, hop=160):
    mag = stft_magnitude(signal, frame_len, hop)
    fb = mel_filterbank(n_mels, frame_len, sr)
    mels = apply_filterbank(mag, fb)
    log = log_transform(mels)
    return [dct_ii(frame, n_mfcc) for frame in log]
```

### 步驟 2：固定長度摘要

```python
def summarize(mfcc_frames):
    n = len(mfcc_frames[0])
    mean = [sum(f[i] for f in mfcc_frames) / len(mfcc_frames) for i in range(n)]
    var = [
        sum((f[i] - mean[i]) ** 2 for f in mfcc_frames) / len(mfcc_frames) for i in range(n)
    ]
    return mean + var
```

簡單卻很強：對時間取 mean + variance，13 個係數的 MFCC 就變成 26 維的固定 embedding。執行瞬間完成。直到 2017 年，它都還能打敗 ESC-50 上最先進的 NN 基準。

### 步驟 3：k-NN

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)) or 1e-12
    nb = math.sqrt(sum(x * x for x in b)) or 1e-12
    return dot / (na * nb)

def knn_classify(q, bank, labels, k=5):
    sims = sorted(range(len(bank)), key=lambda i: -cosine(q, bank[i]))[:k]
    votes = Counter(labels[i] for i in sims)
    return votes.most_common(1)[0][0]
```

### 步驟 4：升級為 log-mel 上的 CNN

用 PyTorch：

```python
import torch.nn as nn

class AudioCNN(nn.Module):
    def __init__(self, n_mels=80, n_classes=50):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.head = nn.Linear(128, n_classes)

    def forward(self, x):  # x: (B, 1, T, n_mels)
        return self.head(self.body(x).flatten(1))
```

3M 個參數。用單張 RTX 4090 在 ESC-50 上訓練約 10 分鐘，準確率 80% 以上。

### 步驟 5：Fine-tune 預訓練的 audio transformer（以 AST 為例）

```python
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ext = ASTFeatureExtractor.from_pretrained("MIT/ast-finetuned-audioset-10-10-0.4593")
model = ASTForAudioClassification.from_pretrained(
    "MIT/ast-finetuned-audioset-10-10-0.4593",
    num_labels=50,
    ignore_mismatched_sizes=True,
)

inputs = ext(audio, sampling_rate=16000, return_tensors="pt")
logits = model(**inputs).logits
```

這個範例從 Hub 載入 AST 來 fine-tune。2026 年的預設 BEATs 不在 Hugging Face Hub 上：請從 [microsoft/unilm 的 BEATs release](https://github.com/microsoft/unilm/tree/master/beats) 下載 checkpoint，並用該 repo 的 `BEATs` 和 `BEATsConfig` 類別載入；fine-tuning 迴圈的結構維持不變。

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 從這裡開始 |
|-----------|-----------|
| 極小資料集（<1000 段） | MFCC 平均值上的 k-NN（你的基準）+ 音訊 augmentation |
| 中型資料集（1K–100K） | BEATs 或 AST fine-tune |
| 大型資料集（>100K） | 從頭訓練，或 fine-tune Whisper-encoder |
| 即時、邊緣裝置 | 40-MFCC CNN，量化成 int8（KWS 風格） |
| 多標籤 (AudioSet) | BEATs-iter3，搭配 BCE loss + mixup + SpecAugment |
| Language ID | MMS-LID、SpeechBrain VoxLingua107 基準 |

決策規則：**從凍結的 backbone 開始，而不是全新的模型**。fine-tune BEATs 的 head，幾小時就能拿到 95% 的 SOTA，不用花好幾週。

## 交付 (Ship It)

存成 `outputs/skill-classifier-designer.md`。針對指定的音訊分類任務，挑選架構、augmentation、類別平衡策略與評估指標。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會在 4 類的合成資料集（不同音高的純音）上訓練 k-NN MFCC 基準。回報 confusion matrix。
2. **中等。** 把 `summarize` 換成 [mean, var, skew, kurtosis]。在同一個合成資料集上，4-moment pooling 有沒有勝過 mean+var？
3. **困難。** 用 `torchaudio` 在 ESC-50 的 fold 1 上訓練 2D CNN。回報 5-fold cross-validation accuracy。加入 SpecAugment（time mask = 20、freq mask = 10），並回報差異。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| AudioSet | 音訊界的 ImageNet | Google 的資料集，200 萬段、632 類、弱標註的 YouTube 音訊。 |
| ESC-50 | 小型分類 benchmark | 環境聲音，50 類 × 每類 40 段。 |
| AST | Audio Spectrogram Transformer | 作用在 log-mel patch 上的 ViT；2021 年的 SOTA。 |
| BEATs | Self-supervised 音訊模型 | Microsoft 的模型，截至 2026 年 iter3 在 AudioSet 上領先。 |
| Mixup | 成對的 augmentation | `x = λ·x1 + (1-λ)·x2; y = λ·y1 + (1-λ)·y2`。 |
| SpecAugment | 以遮蔽為基礎的 augmentation | 把 spectrogram 中隨機的時間與頻率帶歸零。 |
| mAP | 主要的多標籤指標 | 跨類別與閾值的 mean average precision。 |

## 延伸閱讀 (Further Reading)

- [Gong, Chung, Glass (2021). AST: Audio Spectrogram Transformer](https://arxiv.org/abs/2104.01778)：2021–2024 年的標竿架構。
- [Chen et al. (2022, rev. 2024). BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/abs/2212.09058)：2024 年之後的預設選擇。
- [Park et al. (2019). SpecAugment](https://arxiv.org/abs/1904.08779)：主流的音訊 augmentation。
- [Piczak (2015). ESC-50 dataset](https://github.com/karolpiczak/ESC-50)：沿用至今的 50 類 benchmark。
- [Gemmeke et al. (2017). AudioSet](https://research.google.com/audioset/)：632 類的 YouTube 分類體系，至今仍是黃金標準。
