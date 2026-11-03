# 說話者辨識與驗證 (Speaker Recognition & Verification)

> ASR 問的是「他們說了什麼？」說話者辨識問的是「是誰說的？」數學看起來一樣，都是 embeddings 加 cosine，但每個正式環境的決策都取決於一個 EER 數字。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 22 (Embedding Models)
**Time:** ~45 分鐘

## 問題 (The Problem)

使用者說出一句通關密語。你想知道：這個人真的是他所宣稱的那個人嗎（*verification*，1:1）？或者他是註冊名單中的第一個人嗎（*identification*，1:N）？又或者兩者都不是，是個未知的說話者（*open-set*）？

2018 年以前：GMM-UBM + i-vectors。EER 還算合理，但對通道改變（手機 vs 筆電）和情緒很脆弱。2018–2022：x-vectors（用 angular margin 訓練的 TDNN backbone）。2022 年之後：ECAPA-TDNN 和 WavLM-large embeddings。到了 2026 年，這個領域由三個模型和一個指標主導。

這個指標就是 **EER**，Equal Error Rate。調整決策閾值，使 False Accept Rate = False Reject Rate，兩者交會處就是 EER。每篇論文、每個排行榜、每次採購評估都用它。

## 核心概念 (The Concept)

![Enrollment + verification pipeline with embedding + cosine + EER](../assets/speaker-verification.svg)

**Pipeline。** Enrollment：錄下目標說話者 5–30 秒的音訊，計算固定維度的 embedding（ECAPA-TDNN 是 192 維，WavLM-large 是 256 維）。Verification：取得測試語音的 embedding，計算 cosine similarity，與閾值比較。

**ECAPA-TDNN（2020，2026 年仍是主流）。** Emphasized Channel Attention, Propagation and Aggregation - Time-Delay Neural Network。帶有 squeeze-excitation 的 1D conv blocks、multi-head attention pooling，最後接一個線性層輸出 192 維。在 VoxCeleb 1+2（2,700 位說話者、110 萬句語音）上，用 Additive Angular Margin loss (AAM-softmax) 訓練。

**WavLM-SV（2022 年之後）。** 用 AAM loss fine-tune 預訓練的 WavLM-large SSL backbone。品質較高但較慢，300 MB 以上 vs 15 MB。

**x-vector（基準）。** TDNN + statistics pooling。經典做法；在 CPU / 邊緣裝置上仍然有用。

**AAM-softmax。** 標準 softmax，但在角度空間為正確類別加上 margin `m`：`cos(θ + m)`。強制拉開類別間的角度。典型值 `m=0.2`，scale `s=30`。

### 評分 (Scoring)

- 註冊與測試 embeddings 之間的 **Cosine**。以閾值做決策。
- **PLDA (Probabilistic LDA)。** 把 embeddings 投影到一個潛在空間，在那裡同一說話者 vs 不同說話者有封閉形式的 likelihood ratio。疊加在 cosine 之上，EER 可再降 10–20%。2020 年前是標準做法；現在只用在 closed-set 設定。
- **Score normalization。** `S-norm` 或 `AS-norm`：用一組冒充者 (imposter) cohort 的平均值與標準差，來正規化每個分數。跨領域評估時必不可少。

### 你應該知道的數字 (2026)

| Model | VoxCeleb1-O EER | Params | Throughput (A100) |
|-------|-----------------|--------|-------------------|
| x-vector (classic) | 3.10% | 5 M | 400× RT |
| ECAPA-TDNN | 0.87% | 15 M | 200× RT |
| WavLM-SV large | 0.42% | 316 M | 20× RT |
| Pyannote 3.1 segmentation + embedding | 0.65% | 6 M | 100× RT |
| ReDimNet (2024) | 0.39% | 24 M | 100× RT |

### Diarization

在多人說話的音訊中判斷「誰在何時說話」。Pipeline：VAD → 分段 → 對每段做 embedding → 分群（agglomerative 或 spectral）→ 平滑邊界。現代工具組合：`pyannote.audio` 3.1，把說話者分段 + embedding + 分群包在一次呼叫後面。2026 年 AMI 上的 SOTA DER 約 15%（2022 年是 23%）。

```figure
sp-eer-crossover
```

## 動手做 (Build It)

### 步驟 1：用 MFCC 統計量做玩具 embedding

```python
def embed_mfcc_stats(signal, sr):
    frames = featurize_mfcc(signal, sr, n_mfcc=13)
    mean = [sum(f[i] for f in frames) / len(frames) for i in range(13)]
    std = [
        math.sqrt(sum((f[i] - mean[i]) ** 2 for f in frames) / len(frames))
        for i in range(13)
    ]
    return mean + std  # 26-d
```

離 SOTA 差得遠，只用於教學。`code/main.py` 用它在合成的說話者資料上做概念驗證。

### 步驟 2：Cosine similarity + 閾值

```python
def cosine(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    return dot / (na * nb) if na and nb else 0.0

def verify(enroll, test, threshold=0.75):
    return cosine(enroll, test) >= threshold
```

### 步驟 3：由相似度配對計算 EER

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 1.0, 0.0)  # (fa, fr, threshold)
    for t in thresholds:
        fr = sum(1 for s in same_scores if s < t) / len(same_scores)
        fa = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        if abs(fa - fr) < abs(best[0] - best[1]):
            best = (fa, fr, t)
    return (best[0] + best[1]) / 2, best[2]
```

回傳 (eer, threshold_at_eer)。兩者都要回報。

### 步驟 4：用 SpeechBrain 做正式環境版本

```python
from speechbrain.pretrained import EncoderClassifier

clf = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb")

# enroll: average the embeddings of 3-5 clean samples
enroll = torch.stack([clf.encode_batch(load(x)) for x in enrollment_clips]).mean(0)
# verify
score = clf.similarity(enroll, clf.encode_batch(load("test.wav"))).item()
verdict = score > 0.25   # ECAPA typical threshold; tune on your data
```

### 步驟 5：用 pyannote 做 diarization

```python
from pyannote.audio import Pipeline

pipe = Pipeline.from_pretrained("pyannote/speaker-diarization-3.1")
diarization = pipe("meeting.wav", num_speakers=None)
for turn, _, speaker in diarization.itertracks(yield_label=True):
    print(f"{turn.start:.1f}–{turn.end:.1f}  {speaker}")
```

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 選擇 |
|-----------|------|
| Closed-set 1:1 驗證、邊緣裝置 | ECAPA-TDNN + cosine 閾值 |
| Open-set 驗證、雲端 | WavLM-SV + AS-norm |
| Diarization（會議、podcast） | `pyannote/speaker-diarization-3.1` |
| Anti-spoofing（重放 / deepfake 偵測） | AASIST 或 RawNet2 |
| 微型嵌入式（KWS + 註冊） | Titanet-Small (NeMo) |

## 陷阱 (Pitfalls)

- **通道不一致。** 在 VoxCeleb（網路影片）上訓練的模型 ≠ 電話通話音訊。一定要在目標通道上評估。
- **語音太短。** 測試音訊少於 3 秒時，EER 會急遽惡化。
- **用有雜訊的音訊註冊。** 一次有雜訊的註冊會汙染錨點。至少用 3 個乾淨的樣本，並取平均。
- **跨條件使用固定閾值。** 一定要在目標領域的 held-out dev set 上調整閾值。
- **對未正規化的 embeddings 用 cosine。** 先做 L2-normalize；否則 magnitude 會主導結果。

## 交付 (Ship It)

存成 `outputs/skill-speaker-verifier.md`。挑選模型、註冊流程、閾值調整計畫，以及防詐騙的保護措施。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會建立合成的「說話者」（不同的音調特徵）、進行註冊，並在 100 組配對的試驗清單上計算 EER。
2. **中等。** 對 30 句 VoxCeleb1 語音（5 位說話者 × 每人 6 句）使用 SpeechBrain ECAPA。比較用 cosine 與 PLDA 計算的 EER。
3. **困難。** 用 `pyannote.audio` 建立完整的 enroll → diarize → verify pipeline。在 AMI dev set 上評估 DER。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| EER | 最主要的指標 | False Accept = False Reject 時的閾值。 |
| Verification | 1:1 | 「這是 Alice 嗎？」 |
| Identification | 1:N | 「是誰在說話？」 |
| Open-set | 可能有未知者 | 測試集可能包含未註冊的說話者。 |
| Enrollment | 註冊 | 計算說話者的參考 embedding。 |
| AAM-softmax | 那個 loss | 帶有 additive angular margin 的 softmax；強制拉開群集間距。 |
| PLDA | 經典評分 | Probabilistic LDA；在 embeddings 之上做 likelihood-ratio 評分。 |
| DER | Diarization 指標 | Diarization Error Rate，即漏判 + 誤報 + 混淆。 |

## 延伸閱讀 (Further Reading)

- [Snyder et al. (2018). X-Vectors: Robust DNN Embeddings for Speaker Recognition](https://www.danielpovey.com/files/2018_icassp_xvectors.pdf)：經典的深度 embedding 論文。
- [Desplanques et al. (2020). ECAPA-TDNN](https://arxiv.org/abs/2005.07143)：2020–2026 年的主流架構。
- [Chen et al. (2022). WavLM: Large-Scale Self-Supervised Pre-Training for Full Stack Speech Processing](https://arxiv.org/abs/2110.13900)：用於 SV 與 diarization 的 SSL backbone。
- [Bredin et al. (2023). pyannote.audio 3.1](https://github.com/pyannote/pyannote-audio)：正式環境用的 diarization + embedding 工具組合。
- [VoxCeleb leaderboard (updated 2026)](https://www.robots.ox.ac.uk/~vgg/data/voxceleb/)：各模型目前的 EER 排名。
