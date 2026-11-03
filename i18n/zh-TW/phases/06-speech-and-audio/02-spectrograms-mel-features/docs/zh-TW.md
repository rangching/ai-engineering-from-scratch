# Spectrograms、Mel Scale 與音訊特徵 (Spectrograms, Mel Scale & Audio Features)

> 神經網路不擅長直接吃原始 waveform，它們吃 spectrogram，吃 mel spectrogram 更好。2026 年每一個 ASR、TTS 和音訊分類器，成敗都繫於這個前處理的選擇。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 01 (Audio Fundamentals)
**Time:** ~45 分鐘

## 問題 (The Problem)

拿一段 10 秒、16 kHz 的音訊，那是 160,000 個 float，全都落在 `[-1, 1]`，跟「狗叫」或「cat 這個字」這種標籤幾乎完全不相關。原始 waveform 裡有資訊，但形式讓模型很難直接抽取。兩個相同的音素，相隔 100 ms 說出來，原始 samples 會完全不同。

Spectrogram 解決了這件事。它把人類感知會忽略的時間細節（微秒等級的抖動）壓縮掉，同時保留感知會注意的結構（哪些頻率有能量，時間窗約 10–25 ms）。

Mel spectrogram 更進一步。人類對音高的感知是對數式的：100 Hz 對 200 Hz 聽起來「距離相同」於 1000 Hz 對 2000 Hz。mel scale 把頻率軸扭曲成符合這種感知。從 2010 到 2026 年，mel-scaled spectrogram 是語音機器學習中最重要的單一特徵。

## 核心概念 (The Concept)

![Waveform to STFT to mel spectrogram to MFCC ladder](../assets/mel-features.svg)

**STFT (Short-Time Fourier Transform)。** 把 waveform 切成互相重疊的 frames（典型設定：25 ms window、10 ms hop，在 16 kHz 下就是 400 samples / 160 samples）。每個 frame 乘上 window function（預設用 Hann；Hamming 的取捨略有不同），再對每個 frame 做 FFT。把各 frame 的 magnitude spectrum 疊成形狀 `(n_frames, n_freq_bins)` 的矩陣，這就是你的 spectrogram。

**Log-magnitude。** 原始 magnitude 橫跨 5–6 個數量級。取 `log(|X| + 1e-6)` 或 `20 * log10(|X|)` 來壓縮動態範圍。每個正式環境的 pipeline 都用 log-magnitude，而不是原始 magnitude。

**Mel scale。** 頻率 `f`（Hz）對應到 mel `m` 的公式為 $m = 2595 \cdot \log_{10}(1 + f / 700)$。1 kHz 以下大致呈線性，以上大致呈對數。涵蓋 0–8 kHz 的 80 個 mel bins 是標準的 ASR 輸入。

**Mel filterbank。** 一組在 mel scale 上等距排列的三角形 filter。每個 filter 是相鄰 FFT bins 的加權和。把 STFT magnitude 乘上 filterbank 矩陣，一次 matmul 就得到 mel spectrogram。

**Log-mel spectrogram。** `log(mel_spec + 1e-10)`。Whisper 的輸入、Parakeet 的輸入、SeamlessM4T 的輸入，也是 2026 年通用的音訊前端。

**MFCCs。** 對 log-mel spectrogram 做 DCT（type II），保留前 13 個係數。這樣可以去相關並進一步壓縮。這是 2015 年前的主流特徵，之後 CNN/Transformer 直接吃 log-mels 就追上了。MFCC 仍用於 speaker recognition（x-vectors、ECAPA）。

**解析度取捨。** FFT 越大，頻率解析度越好，但時間解析度越差。25 ms / 10 ms 是音訊 ML 的預設；音樂用 50 ms / 12.5 ms；瞬態偵測（鼓擊、爆破音）用 5 ms / 2 ms。

```figure
spectrogram-window
```

## 動手做 (Build It)

### 步驟 1：把 waveform 切成 frames

```python
def frame(signal, frame_len, hop):
    n = 1 + (len(signal) - frame_len) // hop
    return [signal[i * hop : i * hop + frame_len] for i in range(n)]
```

一段 10 秒、16 kHz 的音訊，用 `frame_len=400, hop=160` 會得到 998 個 frames。

### 步驟 2：Hann window

```python
import math

def hann(N):
    return [0.5 * (1 - math.cos(2 * math.pi * n / (N - 1))) for n in range(N)]
```

在 FFT 之前逐元素相乘。這能消除因為在非零端點截斷而造成的 spectral leakage。

### 步驟 3：STFT magnitude

```python
def stft_magnitude(signal, frame_len=400, hop=160):
    win = hann(frame_len)
    frames = frame(signal, frame_len, hop)
    return [magnitudes(dft([w * s for w, s in zip(win, f)])) for f in frames]
```

正式環境用 `torch.stft` 或 `librosa.stft`（FFT 支援、向量化）。這裡的迴圈只是為了教學，在 `code/main.py` 裡跑短音訊。

### 步驟 4：Mel filterbank

```python
def hz_to_mel(f):
    return 2595.0 * math.log10(1.0 + f / 700.0)

def mel_to_hz(m):
    return 700.0 * (10 ** (m / 2595.0) - 1)

def mel_filterbank(n_mels, n_fft, sr, fmin=0, fmax=None):
    fmax = fmax or sr / 2
    mels = [hz_to_mel(fmin) + (hz_to_mel(fmax) - hz_to_mel(fmin)) * i / (n_mels + 1)
            for i in range(n_mels + 2)]
    hzs = [mel_to_hz(m) for m in mels]
    bins = [int(h * n_fft / sr) for h in hzs]
    fb = [[0.0] * (n_fft // 2 + 1) for _ in range(n_mels)]
    for m in range(n_mels):
        for k in range(bins[m], bins[m + 1]):
            fb[m][k] = (k - bins[m]) / max(1, bins[m + 1] - bins[m])
        for k in range(bins[m + 1], bins[m + 2]):
            fb[m][k] = (bins[m + 2] - k) / max(1, bins[m + 2] - bins[m + 1])
    return fb
```

80 個 mels 涵蓋 0–8 kHz，搭配 `n_fft=400`，會得到 `(80, 201)` 的矩陣。把 `(n_frames, 201)` 的 STFT magnitude 乘上它的轉置，就得到 `(n_frames, 80)` 的 mel spectrogram。

### 步驟 5：Log-mel

```python
def log_mel(mel_spec, eps=1e-10):
    return [[math.log(max(v, eps)) for v in frame] for frame in mel_spec]
```

常見的替代做法：`librosa.power_to_db`（以參考值正規化的 dB）、`10 * log10(power + eps)`。Whisper 用的是比較複雜的 clip + normalize 流程（見 Whisper 的 `log_mel_spectrogram`）。

### 步驟 6：MFCCs

```python
def dct_ii(x, n_coeffs):
    N = len(x)
    return [
        sum(x[n] * math.cos(math.pi * k * (2 * n + 1) / (2 * N)) for n in range(N))
        for k in range(n_coeffs)
    ]
```

對每個 log-mel frame 做 DCT，保留前 13 個係數，就是你的 MFCC 矩陣。第一個係數通常會被丟掉（它代表整體能量）。

## 實際使用 (Use It)

2026 年的工具組合：

| 任務 | 特徵 |
|------|----------|
| ASR (Whisper, Parakeet, SeamlessM4T) | 80 log-mels、10 ms hop、25 ms window |
| TTS acoustic model (VITS, F5-TTS, Kokoro) | 80 mels、5–12 ms hop，以便精細控制時間 |
| 音訊分類 (AST, PANNs, BEATs) | 128 log-mels、10 ms hop |
| Speaker embedding (ECAPA-TDNN, WavLM) | 80 log-mels 或 raw-waveform SSL |
| 音樂 (MusicGen, Stable Audio 2) | EnCodec discrete tokens（不用 mels） |
| Keyword spotting | 微型裝置用 40 MFCCs |

經驗法則：**只要不是做音樂，就從 80 log-mels 開始**。任何偏離都要自己舉證。

## 2026 年仍然會出現的陷阱

- **Mel 數量不一致。** 訓練用 80 mels，推論用 128 mels。會悄悄失敗。請在兩端都記錄特徵的 shape。
- **上游 sample rate 不一致。** 用 22.05 kHz 算出的 mels 和 16 kHz 的看起來不同。請在特徵化*之前*先固定 SR。
- **dB 與 log 搞混。** Whisper 預期的是 log-mel，不是 dB-mel。有些 HF pipeline 會自動偵測，你自己寫的程式不會。
- **Normalization 飄移。** 訓練時用每段語音各自正規化，推論時卻用全域正規化。這是會讓 WER 翻倍的正式環境 bug。
- **Padding 造成的洩漏。** 在音訊結尾補零，會讓尾端 frames 出現平坦的頻譜。請對稱 padding 或用複製方式補。

## 交付 (Ship It)

存成 `outputs/skill-feature-extractor.md`。這個 skill 會針對指定的模型目標，挑選特徵類型、mel 數量、frame/hop 與 normalization。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會合成一個 chirp（頻率從 200 掃到 4000 Hz），並印出每個 frame 的 argmax mel bin。（可選）把它畫出來，確認和掃頻吻合。
2. **中等。** 用 `n_mels` 取 `{40, 80, 128}`、`frame_len` 取 `{200, 400, 800}` 重新執行。量測沿時間軸的尖峰頻寬。哪個組合最能解析這個 chirp？
3. **困難。** 實作 `power_to_db`，並用 AudioMNIST 上的小型 CNN 分類器，比較 ASR 準確率：(a) 原始 log-mel，(b) `ref=max` 的 dB-mel，(c) MFCC-13 + delta + delta-delta。回報 top-1 accuracy。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Frame | 一個切片 | 送進一次 FFT 的 25 ms waveform 片段。 |
| Hop | 步幅 | 相鄰 frames 之間相隔的 samples 數；10 ms 是 ASR 預設。 |
| Window | Hann/Hamming 那個東西 | 逐點相乘，把 frame 邊緣逐漸壓到零。 |
| STFT | spectrogram 產生器 | 分 frame、加 window 的 FFT；產出時間 × 頻率矩陣。 |
| Mel | 被扭曲的頻率 | 對數感知尺度；`m = 2595·log10(1 + f/700)`。 |
| Filterbank | 那個矩陣 | 把 STFT 投影到 mel bins 的三角形 filters。 |
| Log-mel | Whisper 的輸入 | `log(mel_spec + eps)`；2026 年的標準做法。 |
| MFCC | 老派特徵 | log-mel 的 DCT；13 個係數，已去相關。 |

## 延伸閱讀 (Further Reading)

- [Davis, Mermelstein (1980). Comparison of parametric representations for monosyllabic word recognition](https://ieeexplore.ieee.org/document/1163420)：MFCC 的論文。
- [Stevens, Volkmann, Newman (1937). A Scale for the Measurement of the Psychological Magnitude Pitch](https://pubs.aip.org/asa/jasa/article-abstract/8/3/185/735757/)：最早的 mel scale。
- [OpenAI — Whisper source, log_mel_spectrogram](https://github.com/openai/whisper/blob/main/whisper/audio.py)：請讀這份參考實作。
- [librosa feature extraction docs](https://librosa.org/doc/latest/api/feature.html)：`mfcc`、`melspectrogram` 以及 hop/window 的參考文件。
- [NVIDIA NeMo — audio preprocessing](https://docs.nvidia.com/deeplearning/nemo/user-guide/docs/en/main/asr/asr_all.html#featurizers)：Parakeet + Canary 模型的大規模正式環境 pipeline。
