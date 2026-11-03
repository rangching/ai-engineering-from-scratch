# 音訊基礎 (Audio Fundamentals) — Waveform、Sampling、Fourier Transform

> Waveform 是原始訊號，spectrogram 是表示方式，mel features 則是對機器學習友善的形式。現代的 ASR 與 TTS pipeline 都沿著這條階梯往上走，而第一階就是搞懂 sampling 和 Fourier。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Vectors & Matrices), Phase 1 · 14 (Probability Distributions)
**Time:** ~45 分鐘

## 問題 (The Problem)

麥克風輸出的是「壓力對時間」的訊號，而你的神經網路吃的是 tensor。兩者之間有一整套慣例，一旦違反就會產生悄悄出現的 bug：模型訓練看起來正常，但 WER 卻翻倍；或是 TTS 輸出一陣嘶嘶聲；或是 voice cloning 系統記住的是麥克風，而不是說話的人。

語音系統裡的每個 bug，最後都能追到三個問題之一：

1. 資料是用什麼 sample rate 錄的？模型預期的又是什麼？
2. 訊號有沒有 aliasing？
3. 你操作的是原始 samples，還是頻率表示？

這三點搞對了，Phase 6 的其餘內容就很好處理。搞錯了，連 Whisper-Large-v4 也會輸出一堆亂碼。

## 核心概念 (The Concept)

![Waveform, sampling, DFT, and frequency bins visualized](../assets/audio-fundamentals.svg)

**Waveform。** 一個一維的 float 陣列，數值落在 `[-1.0, 1.0]`，用 sample 編號當索引。要換算成秒，就除以 sample rate：`t = n / sr`。一段 16 kHz、10 秒的音訊，就是 160,000 個 float 的陣列。

**Sampling rate (sr)。** 每秒有幾個 samples。2026 年常見的取樣率：

| Rate | 用途 |
|------|-----|
| 8 kHz | 電話、舊式 VOIP。Nyquist 在 4 kHz，會把子音砍掉。ASR 請避免使用。 |
| 16 kHz | ASR 標準。Whisper、Parakeet、SeamlessM4T v2 都吃 16 kHz。 |
| 22.05 kHz | 舊款模型的 TTS vocoder 訓練。 |
| 24 kHz | 現代 TTS（Kokoro、F5-TTS、xTTS v2）。 |
| 44.1 kHz | CD 音訊、音樂。 |
| 48 kHz | 電影、專業音訊、高保真 TTS（VALL-E 2、NaturalSpeech 3）。 |

**Nyquist-Shannon。** 取樣率 `sr` 可以無歧義地表示最高 `sr/2` 的頻率。`sr/2` 這條界線叫做 *Nyquist frequency*。超過 Nyquist 的能量會被 *aliased*，也就是折疊到較低的頻率，進而破壞訊號。downsampling 之前一定要先做 low-pass filter。

**Bit depth。** 16-bit PCM（有號 int16，範圍 ±32,767）是通用的交換格式。音樂用 24-bit，內部 DSP 用 32-bit float。像 `soundfile` 這類函式庫會讀 int16，但提供 `[-1, 1]` 範圍的 float32 陣列。

**Fourier Transform。** 任何有限長度的訊號，都可以寫成不同頻率正弦波的總和。Discrete Fourier Transform (DFT) 對 `N` 個 samples 算出 `N` 個複數係數，每個 frequency bin 一個。`bin k` 對應的頻率是 `k · sr / N` Hz。magnitude 是該頻率的振幅，angle 是相位。

**FFT。** Fast Fourier Transform：當 `N` 是 2 的次方時，計算 DFT 的 `O(N log N)` 演算法。所有音訊函式庫底層都用 FFT。16 kHz 下做 1024-sample 的 FFT，會得到 512 個可用的 frequency bins，涵蓋 0–8 kHz，解析度為 15.6 Hz。

**Framing + window。** 我們不會對整段音訊直接做 FFT，而是把它切成互相重疊的 *frames*（通常 25 ms，hop 為 10 ms），每個 frame 乘上 window function（Hann、Hamming）來消除邊緣的不連續，再對每個 frame 做 FFT。這就是 Short-Time Fourier Transform (STFT)。Lesson 02 會從這裡接著講。

```figure
mel-scale
```

## 動手做 (Build It)

### 步驟 1：讀取音訊並畫出 waveform

`code/main.py` 只用標準函式庫的 `wave` 模組，讓 demo 不需要任何依賴。正式環境你會用 `soundfile` 或 `torchaudio.load`（兩者都回傳 `(waveform, sr)` tuple）：

```python
import soundfile as sf
waveform, sr = sf.read("clip.wav", dtype="float32")  # shape (T,), sr=int
```

### 步驟 2：從基本原理合成正弦波

```python
import math

def sine(freq_hz, sr, seconds, amp=0.5):
    n = int(sr * seconds)
    return [amp * math.sin(2 * math.pi * freq_hz * i / sr) for i in range(n)]
```

440 Hz 的正弦波（標準音 A），在 16 kHz 下持續 1 秒，就是 16,000 個 float。用 `wave.open(..., "wb")` 以 16-bit PCM 編碼寫出。

### 步驟 3：手動計算 DFT

```python
def dft(x):
    N = len(x)
    out = []
    for k in range(N):
        re = sum(x[n] * math.cos(-2 * math.pi * k * n / N) for n in range(N))
        im = sum(x[n] * math.sin(-2 * math.pi * k * n / N) for n in range(N))
        out.append((re, im))
    return out
```

`O(N²)`：拿來驗證 `N=256` 的正確性沒問題，但實際音訊完全不能用。真正的程式會呼叫 `numpy.fft.rfft` 或 `torch.fft.rfft`。

### 步驟 4：找出主要頻率

magnitude 的峰值索引 `k_star` 對應到頻率 `k_star * sr / N`。對 440 Hz 的正弦波執行，峰值應該出現在 bin `440 * N / sr`。

### 步驟 5：示範 aliasing

用 10 kHz 取樣一個 7 kHz 的正弦波（Nyquist = 5 kHz）。7 kHz 的音高於 Nyquist，會折疊成 `10 − 7 = 3 kHz`，所以 FFT 的峰值出現在 3 kHz。這是經典的 aliasing 示範，也是每個 DAC/ADC 都內建 brick-wall low-pass filter 的原因。

## 實際使用 (Use It)

2026 年你實際會用在產品上的工具組合：

| 任務 | 函式庫 | 原因 |
|------|---------|-----|
| 讀寫 WAV/FLAC/OGG | `soundfile`（libsndfile wrapper） | 最快、穩定，回傳 float32。 |
| Resample | `torchaudio.transforms.Resample` 或 `librosa.resample` | 內建正確的 anti-aliasing。 |
| STFT / Mel | `torchaudio` 或 `librosa` | 支援 GPU，屬於 PyTorch 生態系。 |
| 即時串流 | `sounddevice` 或 `pyaudio` | 跨平台的 PortAudio bindings。 |
| 檢查檔案 | `ffprobe` 或 `soxi` | CLI，速度快，會回報 sr/channels/codec。 |

決策規則：**先對齊 sample rate，再處理其他事**。Whisper 預期的是 16 kHz mono float32。丟給它 44.1 kHz 立體聲，你會得到看起來像模型 bug 的亂碼。

## 交付 (Ship It)

存成 `outputs/skill-audio-loader.md`。這個 skill 會幫你檢查輸入音訊是否符合下游模型的預期，不符合時還會正確地 resample。

## 練習 (Exercises)

1. **簡單。** 合成 1 秒、16 kHz 的 220 Hz + 440 Hz + 880 Hz 混合音。執行 DFT，確認預期的 bins 出現三個峰值。
2. **中等。** 用 48 kHz 錄一段 3 秒的 WAV 人聲。先用 `torchaudio.transforms.Resample`（含 anti-aliasing）降到 16 kHz，再用單純的 decimation（每三個 sample 取一個）降到 16 kHz。兩者都做 FFT，aliasing 出現在哪裡？
3. **困難。** 只用 `math` 和步驟 3 的 DFT，從零打造 STFT。frame size 400、hop 160、Hann window。用 `matplotlib.pyplot.imshow` 畫出 magnitude。這就是 Lesson 02 的 spectrogram。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Sample rate | 每秒有幾個 samples | ADC 量測訊號的頻率，單位 Hz。 |
| Nyquist | 能表示的最高頻率 | `sr/2`；超過它的能量會 alias 回低頻。 |
| Bit depth | 每個 sample 的解析度 | `int16` = 65,536 個等級；`float32` = `[-1, 1]` 內 24-bit 的精度。 |
| DFT | 針對序列的 Fourier transform | `N` 個 samples → `N` 個複數頻率係數。 |
| FFT | 快速 DFT | `O(N log N)` 演算法，要求 `N` 為 2 的次方。 |
| Bin | 頻率欄位 | `k · sr / N` Hz；解析度 = `sr / N`。 |
| STFT | spectrogram 的底層 | 分 frame、加 window 後，沿時間做 FFT。 |
| Aliasing | 奇怪的頻率幽靈 | 超過 Nyquist 的能量鏡射到較低的 bins。 |

## 延伸閱讀 (Further Reading)

- [Shannon (1949). Communication in the Presence of Noise](https://people.math.harvard.edu/~ctm/home/text/others/shannon/entropy/entropy.pdf)：sampling theorem 背後的論文。
- [Smith — The Scientist and Engineer's Guide to Digital Signal Processing](https://www.dspguide.com/ch8.htm)：免費的經典 DSP 教科書。
- [librosa docs — audio primer](https://librosa.org/doc/latest/auto_tutorials/index.html)：附程式碼的實用導覽。
- [Heinrich Kuttruff — Room Acoustics (6th ed.)](https://www.taylorfrancis.com/books/mono/10.1201/9781315372150/room-acoustics-heinrich-kuttruff)：解釋為什麼現實世界的音訊不是乾淨正弦波的參考書。
- [Steve Eddins — FFT Interpretation notebook](https://blogs.mathworks.com/steve/2020/03/30/fft-spectrum-and-spectral-densities/)：10 分鐘搞懂 frequency bin 的直覺。
