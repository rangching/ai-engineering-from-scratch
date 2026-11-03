# Neural Audio Codecs — EnCodec、SNAC、Mimi、DAC 與語意-聲學分離

> 2026 年的音訊生成幾乎全是 tokens。EnCodec、SNAC、Mimi 和 DAC 把連續的 waveform 變成 transformer 可以預測的離散序列。語意 vs 聲學 token 的分離（第一個 codebook 負責語意，其餘負責聲學），是自 Transformer 以來，對音訊最重要的架構轉變。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 10 · 11 (Quantization), Phase 5 · 19 (Subword Tokenization)
**Time:** ~60 分鐘

## 問題 (The Problem)

語言模型處理的是離散 tokens，音訊卻是連續的。如果你想要語音 / 音樂的 LLM 風格模型（MusicGen、Moshi、Sesame CSM、VibeVoice、Orpheus），首先需要 **neural audio codec**：一個學習得到的 encoder，把音訊離散化成一個小詞彙表的 tokens，再搭配一個重建 waveform 的 decoder。

目前出現了兩個家族：

1. **重建優先的 codecs**：EnCodec、DAC。最佳化感知上的音訊品質。Tokens 是「聲學的」，會捕捉所有東西，包括說話者身分、音色和背景雜訊。
2. **語意優先的 codecs**：Mimi (Kyutai)、SpeechTokenizer。強迫第一個 codebook 編碼語言 / 語音內容（通常是從 WavLM 蒸餾而來）。後續的 codebooks 則是聲學細節。

2024-2026 年的洞見：**純重建的 codec，在你想從文字生成時會得到模糊的語音。** 在 codec tokens 上的 LLM 必須在同一個 codebook 中同時學會語言結構和聲學結構，這無法 scale。把它們分開（語意放在 codebook 0，聲學放在 codebooks 1-N），才讓 Moshi 和 Sesame CSM 得以運作。

## 核心概念 (The Concept)

![Four codec landscape: EnCodec, DAC, SNAC (multi-scale), Mimi (semantic+acoustic)](../assets/codec-comparison.svg)

### 核心技巧：Residual Vector Quantization (RVQ)

為了達到好品質，單一大型 codebook 需要上百萬個 codes，所以所有現代音訊 codecs 都改用 **RVQ**：一串級聯的小 codebooks。第一個 codebook 量化 encoder 的輸出；第二個量化殘差；以此類推。每個 codebook 有 1024 個 codes。8 個 codebooks = 有效詞彙量 $1024^8 = 10^{24}$。

推論時，decoder 把每個 frame 選出的所有 codes 加總起來進行重建。

### 2026 年重要的四個 codecs

**EnCodec（Meta，2022）。** 基準。對 waveform 做 encoder-decoder，瓶頸是 RVQ。24 kHz，最多可用 32 個 codebooks，預設 4 個 codebooks @ 1.5 kbps。使用 `1D conv + transformer + 1D conv` 架構。MusicGen 使用它。

**DAC（Descript，2023）。** 帶有 L2 正規化 codebooks、週期性 activation functions 與改良 losses 的 RVQ。是所有開源 codec 中重建保真度最高的，用 12 個 codebooks 時有時和原始語音無法區分。44.1 kHz 全頻帶。

**SNAC（Hubert Siuzdak，2024）。** 多尺度 RVQ，粗略的 codebooks 以比精細的更低的 frame rate 運作。實際上是以階層方式為音訊建模：約 12 Hz 的粗略「草圖」，加上 50 Hz 的細節。Orpheus-3B 使用它，因為這種階層結構能很好地對應到基於 LM 的生成。

**Mimi（Kyutai，2024）。** 2026 年的改變遊戲規則者。12.5 Hz 的 frame rate（極低），8 個 codebooks @ 4.4 kbps。Codebook 0 是**從 WavLM 蒸餾而來**，被訓練去預測 WavLM 的語音內容特徵。Codebooks 1-7 是聲學殘差。這個分離驅動了 Moshi（第 15 課）和 Sesame CSM。

### Frame rate 對語言建模很重要

Frame rate 越低 = 序列越短 = LM 越快。

| Codec | Frame rate | 1 s = N frames | 適用於 |
|-------|-----------|----------------|---------|
| EnCodec-24k | 75 Hz | 75 | 音樂、一般音訊 |
| DAC-44.1k | 86 Hz | 86 | 高保真音樂 |
| SNAC-24k (coarse) | ~12 Hz | 12 | 對 AR-LM 高效 |
| Mimi | 12.5 Hz | 12.5 | 串流語音 |

在 12.5 Hz 下，10 秒的語音只有 125 個 codec frames，transformer 可以輕鬆預測。

### 語意 vs 聲學 tokens

```
frame_t → [semantic_token_t, acoustic_token_0_t, acoustic_token_1_t, ..., acoustic_token_6_t]
```

- **Semantic token（Mimi 中的 codebook 0）。** 編碼說了什麼：音素、字詞、內容。透過輔助預測 loss 從 WavLM 蒸餾而來。
- **Acoustic tokens（codebooks 1-7）。** 編碼音色、說話者身分、韻律、背景雜訊與細節。

AR LM 先預測 semantic token（以文字為條件），再預測 acoustic tokens（以 semantic + 說話者參考為條件）。這種分解就是現代 TTS 能 zero-shot 複製聲音的原因：語意模型處理內容；聲學模型處理音色。

### 2026 年的重建品質（每秒位元數，位元率越低越好）

| Codec | Bitrate | PESQ | ViSQOL |
|-------|---------|------|--------|
| Opus-20kbps | 20 kbps | 4.0 | 4.3 |
| EnCodec-6kbps | 6 kbps | 3.2 | 3.8 |
| DAC-6kbps | 6 kbps | 3.5 | 4.0 |
| SNAC-3kbps | 3 kbps | 3.3 | 3.8 |
| Mimi-4.4kbps | 4.4 kbps | 3.1 | 3.7 |

像 Opus 這樣的傳統 codecs，在每位元的感知品質上仍然勝出。Neural codecs 勝在**離散 tokens**（Opus 不會產生）以及**生成模型的品質**（LM 能用這些 tokens 做到什麼）。

```figure
rvq-codec-cascade
```

## 動手做 (Build It)

### 步驟 1：用 EnCodec 編碼

```python
from encodec import EncodecModel
import torch

model = EncodecModel.encodec_model_24khz()
model.set_target_bandwidth(6.0)  # kbps

wav = torch.randn(1, 1, 24000)
with torch.no_grad():
    encoded = model.encode(wav)
codes, scale = encoded[0]
# codes: (1, n_codebooks, n_frames), dtype=int64
```

6 kbps 下 `n_codebooks=8`。每個 code 介於 0-1023（10-bit）。

### 步驟 2：解碼並量測重建

```python
with torch.no_grad():
    wav_recon = model.decode([(codes, scale)])

from torchaudio.functional import compute_deltas
import torch.nn.functional as F

mse = F.mse_loss(wav_recon[:, :, :wav.shape[-1]], wav).item()
```

### 步驟 3：語意-聲學分離（Mimi 風格）

```python
from moshi.models import loaders
mimi = loaders.get_mimi()

with torch.no_grad():
    codes = mimi.encode(wav)  # shape (1, 8, frames@12.5Hz)

semantic = codes[:, 0]
acoustic = codes[:, 1:]
```

Semantic codebook 0 與 WavLM 對齊。你可以訓練 text-to-semantic 的 transformer，詞彙量比直接生成音訊小得多。再由另一個獨立的 acoustic-to-waveform decoder，以說話者參考為條件。

### 步驟 4：為什麼在 codec tokens 上做 AR LM 行得通

以 Mimi 的 12.5 Hz × 8 個 codebooks 處理 10 秒語音片段：

```math
N_{\text{tokens}} = 10 \times 12.5 \times 8 = 1000 \text{ tokens}
```

1000 個 tokens 對 transformer 來說是微不足道的 context。256M 參數的 transformer 在現代 GPU 上，可以在數毫秒內生成 10 秒的語音。

## 實際使用 (Use It)

把問題對應到 codec：

| 任務 | Codec |
|------|-------|
| 一般音樂生成 | EnCodec-24k |
| 最高保真度的重建 | DAC-44.1k |
| 語音上的 AR LM (TTS) | SNAC 或 Mimi |
| 串流全雙工語音 | Mimi (12.5 Hz) |
| 搭配文字的音效庫 | EnCodec + T5 condition |
| 細緻的音訊編輯 | DAC + inpainting |

經驗法則：**如果你在打造生成模型，從 Mimi 或 SNAC 開始。如果你在打造壓縮 pipeline，就用 Opus。**

## 陷阱 (Pitfalls)

- **Codebooks 太多。** 增加 codebooks 會讓保真度線性增加，但 LM 序列長度也線性增加。請在 8-12 個停下來。
- **Frame rate 不一致。** 在 12.5 Hz 的 Mimi 上訓練 LM，再用 50 Hz 的 EnCodec 做 fine-tune，會悄悄失敗。
- **假設所有 codebooks 都一樣重要。** 在 Mimi 中，codebook 0 承載內容；失去它會毀掉可懂度。失去 codebook 7 幾乎察覺不到。
- **只用重建品質當作唯一指標。** Codec 可能重建得很好，但如果語意結構不佳，對基於 LM 的生成就毫無用處。

## 交付 (Ship It)

存成 `outputs/skill-codec-picker.md`。針對指定的生成或壓縮任務，挑選 codec。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它實作了玩具版的 scalar + residual quantizer，並在你增加 codebooks 時量測重建誤差。
2. **中等。** 安裝 `encodec`，在 held-out 的語音片段上比較 1、4、8、32 個 codebooks。畫出 PESQ 或 MSE 對 bitrate 的圖。
3. **困難。** 載入 Mimi。編碼一段音訊。把 codebook 0 換成隨機整數後解碼。接著對 codebook 7 做同樣的事。比較兩種破壞，codebook 0 被破壞應該會毀掉可懂度；codebook 7 被破壞則應該幾乎沒有變化。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| RVQ | 殘差量化 | 級聯的小 codebooks；每個都量化前一層的殘差。 |
| Frame rate | Codec 速度 | 每秒有多少 token-frames。越低 = LM 越快。 |
| Semantic codebook | Codebook 0 (Mimi) | 從 SSL 特徵蒸餾而來的 codebook；編碼內容。 |
| Acoustic codebooks | 其他全部 | 音色、韻律、雜訊、細節。 |
| PESQ / ViSQOL | 感知品質 | 與 MOS 相關的客觀指標。 |
| EnCodec | Meta 的 codec | RVQ 基準；MusicGen 使用。 |
| Mimi | Kyutai 的 codec | 12.5 Hz frame rate；語意-聲學分離；驅動 Moshi。 |

## 延伸閱讀 (Further Reading)

- [Défossez et al. (2023). EnCodec](https://arxiv.org/abs/2210.13438)：RVQ 基準。
- [Kumar et al. (2023). Descript Audio Codec (DAC)](https://arxiv.org/abs/2306.06546)：保真度最高的開源方案。
- [Siuzdak (2024). SNAC](https://arxiv.org/abs/2410.14411)：多尺度 RVQ。
- [Kyutai (2024). Mimi codec](https://kyutai.org/codec-explainer)：語意-聲學分離、WavLM 蒸餾。
- [Borsos et al. (2023). AudioLM](https://arxiv.org/abs/2209.03143)：兩階段的語意/聲學範式。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312)：最早可串流的 RVQ codec。
