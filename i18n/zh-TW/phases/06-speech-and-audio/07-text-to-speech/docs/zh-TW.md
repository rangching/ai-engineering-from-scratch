# 文字轉語音 (Text-to-Speech, TTS) — 從 Tacotron 到 F5 和 Kokoro

> ASR 把語音轉成文字；TTS 把文字轉成語音。2026 年的做法分三段：文字 → tokens、tokens → mel、mel → waveform。每一段都有一個預設模型，筆電就跑得動。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 09 (Seq2Seq), Phase 7 · 05 (Full Transformer)
**Time:** ~75 分鐘

## 問題 (The Problem)

你有一個字串："Please remind me to water the plants at 6 pm."。你需要一段 3 秒、聽起來自然的音訊，韻律（停頓、重音）正確，"plants" 的母音唸對，而且在即時語音助理的 CPU 上要在 300 ms 內跑完。你還需要能換聲音、處理語碼轉換 (code-switched) 的輸入（"remind me at 6 pm, daijoubu?"），而且唸人名時不能出糗。

現代 TTS pipeline 長這樣：

1. **Text frontend。** 正規化文字（日期、數字、email），轉成音素 (phonemes) 或 subword tokens，並預測韻律特徵。
2. **Acoustic model。** 文字 → mel spectrogram。Tacotron 2（2017）、FastSpeech 2（2020）、VITS（2021）、F5-TTS（2024）、Kokoro（2024）。
3. **Vocoder。** Mel → waveform。WaveNet（2016）、WaveRNN、HiFi-GAN（2020）、BigVGAN（2022），以及 2024 年之後的 neural codec vocoders。

2026 年，acoustic 與 vocoder 的分界隨著端到端的 diffusion 與 flow-matching 模型而模糊。但在除錯時，三段式的思維模型仍然適用。

## 核心概念 (The Concept)

![Tacotron, FastSpeech, VITS, F5/Kokoro side-by-side](../assets/tts.svg)

**Tacotron 2（2017）。** Seq2seq：char-embedding → BiLSTM encoder → location-sensitive attention → autoregressive LSTM decoder 輸出 mel frames。很慢（AR），文字一長就不穩。仍被當作基準引用。

**FastSpeech 2（2020）。** Non-autoregressive。Duration predictor 輸出每個音素佔幾個 mel frames。一次到位，比 Tacotron 快 10 倍。自然度略遜（單調對齊），但到處都有人用。

**VITS（2021）。** 用變分推論，端到端聯合訓練 encoder + 基於 flow 的 duration + HiFi-GAN vocoder。品質高，只有單一模型。2022–2024 年主導開源 TTS。變體：YourTTS（多說話者 zero-shot）、XTTS v2（2024，Coqui）。

**F5-TTS（2024）。** 建立在 flow matching 上的 diffusion transformer。韻律自然，用 5 秒參考音訊就能做 zero-shot voice cloning。位居 2026 年開源 TTS 排行榜之首。335M 個參數。

**Kokoro（2024）。** 小型（82M），能在 CPU 上執行，是即時應用中同級最佳的英文 TTS。詞彙封閉、僅支援英文，採用 apache-2.0 授權。

**OpenAI TTS-1-HD、ElevenLabs v2.5、Google Chirp-3。** 商用的 state of the art。ElevenLabs v2.5 的情緒標籤（"[whispered]"、"[laughing]"）與角色聲音，在 2026 年主導有聲書製作。

### Vocoder 的演進

| 年代 | Vocoder | 延遲 | 品質 |
|-----|---------|---------|---------|
| 2016 | WaveNet | 僅限離線 | 發表時是 SOTA |
| 2018 | WaveRNN | 接近即時 | 良好 |
| 2020 | HiFi-GAN | 100× 即時 | 接近真人 |
| 2022 | BigVGAN | 50× 即時 | 能泛化到不同說話者/語言 |
| 2024 | SNAC, DAC (neural codecs) | 與 AR 模型整合 | 離散 tokens，位元效率高 |

到了 2026 年，大多數「TTS」模型都是從文字到 waveform 的端到端模型；mel spectrogram 只是內部表示。

### 評估 (Evaluation)

- **MOS (Mean Opinion Score)。** 1–5 分，由群眾外包評分。仍是黃金標準，但慢得要命。
- **CMOS (Comparative MOS)。** A 對 B 的偏好。每筆標註的信賴區間較窄。
- **UTMOS, DNSMOS。** 不需參考音訊的神經網路 MOS 預測器。用於排行榜。
- **透過 ASR 計算 CER (Character Error Rate)。** 把 TTS 輸出丟進 Whisper，與輸入文字計算 CER。作為可懂度 (intelligibility) 的代理指標。
- **SECS (Speaker Embedding Cosine Similarity)。** Voice-cloning 的品質。

LibriTTS test-clean 上的 2026 年數字：

| Model | UTMOS | CER (via Whisper) | Size |
|-------|-------|-------------------|------|
| Ground truth | 4.08 | 1.2% | — |
| F5-TTS | 3.95 | 2.1% | 335M |
| XTTS v2 | 3.81 | 3.5% | 470M |
| VITS | 3.62 | 3.1% | 25M |
| Kokoro v0.19 | 3.87 | 1.8% | 82M |
| Parler-TTS Large | 3.76 | 2.8% | 2.3B |

```figure
sp-tts-stack
```

## 動手做 (Build It)

### 步驟 1：把輸入轉成音素 (phonemize)

```python
from phonemizer import phonemize
ph = phonemize("Hello world", language="en-us", backend="espeak")
# 'həloʊ wɜːld'
```

音素是通用的橋樑。品質低於 VITS 等級的模型，避免直接餵原始文字。

### 步驟 2：執行 Kokoro（2026 年 CPU 預設）

```python
from kokoro import KPipeline
tts = KPipeline(lang_code="a")  # "a" = American English
audio, sr = tts("Please remind me to water the plants at 6 pm.", voice="af_bella")
# audio: float32 tensor, sr=24000
```

可離線執行，單一檔案，82M 個參數。

### 步驟 3：用 F5-TTS 執行 voice cloning

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="my_voice_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please remind me to water the plants.",
)
```

傳入一段 5 秒的參考音訊與它的逐字稿；F5 會複製韻律與音色。

### 步驟 4：從零打造 HiFi-GAN vocoder

太大了，塞不進教學腳本，但整體形狀如下：

```python
class HiFiGAN(nn.Module):
    def __init__(self, mel_channels=80, upsample_rates=[8, 8, 2, 2]):
        super().__init__()
        # 4 upsample blocks, total 256x to go from mel-rate to audio-rate
        ...
    def forward(self, mel):
        return self.blocks(mel)  # -> waveform
```

訓練方式：對抗式（discriminator 作用在短視窗上）+ mel-spectrogram 重建 loss + feature-matching loss。已經商品化，請使用 `hifi-gan` repo 或 nvidia-NeMo 的預訓練 checkpoints。

### 步驟 5：完整 pipeline（pseudocode）

```python
text = "Please remind me at 6 pm."
phones = phonemize(text)
mel = acoustic_model(phones, speaker=alice)      # [T, 80]
wav = vocoder(mel)                                # [T * 256]
soundfile.write("out.wav", wav, 24000)
```

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 選擇 |
|-----------|------|
| 即時英文語音助理 | Kokoro (CPU) 或 XTTS v2 (GPU) |
| 用 5 秒參考音訊做 voice cloning | F5-TTS |
| 商用角色聲音 | ElevenLabs v2.5 |
| 有聲書朗讀 | ElevenLabs v2.5 或 XTTS v2 + fine-tune |
| 低資源語言 | 用 5–20 小時的目標語言資料訓練 VITS |
| 富表現力 / 情緒標籤 | ElevenLabs v2.5 或 StyleTTS 2 fine-tune |

截至 2026 年的開源領先者：**品質選 F5-TTS，效率選 Kokoro**。除非你是歷史學家，否則別碰 Tacotron。

## 陷阱 (Pitfalls)

- **沒有 text normalizer。** "Dr. Smith" 要唸成 "Doctor" 還是 "Drive"？"2026" 要唸成 "twenty twenty six" 還是 "two zero two six"？請在 phonemizer 之前先正規化。
- **OOV 專有名詞。** "Ghumare" → "ghyu-mair"？請為未知 tokens 準備後備的 grapheme-to-phoneme 模型。
- **Clipping。** Vocoder 輸出很少爆音，但推論時 mel 縮放不一致可能超出 ±1.0。一律使用 `np.clip(wav, -1, 1)`。
- **Sample rate 不一致。** Kokoro 輸出 24 kHz；你的下游 pipeline 預期 16 kHz → 要 resample，否則會出現 aliasing。

## 交付 (Ship It)

存成 `outputs/skill-tts-designer.md`。針對指定的聲音、延遲與語言目標，設計 TTS pipeline。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會從玩具詞彙表建立音素字典、估計每個音素的時長，並印出假的「mel」排程。
2. **中等。** 安裝 Kokoro，用 `af_bella` 和 `am_adam` 兩種聲音合成同一句話。比較音訊長度與主觀品質。
3. **困難。** 錄一段自己 5 秒的參考音訊，用 F5-TTS 複製它。回報參考音訊與複製輸出之間的 SECS。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Phoneme | 聲音單位 | 抽象的聲音類別；英文有 39 個 (ARPABet)。 |
| Duration predictor | 每個音素持續多久 | Non-AR 模型的輸出；每個音素對應的整數 frames 數。 |
| Vocoder | Mel → waveform | 把 mel-spec 對應到原始 samples 的神經網路。 |
| HiFi-GAN | 標準 vocoder | 基於 GAN；2020–2024 年主流。 |
| MOS | 主觀品質 | 真人評分者給的 1–5 分 mean opinion score。 |
| SECS | Voice-clone 指標 | 目標與輸出說話者 embedding 之間的 cosine similarity。 |
| F5-TTS | 2024 年開源 SOTA | Flow-matching diffusion；zero-shot cloning。 |
| Kokoro | CPU 英文領先者 | 82M 參數的模型，Apache 2.0。 |

## 延伸閱讀 (Further Reading)

- [Shen et al. (2017). Tacotron 2](https://arxiv.org/abs/1712.05884)：seq2seq 基準。
- [Kim, Kong, Son (2021). VITS](https://arxiv.org/abs/2106.06103)：端到端、基於 flow。
- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885)：目前開源的 SOTA。
- [Kong, Kim, Bae (2020). HiFi-GAN](https://arxiv.org/abs/2010.05646)：2026 年仍在使用的 vocoder。
- [Kokoro-82M on HuggingFace](https://huggingface.co/hexgrad/Kokoro-82M)：2024 年對 CPU 友善的英文 TTS。
