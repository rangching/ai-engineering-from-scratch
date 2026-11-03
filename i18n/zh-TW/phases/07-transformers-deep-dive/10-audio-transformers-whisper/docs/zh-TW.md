# Audio Transformer：Whisper 架構 (Audio Transformers — Whisper Architecture)

> 音訊是頻率隨時間變化的影像。Whisper 是一個吃 mel spectrogram、然後開口說話的 ViT。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 08 (Encoder-Decoder), Phase 7 · 09 (ViT)
**Time:** ~45 分鐘

## 問題 (The Problem)

在 Whisper（OpenAI，Radford et al. 2022）之前，最先進的自動語音辨識（ASR）指的是 wav2vec 2.0 和 HuBERT：自監督的特徵擷取器加上 fine-tune 過的 head。品質高，但資料 pipeline 昂貴，而且對領域很脆弱。多語言語音辨識需要為每個語系各自訓練模型。

Whisper 下了三個賭注：

1. **什麼都拿來訓練。** 從網路上爬取、橫跨 97 種語言、共 680,000 小時的弱標註音訊。沒有乾淨的學術語料庫。沒有音素標籤。
2. **多任務單一模型。** 一個 decoder 透過 task token，同時訓練轉錄、翻譯、語音活動偵測、語言辨識和時間戳記。
3. **標準的 encoder-decoder transformer。** Encoder 吃 log-mel spectrogram。Decoder 以 autoregressive 方式產生文字 token。沒有 vocoder、沒有 CTC、沒有 HMM。

結果是：Whisper large-v3 對口音、雜訊，以及完全沒有乾淨標註資料的語言都很穩健。2026 年，它是每個開源語音助理，以及大多數商用語音助理的預設語音前端。

## 核心概念 (The Concept)

![Whisper pipeline: audio → mel → encoder → decoder → text](../assets/whisper.svg)

### 步驟 1：重新取樣 + 加窗 (Step 1 — resample + window)

音訊為 16 kHz。裁切 / 補齊到 30 秒。計算 log-mel spectrogram：80 個 mel bin，10 ms stride → 約 3,000 個 frame × 80 個特徵。這就是 Whisper 看到的「輸入影像」。

### 步驟 2：Convolutional stem (Step 2 — convolutional stem)

兩個 kernel 為 3、stride 為 2 的 Conv1D 層，把 3,000 個 frame 降到 1,500。在不增加太多參數的情況下，把序列長度減半。

### 步驟 3：Encoder (Step 3 — encoder)

一個 24 層（large 版本）的 transformer encoder，作用於 1,500 個時間步。使用 sinusoidal positional encoding、self-attention、GELU FFN。產生 1,500 × 1,280 的 hidden state。

### 步驟 4：Decoder (Step 4 — decoder)

一個 24 層的 transformer decoder。它以 autoregressive 方式，從 BPE 詞彙中產生 token；這個詞彙是 GPT-2 的超集，另外加了幾個音訊專用的特殊 token。

### 步驟 5：Task token (Step 5 — task tokens)

Decoder 的 prompt 以控制 token 開頭，告訴模型該做什麼：

```
<|startoftranscript|>  <|en|>  <|transcribe|>  <|0.00|>
```

或

```
<|startoftranscript|>  <|fr|>  <|translate|>   <|0.00|>
```

模型就是依照這個慣例訓練的。你用前綴來控制任務。這是 2026 年版的 instruction-tuning，只是套用在語音上。

### 步驟 6：輸出 (Step 6 — output)

Beam search（寬度 5）搭配 log-prob 門檻。當 `<|notimestamps|>` token 不存在時，每 0.02 秒的音訊會預測一個時間戳記。

### Whisper 的各種大小 (Whisper sizes)

| 模型 | 參數量 | 層數 | d_model | Head 數 | VRAM (fp16) |
|-------|--------|--------|---------|-------|-------------|
| Tiny | 39M | 4 | 384 | 6 | ~1 GB |
| Base | 74M | 6 | 512 | 8 | ~1 GB |
| Small | 244M | 12 | 768 | 12 | ~2 GB |
| Medium | 769M | 24 | 1024 | 16 | ~5 GB |
| Large | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3 | 1550M | 32 | 1280 | 20 | ~10 GB |
| Large-v3-turbo | 809M | 32 | 1280 | 20 | ~6 GB (4-layer decoder) |

Large-v3-turbo（2024）把 decoder 從 32 層砍到 4 層。Decoding 快 8 倍，WER 退步不到 1 個點。這個 decode 速度的解鎖，就是 Whisper-turbo 成為 2026 年即時語音 agent 預設選擇的原因。

### Whisper 不做的事 (What Whisper does not do)

- 沒有 diarization（誰在說話）。要搭配 pyannote。
- 原生不支援即時串流，30 秒的視窗是固定的。現代的包裝器（`faster-whisper`、`WhisperX`）透過 VAD + overlap 外掛串流功能。
- 沒有外部切塊的話，超過 30 秒的長篇 context 它處理不了。實務上效果不錯，因為人類語音在轉錄時很少需要長距離的 context。

### 2026 年的版圖 (2026 landscape)

| 任務 | 模型 | 備註 |
|------|-------|-------|
| 英文 ASR | Whisper-turbo, Moonshine | Moonshine 在邊緣裝置上快 4 倍 |
| 多語言 ASR | Whisper-large-v3 | 97 種語言 |
| 串流 ASR | faster-whisper + VAD | 可達到 150 ms 的延遲目標 |
| TTS | Piper, XTTS-v2, Kokoro | Encoder-decoder 模式，但是 Whisper 的形狀 |
| 音訊 + 語言 | AudioLM, SeamlessM4T | 在一個 transformer 裡同時放文字 token 與音訊 token |

```figure
n5-mel-decode
```

## 動手做 (Build It)

請看 `code/main.py`。我們不訓練 Whisper，而是建構 log-mel spectrogram pipeline + task-token prompt 格式化器。這些是你在正式環境中真正會動到的部分。

### 步驟 1：合成音訊 (synthesize audio)

產生 1 秒、440 Hz 的正弦波，取樣率 16 kHz。共 16,000 個樣本。

### 步驟 2：Log-mel spectrogram（簡化版）(log-mel spectrogram (simplified))

完整的 mel spectrogram 需要 FFT。我們做一個簡化版的分幀 + 每 frame 能量，能展示整個 pipeline，又不需要 `librosa`：

```python
def frame_signal(x, frame_size=400, hop=160):
    frames = []
    for start in range(0, len(x) - frame_size + 1, hop):
        frames.append(x[start:start + frame_size])
    return frames
```

Frame = 25 ms，hop = 10 ms。與 Whisper 的加窗方式相符。為了教學，用每 frame 的能量來代替 mel bin。

### 步驟 3：補齊到 30 秒 (pad to 30 s)

Whisper 總是處理 30 秒的區塊。把 spectrogram 補齊（或裁切）到 3,000 個 frame。

### 步驟 4：建構 prompt token (build the prompt tokens)

```python
def whisper_prompt(lang="en", task="transcribe", timestamps=True):
    tokens = ["<|startoftranscript|>", f"<|{lang}|>", f"<|{task}|>"]
    if not timestamps:
        tokens.append("<|notimestamps|>")
    return tokens
```

這就是整個任務控制的介面。一個 4 個 token 的前綴。

## 實際使用 (Use It)

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("meeting.wav", language="en", task="transcribe")
print(result["text"])
print(result["segments"][0]["start"], result["segments"][0]["end"])
```

更快、且與 OpenAI 相容的版本：

```python
from faster_whisper import WhisperModel
model = WhisperModel("large-v3-turbo", compute_type="int8_float16")
segments, info = model.transcribe("meeting.wav", vad_filter=True)
for s in segments:
    print(f"{s.start:.2f} - {s.end:.2f}: {s.text}")
```

**2026 年何時選 Whisper：**

- 用一個模型做多語言 ASR。
- 對雜訊多、種類多樣的音訊做穩健的轉錄。
- 研究 / 原型 ASR，最快的起點。

**何時選別的：**

- 邊緣裝置上超低延遲的串流：在相同品質下 Moonshine 勝過 Whisper。
- 需要 <200 ms 的即時對話 AI：用專門的串流 ASR。
- 說話者 diarization：Whisper 不做這件事；外掛 pyannote。

## 交付 (Ship It)

請看 `outputs/skill-asr-configurator.md`。這個 skill 會為新的語音應用挑選 ASR 模型、decoding 參數與前處理 pipeline。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。確認 16 kHz、10 ms hop 的 1 秒訊號，其 frame 數約為 100。30 秒的話：約 3,000 個 frame。
2. **中等。** 用 `numpy.fft` 建構完整的 log-mel spectrogram。驗證 80 個 mel bin 與 `librosa.feature.melspectrogram(n_mels=80)` 在數值誤差範圍內相符。
3. **困難。** 實作串流推論：把音訊切成 10 秒的視窗並有 2 秒的重疊，對每個區塊執行 Whisper，再合併逐字稿。在 5 分鐘的 podcast 樣本上，測量 word-error rate，並與單次 pass 比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Mel spectrogram | 「音訊影像」 | 2D 表示：一軸是頻率 bin，另一軸是時間 frame；每個格子是 log 縮放後的能量。 |
| Log-mel | 「Whisper 看到的東西」 | 經過 log 的 mel spectrogram；近似人類對響度的感知。 |
| Frame | 「一個時間切片」 | 25 ms 的樣本視窗；以 10 ms 的 stride 重疊。 |
| Task token | 「語音的 prompt 前綴」 | 在 decoder prompt 中的特殊 token，例如 `<\|transcribe\|>` / `<\|translate\|>`。 |
| Voice activity detection (VAD) | 「找出語音」 | 在 ASR 之前去除靜音的閘門；大幅降低成本。 |
| CTC | 「Connectionist Temporal Classification」 | 用於免對齊訓練的經典 ASR loss；Whisper *不*使用它。 |
| Whisper-turbo | 「小 decoder，完整 encoder」 | large-v3 encoder + 4 層 decoder；decoding 快 8 倍。 |
| Faster-whisper | 「正式環境的包裝器」 | CTranslate2 的重新實作；int8 quantization；比 OpenAI 的參考實作快 4 倍。 |

## 延伸閱讀 (Further Reading)

- [Radford et al. (2022). Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356) — Whisper 論文。
- [OpenAI Whisper repo](https://github.com/openai/whisper) — 參考程式碼 + 模型權重。讀 `whisper/model.py`，約 400 行就能從頭到尾看完 Conv1D stem + encoder + decoder。
- [OpenAI Whisper — `whisper/decoding.py`](https://github.com/openai/whisper/blob/main/whisper/decoding.py) — 步驟 5 到 6 所述的 beam-search + task-token 邏輯就在這裡；500 行，完全可讀。
- [Baevski et al. (2020). wav2vec 2.0: A Framework for Self-Supervised Learning of Speech Representations](https://arxiv.org/abs/2006.11477) — 前身；在某些情境下仍是 SOTA 特徵。
- [SYSTRAN/faster-whisper](https://github.com/SYSTRAN/faster-whisper) — 正式環境的包裝器，比參考實作快 4 倍。
- [Jia et al. (2024). Moonshine: Speech Recognition for Live Transcription and Voice Commands](https://arxiv.org/abs/2410.15608) — 2024 年對邊緣裝置友善的 ASR，Whisper 的形狀但更小。
- [HuggingFace blog — "Fine-Tune Whisper For Multilingual ASR with 🤗 Transformers"](https://huggingface.co/blog/fine-tune-whisper) — 標準的 fine-tuning 配方，包含 mel spectrogram 前處理器與 token 時間戳記的處理。
- [HuggingFace `modeling_whisper.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/whisper/modeling_whisper.py) — 完整實作（encoder、decoder、cross-attention、生成），與本課的架構圖相呼應。
