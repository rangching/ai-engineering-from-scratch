# 語音活動偵測與輪替 (Voice Activity Detection & Turn-Taking) — Silero、Cobra 與 Flush 技巧

> 每個語音 agent 的成敗，都取決於兩個決定：使用者現在是不是在說話？他們說完了嗎？VAD 回答第一個問題。Turn-detection（VAD + 靜音 hangover + 語意端點模型）回答第二個。任何一個做錯，你的助理要嘛打斷使用者，要嘛永遠不閉嘴。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 11 (Real-Time Audio), Phase 6 · 12 (Voice Assistant)
**Time:** ~45 分鐘

## 問題 (The Problem)

語音 agent 對每個 20 ms 的 chunk，都要做出三個不同的決定：

1. **這個 frame 是語音嗎？** VAD。二元判斷，每個 frame 一次。
2. **使用者開始新的一句話了嗎？** Onset detection。
3. **使用者說完了嗎？** End-pointing（輪次結束）。

天真的做法（能量 threshold）遇到任何雜訊都會失效，包括車流、鍵盤、人群嘈雜聲。2026 年的答案是：Silero VAD（開源、深度學習）+ turn-detection 模型（語意端點偵測）+ 以 VAD 校準過的靜音 hangover。

## 核心概念 (The Concept)

![VAD cascade: energy → Silero → turn-detector → flush trick](../assets/vad-turn-taking.svg)

### 三層 VAD 級聯

**第 1 層：能量閘門 (energy gate)。** 最便宜。把 RMS 的 threshold 設在 -40 dBFS。能過濾明顯的靜音，但只要有超過 threshold 的雜訊就會觸發。

**第 2 層：Silero VAD**（2020-2026，MIT）。1M 個參數。用 6000 多種語言訓練。在單一 CPU thread 上，每 30 ms chunk 約 1 ms。在 5% FPR 下 TPR 為 87.7%。開源的預設選擇。

**第 3 層：語意 turn detector。** LiveKit 的 turn-detection 模型（2024-2026），或你自己的小型分類器。用來區分「句子中間的停頓」和「說完了」。使用語言脈絡（語調 + 最近的字詞），而不只是靜音。

### 關鍵參數與預設值

- **Threshold。** Silero 輸出一個機率；當 &gt; 0.5（預設）或 &gt; 0.3（敏感）時判定為語音。Threshold 越低 = 第一個字被切掉的情況越少，誤判越多。
- **最短語音長度。** 拒絕短於 250 ms 的語音，通常是咳嗽或椅子的聲音。
- **靜音 hangover（end-pointing）。** VAD 回到 0 之後，等待 500-800 ms 再宣告輪次結束。太短 → 打斷使用者。太長 → 感覺遲鈍。
- **Pre-roll buffer。** 在 VAD 觸發前保留 300-500 ms 音訊。避免 "hey" 被切掉。

### Flush 技巧（Kyutai 2025）

Streaming STT 模型有 look-ahead 延遲（Kyutai STT-1B 為 500 ms，STT-2.6B 為 2.5 秒）。一般來說，語音結束後你得等那麼久才能拿到逐字稿。Flush 技巧：當 VAD 觸發語音結束時，**對 STT 送出 flush 訊號**，強迫它立即輸出。STT 以約 4 倍即時速度處理，所以 500 ms 的 buffer 約 125 ms 就處理完了。

端到端：125 ms VAD + flush STT = 對話級的延遲。

### 2026 年 VAD 比較

| VAD | TPR @ 5% FPR | 延遲 | 授權 |
|-----|--------------|---------|---------|
| WebRTC VAD (Google, 2013) | 50.0% | 30 ms | BSD |
| Silero VAD (2020-2026) | 87.7% | ~1 ms | MIT |
| Cobra VAD (Picovoice) | 98.9% | ~1 ms | commercial |
| pyannote segmentation | 95% | ~10 ms | MIT-ish |

Silero 是正確的預設選擇。Cobra 是合規 / 準確度上的升級選項。只用能量的 VAD 在 2026 年的正式環境中沒有立足之地。

```figure
sp-vad-cascade
```

## 動手做 (Build It)

### 步驟 1：能量閘門

```python
def energy_vad(chunk, threshold_dbfs=-40.0):
    rms = (sum(x * x for x in chunk) / len(chunk)) ** 0.5
    dbfs = 20.0 * math.log10(max(rms, 1e-10))
    return dbfs > threshold_dbfs
```

### 步驟 2：Python 中的 Silero VAD

```python
from silero_vad import load_silero_vad, get_speech_timestamps

vad = load_silero_vad()
audio = torch.tensor(waveform_16k, dtype=torch.float32)
segments = get_speech_timestamps(
    audio, vad, sampling_rate=16000,
    threshold=0.5,
    min_speech_duration_ms=250,
    min_silence_duration_ms=500,
    speech_pad_ms=300,
)
for s in segments:
    print(f"{s['start']/16000:.2f}s - {s['end']/16000:.2f}s")
```

### 步驟 3：輪次結束的狀態機

```python
class TurnDetector:
    def __init__(self, silence_hangover_ms=500, min_speech_ms=250):
        self.state = "idle"
        self.speech_ms = 0
        self.silence_ms = 0
        self.silence_hangover_ms = silence_hangover_ms
        self.min_speech_ms = min_speech_ms

    def update(self, is_speech, chunk_ms=20):
        if is_speech:
            self.speech_ms += chunk_ms
            self.silence_ms = 0
            if self.state == "idle" and self.speech_ms >= self.min_speech_ms:
                self.state = "speaking"
                return "START"
        else:
            self.silence_ms += chunk_ms
            if self.state == "speaking" and self.silence_ms >= self.silence_hangover_ms:
                self.state = "idle"
                self.speech_ms = 0
                return "END"
        return None
```

### 步驟 4：Flush 技巧的骨架

```python
def flush_on_end(stt_client, audio_buffer):
    stt_client.send_audio(audio_buffer)
    stt_client.send_flush()
    return stt_client.recv_transcript(timeout_ms=150)
```

STT（Kyutai、Deepgram、AssemblyAI）必須支援 flush，這招才行得通。Whisper streaming 不支援，它是以區塊為基礎，總是要等 chunks。

## 實際使用 (Use It)

| 情境 | VAD 選擇 |
|-----------|-----------|
| 開源、快速、通用 | Silero VAD |
| 商用客服中心 | Cobra VAD |
| 裝置端（手機） | Silero VAD ONNX |
| 研究 / diarization | pyannote segmentation |
| 零依賴的後備方案 | WebRTC VAD (legacy) |
| 需要好的輪次結束品質 | Silero + LiveKit turn-detector 疊加 |

經驗法則：除非真的沒有其他選擇，否則絕不要上線只用能量的 VAD。

## 陷阱 (Pitfalls)

- **固定的 threshold。** 安靜時有用，吵雜時失效。要嘛在裝置上校準，要嘛改用 Silero。
- **靜音 hangover 太短。** Agent 會在句子中間打斷。500-800 ms 是對話語音的甜蜜點。
- **Hangover 太長。** 感覺遲鈍。請和目標使用者做 A/B 測試。
- **沒有 pre-roll buffer。** 使用者音訊的前 200-300 ms 會遺失。請一律保留滾動式的 pre-roll。
- **忽略語意端點偵測。** "Hmm, let me think..." 含有很長的停頓。使用者討厭思考到一半被打斷。請使用 LiveKit 的 turn-detector 或類似工具。

## 交付 (Ship It)

存成 `outputs/skill-vad-tuner.md`。針對某個工作負載，挑選 VAD 模型、threshold、hangover、pre-roll 與 turn-detection 策略。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它模擬一段語音 + 靜音 + 語音 + 咳嗽的序列，並測試三層 VAD。
2. **中等。** 安裝 `silero-vad`，處理一段 5 分鐘的錄音，調整 threshold，讓第一個字被切掉和誤觸發兩者都降到最低。回報 precision/recall。
3. **困難。** 打造一個迷你 turn-detector：Silero VAD + 一個 3 層 MLP，輸入是最後 10 個字的 embeddings（使用 sentence-transformers）。在人工標註的輪次結束資料集上訓練。F1 要比只用 Silero 高出 10%。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| VAD | 語音偵測器 | 每個 frame 的二元判斷：這是語音嗎？ |
| Turn detection | End-pointing | VAD + 靜音 hangover + 語意端點。 |
| Silence hangover | 說完後的等待 | 宣告輪次結束前要等待的時間；500-800 ms。 |
| Pre-roll | 說話前的 buffer | 在 VAD 觸發前保留 300-500 ms 音訊。 |
| Flush trick | Kyutai 的小技巧 | VAD → flush-STT → 延遲 125 ms，而不是 500 ms。 |
| Semantic endpoint | 「他們是有意停下來的嗎？」 | 看字詞而不只看靜音的 ML 分類器。 |
| TPR @ FPR 5% | ROC 上的一個點 | 標準的 VAD benchmark；Silero 為 87.7%，WebRTC 為 50%。 |

## 延伸閱讀 (Further Reading)

- [Silero VAD](https://github.com/snakers4/silero-vad)：參考用的開源 VAD。
- [Picovoice Cobra VAD](https://picovoice.ai/products/voice/voice-activity-detection/)：商用的準確度領先者。
- [Kyutai — Unmute + flush trick](https://kyutai.org/stt)：讓延遲降到 200 ms 以下的工程技巧。
- [LiveKit — turn detection](https://docs.livekit.io/agents/logic/turns/)：正式環境中的語意端點偵測。
- [WebRTC VAD](https://webrtc.googlesource.com/src/)：舊式基準。
- [pyannote segmentation](https://github.com/pyannote/pyannote-audio)：diarization 等級的分段。
