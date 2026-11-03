# 即時音訊處理 (Real-Time Audio Processing)

> 批次 pipeline 處理一個檔案。即時 pipeline 要在下一個 20 毫秒到達之前，處理完這一個 20 毫秒。每一個對話式 AI、廣播錄音室和電話機器人，都得靠這個延遲預算 (latency budget) 活下去。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 6 · 04 (ASR), Phase 6 · 07 (TTS)
**Time:** ~75 分鐘

## 問題 (The Problem)

你想要一個感覺像活人的語音助理。人類對話的輪替延遲約為 230 ms（從靜默到回應）。超過 500 ms 就會覺得像機器人；超過 1500 ms 就像壞掉了。2026 年，完整的 **聽 → 理解 → 回應 → 說** 迴圈預算如下：

| 階段 | 預算 |
|-------|--------|
| Mic → buffer | 20 ms |
| VAD | 10 ms |
| ASR (streaming) | 150 ms |
| LLM (first token) | 100 ms |
| TTS (first chunk) | 100 ms |
| Render → speaker | 20 ms |
| **Total** | **~400 ms** |

Moshi（Kyutai，2024）做到全雙工 200 ms。GPT-4o-realtime（2024）約 320 ms。2022 年的串接式 pipeline 則是 2500 ms。這 10 倍的改善來自三種技術：(1) 到處都用串流，(2) 搭配部分結果的非同步 pipeline，(3) 可中斷的生成。

## 核心概念 (The Concept)

![Streaming audio pipeline with ring buffer, VAD gate, interruption](../assets/real-time.svg)

**Frame / chunk / window。** 即時音訊以固定大小的區塊流動。常見選擇：20 ms（16 kHz 下是 320 samples）。下游所有東西都必須跟上這個節奏。

**Ring buffer。** 固定大小的環狀 buffer。Producer thread 寫入新的 frames，consumer thread 讀取。避免在熱路徑 (hot path) 配置記憶體。大小 ≈ 最大延遲 × sample rate；2 秒、16 kHz 的 ring 就是 32,000 samples。

**VAD (Voice Activity Detection)。** 沒人說話時，擋住下游的工作。Silero VAD 4.0（2024）在 CPU 上處理每 30 ms 的 frame 不到 1 ms。`webrtcvad` 是較舊的替代方案。

**Streaming ASR。** 隨著音訊到達就輸出部分逐字稿的模型。串流模式下的 Parakeet-CTC-0.6B（NeMo，2024）在 320 ms 延遲下可達 2–5% WER。Whisper-Streaming（Macháček et al., 2023）把 Whisper 切塊，以約 2 秒延遲做到近乎串流。

**中斷 (Interruption)。** 當使用者在助理說話時開口，你必須 (a) 偵測到插話 (barge-in)，(b) 停止 TTS，(c) 丟棄剩下的 LLM 輸出。全部要在 100 ms 內完成，否則使用者會覺得助理聽不見。

**WebRTC Opus transport。** 20 ms frames、48 kHz、可適應的位元率 8–128 kbps。瀏覽器與行動裝置的標準。LiveKit、Daily.co、Pion 是 2026 年打造語音應用的工具組合。

**Jitter buffer。** 網路封包會亂序 / 延遲到達。Jitter buffer 負責重新排序並平滑；太小 → 聽得到斷斷續續，太大 → 延遲增加。典型值為 60–80 ms。

### 常見的坑 (Common gotchas)

- **Thread 競爭。** Python 的 GIL + 重量級模型可能讓 audio thread 餓死。請使用 C callback 的音訊函式庫（sounddevice、PortAudio），並讓 Python 離開熱路徑。
- **Sample-rate 轉換延遲。** 在 pipeline 內部 resample 會增加 5–20 ms。請事先 resample，或使用零延遲的 resampler（PolyPhase、`soxr_hq`）。
- **TTS 暖機。** 即使是 Kokoro 這種很快的 TTS，第一次請求也有 100–200 ms 的暖機時間。請快取模型，並在第一個真正的輪次之前用假資料跑一次來暖機。
- **回音消除。** 沒有 AEC，TTS 輸出會重新進入麥克風，讓 ASR 對機器人自己的聲音起反應。WebRTC AEC3 是開源的預設選擇。

```figure
nyquist-aliasing
```

## 動手做 (Build It)

### 步驟 1：Ring buffer

```python
import collections

class RingBuffer:
    def __init__(self, capacity):
        self.buf = collections.deque(maxlen=capacity)
    def write(self, frame):
        self.buf.extend(frame)
    def read(self, n):
        return [self.buf.popleft() for _ in range(min(n, len(self.buf)))]
    def level(self):
        return len(self.buf)
```

容量決定最大的 buffering 延遲。16 kHz 下 32,000 samples = 2 秒。

### 步驟 2：VAD 閘門

```python
def simple_energy_vad(frame, threshold=0.01):
    return sum(x * x for x in frame) / len(frame) > threshold ** 2
```

正式環境請換成 Silero VAD：

```python
import torch
vad, _ = torch.hub.load("snakers4/silero-vad", "silero_vad")
is_speech = vad(torch.tensor(frame), 16000).item() > 0.5
```

### 步驟 3：Streaming ASR

```python
# Parakeet-CTC-0.6B streaming via NeMo
from nemo.collections.asr.models import EncDecCTCModelBPE
asr = EncDecCTCModelBPE.from_pretrained("nvidia/parakeet-ctc-0.6b")
# chunk_ms=320 ms, look_ahead_ms=80 ms
for chunk in audio_stream():
    partial_text = asr.transcribe_streaming(chunk)
    print(partial_text, end="\r")
```

### 步驟 4：中斷處理器

```python
class Dialog:
    def __init__(self):
        self.tts_task = None

    def on_user_speech(self, frame):
        if self.tts_task and not self.tts_task.done():
            self.tts_task.cancel()   # barge-in
        # then feed to streaming ASR

    def on_final_user_utterance(self, text):
        self.tts_task = asyncio.create_task(self.reply(text))

    async def reply(self, text):
        async for tts_chunk in llm_then_tts(text):
            speaker.write(tts_chunk)
```

關鍵在於 async I/O 和可取消的 TTS 串流。對 audio track 呼叫 WebRTC 的 peerconnection.stop() 是標準做法。

## 實際使用 (Use It)

2026 年的工具組合：

| 層級 | 選擇 |
|-------|------|
| Transport | LiveKit (WebRTC) 或 Pion (Go) |
| VAD | Silero VAD 4.0 |
| Streaming ASR | Parakeet-CTC-0.6B 或 Whisper-Streaming |
| LLM first-token | Groq、Cerebras、vLLM-streaming |
| Streaming TTS | Kokoro 或 ElevenLabs Turbo v2.5 |
| 回音消除 | WebRTC AEC3 |
| 端到端原生 | OpenAI Realtime API 或 Moshi |

## 陷阱 (Pitfalls)

- **為了保險而 buffer 500 ms。** buffer *就是*你的延遲下限。請縮小它。
- **沒有固定 (pin) threads。** audio callback 的優先權低於 UI thread，負載一高就會出現雜音。
- **TTS chunks 太小。** 小於 200 ms 的 chunk 會讓 vocoder 的瑕疵變得聽得見。320 ms 的 chunk 是甜蜜點。
- **沒有 jitter buffer。** 真實網路會抖動；不平滑就會出現爆音。
- **只做一次性的錯誤處理。** 音訊 pipeline 必須不會崩潰。一個 exception 就會讓整個 session 結束。

## 交付 (Ship It)

存成 `outputs/skill-realtime-designer.md`。設計即時音訊 pipeline，並為每個階段訂出具體的延遲預算。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。模擬 ring buffer + 能量式 VAD，並為一段假的 10 秒串流印出各階段延遲。
2. **中等。** 用 `sounddevice` 建立一個 passthrough 迴圈，以 20 ms frames 處理你的麥克風，並在每個 frame 印出 VAD 狀態。
3. **困難。** 用 `aiortc` 建立完整的全雙工回音測試：瀏覽器 → WebRTC → Python → WebRTC → 瀏覽器。用 1 kHz 脈衝量測 glass-to-glass 延遲。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Ring buffer | 環狀佇列 | 固定大小、無鎖（或 SPSC 加鎖）的音訊 frames FIFO。 |
| VAD | 靜音閘門 | 標記語音 vs 非語音的模型或啟發式方法。 |
| Streaming ASR | 即時 STT | 隨著音訊到達輸出部分文字；lookahead 有上限。 |
| Jitter buffer | 網路平滑器 | 重新排序亂序封包的佇列；典型值 60–80 ms。 |
| AEC | 回音消除 | 減去喇叭到麥克風的回授路徑。 |
| Barge-in | 使用者插話 | 系統在 TTS 播放途中偵測到使用者說話；必須取消播放。 |
| Full duplex | 雙向同時 | 使用者與機器人可以同時說話；Moshi 是全雙工。 |

## 延伸閱讀 (Further Reading)

- [Macháček et al. (2023). Whisper-Streaming](https://arxiv.org/abs/2307.14743)：切塊的近乎串流版 Whisper。
- [Kyutai (2024). Moshi](https://kyutai.org/Moshi.pdf)：全雙工、200 ms 延遲。
- [LiveKit Agents framework (2024)](https://docs.livekit.io/agents/)：正式環境的音訊 agent 編排。
- [Silero VAD repo](https://github.com/snakers4/silero-vad)：1 ms 以內的 VAD，Apache 2.0。
- [WebRTC AEC3 paper](https://webrtc.googlesource.com/src/+/main/modules/audio_processing/aec3/)：開源的回音消除。
