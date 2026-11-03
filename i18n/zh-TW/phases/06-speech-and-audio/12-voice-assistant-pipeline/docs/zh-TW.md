# 打造語音助理 Pipeline (Build a Voice Assistant Pipeline) — Phase 6 總結專題

> 把第 01-11 課的所有內容串在一起。打造一個會聽、會推理、會回話的語音助理。2026 年，這已經是個解決了的工程問題，而不是研究問題，但整合的細節決定了它能不能上線。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 05, 06, 07, 11; Phase 11 · 09 (Function Calling); Phase 14 · 01 (Agent Loop)
**Time:** ~120 分鐘

## 問題 (The Problem)

打造一個端到端的助理：

1. 擷取麥克風輸入（16 kHz mono）。
2. 偵測使用者語音的開始與結束。
3. 以串流方式轉錄。
4. 把逐字稿交給可呼叫工具（計時器、天氣、行事曆）的 LLM。
5. 把 LLM 的文字串流給 TTS。
6. 把音訊播放給使用者。
7. 如果使用者在回應途中打斷，就停止。

延遲目標：在筆電 CPU 上，使用者說完話後 800 ms 內輸出第一個 TTS 音訊位元組。品質目標：不漏字、在靜音上不產生幻覺字幕、不洩漏 voice cloning、不被 prompt injection 成功攻擊。

## 核心概念 (The Concept)

![Voice assistant pipeline: mic → VAD → STT → LLM+tools → TTS → speaker](../assets/voice-assistant.svg)

### 七個元件

1. **Audio capture。** Mic → 16 kHz mono → 20 ms chunks。Python 中通常用 `sounddevice`，正式環境則用原生的 AudioUnit/ALSA/WASAPI。
2. **VAD（Lesson 11）。** Silero VAD，threshold 0.5、最短語音 250 ms、靜音 hang-over 500 ms。發出「開始」與「結束」訊號。
3. **Streaming STT（Lesson 4-5）。** Whisper-streaming、Parakeet-TDT，或 Deepgram Nova-3（API）。包含部分與最終逐字稿。
4. **具備 tool calling 的 LLM。** GPT-4o / Claude 3.5 / Gemini 2.5 Flash。工具使用 JSON schema。串流 tokens。
5. **Streaming TTS（Lesson 7）。** Kokoro-82M（最快的開源方案）或 Cartesia Sonic（商用）。在 LLM 產生 20 個 tokens 之後啟動 TTS。
6. **播放。** 輸出到喇叭；低頻寬網路則用 opus 編碼。
7. **中斷處理器。** 如果 TTS 播放期間 VAD 觸發，就停止播放、取消 LLM，並重新啟動 STT。

### 你一定會遇到的三種失敗模式

1. **第一個字被切掉。** VAD 晚了一拍才啟動，使用者的 "hey" 不見了。把啟動 threshold 設成 0.3，而不是 0.5。
2. **回應中途被打斷時的混亂。** 使用者打斷後，LLM 仍繼續生成；助理蓋過使用者的聲音說話。請把 VAD 接到 cancel-LLM。
3. **靜音幻覺。** Whisper 在靜音的暖機 frames 上輸出 "Thanks for watching"。一律要用 VAD 把關。

### 2026 年的正式環境參考架構

| 架構 | 延遲 | 授權 | 備註 |
|-------|---------|---------|-------|
| LiveKit + Deepgram + GPT-4o + Cartesia | 350-500 ms | commercial API | 2026 年業界預設 |
| Pipecat + Whisper-streaming + GPT-4o + Kokoro | 500-800 ms | mostly open | 適合自己動手做 |
| Moshi (full-duplex) | 200-300 ms | CC-BY 4.0 | 單一模型；架構不同，見第 15 課 |
| Vapi / Retell (managed) | 300-500 ms | commercial | 上線最快；客製化有限 |
| Whisper.cpp + llama.cpp + Kokoro-ONNX | offline | open | 隱私 / 邊緣 |

```figure
v4-voice-latency
```

## 動手做 (Build It)

### 步驟 1：切塊的麥克風擷取（pseudocode）

```python
import sounddevice as sd

def mic_stream(chunk_ms=20, sr=16000):
    q = queue.Queue()
    def cb(indata, frames, time, status):
        q.put(indata.copy().flatten())
    with sd.InputStream(channels=1, samplerate=sr, blocksize=int(sr * chunk_ms/1000), callback=cb):
        while True:
            yield q.get()
```

### 步驟 2：VAD 把關的輪次擷取

```python
def capture_turn(stream, vad, pre_roll_ms=300, silence_ms=500):
    buf, pre, triggered = [], collections.deque(maxlen=pre_roll_ms // 20), False
    silent = 0
    for chunk in stream:
        pre.append(chunk)
        if vad(chunk):
            if not triggered:
                buf = list(pre)
                triggered = True
            buf.append(chunk)
            silent = 0
        elif triggered:
            silent += 20
            buf.append(chunk)
            if silent >= silence_ms:
                return b"".join(buf)
```

### 步驟 3：Streaming STT → LLM → TTS

```python
async def turn(audio_bytes):
    transcript = await stt.transcribe(audio_bytes)
    async for token in llm.stream(transcript):
        async for audio in tts.stream(token):
            await speaker.play(audio)
```

### 步驟 4：LLM 迴圈內的 tool calling

```python
tools = [
    {"name": "get_weather", "parameters": {"location": "string"}},
    {"name": "set_timer", "parameters": {"seconds": "int"}},
]

async for chunk in llm.stream(user_text, tools=tools):
    if chunk.type == "tool_call":
        result = dispatch(chunk.name, chunk.args)
        continue_streaming(result)
    if chunk.type == "text":
        await tts.stream(chunk.text)
```

### 步驟 5：中斷處理

```python
tts_task = asyncio.create_task(tts_loop())
while True:
    chunk = await mic.get()
    if vad(chunk):
        tts_task.cancel()
        await speaker.stop()
        await new_turn()
        break
```

## 實際使用 (Use It)

請見 `code/main.py`，那是一個可執行的模擬，用 stub 模型把七個元件全部接起來，即使沒有硬體也能看到 pipeline 的形狀。要做真正的實作，請把 stubs 換成：

- `silero-vad`（`pip install silero-vad`）
- `deepgram-sdk` 或 `openai-whisper`
- `openai`（`gpt-4o`）或 `anthropic`
- `kokoro` 或 `cartesia`
- 用 `sounddevice` 做 I/O

## 陷阱 (Pitfalls)

- **永遠記錄 PII。** 在大多數司法管轄區，完整輪次的音訊都屬於 PII。保存 30 天，並做靜態加密。
- **沒有 barge-in。** 使用者一定會打斷。你的助理必須停止說話。
- **會阻塞的 TTS。** 同步的 TTS 會阻塞 event loop。請使用 async 或獨立的 thread。
- **沒有處理 tool-call 錯誤。** 工具會失敗。LLM 必須拿到錯誤訊息並重試一次，之後再優雅降級。
- **過度積極的幻覺過濾。** 過濾過頭，助理會一直重複「我無法協助這件事」。過濾不足，它什麼都敢說。請在 held-out 資料集上校準。
- **沒有喚醒詞選項。** 一直在聽是隱私上的負擔。請加上喚醒詞閘門（Porcupine 或 openWakeWord）。

## 交付 (Ship It)

存成 `outputs/skill-voice-assistant-architect.md`。給定預算 + 規模 + 語言 + 合規限制，產出完整的架構規格。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它用 stub 模組模擬一個完整輪次的端到端流程，並印出各階段延遲。
2. **中等。** 把 STT stub 換成真正的 Whisper 模型，處理預先錄好的 `.wav`。量測 WER 與端到端延遲。
3. **困難。** 加入 tool calling：實作 `get_weather`（任何 API 皆可）和 `set_timer`。讓 LLM 透過這些工具運作，並驗證當使用者說 "set a 5 minute timer" 時，會觸發正確的函式，而且語音回覆會確認這件事。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Turn | 使用者 + 助理的一來一回 | 一段由 VAD 界定的使用者語音，加上一次 LLM-TTS 的回應。 |
| Barge-in | 打斷 | 助理說話時使用者開口；助理停止。 |
| Wake word | "Hey assistant" | 簡短的關鍵詞偵測器；Porcupine、Snowboy、openWakeWord。 |
| End-pointing | 輪次結束 | VAD + 最短靜音，用來判定使用者已說完。 |
| Pre-roll | 說話前的 buffer | 在 VAD 觸發前先保留 200-400 ms 音訊，避免第一個字被切掉。 |
| Tool call | 函式呼叫 | LLM 輸出 JSON；runtime 派送；結果回饋到迴圈中。 |

## 延伸閱讀 (Further Reading)

- [LiveKit — voice agent quickstart](https://docs.livekit.io/agents/)：正式環境等級的參考。
- [Pipecat — voice agent examples](https://github.com/pipecat-ai/pipecat)：適合自己動手做的框架。
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime)：託管的原生語音路線。
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi)：全雙工參考（第 15 課）。
- [Porcupine wake-word](https://picovoice.ai/products/porcupine/)：喚醒詞閘門。
- [Anthropic — tool use guide](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)：LLM function calling。
