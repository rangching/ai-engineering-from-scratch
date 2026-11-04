# Capstone 03 — 即時語音助理 (Real-Time Voice Assistant，ASR 到 LLM 到 TTS)

> 手感對的語音 agent，端到端延遲低於 800ms、知道你什麼時候講完、能處理 barge-in，而且呼叫工具時不會卡住。Retell、Vapi、LiveKit Agents 和 Pipecat 在 2026 年都達到這個水準。它們用的是同一種形狀：串流 ASR、turn-detector、串流 LLM 和串流 TTS，全部透過 WebRTC 串起來，每一跳都有嚴格的延遲預算。自己做一個，衡量 WER、MOS 與誤截斷率，並在封包遺失下執行。

**Type:** Capstone
**Languages:** Python (agent + pipeline), TypeScript (web client)
**Prerequisites:** Phase 6 (speech and audio), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 17 (infrastructure)
**Phases exercised:** P6 · P7 · P11 · P13 · P14 · P17
**Time:** 30 小時

## 問題 (Problem)

語音是 2025-2026 年變動最快的 AI UX 類別。技術天花板每一季都在下降。OpenAI Realtime API、Gemini 2.5 Live、Cartesia Sonic-2、ElevenLabs Flash v3、LiveKit Agents 1.0 和 Pipecat 0.0.70，全都讓低於 800ms 的 first-audio-out 變得可及。標準不只是延遲，而是互動的手感：不打斷使用者、不被打斷、從句子中途的插話恢復、在對話中途呼叫工具又不讓音訊卡住、在不穩的行動網路上存活。

把三個 REST 呼叫接起來是做不到的。架構必須是端到端的管線式串流。做出來之後，失敗模式就會浮現：針對電話音訊調校的 VAD 被背景電視觸發、turn-detector 在等一個永遠不會來的標點、TTS 在輸出前先緩衝 400ms。這個 capstone 的任務，是在負載下逐一修掉這些問題，並發佈一份延遲與品質報告。

## 核心概念 (Concept)

管線有五個串流階段：**audio in** (來自瀏覽器或 PSTN 的 WebRTC)、**ASR** (來自 Deepgram Nova-3 或 faster-whisper 的串流 partial transcript)、**turn detection** (VAD 加上一個小型 turn-detector 模型，讀取 partial transcript 找完成的線索)、**LLM** (一判定回合結束就串流輸出 token)、**TTS** (在第一個 LLM token 後約 200ms 內串流輸出音訊)。

三個橫切關注點。**Barge-in**：使用者在 agent 說話時開口，TTS 會取消，ASR 立刻接手。**工具使用**：對話中途的函式呼叫 (天氣、行事曆) 必須在旁路執行，不讓音訊卡住；延遲超過 300ms 時，agent 會預先送出一個確認 token (「請稍等……」)。**Backpressure**：在封包遺失時，partial transcript 會被保留、VAD 提高 speech-gate 閾值，agent 也避免蓋過尚未確認的訊息。

衡量標準是量化的。在 15 dB SNR 的 Hamming VAD benchmark 上，WER 低於 8%。100 通量測通話的 first-audio-out p50 低於 800ms。誤截斷率低於 3%。TTS 的 MOS 高於 4.2。單一 g5.xlarge 撐住 50 通並行通話。這些數字就是交付成果。

## 架構 (Architecture)

```
browser / Twilio PSTN
        |
        v
   WebRTC / SIP edge
        |
        v
  LiveKit Agents 1.0  (or Pipecat 0.0.70)
        |
   +----+--------------+--------------+-----------------+
   |                   |              |                 |
   v                   v              v                 v
  ASR              VAD v5         turn-detector     side-channel
(Deepgram         (Silero)          (LiveKit)        tools
 Nova-3 /         speech-gate    completion score    (weather,
 Whisper-v3)      per 20ms        on partials        calendar)
   |                   |              |
   +--------+----------+--------------+
            v
        LLM (streaming)
     GPT-4o-realtime / Gemini 2.5 Flash /
     cascaded Claude Haiku 4.5
            |
            v
        TTS streaming
     Cartesia Sonic-2 / ElevenLabs Flash v3
            |
            v
     audio back to caller
            |
            v
   OpenTelemetry voice traces -> Langfuse
```

## 技術堆疊 (Stack)

- 傳輸：LiveKit Agents 1.0 (WebRTC) 加上 Twilio PSTN 閘道；Pipecat 0.0.70 作為替代框架
- ASR：Deepgram Nova-3 (串流，首個 partial 低於 300ms) 或自行架設的 faster-whisper Whisper-v3-turbo
- VAD：Silero VAD v5 加上 LiveKit turn-detector (讀取 partial transcript 的小型 transformer)
- LLM：緊密整合用 OpenAI GPT-4o-realtime、Gemini 2.5 Flash Live，或串接式的 Claude Haiku 4.5 (串流 completion，獨立的音訊路徑)
- TTS：Cartesia Sonic-2 (最低 first-byte)、ElevenLabs Flash v3，或自行架設用的開源 Orpheus
- 工具：天氣/行事曆/預約用 FastMCP 旁路；工具超過 300ms 時，agent 預先送出填充語
- 可觀測性：OpenTelemetry voice spans、可重播音訊的 Langfuse voice traces
- 部署：自行架設 Whisper + Orpheus 用單一 g5.xlarge (24GB VRAM)；最低延遲用託管 API

```figure
ce-voice-latency
```

## 動手做 (Build It)

1. **WebRTC session。** 架起 LiveKit room 和一個串流麥克風音訊的 web client。在伺服器端，附加一個加入該 room 的 agent worker。

2. **ASR 串流。** 把 20ms 的 PCM frame 餵給 Deepgram Nova-3 (或 GPU 上的 faster-whisper)。訂閱 partial 與 final transcript。記錄每個 partial 的延遲。

3. **VAD 與 turn detector。** 在 frame 串流上跑 Silero VAD v5。收到 speech-end 事件時，對最新的 partial transcript 觸發 LiveKit turn-detector。只有在 VAD 判定靜音 500ms 且 turn-detector 的完成分數 > 0.6 時，才確認「回合結束」。

4. **LLM 串流。** 回合結束時，帶著目前的對話加上最終 transcript 開始 LLM 呼叫。串流輸出 token。收到第一個 token 就交給 TTS。

5. **TTS 串流。** Cartesia Sonic-2 串流回傳音訊 chunk。第一個 chunk 必須在第一個 LLM token 後 200ms 內離開伺服器。把 chunk 送進 LiveKit room；client 透過 WebRTC jitter buffer 播放。

6. **Barge-in。** TTS 播放時，VAD 偵測到新的使用者語音，立刻取消 TTS 串流、丟掉剩下的 LLM 輸出，並重新啟動 ASR。發佈一個 `tts_canceled` span。

7. **工具旁路。** 把天氣和行事曆註冊成 function-calling 工具。被呼叫時並行觸發；若 300ms 內沒有解析完成，讓 LLM 送出「請稍等，我查一下」作為填充語；工具回傳後再繼續。

8. **評測 harness。** 錄下 100 通通話。計算 WER (對照 held-out transcript)、誤截斷率 (使用者說到一半時 TTS 被取消)、first-audio-out p50、TTS MOS (人工或 NISQA)，以及 jitter-loss 測試 (丟掉 3% 的封包)。

9. **負載測試。** 用合成通話者在單一 g5.xlarge 上驅動 50 通並行通話。衡量持續的 first-audio-out p95。

## 實際使用 (Use It)

```
caller: "what is the weather in tokyo tomorrow"
[asr  ] partial @280ms: "what is the"
[asr  ] partial @540ms: "what is the weather"
[turn ] completion score 0.82 at @820ms; commit
[llm  ] tool call emitted @880ms
[tool ] weather.tokyo tomorrow -> 68/52 partly cloudy @940ms
[llm  ] first token @990ms
[tts  ] first audio-out @1040ms: "Tokyo tomorrow will be partly cloudy..."
turn latency: 1040ms user-stop -> audio-out
```

## 交付 (Ship It)

`outputs/skill-voice-agent.md` 是交付成果。給定一個領域 (客服、排程或 kiosk)，它會架起一個 LiveKit agent，ASR/VAD/LLM/TTS 管線依衡量標準調校。評分標準：

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 端到端延遲 | 100 通錄製通話的 first-audio-out p50 低於 800ms |
| 20 | 輪流發言品質 | 在 Hamming VAD benchmark 上誤截斷率低於 3% |
| 20 | 工具使用正確性 | 對話中途的工具呼叫能回傳正確資料，且不讓音訊卡住 |
| 20 | 封包遺失下的可靠性 | 注入 3% 封包遺失時的 WER 與輪流發言穩定度 |
| 15 | 評測 harness 完整度 | 以公開設定重現的量測結果 |
| **100** | | |

## 練習 (Exercises)

1. 在 g5.xlarge 上把 Deepgram Nova-3 換成 faster-whisper v3 turbo。衡量延遲與 WER 的差距，找出 CPU 與 GPU 的取捨在哪裡有影響。

2. 加入插話仲裁策略：使用者在工具呼叫期間 barge-in 時，agent 該怎麼做？比較三種策略 (硬取消、等工具完成再停、把下一回合排隊)。

3. 跑一個對抗式 turn-detector 測試：讓使用者在句子中途停頓很久。調整 VAD 靜音閾值與 turn-detector 分數閾值，在不超過 900ms 的前提下達到最低誤截斷率。

4. 透過 Twilio 把同一個 agent 部署到 PSTN。比較 PSTN 與 WebRTC 的 first-audio-out，解釋 jitter-buffer 與 codec 的差異。

5. 為非英語語言 (日語、西班牙語) 加入語音活動偵測。衡量 Silero VAD v5 的誤觸發率，與針對特定語言 fine-tune 的版本相比如何。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Turn detection | 「語句結束」 | 給定 VAD 靜音與 partial transcript，判斷使用者是否說完的分類器 |
| Barge-in | 「插話處理」 | VAD 偵測到新的使用者語音時，在播放中途取消 TTS |
| First-audio-out | 「延遲」 | 從使用者停止說話，到第一個音訊封包離開伺服器的時間 |
| VAD | 「語音閘門」 | 把音訊 frame 分類為語音或靜音的模型；Silero VAD v5 是 2026 年的預設 |
| Jitter buffer | 「音訊平滑」 | client 端的緩衝區，短暫保留封包以吸收網路波動 |
| Filler | 「確認 token」 | 工具很慢時，agent 為避免沉默而送出的簡短語句 |
| MOS | 「平均意見分數」 | 語音品質的感知評分；NISQA 是自動化的替代指標 |

## 延伸閱讀 (Further Reading)

- [LiveKit Agents 1.0](https://github.com/livekit/agents) — 參考用的 WebRTC agent 框架
- [Pipecat](https://github.com/pipecat-ai/pipecat) — 以 Python 為主的替代串流 agent 框架
- [OpenAI Realtime API](https://platform.openai.com/docs/guides/realtime) — 整合式語音模型的參考
- [Deepgram Nova-3 documentation](https://developers.deepgram.com/docs) — 串流 ASR 參考
- [Silero VAD v5](https://github.com/snakers4/silero-vad) — VAD 參考模型
- [Cartesia Sonic-2](https://docs.cartesia.ai) — 低延遲 TTS 參考
- [Retell AI architecture](https://docs.retellai.com) — 正式環境語音 agent 架構
- [Vapi.ai production stack](https://docs.vapi.ai) — 另一個正式環境參考
