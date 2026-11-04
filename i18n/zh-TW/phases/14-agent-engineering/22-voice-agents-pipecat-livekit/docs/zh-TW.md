# Voice Agent：Pipecat 與 LiveKit (Voice Agents: Pipecat and LiveKit)

> 2026 年，voice agent 是一個一級的正式環境類別。Pipecat 提供以 Python frame 為基礎的 pipeline（VAD → STT → LLM → TTS → transport）。LiveKit Agents 則透過 WebRTC 把 AI 模型橋接給使用者。高階技術堆疊的正式環境延遲目標落在端到端 450–600ms。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 12 (Workflow Patterns)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 Pipecat 以 frame 為基礎的 pipeline：DOWNSTREAM（source→sink）和 UPSTREAM（控制）。
- 說出標準 voice pipeline 的各階段，以及 Pipecat 支援哪些 transport。
- 解釋 LiveKit Agents 的兩個 voice agent 類別（MultimodalAgent、VoicePipelineAgent）以及各自適用的時機。
- 總結 2026 年正式環境的延遲期望，以及它們如何驅動架構選擇。

## 問題 (The Problem)

Voice agent 不是外加 TTS 的文字 loop。延遲預算非常嚴苛（約 600ms）、部分音訊是預設狀態、輪流偵測本身就是一個模型，而 transport 從電話 SIP 到 WebRTC 都有。你要嘛建一個以 frame 為基礎的 pipeline（Pipecat），要嘛倚賴一個平台（LiveKit）。

## 核心概念 (The Concept)

### Pipecat (pipecat-ai/pipecat)

- Python、以 frame 為基礎的 pipeline framework。
- `Frame` → `FrameProcessor` 鏈。
- 兩個流動方向：
  - **DOWNSTREAM**：source → sink（輸入音訊，輸出 TTS）。
  - **UPSTREAM**：回饋與控制（取消、指標、barge-in）。
- `PipelineTask` 以事件（`on_pipeline_started`、`on_pipeline_finished`、`on_idle_timeout`）管理生命週期，並以 observer 處理指標/tracing/RTVI。

典型的 pipeline：

```
VAD (Silero) → STT → LLM (context alternates user/assistant) → TTS → transport
```

Transport：Daily、LiveKit、SmallWebRTCTransport、FastAPI WebSocket、WhatsApp。

Pipecat Flows 加入了結構化對話（狀態機）。Pipecat Cloud 是託管的 runtime。

### LiveKit Agents (livekit/agents)

- 透過 WebRTC 把 AI 模型橋接給使用者。
- 關鍵概念：`Agent`、`AgentSession`、`entrypoint`、`AgentServer`。
- 兩個 voice agent 類別：
  - **MultimodalAgent**：透過 OpenAI Realtime 或同等服務直接處理音訊。
  - **VoicePipelineAgent**：STT → LLM → TTS 串接；提供文字層級的控制。
- 透過 transformer 模型做語意輪流偵測。
- 原生 MCP 整合。
- 透過 SIP 支援電話。
- 透過 LiveKit Inference 提供 50 個以上不需 API key 的模型；另有 200 個以上透過 plugin 提供。

### 商用平台 (Commercial platforms)

Vapi（優化過的高階技術堆疊上約 450–600ms）和 Retell（180 通測試電話的端到端約 600ms）建立在這些之上。當你想要託管的語音技術堆疊、又沒有 WebRTC 團隊時，就選平台。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **沒有處理 barge-in。** 使用者打斷；agent 還繼續講。在 Pipecat 需要 UPSTREAM 取消 frame，LiveKit 有對應機制。
- **忽略 STT 信心值。** 低信心的轉錄文字被當成聖旨餵給 LLM。依信心值設閘門，或請使用者確認。
- **TTS 在句子中間被截斷。** 當 pipeline 在說話中途取消時，TTS 需要知道，否則要截斷音訊。
- **忽略延遲預算。** 每個元件都會增加 50–200ms。出貨前先把你的鏈加總。

### 2026 年的典型延遲 (Typical 2026 latencies)

- VAD：20–60ms
- STT partial：100–250ms
- LLM 第一個 token：150–400ms
- TTS 第一段音訊：100–200ms
- Transport RTT：30–80ms

端到端 450–600ms 屬於高階。800–1200ms 很常見。超過 1500ms 會讓人覺得壞掉了。

```figure
voice-pipeline
```

## 動手做 (Build It)

`code/main.py` 是一個以 frame 為基礎的 toy pipeline，包含：

- `Frame` 型別（audio、transcript、text、tts_audio、control）。
- 具有 `process(frame)` 的 `Processor` 介面。
- 以腳本化 processor 組成的五階段 pipeline（VAD → STT → LLM → TTS → transport）。
- 一個用來展示 barge-in 的 UPSTREAM 取消 frame。

執行：

```
python3 code/main.py
```

Trace 顯示正常流程，以及一次在說話中途停止 TTS 的 barge-in 取消。

## 實際使用 (Use It)

- **Pipecat** 用於完整控制：自訂 processor、Python 優先、可抽換的 provider。
- **LiveKit Agents** 用於以 WebRTC 為優先的部署與電話。
- **Vapi / Retell** 用於沒有 WebRTC 團隊的託管 voice agent。
- **OpenAI Realtime / Gemini Live** 用於直接的音訊輸入/音訊輸出（MultimodalAgent）。

## 交付 (Ship It)

`outputs/skill-voice-pipeline.md` 建立 Pipecat 形狀的 voice pipeline 骨架，包含 VAD + STT + LLM + TTS + transport 以及 barge-in 處理。

## 練習 (Exercises)

1. 為你的 toy pipeline 加一個指標 observer：計算每個階段每秒的 frame 數。延遲在哪裡累積？
2. 實作以信心值為閘門的 STT：低於門檻時，請求「可以再說一次嗎？」
3. 加入語意輪流偵測：簡單規則，如果轉錄文字以 "?" 結尾，就是輪流結束。
4. 讀 Pipecat 的 transport 文件。把 stdlib transport 換成 SmallWebRTCTransport 設定（stub）。
5. 在同一個查詢上量測 OpenAI Realtime 與 STT+LLM+TTS 串接。文字層級的控制要付出多少延遲成本？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Frame | 「事件」 | Pipeline 中有型別的資料單位（audio、transcript、text、control） |
| Processor | 「Pipeline 階段」 | 具有 process(frame) 的 handler |
| DOWNSTREAM | 「正向流」 | Source 到 sink：輸入音訊，輸出語音 |
| UPSTREAM | 「回饋流」 | 控制：取消、指標、barge-in |
| VAD | 「Voice activity detection」 | 偵測使用者何時在說話 |
| Semantic turn detection | 「聰明的輪流結束判斷」 | 以模型判斷使用者已經說完 |
| MultimodalAgent | 「直接音訊 agent」 | 輸入音訊、輸出音訊；中間沒有文字 |
| VoicePipelineAgent | 「串接 agent」 | STT + LLM + TTS；文字層級的控制 |

## 延伸閱讀 (Further Reading)

- [Pipecat docs](https://docs.pipecat.ai/getting-started/introduction)：以 frame 為基礎的 pipeline、processor、transport
- [LiveKit Agents docs](https://docs.livekit.io/agents/)：WebRTC + 語音原語
- [Vapi](https://vapi.ai/)：託管的語音平台
- [Retell AI](https://www.retellai.com/)：託管語音，有延遲 benchmark
