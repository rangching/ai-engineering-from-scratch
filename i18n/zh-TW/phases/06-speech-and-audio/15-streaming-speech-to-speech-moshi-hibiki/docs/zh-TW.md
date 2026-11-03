# 串流語音轉語音 (Streaming Speech-to-Speech) — Moshi、Hibiki 與全雙工對話

> 2024-2026 年重新定義了語音 AI。Moshi 推出單一模型，能以 200 ms 延遲同時聆聽與說話。Hibiki 逐 chunk 進行語音轉語音翻譯。兩者都捨棄了 ASR → LLM → TTS 的 pipeline，改用建立在 Mimi codec tokens 上的統一全雙工架構。這是新的參考設計。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 13 (Neural Audio Codecs), Phase 6 · 11 (Real-Time Audio), Phase 7 · 05 (Full Transformer)
**Time:** ~75 分鐘

## 問題 (The Problem)

用第 11 + 12 課打造的每個語音 agent，延遲下限都在 300-500 ms 左右：VAD 觸發、STT 處理、LLM 推理、TTS 生成。每個階段都有各自的最低延遲。你可以調校並平行化，但 pipeline 的形狀限制了你。

Moshi（Kyutai，2024-2026）問了一個不同的問題：如果根本沒有 pipeline 呢？如果由單一模型直接接收音訊、輸出音訊，而且是連續進行，文字只是中間的「內心獨白 (inner monologue)」，而不是必要的階段呢？

答案就是**全雙工語音轉語音 (full-duplex speech-to-speech)**。理論延遲 160 ms（80 ms Mimi frame + 80 ms 聲學延遲）。實際延遲在單張 L4 GPU 上是 200 ms，只有同級最佳的 pipeline 式語音 agent 的一半。

## 核心概念 (The Concept)

![Moshi architecture: two parallel Mimi streams + inner-monologue text](../assets/moshi-hibiki.svg)

### Moshi 架構

**輸入。** 兩條 Mimi codec 串流，都是 12.5 Hz × 8 個 codebooks：

- 串流 1：使用者音訊（Mimi 編碼，持續到達）
- 串流 2：Moshi 自己的音訊（由 Moshi 生成）

**Transformer。** 一個 7B 參數的 Temporal Transformer 處理這兩條串流，以及一條文字「內心獨白」串流。在每個 80 ms 的步驟，它會：

1. 讀入最新的使用者 Mimi tokens（8 個 codebooks）。
2. 讀入最近的 Moshi Mimi tokens（8 個 codebooks，隨生成而來）。
3. 生成下一個 Moshi 文字 token（內心獨白）。
4. 生成下一個 Moshi Mimi tokens（透過小型 Depth Transformer 生成 8 個 codebooks）。

三條串流（使用者音訊、Moshi 音訊、Moshi 文字）並行運作。Moshi 說話時也能聽到使用者；使用者打斷時能打斷自己；還能附和（"mhm"）而不中斷主要的語句。

**Depth transformer。** 在一個 frame 之內，8 個 codebooks 不是並行預測的，它們之間有相依關係。一個小型的 2 層「depth transformer」會在 80 ms 內依序預測它們。這是 AR codec LM 的標準分解方式（VALL-E、VibeVoice 也這麼做）。

### 為什麼內心獨白文字有幫助

沒有明確的文字，模型就得在它的聲學串流中隱含地為語言建模。Moshi 的洞見：強迫它在輸出音訊的同時也輸出文字 tokens。文字串流基本上就是 Moshi 所說內容的逐字稿。這能提升語意連貫性，讓更換語言模型 head 更容易，還免費附贈逐字稿。

### Hibiki：串流語音轉語音翻譯

相同的架構，改用翻譯配對資料訓練。輸入來源語言音訊，持續輸出目標語言音訊。Hibiki-Zero（2026 年 2 月）不再需要字層級對齊的訓練資料，改用句子層級的資料 + GRPO 強化學習來最佳化延遲。

最初支援四組語言配對；大約有 1000 小時的資料，就能適應新的語言。

### Kyutai 更完整的工具組合 (2026)

- **Moshi**：全雙工對話（先支援法文，英文支援良好）
- **Hibiki / Hibiki-Zero**：同步語音翻譯
- **Kyutai STT**：streaming ASR（500 ms 或 2.5 秒 look-ahead）
- **Kyutai Pocket TTS**：100M 參數的 TTS，可在 CPU 上執行（2026 年 1 月）
- **Unmute**：在公開伺服器上結合以上所有元件的完整 pipeline

L40S GPU 上的吞吐量：64 個並行 session，3 倍即時速度。

### Sesame CSM：表親

Sesame CSM（2025）用了類似的概念：Llama-3 backbone 加上 Mimi codec head。但 CSM 是單向的（輸入 context + 文字，產生語音），而不是全雙工。它是市面上最好的「語音臨場感」TTS，但和 Moshi 的全雙工能力不完全相同。

### 2026 年的效能數字

| Model | 延遲 | 使用情境 | 授權 |
|-------|---------|----------|---------|
| Moshi | 200 ms (L4) | 全雙工英文 / 法文對話 | CC-BY 4.0 |
| Hibiki | 12.5 Hz framerate | 法文 ↔ 英文串流翻譯 | CC-BY 4.0 |
| Hibiki-Zero | same | 5 組語言配對，不需對齊資料 | CC-BY 4.0 |
| Sesame CSM-1B | 200 ms TTFA | 以 context 為條件的 TTS | Apache-2.0 |
| GPT-4o Realtime | ~300 ms | 閉源，OpenAI API | commercial |
| Gemini 2.5 Live | ~350 ms | 閉源，Google API | commercial |

```figure
sp-fullduplex
```

## 動手做 (Build It)

### 步驟 1：介面

Moshi 提供一個 WebSocket 伺服器，接收 80 ms 的 Mimi 編碼音訊 chunks，並回傳 80 ms 的 Mimi 編碼音訊 chunks。雙向，持續不斷。

```python
import asyncio
import websockets
from moshi.client_utils import encode_audio_mimi, decode_audio_mimi

async def moshi_chat():
    async with websockets.connect("ws://localhost:8998/api/chat") as ws:
        mic_task = asyncio.create_task(stream_mic_to(ws))
        spk_task = asyncio.create_task(stream_from_to_speaker(ws))
        await asyncio.gather(mic_task, spk_task)
```

### 步驟 2：全雙工迴圈

```python
async def stream_mic_to(ws):
    async for chunk_80ms in mic_stream_at_12_5_hz():
        mimi_tokens = encode_audio_mimi(chunk_80ms)
        await ws.send(serialize(mimi_tokens))

async def stream_from_to_speaker(ws):
    async for msg in ws:
        mimi_tokens, text_token = deserialize(msg)
        audio = decode_audio_mimi(mimi_tokens)
        await play(audio)
```

兩個方向同時進行。Python asyncio 或 Rust futures 是標準的傳輸方式。

### 步驟 3：訓練目標（概念性）

對每個 80 ms 的 frame `t`：

- 輸入：`user_mimi[0..t]`、`moshi_mimi[0..t-1]`、`moshi_text[0..t-1]`
- 預測：`moshi_text[t]`，接著是 `moshi_mimi[t, codebook_0..7]`

文字先於音訊被預測（內心獨白）；音訊在 depth transformer 內依 codebook 順序預測。

### 步驟 4：Moshi 贏在哪、輸在哪

Moshi 贏在：

- 在便宜的硬體上端到端低於 250 ms。
- 自然的附和 (back-channels) 與打斷。
- 不需要 pipeline 的黏合程式碼。

Moshi 贏不了的地方：

- Tool calling（沒有針對它訓練；你需要另外的 LLM 路徑）。
- 長時間推理（Moshi 是約 8B 的對話模型，不是 Claude/GPT-4）。
- 冷門主題的事實準確度。
- 大多數正式環境的企業使用情境（2026 年仍然使用 pipeline）。

## 實際使用 (Use It)

| 情境 | 選擇 |
|-----------|------|
| 最低延遲的語音陪伴 | Moshi |
| 即時翻譯通話 | Hibiki |
| 語音 demo / 研究 | Moshi、CSM |
| 帶工具的企業 agent | Pipeline（第 12 課），而不是 Moshi |
| 有 context 的自訂聲音 TTS | Sesame CSM |
| 任何語言的語音轉語音 | GPT-4o Realtime 或 Gemini 2.5 Live（商用） |

## 陷阱 (Pitfalls)

- **有限的 tool calling。** Moshi 是對話模型，不是 agent 框架。需要工具時請與 pipeline 結合。
- **特定聲音的條件控制。** Moshi 使用單一訓練出來的角色；要複製聲音需要另外跑一次訓練。
- **語言覆蓋範圍。** 法文 + 英文很出色；其他語言有限。Hibiki-Zero 有幫助，但你仍然需要訓練資料。
- **資源成本。** 一個完整的 Moshi session 會佔用一個 GPU 名額；不是便宜的共享租戶部署模式。

## 交付 (Ship It)

存成 `outputs/skill-duplex-pipeline.md`。針對語音 agent 的工作負載，在 pipeline 與全雙工架構之間做選擇，並說明理由。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它以符號方式模擬雙串流 + 內心獨白的架構。
2. **中等。** 從 HuggingFace 拉下 Moshi，執行伺服器，測試一段對話。量測從使用者說完到 Moshi 開始回應的實際牆上時間延遲。
3. **困難。** 拿你第 12 課的 pipeline agent，用 20 句配對的測試語句，比較它與 Moshi 的 P50 延遲。寫下什麼時候 pipeline 在架構上仍然勝出。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Full-duplex | 同時聽與說 | 同一個模型上同時有兩條音訊串流在運作。 |
| Inner monologue | 模型的文字串流 | Moshi 在輸出音訊的同時也輸出文字 tokens。 |
| Depth transformer | Codebook 之間的預測器 | 在單一 80 ms frame 內預測 8 個 codebooks 的小型 transformer。 |
| Mimi | Kyutai 的 codec | 12.5 Hz × 8 個 codebooks；語意+聲學；驅動 Moshi。 |
| Streaming S2S | 即時音訊 → 音訊 | 逐 chunk 的翻譯/對話，沒有 pipeline 階段。 |
| Back-channeling | 「嗯哼」之類的回應 | Moshi 可以發出簡短的附和，而不中斷自己的輪次。 |

## 延伸閱讀 (Further Reading)

- [Défossez et al. (2024). Moshi — speech-text foundation model](https://arxiv.org/html/2410.00037v2)：論文本身。
- [Kyutai Labs (2026). Hibiki-Zero](https://arxiv.org/abs/2602.12345)：不需對齊資料的串流翻譯。
- [Sesame (2025). Crossing the uncanny valley of voice](https://www.sesame.com/research/crossing_the_uncanny_valley_of_voice)：CSM 規格。
- [Kyutai — Moshi repo](https://github.com/kyutai-labs/moshi)：安裝 + 伺服器。
- [OpenAI — Realtime API](https://platform.openai.com/docs/guides/realtime)：閉源的商用同類產品。
- [Kyutai — Delayed Streams Modeling](https://github.com/kyutai-labs/delayed-streams-modeling)：底層的 STT/TTS 框架。
