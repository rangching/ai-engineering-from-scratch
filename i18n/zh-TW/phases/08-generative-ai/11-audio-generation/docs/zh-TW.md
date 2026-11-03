# 音訊生成 (Audio Generation)

> 音訊是 16-48 kHz 的 1-D 訊號。一段五秒的片段有 80-240k 個樣本。沒有 transformer 能直接對這麼長的序列做 attention。2026 年每個正式音訊模型的解法都一樣：用 neural codec（Encodec、SoundStream、DAC）把音訊壓縮成 50-75 Hz 的離散 token，再由 transformer 或 diffusion model 生成 token。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Audio Features), Phase 6 · 04 (ASR), Phase 8 · 06 (DDPM)
**Time:** ~45 分鐘

## 問題 (The Problem)

三種音訊生成任務：

1. **Text-to-speech。** 給定文字，產生語音。乾淨的語音是窄頻帶、語音結構很強，用 transformer-over-tokens 就能解得很好。VALL-E (Microsoft)、NaturalSpeech 3、ElevenLabs、OpenAI TTS。
2. **音樂生成。** 給定 prompt（文字、旋律、和弦進行、曲風），產生音樂。分佈廣得多。MusicGen (Meta)、Stable Audio 2.5、Suno v4、Udio、Riffusion。
3. **音效 / 聲音設計。** 給定 prompt，產生環境音或 Foley 音效。AudioGen、AudioLDM 2、Stable Audio Open。

三者都跑在同一個底座上：neural audio codec + token-AR 或 diffusion 的 generator。

## 核心概念 (The Concept)

![音訊生成：codec token + transformer 或 diffusion](../assets/audio-generation.svg)

### Neural audio codec

Encodec (Meta, 2022)、SoundStream (Google, 2021)、Descript Audio Codec (DAC, 2023)。卷積 encoder 把波形壓縮成每個 timestep 一個向量；residual vector quantization (RVQ) 把每個向量轉成 K 個 codebook 索引的串接。Decoder 則反向還原。24 kHz 音訊在 2 kbps 下，以 75 Hz 使用 8 個 RVQ codebook = 600 tokens/sec。

```
waveform (16000 samples/sec)
    └─ encoder conv ─┐
                     ├─ RVQ layer 1 → indices at 75 Hz
                     ├─ RVQ layer 2 → indices at 75 Hz
                     ├─ ...
                     └─ RVQ layer 8
```

### 其上的兩種生成範式

**Token-autoregressive。** 把 RVQ token 攤平成序列，跑 decoder-only transformer。MusicGen 使用「delayed parallel」，以各自錯開的偏移量平行輸出 K 條 codebook 串流。VALL-E 則根據文字 prompt + 3 秒的聲音樣本生成語音 token。

**Latent diffusion。** 把 codec token 打包成連續的 latent，或用 categorical diffusion 對它們建模。Stable Audio 2.5 對連續的音訊 latent 使用 flow matching。AudioLDM 2 使用 text-to-mel-to-audio 的 diffusion。

2024-2026 年的趨勢：flow matching 在音樂上勝出（推論較快、樣本較乾淨），而 token-AR 仍主導語音，因為它天生因果性，且很適合串流。

## 正式產品版圖 (Production landscape)

| 系統 | 任務 | 骨幹 | 延遲 |
|------|------|------|------|
| ElevenLabs V3 | TTS | Token-AR + neural vocoder | 首個 token 約 300ms |
| OpenAI GPT-4o audio | 全雙工語音 | 端到端多模態 AR | ~200ms |
| NaturalSpeech 3 | TTS | Latent flow matching | 非串流 |
| Stable Audio 2.5 | 音樂 / SFX | 在音訊 latent 上的 DiT + flow matching | 1 分鐘片段約 10s |
| Suno v4 | 完整歌曲 | 未公開；懷疑是 token-AR | 每首歌約 30s |
| Udio v1.5 | 完整歌曲 | 未公開 | 每首歌約 30s |
| MusicGen 3.3B | 音樂 | 基於 Encodec 32kHz 的 Token-AR | 即時 |
| AudioCraft 2 | 音樂 + SFX | Flow matching | 5s 片段約 5s |
| Riffusion v2 | 音樂 | 頻譜圖 diffusion | ~10s |

```figure
score-matching
```

## 動手做 (Build It)

`code/main.py` 模擬核心概念：在合成的「音訊 token」序列上訓練一個小型 next-token transformer，序列由兩種不同的「風格」產生（風格 A 是低 token 與高 token 交替，風格 B 是單調遞增）。以風格為條件並進行 sampling。

### 步驟 1：合成的音訊 token

```python
def make_tokens(style, length, vocab_size, rng):
    if style == 0:  # "speech-like": alternating
        return [i % vocab_size for i in range(length)]
    # "music-like": ramp
    return [(i * 3) % vocab_size for i in range(length)]
```

### 步驟 2：訓練一個小型 token 預測器

一個以風格為條件的 bigram 式預測器。重點在於這個模式：codec token → cross-entropy 訓練 → autoregressive sampling。

### 步驟 3：條件式 sampling

給定風格 token 和一個起始 token，從預測出的分佈中 sample 下一個 token。持續 20-40 個 token。

## 常見陷阱 (Pitfalls)

- **Codec 的品質決定輸出品質的上限。** 如果 codec 無法忠實表示某個聲音，再好的 generator 也沒用。DAC 是目前開放權重中最好的。
- **RVQ 的誤差累積。** 每一層 RVQ 都對前一層的殘差建模。第 1 層的誤差會往後傳。在較高的層以 temperature 0 做 sampling 有幫助。
- **音樂結構。** 30 秒的 token 在 75 Hz 下是 20k+ 個 token。對 transformer 來說很困難。MusicGen 使用滑動視窗 + prompt 延續；Stable Audio 使用較短的片段 + 交叉淡化。
- **邊界處的瑕疵。** 在生成的片段之間做交叉淡化，需要仔細的 overlap-add。
- **對乾淨資料的胃口。** 音樂生成器需要數萬小時有授權的音樂。Suno / Udio 的 RIAA 訴訟（2024）讓這個問題浮上檯面。
- **聲音複製的倫理。** 3 秒的樣本加上文字 prompt，就足以讓 VALL-E / XTTS / ElevenLabs 複製一個聲音。每個正式模型都需要濫用偵測 + 選擇退出名單。

## 實際使用 (Use It)

| 任務 | 2026 年的技術組合 |
|------|-----------------|
| 商用 TTS | ElevenLabs、OpenAI TTS 或 Azure Neural |
| 聲音複製（經同意驗證） | XTTS v2（開放）或 ElevenLabs Pro |
| 背景音樂，快速 | Stable Audio 2.5 API、Suno 或 Udio |
| 帶歌詞的音樂 | Suno v4 或 Udio v1.5 |
| 音效 / Foley | AudioCraft 2、ElevenLabs SFX 或 Stable Audio Open |
| 即時語音 agent | GPT-4o realtime 或 Gemini Live |
| 開放權重的音樂研究 | MusicGen 3.3B、Stable Audio Open 1.0、AudioLDM 2 |
| 配音 / 翻譯 | HeyGen、ElevenLabs Dubbing |

## 交付 (Ship It)

存成 `outputs/skill-audio-brief.md`。這個 skill 接收音訊簡報（任務、長度、風格、聲音、授權），輸出：模型 + 託管方式、prompt 格式（曲風標籤、風格描述詞、結構標記）、codec + generator + vocoder 的處理鏈、seed 流程，以及評估計畫（MOS / CLAP score / TTS 的 CER / 使用者 A/B）。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py` 並明確指定風格。驗證生成的序列符合該風格的模式。
2. **中等。** 加入 delayed parallel decoding：模擬 2 條必須保持 1 步偏移的 token 串流。訓練一個聯合預測器。
3. **困難。** 使用 HuggingFace transformers 在本機執行 MusicGen-small。用三種不同的 prompt 各生成一段 10 秒的片段；做 A/B 比較風格遵循度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| Codec | 「Neural 壓縮」 | 音訊的 encoder / decoder；典型輸出是 50-75 Hz 的 token。 |
| RVQ | 「Residual VQ」 | K 個 quantizer 的串接；每一個都對前一個的殘差建模。 |
| Token | 「一個 codec 符號」 | codebook 的離散索引；通常是 1024 或 2048。 |
| Delayed parallel | 「錯開的 codebook」 | 以錯開的偏移量輸出 K 條 token 串流，以縮短序列長度。 |
| Flow matching | 「2024 年音訊的突破」 | diffusion 的較直路徑替代方案；sampling 更快。 |
| Voice prompt | 「3 秒樣本」 | 引導複製聲音的 speaker embedding 或 token 前綴。 |
| Mel spectrogram | 「視覺化的那個」 | 對數幅度的感知頻譜圖；許多 TTS 系統使用。 |
| Vocoder | 「Mel 轉波形」 | 把 mel 頻譜圖轉回音訊的 neural 元件。 |

## 實務筆記 (Production note)：音訊是串流問題

音訊是唯一一種使用者期待 *邊生成邊收到*，而不是一次到位的輸出模態。在正式環境中，這代表 TPOT（Time Per Output Token）很重要，因為目標吞吐量是使用者的聆聽速度，而不是閱讀速度。對以約 75 tokens/秒 token 化的 16kHz 音訊（Encodec）而言，伺服器必須對每個使用者生成 ≥75 tokens/秒，才能維持流暢播放。

兩個架構上的結果：

- **Flow-matching 音訊模型無法輕易串流。** Stable Audio 2.5 和 AudioCraft 2 一次 pass 就渲染固定長度的片段。要串流，你得把片段切塊並讓邊界重疊（類似滑動視窗 diffusion），相較 codec AR 模型要多增加 100-300ms 的延遲開銷。

如果產品是「即時語音聊天」或「即時音樂接續」，請選 codec AR 路線。如果是「按下送出後渲染 30 秒的片段」，flow-matching 在品質與總延遲上勝出。

## 延伸閱讀 (Further Reading)

- [Défossez et al. (2022). Encodec: High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — codec 標準。
- [Zeghidour et al. (2021). SoundStream](https://arxiv.org/abs/2107.03312) — 第一個被廣泛使用的 neural audio codec。
- [Kumar et al. (2023). High-Fidelity Audio Compression with Improved RVQGAN (DAC)](https://arxiv.org/abs/2306.06546) — DAC。
- [Wang et al. (2023). Neural Codec Language Models are Zero-Shot Text to Speech Synthesizers (VALL-E)](https://arxiv.org/abs/2301.02111) — VALL-E。
- [Copet et al. (2023). Simple and Controllable Music Generation (MusicGen)](https://arxiv.org/abs/2306.05284) — MusicGen。
- [Liu et al. (2023). AudioLDM 2: Learning Holistic Audio Generation with Self-supervised Pretraining](https://arxiv.org/abs/2308.05734) — AudioLDM 2。
- [Stability AI (2025). Stable Audio 2.5](https://stability.ai/news-updates/stability-ai-introduces-stable-audio-25-the-first-audio-model-built-for-enterprise-sound-production-at-scale) — 2025 年搭配 flow matching 的 text-to-music。
