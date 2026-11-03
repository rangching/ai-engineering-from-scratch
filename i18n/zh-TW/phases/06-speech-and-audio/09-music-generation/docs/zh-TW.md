# 音樂生成 (Music Generation) — MusicGen、Stable Audio、Suno 與授權地震

> 2026 年的音樂生成：Suno v5 和 Udio v4 主導商用市場；MusicGen、Stable Audio Open 和 ACE-Step 領先開源。技術問題大致解決了。法律問題（Warner Music 5 億美元和解、UMG 和解）則在 2025-2026 年重塑了這個領域。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms), Phase 4 · 10 (Diffusion Models)
**Time:** ~75 分鐘

## 問題 (The Problem)

文字 → 30 秒到 4 分鐘的音樂片段，包含歌詞、人聲與結構。分成三個子問題：

1. **器樂生成。** 像 "lo-fi hip-hop drums with warm keys" 這樣的文字 → 音訊。MusicGen、Stable Audio、AudioLDM。
2. **歌曲生成（含人聲 + 歌詞）。** "Country song about rainy Texas nights" → 完整歌曲。Suno、Udio、YuE、ACE-Step。
3. **條件式 / 可控制。** 延伸既有片段、重新生成橋段、換曲風、分離 stems，或做 inpainting。Udio 的 inpainting + stem separation 是 2026 年大家要追的功能。

## 核心概念 (The Concept)

![Music generation: token-LM vs diffusion, the 2026 model map](../assets/music-generation.svg)

### 基於 neural-codec tokens 的 Token LM

Meta 的 **MusicGen**（2023，MIT）和許多衍生模型：以文字/旋律 embeddings 為條件，autoregressive 地預測 EnCodec tokens（32 kHz、4 個 codebooks），再用 EnCodec 解碼。300M 到 3.3B 個參數。是很強的基準；超過 30 秒就吃力。

**ACE-Step**（開源，4B XL 於 2026 年 4 月發布）把它延伸到以歌詞為條件的完整歌曲生成。是開源社群最接近 Suno 的東西。

### 基於 mels 或 latents 的 Diffusion

**Stable Audio（2023）**與 **Stable Audio Open（2024）**：在壓縮音訊上做 latent diffusion。擅長 loop、音效設計、氛圍質感。不擅長有結構的完整歌曲。

**AudioLDM / AudioLDM2**：透過 T2I 風格的 latent diffusion 做 text-to-audio，推廣到音樂、音效與語音。

### 混合式（正式商用）— Suno、Udio、Lyria

權重不公開。可能是 AR codec LM + 基於 diffusion 的 vocoder，再加上專門的人聲 / 鼓 / 旋律 heads。Suno v5（2026）是 ELO 1293 的品質領先者。Udio v4 加入 inpainting + stem separation（貝斯、鼓、人聲可分別下載）。

### 評估 (Evaluation)

- **FAD (Fréchet Audio Distance)。** 用 VGGish 或 PANNs 特徵，計算生成與真實音訊分布之間 embedding 層級的距離。越低越好。MusicGen small 在 MusicCaps 上為 4.5 FAD；SOTA 約 3.0。
- **Musicality（主觀）。** 人類偏好。Suno v5 以 ELO 1293 領先。
- **文字-音訊對齊。** prompt 與輸出之間的 CLAP score。
- **Musicality 瑕疵。** 節拍錯位的轉場、人聲樂句飄移、超過 30 秒後失去結構。

## 2026 年的模型地圖

| Model | Params | Length | Vocals | License |
|-------|--------|--------|--------|---------|
| MusicGen-large | 3.3B | 30 s | no | MIT |
| Stable Audio Open | 1.2B | 47 s | no | Stability non-commercial |
| ACE-Step XL (Apr 2026) | 4B | &gt; 2 min | yes | Apache-2.0 |
| YuE | 7B | &gt; 2 min | yes, multilingual | Apache-2.0 |
| Suno v5 (closed) | ? | 4 min | yes, ELO 1293 | commercial |
| Udio v4 (closed) | ? | 4 min | yes + stems | commercial |
| Google Lyria 3 (closed) | ? | real-time | yes | commercial |
| MiniMax Music 2.5 | ? | 4 min | yes | commercial API |

## 法律環境 (2025-2026)

- **Warner Music vs Suno 和解。** 5 億美元。WMG 現在對 Suno 上的 AI 肖像、音樂權利與使用者生成的曲目擁有監督權。UMG 與 Udio 也有類似的和解。
- **EU AI Act** + **California SB 942**：AI 生成的音樂必須揭露。
- 採用 MIT 授權的 **Riffusion / MusicGen** 沒有合規包袱，但也沒有可商用的人聲。

可以放心上線的做法：

1. 只生成器樂（MusicGen、Stable Audio Open，輸出為 MIT/CC0）。
2. 使用商用 API（Suno、Udio、ElevenLabs Music），每次生成都有授權。
3. 用自有或已授權的曲庫訓練（大多數企業最後都走這條路）。
4. 為生成結果加上浮水印 + metadata 標記。

```figure
sp-codec-tokens
```

## 動手做 (Build It)

### 步驟 1：用 MusicGen 生成

```python
from audiocraft.models import MusicGen
import torchaudio

model = MusicGen.get_pretrained("facebook/musicgen-small")
model.set_generation_params(duration=10)
wav = model.generate(["upbeat synthwave with driving drums, 128 BPM"])
torchaudio.save("out.wav", wav[0].cpu(), 32000)
```

三種大小：`small`（300M，快）、`medium`（1.5B）、`large`（3.3B）。要驗證「這個點子行不行得通」，small 就夠了。

### 步驟 2：旋律條件

```python
melody, sr = torchaudio.load("humming.wav")
wav = model.generate_with_chroma(
    ["jazz piano cover"],
    melody.squeeze(),
    sr,
)
```

MusicGen-melody 接收 chromagram，保留旋律，同時替換音色。適合用在「把這段旋律改成弦樂四重奏」。

### 步驟 3：FAD 評估

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()

fad.get_fad_score("generated_folder/", "reference_folder/")
```

計算 VGGish embedding 的距離。適合用於曲風層級的回歸測試；無法取代真人聆聽。

### 步驟 4：加進 LLM 音樂工作流程

結合 Lessons 7-8 的想法：

```python
prompt = "Write a 30-second jazz loop. Describe the drums, bass, and piano voicing."
description = llm.complete(prompt)
music = musicgen.generate([description], duration=30)
```

## 實際使用 (Use It)

| 目標 | 工具組合 |
|------|-------|
| 器樂音效設計 | Stable Audio Open |
| 遊戲 / 適應性音樂 | Google Lyria RealTime（封閉） |
| 含人聲的完整歌曲（商用） | Suno v5 或 Udio v4，並取得明確授權 |
| 含人聲的完整歌曲（開源） | ACE-Step XL 或 YuE |
| 短廣告 jingle | 以哼唱參考音訊為條件的 MusicGen melody |
| 音樂影片背景 | MusicGen + Stable Video Diffusion |

## 2026 年仍然會出現的陷阱

- **用 prompt 洗白版權。** "Song in the style of Taylor Swift"：商用的 Suno/Udio 現在會過濾這類 prompt，開源模型則不會。請自行加上過濾清單。
- **超過 30 秒後重複 / 飄移。** AR 模型會循環。請把多段生成結果做 crossfade，或使用 ACE-Step 來維持結構連貫。
- **Tempo 飄移。** 模型會偏離 BPM。請在 prompt 中加入 BPM 標籤，並用 librosa 的 `beat_track` 做後處理過濾。
- **人聲可懂度。** Suno 非常出色；開源模型的咬字常常含糊。如果歌詞很重要，請使用商用 API 或自行 fine-tune。
- **單聲道輸出。** 開源模型產生的是單聲道或假立體聲。請用正確的立體聲重建來升級（ezst、Cartesia 的 stereo diffusion）。

## 交付 (Ship It)

存成 `outputs/skill-music-designer.md`。針對音樂生成的部署，挑選模型、授權策略、長度 / 結構規劃，以及揭露用的 metadata。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會產生「生成式」的和弦進行 + 鼓點模式，以 ASCII 符號呈現，是個音樂生成的卡通版。想的話可以用任何 MIDI renderer 播放。
2. **中等。** 安裝 `audiocraft`，用 MusicGen-small 針對 4 種曲風 prompt 各生成 10 秒的片段，並對參考曲風集量測 FAD。
3. **困難。** 用 ACE-Step（或 MusicGen-melody），以不同的音色 prompt 生成同一段旋律的三種變化。計算與 prompt 的 CLAP similarity 以驗證對齊。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| FAD | 音訊版的 FID | 真實與生成音訊 embedding 分布之間的 Fréchet distance。 |
| Chromagram | 以音高表示的旋律 | 每個 frame 12 維的向量；旋律條件的輸入。 |
| Stems | 樂器軌 | 分離出的貝斯 / 鼓 / 人聲 / 旋律 WAV 檔。 |
| Inpainting | 重新生成某一段 | 遮住一個時間視窗；模型只重新生成那一段。 |
| CLAP | 文字-音訊版的 CLIP | 對比式的音訊-文字 embedding；用來評估文字-音訊對齊。 |
| EnCodec | 音樂 codec | Meta 的 neural codec，MusicGen 使用；32 kHz、4 個 codebooks。 |

## 延伸閱讀 (Further Reading)

- [Copet et al. (2023). MusicGen](https://arxiv.org/abs/2306.05284)：開源的 autoregressive 基準。
- [Evans et al. (2024). Stable Audio Open](https://arxiv.org/abs/2407.14358)：音效設計的預設選擇。
- [ACE-Step](https://github.com/ace-step/ACE-Step)：開源的 4B 完整歌曲生成器，2026 年 4 月。
- [Suno v5 platform docs](https://suno.com)：商用品質的領先者。
- [AudioLDM2](https://arxiv.org/abs/2308.05734)：用於音樂 + 音效的 latent diffusion。
- [WMG-Suno settlement coverage](https://www.musicbusinessworldwide.com/warner-music-group-settles-with-suno-strikes-first-of-its-kind-deal-with-ai-song-generator/)：2025 年 11 月的先例。
