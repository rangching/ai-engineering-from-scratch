# 聲音複製與聲音轉換 (Voice Cloning & Voice Conversion)

> Voice cloning 是用別人的聲音讀出你的文字。Voice conversion 是把你的聲音改寫成別人的，同時保留你說的內容。兩者都建立在同一種分解上：把說話者身分和內容分開。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 07 (TTS)
**Time:** ~75 分鐘

## 問題 (The Problem)

2026 年，一段 5 秒的音訊，加上一張消費級 GPU，就足以產生任何人聲音的高品質複製。ElevenLabs、F5-TTS、OpenVoice v2、VoiceBox 都提供 zero-shot 或 few-shot 的複製。這項技術是福音（無障礙 TTS、配音、輔助用語音），也是武器（詐騙電話、政治 deepfake、智慧財產權竊取）。

兩個密切相關的任務：

- **Voice cloning（TTS 端）：** 文字 + 5 秒參考聲音 → 用該聲音說出的音訊。
- **Voice conversion（語音端）：** 來源音訊（A 說了 X）+ B 的參考聲音 → B 說 X 的音訊。

兩者都把 waveform 分解成（內容、說話者、韻律），再把某個來源的內容與另一個來源的說話者重新組合。

2026 年你必須遵守的關鍵限制：**浮水印 (watermarking) 與同意閘門 (consent gates) 在歐盟（AI Act，2026 年 8 月起強制執行）和加州（AB 2905，2025 年生效）都是法律要求**。你的 pipeline 必須輸出聽不見的浮水印，並拒絕未經同意的複製。

## 核心概念 (The Concept)

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning。** 把 5 秒的音訊丟給一個在數千位說話者上訓練過的模型。Speaker encoder 把音訊對應到 speaker embedding；TTS decoder 以該 embedding 加上文字為條件。

代表模型：F5-TTS（2024）、YourTTS（2022）、XTTS v2（2024）、OpenVoice v2（2024）。

**Few-shot fine-tuning。** 錄下 5-30 分鐘的目標聲音，用 LoRA fine-tune 基礎模型一小時。品質從「還行」躍升到「無法分辨」。Coqui 和 ElevenLabs 都支援這種模式；社群則搭配 F5-TTS 使用。

**Voice conversion (VC)。** 有兩大類：

- **Recognition-synthesis。** 先用類似 ASR 的模型抽取內容表示（例如 soft phoneme posteriors，PPGs），再用目標說話者的 embedding 重新合成。對語言與口音很穩健。代表模型：KNN-VC（2023）、Diff-HierVC（2023）。
- **Disentanglement。** 訓練一個 autoencoder，在瓶頸處的潛在空間中把內容、說話者和韻律分開。推論時替換 speaker embedding。品質較低但較快。代表模型：AutoVC（2019）、VITS-VC 變體。

**基於 neural codec 的複製（2024 年之後）。** VALL-E、VALL-E 2、NaturalSpeech 3、VoiceBox：把音訊視為來自 SoundStream / EnCodec 的離散 tokens，在 codec tokens 上訓練大型 autoregressive 或 flow-matching 模型。短 prompt 的品質與 ElevenLabs 相當。

### 倫理不是外掛

**Watermarking。** PerTh (Perth) 和 SilentCipher（2024）把約 16-32 bit 的 ID 以人耳無法察覺的方式嵌入音訊。能撐過重新編碼、串流與常見的編輯。是可用於正式環境的開源方案。

**Consent gates。** 每一個複製輸出都必須搭配可驗證的同意紀錄。「我，Rohit，於 2026-04-22 授權此聲音用於 X 目的。」存放在防竄改的日誌中。

**偵測。** AASIST、RawNet2 和 Wav2Vec2-AASIST 都有現成的偵測器。ASVspoof 2025 挑戰賽公布，最先進的偵測器對 ElevenLabs、VALL-E 2 和 Bark 的輸出，EER 為 0.8–2.3%。

### 數字 (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 時，對大多數聽者而言，通常已無法與目標區分。

```figure
sp-voice-factorize
```

## 動手做 (Build It)

### 步驟 1：用 recognition-synthesis 分解（main.py 中只有程式碼的 demo）

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

概念上很簡單；實作的主要份量在 `tts_model` 和 speaker encoder。

### 步驟 2：用 F5-TTS 做 zero-shot 複製

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

參考逐字稿必須和音訊完全一致；不一致會破壞對齊。

### 步驟 3：用 KNN-VC 做聲音轉換

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC 會用 WavLM 對來源與目標池抽取每個 frame 的 embedding，再把每個來源 frame 換成池中最近的鄰居。屬於 non-parametric，只要一分鐘的目標語音就能運作。

### 步驟 4：嵌入浮水印

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

約 32 bits 的 payload，經過 MP3 重新編碼與輕微雜訊後仍可偵測。

### 步驟 5：同意閘門

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 選擇 |
|-----------|------|
| 5 秒 zero-shot 複製、開源 | F5-TTS 或 OpenVoice v2 |
| 商用的正式環境複製 | ElevenLabs Instant Voice Clone v2.5 |
| 聲音轉換（改寫） | KNN-VC 或 Diff-HierVC |
| 多說話者 fine-tune | StyleTTS 2 + speaker adapter |
| 跨語言複製 | XTTS v2 或 VALL-E X |
| Deepfake 偵測 | Wav2Vec2-AASIST |

## 陷阱 (Pitfalls)

- **參考逐字稿沒對齊。** F5-TTS 和類似模型要求參考文字必須與參考音訊完全一致，連標點都要。
- **參考音訊有殘響。** 回音會毀掉複製結果。請在乾燥環境、近距離收音錄製。
- **情緒不一致。** 參考音訊是「開心」的，複製出來的每句話都會很開心。請讓參考音訊的情緒符合目標用途。
- **語言洩漏。** 複製一位英文說話者，再要求模型說法文，口音往往還是會帶過去；請使用跨語言模型（XTTS、VALL-E X）。
- **沒有浮水印。** 自 2026 年 8 月起，在歐盟無法合法上線。

## 交付 (Ship It)

存成 `outputs/skill-voice-cloner.md`。設計包含同意閘門 + 浮水印 + 品質目標的複製或轉換 pipeline。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。透過計算兩位「說話者」在替換前後的 cosine，示範 speaker-embedding 的替換。
2. **中等。** 用 OpenVoice v2 複製你自己的聲音。量測參考音訊與複製結果之間的 SECS，並透過 Whisper 量測 CER。
3. **困難。** 對 20 個複製結果加上 SilentCipher 浮水印，經過 128 kbps MP3 編碼再解碼，然後偵測 payload。回報 bit-accuracy。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Zero-shot clone | 5 秒就夠了 | 預訓練模型 + speaker embedding；不需訓練。 |
| PPG | Phonetic posteriorgram | 每個 frame 的 ASR posteriors，作為與語言無關的內容表示。 |
| KNN-VC | 最近鄰轉換 | 把每個來源 frame 換成目標池中最近的 frame。 |
| Neural codec TTS | VALL-E 風格 | 在 EnCodec/SoundStream tokens 上的 AR 模型。 |
| Watermark | 聽不見的簽章 | 嵌入音訊的位元，能撐過重新編碼。 |
| SECS | 複製的擬真度 | 目標與複製結果 speaker embeddings 之間的 cosine。 |
| AASIST | Deepfake 偵測器 | Anti-spoof 模型；偵測合成語音。 |

## 延伸閱讀 (Further Reading)

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885)：開源的 SOTA zero-shot 複製。
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) 與 [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370)：neural-codec TTS。
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879)：基於 disentanglement 的 voice conversion。
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975)：基於檢索的 VC。
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher)：可用於正式環境的 32-bit 音訊浮水印。
- [ASVspoof 2025 results](https://www.asvspoof.org/)：偵測器與合成器的軍備競賽，2026 年更新。
