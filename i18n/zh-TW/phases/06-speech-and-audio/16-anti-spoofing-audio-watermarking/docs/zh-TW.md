# 語音 Anti-Spoofing 與音訊浮水印 (Voice Anti-Spoofing & Audio Watermarking) — ASVspoof 5、AudioSeal、WaveVerify

> Voice cloning 上線的速度比防禦快。2026 年正式環境的語音系統需要兩樣東西：一個把真假語音分類的偵測器（AASIST、RawNet2），以及一個能撐過壓縮與編輯的浮水印（AudioSeal）。兩者都要出貨，否則就別出貨 voice cloning。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 06 (Speaker Recognition), Phase 6 · 08 (Voice Cloning)
**Time:** ~75 分鐘

## 問題 (The Problem)

三種相關的防禦：

1. **Anti-spoofing / deepfake 偵測。** 給定一段音訊，它是合成的還是真實的？ASVspoof benchmarks（ASVspoof 2019 → 2021 → 5）是黃金標準。
2. **音訊浮水印 (Audio watermarking)。** 在生成的音訊中嵌入聽不見的訊號，讓偵測器之後能把它取出。AudioSeal（Meta）和 WavMark 是開源的選項。
3. **經過驗證的來源證明 (Authenticated provenance)。** 對音訊檔案 + metadata 做密碼學簽章。C2PA / Content Authenticity Initiative。

偵測處理的是不合作的對手。浮水印處理的是合規，也就是 AI 生成的音訊應該可以被辨認出來。2026 年兩者都是必要的。

## 核心概念 (The Concept)

![Anti-spoofing vs watermarking vs provenance — three defense layers](../assets/spoofing-watermark.svg)

### ASVspoof 5：2024-2025 年的 benchmark

與前幾屆相比最大的改變：

- **群眾外包的資料**（不是乾淨的錄音室品質），更貼近真實條件。
- **約 2000 位說話者**（之前約 100 位）。
- **32 種攻擊演算法。** TTS + voice conversion + 對抗式擾動。
- **兩個賽道。** Countermeasure (CM) 獨立偵測；以及針對生物辨識系統的 Spoofing-robust ASV (SASV)。

ASVspoof 5 上的 state-of-the-art：約 7.23% EER。在較舊的 ASVspoof 2019 LA 上：0.42% EER。實際部署時：在真實環境的音訊上預期會有 5-10% 的 EER。

### AASIST 與 RawNet2：偵測模型家族

**AASIST**（2021，更新到 2026）。作用在頻譜特徵上的 graph-attention。目前是 ASVspoof 5 countermeasure 任務的 SOTA。

**RawNet2。** 對原始 waveform 做卷積前端 + TDNN backbone。較簡單的基準；經過 fine-tune 後仍有競爭力。

**NeXt-TDNN + SSL 特徵。** 2025 年的變體：ECAPA 風格 + WavLM 特徵 + focal loss。在 ASVspoof 2019 LA 上達到 0.42% EER。

### AudioSeal：2024 年預設的浮水印

Meta 的 **AudioSeal**（2024 年 1 月，v0.2 為 2024 年 12 月）。關鍵設計：

- **Localized（局部化）。** 以 16 kHz sample 解析度（1/16000 秒），逐 frame 偵測浮水印。
- **Generator + detector 聯合訓練。** Generator 學習嵌入聽不見的訊號；detector 學習在各種 augmentations 下找出它。
- **穩健。** 能撐過 MP3 / AAC 壓縮、EQ、±10% 的速度變化，以及 +10 dB SNR 的雜訊混入。
- **快速。** Detector 以 485 倍即時速度運作；比 WavMark 快 1000 倍。
- **容量。** 每句話可嵌入 16-bit 的 payload（可編碼模型 ID、生成時間戳記、使用者 ID）。

### WavMark

AudioSeal 之前的開源基準。Invertible neural network，32 bits/sec。問題：

- 同步的暴力搜尋很慢。
- 可被高斯雜訊或 MP3 壓縮移除。
- 不適合即時使用。

### WaveVerify（2025 年 7 月）

處理 AudioSeal 的弱點，特別是時間操作（反轉、變速）。使用基於 FiLM 的 generator + Mixture-of-Experts detector。在標準攻擊上與 AudioSeal 不相上下；還能處理時間編輯。

### 對手利用的缺口

出自 AudioMarkBench：「在 pitch shift 下，所有浮水印的 Bit Recovery Accuracy 都低於 0.6，代表幾乎被完全移除。」**Pitch-shift 是通用的攻擊。** 2026 年沒有任何浮水印能完全抵抗強烈的音高修改。這就是為什麼除了浮水印之外，你還需要偵測（AASIST）。

### C2PA / Content Authenticity Initiative

這不是 ML 技術，而是一種 manifest 格式。音訊檔案帶有關於製作工具、作者、日期且經密碼學簽章的 metadata。Audobox / Seamless 使用它。適合用於來源證明；但如果壞人重新編碼並移除 metadata，它就起不了作用。

```figure
v4-audio-watermark
```

## 動手做 (Build It)

### 步驟 1：簡單的頻譜特徵偵測器（玩具）

```python
def spectral_rolloff(spec, percentile=0.85):
    cum = 0
    total = sum(spec)
    if total == 0:
        return 0
    threshold = total * percentile
    for k, v in enumerate(spec):
        cum += v
        if cum >= threshold:
            return k
    return len(spec) - 1

def is_suspicious(audio):
    spec = magnitude_spectrum(audio)
    rolloff = spectral_rolloff(spec)
    return rolloff / len(spec) > 0.92
```

合成語音的高頻能量往往異常平坦。正式環境的偵測器用的是 AASIST，而不是這個。但直覺是成立的。

### 步驟 2：AudioSeal 嵌入 + 偵測

```python
from audioseal import AudioSeal
import torch

generator = AudioSeal.load_generator("audioseal_wm_16bits")
detector = AudioSeal.load_detector("audioseal_detector_16bits")

audio = load_wav("generated.wav", sr=16000)[None, None, :]
payload = torch.tensor([[1, 0, 1, 1, 0, 1, 0, 0, 1, 1, 0, 1, 0, 1, 1, 0]])
watermark = generator.get_watermark(audio, sample_rate=16000, message=payload)
watermarked = audio + watermark

result, decoded_payload = detector.detect_watermark(watermarked, sample_rate=16000)
# result: float in [0, 1] — probability of watermark presence
# decoded_payload: 16 bits; match against embedded payload
```

### 步驟 3：評估：EER

```python
def eer(real_scores, fake_scores):
    thresholds = sorted(set(real_scores + fake_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in fake_scores if s >= t) / len(fake_scores)
        frr = sum(1 for s in real_scores if s < t) / len(real_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

### 步驟 4：正式環境整合

```python
def safe_tts(text, voice, clone_reference=None):
    if clone_reference is not None:
        verify_consent(user_id, clone_reference)
    audio = tts_model.synthesize(text, voice)
    audio_with_wm = audioseal_embed(audio, payload=build_payload(user_id, model_id))
    manifest = c2pa_sign(audio_with_wm, user_id, timestamp=now())
    return audio_with_wm, manifest
```

每次生成都要附上：(1) 浮水印，(2) 已簽章的 manifest，(3) 符合保存政策的稽核日誌。

## 實際使用 (Use It)

| 使用情境 | 防禦方式 |
|----------|---------|
| 出貨 TTS / voice cloning | 每個輸出都嵌入 AudioSeal（不容妥協） |
| 生物辨識語音解鎖 | AASIST + ECAPA 集成；活體挑戰 (liveness challenge) |
| 客服中心詐騙偵測 | 對 20% 的來電抽樣執行 AASIST |
| Podcast 真實性 | 上傳時做 C2PA 簽章，若為 AI 生成則加 AudioSeal |
| 研究 / 訓練偵測器 | ASVspoof 5 的 train/dev/eval 資料集 |

## 陷阱 (Pitfalls)

- **有浮水印卻從不執行偵測器。** 毫無意義。請把偵測器放進你的 CI。
- **偵測沒有校準。** 在 ASVspoof LA 上訓練的 AASIST 會 overfit；真實世界的準確率會下降。請在你的領域上校準。
- **Pitch-shift 缺口。** 強烈的 pitch shift 會移除大多數浮水印。請準備偵測作為後備。
- **移除 metadata 後重新託管。** C2PA 只要重新編碼就能輕易繞過。請一律同時加上密碼學 + 感知（浮水印）防禦。
- **把活體驗證當成偵測。** 要求使用者說出隨機的句子。能防止重放攻擊，但擋不住即時的複製。

## 交付 (Ship It)

存成 `outputs/skill-spoof-defender.md`。針對語音生成的部署，挑選偵測模型、浮水印、來源證明 manifest 與營運手冊。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。在合成音訊上執行玩具偵測器 + 玩具浮水印的嵌入/偵測。
2. **中等。** 安裝 `audioseal`，在 TTS 輸出中嵌入 16-bit 的 payload，再解碼。用雜訊破壞音訊並量測 Bit Recovery Accuracy。
3. **困難。** 在 ASVspoof 2019 LA 上 fine-tune RawNet2 或 AASIST。量測 EER。在一組 held-out 的 F5-TTS 生成音訊上測試，看看 OOD 偵測退化多少。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| ASVspoof | 那個 benchmark | 兩年一度的挑戰賽；2024 年是 ASVspoof 5。 |
| CM (countermeasure) | 偵測器 | 分類器：真實語音 vs 合成 / 轉換語音。 |
| SASV | 說話者驗證 + CM | 整合生物辨識與 spoof 偵測。 |
| AudioSeal | Meta 的浮水印 | 局部化、16-bit payload、比 WavMark 快 485 倍。 |
| Bit Recovery Accuracy | 浮水印存活率 | 攻擊後還原出的 payload 位元所占比例。 |
| C2PA | 來源證明 manifest | 關於製作 / 作者的密碼學 metadata。 |
| AASIST | 偵測器家族 | 基於 graph-attention 的 anti-spoofing SOTA。 |

## 延伸閱讀 (Further Reading)

- [Todisco et al. (2024). ASVspoof 5](https://dl.acm.org/doi/10.1016/j.csl.2025.101825)：目前的 benchmark。
- [Defossez et al. (2024). AudioSeal](https://arxiv.org/abs/2401.17264)：預設的浮水印。
- [Chen et al. (2025). WaveVerify](https://arxiv.org/abs/2507.21150)：針對時間攻擊的 MoE 偵測器。
- [Jung et al. (2022). AASIST](https://arxiv.org/abs/2110.01200)：SOTA 偵測 backbone。
- [AudioMarkBench (2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/5d9b7775296a641a1913ab6b4425d5e8-Paper-Datasets_and_Benchmarks_Track.pdf)：穩健性評估。
- [C2PA specification](https://spec.c2pa.org/specifications/specifications/2.4/index.html)：來源證明 manifest 格式。
