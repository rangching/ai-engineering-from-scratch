# 音訊評估 (Audio Evaluation) — WER、MOS、UTMOS、MMAU、FAD 與開放排行榜

> 無法衡量的東西就無法上線。本課列出 2026 年每種音訊任務的指標：ASR（WER、CER、RTFx）、TTS（MOS、UTMOS、SECS、經 ASR 來回轉換後的 WER）、audio-language（MMAU、LongAudioBench）、音樂（FAD、CLAP），以及說話者（EER）。另外還有可以拿來比較的排行榜。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04, 06, 07, 09, 10; Phase 2 · 09 (Model Evaluation)
**Time:** ~60 分鐘

## 問題 (The Problem)

每種音訊任務都有多個指標，各自衡量不同的面向。用錯指標，就會上線一個在儀表板上看起來很棒、在正式環境卻很糟的模型。2026 年的標準清單：

| 任務 | 主要指標 | 次要指標 |
|------|---------|-----------|
| ASR | WER | CER · RTFx · first-token latency |
| TTS | MOS / UTMOS | SECS · WER-on-ASR-round-trip · CER · TTFA |
| Voice cloning | SECS (ECAPA cosine) | MOS · CER |
| Speaker verification | EER | minDCF · 操作點上的 FAR / FRR |
| Diarization | DER | JER · speaker confusion |
| Audio classification | top-1 · mAP | macro F1 · 每類別 recall |
| Music generation | FAD | CLAP · 聆聽小組 MOS |
| Audio language model | MMAU-Pro | LongAudioBench · AudioCaps FENSE |
| Streaming S2S | latency P50/P95 | WER · MOS |

## 核心概念 (The Concept)

![Audio evaluation matrix — metrics vs tasks vs 2026 leaderboards](../assets/eval-landscape.svg)

### ASR 指標

**WER (Word Error Rate)。** `(S + D + I) / N`。計分前先轉小寫、去除標點、正規化數字。請使用 `jiwer` 或 OpenAI 的 `whisper_normalizer`。&lt; 5% = 達到朗讀語音的人類水準。

**CER (Character Error Rate)。** 相同公式，但在字元層級。用於聲調語言（普通話、廣東話），因為這些語言的斷詞有歧義。

**RTFx（inverse real-time factor）。** 每個實際牆上時間秒數所處理的音訊秒數。越高越好。Parakeet-TDT 達到 3380 倍。Whisper-large-v3 約 30 倍。

**First-token latency。** 從音訊輸入到第一個逐字稿 token 的實際牆上時間。對串流至關重要。Deepgram Nova-3：約 150 ms。

### TTS 指標

**MOS (Mean Opinion Score)。** 1-5 分的人類評分。黃金標準，但很慢。每個樣本要收集 20 位以上的聆聽者，每個模型要 100 個以上的樣本。

**UTMOS（2022-2026）。** 學習得到的 MOS 預測器。在標準 benchmark 上與人類 MOS 的相關性約為 0.9。F5-TTS：UTMOS 3.95；ground truth：4.08。

**SECS (Speaker Encoder Cosine Similarity)。** 用於 voice cloning。參考音訊與複製輸出之間的 ECAPA embedding cosine。&gt; 0.75 = 可辨認的複製。

**WER-on-ASR-round-trip。** 對 TTS 輸出跑 Whisper，並與輸入文字計算 WER。能抓出可懂度的退步。2026 年 SOTA：英文測試集上 &lt; 2% WER（普通話測試集則回報 CER）。

**TTFA (time-to-first-audio)。** 實際牆上時間延遲。Kokoro-82M：約 100 ms；F5-TTS：約 1 秒。

### Voice cloning 專用

**SECS + MOS + CER** 三者並用。SECS 高但 MOS 低的複製，代表音色對但不自然；反過來則代表聲音自然但說話者不對。

### Speaker verification

**EER (Equal Error Rate)。** False Accept Rate 等於 False Reject Rate 的閾值。VoxCeleb1-O 上的 ECAPA：0.87%。

**minDCF (min Detection Cost)。** 在選定的操作點（通常是 FAR=0.01）上的加權成本。比 EER 更貼近正式環境。

### Diarization

**DER (Diarization Error Rate)。** `(FA + Miss + Confusion) / total_speaker_time`。漏判的語音 + 誤報的語音 + 說話者混淆，各自以比例表示。AMI 會議：DER 約 10-20% 是實際的水準。pyannote 3.1 + 商用的 Precision-2：在錄音良好的音訊上 DER &lt;10%。

**JER (Jaccard Error Rate)。** DER 的替代指標，對短片段偏誤比較穩健。

### Audio classification

多標籤：對所有類別計算 **mAP (mean Average Precision)**。AudioSet：BEATs-iter3 的 mAP 為 0.548。

多類別互斥：**top-1、top-5 accuracy**。Speech Commands v2：99.0% top-1（Audio-MAE）。

不平衡：**macro F1** + **每類別 recall**。請逐類別回報，總體準確率會掩蓋哪些類別失敗。

### Music generation

**FAD (Fréchet Audio Distance)。** 真實與生成音訊的 VGGish-embedding 分布之間的距離。MusicCaps 上的 MusicGen-small：4.5。MusicLM：4.0。越低越好。

**CLAP Score。** 使用 CLAP embeddings 的文字-音訊對齊分數。&gt; 0.3 = 合理的對齊。

**聆聽小組 MOS。** 對消費級音樂來說仍是最終結論。Suno v5 在 Suno 自己的成對偏好 benchmark 上得到 ELO 1293。那是廠商自己的數字；TTS Arena 排名的是語音，不是音樂。

### Audio-language benchmarks

**MMAU (Massive Multi-Audio Understanding)。** 1 萬組音訊 QA。

**MMAU-Pro。** 1800 道困難題目，四個類別：speech / sound / music / multi-audio。四選一的隨機猜測機率是 25%。Gemini 2.5 Pro 總體約 60%；所有模型的 multi-audio 約 22%。

**LongAudioBench。** 數分鐘長的音訊，搭配語意查詢。Audio Flamingo Next 擊敗 Gemini 2.5 Pro。

**AudioCaps / Clotho。** Captioning benchmarks。指標有 SPICE、CIDEr、FENSE。

### 串流語音轉語音

**Latency P50 / P95 / P99。** 從使用者說完到第一個聽得見的回應的實際牆上時間。Moshi：200 ms；GPT-4o Realtime：300 ms。

輸出的 **WER / MOS**。

**Barge-in 回應速度。** 從使用者打斷到助理靜音的時間。目標 &lt; 150 ms。

### 2026 年的排行榜

| 排行榜 | 賽道 | URL |
|------------|--------|-----|
| Open ASR Leaderboard (HF) | 英文 + 多語言 + 長音訊 | `huggingface.co/spaces/hf-audio/open_asr_leaderboard` |
| TTS Arena (HF) | 英文 TTS | `huggingface.co/spaces/TTS-AGI/TTS-Arena` |
| Artificial Analysis Speech | TTS + STT，由成對投票得出 ELO | `artificialanalysis.ai/speech` |
| MMAU-Pro | LALM 推理 | `sonalkum.github.io/mmau-pro` |
| SpeakerBench / VoxSRC | Speaker recognition | `voxsrc.github.io` |
| MMAU music subset | Music LALM | (within MMAU) |
| HEAR benchmark | Self-supervised 音訊 | `hearbenchmark.com` |

```figure
sp-wer-align
```

## 動手做 (Build It)

### 步驟 1：含 normalization 的 WER

```python
from jiwer import wer, Compose, ToLowerCase, RemovePunctuation, Strip

transform = Compose([ToLowerCase(), RemovePunctuation(), Strip()])
score = wer(
    truth="Please turn on the lights.",
    hypothesis="please turn on the light",
    truth_transform=transform,
    hypothesis_transform=transform,
)
# ~0.17
```

### 步驟 2：TTS 來回轉換 WER

```python
def ttr_wer(tts_model, asr_model, texts):
    errors = []
    for txt in texts:
        audio = tts_model.synthesize(txt)
        recog = asr_model.transcribe(audio)
        errors.append(wer(truth=txt, hypothesis=recog))
    return sum(errors) / len(errors)
```

### 步驟 3：Voice cloning 的 SECS

```python
from speechbrain.inference.speaker import EncoderClassifier
sv = EncoderClassifier.from_hparams("speechbrain/spkrec-ecapa-voxceleb")

emb_ref = sv.encode_batch(load_wav("reference.wav"))
emb_clone = sv.encode_batch(load_wav("cloned.wav"))
secs = torch.nn.functional.cosine_similarity(emb_ref, emb_clone, dim=-1).item()
```

### 步驟 4：Music generation 的 FAD

```python
from frechet_audio_distance import FrechetAudioDistance
fad = FrechetAudioDistance()
score = fad.get_fad_score("generated_folder/", "reference_folder/")
```

### 步驟 5：Speaker verification 的 EER（與第 6 課相同的程式碼）

```python
def eer(same_scores, diff_scores):
    thresholds = sorted(set(same_scores + diff_scores))
    best = (1.0, 0.0)
    for t in thresholds:
        far = sum(1 for s in diff_scores if s >= t) / len(diff_scores)
        frr = sum(1 for s in same_scores if s < t) / len(same_scores)
        if abs(far - frr) < best[0]:
            best = (abs(far - frr), (far + frr) / 2)
    return best[1]
```

## 實際使用 (Use It)

每次部署都要搭配一套固定的評估工具 (eval harness)，在每次模型更新時執行。三條基本規則：

1. **計分前先正規化。** 轉小寫、去標點、展開數字。並回報你使用的正規化規則。
2. **回報分布，而不是平均值。** 延遲回報 P50/P95/P99。分類回報每類別 recall。MMAU 回報每個類別。
3. **跑一個標準的公開 benchmark。** 即使你的正式環境資料不同，在 Open ASR / TTS Arena / MMAU 上回報，能讓審查者做同條件的比較。

## 陷阱 (Pitfalls)

- **UTMOS 外推。** 它是在 VCTK 風格的乾淨語音上訓練的；對吵雜 / 複製 / 帶情緒的音訊評分不佳。
- **MOS 小組偏誤。** 20 位 Amazon Mechanical Turk 工作者 ≠ 20 位目標使用者。如果風險高，請付費找領域專家小組。
- **FAD 取決於參考集。** 跨模型比較時，請對照同一個參考分布。
- **只看整體 WER。** 整體 5% 的 WER 可能掩蓋帶口音語音上 30% 的 WER。請依人口統計切片回報。
- **公開 benchmark 飽和。** 大多數前沿模型在標準 benchmark 上都已接近天花板。請建立能反映你實際流量的內部 held-out 資料集。

## 交付 (Ship It)

存成 `outputs/skill-audio-evaluator.md`。為任何音訊模型的發布，挑選指標、benchmarks 與回報格式。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。在玩具輸入上計算 WER / CER / EER / SECS / 類 FAD / 類 MMAU。
2. **中等。** 建立 TTS 來回轉換 WER 的評估工具。把你的 Kokoro 或 F5-TTS 輸出丟進 Whisper。對 50 個 prompts 計算 WER。標記 WER &gt; 10% 的 prompts。
3. **困難。** 在 MMAU-Pro 的 speech + multi-audio 子集（各 50 題）上，為你在第 10 課選擇的 LALM 評分。回報每個類別的準確率，並與公布的數字比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| WER | ASR 分數 | 正規化後，字層級的 `(S+D+I)/N`。 |
| CER | 字元層級的 WER | 用於聲調語言或字元層級的系統。 |
| MOS | 人類的意見 | 1-5 分評分；20 位以上聆聽者 × 100 個樣本。 |
| UTMOS | ML 的 MOS 預測器 | 學習得到的模型；與人類 MOS 的相關性約 0.9。 |
| SECS | Voice-clone 相似度 | 參考音訊與複製結果之間的 ECAPA cosine。 |
| EER | Speaker verif 分數 | FAR = FRR 時的閾值。 |
| DER | Diarization 分數 | (FA + Miss + Confusion) / total。 |
| FAD | Music-gen 品質 | VGGish embeddings 上的 Fréchet distance。 |
| RTFx | 吞吐量 | 每個實際牆上時間秒數所處理的音訊秒數。 |

## 延伸閱讀 (Further Reading)

- [jiwer](https://github.com/jitsi/jiwer)：附有 normalization 工具的 WER/CER 函式庫。
- [UTMOS (Saeki et al. 2022)](https://arxiv.org/abs/2204.02152)：學習得到的 MOS 預測器。
- [Fréchet Audio Distance (Kilgour et al. 2019)](https://arxiv.org/abs/1812.08466)：music-gen 的標準指標。
- [Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)：2026 年即時排名。
- [TTS Arena](https://huggingface.co/spaces/TTS-AGI/TTS-Arena)：以真人投票的 TTS 排行榜。
- [MMAU-Pro benchmark](https://sonalkum.github.io/mmau-pro/)：LALM 推理排行榜。
- [HEAR benchmark](https://hearbenchmark.com/)：音訊 SSL benchmarks。
