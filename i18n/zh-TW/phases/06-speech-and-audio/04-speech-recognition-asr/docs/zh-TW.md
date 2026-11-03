# 語音辨識 (Speech Recognition, ASR) — CTC、RNN-T、Attention

> 語音辨識就是對每個時間點做音訊分類，再由一個懂英文和靜音的 sequence model 把結果串起來。CTC、RNN-T 和 attention 是三種做法。選一個，並且搞懂為什麼。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 02 (Spectrograms & Mel), Phase 5 · 08 (CNNs & RNNs for Text), Phase 5 · 10 (Attention)
**Time:** ~45 分鐘

## 問題 (The Problem)

你有一段 10 秒、16 kHz 的音訊，想得到一個字串："turn on the kitchen lights"。難處在結構上：音訊 frames 和字元不是一對一對齊的。"okay" 這個字可能花 200 ms，也可能花 1200 ms。靜音會把語句斷開。有些音素比較長。輸出 token 的數量事先也不知道。

有三種寫法可以解決：

1. **CTC (Connectionist Temporal Classification)。** 對每個 frame 輸出 token 機率，其中包含一個特殊的 *blank*。解碼時把重複和 blank 合併掉。non-autoregressive，速度快。wav2vec 2.0、MMS 都用它。
2. **RNN-T (Recurrent Neural Network Transducer)。** Joint network 根據 encoder frame 和先前的 tokens 預測下一個 token。可以串流。Google 的裝置端 ASR、NVIDIA Parakeet 都用它。
3. **Attention encoder-decoder。** Encoder 把音訊壓縮成 hidden states，decoder 用 cross-attention 以 autoregressive 方式產生 tokens。Whisper、SeamlessM4T 都用它。

2026 年，LibriSpeech test-clean 上的 SOTA WER 是 1.4%（Parakeet-TDT-1.1B，NVIDIA）和 1.58%（Whisper-Large-v3-turbo）。差距很小，但部署上的差異非常大。

## 核心概念 (The Concept)

![Three ASR formulations: CTC, RNN-T, attention-encoder-decoder](../assets/asr-formulations.svg)

**CTC 的直覺。** 讓 encoder 輸出 `T` 個 frame 層級、涵蓋 `V+1` 個 token（V 個字元 + blank）的分布。對於長度 `U < T` 的目標字串 `y`，任何能合併成 `y` 的 frame 對齊方式都算數。CTC loss 會對所有這類對齊方式加總。推論時：每個 frame 取 argmax，合併重複，移除 blank。

優點：non-autoregressive、可串流、不需要 lookahead。缺點：*conditional independence assumption*，也就是每個 frame 的預測彼此獨立，所以沒有內建的語言模型。解法是透過 beam search 或 shallow fusion 加上外部 LM。

**RNN-T 的直覺。** 加入一個 *predictor* 網路來 embed token 歷史，以及一個 *joiner* 把 predictor 狀態和 encoder frame 結合成 `V+1` 上的聯合分布（`+1` 代表 null / 不輸出）。它明確建模了 CTC 忽略的條件相依性。因為每一步只取決於過去的 frames 與過去的 tokens，所以可以串流。

優點：可串流，而且有內建 LM。缺點：訓練比較複雜，也更吃記憶體（3D loss lattice）；RNN-T loss 的 kernel 本身就自成一整個函式庫類別。

**Attention encoder-decoder。** Encoder（6-32 層 transformer）處理 log-mel frames。Decoder（6-32 層 transformer）用 cross-attention 看 encoder 輸出，以 autoregressive 方式產生 tokens。沒有對齊限制，attention 可以看音訊中的任何位置。除非限制 attention（chunked 的 Whisper-Streaming，2024），否則無法串流。

優點：離線 ASR 品質最高，用標準 seq2seq 工具就能輕鬆訓練。缺點：autoregressive 的延遲和輸出長度成正比；不額外做工程就無法串流。

### WER：唯一的數字

**Word Error Rate** = $(S + D + I) / N$，其中 $S$ 是替換數、$D$ 是刪除數、$I$ 是插入數、$N$ 是參考文字的字數。等同於在字層級計算 Levenshtein edit distance，越低越好。WER 高於 20% 通常無法使用；低於 5% 則達到朗讀語音的人類水準。標準 benchmark 上的 2026 年數字：

| Model | LibriSpeech test-clean | LibriSpeech test-other | Size |
|-------|------------------------|------------------------|------|
| Parakeet-TDT-1.1B | 1.40% | 2.78% | 1.1B params |
| Whisper-Large-v3-turbo | 1.58% | 3.03% | 809M |
| Canary-1B Flash | 1.48% | 2.87% | 1B |
| Seamless M4T v2 | 1.7% | 3.5% | 2.3B |

這些全都是 encoder-decoder 或 RNN-T 架構。純 CTC 系統（wav2vec 2.0）在 test-clean 上約為 1.8–2.1%。

```figure
ctc-collapse
```

## 動手做 (Build It)

### 步驟 1：Greedy CTC 解碼

```python
def ctc_greedy(frame_logits, blank=0, vocab=None):
    # frame_logits: list of per-frame probability vectors
    preds = [max(range(len(p)), key=lambda i: p[i]) for p in frame_logits]
    out = []
    prev = -1
    for p in preds:
        if p != prev and p != blank:
            out.append(p)
        prev = p
    return "".join(vocab[i] for i in out) if vocab else out
```

兩條規則：合併連續重複，丟掉 blank。例如：`a a _ _ a b b _ c` → `a a b c`。

### 步驟 2：Beam-search CTC

```python
def ctc_beam(frame_logits, beam=8, blank=0):
    import math
    beams = [([], 0.0)]  # (tokens, log_prob)
    for p in frame_logits:
        log_p = [math.log(max(pi, 1e-10)) for pi in p]
        candidates = []
        for seq, lp in beams:
            for t, lpt in enumerate(log_p):
                new = seq[:] if t == blank else (seq + [t] if not seq or seq[-1] != t else seq)
                candidates.append((new, lp + lpt))
        candidates.sort(key=lambda x: -x[1])
        beams = candidates[:beam]
    return beams[0][0]
```

正式環境會用結合 LM fusion 的 prefix tree beam search；這裡只是概念骨架。

### 步驟 3：WER

```python
def wer(ref, hyp):
    r, h = ref.split(), hyp.split()
    dp = [[0] * (len(h) + 1) for _ in range(len(r) + 1)]
    for i in range(len(r) + 1):
        dp[i][0] = i
    for j in range(len(h) + 1):
        dp[0][j] = j
    for i in range(1, len(r) + 1):
        for j in range(1, len(h) + 1):
            cost = 0 if r[i - 1] == h[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,
                dp[i][j - 1] + 1,
                dp[i - 1][j - 1] + cost,
            )
    return dp[len(r)][len(h)] / max(1, len(r))
```

### 步驟 4：用 Whisper 做推論

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe("clip.wav")
print(result["text"])
```

一行程式就能用上 2026 年最強的通用 ASR。在 24 GB 的 GPU 上，速度約為即時的 20 倍。

### 步驟 5：用 Parakeet 或 wav2vec 2.0 做串流

```python
from transformers import pipeline
asr = pipeline("automatic-speech-recognition", model="nvidia/parakeet-tdt-1.1b")
for chunk in streaming_audio():
    print(asr(chunk, return_timestamps=True))
```

串流 ASR 需要 chunked 的 encoder attention 與延續狀態 (carryover state)；請使用支援它的函式庫（Parakeet 用 NeMo，或帶有 `chunk_length_s` 的 `transformers` pipeline）。

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 選擇 |
|-----------|------|
| 英文、離線、追求最高品質 | Whisper-large-v3-turbo |
| 多語言、穩健 | SeamlessM4T v2 |
| 串流、低延遲 | Parakeet-TDT-1.1B 或 Riva |
| 邊緣、行動裝置、延遲 <500 ms | 量化的 Whisper-Tiny 或 Moonshine (2024) |
| 長音訊 | 搭配 VAD 切塊的 Whisper (WhisperX) |
| 特定領域（醫療、法律） | Fine-tune wav2vec 2.0 + 領域 LM fusion |

## 2026 年仍然會出現的陷阱

- **沒有 VAD。** 對靜音執行 Whisper 會產生幻覺（"Thanks for watching!"）。一定要用 VAD 把關。
- **字元、字、subword 的 WER 混用。** 請在 normalization（轉小寫、去除標點）*之後*，回報字層級的 WER。
- **Language ID 飄移。** Whisper 的自動 LID 會把吵雜的音訊誤判成日文或威爾斯文；知道語言時請強制設定 `language="en"`。
- **長音訊沒有切塊。** Whisper 的視窗是 30 秒。更長的音訊請用 `chunk_length_s=30, stride=5`。

## 交付 (Ship It)

存成 `outputs/skill-asr-picker.md`。針對指定的部署目標，挑選模型、解碼策略、切塊方式與 LM fusion。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會 greedy 解碼一份手工製作的 CTC 輸出，並對照參考文字計算 WER。
2. **中等。** 正確實作步驟 2 的 prefix-tree beam search（要處理 blank 合併規則）。在 10 筆合成資料上與 greedy 比較。
3. **困難。** 在 [LibriSpeech test-clean](https://www.openslr.org/12) 上使用 `whisper-large-v3-turbo`。計算前 100 句的 WER，並與公開數字比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| CTC | 有 blank token 的 loss | 對所有 frame 到 token 對齊方式取邊際機率；non-AR。 |
| RNN-T | 串流用的 loss | CTC + 下一個 token 的 predictor；能處理字詞順序。 |
| Attention enc-dec | Whisper 風格 | Encoder + 用 cross-attention 的 decoder；離線品質最佳。 |
| WER | 你要回報的那個數字 | 字層級的 `(S+D+I)/N`。 |
| Blank | 空白 | CTC 中的特殊 token，表示「這個 frame 不輸出」。 |
| LM fusion | 外部語言模型 | 在 beam search 時加入加權的 LM log-probs。 |
| VAD | 靜音閘門 | Voice activity detector；裁掉非語音部分。 |

## 延伸閱讀 (Further Reading)

- [Graves et al. (2006). Connectionist Temporal Classification](https://www.cs.toronto.edu/~graves/icml_2006.pdf)：CTC 的論文。
- [Graves (2012). Sequence Transduction with RNNs](https://arxiv.org/abs/1211.3711)：RNN-T 的論文。
- [Radford et al. / OpenAI (2022). Whisper: Robust Speech Recognition via Large-Scale Weak Supervision](https://arxiv.org/abs/2212.04356)：2022 年的標竿論文；2024 年有 v3-turbo 延伸版本。
- [NVIDIA NeMo — Parakeet-TDT card](https://huggingface.co/nvidia/parakeet-tdt-1.1b)：2026 年 Open ASR Leaderboard 的領先者。
- [Hugging Face — Open ASR Leaderboard](https://huggingface.co/spaces/hf-audio/open_asr_leaderboard)：橫跨 25 個以上模型的即時 benchmark。
