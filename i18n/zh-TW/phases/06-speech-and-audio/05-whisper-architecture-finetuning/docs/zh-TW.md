# Whisper — 架構與 Fine-Tuning (Architecture & Fine-Tuning)

> Whisper 是 30 秒視窗的 transformer encoder-decoder，用 68 萬小時的多語言弱監督音訊-文字配對訓練而成。一個架構、多種任務，在 99 種語言上都很穩健。它是 2026 年的參考 ASR。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 5 · 10 (Attention), Phase 7 · 05 (Full Transformer)
**Time:** ~75 分鐘

## 問題 (The Problem)

OpenAI 在 2022 年 9 月發表的 Whisper，是第一個以日常用品形式推出的 ASR 模型：貼上音訊、拿到文字，支援 99 種語言，對雜訊很穩健，在筆電上就能跑。到了 2024 年，OpenAI 推出了 Large-v3 與 Turbo 變體；到了 2026 年，從 podcast 轉錄、語音助理到 YouTube 字幕，Whisper 都是預設的基準。

但你不能永遠把 Whisper 當成黑盒 pipeline。Domain shift 會毀了它：技術術語、說話者的口音、專有名詞、很短的音訊、靜音。你需要知道：

1. 它內部到底是什麼。
2. 怎麼正確地餵給它切塊、串流或長音訊。
3. 什麼時候該 fine-tune，以及怎麼做。

## 核心概念 (The Concept)

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**架構。** 標準的 transformer encoder-decoder。

- 輸入：30 秒的 log-mel spectrogram，80 個 mels，10 ms hop → 3000 個 frames。較短的音訊補零，較長的音訊切塊。
- Encoder：conv-downsample（stride 2）+ `N` 個 transformer blocks。以 Large-v3 為例：32 層、1280 維、20 個 heads。
- Decoder：`N` 個 transformer blocks，包含 causal self-attn 與對 encoder 輸出的 cross-attn。大小與 encoder 相同。
- 輸出：在 51,865 個 token 的詞彙表上的 BPE tokens。

Large-v3 有 1.55B 個參數。Turbo 把 decoder 從 32 層減成 4 層，延遲降為 1/8，WER 只退步不到 1%。

**Prompt 格式。** Whisper 是多任務模型，由 decoder prompt 中的特殊 tokens 來引導：

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>`：語言標籤；指出音訊中說的是哪種語言。省略的話，Whisper 會先偵測語言。
- `<|transcribe|>` 或 `<|translate|>`：任務標籤；用原本的語言轉錄語音，或翻譯成英文文字。
- `<|notimestamps|>`：跳過字層級的 timestamps（較快）。

靠這個 prompt，一個模型就能做很多任務。把 `<|en|>` 換成 `<|fr|>`，它就會轉錄法文。

**30 秒視窗。** 一切都固定在 30 秒。較長的音訊需要切塊，較短的會被 padding。視窗本身不支援原生串流，這就是 WhisperX、Whisper-Streaming 和 faster-whisper 存在的原因。

**Log-mel normalization。** `(log_mel - mean) / std`，其中統計值來自 Whisper 自己的訓練資料。你*必須*使用 Whisper 的前處理（`whisper.audio.log_mel_spectrogram`），不能用 `librosa.feature.melspectrogram`。

### 2026 年的變體

| 變體 | Params | 延遲 (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Fine-tuning

2026 年的標準流程：

1. 收集 10–100 小時、有對齊逐字稿的目標領域音訊。
2. 執行 `transformers.Seq2SeqTrainer`，搭配 `generate_with_loss` callback。
3. Parameter-efficient：在 attention 層的 `q_proj`、`k_proj`、`v_proj` 上用 LoRA，GPU 記憶體降為 1/4，WER 代價不到 0.3。
4. 如果資料少於 10 小時，就凍結 encoder，只調 decoder。
5. 使用 Whisper 自己的 tokenizer 和 prompt 格式；絕對不要換 tokenizer。

社群的成果：用 20 小時的醫療口述資料 fine-tune Medium，醫療詞彙上的 WER 從 12% 降到 4.5%。用 4 小時的冰島語 fine-tune Turbo，WER 從 18% 降到 6%。

```figure
sp-asr-attention
```

## 動手做 (Build It)

### 步驟 1：直接執行 Whisper

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

你應該一律覆寫的關鍵預設值：`temperature=0.0`（取樣預設會走 0.0 → 0.2 → 0.4 … 的 fallback 鏈）、`condition_on_previous_text=False`（避免連鎖幻覺問題），以及 `no_speech_threshold=0.6`（靜音偵測）。

### 步驟 2：Chunked 長音訊

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX 加上了 (1) Silero VAD 把關、(2) 透過 wav2vec 2.0 做字層級對齊、(3) 透過 `pyannote.audio` 做 diarization。這是 2026 年正式環境轉錄的主力。

### 步驟 3：用 LoRA 做 Fine-tune

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

接著是標準的 Trainer 迴圈。每 1000 步存一次 checkpoint，並在 held-out 資料上用 WER 評估。

### 步驟 4：檢視每一層學到什麼

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

用 heatmap 視覺化，你會看到 decoder 逐步掃過 encoder frames 時形成的對角線對齊。那條對角線就是 Whisper 對字詞 timestamps 的概念。

## 實際使用 (Use It)

2026 年的工具組合：

| 情境 | 選擇 |
|-----------|------|
| 一般英文、離線 | 透過 `whisperx` 使用 Large-v3-turbo |
| 行動裝置 / 邊緣 | 量化 (int8) 的 Whisper-Tiny 或 Moonshine |
| 多語言長音訊 | 透過 `whisperx` 使用 Large-v3 + diarization |
| 低資源語言 | 用 LoRA fine-tune Medium 或 Turbo |
| 串流（2 秒延遲） | Whisper-Streaming 或 Parakeet-TDT |
| 字層級 timestamps | WhisperX（透過 wav2vec 2.0 做 forced alignment） |

`faster-whisper`（CTranslate2 後端）是 2026 年最快的 CPU+GPU 推論 runtime，輸出相同，速度是原版的 4 倍。

## 2026 年仍然會出現的陷阱

- **在靜音上產生幻覺文字。** Whisper 是用字幕訓練的，裡面有 "Thanks for watching!"、"Subscribe!" 和歌詞。呼叫之前一定要用 VAD 把關。
- **`condition_on_previous_text` 連鎖效應。** 一次幻覺會汙染後續的視窗。除非你需要跨區塊的流暢度，否則設為 `False`。
- **短音訊的 padding。** 把 2 秒的音訊補到 30 秒，尾端的靜音可能產生幻覺。請用 `pad=False` 或 VAD 把關。
- **錯誤的 mel 統計值。** 用 librosa 的 mels 取代 Whisper 的，輸出會近乎隨機。請使用 `whisper.audio.log_mel_spectrogram`。

## 交付 (Ship It)

存成 `outputs/skill-whisper-tuner.md`。針對指定的領域，設計 Whisper 的 fine-tune 或推論 pipeline。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。它會對 Whisper 風格的 prompt 做 tokenize、計算解碼 shape 的預算，並印出 10 分鐘音訊的切塊排程。
2. **中等。** 安裝 `faster-whisper`，轉錄一段 10 分鐘的 podcast，並與人工逐字稿比較 WER。試試 `language="auto"` 與強制 `language="en"` 的差別。
3. **困難。** 用 HF `datasets` 挑一個 Whisper 表現不佳的語言（例如 Urdu），用 LoRA 在 2 小時的資料上 fine-tune Medium 2 個 epochs，並回報 WER 的變化。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| 30-sec window | Whisper 的限制 | 輸入長度的硬上限；更長的音訊要切塊。 |
| SOT | Start-of-transcript | `<\|startoftranscript\|>` 啟動 decoder 的 prompt。 |
| Timestamps token | 時間對齊 | 每 0.02 秒的偏移量，都是 51k 詞彙表中的一個特殊 token。 |
| Turbo | 快速版本 | 4 層 decoder，快 8 倍，WER 退步不到 1%。 |
| WhisperX | 長音訊的包裝工具 | VAD + Whisper + wav2vec 對齊 + diarization。 |
| LoRA fine-tune | 高效率調校 | 在 attention 加上 low-rank adapters；只訓練約 0.3% 的參數。 |
| Hallucination | 悄悄出現的失敗 | Whisper 會從雜訊/靜音中產生流暢的英文。 |

## 延伸閱讀 (Further Reading)

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356)：原始架構與訓練配方。
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363)：4 層 decoder，加速 8 倍。
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747)：長音訊、字層級對齊、含 diarization。
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper)：以 CTranslate2 為後端，快 4 倍。
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper)：標準的 LoRA / 完整 fine-tune 教學。
