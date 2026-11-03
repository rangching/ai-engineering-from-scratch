# 音訊語言模型 (Audio-Language Models) — Qwen2.5-Omni、Audio Flamingo、GPT-4o Audio

> 2026 年的 audio-language models 能同時對語音、環境聲音和音樂做推理。Qwen2.5-Omni-7B 在 MMAU-Pro 上追平 GPT-4o Audio。Audio Flamingo Next 在 LongAudioBench 上擊敗 Gemini 2.5 Pro。開源與閉源的差距基本上已經消失，只剩 multi-audio 任務例外，在那裡大家都接近隨機猜測。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 6 · 04 (ASR), Phase 12 · 03 (Vision-Language Models), Phase 7 · 10 (Audio Transformers)
**Time:** ~45 分鐘

## 問題 (The Problem)

你有 5 秒的音訊：狗叫、有人大喊 "stop!"，然後一片寂靜。有用的問題橫跨好幾個面向：

- **轉錄 (Transcription)。** 「說了什麼？」，這是 ASR 的地盤。
- **語意推理 (Semantic reasoning)。** 「這個人有危險嗎？」，需要同時理解狗叫 + 大喊 + 寂靜。
- **音樂推理 (Music reasoning)。** 「旋律由哪些樂器演奏？」
- **長音訊檢索 (Long-audio retrieval)。** 「在這場 90 分鐘的講座中，講者是在哪裡解釋 gradient descent 的？」

用單一個 prompt 就能回答以上所有問題的單一模型，就是 **audio-language model**（LALM / ALM）。它和純 ASR 不同：LALM 產生的是自由形式的自然語言回答，而不只是逐字稿。

## 核心概念 (The Concept)

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### 三元件模板

2026 年的每個 LALM 都有相同的骨架：

1. **Audio encoder。** Whisper encoder · BEATs · CLAP · WavLM · 或每個模型自訂的 encoder。
2. **Projector。** 線性層或 MLP，把 audio-encoder 的特徵橋接到 LLM 的 token embedding 空間。
3. **LLM。** 基於 Llama / Qwen / Gemma 的 decoder。接收交錯的文字 + 音訊 tokens，產生文字。

訓練：

- **階段 1。** 凍結 encoder + LLM；只用 ASR / captioning 資料訓練 projector。
- **階段 2。** 在遵循指令的音訊任務（QA、推理、音樂理解）上做完整 / LoRA fine-tune。
- **階段 3（可選）。** 語音輸入 / 語音輸出 (voice-in / voice-out) 會加上 speech decoder。Qwen2.5-Omni 和 AF3-Chat 就是這樣做的。

### 2026 年的模型地圖

| Model | Backbone | Audio encoder | 輸出模態 | 取得方式 |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Benchmark 實況檢視 (2026)

**MMAU-Pro。** 1800 組 QA，涵蓋語音 / 聲音 / 音樂 / 混合。包含 multi-audio 子集。

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**multi-audio 這一欄對所有模型都很難看。** 四選一選擇題的隨機猜測機率是 25%；大多數模型的分數就落在那附近。LALM 仍然很難比較兩段音訊。

### 2026 年 LALM 有用的地方

- **客服中心錄音的合規稽核。** 「客服人員有沒有提到必要的揭露事項？」
- **無障礙。** 向聽障使用者描述聲音事件（而不只是轉錄）。
- **內容審核。** 偵測暴力言語 + 威脅語氣 + 背景脈絡。
- **Podcast / 會議章節劃分。** 語意摘要，而不只是說話者輪替。
- **音樂曲庫分析。** 「找出所有 B 段有轉調的曲目。」

### 它們（目前）還不好用的地方

- 細緻的樂理（和弦層級以下）。
- 長對話中依說話者歸屬的推理（超過 10 分鐘就會退化）。
- Multi-audio 比較（22-26% 勉強高於隨機）。
- 即時串流推理（大多數是離線批次推論）。

```figure
v4-alm-tokens
```

## 動手做 (Build It)

### 步驟 1：查詢 Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### 步驟 2：Projector 模式

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

就這樣。Projector 通常是 1-3 個線性層。用 ASR 配對資料（音訊 → 逐字稿）訓練它，就是階段 1 的前置任務 (pretext task)。

### 步驟 3：對 MMAU / LongAudioBench 做 benchmark

```python
from datasets import load_dataset
mmau = load_dataset("gamma-lab-umd/MMAU-Pro", split="test")
mcq = mmau.filter(lambda item: len(item["choices"] or []) > 1)

correct = 0
for item in mcq:
    answer = call_model(item["audio_path"], item["question"], item["choices"])
    if answer == item["answer"]:
        correct += 1
print(f"Accuracy: {correct / len(mcq):.3f}")
```

`audio_path` 指向資料集 repo 中的 `data.zip`（約 47 GB），所以評分前要先下載並解壓縮。這個 exact-match 迴圈只是合理性檢查 (sanity check)，不是 benchmark 的評分器，因此它的數字無法與已發表的 MMAU-Pro 結果比較。官方評估器是用 embedding similarity (NV-Embed-v2) 比對選擇題答案，用 LLM judge 評分開放式答案，並用 regex 規則檢查遵循指令的答案：請把預測寫入 `model_output` 欄位，並從 [MMAU-Pro repo](https://github.com/sonalkum/MMAUPro) 執行 `evaluate_mmau_pro_comprehensive.py`。請分開回報每個 `category`（speech、sound、music、multi 以及其他）。總體數字會掩蓋模型失敗的地方。

## 實際使用 (Use It)

| 任務 | 2026 年的選擇 |
|------|-----------|
| 自由形式音訊 QA（開源） | Qwen2.5-Omni-7B |
| 長音訊最佳開源模型 | Audio Flamingo Next |
| 最佳閉源模型 | Gemini 2.5 Pro |
| 語音輸入 / 語音輸出 agent | Qwen2.5-Omni 或 GPT-4o Audio |
| 音樂推理 | Audio Flamingo 3 或 2（針對音樂特化的 AF-CLAP） |
| 客服中心稽核 | 透過 API 使用 Gemini 2.5 Pro，並對你的政策文件做 RAG |

## 陷阱 (Pitfalls)

- **過度相信 multi-audio。** 如果你的任務需要「哪一段音訊有 X」，隨機猜測等級的表現是真實存在的。
- **長音訊退化。** 超過 10 分鐘，大多數模型的說話者歸屬就會壞掉。先做 diarize（Lesson 6），再做摘要。
- **在靜音上產生幻覺。** 與 Whisper 相同的問題，會被使用 Whisper encoder 的 LALM 繼承。請用 VAD 把關。
- **Benchmark 挑好看的數字。** 廠商的部落格文章會強調最佳情況的類別。請自己跑 MMAU-Pro 的 multi-audio 子集。

## 交付 (Ship It)

存成 `outputs/skill-alm-picker.md`。針對指定的音訊理解任務，挑選 LALM + benchmark 子集 + 輸出模態（文字 vs 語音）。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`，看一個玩具版 projector 模式，以及假的 LALM 對 (audio-embedding, text-tokens) → 輸出 tokens 的路由。
2. **中等。** 在 100 題 MMAU-Pro 語音題上為 Qwen2.5-Omni-7B 評分，並與論文公布的數字比較。
3. **困難。** 建立最小的音訊 captioning 基準：BEATs encoder + 2 層 projector + 凍結的 Llama-3.2-1B。只在 AudioCaps 上 fine-tune projector。在 Clotho-AQA 上與 SALMONN 比較。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| LALM | 音訊版 ChatGPT | Audio encoder + projector + LLM decoder。 |
| Projector | 轉接器 | 把音訊特徵對應到 LLM embedding 空間的小型 MLP。 |
| MMAU | 那個 benchmark | 橫跨語音、聲音、音樂的 1 萬組音訊 QA。 |
| MMAU-Pro | 更難的 MMAU | 1800 道 multi-audio / 重推理的題目。 |
| LongAudioBench | 長音訊評估 | 數分鐘長的音訊，搭配語意查詢。 |
| Voice-in / voice-out | 原生語音 | 模型接收語音並輸出語音，中間不繞道文字。 |

## 延伸閱讀 (Further Reading)

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759)：參考架構。
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B)：語音輸入、語音輸出。
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128)：開源長音訊的領先者。
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905)：LongAudioBench 的 SOTA。
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289)：雙 encoder 的先驅。
- [MMAU-Pro leaderboard](https://sonalkum.github.io/mmau-pro/)：2026 年即時排名。
