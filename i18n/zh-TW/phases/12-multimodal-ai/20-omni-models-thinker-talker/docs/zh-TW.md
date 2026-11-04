# Omni 模型：Qwen2.5-Omni 與 Thinker-Talker 拆分 (Omni Models: Qwen2.5-Omni and the Thinker-Talker Split)

> GPT-4o 在 2024 年 5 月的產品示範之所以具破壞性，不是因為底層模型，而是因為產品的形態：一個語音介面，你說話，模型看到鏡頭所見，並在 250ms 內開口回應。開源生態系在 2024 年剩下的時間與 2025 年都在競相追上這個產品介面。Qwen2.5-Omni (2025 年 3 月) 是參考用的開源設計：一個 Thinker（大型文字生成 transformer）加上一個 Talker（平行的語音生成 transformer），以串流的語音 token 連接。Mini-Omni 加以簡化，Moshi 追平了它的延遲，GLM-4-Voice 把它擴展到中文。這堂課讀 Thinker-Talker 架構，以及讓串流即時對話得以運作的延遲預算。

**Type:** Build
**Languages:** Python (stdlib, streaming pipeline latency simulator + VAD loop)
**Prerequisites:** Phase 12 · 19 (audio-LLMs), Phase 12 · 16 (any-to-any)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 把推論管線拆成 Thinker（文字推理）與 Talker（語音合成），並解釋為什麼平行串流有效。
- 逐元件計算對話式互動的首個音訊位元組時間 (TTFAB) 預算。
- 描述 TMRoPE 在 Thinker 內，跨視覺、音訊與文字的時間對齊位置編碼。
- 說出兩種即時對話模式，half-duplex 與 full-duplex，並解釋 VAD 輪流發言如何驅動 half-duplex 的交接。

## 問題 (The Problem)

即時語音助理要快速做很多事：

1. 聽使用者說話。即時的語音 tokenization，以及語音活動偵測 (VAD) 來知道他們何時說完。
2. 選擇性地看。2-4 FPS 的鏡頭輸入，與音訊一起串流進 Thinker。
3. 思考。依對話歷史組織回應。
4. 說話。合成音訊 token，解碼成波形，串流到使用者的喇叭。

每個步驟都增加延遲。要有對話的感覺，總來回時間必須 < 500ms，低於這個值，使用者就不再察覺延遲。GPT-4o 宣稱約 250ms。Moshi 約 160ms。Qwen2.5-Omni 約 350-500ms。

每個元件都必須串流。不能有任何「全部批次處理再解碼」。

## 核心概念 (The Concept)

### Thinker 與 Talker

Qwen2.5-Omni 的拆解：

- Thinker：7B-80B 的文字生成 transformer。消化交錯的文字 + 圖片 + 音訊 token。輸出代表要說什麼的文字 token。
- Talker：較小的語音生成 transformer（2 億-10 億）。消化 Thinker 輸出的文字 token 加上近期的語音 context token。輸出離散的語音 token（residual-VQ 索引）。
- 語音 decoder：串流的波形 decoder（SNAC、MoVQGAN 家族），即時把語音 token 轉成音訊取樣。

這個分離很重要。Thinker 必須夠大才能推理得好。Talker 可以很小，因為它的工作是局部的：把文字轉成語音 token。更大的 Talker 並不更有表現力，只是更慢。

兩者平行執行：

1. Thinker 輸出文字 token $t_i$。
2. Talker 消化 $t_i$（透過串流）並輸出語音 token $s_i, s_{i+1}, \ldots, s_{i+k}$。
3. 語音 decoder 在語音 token 到達時消化它們，並輸出音訊取樣。
4. 等 Thinker 到達文字 token $t_{i+3}$ 時，Talker 已經串流出 $t_0, \ldots, t_{i+2}$ 的音訊。

### TMRoPE：時間對齊的 multimodal 位置 (TMRoPE)

Thinker 需要整合影像影格（以 4 FPS 到達）、音訊影格（以每秒 50 個影格到達），以及對話歷史中的文字。單純的序列順序（所有圖片，然後所有音訊，然後文字）會失去時間對齊。

TMRoPE 為每個 token 指派絕對時間戳。t=2.3s 的視覺 token。t=2.32s 的音訊 token。使用者在 t=2.35s 說的「stop」文字 token。RoPE 依時間戳旋轉 attention；模型把它們視為時間上同時發生。

這是讓「他一邊揮手一邊說哈囉」得以成立的基礎設施：模型在概念上的同一時刻看到影片影格和音訊。

### 串流語音合成 (Streaming speech synthesis)

語音 token 必須串流。Mini-Omni (Xie & Wu, 2024) 提出了「語言模型能一邊思考一邊以串流方式聽與說」：Thinker 輸出 token 與 Talker 輸出 token 在同一條序列中交錯。Talker 一等 Thinker 確定下一個文字 token 就立刻啟動。沒有批次邊界。

Moshi (Défossez et al., 2024 年 10 月) 是最快的開源實作。在單張 A100 上 TTFAB 為 160ms。架構：單一 7B transformer，在交替的位置輸出文字與語音 token，並有「inner monologue」把思考串流與說話串流分開。這實際上是把 Thinker + Talker 透過仔細訓練融合成一個模型。

### VAD 與輪流發言 (VAD and turn-taking)

語音活動偵測在輸入端執行。兩種模式：

- Half-duplex：使用者說話，模型聆聽。模型說話，使用者聆聽。透過 VAD 靜音偵測（約 200ms）清楚交接。
- Full-duplex：雙方可以同時說話。模型可以回應語（「嗯哼」）或打斷。難得多。Moshi 支援。

Qwen2.5-Omni 預設支援 half-duplex，以靜音閾值做輪流發言。Full-duplex 需要應用層處理。

### Qwen3-Omni (2025 年 11 月)

後繼者。Qwen3-80B Thinker、更大的 Talker、改良的 TMRoPE-v2。延遲接近 GPT-4o 的 250ms。開放權重。在 OmniBench 上的 benchmark 與 Gemini 2.0 Live 相當。

### Production 延遲預算 (Production latency budget)

典型的串流互動：

- 麥克風 -> 音訊 token：40-80ms。
- Prefill（prompt + 歷史）：7B 為 100-200ms，70B 則多得多。
- 第一個 Thinker 文字 token：40ms。
- Talker 處理第一個文字 token：20ms。
- 第一批語音 token 確定：40ms。
- Residual-VQ 解碼：30ms。
- 語音波形解碼：50-80ms。

TTFAB 總計：7B 為 320-490ms，70B 為 600-900ms。前沿品質通常意味著 70B 以上；因此有前沿延遲差距。

### Token 速率計算 (Token-rate math)

16kHz 語音搭配 50 Hz 基底語音 token，輸出每秒需要 50 個語音 token。Talker 必須輸出 ≥50 tok/s 才跟得上。在 H100 上典型的 LLM 吞吐量為 30-80 tok/s，小型（2-3 億）Talker 夠快；7B 的 Talker 就會落後。

這就是為什麼有小型專用的 Talker 模型，而不是「直接用主模型」。

```figure
l5-thinker-talker
```

## 實際使用 (Use It)

`code/main.py`：

- 以模擬的 token 輸出速率，模擬 Thinker-Talker 管線。
- 為可設定的模型大小與麥克風取樣率計算 TTFAB。
- 示範以 VAD 靜音閾值做 half-duplex 輪流發言。

## 交付 (Ship It)

這堂課產出 `outputs/skill-omni-streaming-budget.md`。給定即時語音產品的目標 TTFAB 與功能集（視覺輸入、雙語、full-duplex），它會在 Qwen2.5-Omni、Qwen3-Omni、Moshi 或 Mini-Omni 之間挑選，並決定 Thinker/Talker 的大小。

## 練習 (Exercises)

1. 你的目標 TTFAB 是 300ms。在 7B Thinker 與 300M Talker 下，寫出每個元件的延遲。

2. Qwen2.5-Omni 使用 TMRoPE。描述在使用者於 t=1s 開始說話、鏡頭在 t=1.2s 捕捉到一個手勢的 prompt 中，模型看到什麼。

3. Full-duplex 支援需要模型在聆聽的同時輸出音訊。提出一種教它這件事的訓練資料格式。

4. 閱讀 Moshi 論文第 4 節。描述「inner monologue」的分離，以及為什麼它避免了 Thinker-Talker 拆分。

5. 計算吞吐量預算：Talker 必須多快輸出 token，才能跟上每秒 50 個基底層 token 的 16kHz 語音？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Thinker | "Reasoning brain" | 產生要說什麼的大型文字生成 transformer |
| Talker | "Speech-generating mouth" | 從 Thinker 的文字產生離散語音 token 的小型 transformer |
| TTFAB | "Latency budget" | 首個音訊位元組的時間：從使用者說完話到第一個音訊取樣輸出 |
| TMRoPE | "Time-aligned RoPE" | 跨視覺、音訊、文字使用絕對時間戳的位置編碼 |
| Half-duplex | "Turn-taking" | 使用者與模型輪流；VAD 靜音偵測使用者說完 |
| Full-duplex | "Simultaneous" | 模型可同時說話與聆聽；具備回應語的能力 |
| Inner monologue | "Moshi separation" | 單一模型設計，思考串流與說話串流交錯 |

## 延伸閱讀 (Further Reading)

- [Xu et al. — Qwen2.5-Omni (arXiv:2503.20215)](https://arxiv.org/abs/2503.20215)
- [Qwen Team — Qwen3-Omni (arXiv:2509.17765)](https://arxiv.org/html/2509.17765v1)
- [Xie & Wu — Mini-Omni (arXiv:2408.16725)](https://arxiv.org/abs/2408.16725)
- [Défossez et al. — Moshi (arXiv:2410.00037)](https://arxiv.org/abs/2410.00037)
- [Zeng et al. — GLM-4-Voice (arXiv:2412.02612)](https://arxiv.org/abs/2412.02612)
