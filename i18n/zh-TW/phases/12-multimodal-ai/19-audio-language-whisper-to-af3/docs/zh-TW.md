# Audio-Language 模型：從 Whisper 到 Audio Flamingo 3 的演進 (Audio-Language Models: the Whisper to Audio Flamingo 3 Arc)

> Whisper (Radford et al., 2022 年 12 月) 底定了語音辨識：68 萬小時弱監督的多語言語音、簡單的 encoder-decoder transformer，以及一個讓之後每個 ASR 發表都得引用的 benchmark。但辨識不等於推理。問「這段錄音裡有哪些樂器」、「說話者在表達什麼情緒」或「第 3 分鐘發生了什麼」，需要的是音訊理解，不是轉錄。Qwen-Audio、SALMONN、LTU 與 NVIDIA 的 Audio Flamingo 3 (AF3, 2025 年 7 月) 逐步建立了這個堆疊：保留 Whisper 等級的 encoder，接上 Q-former，用音訊-文字 instruction 資料訓練，再加入 chain-of-thought 推理。這堂課走過這段演進。

**Type:** Build
**Languages:** Python (stdlib, log-Mel spectrogram + audio Q-former skeleton)
**Prerequisites:** Phase 6 (Speech and Audio), Phase 12 · 03 (Q-Former)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 從波形計算 log-Mel spectrogram：加窗、FFT、filter bank、log 轉換。
- 比較 encoder 選項：Whisper encoder、BEATs、AF-Whisper 混合型。各自何時勝出。
- 建立音訊 Q-former：N 個 learnable query 對 spectrogram patch 做 cross-attention。
- 解釋級聯式 (Whisper 後接 LLM) 與端到端音訊-LLM 訓練的差別：為什麼端到端在推理上擴展得更好。

## 問題 (The Problem)

語音辨識已經被 Whisper 解決了。音訊的 OCR 是現成商品。但「現成商品」止於轉錄。如果模型無法對聽到的內容做推理（時間、說話者、情緒、音樂結構、環境聲音），光靠轉錄無法驅動產品功能。

三條明顯的路線：

1. 級聯：Whisper 轉錄，LLM 對逐字稿做推理。適用於純語音情境。對音樂、環境音訊、多人重疊說話、情緒則會失敗。

2. 端到端音訊-LLM：音訊 encoder 把音訊 token 直接餵給 LLM，跳過轉錄。保留聲學資訊（情緒、說話者、環境）。需要新的訓練資料。

3. 混合：音訊 encoder + 同時能轉錄與推理的文字 decoder。Qwen-Audio 和 Audio Flamingo 選這條路。

## 核心概念 (The Concept)

### Log-Mel spectrogram：輸入特徵

每個音訊 encoder 都從同一個特徵開始：log-Mel spectrogram。

1. 重新取樣到 16 kHz。
2. 以 25ms 視窗、10ms hop 做短時傅立葉轉換。
3. 取 FFT 結果的幅度。
4. 套用 Mel filter bank（通常是 0-8000 Hz 內以 log 間距排列的 80 個濾波器），轉換成感知頻率。
5. 做 log 壓縮 $(\log(1 + x))$ 以取得動態範圍。

結果：形狀為 (T, 80) 的 2D 陣列，其中 T 是時間影格數。對於 100 Hz 影格率的 30 秒片段：(3000, 80)。

### Whisper 的 encoder

Whisper 的 encoder 是 12 層 ViT 風格的 transformer，把 log-Mel spectrogram 當作時間影格序列來處理。輸出：每個時間影格一個 hidden-state 向量。

對 ASR 來說，Whisper 的 decoder 是 cross-attention transformer，以 encoder 輸出為條件生成文字 token。標準的 encoder-decoder。

對 ALM（音訊-LLM）來說，你想把 encoder 輸出當作另一個 LLM 的輸入。模式是：Whisper encoder 凍結，Q-former 可訓練，LLM 凍結或調校。

### BEATs 與音訊專用 encoder (BEATs and audio-specific encoders)

Whisper 是在以語音為主的資料上訓練的。對音樂和環境音訊較弱。

BEATs (Chen et al., 2022) 是在 AudioSet 上訓練的 self-supervised transformer。在相同參數量下，比 Whisper 更能捕捉音樂和環境聲音。

AF-Whisper（Audio Flamingo 3 的 encoder）：從 Whisper-large-v3 encoder 開始，持續在語音、聲音與音樂上一起訓練，讓單一 encoder 同時承載語言與聲學訊號。SALMONN 等較早的系統則是透過串接 Whisper 與 BEATs 的特徵，達到相同的涵蓋範圍。

### 音訊 Q-former (Audio Q-former)

與 BLIP-2 視覺 Q-former 相同的模式。固定數量的 learnable query（通常 32 或 64）對音訊 encoder 的輸出影格做 cross-attention。這些 query 成為 LLM 消化的音訊 token。

Alignment 階段的訓練：只訓練 Q-former，在音訊-文字配對 (AudioCaps, Clotho) 上使用 contrastive + captioning loss。Instruction 階段：端到端，解凍 LLM，在 instruction 資料上訓練。

### 演進：SALMONN、Qwen-Audio、AF3 (The arc)

SALMONN (Tang et al., 2023)：Whisper + BEATs + Q-former + LLaMA。第一個具備相當推理能力的開源音訊-LLM。在 MMAU 上的 benchmark 約 0.55 綜合分。

Qwen-Audio (Chu et al., 2023)：架構相似，用更豐富的資料集訓練，針對多輪對話調校。MMAU 約 0.60。

LTU，Listen, Think, Understand (Gong et al., 2023)：明確的推理資料，專注於對音訊片段做 chain-of-thought。規模較小但更專注。

Audio Flamingo 3 (Goel et al., 2025 年 7 月)：目前的開源 SOTA。7B LLM backbone (Qwen2.5-7B)、由 Whisper-large-v3 打造的 AF-Whisper 音訊 encoder、通往 LLM 的小型 adaptor 層，以及 100 萬筆以上音訊-文字 instruction 配對的訓練。MMAU 0.72，在某些子任務上追平專有前沿模型。

AF3 還為音訊引入了按需的 chain-of-thought：模型可以選擇性地在最終答案之前輸出思考 token（「讓我先辨識樂器：...」）。啟用思考時，複雜推理任務的準確率提升 3-5 分。

### 級聯式與端到端 (Cascaded vs end-to-end)

級聯式管線：

1. Whisper 把音訊轉錄成文字。
2. LLM 對文字做推理。

對「摘要這集 podcast」完美運作。以下情況則失敗：
- 「這首歌的氛圍是什麼？」：氛圍在聲音裡，不在文字裡。
- 「說話的是 Alice 還是 Bob？」：需要說話者辨識。
- 「爆炸發生在第幾秒？」：時間 grounding 在文字中流失了。
- 「這是真實還是生成的音訊？」：deepfake 偵測需要聲學特徵。

端到端保留了聲學訊號。Qwen-Audio 和 AF3 原生就能處理音樂、環境與情緒。

### 2026 年的 production 配方 (2026 production recipe)

對新的音訊理解產品：

- 級聯式，如果：目標是轉錄，沒有音樂，不需要情緒推論。
- AF3 / Qwen-Audio 家族，如果：有音樂、情緒、多人說話或複雜的音訊推理。

級聯式更便宜、更簡單。端到端能力更強。

### MMAU：音訊推理 benchmark

MMAU (Massive Multimodal Audio Understanding) 是 2024-2025 年的音訊推理 benchmark：

- 橫跨語音、音樂、環境聲音的 10,000 組音訊-文字 QA 配對。
- 涵蓋分類、時間推理、因果推理、開放式 QA。
- 測試級聯式管線系統性遺漏的東西。

開源 SOTA (AF3) 為 0.72；專有前沿約 0.78（Gemini 2.5 Pro、Claude Opus 4.7）。這個差距比 VideoMME 的開源對封閉差距小，顯示音訊-LLM 正在成熟。

```figure
audio-text-ctc
```

## 實際使用 (Use It)

`code/main.py`：

- 用標準函式庫實作 log-Mel spectrogram 計算：加窗、樸素 DFT、Mel filter-bank。
- 音訊 Q-former 骨架：給定 encoder 輸出影格，計算 Q、K、V、attention，並輸出 N 個 token。
- 在玩具任務上比較級聯式與端到端。

## 交付 (Ship It)

這堂課產出 `outputs/skill-audio-llm-pipeline-picker.md`。給定音訊任務（轉錄、音樂標記、情緒推論、多人說話者分離、環境分類），它會選擇級聯式、端到端 AF3，或混合方案。

## 練習 (Exercises)

1. 計算 30 秒、16kHz、25ms 視窗、10ms hop、80 個 Mel bin 的片段的 log-Mel spectrogram 維度。在 48kHz 下會怎麼變？

2. 為什麼 Whisper 在音樂上表現不佳？BEATs 捕捉了哪些 Whisper 沒有的音訊特徵？

3. 64 個 query 與 32 個 query 的音訊 Q-former：任務複雜度到什麼程度時 64 才划算？32 又為什麼能省運算？

4. 閱讀 AF3 第 4 節關於按需思考的內容。提出三個 chain-of-thought 幫助最大的音訊任務。

5. 用 AF3 的輸出實作一個最小的說話者分離管線。你如何標示說話者的切換？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Log-Mel spectrogram | "Mel features" | 經 Mel filter bank 後的 log 幅度值組成的 2D（時間、頻率）陣列 |
| Audio Q-former | "Audio Perceiver" | 從音訊 encoder 輸出到餵入 LLM 的固定長度 query 的 cross-attention 瓶頸 |
| Cascaded | "ASR-then-LLM" | Whisper 轉錄、文字 LLM 推理的管線；會損失聲學資訊 |
| End-to-end | "Audio-LLM" | 音訊特徵透過 Q-former 直接進入 LLM；保留聲學訊號 |
| BEATs | "Audio AudioSet encoder" | 在 AudioSet 上訓練的 SSL transformer；擅長音樂與環境聲音 |
| MMAU | "Audio reasoning bench" | 橫跨語音、音樂、環境的 1 萬組 QA 配對；2024 年的評估標準 |
| On-demand thinking | "Audio CoT" | 模型可選擇在最終答案前輸出推理 token，準確率提升 3-5 分 |

## 延伸閱讀 (Further Reading)

- [Radford et al. — Whisper (arXiv:2212.04356)](https://arxiv.org/abs/2212.04356)
- [Chu et al. — Qwen-Audio (arXiv:2311.07919)](https://arxiv.org/abs/2311.07919)
- [Goel et al. — Audio Flamingo 3 (arXiv:2507.08128)](https://arxiv.org/abs/2507.08128)
- [Tang et al. — SALMONN (arXiv:2310.13289)](https://arxiv.org/abs/2310.13289)
- [Gong et al. — LTU (arXiv:2305.10790)](https://arxiv.org/abs/2305.10790)
