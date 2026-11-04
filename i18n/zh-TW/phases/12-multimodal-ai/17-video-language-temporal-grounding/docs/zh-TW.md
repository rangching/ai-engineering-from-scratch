# Video-Language 模型：時間 Token 與 Grounding (Video-Language Models: Temporal Tokens and Grounding)

> 影片不是一疊照片。一段 5 秒的片段有因果順序、動作動詞與事件時間點，這些都是圖片模型無法表示的。Video-LLaMA (Zhang et al., 2023 年 6 月) 推出了第一個具備音訊-視覺 grounding 的開源 video-LLM。VideoChat 和 Video-LLaVA 把這個模式擴大。到了 2025 年，Qwen2.5-VL 的時間對齊 M-RoPE 縮小了與前沿專有模型的差距。每個系統對時間 token 的處理方式都不同：每個片段一個 Q-former、每個影格做 concat-pool、每個 token 一個時間對齊的 M-RoPE。這堂課讀這些模式，做出均勻對動態的影格取樣器，並在時間 grounding 任務上評估。

**Type:** Build
**Languages:** Python (stdlib, frame sampler + temporal-grounding evaluator)
**Prerequisites:** Phase 12 · 08 (LLaVA-OneVision)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼時間位置編碼會獨立於 vision encoder，改變 video VLM 的表現。
- 在每秒 token 數與 grounding 準確率上，比較均勻、dynamic-FPS 與事件驅動的影格取樣。
- 描述每個片段一個 Q-former (Video-LLaMA)、每個影格 pool (Video-LLaVA) 與每個 token 時間對齊 M-RoPE (Qwen2.5-VL) 的設計。
- 說出四個影片 benchmark：VideoMME、TempCompass、EgoSchema、Video-MMMU。

## 問題 (The Problem)

1 分鐘、30 FPS 的影片有 1800 個影格。以每影格 196 個 visual token（224 的 ViT-B）計算，就是 35.2 萬個 token，比任何 2024 年代的 LLM context 都大。

存在三種縮減策略：

1. 對影格做子取樣（依內容為 1-8 FPS）。
2. 積極地 pool 每個影格的 patch token（3x3 或 4x4 雙線性 pool）。
3. 透過 Q-former 壓縮，它接收 16 個影格的片段並輸出 64 個 token。

每種取捨都不同。子取樣損失時間細節。Pooling 損失空間細節。Q-former 兩者都損失一點，但省下 token。

時間位置編碼是另一個軸：模型怎麼知道影格 5 在影格 6 之前？選項包括簡單的 1D 時間 RoPE (Video-LLaMA)、可學習的時間 embedding (Video-LLaVA)，以及時間對齊的 M-RoPE (Qwen2.5-VL，完整 3D)。

## 核心概念 (The Concept)

### Video-LLaMA：每個片段一個 Q-former + 音訊分支

Video-LLaMA (2023) 是第一個開源 video-LLM。架構：

- 以 2 FPS 取樣的 16 影格片段（所以是 8 秒）。
- 逐影格的 ViT 特徵 -> Video Q-former 對全部 16 個影格做 cross-attention -> 32 個可學習 query -> LLM。
- 平行的音訊分支：波形 -> ImageBind 音訊 encoder -> Audio Q-former -> 32 個 query -> LLM。

優點：音訊-視覺的聯合推理。缺點：片段長度固定，無法對任意時間做 grounding。

### VideoChat 與 Video-LLaVA

VideoChat 保留 Video-LLaMA 的想法，但去掉音訊並簡化。Video-LLaVA (Lin et al., 2023) 用同一個 visual encoder 同時訓練圖片與影片影格（「projection 之前先 alignment」），得到統一的表示。兩者都是凍結的 CLIP encoder + MLP + LLM。

兩者都不處理長影片。都是 8-16 影格的系統。

### Qwen2.5-VL 與時間對齊的 M-RoPE (Time-aligned M-RoPE)

Qwen2.5-VL 保留 Qwen2-VL 的 M-RoPE (Multimodal Rotary Position Embedding)，並讓它的時間軸與絕對時間對齊。每個 patch token 帶有 (t, h, w) 位置，其中 t 跟隨實際時間戳（而非影格索引）。Qwen2.5-Omni（課程 12.20）後來把同一個想法的音訊與影片版本命名為 TMRoPE，Time-aligned Multimodal RoPE。

與簡單時間 embedding 的主要差異：

- 絕對時間，而非索引。模型看到的是「在 4.2 秒」，而不是「在影格 15」。
- 每個 token 旋轉，而非每個片段。每個 visual token 依其時間戳獨立旋轉。
- 相容於 dynamic FPS。如果這裡以 2 FPS 取樣、那裡以 4 FPS 取樣，時間對齊的 M-RoPE 原生就能處理不均勻的間隔。

時間對齊的 M-RoPE 使「貓在第幾秒跳起來？」這類查詢成為可能。模型可以輸出「在 4.2 秒」。Video-LLaMA 只能說「在片段前段」。

### 影格取樣策略 (Frame sampling strategies)

均勻：在長度上平均取樣 N 個影格。簡單，但會漏掉動作高峰。

Dynamic FPS：根據動作強度自適應取樣。光流或影格差分挑出高動作的區段做更密的取樣。Qwen2.5-VL 就用這個訓練。

事件驅動：跑一個輕量偵測器，在動作發生處多取樣。VideoAgent 使用。

關鍵影格 + context：在鏡頭邊界加上幾個相鄰影格取樣。用於電影內容。

### 每個影格的 pooling (Pooling per frame)

以 1 FPS、每影格 576 個 token 計算，5 分鐘的片段是 172,800 個 token。這會溢出 Qwen2.5-VL-72B 的 128k context，所以你必須先降低影格率，或先 pool 每個影格的 token。

3x3 雙線性 pool 把每影格縮減到 64 個 token -> 5 分鐘 19,200 個 token。對大多數任務是甜蜜點。

對空間細節較不重要的 agent 工作流程，可以 pool 得更積極（6x6 -> 每影格 16 個 token）。

### 四個影片 benchmark (The four video benchmarks)

- VideoMME：全面的影片理解，短 + 中 + 長。
- TempCompass：細緻的時間推理，「之前」/「之後」類問題。
- EgoSchema：長時間範圍的第一人稱影片。
- Video-MMMU：multimodal 多學科的影片問題。

完整的 video-VLM 評估要涵蓋全部四個。它們壓力測試不同的面向：TempCompass 全在於順序，EgoSchema 在於 3 分鐘以上的推理，VideoMME 橫跨不同長度。

### Grounding 輸出格式 (Grounding output formats)

時間 grounding 的輸出格式：

- 自由文字：「貓大約在 4 秒左右跳起來。」容易解析，但不精確。
- 結構化 JSON：`{"event": "jump", "start": 4.1, "end": 4.3}`。Qwen2.5-VL 訓練了這個。
- 基於 token：特殊的 `<time>4.1</time>` token 與答案交錯。部分影片模型使用；Qwen2.5-VL 則是把時間編碼在 M-RoPE 的 position ID 中，並以文字或 JSON 回答。

基於 token 的對下游使用最精確。Qwen2.5-VL 的 JSON 輸出格式可以直接解析。

### 2026 年的最佳實務 (2026 best practice)

2026 年的 video VLM：

- Encoder：SigLIP 2 或原生解析度 ViT，LLM 中搭配時間對齊的 M-RoPE (Qwen2.5-VL)。
- 影格取樣：dynamic FPS（依動作為 1-4），並設最大影格數上限。
- 每影格 pooling：3x3 雙線性。
- 輸出：帶有時間 + 事件欄位的結構化 JSON。
- Benchmark：一般用 VideoMME + TempCompass；長時間範圍用 EgoSchema。

```figure
video-temporal-patches
```

## 實際使用 (Use It)

`code/main.py` 包含：

- 均勻與 dynamic-FPS 影格取樣器。
- 一個玩具版時間 grounding 評估器：給定時間 T 的「真實」事件與模型輸出，在容許範圍內為準確度評分。
- Video-LLaMA（16 影格，Q-former）、Video-LLaVA（8 影格，MLP）、Qwen2.5-VL（dynamic FPS + 時間對齊 M-RoPE）的比較。

## 交付 (Ship It)

這堂課產出 `outputs/skill-video-vlm-frame-planner.md`。給定影片任務（監控、動作辨識、時間 grounding、摘要），它會挑選影格取樣器、pooling 因子、輸出格式與預期的準確率等級。

## 練習 (Exercises)

1. 對一段 3 分鐘的烹飪示範，選擇均勻或 dynamic FPS。用 token 數量證明你的選擇。

2. 時間對齊的 M-RoPE 具體增加了什麼，是簡單的時間 embedding 表做不到的？

3. 寫出一個 VLM 能學會輸出的時間 grounding JSON schema。包含錯誤情況。

4. 閱讀 Video-LLaVA 第 3 節的「Alignment Before Projection」。為什麼這比分開訓練圖片與影片 encoder 更好？

5. 根據 VideoMME 排行榜，到 2026 年為止，最佳開源模型與最佳專有模型之間的差距是多少？其中有多少歸因於時間編碼，又有多少歸因於基礎 LLM 的規模？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Temporal grounding | "Time-localized answers" | VLM 針對事件發生的時間，輸出特定的時間戳範圍 |
| Time-aligned M-RoPE | "TMRoPE" | 時間軸跟隨絕對時間戳的 3D 旋轉位置編碼；Qwen2.5-VL 用於影片，Qwen2.5-Omni 則把其音訊與影片形式命名為 TMRoPE |
| Dynamic FPS | "Motion-aware sampling" | 在高動作區段多取樣影格，在靜態區段少取樣 |
| Frame pooling | "Spatial compress per frame" | 在進入 LLM 前，以雙線性插值減少每個影格的 patch 數 |
| Video Q-former | "Clip compressor" | 把 N 個影格對應到 K 個可學習 query 的 cross-attention 瓶頸 |
| VideoMME | "Video bench" | 涵蓋短/中/長影片的全面 benchmark，2500 多個樣本 |

## 延伸閱讀 (Further Reading)

- [Zhang et al. — Video-LLaMA (arXiv:2306.02858)](https://arxiv.org/abs/2306.02858)
- [Li et al. — VideoChat (arXiv:2305.06355)](https://arxiv.org/abs/2305.06355)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Qwen Team — Qwen2.5-VL (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Lin et al. — VILA-1.5 (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
