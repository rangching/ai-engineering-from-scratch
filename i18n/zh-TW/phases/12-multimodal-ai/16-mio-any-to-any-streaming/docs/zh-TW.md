# MIO 與 Any-to-Any Streaming Multimodal 模型 (MIO and Any-to-Any Streaming Multimodal Models)

> GPT-4o 推出了大多數開源模型無法複製的產品：一個能聽語音、看影片，並即時開口回應的 agent。到 2024 年底，開源生態系的答案是 MIO (Wang et al., 2024 年 9 月)。MIO 把文字、圖片、語音與音樂 token 化，在交錯序列上訓練一個 causal transformer，並能從任何 modality 生成任何 modality。AnyGPT (Zhan et al., 2024 年 2 月) 是概念驗證；MIO 是擴大規模的版本；Unified-IO 2 (Allen AI, 2023 年 12 月) 是帶有視覺 + 動作 grounding 的表親。這堂課讀 any-to-any 的模式：四個 tokenizer、一個 transformer、適合串流的解碼。

**Type:** Learn
**Languages:** Python (stdlib, four-modality token allocator + streaming decode loop)
**Prerequisites:** Phase 12 · 11 (Chameleon), Phase 6 (Speech and Audio)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 設計一個共用詞彙表，容納文字、圖片、語音與音樂 token 而不發生衝突。
- 在壓縮與重建的取捨上，比較 SEED-Tokenizer（圖片）與 SpeechTokenizer residual-VQ（語音）。
- 解釋逐步建立 any-to-any 生成能力的四階段 curriculum。
- 說出三種開源 any-to-any 配方及其主要取捨：MIO、AnyGPT、Unified-IO 2。

## 問題 (The Problem)

統一的 multimodal 模型很容易宣稱，卻很難大規模打造。直到 2024 年，大多數「any-to-any」系統都是管線式的：視覺模型 → 文字表示 → 語音模型 → 音訊。每一跳都會損失資訊、增加延遲，並讓訓練複雜化。GPT-4o 的示範影片展示了單一模型、亞秒級回應的替代方案；開源系統落後了好幾個月。

工程上的挑戰：

- 每種 modality 都必須有 tokenizer，壓縮得夠不失真以便重建，並以 transformer 能消化的速率產生 token。
- 單一詞彙表必須為文字 (32k+)、圖片 (8k+)、語音 (4k+)、音樂 (8k+) 配置空間。至少五萬多個項目。
- 訓練資料必須涵蓋每一種輸入輸出配對（text→image、image→speech、speech→image 等），否則模型必須自行組合。
- 推論必須足夠快地串流輸出 token，才能達到對話般的延遲（首個音訊位元組的時間 <500ms）。

## 核心概念 (The Concept)

### 四種 modality 的四個 tokenizer (Four tokenizers for four modalities)

MIO 的 tokenizer 堆疊：

- 文字：標準 BPE，詞彙表約 32000。
- 圖片：SEED-Tokenizer (2023)，一個離散的圖片 tokenizer，codebook 有 8192 個項目，每張圖片輸出 32 個 token。
- 語音：SpeechTokenizer residual-VQ (2023)，把 16kHz 波形編碼成 8 個階層式 codebook；第一層是粗略的內容，後面各層加入韻律和說話者身分。MIO 保留內容層加上前三個音色層，各 1024 個 code，共 4096 個語音 ID。
- 音樂：類似的 residual-VQ（Meta 的 MusicGen / Encodec 家族），4-8 個 codebook。MIO 本身並未提供音樂 tokenizer（它的第四種 modality 是影片，以影格通過圖片 tokenizer）；這堂課的玩具版加入音樂，是為了展示新 modality 如何取得一段 ID 範圍。

每種 modality 都產生整數 token。這些 token 在共用詞彙表中取得互不重疊的 ID 範圍：

```
text:   0..31999
image:  32000..40191  (8192 image tokens)
speech: 40192..44287  (4 codebooks x 1024 speech tokens)
music:  44288..52479  (8192 music tokens, the toy's extension)
sep:    52480..52485  (<image>, <speech>, <music>, </...>)
```

總計：約 5.2 萬的詞彙表。輸入 embedding 與輸出投影橫跨全部範圍。

### 串流解碼 (Streaming decode)

語音生成使用 residual-VQ。transformer 預測基底（第 0 層）語音 token；平行解碼的 residual quantizer 預測後續各層。每個第 0 層 token 在 16kHz 下大約是 50ms 的音訊。

串流模式：

1. 使用者對麥克風說話；即時音訊 tokenizer 每 50ms 輸出語音 token。
2. MIO 在 token 抵達時就消化它們（prompt prefill + 增量 forward）。
3. 輸出 token 邊生成邊串流輸出；平行語音 decoder 以約 50-150ms 的延遲把它們轉成音訊取樣。
4. 首個音訊位元組的時間：MIO 論文沒有報告延遲數字。以下預算把這個設計估計在最好約 300ms，而 GPT-4o 約為 250ms。

Mini-Omni (arXiv:2408.16725)、GLM-4-Voice (arXiv:2412.02612) 和 Moshi (arXiv:2410.00037) 是互補的串流 speech-LLM 設計。尤其是 Moshi，在單張 GPU 上達到 160ms 的來回延遲。

### 四階段 curriculum (Four-stage curriculum)

MIO 的訓練 curriculum：

1. 階段 1：alignment。大規模的 modality 配對語料：文字-圖片、文字-語音、文字-音樂。每種配對使用自己的 token 詞彙表區段。訓練共用詞彙表。
2. 階段 2：交錯。多 modality 交錯的文件（含圖片 + 影片的部落格、附逐字稿的 podcast 等）。訓練跨 modality 的 context。
3. 階段 3：語音強化。額外的音訊資料，在不損失文字能力的前提下提升語音品質。
4. 階段 4：SFT。跨 modality 的 instruction tuning：VQA、captioning、旁白、語音對語音的對話。

少了任何一個階段，特定能力就會退化：跳過階段 2，模型會失去跨 modality 的 context；跳過階段 3，語音品質就差。

### Chain-of-visual-thought

MIO 引入了 chain-of-visual-thought：模型輸出中間的圖片 token 作為推理步驟。對於「貓正在爬樹嗎？」模型會：

1. 輸出渲染場景的 `<image>` token（來自輸入圖片或草圖）。
2. 輸出分析該草圖的文字。
3. 輸出最終答案。

渲染出的中間圖片充當草稿紙。空間推理任務的 benchmark 因此進步。這個想法呼應了文字推理的 chain-of-thought。

### any-to-any 的競爭者 (Competitors in any-to-any)

- AnyGPT (arXiv:2402.12226)：4 種 modality（文字、圖片、語音、音樂），設計相似。
- Unified-IO 2 (arXiv:2312.17172)：加入視覺動作輸出、深度、法線。任務更多元，規模較小。
- NExT-GPT (arXiv:2309.05519)：LLM + 各 modality 專屬的 diffusion decoder。不是單一模型的做法。
- CoDi (arXiv:2305.11846)：可組合的 diffusion；透過共用 latent 達成 any-to-any。

MIO 最接近純 token 的 any-to-any。AnyGPT 是它概念上的祖先。

### 延遲預算 (Latency budget)

對對話式產品來說，每個元件的延遲都很重要：

- 麥克風到音訊 token：約 50ms。
- Prefill（音訊 token + 歷史）：8B 模型約 100ms。
- 第一個輸出 token：約 50ms。
- 平行 residual-VQ + 語音 decoder：約 100-150ms。

首個音訊位元組的總時間：最少約 300ms。GPT-4o 宣稱約 250ms。Moshi 宣稱 160ms。MIO 和 AnyGPT 都沒有公布實測數字，所以請把這個預算當作估計，而非 benchmark。

### 為什麼 any-to-any 仍然困難 (Why any-to-any stays hard)

即使到了 2026 年，開源 any-to-any 模型在兩個面向上仍落後封閉模型：

- 語音品質。residual-VQ tokenizer 是有損的；對話語音聽起來比 ElevenLabs 等級的聲音更機械。
- 跨 modality 推理。要求模型「唱出你看到的東西」，失敗的頻率仍高於純視覺任務。

這些都是尚未解決的研究問題。Qwen3-Omni（課程 12.20）是 2025 年最先進的開源嘗試。

```figure
any-to-any-stream
```

## 實際使用 (Use It)

`code/main.py`：

- 定義四種 modality 的詞彙表配置並印出。
- 讓一串 multimodal 輸入（文字、圖片、音訊片段、音樂）通過 tokenizer 路由器。
- 模擬文字轉語音回應的串流解碼，並計算延遲。
- 根據 encoder、prefill 與 decoder 的延遲，計算預期的首個音訊位元組時間。

## 交付 (Ship It)

這堂課產出 `outputs/skill-any-to-any-pipeline-auditor.md`。給定對話式產品規格（輸入的 modality、輸出的 modality、延遲目標），它會審查 MIO 家族的設計選擇並計算延遲預算。

## 練習 (Exercises)

1. 你的產品接受語音輸入並回傳語音輸出。端到端延遲預算目標是多少？列出耗時的元件。

2. SpeechTokenizer residual-VQ 使用 8 個 codebook。說明為什麼需要平行解碼 residual 層級（相對於循序），以及它省下多少延遲。

3. 你的詞彙表有 32k 文字 + 8k 圖片 + 4k 語音。再加上 8k 音樂和約 10 個分隔符。在 hidden 維度 4096 下，embedding 矩陣的參數成本是多少？

4. Chain-of-visual-thought 會輸出中間圖片。哪些問題從中受益？哪些問題會被額外的 token 拖累？

5. 閱讀 Moshi (arXiv:2410.00037)。描述它的「inner monologue」技巧，並與 MIO 的 chain-of-visual-thought 比較。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Any-to-any | "Multimodal in/out" | 單一模型，可朝任何方向接受並輸出文字、圖片、語音和音樂 |
| Residual-VQ | "Speech tokenizer stack" | 多 codebook 的 tokenization，每一層都增加資訊；基底層是內容，後面各層是韻律 |
| SEED-Tokenizer | "Image codes" | 具有 8192 個項目 codebook、每張圖片 32 個 token 的離散圖片 tokenizer，MIO 使用 |
| Chain-of-visual-thought | "Visual scratchpad" | 模型在給出最終答案前，先生成中間圖片作為推理步驟 |
| Time-to-first-audio-byte | "TTFAB" | 從使用者說話到第一個音訊輸出的延遲；<500ms 才有對話的感覺 |
| Four-stage curriculum | "Training recipe" | Alignment -> interleaved -> speech-enhanced -> SFT，依此順序 |

## 延伸閱讀 (Further Reading)

- [Wang et al. — MIO (arXiv:2409.17692)](https://arxiv.org/abs/2409.17692)
- [Zhan et al. — AnyGPT (arXiv:2402.12226)](https://arxiv.org/abs/2402.12226)
- [Lu et al. — Unified-IO 2 (arXiv:2312.17172)](https://arxiv.org/abs/2312.17172)
- [Wu et al. — NExT-GPT (arXiv:2309.05519)](https://arxiv.org/abs/2309.05519)
- [Tang et al. — CoDi (arXiv:2305.11846)](https://arxiv.org/abs/2305.11846)
