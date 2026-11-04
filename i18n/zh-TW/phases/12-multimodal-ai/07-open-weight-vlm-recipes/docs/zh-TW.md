# Open-Weight VLM 配方：真正重要的是什麼 (Open-Weight VLM Recipes: What Actually Matters)

> 2024-2026 年的 open-weight VLM 文獻，是一片 ablation 表格的森林。Apple 的 MM1 測試了 13 種 image encoder、connector 與資料混合的組合。Allen AI 的 Molmo 證明詳細的人工 caption 勝過 GPT-4V distillation。Cambrian-1 做了 20 多種 encoder 比較。Idefics2 把五軸設計空間正式化。Prismatic VLMs 在受控 benchmark 上比較了 27 種訓練配方。在這些雜訊中，有一小組結果在各篇論文間都成立：image encoder 比 connector 架構更重要，資料混合又比兩者都重要，而詳細的人工 caption 勝過蒸餾出來的合成資料。這堂課幫你讀完這些表格，你就不用自己讀了。

**Type:** Learn + lab
**Languages:** Python (stdlib, ablation table parser + recipe picker)
**Prerequisites:** Phase 12 · 05 (LLaVA baseline)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 說出 VLM 的五軸設計空間：image encoder、connector、LLM、資料混合、解析度排程。
- 讀懂 MM1 / Idefics2 / Cambrian-1 的 ablation 表格，並預測哪個旋鈕會影響特定 benchmark。
- 給定運算預算與任務組合，為新的 VLM 挑選配方（encoder、connector、資料、解析度）。
- 解釋為什麼在相同 token 數下，詳細的人工 caption 勝過 GPT-4V distillation。

## 問題 (The Problem)

現有數百個 open-weight VLM。「好」與「最先進」之間的差距，大部分不在架構，而在資料、解析度排程與 encoder 的選擇。知道模型表現不佳時該先轉哪個旋鈕，能讓你避免一個 500 萬 GPU 小時的錯誤。

2023 年那一波 (LLaVA-1.5、InstructBLIP、MiniGPT-4) 靠 caption 配對 pretraining 加 LLaVA-Instruct-150k。是不錯的 baseline，MMMU 大約停在 35%。

2024 年那一波 (MM1、Idefics2、Molmo、Cambrian-1、Prismatic VLMs) 做了詳盡的 ablation。結果出人意料，而且很實用。

## 核心概念 (The Concept)

### 五軸設計空間 (The five-axis design space)

Idefics2 (Laurençon et al., 2024) 為這些軸命名：

1. Image encoder。CLIP ViT-L/14、SigLIP SO400m/14、DINOv2 ViT-g/14、InternViT-6B。各 encoder 在 patch size、解析度與 pretraining 目標上不同。
2. Connector。MLP（2-4 層）、Q-Former（32 個 query + cross-attn）、Perceiver Resampler（64 個 query）、C-Abstractor（convolution + 雙線性 pooling）。
3. 語言模型。Llama-3 8B / 70B、Mistral 7B、Phi-3、Gemma-2、Qwen2.5。LLM 大小是主要的參數成本。
4. 訓練資料。Caption 配對 (CC3M, LAION)、交錯資料 (OBELICS, MMC4)、instruction 資料 (LLaVA-Instruct, ShareGPT4V, PixMo, Cauldron)。
5. 解析度排程。固定 224/336/448、AnyRes、原生動態。訓練期間逐步提高或維持固定。

每個 production VLM 都在每個軸上做出選擇。MMMU 分數的大部分變異，可以由軸 1、4、5 解釋，而不是你選了哪種 connector。

### 軸 1：encoder > connector

MM1 第 3.2 節顯示：從 CLIP ViT-L/14 換成 SigLIP SO400m/14，MMMU 增加 3 分以上。把 connector 從 MLP 換成 Perceiver Resampler，增加不到 1 分。Idefics2 重現了這個結果：SigLIP > CLIP，在相同 token 數下 Q-Former ≈ MLP ≈ Perceiver。

Cambrian-1 的「Cambrian Vision Encoders Match-Up」(Tong et al., 2024) 在以視覺為中心的 benchmark (CV-Bench) 上跑了 20 多種 encoder。排行榜頂端是 DINOv2 與 SigLIP 的混合；CLIP 在中間；ImageBind 和 ViT-MAE 較低。從 CLIP ViT-L 到 DINOv2 ViT-g/14 的差距，在 CV-Bench 上約 5-7 分。

2026 年開源 VLM 的預設 encoder 是 SigLIP 2 SO400m/14，負責語意加 dense feature，有時會與 DINOv2 ViT-g/14 的特徵串接（Cambrian 的「Spatial Vision Aggregator」就這樣做）。

### 軸 2：connector 設計差不多 (Connector design is a wash)

MM1、Idefics2、Prismatic 與 MM-Interleaved 都得到同樣的結論：在固定的 visual-token 數量下，connector 架構幾乎沒差別。在相同 token 預算下，對 mean-pool 後 patch 做的 2 層 MLP，表現與 32 個 query 的 Q-Former 相差在 1 分以內。

真正重要的是 token 數量。更多的 visual token = 更多 LLM 運算 = 到某個程度之前表現更好，之後報酬遞減。每張圖片 64 個 token 對 OCR 太少。576-1024 個 token 是大多數開源 VLM 的甜蜜點。2048 以上只對文件和圖表有幫助。

Q-Former 與 MLP 是成本問題，不是品質問題：Q-Former 不論圖片解析度，都把 token 限制在 32-64；MLP 則輸出全部 patch token。對高解析度輸入，Q-Former 節省 LLM context；對低解析度，差別只是雜訊。

### 軸 3：LLM 大小決定天花板 (LLM size sets the ceiling)

把 LLM 從 7B 加倍到 13B，在每篇 VLM 論文中，MMMU 都穩定增加 2-4 分。到 70B 時，大多數 benchmark 都飽和了。VLM 的 multimodal 推理天花板，就是 LLM 的文字推理天花板：visual encoder 只能餵資料給它，不能替它推理。

這就是為什麼 Qwen2.5-VL-72B 和 Claude Opus 4.7 在 MMMU-Pro 和 ScreenSpot-Pro 上表現壓倒性：語言大腦夠大。7B 的 VLM 無法靠巧妙的 connector 設計來取代 70B 的 VLM。

### 軸 4：資料，詳細的人工 caption 勝過 distillation (Data)

Molmo + PixMo (Deitke et al., 2024) 是人人都該讀的 2024 年成果。Allen AI 請人工標註者用 1-3 分鐘的密集語音轉文字方式描述圖片，得到 71.2 萬張密集 caption 的圖片。訓練資料中完全沒有 GPT-4V distillation。

Molmo-72B 在 11 項 benchmark 中全部勝過 Llama-3.2-90B-Vision。差別不在架構，而在 caption 品質。詳細的人工 caption 每張圖片包含的資訊是簡短網路 caption 的 5-10 倍，而且在 GPT-4V distillation 會產生幻覺的地方，仍能維持事實依據。

ShareGPT4V (Chen et al., 2023) 與 Cauldron (Idefics2) 沿用同樣的做法，混合人工與 GPT-4V 的 caption。趨勢很清楚：對 2026 年的前沿，caption 密度 > caption 數量 > distillation 的便利性。

### 軸 5：解析度與它的排程 (Resolution and its schedule)

Idefics2 的 ablation：384 -> 448 增加 1-2 分。448 -> 980 搭配圖片切分 (AnyRes)，在 OCR benchmark 上再增加 3-5 分。固定解析度訓練會停在中等準確率；解析度逐步提高（從 224 開始，結束於 448 或原生）訓練得更快，也結束在更高的地方。

Cambrian-1 做了解析度對 token 的取捨：在固定運算量下，你可以選較低解析度加更多 token，或較高解析度加較少 token。OCR 是高解析度勝出；一般場景理解則是低解析度多 token 勝出。

2026 年的 production 配方：階段 1 以固定 384 訓練，階段 2 在重 OCR 的任務上使用最高 1280 的動態解析度。

### Prismatic 的受控比較 (The Prismatic controlled comparison)

Prismatic VLMs (Karamcheti et al., 2024) 是控制了所有軸的論文。相同的 13B LLM、相同的 instruction 資料、相同的評估，一次只改變一個軸。結果：

- 每張圖片的 visual-token 數量解釋約 60% 的變異。
- Encoder 的選擇解釋約 20%。
- Connector 架構解釋約 5%。
- 其他一切（資料混合、排程器、LR）占剩下的約 15%。

這是粗略的分解，但它是文獻中對「我該先 ablate 什麼」最乾淨的答案。

### 2026 年的選擇器 (A picker for 2026)

根據證據，2026 年新專案的預設 open-VLM 配方：

- Encoder：原生解析度、帶 NaFlex 的 SigLIP 2 SO400m/14；若需要 segmentation/grounding，再串接 DINOv2 ViT-g/14 做 dense feature。
- Connector：對 patch token 做 2 層 MLP。除非受 token 限制，否則跳過 Q-Former。
- LLM：Qwen2.5 / Llama-3.1 / Gemma 2，為了成本選 7B，為了品質選 70B，依目標延遲決定。
- 資料：PixMo + ShareGPT4V + Cauldron，再補上特定任務的 instruction 資料。
- 解析度：動態（長邊最小 256、最大 1280 像素）。
- 排程：階段 1 alignment（只訓練 projector），階段 2 完整 fine-tune，階段 3 特定任務 fine-tune。

這些預設值每一項，都能追溯到本課結尾所引用論文中的實測 ablation。

```figure
l5-vlm-recipe-knobs
```

## 實際使用 (Use It)

`code/main.py` 是 ablation 表格解析器與配方選擇器。它編入了 MM1 與 Idefics2 的 ablation 表格（精簡版），讓你查詢：

- 「給定預算 X 與任務 Y，哪個配方勝出？」
- 「如果我在 7B Llama 上把 SigLIP 換成 CLIP，預期的 MMMU 差距是多少？」
- 「要得到 80% 信心的答案，我該先 ablate 哪個軸？」

輸出是附預期 benchmark 差距的配方排名清單，以及「先 ablate 這個」的建議。

## 交付 (Ship It)

這堂課產出 `outputs/skill-vlm-recipe-picker.md`。給定目標任務組合、運算預算與延遲目標，它會輸出完整配方（encoder、connector、LLM、資料混合、解析度排程），並為每個選擇附上佐證的 ablation 引用。避免工程師每次啟動新的 VLM 專案，就重新發明一次 Idefics2 的 ablation 表格。

## 練習 (Exercises)

1. 閱讀 MM1 第 3.2 節。在固定 2B LLM、5000 萬張圖片的預算下，哪個 encoder 勝出？換成 13B LLM 時答案會反過來嗎？為什麼？

2. Cambrian-1 發現串接 DINOv2 + SigLIP 在以視覺為中心的 benchmark 上勝過單獨任一個，但在 MMMU 上沒有增加訊號。預測哪些 benchmark 會進步、哪些維持不變。

3. 你的目標是 2B LLM 上的行動 UI agent。挑選 encoder、connector、解析度與資料混合。用具體的 ablation 表格為每個選擇辯護。

4. Molmo 推出 4B 與 72B 模型。4B 與封閉的 7B VLM 不相上下；72B 在 11/11 項 benchmark 上勝過 Llama-3.2-90B-Vision。這告訴你關於 LLM 大小平台期假說什麼？

5. 設計一個 ablation 表格，在 7B VLM 上把資料混合品質與 encoder 品質分開。最少需要幾次訓練？提出四種軸的設定。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Ablation | "Turning one knob" | 進行多次訓練，彼此只在設計空間的一個軸上不同，其他一切固定 |
| Connector | "Bridge" / "projector" | 可訓練的模組，把 vision encoder 的輸出對應到 LLM 的 token 空間（MLP、Q-Former、Perceiver） |
| Detailed human caption | "Dense caption" | 人工撰寫的多句描述（通常 80-300 個 token），比網路 alt text 豐富 |
| Distillation | "GPT-4V captions" | 由更強的專有 VLM 產生的訓練資料；方便，但容易繼承幻覺 |
| AnyRes / dynamic res | "High-res path" | 透過 tiling 或 M-RoPE，餵入大於 encoder 原生解析度的圖片的策略 |
| Resolution ramp | "Curriculum" | 從低解析度開始、逐步提高的訓練排程，加快 alignment 學習 |
| Vision-centric bench | "CV-Bench / BLINK" | 強調細緻視覺感知、而非重語言推理的評估 |
| PixMo | "Molmo's data" | Allen AI 的 71.2 萬張密集 caption 圖片資料集；人類語音轉錄成密集 caption |

## 延伸閱讀 (Further Reading)

- [McKinzie et al. — MM1 (arXiv:2403.09611)](https://arxiv.org/abs/2403.09611)
- [Laurençon et al. — Idefics2 / What matters building VLMs (arXiv:2405.02246)](https://arxiv.org/abs/2405.02246)
- [Deitke et al. — Molmo and PixMo (arXiv:2409.17146)](https://arxiv.org/abs/2409.17146)
- [Tong et al. — Cambrian-1 (arXiv:2406.16860)](https://arxiv.org/abs/2406.16860)
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865)
