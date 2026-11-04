# LLaVA 與 Visual Instruction Tuning (LLaVA and Visual Instruction Tuning)

> LLaVA (2023 年 4 月) 是地球上被複製最多次的 multimodal 架構。它用 2 層 MLP 取代 BLIP-2 的 Q-Former，用單純的 token 串接取代 Flamingo 的 gated cross-attention，並用 GPT-4 從純文字 caption 產生的 15.8 萬輪 visual-instruction 對話來訓練。2023 到 2026 年間任何打造過 VLM 的人，做出來的都是 LLaVA 的某個變體。LLaVA-1.5 換上 MLP projector 和 336px 的 encoder。LLaVA-NeXT 加入 AnyRes tiling 來處理高解析度輸入。LLaVA-OneVision 在同一套配方中統一了單圖、多圖與影片。這堂課會讀這份配方、實作 projector，並解釋為什麼「簡單的贏了」。

**Type:** Build
**Languages:** Python (stdlib, projector + instruction-template builder)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 建立一個 2 層 MLP projector，把 ViT patch embedding（維度 1024）對應到 LLM 的 embedding 維度（維度 4096）。
- 走一遍 LLaVA 的兩階段配方：(1) 在 55.8 萬組 caption 配對上做 projector alignment，(2) 在 15.8 萬輪 GPT-4 產生的對話上做 visual instruction tuning。
- 建構 LLaVA 格式的 prompt，包含圖片 token 佔位符、system prompt 與 user/assistant 輪次。
- 解釋為什麼儘管 Q-Former 在 token 預算上占優，社群還是從 Q-Former 轉向 MLP。

## 問題 (The Problem)

BLIP-2 的 Q-Former（課程 12.03）把一張圖片壓縮成 32 個 token。乾淨、有效率、benchmark 表現好。但它有兩個問題。

第一，Q-Former 可以訓練，但它的 loss 不是最終任務。階段 1 訓練 ITC+ITM+ITG，階段 2 訓練 LM loss。query 學到某種中介表示，LLM 還得再解碼它。資訊在瓶頸處流失了。

第二，Q-Former 有 1.88 億參數，在 LLaVA 2023 年的規模下，你得跟目標 LLM 一起共同設計。換 LLM，就要重訓 Q-Former。換 vision encoder，也要重訓。每種組合都是獨立的研發專案。

LLaVA 的答案簡單得令人難為情：拿 ViT 的 576 個 patch token，各自通過一個 2 層 MLP (`1024 → 4096 → 4096`)，然後把全部 576 個倒進 LLM 的輸入序列。沒有瓶頸。沒有針對怪異目標的階段 1 pretraining。只要用直接的 LM loss 訓練 MLP。

資料從哪來？LLaVA 的第二個洞見：用 GPT-4（純文字）來產生 instruction 資料。把某張圖片的 COCO caption 和 bounding box 資料餵給 GPT-4，請它產生對話、描述與複雜的推理問題。免費得到 15.8 萬輪 instruction-response。不需要人工標註。

結果：一個 VLM 用 8 張 A100 跑一天，在 MMMU 上勝過 Flamingo，並釋出社群可以延伸的開源 checkpoint。到 2023 年底，它已催生 50 多個 fork。

## 核心概念 (The Concept)

### 架構 (The architecture)

7B 的 LLaVA-1.5：
- Vision encoder：CLIP ViT-L/14 @ 336（階段 1 凍結，階段 2 可選擇解凍）。
- Projector：帶 GELU 激活的 2 層 MLP，`1024 → 4096 → 4096`。
- LLM：Vicuna-7B v1.5（13B 版本用 Vicuna-13B；後來的 fork 改用 Llama-3.1-8B）。

對圖片 + 文字 prompt 的 forward pass：

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

圖片占用 LLM context 的 576 個 token。在 2048 的 context 下，剩 1472 個 token 給文字。在 32k context 下，這只是四捨五入的誤差。

### 階段 1：projector alignment

凍結 ViT，凍結 LLM。只訓練 2 層 MLP。資料集：55.8 萬組圖片-caption 配對 (LAION-CC-SBU)。Loss：以投影後的圖片 token 為條件，對 caption 做 language modeling。

一個 epoch、batch 128，幾個小時就完成。projector 學會把 ViT 空間對應到 LLM 空間。沒有任何特定任務的監督。

### 階段 2：visual instruction tuning

解凍 projector（仍可訓練）。解凍 LLM（通常是完整解凍，有時用 LoRA）。在 15.8 萬輪 visual-instruction 對話上訓練。

instruction 資料是關鍵技巧。Liu 等人的產生方式：
1. 取一張 COCO 圖片。
2. 擷取文字描述（5 條人工 caption + bounding box 清單）。
3. 用三種 prompt 模板送給 GPT-4：
   - 對話：「產生使用者與助理之間關於這張圖片的來回對話。」
   - 詳細描述：「對這張圖片給出豐富、詳細的描述。」
   - 複雜推理：「問一個需要對圖片做推理的問題，然後回答它。」
4. 把 GPT-4 的輸出解析成 (instruction, response) 配對。

這整個過程完全沒有直接碰到圖片，只用文字描述。GPT-4 會幻想出合理的圖片內容。有些雜訊，但行得通：15.8 萬輪就足以解鎖對話能力。

### 為什麼社群複製這套做法 (Why the community copied this)

- 沒有階段 1 專屬的 loss 要調。全程 LM loss。
- projector 幾小時就訓練完，不是幾天。
- 只要重訓 projector，就能更換 LLM（LLaVA-Llama2、LLaVA-Mistral、LLaVA-Llama3）。
- visual-instruction 資料管線使用 GPT-4，換一個新領域重新產生很便宜。

### LLaVA-1.5 與 LLaVA-NeXT

LLaVA-1.5 (2023 年 10 月) 加入：
- 把學術任務資料 (VQA、OKVQA、RefCOCO) 混進 instruction tuning。
- 更好的 system prompt。
- 用 2 層 MLP projector 取代原本單一的線性層，並用 336px 的 CLIP ViT-L/14（576 個 token）取代 224px（256 個 token）。訓練序列維持在 2048 個 token。
- 論文中提出一種網格切分的 HD 變體，後來 LLaVA-NeXT 以 AnyRes 的名稱正式推出。

LLaVA-NeXT (2024 年 1 月) 加入：
- AnyRes：把高解析度圖片切成 2x2 或 1x3 的 336x336 裁切網格，外加一張全域低解析度縮圖。每個裁切變成 576 個 token。2x2 網格加縮圖，每張圖片約 2880 個 visual token。1x3 網格的裁切較少，所以 token 也較少。OCR 與圖表任務的表現大幅躍升。
- 以 ShareGPT4V（高品質 GPT-4V caption）改善的 instruction 資料混合。
- 更強的基礎 LLM (Mistral-7B、Yi-34B)。

### LLaVA-OneVision

課程 12.08 會深入介紹 OneVision。簡短版本：相同的 projector，但用一套 curriculum 訓練，在單一模型中涵蓋單圖、多圖與影片，並共用 visual-token 預算。

### 與 Q-Former 的比較 (The comparison to Q-Former)

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| 每張圖片的 visual token | 32 | 576（基本）或約 2880（AnyRes，2x2 網格） |
| 可訓練參數 | 1.88 億 + LM | 約 2200 萬 (7B) + LM |
| 階段 1 loss | ITC+ITM+ITG | 只有 LM |
| 更換 LLM | 需要重訓 | 只需少量重訓即可替換 |
| 多圖 | 不方便 | 自然（串接） |
| 影片 | 不方便 | 自然（逐影格串接） |
| Token 預算 | 小 | 大 |

MLP 贏在簡單與 token 彈性。Q-Former 贏在 token 預算。到 2023 年底，token 預算不再是決定性的限制（LLM context 成長到 32k-128k 以上），簡單性勝出。

### Prompt 格式 (The prompt format)

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` 是佔位 token。在 tokenization 之前，它會被換成 576 個 visual token（使用 2x2 AnyRes 網格時約 2880 個）。tokenizer 看到的序列比訓練時略長，但 LLM 能處理這種新的輸入，因為階段 1 已經教過它了。

### 參數經濟學 (Parameter economy)

LLaVA-1.5-7B 的拆解：
- CLIP ViT-L/14 @ 336：3.03 億（階段 1 凍結，階段 2 常常解凍）。
- Projector（2 層線性）：約 2200 萬可訓練。
- Llama-7B：70 億。
- 總計：73 億參數。階段 2 期間可訓練：完整的 70 億 + 2200 萬 projector。

階段 2 的訓練成本：8 張 A100 約 20 小時。這是關鍵數字，一天、一個節點、可重現。這就是 LLaVA 能廣為流傳的原因。

```figure
mm-llava-projector
```

## 實際使用 (Use It)

`code/main.py` 實作了：

1. 純 Python 的 2 層 MLP projector（玩具規模：維度 16 → 32 → 32）。
2. prompt 建構流程：system prompt + 被換成 N 個投影 token 的 `<image>` + user 輪次 + assistant 生成佔位符。
3. 一個視覺化工具，顯示 576 個 token 的 visual 區塊在 LLM context 中看起來如何（占 2k / 32k / 128k context 的百分比）。

## 交付 (Ship It)

這堂課產出 `outputs/skill-llava-vibes-eval.md`。給定一個 LLaVA 家族的 checkpoint，它會執行 10 題 prompt 的 vibes-eval 套件（3 題 captioning、3 題 VQA、2 題推理、2 題拒答），並回報人類可讀的成績單。這不是 benchmark；是用來確認 projector 與 LLM 銜接良好的 smoke test。

## 練習 (Exercises)

1. 計算 `1024 → 4096 → 4096` 的 2 層 MLP projector 的可訓練參數量。含 GELU 與 bias，它占 LLaVA-13B 的多少比例？

2. 為「拒答」情境建構一個 LLaVA prompt：圖片中有一位私人個體。寫出預期的 assistant 回應。為什麼 LLaVA 應該在 zero-shot 時拒答，又需要什麼訓練資料來強化這種拒答？

3. 閱讀 LLaVA-NeXT 部落格中的 AnyRes 章節。計算 1344x672 圖片在 AnyRes 下的 visual token 數量，並與 336x336 的基本 576 個 token 比較。

4. LLaVA 階段 1 的 projector 是以 caption 的 LM loss 訓練的。如果跳過階段 1，直接進入階段 2 (visual instruction tuning) 會怎樣？請引用 Prismatic VLMs 的 ablation (arXiv:2402.07865) 來回答。

5. LLaVA-Instruct-150k 使用 GPT-4 搭配 COCO caption 來產生 instruction。針對新領域（醫療 X 光、衛星影像），描述產生領域 instruction 的四步驟資料管線。每個步驟可能出什麼問題？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|----------------|------------------------|
| Projector | "MLP bridge" | 帶 GELU 的 2 層 MLP，把 ViT 維度對應到 LLM 維度 |
| Image token | "<image> placeholder" | prompt 中的標記，推論前被換成 N 個投影後的 visual token |
| Visual instruction tuning | "LLaVA stage 2" | 在 GPT-4 產生的 (image, instruction, response) 三元組上訓練 |
| Stage 1 alignment | "Projector pretraining" | 凍結 ViT 和 LLM，以 caption 的 LM loss 訓練 projector |
| AnyRes | "Multi-crop tiling" | 把高解析度圖片切成 tile 網格，並串接每個 tile 的 visual token |
| LLaVA-Instruct | "GPT-4-generated" | 由 COCO caption + GPT-4 合成的 15.8 萬組 instruction-response 配對 |
| Vision encoder freeze | "Backbone locked" | CLIP 權重在階段 1 不更新，有時階段 2 也不更新 |
| ShareGPT4V | "Better captions" | 由 GPT-4V 產生的 100 萬筆密集 caption，用於更高品質的 alignment |
| VQA | "Visual question answering" | 針對圖片回答自由形式問題的任務 |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 的 ablation，系統性測試 projector 與資料的選擇 |

## 延伸閱讀 (Further Reading)

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — LLaVA 論文。
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5。
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — 密集 caption 資料集。
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — 設計空間的 ablation。
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — 統一單圖、多圖與影片。
