# 從 CLIP 到 BLIP-2：Q-Former 作為 Modality 之間的橋樑 (From CLIP to BLIP-2 — Q-Former as Modality Bridge)

> CLIP 能對齊圖片與文字，卻無法產生 caption、回答問題或進行對話。BLIP-2 (Salesforce, 2023) 用一座小型、可訓練的橋樑解決了這件事：32 個可學習的 query 向量，透過 cross-attention 去關注凍結 ViT 的特徵，然後直接放進凍結 LLM 的輸入串流。這座只有 1.88 億參數的橋，把 ViT-g/14 接上了大到 110 億參數的 Flan-T5-XXL 等凍結 LLM。到 2026 年為止，所有基於 adapter 的 VLM（MiniGPT-4、InstructBLIP、LLaVA 的親戚們）都是它的後代。這堂課會讀 Q-Former 的架構、解釋它的兩階段訓練，並做出一個玩具版本，把 visual token 餵進凍結的文字 decoder。

**Type:** Build
**Languages:** Python (stdlib, cross-attention + learnable-query demo)
**Prerequisites:** Phase 12 · 02 (CLIP), Phase 7 (Transformers)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼在凍結的 vision encoder 與凍結的 LLM 之間放一個可訓練的瓶頸，在成本與穩定性上都勝過 end-to-end finetuning。
- 實作一個 cross-attention block，讓一組固定的 learnable query 去關注外部的圖片特徵。
- 走一遍 BLIP-2 的兩階段 pretraining：先是表示學習 (ITC + ITM + ITG)，再是生成式學習（搭配凍結 decoder 的 LM loss）。
- 比較 Q-Former 與 LLaVA 使用的較簡單 MLP projector，並論證各自在什麼情況下勝出。

## 問題 (The Problem)

你有一個凍結的 ViT，每張圖片產生 256 個維度 1408 的 patch token。你還有一個凍結的 7B LLM，它預期的是維度 4096 的 token embedding。最直觀的橋樑，是從 1408 到 4096 的一個線性層。這樣可行，但把全部 256 個 patch token 餵進 LLM 的 context，每張圖片就多花 256 個 token。一個 batch 有 32 張圖片的話，光是視覺 modality 就吃掉 8192 個 token。

BLIP-2 的問題是：能不能把 256 個 token 的圖片表示壓縮成少得多的 token（比方 32 個），同時保留足夠的資訊，讓 LLM 能產生 caption、回答問題，並對圖片進行推理？而且能不能在完全不動凍結 backbone 的情況下訓練這座橋，讓訓練成本只有橋本身的參數？

答案是 Q-Former。32 個可學習的「query」向量，對 ViT 的 patch token 做 cross-attention，產生 32 個 token 的視覺摘要，交給 LLM 使用。總共 1.88 億參數。在碰到 LLM 之前，先用 contrastive、matching 與 generative 目標訓練。

## 核心概念 (The Concept)

### Learnable query

Q-Former 的核心技巧：不讓 LLM 的文字 token 去關注圖片 patch，而是引入一組新的 32 個可學習 query 向量 `Q`，讓*它們*去關注圖片 patch。這些 query 是模型的參數，在訓練中學習，而且每張圖片都用同樣的 32 個 query。

經過 cross-attention 後，每個 query 都帶著圖片的壓縮摘要，例如「描述主要物體」、「描述背景」、「數物體」等等。這些 query 並不會真的專精於語意標籤，它們學到的是任何能讓下游 loss 下降的編碼方式。

### 架構 (Architecture)

Q-Former 是一個小型 transformer（12 層，約 1 億參數），有兩條路徑：

1. Query 路徑：32 個 query 向量先經過 self-attention（彼此之間），再對凍結 ViT 的 patch token 做 cross-attention，最後是 FFN。
2. 文字路徑：類似 BERT 的 text encoder，與 query 路徑共用 self-attention 和 FFN 的權重。文字路徑關閉 cross-attention。

訓練時兩條路徑都會跑。query 與文字透過共用的 self-attention 互動，代表 query 可以在需要的任務 (ITM、ITG) 中以文字為條件。在交給 VLM 的推論階段，只有 query 會流過，產生 32 個 visual token。

### 兩階段訓練 (Two-stage training)

BLIP-2 分兩個階段做 pretraining：

階段 1：表示學習（沒有 LLM）。三個 loss：
- ITC (image-text contrastive)：在 pooled 的 query token 與文字 CLS token 之間，做 CLIP 風格的 contrastive。
- ITM (image-text matching)：二元分類器，判斷這組圖文是否配對。使用 hard-negative mining。
- ITG (image-grounded text generation)：針對文字的 causal LM head，以 query 為條件。迫使 query 編碼出可用來生成文字的內容。

只有 Q-Former 在訓練，ViT 是凍結的，沒有 LLM 參與。

階段 2：生成式學習。接上一個凍結的 LLM（OPT-2.7B 或 Flan-T5-XL 等）。透過一個小的線性層，把 32 個 query 輸出投影到 LLM 的 embedding 維度，加在文字 prompt 前面。在串接後的 prompt + 圖片 + caption 序列上，以 LM loss 只訓練線性投影與 Q-Former。

階段 2 之後，Q-Former 加投影層就是完整的視覺 adapter。推論時：圖片 → ViT → Q-Former → 線性投影 → 加到文字前面 → 凍結 LLM 輸出結果。

### 參數經濟學 (Parameter economics)

BLIP-2 搭配 ViT-g/14（11 億，凍結）+ OPT-6.7B（67 億，凍結）+ Q-Former（1.88 億，訓練）= 總共 80 億，其中 1.88 億參與訓練。光 Q-Former 就約占整個堆疊參數的 2.4%。訓練成本也反映了這點：用少數幾張 A100 跑幾天，而 end-to-end 要幾週。

品質：BLIP-2 在 zero-shot VQA 上追平或勝過 Flamingo-80B，而體積小了 50 倍。這座橋有用。

### InstructBLIP 與具備指令感知的 Q-Former

InstructBLIP (2023) 為 Q-Former 增加一個額外輸入：指令文字本身。在 cross-attention 時，query 現在能同時取得圖片 patch 與指令。query 可以針對每個指令專精（「數車子」、「描述氛圍」），而不是學一份固定的摘要。在 held-out 任務上的 benchmark 有進步。

### MiniGPT-4 與只訓練 projector 的做法

MiniGPT-4 保留 Q-Former，但凍結其餘一切，只訓練輸出的線性投影。便宜，但代價是品質：query 是 BLIP-2 的，不是你的。適合快速迭代，但不是最好的架構。

### 為什麼 LLaVA 選擇更簡單的做法 (Why LLaVA went simpler)

LLaVA (2023，課程 12.05) 用一個單純的 2 層 MLP 取代 Q-Former，把每個 ViT patch token 都投影到 LLM 空間：24x24 網格每張圖片 576 個 token，全部餵給 LLM。壓縮得比較差，但讓 LLM 能關注原始 patch。當時這有爭議；到 2023 年底它成為主流，因為 visual instruction 資料 (LLaVA-Instruct-150k) 證明 MLP 可以被訓練到保留足夠的訊號。取捨是：LLaVA 的 context 填得更快，但自然地擴展到多圖片與影片。

到 2026 年，領域分成兩派：token 預算要緊的地方（長影片、多圖片）Q-Former 還活著；每個 token 的原始品質優先時，MLP projector 占主導。

### Gated cross-attention：Flamingo，祖先

Flamingo（課程 12.04）早於 BLIP-2，用的是同樣的 cross-attention 想法，但是放在每一個凍結 LLM 的層，而不是單一座橋。BLIP-2 展示了只壓縮到輸入層也行得通。Gemini 與 Idefics 把兩者結合：交錯的輸入 token，加上可選的 gated cross-attention 以支援 in-context few-shot。

### 2026 年的後代

- Q-Former：BLIP-2、InstructBLIP、MiniGPT-4，以及多數為了 token 預算考量的 video-language 模型。
- Perceiver resampler：Flamingo 的變體（課程 12.04）；Idefics 家族、Eagle、OmniMAE。
- MLP projector：LLaVA、LLaVA-NeXT、LLaVA-OneVision、Cambrian-1。
- Attention pool：VILA、PaliGemma。

四種都有效。決定性的問題是：你受限於 token 預算，還是受限於每個 token 的品質。

```figure
modality-projection
```

## 實際使用 (Use It)

`code/main.py` 建立了一個標準函式庫版的 Q-Former 式 cross-attention：

1. 模擬 256 個圖片 patch token（維度 128）。
2. 建立 32 個 learnable query（維度 128）。
3. 執行 scaled-dot-product cross-attention（Q 來自 query，K/V 來自 patch）。
4. 透過一個線性層投影到 LLM 維度 (512)。
5. 輸出 32 個可直接給 LLM 使用的 visual token。

所有數學都用純 Python（對向量做巢狀迴圈）。是玩具，但形狀正確。程式會印出 attention 權重矩陣，讓你看到每個 query 從哪些 patch 取資訊。

## 交付 (Ship It)

這堂課產出 `outputs/skill-modality-bridge-picker.md`。給定目標 VLM 設定（vision encoder 的 token 數、LLM 的 context 預算、部署限制、品質目標），它會推薦 Q-Former、MLP 或 Perceiver resampler，附上簡短理由，以及每種橋樑的參數量估算。

## 練習 (Exercises)

1. 用 PyTorch 實作 cross-attention block。驗證在 32 個 query 與 256 個 key/value 時，attention 權重矩陣是 32 x 256，而且 softmax 之後每個 row 的和為 1。

2. 在 BLIP-2 階段 1，Q-Former 同時跑三個 loss：ITC、ITM、ITG。用 pseudo-code 寫出各自的 forward signature。哪一個需要啟用文字 encoder 路徑？

3. 比較參數量：Q-Former（12 層、768 hidden）對上 2 層 MLP projector（1408 → 4096，兩層）。LLM 要到多大規模，1.88 億參數 Q-Former 的成本才能在訓練效率上回本？

4. 閱讀 BLIP-2 論文 (arXiv:2301.12597) 的第 3.2 節，了解 Q-Former 如何初始化。解釋為什麼從 BERT-base 初始化（而不是隨機）能加速收斂。

5. 對一段 10 分鐘、以 1 FPS 取樣成 60 個影格的影片，計算每個影格在 (Q-Former → 每影格 32 個 token) 與 (MLP projector → 每影格 576 個 token) 下的 token 成本。在 prompt 文字還沒占到任何空間之前，兩種橋各能在 128k token 的 LLM context window 中塞進多少影格？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|----------------|------------------------|
| Q-Former | "Querying transformer" | 帶有 32 個 learnable query 向量的小型 transformer，對凍結的 ViT 特徵做 cross-attention |
| Learnable queries | "Soft prompt for vision" | 一組固定的參數，作為 cross-attention 的 query 端；每個模型各自學習，所有輸入共用 |
| Cross-attention | "Q from here, K/V from there" | query、key、value 來自不同來源的 attention；query 就是靠它從 ViT patch 取資訊 |
| ITC | "Image-text contrastive" | 套用在 Q-Former pooled query 與文字 CLS 之間的 CLIP 風格 loss |
| ITM | "Image-text matching" | 針對 hard-negative mining 配對的二元分類器；迫使 query 區分細微的不匹配 |
| ITG | "Image-grounded text generation" | 以 query 為條件生成文字的 causal LM loss；迫使 query 編碼出可被解碼成文字的內容 |
| Two-stage pretraining | "Representation then generative" | 階段 1 單獨訓練 Q-Former (ITC/ITM/ITG)；階段 2 接上凍結 LLM，只訓練投影層與 Q-Former |
| Frozen backbone | "Do not finetune" | vision encoder 與 LLM 的權重固定；只有橋在訓練 |
| Projection head | "Linear to LLM dim" | 最後一個線性層，把 Q-Former 輸出對應到 LLM 的 embedding 維度 |
| Perceiver resampler | "Flamingo's version" | 類似的 learnable-query cross-attention；Flamingo 每張圖片跑一次產生固定 latent，再於每個 LLM 層透過 gated cross-attention 讀取 |

## 延伸閱讀 (Further Reading)

- [Li et al. — BLIP-2 (arXiv:2301.12597)](https://arxiv.org/abs/2301.12597) — 核心論文。
- [Li et al. — BLIP (arXiv:2201.12086)](https://arxiv.org/abs/2201.12086) — 前身，有 ITC/ITM/ITG 三件組。
- [Li et al. — ALBEF (arXiv:2107.07651)](https://arxiv.org/abs/2107.07651) — 「先對齊再融合」，階段 1 訓練的概念祖先。
- [Dai et al. — InstructBLIP (arXiv:2305.06500)](https://arxiv.org/abs/2305.06500) — 具備指令感知的 Q-Former。
- [Zhu et al. — MiniGPT-4 (arXiv:2304.10592)](https://arxiv.org/abs/2304.10592) — 只訓練 projector 的做法。
- [Jaegle et al. — Perceiver IO (arXiv:2107.14795)](https://arxiv.org/abs/2107.14795) — learnable-query cross-attention 的通用架構。
