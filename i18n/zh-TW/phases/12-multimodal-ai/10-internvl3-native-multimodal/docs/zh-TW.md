# InternVL3：原生 Multimodal Pretraining (InternVL3: Native Multimodal Pretraining)

> 在 InternVL3 之前，每個開源 VLM 都遵循同樣的三步驟配方：拿一個用數兆文字 token 訓練好的文字 LLM，接上 vision encoder，再 fine-tune 接縫處。這樣可行，但有 alignment debt：文字 LLM 已經把全部 pretraining 預算花在純文字上，並不原生理解 visual token。事後才加入視覺時，LLM 必須重新學習如何把視覺輸入與它的文字推理關聯起來，又不能忘掉文字能力。InternVL3 (Zhu et al., 2025 年 4 月) 拒絕事後補上的做法：只做一次 pretraining，從第一步就交錯文字與 multimodal 資料。結果在 780 億參數的開源模型上，於 MMMU-Pro 追平 Gemini 2.5 Pro。這堂課讀原生 pretraining 的論點，以及採用它之後有什麼改變。

**Type:** Learn
**Languages:** Python (stdlib, training-corpus mixer)
**Prerequisites:** Phase 12 · 05, Phase 12 · 07 (recipes)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼事後訓練 VLM 會累積 alignment debt，並引用三個可量測的症狀（catastrophic forgetting、答案漂移、視覺與文字不一致）。
- 描述 InternVL3 的原生 pretraining 語料混合，以及文字 : 交錯 : caption 的比例為什麼重要。
- 比較 V2PE（variable visual position encoding）與 Qwen2-VL 的 M-RoPE。
- 說出 Visual Resolution Router (ViR) 與 Decoupled Vision-Language (DvD) 這兩個部署最佳化。

## 問題 (The Problem)

事後訓練 VLM 是預設做法。LLaVA、BLIP-2、Qwen-VL、Idefics 都是拿已經預訓練好的 LLM（Llama、Vicuna、Qwen、Mistral）再加上視覺。訓練階段通常長這樣：

1. 凍結 LLM + 凍結 vision encoder + 可訓練的 projector，在 caption 配對上訓練來對齊 embedding。
2. 解凍 LLM，在 instruction 資料 (LLaVA-Instruct, ShareGPT4V) 上訓練。
3. 可選的特定任務 fine-tune。

會出現三個 alignment debt 的症狀：

- Catastrophic forgetting。事後訓練的 VLM 忘掉純文字技能。GSM8K 分數掉 5-10 分，Hellaswag 分數下降，純文字 agent 退步。
- 答案漂移。同一個視覺問題只是措辭稍微不同，就得到不同答案。vision encoder 與 LLM 的連結，比 LLM 自己的 token 綁得更弱。
- 視覺與文字不一致。VLM 可以正確描述一張圖片，然後回答一個與自己描述矛盾的問題。visual token 參與 LLM 內部一致性檢查的方式，與文字不同。

這些症狀都有充分的記錄。MM1.5 第 4 節量化了它們。LLaVA-OneVision 的 ablation 也有暗示。原生 pretraining 就是答案。

## 核心概念 (The Concept)

### 原生 multimodal pretraining (Native multimodal pretraining)

InternVL3 在從第一步就是原生 multimodal 的語料上從頭訓練。混合比例是：

- 40% 純文字資料 (FineWeb、Proof-Pile-2 等)
- 35% 交錯圖文資料 (OBELICS、MMC4 風格)
- 20% 成對的圖片-caption 資料
- 5% 影片-文字資料

vision token、文字 token 與跨 modality 的互動，從第一個梯度步驟起就參與同一個 loss。沒有 alignment pretraining，沒有凍結 projector 的階段，也沒有需要復原的 catastrophic forgetting。

基礎模型的訓練是單一階段。之後才做 instruction tuning，但基礎模型已經把 visual token 當成第一級公民來理解。

### V2PE (variable visual position encoding)

Qwen2-VL 使用軸分配固定的 M-RoPE。InternVL3 引入 V2PE：位置編碼依 modality 類型（文字、圖片、影片）而變，並帶有可學習的縮放。實務上：

- 文字 token 得到 1D 位置（文字索引）。
- 圖片 patch 得到 2D 位置 (row, col)。
- 影片影格得到 3D 位置 (time, row, col)。

三者共用相同的 RoPE 頻率基底，但每個頻帶在 hidden 維度上的分配是可學習的參數，而不是固定切分。這讓 pretraining 期間能自由取捨時間與空間頻率的解析度。

V2PE 的 ablation 主張：在相同運算量下，影片 benchmark 比 M-RoPE 多 1-2 分。不是革命，但更乾淨。

### Visual Resolution Router (ViR)

部署最佳化。不是所有圖片都需要完整解析度編碼。一張只有一個低細節物體的照片，以 1280px 原生解析度編碼是浪費 token。ViR 是個小型分類器，在編碼之前預測回答問題所需的最低解析度。

路由有三個層級：低解析度（256 個 token）、中（576）、高（2048+）。在 production 流量中，60% 的查詢低或中解析度就足夠。淨效果：在相同品質下吞吐量提升 2-3 倍。

### Decoupled Vision-Language 部署 (DvD)

服務大型 VLM 時，vision encoder 每張圖片只跑一次，但 LLM 每個輸出 token 都要 autoregressive 地跑一次。兩個元件有不同的瓶頸（vision = conv + attention 的 GPU 記憶體頻寬；LLM = KV cache）。DvD 把它們拆到不同的 GPU，中間以串流連接。

對 8B + 4 億參數 encoder 的模型，DvD 相較於放在同處，大致讓每個節點的吞吐量加倍。

### 單階段與多階段的品質 (Single-stage vs multi-stage quality)

InternVL3 主要的 benchmark 主張：在 780 億參數下，追平 Gemini 2.5 Pro 的 MMMU-Pro。在 380 億時，追平 GPT-4o。在 80 億時，領先開源 8B 排行榜。全部都是靠單階段 pretrain + instruction-tune 的配方。

alignment-debt 假說是可量測的：每單位視覺 benchmark 的進步，InternVL3-8B 損失的文字 benchmark 分數（MMLU、GSM8K）比 Qwen2.5-VL-7B 少。模型更像通才，因為訓練是一整塊，而不是兩塊。

### InternVL3.5 與 InternVL-U

InternVL3.5 (2025 年 8 月) 擴大了配方。相同的原生 pretrain 做法，更多資料、更多參數。MMMU 的進步是漸進式的。

InternVL-U (2026) 加入統一生成：在同一個 backbone 之上，用 MMDiT head 輸出圖片。「U」代表「Understanding + generation」，追趕 Transfusion 式的統一模型（課程 12.13）。同一個原生 pretrain 的 backbone 同時支援理解與生成的 head。

### 原生 pretraining 的取捨 (Trade-offs of native pretraining)

原生 pretraining 並非免費：

- 運算。從頭訓練新的 VLM，成本與訓練文字 LLM 相同，要數百萬 GPU 小時。事後調整重用既有的 LLM 權重，省下大部分成本。
- 資料。大規模交錯圖文語料庫很稀少。OBELICS 是 1.41 億份文件；MMC4 是 5.71 億。光是文字就有 15T 個 token。multimodal pretraining 資料的稀缺是硬性限制。
- 基礎 LLM 的重用。原生 pretraining 放棄了之後換入新 LLM 的選項。事後調整讓你只需重訓 adapter，就能把 Llama-3.1 換成 Llama-4。

InternVL3 下的賭注是：alignment debt 比失去重用性更糟。benchmark 支持這個說法。但生產成本讓未來的實驗室很難便宜地複製。事後調整的 VLM 會繼續存在，因為對大多數專案來說仍然比較便宜。

```figure
l5-native-pretrain
```

## 實際使用 (Use It)

`code/main.py` 是訓練語料混合器與 ViR 路由模擬器。它會：

- 接收目標語料混合 (%text, %interleaved, %caption, %video)，並計算各 modality 的預期步數。
- 在一批查詢上模擬 ViR 路由（分布：50% 低細節、30% 中、20% 高細節），並回報平均 token 數。
- 根據 encoder 與 LLM 的 FLOPs，回報 DvD 吞吐量估算。
- 並排印出事後調整與原生 pretraining 在參數、運算、資料與預期 alignment-debt 症狀上的比較。

## 交付 (Ship It)

這堂課產出 `outputs/skill-native-vs-posthoc-auditor.md`。給定一份 VLM 訓練計畫，它會審查應該走原生還是事後路線，標出 alignment-debt 風險，並建議語料混合。當你評估新的開源 VLM 專案規模、需要挑選訓練策略時就用它。

## 練習 (Exercises)

1. 估算 InternVL3-8B（原生 pretrain）與 LLaVA-OneVision-7B（事後調整）之間的運算差距。GPU 小時的大致比例是多少？差距由什麼解釋？

2. InternVL3 回報 40% 文字 / 35% 交錯 / 20% caption / 5% 影片。如果你的目標任務偏重影片，提出新的比例，並論證為什麼基礎模型仍需要大量文字和 caption 資料。

3. 閱讀 MM1.5 第 4 節關於 forgetting 的內容。說出事後訓練出現最大退步的確切 benchmark。這個退步造成多大的代價？

4. ViR 把 60% 的流量路由到低解析度編碼。它會誤判哪些種類的查詢（需要高解析度卻送到低解析度）？提出三種 router 失敗模式。

5. DvD 把 vision 和 LLM 拆到不同的 GPU。在什麼流量模式下，DvD 反而會降低吞吐量而不是提升？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Native multimodal pretraining | "From scratch together" | 文字 + 圖片 + 影片 token 從第 1 步就參與 loss，而不是事後才接上 |
| Alignment debt | "Post-hoc penalty" | 把視覺接到凍結 LLM 上所造成、可量測的文字技能與答案一致性退步 |
| V2PE | "Variable visual pos encoding" | 每個 modality 可學習的位置編碼分配；InternVL3 對 M-RoPE 的後繼者 |
| ViR | "Resolution router" | 小型分類器，在編碼前為每個查詢選擇所需的最低解析度，節省推論 token |
| DvD | "Decoupled deployment" | vision encoder 在一張 GPU，LLM 在另一張，以串流交接；讓大型 VLM 的吞吐量加倍 |
| InternVL-U | "Unified understanding + generation" | 2026 年的後續作品，在原生 pretrain 的 backbone 上加入圖片生成 head |
| Interleaved corpus | "OBELICS / MMC4" | 文字與圖片依自然閱讀順序排列的文件；原生 pretraining 的原料 |

## 延伸閱讀 (Further Reading)

- [Chen et al. — InternVL 1 (arXiv:2312.14238)](https://arxiv.org/abs/2312.14238)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
- [InternVL3.5 (arXiv:2508.18265)](https://arxiv.org/abs/2508.18265)
- [InternVL-U (arXiv:2603.09877)](https://arxiv.org/abs/2603.09877)
- [Zhang et al. — MM1.5 (arXiv:2409.20566)](https://arxiv.org/abs/2409.20566)
