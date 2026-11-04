# DeepSeek-V3 架構導覽 (DeepSeek-V3 Architecture Walkthrough)

> Phase 10 · Lesson 14 點出了每個開放模型都會轉動的六個架構旋鈕。DeepSeek-V3（2024 年 12 月，總共 671B 參數、37B 活躍）把這六個全都轉了，還加上四個：Multi-Head Latent Attention、無輔助 loss 的負載平衡、Multi-Token Prediction，以及 DualPipe 訓練。這堂課由上到下閱讀 DeepSeek-V3 的架構，並從公開的 config 推導出每一個參數量。學完之後，你可以解釋為什麼 671B/37B 這個比例是對的賭注，以及為什麼 MLA + MoE 合在一起，在 frontier 上勝過任何單獨一個。

**Type:** Learn
**Languages:** Python (stdlib, parameter calculator)
**Prerequisites:** Phase 10 · 14 (open-model walkthroughs), Phase 10 · 17 (NSA), Phase 10 · 18 (MTP), Phase 10 · 19 (DualPipe)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 由上到下閱讀 DeepSeek-V3 的 config，並以六個 GPT-2 旋鈕加上四個 DeepSeek 特有的新增項目，來解釋每個欄位。
- 推導總參數量 (671B)、活躍參數量 (37B)，以及對各自有貢獻的元件。
- 計算 MLA 在 128k context 下的 KV cache 佔用量，並與一個活躍參數相同、搭配 GQA 的 dense 模型所需付出的相比。
- 陳述 DeepSeek 特有的四項創新（MLA、MTP、無輔助 loss 的 routing、DualPipe），並指出各自針對架構/訓練堆疊的哪個部分。

## 問題 (The Problem)

DeepSeek-V3 是第一個在架構上與 Llama 家族有實質差異的 frontier 開放模型。Llama 3 405B 是「轉了六個旋鈕的 GPT-2」。DeepSeek-V3 則是轉了全部六個旋鈕、外加另外四個的 GPT-2。讀 Llama 3 的 config 是讀 DeepSeek config 的暖身，但深層結構，包括 attention block 的形狀、routing 的邏輯、訓練時的目標，差異大到你需要另一份獨立的導覽。

學會它的回報：DeepSeek-V3 開放權重的釋出，改變了開放模型中「frontier 能力」的意義。這個架構是許多 2026 年訓練在抄襲的藍圖。了解它是任何接觸 frontier LLM 訓練或推論的職位的基本門檻。

## 核心概念 (The Concept)

### 再談不變的核心 (The invariant core, again)

DeepSeek-V3 仍然是自迴歸的。它仍然堆疊 decoder block。每個 block 仍然有 attention 加 MLP 加兩個 RMSNorm。它在 MLP 中仍然使用 SwiGLU。它仍然使用 RoPE。Pre-norm。Embedding 的 weight tying。與每個 Llama 或 Mistral 的 baseline 相同。

### 轉折：用 MLA 取代 GQA (The twist: MLA instead of GQA)

從 Phase 10 · 14 你知道 GQA 透過讓多組 Q head 共用 K 和 V，來縮小 KV cache。Multi-Head Latent Attention (MLA) 更進一步：K 和 V 被壓縮成共用的低秩潛在表示（`kv_lora_rank`），然後在運算時即時為每個 head 解壓縮。KV cache 只儲存潛在表示，通常是每個 token 每層 512 個 float，而不是 8 x 128 = 1024 個 float。

在 128k context 下，使用 MLA 的 DeepSeek-V3（每個 token 每層一個共用的潛在 `c^{KV}`；K 和 V 都是透過 up-projection 從這個潛在表示導出，而 up-projection 可以被吸收進後續的 matmul）：

```
kv_cache = num_layers * kv_lora_rank * max_seq_len * bytes_per_element
         = 61 * 512 * 131072 * 2
         = 7.6 GB
```

一個假設的 GQA baseline（Llama 3 70B 的形狀，8 個 KV head，head dim 128）則要付出：

```
kv_cache = 2 * 61 * 8 * 128 * 131072 * 2
         = 30.5 GB
```

在 128k context 下，MLA 比 Llama-3-70B 風格的 GQA cache 小 4 倍。

取捨：MLA 在每次 attention 計算中（每個 head）多了一個解壓縮步驟。相較於省下的頻寬，額外的運算很小。對長 context 推論是淨贏。

### Routing：無輔助 loss 的負載平衡 (The routing: auxiliary-loss-free load balancing)

MoE router 決定哪些 top-k 個 expert 處理每個 token。天真的 router 會把太多工作集中在少數 expert，讓其他的閒置。標準的修正：加一個輔助 loss 項來懲罰負載不平衡。這有效，但會略微降低主要任務的表現。

DeepSeek-V3 引入了無輔助 loss 的方案。每個 expert 的 bias 項被加到 router 的 logit 上，並在訓練期間以一條簡單的規則調整：如果 expert `e` 超載，就降低 `bias_e`；如果負載不足，就提高它。沒有額外的 loss 項。訓練保持乾淨。Expert 的負載維持平衡。

對主要 loss 的影響：沒有可量測的影響。對 MoE 架構的影響：更乾淨，沒有需要調整的輔助 loss 超參數。

### MTP：更密集的訓練 + 免費的 draft (The MTP: denser training + free draft)

從 Phase 10 · 18 你知道 DeepSeek-V3 加入了 D=1 個 MTP 模組，預測往後兩個位置的 token。推論時，訓練好的模組被重新用作 speculative-decoding 的 draft，接受率達 80% 以上。訓練時，每個 hidden state 在 D+1 = 2 個目標上被監督，提供更密集的訊號。

參數：在 671B 主模型之上多 14B。開銷：2.1%。

### 訓練：DualPipe (The training: DualPipe)

從 Phase 10 · 19 你知道 DualPipe 是雙向 pipeline，讓 forward 與 backward 的 chunk，與跨節點的 all-to-all 通訊重疊。在 DeepSeek-V3 2,048 張 H800 的規模下，它回收了大約 24.5 萬 GPU 小時，這些是 1F1B 會因 pipeline bubble 而損失的。

### 逐欄位解讀 config (The config, field by field)

這是 DeepSeek-V3 的 config（簡化版）：

```
hidden_size: 7168
intermediate_size: 18432   (dense MLP hidden size, used on first few layers)
moe_intermediate_size: 2048 (expert MLP hidden size)
num_hidden_layers: 61
first_k_dense_layers: 3    (first 3 layers use dense MLP)
num_attention_heads: 128
num_key_value_heads: 128   (formally equal to num_heads under MLA, but
                           the real compression is in kv_lora_rank)
kv_lora_rank: 512          (MLA latent dimension)
num_experts: 256            (MoE expert count per block)
num_experts_per_tok: 8      (top-8 routing)
shared_experts: 1           (always-on shared expert per block)
max_position_embeddings: 163840
rope_theta: 10000.0
vocab_size: 129280
mtp_module: 1               (1 MTP module at depth 1)
```

逐項解析：

- `hidden_size=7168`：embedding 維度。
- `num_hidden_layers=61`：block 總深度。
- `first_k_dense_layers=3`：前 3 個 block 使用大小為 18432 的 dense MLP。其餘 58 個使用 MoE。
- `num_attention_heads=128`：128 個 query head。
- `kv_lora_rank=512`：K 和 V 被壓縮到這個潛在維度，再為每個 head 解壓縮。
- `num_experts=256, num_experts_per_tok=8`：每個 MoE block 有 256 個 expert，routing top-8。
- `shared_experts=1`：除了 256 個 routed expert 之外，還有 1 個永遠啟用的 expert 對每個 token 都有貢獻。把它想成「dense 的底線」，確保每個 token 都能得到一些可靠的東西。
- `moe_intermediate_size=2048`：每個 expert 的 MLP hidden 大小。比 dense MLP 小，因為有 256 個。

### 參數計算 (Parameter accounting)

完整的計算在 `code/main.py` 中。重點如下：

- Embedding：`vocab * hidden = 129280 * 7168 = ~0.93B`。
- 前 3 個 dense block：使用 MLA 的 attention（計算器的簡化計算中每個 block 約 84M）+ dense SwiGLU MLP（每個 block `3 * 7168 * 18432 = ~396M`）+ norm。總計約 1.4B。
- 58 個 MoE block：使用 MLA 的 attention（~84M）+ 256 個 routed expert（每個 `3 * 7168 * 2048 = ~44M`）+ 1 個 shared expert（~44M）+ router + norm。每個 block 約 11.4B，包含所有 expert。58 個 MoE block 約 661B。
- MTP 模組：公開的模組是 14B。計算器使用約 0.8B 的 dense 替身。

總計：計算器得出約 664B，而主模型公開的是 671B。Hugging Face 的 checkpoint 是 685B，因為它在 671B 之上加了 14B 的 MTP 模組。其餘約 1% 的差距來自簡化：計算器略過了 MLA 中解耦的 RoPE 維度、未 tying 的輸出 head，以及一些小的 tensor。

每次 forward 的活躍參數：

- Attention：每層 84M * 61 = 約 5.1B（所有層都會觸發）。
- MLP 活躍部分：前 3 層是 dense (`3 * 396M = ~1.2B`)，58 個 MoE 層各自啟用 8 個 routed + 1 個 shared expert (`9 * 44M = ~396M` 每層，總計約 23B)。合計約 24B。
- Embedding + norm：約 0.9B。
- 總活躍：計算器得出約 30B，而公開的是 37B。大部分差距在 attention：完整的 MLA 帶有 RoPE 維度以及更寬的 query 與 key head，比簡化的計算更大。MTP 模組不計入 37B。

### 671B / 37B 的比例 (The 671B / 37B ratio)

18 倍的稀疏比（活躍參數是總量的 5.5%）。DeepSeek-V3 是已釋出開放權重的 frontier MoE 模型中最稀疏的。Mixtral 8x7B 的比例 13/47 (28%) 要密得多。Llama 4 Maverick 的比例 17B/400B (4.25%) 則相當。DeepSeek 的賭注是：在 frontier 規模下，更多 expert 搭配更低的啟用比例，能在每個活躍 FLOP 上產生更好的品質。

### DeepSeek-V3 的位置 (Where DeepSeek-V3 sits)

| 模型 | 總量 | 活躍 | 比例 | Attention | 新穎想法 |
|-------|------|-------|-------|-----------|-------------|
| Llama 3 70B | 70B | 70B | 100% | GQA 64/8 | — |
| Llama 4 Maverick | 400B | 17B | 4.25% | GQA | — |
| Mixtral 8x22B | 141B | 39B | 27% | GQA | — |
| DeepSeek V3 | 671B | 37B | 5.5% | MLA 512 | MLA + MTP + aux-free + DualPipe |
| Qwen 2.5 72B | 72B | 72B | 100% | GQA 64/8 | YaRN 延伸 |

### 後續：R1、V4 (The follow-on: R1, V4)

DeepSeek-R1 (2025) 是在 V3 骨幹上做的推理訓練。R1 使用相同的架構。改變的是 post-training 的配方（在可驗證任務上做大規模 RL），而不是 pretraining 的架構。

DeepSeek-V4（如果發布的話）預期會保留 MLA + MoE + MTP，並加入 DSA (DeepSeek Sparse Attention)，也就是 Phase 10 · 17 中 NSA 的後繼者。這個系譜是穩定的：架構層級的創新不斷累積；每個版本再轉動額外的旋鈕。

```figure
moe-routing
```

## 實際使用 (Use It)

`code/main.py` 是專門針對 DeepSeek-V3 形狀的參數計算器。執行它，把輸出與論文中的數字比較，並用在假設的變體上（256 個 expert 對 512 個、top-8 對 top-16、MLA rank 512 對 1024）。

要看的重點：

- 總參數量對照公開的 671B。
- 活躍參數量對照公開的 37B。
- 128k context 下的 KV cache，也就是 MLA 與 GQA 的比較。
- 每層的拆解，看參數預算實際上花在哪裡。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-deepseek-v3-reader.md`。給定一個 DeepSeek 家族的模型（V3、R1 或任何未來的變體），它會逐元件地閱讀架構，說出 config 的每個欄位、按元件推導參數量，並指出該模型使用了 DeepSeek 特有四項創新中的哪幾項。

## 練習 (Exercises)

1. 執行 `code/main.py`。將計算器的總參數量估計與公開的 671B 比較，並找出差異來自哪裡。論文的第 2 節有完整的逐項明細。

2. 修改 config，使用 MLA rank 256 而不是 512。計算在 128k context 下得到的 KV cache 大小。它減少了百分之多少，又以每個 head 的表達力為代價？

3. 比較 DeepSeek-V3 的（256 個 expert、top-8）routing 與假設的（512 個 expert、top-8）變體。總參數量成長；活躍參數量維持不變。額外的 expert 容量在理論上帶來什麼，在推論時又要付出什麼代價？

4. 閱讀 DeepSeek-V3 技術報告 (arXiv:2412.19437) 關於 MLA 的第 2.1 節。用三句話解釋，為什麼 K 和 V 的解壓縮矩陣可以被「吸收」進後續的 matmul，以提高推論效率。

5. DeepSeek-V3 在大部分運算中使用 FP8 訓練。計算 FP8 相對於 BF16 儲存 671B 權重所節省的記憶體。這與 14.8T token 的訓練預算有什麼交集？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MLA | 「Multi-Head Latent Attention」 | 把 K 和 V 壓縮成共用的低秩潛在表示（kv_lora_rank，通常是 512），並即時為每個 head 解壓縮，KV cache 只儲存潛在表示 |
| kv_lora_rank | 「MLA 的壓縮維度」 | K 和 V 共用的潛在表示大小，DeepSeek-V3 使用 512 |
| First k dense layers | 「前面的層維持 dense」 | MoE 模型最前面的幾層跳過 MoE router，改用 dense MLP 以求穩定 |
| num_experts_per_tok | 「Top-k routing」 | 每個 token 觸發多少個 routed expert，DeepSeek-V3 使用 8 |
| Shared experts | 「永遠啟用的 expert」 | 不論 routing 如何，都處理每個 token 的 expert，DeepSeek-V3 使用 1 個 |
| Auxiliary-loss-free routing | 「以 bias 調整的負載平衡」 | 訓練期間調整每個 expert 的 bias 項，在不加 loss 項的情況下維持 expert 負載平衡 |
| MTP module | 「額外的預測 head」 | 從 $h^{(1)}$ 與 $E(t+1)$ 預測 $t+2$ 的 transformer block，更密集的訓練，加上免費的 speculative-decoding draft |
| DualPipe | 「雙向 pipeline」 | 讓 forward/backward 運算與跨節點 all-to-all 重疊的訓練排程 |
| Active parameter ratio | 「稀疏度」 | active_params / total_params，DeepSeek-V3 達到 5.5% |
| FP8 training | 「8-bit 訓練」 | 以 FP8 進行訓練的儲存與許多運算，相較於 BF16 約省一半記憶體，只付出少量品質代價 |

## 延伸閱讀 (Further Reading)

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：完整的架構、訓練與結果文件
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)：config 檔案與部署說明
- [DeepSeek-V2 paper (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)：引入 MLA 的前身
- [DeepSeek-R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948)：建立在 V3 架構上的推理訓練後繼者
- [Native Sparse Attention (arXiv:2502.11089)](https://arxiv.org/abs/2502.11089)：DeepSeek 家族 attention 的未來方向
- [DualPipe repository](https://github.com/deepseek-ai/DualPipe)：訓練排程的參考實作
