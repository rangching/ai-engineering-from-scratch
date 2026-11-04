# Multi-Token Prediction (MTP) (Multi-Token Prediction (MTP))

> 從 GPT-2 到 Llama 3，每個自迴歸 LLM 在每個位置都用一個 loss 訓練：預測下一個 token。DeepSeek-V3 在每個位置加了第二個 loss：預測再下一個 token。額外的 14B 參數（在一個 671B 的模型上）透過 gradient 流動被蒸餾回主模型，而訓練好的 MTP head 在推論時被重新用作 speculative-decoding 的 drafter，接受率達 80% 以上。1.8 倍的生成吞吐量是白送的。這堂課打造 DeepSeek 技術報告中的 sequential MTP 模組，計算 loss 與共用 head 的參數配置，並解釋為什麼 MTP 保留了因果鏈，而 Gloeckle 等人最初的平行 MTP 卻破壞了它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 陳述 MTP 的訓練目標，並推導跨預測深度的聯合 loss。
- 解釋 Gloeckle 等人的平行 MTP head (2024) 與 DeepSeek-V3 的 sequential MTP 模組之間的差異，以及為什麼 sequential 設計能保留因果鏈。
- 計算在 pre-training 中加入 MTP 模組所增加的參數與記憶體開銷。
- 從零實作一個 MTP 模組：共用的 embedding、每個深度的 transformer block、投影，以及共用的輸出 head。

## 問題 (The Problem)

Next-token prediction 是標準的 LLM 訓練目標。每個 hidden state 都只被監督去預測一件事：緊接著的下一個 token。這是個出奇薄弱的訊號。序列中的大部分資訊延伸到超過一個 token 之外：結構、連貫性、事實性、算術的流程。模型得靠在數兆個 token 上累積許多單一 token 的訊號，才能學到這些。

MTP 問：如果每個 hidden state 都被監督去同時預測多個未來的 token 呢？Gloeckle 等人 (Meta, 2024) 證明這有幫助。他們的實作是在 backbone 之上放了數個獨立的輸出 head，每個預測不同的位移。平行、簡單，但這些 head 看的是同一個 hidden state，沒有任何階層式的精煉，而且預測並沒有因果地串接，所以不能用於 speculative decoding。

DeepSeek-V3（2024 年 12 月）把 MTP 重新設計成 sequential 模組，在每個預測深度都保留因果鏈。模型從 `h_i^(0)` 預測 `t+1`，然後從新的 hidden state `h_i^(1)` 預測 `t+2`，而 `h_i^(1)` 結合了 `h_i^(0)` 與 `E(t+1)` 的 embedding，依此類推。每個深度都是自己的小型 transformer block。共用的 embedding 與共用的輸出 head 讓參數開銷維持在適度範圍。在 DeepSeek-V3 的規模下，671B 的主模型權重之上，MTP 模組增加了 14B 額外參數。這 2% 的開銷買到了更密集的訓練訊號，以及推論時現成的 speculative-decoding draft。

這堂課從零打造單一個 MTP 模組以及 D 深度的 loss。數學很整齊。實作約 150 行。

## 核心概念 (The Concept)

### Sequential MTP 的配方 (The sequential MTP recipe)

DeepSeek-V3 在主模型之上加了 `D` 個 MTP 模組。主模型預測 `t_{i+1}`。每個模組 `k`（`k = 1..D`）在給定到位置 `i` 為止的前綴下，預測再往後 `k` 步的 token `t_{i+k+1}`。

模組 `k` 包含：

- 一個有自己的 attention 與 MLP 的 transformer block `T_k`。
- 一個投影矩陣 `M_k`，把前一深度的 hidden state 與下一深度的 ground-truth token 的 embedding 結合起來。
- 共用的 embedding `E`（與主模型相同）。
- 共用的輸出 head `Out`（與主模型相同）。

訓練時，對於到位置 `i` 為止的前綴，每個深度的 hidden state 為：

```math
\begin{aligned}
h_i^{(0)} &= \text{main model backbone at position } i \\
h_i^{(k)} &= T_k\Big( M_k \cdot \mathop{\mathrm{concat}}\big(\mathop{\mathrm{RMSNorm}}(h_i^{(k-1)}), \mathop{\mathrm{RMSNorm}}(E(t_{i+k}))\big) \Big) \quad \text{for } k \ge 1
\end{aligned}
```

每個深度的預測為：

```math
\text{logits}_{i+k+1} = \mathop{\mathrm{Out}}(h_i^{(k)}) \quad \text{for } k = 1, \ldots, D
```

每個深度的 loss 是對 ground-truth `t_{i+k+1}` 的 cross-entropy：

```math
L_k = \mathop{\mathrm{CE}}(\text{logits}_{i+k+1}, t_{i+k+1})
```

跨深度的聯合 loss：

```math
L_{\text{MTP}} = \frac{\lambda}{D} \sum_{k=1}^{D} L_k
```

`lambda` 是一個小的加權係數。DeepSeek-V3 在訓練的前 10% 使用 0.3，之後使用 0.1。總訓練 loss 是 `L_main + L_MTP`。

### 為什麼是 sequential，而不是 parallel (Why sequential, not parallel)

Gloeckle 最初的平行 MTP 有 D 個輸出 head，每個直接套用在 `h_i^(0)` 上。每個 head 從相同的 backbone hidden state 預測 `t_{i+k}`。這樣訓練沒問題，但預測之間並不互相以對方為條件。你無法用 `head_1` 的輸出去幫助 `head_2`，因為這些 head 是平行觸發的。

DeepSeek-V3 的 sequential 設計，從 `h_i^(k-1)` 加上實際的下一個 token 的 embedding `E(t_{i+k})` 來建立 `h_i^(k)`。這保留了因果鏈：為了預測 `t_{i+k+1}`，深度 `k` 的模組看得到 `t_{i+k}` 是什麼。這在結構上與自迴歸 decoder 消耗自己輸出的方式完全相同，這使得 MTP 模組可以直接作為 speculative-decoding 的 drafter 使用。

推論時：把 `h_i^(k-1)` 和草擬的 `t_{i+k}` 餵進模組 `k`，得到對 `t_{i+k+1}` 的預測。重複。這正是 EAGLE 風格的 draft，把訓練好的 MTP 模組當作 draft 網路。DeepSeek-V3 回報第一個 MTP 模組有 80% 以上的接受率，以及約 1.8 倍的加速。

### 參數計算 (Parameter accounting)

對於 hidden 為 `h`、vocabulary 為 `V` 的模型：

- 主模型：數十億參數，加上一個大小為 `V * h` 的輸出 head。
- 共用的輸出 head：重複使用主模型的 head。沒有額外參數。
- 共用的 embedding：重複使用主模型的 embedding。沒有額外參數。
- 每個 MTP 模組：
  - 投影 `M_k`：`(2h) * h = 2h^2`。
  - Transformer block `T_k`：attention（MHA 為 `4h^2`）加上 MLP（SwiGLU 比例 8/3，通常為 `8h^2`）。每個 block 約 `12h^2`。

每個模組的額外參數總計：`~14h^2`。對 DeepSeek-V3 的 `h = 7168`、D = 1 個模組：帳面上 `~14 * 7168^2 = ~720M` 個參數。DeepSeek-V3 回報的是 14B，差異主要是因為 MTP 模組中的 expert 層也是 MoE。

### Speculative-decoding 的回報 (The speculative-decoding payoff)

Pre-training 期間，MTP 模組會讓訓練慢約 10%（更多的 forward 運算、額外的 loss）。回報有兩方面：

1. 更密集的訓練訊號。每個 hidden state 看到 D+1 個監督目標。在 MMLU、GSM8K、MATH、HumanEval 上量測到的效果：DeepSeek-V3 的 ablation 中，有一致的數個百分點的進步。

2. 推論時免費的 speculative decoding draft。MTP 模組本來就被訓練去預測接下來的幾個 token。重新用作 draft 網路時，接受率達 80% 以上。在這個水準下，N=3 或 N=5 的 spec decoding 能提供 1.8 倍的吞吐量。訓練時多付出的 10% 成本，在你第一次執行推論時就回本了。

### 與 EAGLE 的關係 (Relation to EAGLE)

EAGLE 在 pre-training 之後另外訓練一個小型 draft model。MTP 則把 draft 烘焙進 pre-training。兩種做法收斂到相近的接受率，但走的管線不同：

| 面向 | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| 何時訓練 | Pre-training 之後 | Pre-training 期間 |
| 與既有權重向後相容 | 是 | 否（需要重新訓練） |
| Draft 參數 | 1-2 層 transformer | 1 個 transformer block + 投影 |
| 接受率 | 0.88-0.92 | 深度 1 時 0.80+ |
| 加速之外的好處 | 只有 speculative decoding | 更密集的訓練訊號 + 加速 |

```figure
multi-token-predict
```

## 動手做 (Build It)

`code/main.py` 端到端打造單一個 MTP 模組：共用 embedding、投影、transformer block、共用輸出 head。接著它在一條簡短的合成序列上計算每個深度的 cross-entropy loss，並印出各元件的參數量。只有 32 個 token 的玩具 vocabulary，讓數字保持易讀。

### 步驟 1：共用的 embedding 表 (Step 1: shared embedding table)

單一個 `vocab_size x hidden` 的表，被主模型以及每個深度的每個 MTP 模組共用。不是第二份副本，而是字面上相同的 tensor。

### 步驟 2：每個深度的組合 (Step 2: the per-depth combination)

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

真正的 DeepSeek-V3 把兩個經過 RMSNorm 的向量串接成 `[2h]`，並用一個 `h x 2h` 矩陣投影。為了標準函式庫的簡潔，玩具版使用向量相加。

### 步驟 3：深度 k 的 transformer block (Step 3: the transformer block at depth k)

Self-attention 加上 MLP。在玩具版中，一層 linear attention block 與一個 SwiGLU MLP，在不使用 numpy 的情況下保持結構清晰可見。

### 步驟 4：共用的輸出 head (Step 4: the shared output head)

重複使用主模型的輸出投影。得到 vocabulary 上的 logit。

### 步驟 5：每個深度的 loss (Step 5: per-depth loss)

softmax(logits) 對偏移 `k+1` 處 ground-truth token 的 cross-entropy。以 `lambda / D` 的縮放係數跨深度彙整。

### 步驟 6：參數計算 (Step 6: parameter accounting)

印出總參數量、共用（embedding、head）的數量，以及每個模組額外的數量。展示 MTP 額外參數與主模型大小的比值。

## 實際使用 (Use It)

MTP 已整合在 DeepSeek-V3（2024 年 12 月）與 DeepSeek-R1 系列中。推論時：

- DeepSeek 自己的服務堆疊，開箱即用地把 MTP 模組當作 speculative decoder 使用。
- 截至 2026 年 4 月，vLLM 與 SGLang 已有 DeepSeek-V3 MTP 的整合路徑。
- AMD 的 ROCm SGLang 教學展示了特定的 MTP speculative-decoding 設定，在 V3 checkpoint 上實測加速 1.8 倍。

何時在新的 pre-training 中使用 MTP：

- 你掌控完整的 pre-training 管線，並想累積更密集的訓練訊號。
- 你知道會大規模服務這個模型，並想免費得到 speculative decoding。
- 你的 hidden size 至少為 4096。在 1B 規模時，開銷造成的傷害大於增益帶來的幫助。

何時不該用：

- Fine-tune 既有的 pre-trained dense 模型。MTP 模組沒有被訓練過。
- 你想要乾淨 baseline 來比較的研究模型。MTP 會改變架構。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-mtp-planner.md`。給定一份 pre-training 的規格（模型大小、資料、運算），它會回傳整合 MTP 的計畫：深度數 D、`lambda` 排程、記憶體開銷，以及推論時 speculative-decoding 的接線。

## 練習 (Exercises)

1. 執行 `code/main.py`。展示隨著合成訊號增強，每個深度的 loss 單調下降。把合成資料改成固定模式，並驗證深度 1 與深度 2 的 loss 都會收斂。

2. 計算一個 dense 70B 模型（hidden 8192、80 層）加上 D=1 個 MTP 模組的參數開銷。與 DeepSeek-V3 回報的 14B 開銷比較。解釋為什麼 DeepSeek 的數字較高：MTP 的 transformer block 繼承了相同的 MoE 結構，墊高了每個模組的參數量。

3. 在玩具版中實作 D=2：加入第二個 MTP 模組，它接收 $h^{(1)}$ 並預測 `t_{i+3}`。驗證聯合 loss 與參數計算符合 DeepSeek 論文的公式 19-21。

4. 把玩具版切換成平行 MTP（Gloeckle 風格）：在主 hidden state 之上加 D 個輸出 head，各預測不同的位移。量測在相同合成訊號上，每個深度的 loss 與 sequential 版本相比如何。對 k > 1，sequential 版本應該產生較低的深度 k loss，因為它以中間的預測為條件。

5. 把訓練好的 MTP 模組當作 EAGLE 風格的 draft：在推論時呼叫模組 k 來提出 `t_{i+k+1}`。在保留序列上，量測這些 draft token 與主模型預測的接受率。如果你在玩具版上達到 50% 以上，就重現了 MTP 可作為 draft 的經驗性質。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| MTP module | 「額外的 loss block」 | 一個小型 transformer block 加上投影，預測比主模型再往前 `k` 個位置的 token |
| Prediction depth | 「哪個位移」 | 整數 `k`，使得模組 `k` 從到位置 `i` 為止的前綴預測 `t_{i+k+1}`（主模型是深度 0，預測 `t_{i+1}`） |
| Parallel MTP | 「Gloeckle 風格」 | 在同一個 backbone hidden state 上的 D 個獨立 head，沒有條件鏈 |
| Sequential MTP | 「DeepSeek-V3 風格」 | 每個模組以前一深度的 hidden state 加上下一個 token 的 embedding 為條件，保留因果鏈 |
| Shared output head | 「重複使用主 head」 | MTP 模組呼叫主模型的 LM head，而不是獨立的輸出投影 |
| Shared embedding | 「重複使用主表」 | 同一個 vocabulary embedding 表到處使用，沒有重複的參數 |
| Projection matrix $M_k$ | 「結合 hidden + 下一個 token」 | 一個 `h x 2h` 的線性層，把前一個 hidden state 與目標 token 的 embedding 折疊成下一個深度的輸入 |
| Joint loss $L_{\text{MTP}}$ | 「平均的額外 loss」 | 各深度 cross-entropy loss 的算術平均，以 `lambda` 縮放 |
| Acceptance rate at depth 1 | 「MTP draft 猜對的頻率」 | D=1 的 MTP 模組的 top-1 預測，等於主模型 top-1 預測的比率，DeepSeek-V3 上達 80% 以上 |
| Lambda weighting | 「額外 loss 的重要性」 | 每個深度的縮放係數，DeepSeek-V3 在訓練初期為 0.3，之後為 0.1 |

## 延伸閱讀 (Further Reading)

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437)：完整的 sequential MTP 描述（第 2.2 節），包含聯合 loss 的公式以及推論時 1.8 倍的加速
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737)：DeepSeek 的設計所改進的平行 MTP baseline
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3)：總共 685B（671B 主模型 + 14B MTP）、部署說明
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)：MTP 所處的 speculative-decoding 框架
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840)：EAGLE 2025 年的 draft 架構，是 MTP 所競爭的對應做法
