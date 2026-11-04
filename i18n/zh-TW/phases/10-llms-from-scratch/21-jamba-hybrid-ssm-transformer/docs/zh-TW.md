# Jamba：混合 SSM-Transformer (Jamba — Hybrid SSM-Transformer)

> State space model (SSM) 與 transformer 想要的東西不同。Transformer 以二次方的成本，透過 attention 換取品質。SSM 透過 recurrence 換取線性時間的推論與固定的記憶體，但品質略遜。AI21 的 Jamba（2024 年 3 月）與 Jamba 1.5（2024 年 8 月）把兩者放進同一個模型：每 7 層 Mamba 搭配 1 層 Transformer，每隔一個 block 使用 MoE，並有可放進單張 80GB GPU 的 256k context window。Mamba-3 (ICLR 2026) 則以複數值的 state space 與 MIMO 投影強化了 SSM 這一側。這堂課端到端閱讀這兩個架構，並解釋為什麼在純 SSM 與純 Transformer 的長 context 嘗試都沒能延續下來的三年間，混合配方卻存活了下來。

**Type:** Learn
**Languages:** Python (stdlib, layer-mix calculator)
**Prerequisites:** Phase 10 · 14 (open-model architectures), Phase 10 · 17 (native sparse attention)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋 Jamba block 中的三個基本元件（Transformer 層、Mamba 層、MoE），以及 1:7:偶數 的交錯配方。
- 說明 SSM 的 recurrence 在高層次上長什麼樣子，以及為什麼它能做到固定記憶體的推論。
- 計算 Jamba 模型在 256k context 下的 KV cache 佔用量，並與純 Transformer 模型所需的相比。
- 說出 Mamba-3 的三項創新（exponential-trapezoidal discretization、複數值 state 更新、MIMO），以及各自針對的問題。

## 問題 (The Problem)

Attention 對序列長度是二次方。State space model 是線性的。這個差異會不斷放大：在 256k 個 token 時，Transformer 每個 head 的 attention map 有 650 億個項目；而 SSM 的 recurrent state 不論序列多長，大小都是固定的。

純 SSM 模型（Mamba、Mamba-2）在小規模下追平 Transformer 的 perplexity，但在狀態追蹤任務上落後，並且在某些類別的 in-context retrieval 上失敗。直覺是：SSM 把歷史壓縮成固定大小的 state，當歷史很長時，資訊就會流失。Attention 能精確地記住一切，但要付出二次方的成本。

顯而易見的修正：兩者並用。在需要精確回想的地方放 Transformer 層。其他地方用 SSM 層。調整比例。Jamba 是第一個大規模出貨這種混合配方的正式等級模型（總共 52B、12B 活躍、256k context、單張 80GB GPU）。Jamba 1.5 將這個家族擴展到總共 398B / 94B 活躍。Mamba-3 (ICLR 2026) 則是目前最好的純 SSM baseline，混合模型可以圍繞它重建。

這堂課閱讀這三篇論文，並產生「挑選正確比例」的心智模型。

## 核心概念 (The Concept)

### 一頁讀懂 SSM (An SSM in one page)

State space model 透過固定大小的 state `h` 處理序列 `x_1, ..., x_N`：

```math
\begin{aligned}
h_t &= A h_{t-1} + B x_t \\
y_t &= C h_t
\end{aligned}
```

每一步，state 透過線性動態 `A` 演化，接收輸入 `B x_t`，並輸出 `C h_t`。`A, B, C` 可以被學習。注意這個關鍵性質：計算 `y_t` 只需要 `h_{t-1}` 和 `x_t`，不需要任何更早的 `x`。記憶體是固定的。推論每個 token 是 $O(1)$。

建模品質的關鍵在於 `A` 的結構。S4 (Gu 2021) 使用高度結構化的矩陣，在訓練期間可以被有效率地當作長卷積來計算。Mamba (Gu, Dao 2023) 把固定的 `A, B, C` 換成與資料相關的版本（這就是「selective」的部分）。Mamba-2 (2024) 進一步簡化了結構。Mamba-3 (2026) 則在特定地方重新加回複雜度。

關鍵性質：對於 decoder LLM，SSM 層可以直接取代 attention 層，用固定大小的每層 state，取代不斷成長的 KV cache。

### Jamba block (The Jamba block)

Jamba block 依兩個數字交錯排列各層：

- `l`：attention 對 Mamba 的比例。Jamba 使用 `l = 8`，意思是每 7 層 Mamba 有 1 層 Transformer（每組 7 層 Mamba + 1 層 Attention = 8 層）。
- `e`：MoE 頻率。Jamba 使用 `e = 2`，意思是每隔一層套用 MoE。

一個 block 內的層序列：

```
M  M  M  M  M  M  M  A    (7 Mamba + 1 Attention)
|  M  |  M  |  M  |  M    (where | marks MoE applied)
```

每個 Jamba block 有 8 層。深度為 4 個 block（共 32 層）時，你會得到 28 層 Mamba 與 4 層 Attention。其中 16 層使用 MoE。

### 為什麼是 1:7 的比例 (Why the 1:7 ratio)

AI21 做了 ablation：在他們的長 context eval 上，什麼樣的 attention 對 Mamba 比例，能同時得到最佳的每參數 perplexity 與 in-context 回想？

- Attention 太多（1:1）：品質上升，但記憶體與速度變差。
- Attention 太少（1:15）：記憶體很棒，但 in-context retrieval 失敗。
- 最佳點：1:7 或 1:8。

直覺是：Transformer 層負責精確回想與狀態追蹤。Mamba 層負責便宜的大宗處理。

### 位置編碼 (Positional encoding)

Mamba 層本身就具備位置感知（透過 recurrence）。最初基於 Mamba 的混合模型中，attention 層沒有使用 RoPE，位置資訊由 SSM 層提供。Jamba 1.5 在 attention 層加上 RoPE，以提升更長 context 的泛化，這是根據實證的長 context 評估所做的事後改良。

### 記憶體預算 (The memory budget)

以 Jamba-1 的形狀（32 層：28 層 Mamba + 4 層 Attention，hidden 4096，32 個 attention head）為例：

- KV cache（僅 attention 層）：BF16 在 256k 下 `2 * 4 * 32 * 128 * 256k * 2 = 16 GB`。只有 4 層 attention 有貢獻。實際釋出的 Jamba 在這些層使用 8 個 KV head 的 GQA，使這個數字降到 4 GB，這就是 Jamba 論文中的數字。
- SSM state：每個 token 前綴 `28 * hidden * state_size`，但這是每層固定大小，不隨序列長度成長。典型的 Mamba state 是每個特徵 16，hidden 4096：總共 `28 * 4096 * 16 * 2 = 3.7 MB`。

對比 32 層、相同 hidden、32 個 head 的完整 MHA 純 Transformer：BF16 在 256k 下 `2 * 32 * 32 * 128 * 256k * 2 = 128 GB`。KV cache 減少 8 倍。即使對比大多數 2024 年模型使用的 GQA(8) baseline（`2 * 32 * 8 * 128 * 256k * 2 = 32 GB`），擁有 32 個 KV head、16 GB 的 1:7 混合模型仍然小 2 倍。實際釋出的 Jamba 是 4 GB，小 8 倍。

這就是 AI21 所說的「單張 80GB GPU 上的 256k context」。完整 MHA 純 Transformer 的 KV cache 放不下；即使是 GQA baseline，也沒有空間留給權重與 activation；而 Jamba 的可以。

### Mamba-3：2026 年的純 SSM baseline (Mamba-3: the pure-SSM baseline in 2026)

Mamba-3 (ICLR 2026, arXiv:2603.15569) 在純 SSM 這一側引入三項創新：

1. **Exponential-trapezoidal discretization。** 以更具表達力的 recurrence，取代 Mamba-2 中的 Euler 方法 discretization。在核心 recurrence 內，對 state 輸入套用類似卷積的運算，而不是作為對 `x_t` 的外部卷積。

2. **複數值的 state 更新。** 先前的 Mamba 把 state 矩陣從複數（S4）簡化為實數對角（Mamba），再到縮放的單位矩陣（Mamba-2）。Mamba-3 重新加入複數值，相當於對 state 套用與資料相關的 rotary embedding。這恢復了先前實數值簡化所損失的狀態追蹤能力。

3. **Multi-input multi-output (MIMO) 投影。** 不使用每個特徵的純量投影，而是使用矩陣值的投影。在不增加 decode 延遲的情況下，提升建模能力與推論時的硬體使用率。

在 1.5B 參數時，Mamba-3 的平均下游準確度比 Gated DeltaNet 高 0.6 分；MIMO 變體再多加 1.2 分，總共進步 1.8 分。在相同的 state 大小下，Mamba-3 用一半的 state 就能追平 Mamba-2。

Mamba-3 還沒有在大規模的正式混合模型中出貨，但它顯然是下一個 Jamba 等級模型 SSM 這一側的候選。

### 何時該選混合模型 (When to reach for a hybrid)

混合模型勝出的情況：

- Context 長到純 Transformer 的 KV cache 變得痛苦（64k 以上）。
- 任務混合了短距離結構（適合 SSM）與長距離回想（需要 Transformer）。
- 你想部署在單 GPU 記憶體預算下，而光是 Transformer 的 KV cache 就放不下。

混合模型落敗的情況：

- Context 短（小於 16k）。SSM 的開銷被浪費；純 Transformer 就很好。
- 任務需要處處到處的 attention（深度推理、多文件交叉參照）。混合模型中 attention 層的稀疏性會造成傷害。
- 你正在擴展到兆級參數的 frontier 模型。純 Transformer + MLA + MoE（DeepSeek-V3 風格）目前在能力競賽中領先。

### 競爭格局 (The competitive landscape)

| 模型 | 家族 | 規模 | 獨特主張 |
|-------|--------|------|-------------|
| Mamba-2 | 純 SSM | 3B | 線性時間、固定記憶體 |
| Jamba | 混合 | 52B/12B | 80GB 上 256k |
| Jamba 1.5 Large | 混合 | 398B/94B | 企業級長 context |
| Mamba-3 | 純 SSM | 1.5B（論文） | 恢復狀態追蹤 |
| DeepSeek-V3 | 純 Transformer + MoE | 671B/37B | frontier 能力 |

2026 年的格局：純 Transformer MoE 主宰 frontier，但混合模型佔據 256k 以上 context 的利基。Mamba-3 在狀態追蹤上的進展，可能推動下一代的混合比例更低（更多 SSM、更少 attention）。

```figure
swiglu-ffn
```

## 實際使用 (Use It)

`code/main.py` 是混合架構的記憶體計算器。給定 SSM-Transformer 的比例，以及 hidden-size / 層數設定，它會計算：

- 目標 context 下的 KV cache。
- SSM state 記憶體。
- 一系列模型形狀在 context N 下的總記憶體。

這個計算器支援：

- 純 Transformer baseline（KV cache 隨 N 成長）。
- Jamba 風格的 1:7 混合。
- 純 SSM（完全沒有 KV cache）。

這些數字，對已發表的形狀直接取自 Jamba-1 與 Jamba-1.5 論文，對假設的變體則是外推而來。

真實部署的整合考量：

- 大多數正式的推論伺服器（vLLM、SGLang）支援 Jamba 與 Mamba。請檢查特定版本。
- 在 256k context 下，Jamba 的記憶體優勢展現在並行請求的吞吐量上。在相同的 VRAM 上，你能塞進比 Transformer 序列更多的 Jamba 序列。
- 作為獨立模型的 Mamba-3 還沒有在正式環境出貨，目前是 1.5B 的研究預覽。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-hybrid-picker.md`。給定工作負載規格（context 長度分布、任務組合、記憶體預算），它會在純 Transformer、Jamba 風格混合模型與純 SSM 之間做推薦，並對記憶體與品質的取捨給出明確的理由。

## 練習 (Exercises)

1. 執行 `code/main.py`，計算 32 層純 Transformer（hidden 4096、32 個 head）與相同形狀的 Jamba-1 混合模型在 256k context 下的 KV cache。驗證 AI21 論文所主張的約 8 倍記憶體縮減。

2. 修改計算器，模擬 1:3 混合（4 層 Mamba : 1 層 Attention）與 1:15 混合（14 層 Mamba : 1 層 Attention）。畫出 KV cache 對比例的圖。在什麼比例下，KV cache 等於 SSM state 記憶體？

3. 閱讀 Jamba 論文 (arXiv:2403.19887) 的第 3 節。解釋為什麼 AI21 使用 Mamba-1 而不是 Mamba-2，儘管 Mamba-2 更快。提示：混合模型的 ablation 章節有記載。

4. 計算 Jamba 1.5 Large（總共 398B、94B 活躍）中每隔一層使用 MoE 的參數開銷。把活躍比例與 DeepSeek-V3 (37B/671B) 比較，並解釋為什麼 Jamba 的架構把活躍比例推得更高。

5. 閱讀 Mamba-3 論文 (arXiv:2603.15569) 的第 3 節。用三句話解釋，為什麼複數值的 state 更新等價於與資料相關的 rotary embedding。把答案與 Phase 7 · Lesson 04 的 RoPE 推導連結起來。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| State space model (SSM) | 「帶固定 state 的 recurrence」 | 具有學習式 recurrence `h_t = A h_{t-1} + B x_t` 的層，每個 token 的記憶體固定 |
| Selective SSM | 「Mamba 的技巧」 | 與資料相關的 A、B、C 參數，讓模型在線性時間內具備類似閘控的選擇性 |
| Attention-to-Mamba ratio | 「有多少 attention 層」 | 在 Jamba 中，`l = 8` 代表每 7 層 Mamba 有 1 層 attention |
| Jamba block | 「8 層一組」 | 一層 attention + 七層 Mamba，並在交替位置使用 MoE |
| SSM state | 「hidden buffer」 | 取代 Mamba 層 KV cache 的、固定大小的每層 state |
| 256k context | 「Jamba 的旗艦數字」 | Jamba-1 能放進單張 80GB GPU 的序列長度，純 Transformer 在該大小下做不到 |
| Mamba-3 | 「2026 年的純 SSM」 | 目前最好的純 SSM 架構，具備複數 state + MIMO，是混合模型所圍繞重建的 baseline |
| MIMO | 「Multi-input multi-output」 | Mamba-3 的創新，使用矩陣值投影而不是每個特徵的純量 |
| Exponential-trapezoidal discretization | 「Mamba-3 的 recurrence」 | 更具表達力的 recurrence，涵蓋 Mamba-2 的 Euler 方法 discretization |
| Hybrid architecture | 「混合 attention 與 SSM」 | 任何交錯排列 Transformer 與 SSM 層的模型，Jamba 是正式環境的原型 |

## 延伸閱讀 (Further Reading)

- [Lieber et al. — Jamba: A Hybrid Transformer-Mamba Language Model (arXiv:2403.19887)](https://arxiv.org/abs/2403.19887)：最初的 Jamba 論文、比例 ablation、256k context 的主張
- [AI21 — Jamba 1.5: Hybrid Transformer-Mamba at Scale (arXiv:2408.12570)](https://arxiv.org/abs/2408.12570)：擴大規模的家族，398B/94B 與 12B/52B 的公開釋出
- [Gu, Dao — Mamba: Linear-Time Sequence Modeling with Selective State Spaces (arXiv:2312.00752)](https://arxiv.org/abs/2312.00752)：Jamba 所建立的 selective SSM 論文
- [Dao, Gu — Mamba-2 (arXiv:2405.21060)](https://arxiv.org/abs/2405.21060)：簡化的結構化 state-space 後繼者
- [Lahoti et al. — Mamba-3 (arXiv:2603.15569, ICLR 2026)](https://arxiv.org/abs/2603.15569)：複數值 state、MIMO，2026 年的純 SSM 前沿
- [Gu et al. — Efficiently Modeling Long Sequences with Structured State Spaces (arXiv:2111.00396)](https://arxiv.org/abs/2111.00396)：S4 論文，是 LLM 領域 SSM 系譜的起點
