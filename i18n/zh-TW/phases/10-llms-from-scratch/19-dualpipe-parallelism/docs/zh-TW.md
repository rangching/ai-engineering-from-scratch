# DualPipe Parallelism (DualPipe Parallelism)

> DeepSeek-V3 在 2,048 張 H800 GPU 上訓練，MoE 的 expert 分散在各個節點。跨節點的 expert all-to-all 通訊，每花 1 個 GPU 小時做運算，就要花 1 個 GPU 小時做通訊。GPU 有一半的時間在閒置。DualPipe (DeepSeek, 2024 年 12 月) 是一種雙向 pipeline，讓 forward 與 backward 的運算，與它們所觸發的 all-to-all 通訊重疊。Bubble 下降，吞吐量上升，而且一旦 Expert Parallelism 本來就把 expert 分散到各 rank，保留兩份模型參數副本（這就是名稱中「dual」的由來）的成本就很低。這堂課是一份 Learn 類型的導覽，說明 DualPipe 實際做了什麼，以及為什麼 Sea AI Lab 的 DualPipeV 改良版，以稍微更緊的 bubble 為代價，去掉了 2 倍的參數成本。

**Type:** Learn
**Languages:** Python (stdlib, schedule simulator)
**Prerequisites:** Phase 10 · 05 (distributed training, FSDP, DeepSpeed), Phase 10 · 14 (open-model architectures and MoE)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 DualPipe forward-backward chunk 的四個組成，以及為什麼每一個都有自己的重疊視窗。
- 解釋大規模下的 pipeline bubble 問題，以及「無 bubble」在實務上與在行銷用語上分別是什麼意思。
- 手動追蹤 8 個 PP rank、16 個 micro-batch 的 DualPipe 排程，並確認 forward 與反向的串流會填滿彼此的閒置時段。
- 陳述 DualPipeV (Sea AI Lab, 2025) 所做的取捨：在 Expert Parallelism 沒有啟用時，以稍大的 bubble 為代價，去掉 2 倍的參數複製。

## 問題 (The Problem)

在 2k 張 H800 GPU 上訓練 671B 的 MoE 模型，會遇到三個互相加成的瓶頸：

1. **記憶體壓力。** 每張 GPU 持有模型的一個切片。序列長度 8k、61 層、128 個 head 的 activation 記憶體非常龐大。
2. **Pipeline bubble。** 傳統的 pipeline parallelism (GPipe、1F1B) 會讓 GPU 在等待它那一階段的輸入或 gradient 時閒置。在 8 個階段時，即使採用 1F1B 排程，大約仍有 12% 的 GPU 時間會是 bubble。
3. **跨節點 all-to-all。** 搭配 expert parallelism 的 MoE，把 expert 分散到各節點。每次 forward pass 都會觸發一次 all-to-all 把 token 分派到它們的 expert，以及另一次把結果合併回來。在 2k 張 GPU 時，這很容易變成 1:1 的運算對通訊比。

這三者各有獨立的解法：用 gradient checkpointing 處理記憶體、用 Zero Bubble (Sea AI Lab, 2023) 處理 pipeline bubble、用 expert-parallel 通訊 kernel 處理 all-to-all。DualPipe 做的，是讓它們一起運作。這個排程在單一個 forward-backward chunk 內重疊運算與通訊，從 pipeline 的兩端同時注入 micro-batch，並利用得到的排程，把 all-to-all 藏在運算視窗裡。

回報的結果：pipeline bubble 幾乎被消除，在 DeepSeek-V3 14.8T token 的訓練中，GPU 使用率超過 95%。

## 核心概念 (The Concept)

### Pipeline parallelism 複習 (Pipeline parallelism refresher)

把 N 層的模型分到 P 個裝置上。裝置 `i` 持有第 `i * N/P .. (i+1) * N/P - 1` 層。一個 micro-batch 從裝置 0 到 P-1 往前流動，然後從 P-1 回到 0 往後流動。每個裝置只能在前一個裝置送出輸出之後，才能開始它的 forward 階段，並且只能在下游裝置送出上游 gradient 之後，才能開始 backward。

GPipe (Huang et al., 2019) 一次排程一個 micro-batch，這浪費了大部分 GPU 時間。1F1B (Narayanan et al., 2021) 對多個 micro-batch 交錯進行 forward 與 backward。Zero Bubble (Qi et al., 2023) 把 backward 拆成兩個部分：對輸入的 backward (B) 與對權重的 backward (W)，並排程它們去填補 bubble。在 Zero Bubble 之後，pipeline 幾乎是緊密的。

DualPipe 是下一步。它在此之上加了兩個想法：

### 想法 1：chunk 分解 (Idea 1: chunk decomposition)

每個 forward chunk 被切成四個組成：

- **Attention。** Q/K/V 投影、attention、輸出投影。
- **All-to-all dispatch。** 把 token 送到它們的 expert 的跨節點通訊。
- **MLP。** MoE expert 的運算。
- **All-to-all combine。** 把 expert 輸出帶回來的跨節點通訊。

Backward chunk 為每一個加上 gradient 版本。DualPipe 排程它們，使得 all-to-all dispatch 與下一個 chunk 的 attention 運算平行發生，而 all-to-all combine 與後續 chunk 的 MLP 運算平行發生。

### 想法 2：雙向排程 (Idea 2: bidirectional scheduling)

大多數 pipeline 排程從階段 0 注入 micro-batch，朝階段 P-1 流動。DualPipe 從*兩端*注入 micro-batch。階段 0 看到從那裡出發的 forward micro-batch；階段 P-1 也看到從那裡出發的 forward micro-batch。兩條串流在中間相遇。

要讓這個運作，裝置 `i` 必須同時持有 pipeline 前段的層 `i` 以及 pipeline 後段的層 `P - 1 - i`。這就是 DualPipe 中「dual」的部分：每個裝置為它需要服務的模型層保留兩份副本（每個方向一份）。在 DeepSeek-V3 的規模下，這是 2 倍的參數複製成本。這是負擔得起的，因為 Expert Parallelism 已經把 MoE 的 expert 攤得很薄，把非 expert 的層複製兩份只是小事。

關鍵在於，一個方向的 forward 串流與另一個方向的 backward 串流，剛好在單向排程中會出現 bubble 的地方重疊。Bubble 因而消失。

### 手動追蹤的排程 (A hand-traced schedule)

考慮 P = 4 個 rank、8 個 micro-batch，分成 4 個 forward / 4 個反向。時間由左向右；每一列是一個裝置 rank。

```
           Time →
rank 0:  F1 F2 F3 F4  F5R F6R F7R F8R  B1 B2 B3 B4  ...
rank 1:     F1 F2 F3  F4/F5R F6R F7R   B1 B2 ...
rank 2:        F1 F2  F3/F5R F4/F6R    B1 ...
rank 3:           F1  F2/F5R F3/F6R    ...
```

閱讀「F4/F5R」的記號：rank 1 正在同一個時段內執行 micro-batch 4 的 forward（在 pipeline 中由左到右）以及 micro-batch 5 的 forward（由右到左）。這就是「雙向」在運作上的意義。

在 rank 2，交叉的串流較早重疊，在 rank 0 和 P-1 則最晚重疊。在排程穩定的中段階段，每個 rank 都在執行 X 方向的 forward，與 Y 方向的 backward 重疊。運算保持忙碌。Forward pass 的 all-to-all dispatch 藏在 backward 運算裡。All-to-all combine 藏在 forward 運算裡。Bubble 被擠掉了。

### Bubble 的計算 (Bubble accounting)

標準 1F1B pipeline 的 bubble（每個 rank 浪費的時間）：

```
bubble_1F1B = (P - 1) * forward_chunk_time
```

Zero Bubble 的改良把它降低，但不是降到零。DualPipe 在穩定階段，如果 micro-batch 數量能被 pipeline 深度的 2 倍整除，bubble 為零。在穩定階段之外（暖機與冷卻），有一些 bubble，但它不會隨 micro-batch 數量成長，這是論文強調的關鍵性質。

用行銷的說法：「無 bubble」。用技術的說法：bubble 不會隨 micro-batch 數量成長。Sea AI Lab 的後續分析 (DualPipeV / Cut-in-half) 顯示，只有在 Expert Parallelism 不是瓶頸時，才有完整的零 bubble；在由 EP 驅動的 all-to-all 下，總是存在某種排程上的妥協。

### DualPipeV：改良版 (DualPipeV — the refinement)

Sea AI Lab (2025) 觀察到，當 EP 通訊的重疊不是重點時，2 倍的參數複製是浪費的。他們的 DualPipeV 排程把雙向注入折疊成「V 形」排程，只用單一份參數副本運作。Bubble 比 DualPipe 的稍大，但記憶體節省相當可觀。DeepSeek 在其開源的 DualPipe 實作中，把 DualPipeV 納入為 EP 關閉模式。

取捨如下：

| 特性 | DualPipe | DualPipeV | 1F1B | Zero Bubble |
|---------|---------|-----------|------|------------|
| 每個裝置的參數副本數 | 2 | 1 | 1 | 1 |
| Bubble 對 micro-batch 數 | 常數 | 小幅成長 | 成長 | 成長 |
| 運算與通訊的重疊 | 完整 | 部分 | 極少 | 部分 |
| 何時使用 | EP 繁重的 MoE | dense 或 EP 輕量 | baseline | 任何 pipeline |

### 對 14.8T token 的訓練意味著什麼 (What it means for a 14.8T-token run)

DeepSeek-V3 的 pre-training 在 2,048 張 H800 GPU 上消耗了 14.8T 個 token，大約花了 280 萬 GPU 小時。若採用天真的 1F1B，他們會有 12-15% 浪費在 pipeline bubble 上，也就是 34-42 萬 GPU 小時，足以訓練一個完整的 70B 模型。DualPipe 回收了其中大部分。沒有內部紀錄很難直接量化這項貢獻，但論文中的主張是整個訓練平均的 GPU 使用率超過 95%。

對於較小的訓練（少於 1k 張 GPU），DualPipe 是殺雞用牛刀：pipeline bubble 相對於總成本較小，而 dense 模型訓練很少碰到 all-to-all 瓶頸。對於數千張 GPU 規模的 frontier MoE 訓練，它實質上是必要的。

### 它在堆疊中的位置 (Where it sits in the stack)

- 與 **FSDP** (Phase 10 · 05) 互補。FSDP 把模型參數分片到各 rank；DualPipe 排程各 rank 的運算。兩者可以結合。
- 相容於 **ZeRO-3** 的 gradient 分片。兩份副本複製的記帳，需要與 ZeRO 分片後的 gradient 配合。
- 需要針對特定叢集拓撲調校的**自訂 all-to-all kernel**。DeepSeek 的開源 kernel 是參考實作。

```figure
expert-capacity
```

## 實際使用 (Use It)

`code/main.py` 是一個 pipeline 排程模擬器。它接受 `(P, n_micro_batches, schedule)`，並印出 1F1B、Zero Bubble、DualPipe 與 DualPipeV 各自的穩定階段使用率。它是教學工具，數字符合論文中的定性主張，但並非對正式環境實測加速的主張。

模擬器的價值：用不同的 P 與 micro-batch 數量執行，看看 1F1B 的 bubble 比例如何成長，而 DualPipe 不會。

真實訓練的整合考量：

- 選擇一個能整除你 micro-batch 數量的 pipeline-parallel 深度。
- 確認你的 expert-parallel mesh 支援雙向 all-to-all。DeepSeek 的 kernel 是參考。
- 預期第一次會花上一週的時間在排程本身的除錯上。記帳很繁瑣。
- 監控每個 rank 的 GPU 使用率，而不只是總和。DualPipe 的好處來自收緊落後的 rank。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-dualpipe-planner.md`。給定一個訓練叢集規格（GPU 數量、拓撲、互連、模型形狀），它會推薦 pipeline parallelism 策略、要使用的排程演算法，以及在目標規模下預期的 bubble 比例。

## 練習 (Exercises)

1. 對 `(P=8, micro_batches=16, schedule=dualpipe)` 與 `(P=8, micro_batches=16, schedule=1f1b)` 執行 `code/main.py`。計算 GPU 使用率的差異，並將其表示為每百萬個訓練 token 回收的 GPU 小時數。

2. 手動畫出 `(P=4, micro_batches=8, schedule=dualpipe)` 的排程表。在每個時段標上 micro-batch ID 與方向。找出第一個沒有 bubble 的時段。

3. 閱讀 DeepSeek-V3 技術報告 (arXiv:2412.19437) 的圖 5。找出 DualPipe forward chunk 內 all-to-all dispatch 的重疊視窗。解釋運算排程如何把它藏起來。

4. 計算 DualPipe 對於 P=8 個 pipeline 階段的 70B dense 模型，以及 P=16 個 pipeline 階段的 671B MoE 模型的 2 倍參數開銷。說明為什麼 MoE 情況的開銷在比例上較小（大部分參數是 expert，分片在很大的 EP group 中）。

5. 比較 DualPipe 與 Chimera（2021 年一個競爭的雙向排程器）。以論文第 3.4 節為參考，找出 DualPipe 新增、而 Chimera 沒有的兩個特定性質。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Pipeline bubble | 「每個 rank 的閒置時間」 | 因為某個 pipeline 階段在等待它的輸入或 gradient 而浪費的 GPU 週期 |
| 1F1B | 「預設的 pipeline 排程」 | 一個 forward / 一個 backward 交錯排程，是 DualPipe 所超越的 baseline |
| Zero Bubble | 「Sea AI Lab 2023」 | 把 backward 拆成 B（輸入 gradient）與 W（權重 gradient），幾乎完全收緊 pipeline |
| DualPipe | 「DeepSeek-V3 排程」 | 雙向 pipeline + 運算-通訊重疊，bubble 不會隨 micro-batch 數量成長 |
| DualPipeV | 「Cut-in-half」 | V 形改良，以稍大的 bubble 為代價，去掉 2 倍的參數複製 |
| Chunk | 「pipeline 工作的單位」 | 一個 micro-batch 通過一個 pipeline 階段的一次 forward 或 backward pass |
| All-to-all dispatch | 「把 token 送到 expert」 | 把 token 路由到被指派的 MoE expert 的跨節點通訊 |
| All-to-all combine | 「把 expert 輸出帶回來」 | 在 MLP 之後收集 expert 輸出的跨節點通訊 |
| Expert Parallelism (EP) | 「把 expert 放在各 GPU 上」 | 把 MoE expert 分片到各 rank，讓不同 GPU 持有不同 expert |
| Pipeline Parallelism (PP) | 「把層放在各 GPU 上」 | 把模型層分片到各 rank，是 DualPipe 所排程的維度 |
| Bubble fraction | 「浪費的 GPU 時間」 | (bubble_time / total_time)，DualPipe 將它推向零的那個比例 |

## 延伸閱讀 (Further Reading)

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437), Section 3.3.2 and Figure 5](https://arxiv.org/abs/2412.19437)：DualPipe 的主要參考資料
- [DeepSeek — DualPipe GitHub repository](https://github.com/deepseek-ai/DualPipe)：開源參考實作，包含 DualPipeV (Cut-in-half) 模式
- [Qi et al. — Zero Bubble Pipeline Parallelism (arXiv:2401.10241, Sea AI Lab 2023)](https://arxiv.org/abs/2401.10241)：Zero Bubble 的前身
- [Sea AI Lab — DualPipe could be better without the Dual](https://sail.sea.com/blog/articles/63)：影響 DeepSeek EP 關閉模式的 DualPipeV 分析
- [Narayanan et al. — PipeDream / 1F1B (arXiv:1806.03377, 2018-2021)](https://arxiv.org/abs/1806.03377)：DualPipe 所比較的 1F1B 排程
- [Huang et al. — GPipe (arXiv:1811.06965, 2018)](https://arxiv.org/abs/1811.06965)：最初的 pipeline parallelism 論文與 bubble 問題
