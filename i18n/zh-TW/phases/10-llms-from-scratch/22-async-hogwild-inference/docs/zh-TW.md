# Async 與 Hogwild! Inference (Async and Hogwild! Inference)

> Speculative decoding（Phase 10 · 15）在單一序列內平行化 token。Multi-agent 框架則跨整條序列平行化，但必須明確協調（投票、子任務切分）。Hogwild! Inference（Rodionov et al., arXiv:2504.06261）做的是另一件事：讓同一個 LLM 的 N 個實例，針對一份*共用*的 key-value cache 平行執行。每個 worker 都能立即看到其他 worker 產生的每個 token。現代的推理模型，例如 QwQ、DeepSeek-R1，不需要任何 fine-tuning，就能透過這個共用 cache 自我協調。這個方法還處於實驗階段，但它開啟了一條與 spec decode 正交的、全新的推論平行化軸線。這堂課用標準函式庫 Python 實作一個雙 worker 的 Hogwild! 模擬器，並解釋共用 cache 的協作為什麼能從既有模型的推理能力中浮現。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 12 (inference optimization), Phase 10 · 15 (speculative decoding)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述三種常見的平行 LLM 拓撲（投票、子任務、Hogwild!），並指出各自針對哪些問題。
- 陳述 Hogwild! 的核心設定：多個 worker、一份共用 KV cache、透過 self-prompting 浮現的協調。
- 以 worker 數量 `N`、任務層級的平行度 `p`，以及協調開銷 `c` 的函數，計算 Hogwild! 的實際耗時加速。
- 在玩具問題上實作雙 worker 的 Hogwild! 模擬器，並觀察浮現的任務分工。

## 問題 (The Problem)

現代 LLM 靠產生長長的推理鏈來解決困難的問題，5000 個 token 的逐步邏輯很常見，在深度數學問題上則會到數萬個 token。在 70B 模型上以每秒 35 個 token 的速度 decode，50k 個 token 要 24 分鐘。這個模型不算互動式。

Speculative decoding（Phase 10 · 15）透過在單一序列內平行化，帶給你 3-5 倍的加速。超過這個之後，自迴歸 decoding 的循序相依就是硬性的天花板。每個新 token 都依賴所有先前的 token。

顯而易見的問題：我們能跨序列平行化嗎？對同一個問題執行同一模型的多個副本，讓它們合作，並分工？

先前的工作：投票集成（執行 N 個模型，挑出多數答案）、tree-of-thought（分支推理路徑再重新組合），以及 multi-agent 框架（為每個 agent 指派子任務，使用協調者）。這些在特定任務領域都有幫助。它們也都引入了明確的協調機制：投票規則、分支與剪枝的邏輯、agent 之間的訊息協定。

Hogwild! Inference 採取不同的做法。N 個 worker 共用單一個 KV cache。每個 worker 立即看到其他 worker 產生的每個 token，就像那是它自己的 context 一樣。這些 worker 無需任何訓練或 fine-tuning，就想出如何分工。現代的推理模型（QwQ、DeepSeek-R1、Claude 家族的推理模式）能讀取共用 cache，並說出像是「我看到 worker 2 已經處理了 base case，所以我來處理 inductive step」這樣的話。

加速幅度取決於工作負載，且截至 2026 年 4 月仍屬實驗階段。但這個想法值得了解，因為它開啟了推論平行化的新軸線。

## 核心概念 (The Concept)

### 設定 (The setup)

初始化 N 個 worker 程序，全都執行同一個 LLM。不使用每個 worker 各自的 KV cache，而是維護*一個*共用 cache。當 worker `i` 產生 token `t_j`，這個 token 會被寫入共用 cache 的下一個位置。當 worker `k` 進行它的下一步時，它讀取 cache 目前的狀態（包含全部 N 個 worker 到目前為止產生的所有東西）。

在步驟時間，worker 們競相寫入 token。沒有每個 worker 的位置索引，cache 就是一條不斷成長的單一序列。順序由寫入到達的時間決定。

### 為什麼協調會浮現 (Why coordination emerges)

這些 worker 共用一個 prompt。通常類似「你是共同處理這個問題的 N 個實例之一。每個實例都讀取共用記憶體，並可以看到其他實例寫了什麼。避免重複的工作。」這個 prompt 加上共用 cache 就夠了。推理模型讀取 cache，注意到問題的哪些部分已被嘗試過，並（常常但不總是）轉向尚未探索的部分。

Hogwild! 論文 (Rodionov et al., 2025) 回報了類似下列的觀察：

- Worker 制定計畫，並透過 cache 將計畫傳達給其他 worker。
- Worker 注意到其他 worker 推理中的錯誤並指出來。
- 當計畫失敗時，worker 會調整並提出替代方案。
- 當被提示檢查重複時，worker 會偵測到它並轉向。

這些都不需要 fine-tuning。浮現的行為來自模型本來就具備的推理能力。

### 命名 (The naming)

論文的名稱借用了 Hogwild! SGD (Recht et al., 2011)，一種非同步更新的 optimizer。類比是：SGD 的非同步 worker 都寫入共用的參數向量；Hogwild! Inference 的 worker 都寫入共用的 KV cache。兩者都仰賴經驗上的收斂，而不是同步的保證。

### RoPE 讓這件事可行 (RoPE makes this tractable)

Rotary Position Embeddings (RoPE, Su et al. 2021) 透過 Q 與 K 向量中的旋轉來編碼位置資訊。因為旋轉後的 query 與旋轉後的 key 之間的 attention 分數，只取決於它們的相對距離，所以整塊已快取的項目可以移到新的偏移位置而不需重新計算：旋轉 query，而不是重新旋轉每個已快取的 key。Hogwild! Inference 對每個 cache block 做一次這樣的事：對於位置 `i_q` 的 query，以及在該 worker 視角中從 `i_k` 開始的 block，query 會依相對距離 `i_q - i_k` 旋轉。這就是每個 worker 以自己的順序看到共用 cache block 的方式。這個技巧適用於連續 block 的整體平移。把個別的 token 移到不同的位置會改變它們的旋轉，仍然需要重新計算。

在使用學習式位置或絕對位置的模型中，Hogwild! 每次並行寫入都需要讓 cache 失效。RoPE 讓 cache 保持穩定。

### 耗時的數學 (Wall-time math)

令 `T_serial` 為單一 worker 獨自解決問題的時間。令 `p` 為任務層級可平行化的比例。令 `c` 為每一步的協調開銷（讀取擴大後的 cache、決定要寫什麼）。

單一 worker 的時間：`T_serial`。
N 個 worker 的 Hogwild! 時間，若協調免費：`T_serial * ((1 - p) + p / N)`。經典的 Amdahl。
加上協調開銷：`T_serial * ((1 - p) + p / N) + c * steps_per_worker`。

要讓 worker 有生產力，`c` 必須相對於每一步的 decode 時間很小。在產生 5k 個以上 token 的推理模型上，worker 可以負擔數百個 token 的協調開銷，仍然領先。在短的聊天任務上，協調佔主導，Hogwild! 比循序還差。

### 具體範例 (Concrete example)

推理問題：10k 個 token 的 chain-of-thought。假設問題有 `p = 0.7` 的可平行化內容（不同的證明策略、不同的案例分析），且每個 worker 有 `c = 200` 個 token 的協調開銷。使用 `N = 4` 個 worker：

- 循序時間：10000 個 decode 步驟。
- Hogwild! 時間：10000 * (0.3 + 0.7 / 4) + 200 * 4 = 10000 * 0.475 + 800 = 5550 個 decode 步驟。
- 加速：10000 / 5550 = 1.8 倍。

這很溫和。但在更長的推理問題（50k 個 token）上，協調開銷會被攤提，加速會逼近 2.5-3 倍。Hogwild! 是推論版的執行緒層級平行化，在一種讓你自然撰寫多執行緒程式碼的語言裡。

### 何時該用 Hogwild! (When to reach for Hogwild!)

- 長推理問題（數千個 token），且任務可以跨獨立的子目標平行化。
- 被訓練成一步一步思考的推理模型。非推理模型無法良好地自我協調。
- 有足夠 VRAM 容納共用 cache 加上 N 個 worker 程序的單節點部署。Cache 是共用的，但每個 worker 都有自己的 activation 記憶體。

### 何時不該用 (When not to)

- 短的互動式聊天。協調開銷佔主導。
- 無法平行化的任務（單一線性的證明、單次編譯）。N=1 就是上限。
- 非推理模型。不會浮現協調。
- 多節點部署。共用 cache 需要非常快速的跨 worker 同步。節點內沒問題；跨節點則是延遲的災難。

### 實驗性的現況 (The experimental status)

截至 2026 年 4 月，Hogwild! 是一個有開源 PyTorch 實作的研究方法。尚未有正式環境採用。三個阻礙：

1. 跨並行程序管理共用 KV cache 的工程並不簡單。
2. 浮現的協調取決於任務；benchmark 仍在建立中。
3. 相較於 speculative decoding 已經提供的加速，這些加速是溫和的，兩者可以結合，但結合後的工程又多了一層。

值得了解。值得實驗。還不值得把產品押在上面。

```figure
continuous-batching
```

## 動手做 (Build It)

`code/main.py` 實作了一個玩具版的 Hogwild! 模擬器：

- 兩個 worker 程序，每個都是確定性的「LLM」，以已知機率產生數種 token 類別之一（work-token、observe-token、coordinate-token）。
- 一份共用 cache（就是一個 token 的 list），兩個 worker 都讀寫。
- 一個簡單的協調邏輯：當 worker 看到另一個 worker 已經在某類別中產生足夠的 work token 時，它就挑選不同的類別。

模擬器在固定的步驟預算下執行，並回報：

- 產生的 work-token 總數。
- 總耗時（worker 步驟數）。
- 相對於單一 worker 的有效加速。
- 哪個 worker 寫了哪個 token 的追蹤紀錄。

### 步驟 1：共用 cache (Step 1: the shared cache)

一個兩個 worker 都會附加的 list。真實實作中會用簡單的鎖（Python `threading.Lock`）；我們用計數器來模擬。

### 步驟 2：Worker 迴圈 (Step 2: the worker loop)

每個 worker 在每一步：

- 讀取目前的共用 cache。
- 根據已經在那裡的內容，決定要寫哪個類別的 token。
- 寫入一個 token。

### 步驟 3：協調啟發式規則 (Step 3: the coordination heuristic)

如果類別 X 在 cache 中已有 K 個 token，而 worker 打算寫的類別是 X，worker 就切換到類別 Y。這是推理模型「注意到這已經被涵蓋，改做別的事」這種行為的玩具替身。

### 步驟 4：量測的加速 (Step 4: measured speedup)

在相同的總步驟預算下，分別以 N=1 個 worker 與 N=2 個 worker 執行模擬器。計算產生的 work-token。因為由協調驅動的任務分工，N=2 應該產生大約 1.5-1.8 倍的 work-token。

### 步驟 5：對協調施壓 (Step 5: stress the coordination)

降低協調啟發式規則的敏感度。再次執行。觀察到沒有良好的協調時，N=2 會重複產生相同的 token，加速下降到 1 以下。這符合論文的觀察：這個技巧只有在 worker 具備自我協調的推理能力時才有效。

## 實際使用 (Use It)

截至 2026 年 4 月，Hogwild! 在正式環境中的整合是研究等級的。Yandex/HSE/IST 的參考實作以 PyTorch 為基礎，目標是 DeepSeek-R1 與 QwQ 模型上的單節點多程序設定。

務實的採用路徑：

1. 剖析你的推理任務工作負載。量測探索性（多種策略、案例分析、搜尋）token 與線性 token 的比例。
2. 如果探索性佔主導，執行雙 worker 的 Hogwild! 實驗。量測耗時的改善。
3. 如果改善低於 1.3 倍，你處於協調主導的區間。退回單一 worker。
4. 如果改善超過 1.5 倍，推進到 N=4 並再次量測。報酬遞減通常在 N=4-8 左右出現。

與 speculative decoding 結合：每個 Hogwild! worker 可以獨立使用 spec decode。兩種加速（大致上）會相乘，使 3 倍的 spec decode 與 1.8 倍的 Hogwild!，相對於天真的單一 worker decoding，達到有效的 5.4 倍。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-parallel-inference-router.md`。給定推理工作負載的輪廓（token 預算、任務平行度輪廓、模型家族、部署目標），它會在投票、tree-of-thought、multi-agent、Hogwild! 與 speculative decoding 策略之間做路由。

## 練習 (Exercises)

1. 以預設設定執行 `code/main.py`。確認 N=2 的 Hogwild! 設定，在相同耗時下，比 N=1 baseline 產生更多的 work-token。

2. 降低協調啟發式規則的強度（設定 `coordination_weight=0.1`）。重新執行。展示加速崩潰。解釋原因：當 worker 無法協調時，它們會重複付出努力。

3. 計算 50k 個 token 的推理任務，在 `p=0.8, c=500` 且 N=4 個 worker 下，預期的 Hogwild! 加速。對 1k 個 token 的聊天任務，在 `p=0.3, c=200` 且 N=4 個 worker 下做同樣的計算。為什麼一個是贏、另一個是輸？

4. 閱讀 Hogwild! 論文的第 4 節（初步評估）。找出作者回報的兩種失敗模式。描述更好的協調 prompt 可能如何減輕各自的問題。

5. 在玩具版中結合 Hogwild! 與 speculative decoding：每個 worker 內部使用 2 個 token 的 spec-decode。回報相乘的加速。當兩個 worker 都想延伸同一個共用 cache 前綴時，會出現什麼記帳問題？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Hogwild! | 「平行 worker，共用 cache」 | 同一個 LLM 的 N 個實例並行執行，共用一份 KV cache，透過 self-prompting 浮現協調 |
| Shared KV cache | 「協調的媒介」 | 所有 worker 都讀寫的單一個不斷成長的 KV buffer，讓 token 在各 worker 之間即時可見 |
| Emergent coordination | 「不需要訓練」 | 具備推理能力的 LLM 能讀取共用 cache 並分工，不需要任何 fine-tuning 或明確協定 |
| Coordination overhead (c) | 「花在定位上的 token」 | 每個 worker 讀取擴大後的 cache 並決定要做什麼的成本，必須相對於總 decode 時間保持很小 |
| Parallelizable fraction (p) | 「能平行執行的部分」 | 任務層級的平行度：總工作量中，並非本質上循序的部分所佔的比例 |
| RoPE enables Hogwild! | 「Rotary 位置對平移不變」 | 因為位置是旋轉，寫入共用 cache 不需要重新計算先前的 token |
| Voting ensemble | 「執行 N 個，挑多數」 | 最簡單的平行推論拓撲，適合分類，對長篇推理較不適合 |
| Tree of thought | 「分支與剪枝」 | 探索多個分支並剪枝的推理策略，有明確的協調邏輯 |
| Multi-agent framework | 「指派子任務」 | 每個 agent 取得一個角色，由協調者統籌，協定開銷沉重 |

## 延伸閱讀 (Further Reading)

- [Rodionov et al. — Hogwild! Inference: Parallel LLM Generation via Concurrent Attention (arXiv:2504.06261)](https://arxiv.org/abs/2504.06261)：Hogwild! 論文，在 QwQ 與 DeepSeek-R1 上的初步評估
- [Recht, Re, Wright, Niu — Hogwild!: A Lock-Free Approach to Parallelizing Stochastic Gradient Descent (arXiv:1106.5730, NeurIPS 2011)](https://arxiv.org/abs/1106.5730)：最初的 Hogwild!，命名的由來
- [Su et al. — RoFormer: Enhanced Transformer with Rotary Position Embedding (arXiv:2104.09864)](https://arxiv.org/abs/2104.09864)：RoPE，讓共用 cache 推論可行的性質
- [Yao et al. — Tree of Thoughts: Deliberate Problem Solving with Large Language Models (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601)：與 Hogwild! 正交的 tree-of-thought 推理策略
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192)：speculative decoding，Hogwild! 可與之組合的單序列內平行化
- [Hogwild! reference PyTorch implementation](https://github.com/eqimp/hogwild_llm)：論文實驗的唯一事實來源
