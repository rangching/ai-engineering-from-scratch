# Production 中的 EAGLE-3 Speculative Decoding

> Speculative decoding 讓一個快速的 draft 模型搭配 target 模型。Draft 提出 K 個 tokens；target 在一次 forward 中驗證；被接受的 tokens 等於免費。2026 年，EAGLE-3 是 production 等級的變體：它在 target 模型的 hidden states 上訓練 draft head，而不是在原始 tokens 上，讓一般聊天的 acceptance rate alpha 達到 0.6-0.8 區間。正確的問題不是「draft 有多快」，而是「我的流量上 alpha 是多少？」如果 alpha 低於約 ~0.55，在高並行下 speculative decoding 是淨負效益，因為每次被拒絕的 draft 都要多付一次 target forward pass。這堂課教你先量測 alpha，再打開旗標。

**Type:** Learn
**Languages:** Python (stdlib, toy acceptance-rate simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 18 (Multi-Token Prediction)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出 speculative decoding 的三個世代，並解釋 EAGLE-3 相對於 EAGLE-2 和傳統 draft 模型改變了什麼。
- 定義 acceptance rate alpha，從 alpha 和 K（draft 長度）計算預期加速，並找出你目標並行度下的損益平衡 alpha。
- 解釋為什麼 speculative decoding 在 2026 年的 vLLM 中是 opt-in（而非預設），以及為什麼沒量測 alpha 就打開是 production 的反模式。
- 寫出量測計畫：用哪個 benchmark、哪種 prompt 分布、哪個並行點、以哪個指標作為關卡。

## 問題 (The Problem)

Decode 是 memory-bound。在 H100 上執行 Llama 3.3 70B FP8，每個 decode 步驟都要從 HBM 讀取全部約 ~70 GB 的權重，並為每個 sequence 產生一個 token。Decode 期間 GPU 運算幾乎閒置，瓶頸是 HBM 頻寬，而不是 matmul throughput。

Speculative decoding 利用了這個落差。用便宜的 draft 模型產生 K 個候選 tokens，再請 target 模型在單次 forward pass 中驗證全部 K 個。每個被驗證的 token 實質上是免費的（攤銷到 target 本來就得做的 batch-of-K forward 中）。

傳統的 draft-model 做法使用同家族較小的模型（用 Llama 3.2 1B 為 Llama 3.3 70B 做 draft）。這行得通，但 acceptance rate 平庸，因為較小模型的分布會偏離 target。EAGLE，然後 EAGLE-2，再到 EAGLE-3，直接在 target 模型的內部狀態上訓練輕量的 draft head，讓 draft 的分布更貼近 target。這就是為什麼 alpha 從 draft-model 的 0.4 提升到 EAGLE-3 的 0.6-0.8。

但有個陷阱：EAGLE-3 在 2026 年的 vLLM 中是 opt-in。`speculative_config` 必須明確設定。沒有旗標就沒有加速。沒在真實流量上量測 alpha 就打開的團隊，常常看到尾端 latency 變差而不是變好。

## 核心概念 (The Concept)

### Speculative decoding 實際買到什麼

沒有 spec decode 時，每個 token 的成本是一次 target forward。使用 draft 長度 K 和 acceptance alpha 的 spec decode，每次 target forward 的預期 tokens 是 `1 + K * alpha`。加速為 `(1 + K * alpha) / (1 + epsilon)`，其中 epsilon 是 draft 加 verify 的額外開銷。以 K=5、alpha=0.7 為例：`(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`。實際數字多落在 2-3x 附近，因為 production 流量的 alpha 很少那麼高，且 epsilon 會隨 batch size 增加。

### 為什麼 alpha 是唯一重要的指標

被拒絕的 tokens 不會消失，它們會讓第一個被拒絕的 token 多付一次 target forward。在 alpha 掉到 0.4 的工作負載上，你要付 draft 開銷、加驗證、再加重新抽樣。在高並行（例如 256 並行）下，decode batch 本身已經夠大，「只用 target」與「target 加 verify」之間的記憶體頻寬差距會縮小。在 2026 年多數硬體上，alpha 低於 0.55 時，spec decode 是淨負效益。

Alpha 隨工作負載而異。在 ShareGPT 風格的一般聊天上，以 ShareGPT 訓練的 EAGLE-3 可達 0.6-0.8。在特定領域流量（程式碼、醫療、法律）上，以一般資料訓練的 draft head 會掉到 0.4-0.6。訓練領域專屬的 draft head 能找回 alpha，這相較 target 的 fine-tuning 是輕量又快速的訓練工作。Phase 7 · 16 和 Phase 10 · 15 引用較高的數值（0.85 以上）。那些是符合 draft 訓練資料的聊天 benchmark 上的 per-token 比率。請依本頁的區間做規劃，並在你自己的流量上量測 alpha。

### EAGLE 各世代一覽

- **傳統 draft 模型**：同家族的小模型。Alpha 0.3-0.5。基礎設施簡單：載入兩個模型，每次 target forward 對應 draft 跑 K 次 forward。
- **EAGLE-1 (2024)**：在 target hidden states（最後一層）上訓練的單一 draft head。Alpha ~0.5-0.6。在 target 之上只多一點參數開銷。
- **EAGLE-2 (2025)**：自適應 draft 長度與樹狀 draft（在一次 target pass 中驗證多個分支）。Alpha ~0.6-0.7。Draft scheduler 更複雜。
- **EAGLE-3 (2025-2026)**：在多個 target 層（不只最後一層）上訓練 draft head，對齊更好。一般聊天上 alpha ~0.6-0.8。

### 2026 年的 production 配方

1. 先原樣交付 target 模型。在目標並行度下量測 baseline TTFT、ITL、throughput。
2. 透過 vLLM `speculative_config` 啟用 EAGLE-3 draft。重新執行 benchmark。
3. 記錄 acceptance rate alpha。vLLM V1 以 `spec_decode_metrics.accepted_tokens_per_request` 回報。除以要求的 draft 長度即得 alpha。
4. 如果在 production 流量分布上 alpha < 0.55，就停用 spec decode，或訓練領域專屬的 EAGLE-3 draft。
5. 在 production 並行度下重新執行。確認 P99 ITL 沒有變差。

### Production 的陷阱：P99 尾端

Spec decode 會降低平均 ITL。如果沒有調校，P99 可能變差。被拒絕的 draft 會觸發兩次 pass 的序列（draft + verify 失敗 + 重新抽樣）。在滿 batch 下，這兩次 pass 會串行。請看 P99 ITL，不要看 P50。

### EAGLE-3 已在哪裡部署

Google 在 2025 年於 AI Overviews 部署 speculative decoding（品質相同、回應更快）。vLLM V1 以 `speculative_config` 作為文件化的介面，其功能矩陣把 speculative decoding 標示為與 chunked prefill 相容。SGLang 支援 EAGLE-3，作為重度使用 prefix 的工作負載的建議 draft 路徑。

### 一行搞定損益平衡計算

預期加速：`S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`。令 `S = 1` 解出 alpha：`alpha_breakeven = verify_overhead / K`。以典型的 verify_overhead ~0.15 和 K=5：`alpha_breakeven = 0.03`。但那只是原始的 decode 數學。在高並行下 verify 開銷上升，而 decode batch 已經把記憶體讀取攤銷到各個 sequence，所以實務上有效的 alpha_breakeven 會升到約 ~0.45-0.55。

### 什麼時候不要用 speculative decoding

- latency 不重要的 batch-1 離線生成。用單純的 target。
- 非常短的輸出（少於 50 tokens）。Draft 開銷與 verify 成本占主導。
- 沒有領域訓練 draft head 的專門領域。Alpha 太低。
- 假設每一對功能都能組合。請檢查你的版本的 vLLM 相容性矩陣：v0.18.0 把 speculative decoding 標示為與 chunked prefill 相容。

```figure
mx-speculative-tree
```

## 實際使用 (Use It)

`code/main.py` 在一系列 alpha 值與 draft 長度 K 下，模擬有無 speculative decoding 的 decode 迴圈。它會印出損益平衡 alpha、實測加速與尾端行為。對多組 (alpha, K) 組合執行，確切看出 speculative decoding 在哪裡不再划算。

## 交付 (Ship It)

這堂課產出 `outputs/skill-eagle3-rollout.md`。給定 target 模型、流量分布描述與並行目標，它會產出分階段的 EAGLE-3 rollout 計畫：benchmark baseline、啟用設定、量測 alpha、以 alpha >= 0.55 作為關卡、監看 P99 ITL。

## 練習 (Exercises)

1. 執行 `code/main.py`。在 K=5 時，要達到 2x 加速需要多少 alpha？3x 呢？這對 verify_overhead 有多敏感？
2. 假設 production 流量是 70% 一般聊天、30% 程式碼。一般聊天在以 ShareGPT 訓練的 EAGLE-3 上達到 alpha 0.7；程式碼為 alpha 0.4。混合後的 alpha 是多少？spec decode 是淨正效益嗎？
3. 閱讀 vLLM `speculative_config` 文件。說出三種模式（draft model、EAGLE、N-gram），並檢查在你的 vLLM 版本中，每一種能與哪些功能組合。
4. 你看到啟用 EAGLE-3 後平均 ITL 下降 25%，但 P99 ITL 上升 15%。診斷並提出緩解方式。
5. 計算 Llama 3.3 70B 的 EAGLE-3 draft head 的記憶體成本。它和用 Llama 3.2 1B 當傳統 draft 相比如何？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Speculative decoding | 「draft 加 verify」 | 用便宜的模型提出 K 個 tokens，在一次 target forward 中驗證全部 K 個 |
| Acceptance rate alpha | 「spec 接受率」 | draft tokens 被 target 接受的比例；唯一重要的指標 |
| Draft length K | 「spec k」 | draft 每次 target forward 提出的 tokens 數；典型為 4-8 |
| Verify overhead epsilon | 「spec 開銷」 | 相對單純 target forward，驗證並重新抽樣的額外成本；隨 batch 增加 |
| EAGLE-3 | 「最新的 EAGLE」 | 2025-2026 變體；在多個 target 層上訓練 draft head；一般聊天 alpha 0.6-0.8 |
| `speculative_config` | 「vLLM spec 設定」 | vLLM V1 中的明確 opt-in；沒有預設代表沒有加速 |
| N-gram spec decode | 「N-gram draft」 | 在 GPU 端用 prompt 中的 N-gram 查找做 draft；與 chunked-prefill 相容 |
| Break-even alpha | 「無效 alpha」 | spec decode 加速為零時的 alpha；請在 production 並行度下留意 |
| Rejected-draft two-pass | 「重新抽樣成本」 | draft 被拒絕時的兩次 target forward；造成 P99 尾端 |

## 延伸閱讀 (Further Reading)

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/)：V1 中 `speculative_config` 與 chunked-prefill 相容性的權威來源。
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/)：確切的欄位集合。
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077)：原始 EAGLE draft-head 的公式化。
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858)：自適應 draft 與樹狀結構。
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html)：使用 speculative decoding 的高效 LLM 系統。
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding)：production rollout 檢查清單。
