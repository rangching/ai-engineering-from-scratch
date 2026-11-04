# Differential Attention (V2) (Differential Attention (V2))

> Softmax attention 會把一小部分的機率，分散到每個不相符的 token 上。在 100k 個 token 上，這些雜訊會累積起來，淹沒訊號。Differential Transformer (Ye et al., ICLR 2025) 把 attention 算成兩個 softmax 的差，減掉共有的雜訊底線，藉此解決這個問題。DIFF V2 (Microsoft, 2026 年 1 月) 是為正式堆疊所做的重寫：decode 延遲與 baseline Transformer 相當，不需要自訂 kernel，相容 FlashAttention。這堂課從頭到尾走過 V1 到 V2，並附上一個可用標準函式庫 Python 執行的差分運算的玩具實作。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 02 (self-attention), Phase 7 · 15 (attention variants), Phase 10 · 14 (architecture walkthrough)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 精確說明為什麼 softmax attention 有雜訊底線，以及為什麼它會隨 context 長度成長。
- 推導 differential attention 的公式，並解釋為什麼相減能抵消共有的雜訊成分，同時保留訊號。
- 走過 V1 到 V2 的 diff：什麼變快了、什麼變簡單了、什麼變穩定了，以及為什麼每項改變對正式的 pre-training 都是必要的。
- 用純 Python 從零實作 differential attention，並在合成的訊號加雜訊 query 上，經驗性地驗證雜訊抵消的特性。

## 問題 (The Problem)

標準的 softmax attention 有一個數學性質，在規模變大時會變成營運上的頭痛問題。對一個 query `q`，attention 權重是 `softmax(qK^T / sqrt(d))`。Softmax 永遠不會產生精確的零：每個不相符的 token 都會分到一些正的質量。這些殘餘的質量就是雜訊，而且會隨 context 長度擴大。在 128k 個 token 時，即使每個不相符的 token 只拿到 0.0001% 的機率，127,999 個加起來也貢獻了大約 13% 的總量。模型必須學會繞過一個隨 context 成長的雜訊底線。

經驗上，這會表現為 attention head 之間的干擾：長 context RAG 中憑空捏造的引用、100k token 檢索任務上的 lost-in-the-middle 失敗，以及在 32k 以上的 needle-in-haystack benchmark 上細微的準確度下降。Differential Transformer 論文 (arXiv:2410.05258, ICLR 2025) 量測了這個差距：DIFF Transformer 比同樣大小的 baseline，達到更低的 perplexity、更高的長 context 準確度，以及更少的幻覺。

DIFF V1 有三個問題，讓它進不了 frontier 的 pre-training 管線。它的 value cache 在每個 decode 步驟都必須載入兩次，它需要自訂的 CUDA kernel，破壞了與 FlashAttention 的相容性，而且它每個 head 的 RMSNorm 在 70B 以上規模的長時間訓練中會造成不穩定。DIFF V2 (Microsoft unilm blog, 2026 年 1 月 20 日) 修正了這三點。這堂課走過兩個版本，建立差分運算子，並在玩具 query 上對雜訊抵消做 benchmark。

## 核心概念 (The Concept)

### Softmax 的雜訊底線 (The noise floor of softmax)

對於 query `q` 與 key `K = [k_1, ..., k_N]`，attention 權重為：

```math
w_i = \frac{\exp(q \cdot k_i / \sqrt{d})}{\sum_j \exp(q \cdot k_j / \sqrt{d})}
```

沒有任何 `w_i` 會是零。如果 `k_i` 與 `q` 完全無關，分數 `q . k_i` 並不是 0，而是在零附近以 `||q||^2 / d` 的變異數波動。經過 softmax 正規化後，每個無關的 token 仍然對加權和貢獻 `O(1/N)`。無關 token 的總貢獻是 `O((N-1)/N) = O(1)`，這不是個小量。

模型想要的是類似硬性 top-k 的東西：對相符的 token 給高權重，其他地方接近零。Softmax 太平滑，無法直接做到。

### 差分的想法 (The differential idea)

把每個 head 的 Q 與 K 投影切成兩份：$Q = (Q_1, Q_2)$ 和 $K = (K_1, K_2)$。計算兩張 attention map：

```math
\begin{aligned}
A_1 &= \mathop{\mathrm{softmax}}(Q_1 K_1^T / \sqrt{d}) \\
A_2 &= \mathop{\mathrm{softmax}}(Q_2 K_2^T / \sqrt{d})
\end{aligned}
```

輸出：

```math
\mathop{\mathrm{DiffAttn}} = (A_1 - \lambda \cdot A_2) V
```

相減會抵消兩張 map 共有的任何雜訊分布。如果兩張 map 在 127k 個無關 token 上都有大致均勻的權重（在隨機初始化時就是如此），這些就會抵消。訊號，也就是集中在少數真正相關 token 上的尖峰權重，只有在兩張 map 中以相同的大小出現時才會被抵消，而模型一旦訓練之後就不會是這樣。

`lambda` 是每個 head 的可學習純量，參數化為 `lambda = exp(lambda_q1 dot lambda_k1) - exp(lambda_q2 dot lambda_k2) + lambda_init`。它可以是負的。`lambda_init` 預設是一個小的正數，例如 0.8。

### 為什麼這就像分頭的降噪 (Why this matches headed noise-canceling)

想像兩支錄同一個人聲音的有雜訊麥克風。兩者都收到說話者加上相關的背景雜訊。把其中一個減去另一個，共有的雜訊就消失了。人聲得以保留，因為兩個訊號在相位或振幅上的差異，足以防止完全抵消。每個 head 的 `lambda` 學到的就是這個平衡。

### V1 與 V2：diff (V1 vs V2: the diff)

V1 讓參數量與 baseline Transformer 相等。為了在每個 head 得到兩個 query，它把 head 維度減半。這損失了 head 的表達力，更痛的是，也讓每個 head 的 value cache 減半。Decode 必須在每一步載入 value cache 兩次（每個 softmax 分支一次）。結果：儘管參數量相同，decode 卻比 baseline 慢。

V2 把 query head 的數量加倍，並維持 KV head 不變（從 up-projection 借用參數）。Head 維度與 baseline 相同。相減之後，多出的維度會被投影回去，以符合 baseline Transformer 的 $O_W$ 投影。三件事同時發生：

1. Decode 速度與 baseline 相當（KV cache 只載入一次）。
2. FlashAttention 不用修改就能運作（不需要自訂 kernel）。
3. Decode 時的 arithmetic intensity 上升（每從 HBM 載入一個 byte，有更多運算）。

V2 也移除了 V1 用來穩定相減的每個 head 的 RMSNorm。在 70B 等級的 pre-training 規模下，那個 RMSNorm 會讓訓練後期不穩定。V2 以更簡單的初始化方案取代它，不需要額外的模組就能維持訓練穩定。

### 何時該用它 (When to reach for it)

| 工作負載 | 好處 |
|----------|---------|
| 長 context RAG (64k+) | 更乾淨的 attention map、更少憑空捏造的引用 |
| Needle-in-haystack benchmark | 在 32k 以上準確度顯著提升 |
| 多文件 QA | 文件之間的干擾較少 |
| 8k 的程式碼補全 | 邊際效果，不值得改架構 |
| 短聊天 (< 4k) | 與 baseline 基本上無法區分 |

價值隨 context 長度增加。在 4k 個 token 時，雜訊底線小到標準 attention 就夠了。在 128k 時，它就在傷害你。

### 它如何與 2026 年的其他旋鈕搭配 (How it stacks with other 2026 knobs)

| 特性 | 與 DIFF V2 相容嗎？ |
|---------|------------------------|
| GQA | 是（V2 增加的是 Q head，不是 KV head） |
| MLA (DeepSeek) | 原則上可以，目前沒有結合兩者的已發表論文 |
| MoE | 是（attention 與 MLP block 無關） |
| RoPE | 是（不變） |
| YaRN / 長 context scaling | 是（正是 DIFF 最有幫助的地方） |
| FlashAttention | V2 可以（V1 不行） |
| Speculative decoding | 是（attention 的改變對 spec-decode 迴圈是不可見的） |

```figure
differential-attention
```

## 動手做 (Build It)

`code/main.py` 用純 Python 實作 differential attention。一個具有已知訊號加雜訊結構的玩具 query，讓你直接量測雜訊抵消比。

### 步驟 1：標準 softmax attention (Step 1: standard softmax attention)

標準函式庫的矩陣運算：list 的 list、手寫 matmul、以減去最大值來維持數值穩定的 softmax。

```python
def softmax(row):
    m = max(row)
    exps = [math.exp(x - m) for x in row]
    s = sum(exps)
    return [e / s for e in exps]
```

### 步驟 2：把 Q、K 切成兩半 (Step 2: split Q, K into two halves)

V1 風格：把 head 維度減半。V2 風格：維持 head 維度，並把 head 數量加倍。為了教學清晰，玩具實作使用 V1，數學是相同的，只有記帳方式不同。

### 步驟 3：兩個 softmax 分支 + 相減 (Step 3: two softmax branches + subtraction)

```python
A1 = [softmax([dot(q1, k) / scale for k in K1]) for q1 in Q1]
A2 = [softmax([dot(q2, k) / scale for k in K2]) for q2 in Q2]
diff_weights = [[a1 - lam * a2 for a1, a2 in zip(r1, r2)] for r1, r2 in zip(A1, A2)]
out = [[sum(w * v[j] for w, v in zip(row, V)) for j in range(d_v)] for row in diff_weights]
```

注意：輸出權重可以是負的。這沒問題，value cache 仍然能處理帶符號的貢獻。後續的 V 投影會吸收這個符號。

### 步驟 4：雜訊抵消的量測 (Step 4: noise cancellation measurement)

建立長度 1024 的合成序列。把訊號 token 放在已知位置，其餘用雜訊填滿。計算 (a) 標準 softmax attention 在訊號位置上的權重，以及 (b) differential attention 的權重。量測兩者的訊雜比。DIFF attention 能穩定地產生更高的訊雜比，倍數為 3 倍到 10 倍，取決於兩個分支被訓練得有多不同。

### 步驟 5：V1 與 V2 的參數計算 (Step 5: V1 vs V2 parameter accounting)

給定一個設定 (hidden=4096, heads=32, d_head=128)，印出：

- Baseline Transformer：Q、K、V 各為 `hidden * hidden` 大小，MLP 為 4 * hidden。
- DIFF V1：Q、K 各為 `hidden * hidden`，V 的大小為 `hidden * hidden`（不變），內部 head 維度減半。增加每個 head 的 `lambda` 參數 $(O(\text{heads} \cdot d_{\text{head}}))$。
- DIFF V2：Q 的大小為 `2 * hidden * hidden`，K 為 `hidden * hidden`，V 為 `hidden * hidden`。多出的維度在 $O_W$ 之前被投影回去。增加相同的 `lambda` 參數。

玩具程式會量測 V2 額外的參數成本（每個 attention block 大約多出 `hidden * hidden`）並印出來。

## 實際使用 (Use It)

截至 2026 年 4 月，DIFF V2 還沒有內建在每個正式的推論伺服器中，但 vLLM 與 SGLang 的整合正在進行。同時，這個模式出現在：

- Microsoft 內部的長 context 正式模型。
- 數個以 256k 以上 context 為目標的開放模型訓練的研究重現。
- 在交替層結合 DIFF attention 與 sliding-window attention 的混合架構。

2026 年你會在以下情況使用它：

- 從頭訓練一個以 64k 以上有效 context 為目標的新模型。從一開始就加入 differential attention，之後再重新訓練的代價很高。
- Fine-tune 一個 lost-in-the-middle 失敗主導你的 eval 的長 context 模型。對 Q 投影做 LoRA，可以近似 DIFF 的結構。

以下情況你不會用它：

- 你正在服務一個長 context 表現穩定的 pre-trained dense 模型。在既有的權重上，重新訓練的成本很少能回本。
- 你的 context 總是小於 16k。雜訊底線可以忽略。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-diff-attention-integrator.md`。給定一個模型架構、目標 context 長度、幻覺輪廓與訓練預算，它會產出將 differential attention 加進新的 pre-training 或 LoRA fine-tune 的整合計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。驗證在合成 query 上，differential attention 回報的訊雜比高於標準 softmax attention。改變雜訊振幅，並展示標準 attention 變得不堪使用的交會點。

2. 為 7B 等級的模型 (hidden=4096, heads=32, d_head=128, 32 層)，計算從 baseline 到 DIFF V1，以及從 baseline 到 DIFF V2 的參數量差異。展示哪些元件增加了參數，哪些維持不變。

3. 閱讀 DIFF V1 論文 (arXiv:2410.05258) 的第 3 節，以及 DIFF V2 Hugging Face blog 的第 2 節。用兩句話解釋，為什麼 V1 每個 head 的 RMSNorm 是必要的，以及為什麼 V2 能移除它，而不造成訓練發散。

4. 實作一個 ablation：以 `lambda = 0`（純第一個 softmax）和 `lambda = 1`（完整相減）計算 differential attention。在合成 query 上，量測訊雜比在整個掃描範圍內如何變化。找出使訊雜比最大的 `lambda`。

5. 把玩具擴充到 GQA + DIFF V2。選擇 8 個 KV head 與 32 個 Q head。展示 KV cache 大小與具有相同 (8, 32) 設定的 baseline GQA 模型相符。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Differential attention | 「兩個 softmax 相減」 | 把 Q、K 切成兩半，計算兩張 softmax map，從第一張減去第二張（以 $\lambda$ 縮放），再乘上 V |
| Noise floor | 「softmax 非零的尾巴」 | Softmax 放在每個無關 token 上的 $O(1/N)$ 權重，在長 context 上加總為 $O(1)$ |
| lambda | 「相減的縮放」 | 每個 head 可學習的純量，參數化為 `exp(lq1.lk1) - exp(lq2.lk2) + lambda_init`，可以是負的 |
| DIFF V1 | 「ICLR 2025 版本」 | 最初的 Differential Transformer，把 head 維度減半以保持參數量，需要自訂 kernel，decode 較慢 |
| DIFF V2 | 「2026 年 1 月的修正」 | 加倍 Q head，維持 KV head，decode 速度與 baseline 相當，且可搭配 FlashAttention |
| Per-head RMSNorm | 「V1 的穩定器」 | V1 在相減之後套用的額外 norm，V2 為避免訓練後期不穩定而移除它 |
| Signal-to-noise ratio | 「有多少 attention 被浪費」 | 真正訊號位置上的權重，與無關位置的平均權重之比 |
| Lost in the middle | 「長 context 的失敗模式」 | 一種經驗現象：位於長 context 中段的文件，檢索準確度會下降，DIFF attention 能減輕它 |
| Arithmetic intensity | 「每載入一個 byte 的 FLOPs」 | V2 在 decode 時，透過每次 KV 載入加倍 query 數而提高的比值，對 memory-bound 的 decode 很重要 |

## 延伸閱讀 (Further Reading)

- [Ye et al. — Differential Transformer (arXiv:2410.05258, ICLR 2025)](https://arxiv.org/abs/2410.05258)：原始論文，包含雜訊抵消的理論與長 context 的 ablation
- [Microsoft unilm — Differential Transformer V2 (Hugging Face blog, January 2026)](https://huggingface.co/blog/microsoft/diff-attn-v2)：為正式堆疊所做的重寫，與 baseline 的 decode 相當，相容 FlashAttention
- [Understanding Differential Transformer Unchains Pretrained Self-Attentions (arXiv:2505.16333)](https://arxiv.org/abs/2505.16333)：從理論分析為什麼相減能恢復 pretrained 的 attention 結構
- [Shared DIFF Transformer (arXiv:2501.17900)](https://arxiv.org/html/2501.17900)：參數共用的變體
- [Vaswani et al. — Attention Is All You Need (arXiv:1706.03762)](https://arxiv.org/abs/1706.03762)：DIFF 所減去的 baseline Transformer
- [Liu et al. — Lost in the Middle (arXiv:2307.03172)](https://arxiv.org/abs/2307.03172)：DIFF attention 所針對的長 context benchmark
