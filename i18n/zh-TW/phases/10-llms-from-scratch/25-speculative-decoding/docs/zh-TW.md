# Speculative Decoding 與 EAGLE (Speculative Decoding and EAGLE)

> Frontier LLM 每產生一個 token，就需要對數十億個參數做一次完整的 forward pass。這次 forward pass 的資源配置嚴重過剩：多數時候，一個小得多的模型就能正確猜出接下來的 3-5 個 token，大模型只需要*驗證*這個猜測。當猜對時，你用一個 token 的價錢拿到 5 個 token。Speculative decoding（Leviathan et al. 2023）讓這件事變得精確，而 EAGLE-3（2025）把接受率推到每次驗證約 4.5 個 token，在輸出分布相符的情況下達到 4-5 倍的加速。

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 分鐘

## 問題 (The Problem)

70B 等級的模型在 H100 上的 decode 吞吐量，通常是每秒 40-80 個 token。每個 token 都需要一次完整的 forward pass，從 HBM 讀取所有模型權重。你不能在不改變輸出的前提下把模型變小。你不能把 batch size 增加到超過記憶體。你被卡住了，除非你能讓模型在每次 forward pass 中輸出不只一個 token。

自迴歸生成看起來本質上是循序的：`x_{t+1} = sample(p(· | x_{1:t}))`。但這裡有一個並行的機會。如果你有一個便宜的預測器說「接下來 4 個 token 大概是 [a, b, c, d]」，你就可以在大模型的**單一次 forward pass** 中驗證全部 5 個位置，並接受最長的相符前綴。

Leviathan、Kalai、Matias（2023，「Fast Inference from Transformers via Speculative Decoding」）透過一個巧妙的接受/拒絕規則，讓這件事變得精確，這個規則保留了 target model 的取樣分布。相同的輸出分布，快 2-4 倍。

## 核心概念 (The Concept)

### 雙模型設定 (The Two-Model Setup)

- **Target model** `M_p`：你真正想要從中取樣的大型、慢速、高品質模型。分布：`p(x)`。
- **Draft model** `M_q`：小型、快速、品質較低的模型。分布：`q(x)`。小 5-30 倍。

每一步：

1. Draft model 以自迴歸方式提出 `K` 個 token：`x_1, x_2, ..., x_K ~ q`。
2. Target model 對全部 `K+1` 個位置平行執行**一次** forward pass，為每個被提出的 token 產生 `p(x_k)`。
3. 依下面修改過的 rejection-sampling 規則，由左至右接受/拒絕每個 token。接受最長的相符前綴。
4. 如果任何 token 被拒絕，就從修正後的分布中抽樣出替代 token 並停止。否則從 `p(· | x_1...x_K)` 抽樣一個 bonus token。

如果 draft 與 target 完全吻合，每次 target forward 你會得到 K+1 個 token。如果 draft 在位置 1 就錯了，你只會得到 1 個 token。

### 精確性規則 (The Exactness Rule)

Speculative decoding 在分布上**可證明等價於從 p 取樣**。拒絕規則：

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

其中 `(p - q)+` 表示逐點差的正部分。當 draft 與 target 一致 (`p ≈ q`) 時，接受率接近 1。當它們不一致時，residual 分布的建構方式，使整體樣本仍然恰好是 `p`。

**貪婪情況。** 對於 temperature=0 的取樣，只要檢查 `argmax(p) == x_t`。如果是，就接受；如果不是，就輸出 `argmax(p)` 並停止。

### 預期加速 (Expected Speedup)

如果 draft model 在 token 層級的接受率是 `α`，每次 target forward pass 預期產生的 token 數為：

```math
\mathbb{E}[\text{tokens}] = \frac{1 - \alpha^{K+1}}{1 - \alpha} \qquad K = \text{draft length}, \ \alpha \in [0, 1]
```

在 `α = 0.8, K = 4` 時：每次 forward 為 `(1 - 0.8^5)/(1 - 0.8) = 3.36` 個 token。單次 target forward 的成本大約是 `cost_q * K + cost_p`（K 個 draft 步驟加一次 target 驗證）。如果 `cost_p >> cost_q * K`，吞吐量的加速比為 `3.36× / 1 = 3.36×`。

唯一真正的參數是 `α`，它完全取決於 draft 與 target 的對齊程度。一個好的 draft 就是一切。

### 訓練 Draft：蒸餾 (Training the Draft: Distillation)

隨機的小模型是很差的 draft。標準配方是從 target 蒸餾：

1. 挑選一個小架構（70B 的 target 用約 1B，7B 的 target 用約 500M）。
2. 讓 target model 跑過大型文字語料，儲存它的 next-token 分布。
3. 以對 target 分布的 KL divergence 訓練 draft（而不是對 ground-truth token）。

結果：`α` 在程式碼上通常是 0.6-0.8，在自然語言聊天上是 0.7-0.85。正式環境的加速為 2-3 倍。

### EAGLE：樹狀草擬 + 特徵重用 (EAGLE: Tree Drafting + Feature Reuse)

Li、Wei、Zhang、Zhang（2024，「EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty」）觀察到標準 speculative decoding 中的兩個低效率之處：

1. Draft 做 K 次循序步驟，每次都是完整堆疊。但 draft 可以重用 target 在最近一次驗證中的特徵（hidden state），target 已經算出了豐富的表示，而 draft 卻在從頭重新推導。
2. Draft 輸出的是線性鏈。如果 draft 能輸出候選的*樹*（每個節點有多個猜測），target 的單一次 forward pass 就可以透過 tree attention mask 平行驗證多條候選路徑，並挑出最長的被接受分支。

EAGLE-1 的改變：
- Draft 輸入 = target 在位置 t 的最終 hidden state，而不是原始 token。
- Draft 架構 = 1 層 transformer decoder（而不是獨立的小模型）。
- 輸出 = 每個深度 K = 4-8 個候選的樹，深度 4-6。

EAGLE-2（2024）加入動態樹拓撲：樹在 draft 不確定的地方變寬，在有把握的地方保持窄。在不增加驗證成本的情況下，提升 `α_effective`。

EAGLE-3（Li et al. 2025，「EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test」）移除了對固定頂層特徵的依賴，並以新的「test-time simulation」loss 訓練 draft，也就是 draft 是在與 target 測試時分布相符的輸出上訓練的，而不是 teacher-forced 的訓練分布。接受率從 0.75（EAGLE-2）上升到 0.82（EAGLE-3），平均每次驗證的 token 數從 3.0 上升到 4.5。這些數字來自單一個參考設定。接受率取決於模型配對、draft 樹與流量：Phase 10 · 15 引用在聊天 benchmark 上每個被接受路徑 token 為 0.88 到 0.92，而 Phase 17 · 05 對正式環境的一般聊天則以 0.6 到 0.8 來規劃。

### Tree Attention 驗證 (Tree Attention Verification)

當 draft 輸出一棵樹時，target model 使用 **tree attention mask** 在單一次 forward pass 中驗證它，這是一個編碼樹拓撲而非單純一條直線的 causal mask。每個 token 只 attend 到它在樹中的祖先。驗證 pass 仍然是一次 forward、一次 matmul；拓撲 mask 只多花幾個額外的 KV 項目。

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

如果 `a, b` 是競爭的第一個 token 候選，`c, d, e, f` 是第二個 token 的候選，全部六個位置都在一次 forward pass 中被驗證。輸出是沿著任何被接受路徑的最長前綴。

### 何時有效，何時無效 (When It Wins, When It Doesn't)

**有效：**
- 文字可預測的聊天/補全（程式碼、常見英文、結構化輸出）。`α` 很高。
- decode 期間有未使用的 GPU 運算的情況（memory-bound 階段）。樹狀草擬利用了可用的 FLOPs。

**失敗/沒有好處：**
- 高度隨機的輸出（高 temperature 的創意寫作）。`α` 趨向 `1/|vocab|`。
- 極高並行度的 batch 服務：batching 已經填滿了 FLOPs，樹狀驗證幾乎沒有空間。
- 非常小的 target model，draft 並沒有小多少。

正式環境的團隊通常回報，聊天上實際耗時加速 2-3 倍，程式碼生成上 3-5 倍，創意寫作上幾乎為零。

```figure
speculative-decoding
```

## 動手做 (Build It)

`code/main.py`：

- 一個參考的 `speculative_decode(target, draft, prompt, K, temperature)`，實作精確的拒絕規則，並驗證它保留了 target 的分布（與單純 target 取樣相比，經驗 KL < 0.01）。
- 一個 EAGLE 風格的樹狀 drafter，以 top-p 分支建立深度 K 的樹。
- 一個 tree attention mask 建構器，為 verifier 產生正確的 causal 模式。
- 一個接受率測試工具，在一個小型 LM 上執行兩者（從 GPT-2-medium 的 target 蒸餾出一個 GPT-2-small）。

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_dists = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_dists.append(probs)
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_dists[k][tok]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_dists[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## 實際使用 (Use It)

- **vLLM** 與 **SGLang** 內建一等公民的 speculative decoding 支援。在 vLLM 中，傳入 `--speculative-config` 一個包含 `method`、`model` 與 `num_speculative_tokens` 的 JSON 物件；EAGLE-3 是 `"method": "eagle3"`。
- **NVIDIA TensorRT-LLM** 原生支援 Medusa 與 EAGLE 樹。
- **參考的 draft model**：`Qwen/Qwen3-0.6B`（為 Qwen3-32B 草擬）、`meta-llama/Llama-3.2-1B-Instruct`（為 Llama 3.x 70B 草擬）。
- **Medusa head**（Cai et al. 2024，「Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads」）：不使用 draft model，而是在 target 本身加上 K 個平行的預測 head。更容易部署，接受率略低於 EAGLE。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-speculative-tuning.md`，這個 skill 會剖析 target model 的工作負載，並選擇：draft model、K（draft 長度）、樹的寬度、temperature，以及何時退回單純 decode。

## 練習 (Exercises)

1. 實作精確的拒絕規則並以經驗驗證。透過 `speculative_decode` 與單純 target 取樣各執行 10K 個樣本；計算兩個輸出分布之間的 TV distance。應該 < 0.01。

2. 計算加速公式。給定固定的 `α` 與 `K`，畫出每次 target forward 預期的 token 數。為 $\alpha \in \lbrace 0.5, 0.7, 0.9 \rbrace$ 找出最佳的 $K$。

3. 訓練一個小型 draft。取一個 124M 的 GPT-2 target，用 KL loss 在 100M 個 token 上蒸餾一個 30M 的 GPT-2 draft。在保留的文字上量測 `α`。預期：0.6-0.7。

4. 實作 EAGLE 風格的樹狀草擬。不用鏈，而是讓 draft 在每個深度輸出 top-3 分支。建立 tree attention mask。驗證 target 接受最長的正確分支。

5. 量測失敗模式。在 temperature=1.5（高隨機性）下執行 speculative decode。展示 α 崩潰，且由於 draft 的開銷，演算法比單純 decode 還慢。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Target model | 「大模型」 | 你想要從中取樣的慢速、高品質模型（p 分布） |
| Draft model | 「推測者」 | 小型、快速的預測器（q 分布），小 5-30 倍 |
| K / draft length | 「向前看」 | 每次驗證 pass 所推測的 token 數 |
| α / acceptance rate | 「命中率」 | draft 的提議被接受的每個 token 機率 |
| Exact rejection rule | 「接受測試」 | 保留 target 分布的 r < p/q 比較 |
| Residual distribution | 「修正後的 p-q」 | (p - q)+ / ||(p - q)+||_1，拒絕時該從中抽樣的分布 |
| Tree drafting | 「分支式推測」 | Draft 輸出候選的樹，用樹狀結構的 attention mask 在一次 pass 中驗證 |
| Tree attention mask | 「拓撲 mask」 | 編碼樹拓撲的 causal mask，讓每個節點只 attend 到它的祖先 |
| Medusa heads | 「平行 head」 | target 本身額外的 K 個預測 head，不需要獨立的 draft model |
| EAGLE feature reuse | 「Hidden-state draft」 | Draft 的輸入是 target 最後的 hidden state，而不是原始 token，讓 draft 縮小 |
| Test-time simulation loss | 「EAGLE-3 訓練」 | 在與 target 測試時分布相符的輸出上訓練 draft，而不是 teacher forcing |

## 延伸閱讀 (Further Reading)

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192)：精確的拒絕規則與理論上的加速分析
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318)：DeepMind 同時期的 speculative-sampling 論文
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774)：取代 draft model 的平行 head 做法
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077)：特徵重用與樹狀草擬
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858)：動態樹拓撲
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840)：訓練時與測試時的匹配
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057)：Jacobi/lookahead decoding，不需要推測者的替代方案
