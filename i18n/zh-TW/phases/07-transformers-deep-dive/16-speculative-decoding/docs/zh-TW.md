# Speculative Decoding：Draft、Verify、重複 (Speculative Decoding — Draft, Verify, Repeat)

> Autoregressive decoding 是串行的。每個 token 都要等前一個。Speculative decoding 打斷了這條鏈：便宜的模型 draft N 個 token，昂貴的模型用一次 forward pass 驗證全部 N 個。當 draft 猜對時，你只付了一次大模型 forward，就換到 N 次生成。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 07 (GPT Causal LM), Phase 7 · 12 (KV Cache & Flash Attention)
**Time:** ~60 分鐘

## 問題 (The Problem)

一個 70B LLM 在 H100 上取樣一個 token 約需 30 ms。一個 3B 的 draft 模型約需 3 ms。如果我們讓 3B 先 draft 5 個 token，再執行 70B *一次*來驗證全部 5 個，總共是 $5 \times 3 + 30 = 45$ ms，最多可得到 5 個被接受的 token，相較之下直線式生成需要 `5×30 = 150 ms`。這就是 speculative decoding 完整的賣點：用少量額外的 GPU 記憶體（draft 模型），換取 2 到 4 倍更低的 decode 延遲。

這個技巧必須保持分布不變。Speculative sampling 由 Leviathan et al. (2023) 與 Chen et al. 同時提出，它保證輸出序列與大模型自己產生的結果**分布完全相同**。沒有品質上的取捨，只是更快。

2026 年的推論由四個 draft-verifier 家族主導：

1. **Vanilla speculative（Leviathan 2023）。** 獨立的 draft 模型（例如 Llama 3 1B）+ verifier（例如 Llama 3 70B）。
2. **Medusa（Cai 2024）。** Verifier 上的多個 decoding head 平行預測位置 `t+1..t+k`。不需要獨立的 draft 模型。
3. **EAGLE 家族（Li 2024、2025）。** 輕量的 draft，重複使用 verifier 的 hidden state；接受率比 vanilla 更高；典型為 3 到 4 倍。
4. **Lookahead decoding（Fu 2024）。** Jacobi 迭代；完全不需要 draft 模型。Self-speculation。小眾但不需要額外依賴。

2026 年每個正式上線的推論堆疊預設都附帶 speculative decoding。vLLM、TensorRT-LLM、SGLang 和 llama.cpp 至少都支援 vanilla + EAGLE-2。

## 核心概念 (The Concept)

### 核心演算法 (The core algorithm)

給定 verifier $M_q$ 與較便宜的 draft $M_p$：

1. 令 $x_1 \ldots x_k$ 為已經解碼完成的 prefix。
2. **Draft**：用 $M_p$ 以 autoregressive 方式提議 $d_{k+1}, d_{k+2}, \ldots, d_{k+N}$，其 draft 機率為 $p_1 \ldots p_N$。
3. **平行驗證**：對 $x_1 \ldots x_k, d_{k+1}, \ldots, d_{k+N}$ 執行一次 $M_q$，得到位置 $k+1 \ldots k+N+1$ 的 verifier 機率 $q_1 \ldots q_{N+1}$。
4. **由左至右接受 / 拒絕每個 draft token**：對每個 $i$，以機率 $\min(1, q_i(d_i) / p_i(d_i))$ 接受。
5. 在位置 $j$ 第一次被拒絕時：從經過正規化的「residual」分布 $(q_j - p_j)_+$ 取樣 $t_j$。$j$ 之後的所有 draft 都被丟棄。
6. 當全部 $N$ 個都被接受時：從 $q_{N+1}$ 額外取樣一個 token $t_{N+1}$（免費的 bonus token）。

Residual 分布的技巧，是讓輸出分布與 $M_q$ 從頭取樣完全一致的數學洞見。

### 什麼決定加速幅度 (What determines speedup)

令 $\alpha$ = 每個 draft token 的期望接受率。令 $c$ = draft 對 verifier 的成本比。每一步：

- 天真的生成每個 token 呼叫 1 次大模型。
- 當 $\alpha$ 很高時，speculative 每 $(1 - \alpha^{N+1}) / (1 - \alpha) \approx 1/(1-\alpha)$ 個 token 才呼叫 1 次大模型。

在 $\alpha = 0.75$、$N = 5$ 時的典型經驗法則：大模型呼叫次數少 3 倍。Draft 的成本便宜 5 倍。實際耗時總共降低約 2.5 倍。

**$\alpha$ 取決於：**

- Draft 對 verifier 的近似程度。同一家族 / 相同訓練資料會大幅提高 $\alpha$。
- Decoding 策略。Greedy draft 對上 greedy verifier：$\alpha$ 高。Temperature 取樣：較難匹配，接受率下降。
- 任務類型。程式碼和結構化輸出接受得較多（可預測）；自由創作的寫作接受得較少。

### Medusa：沒有 draft 模型的 draft (Medusa — drafts without a draft model)

Medusa 用 verifier 上額外的輸出 head 取代 draft 模型。在位置 `t`：

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

每個 head 輸出自己的 logit。推論時從每個 head 取樣得到候選序列，再用一次 forward pass 搭配 tree-attention 機制來驗證，該機制會一次考慮所有候選的後續延伸。

優點：不需要第二個模型。缺點：增加可訓練參數；需要一個 supervised fine-tuning 階段（約 1B 個 token）；接受率比搭配好 draft 的 vanilla speculative 略低。

### EAGLE：重複使用 hidden state 得到更好的 draft (EAGLE — better draft by reusing hidden states)

EAGLE-1/2/3（Li et al., 2024–2025）讓 draft 模型成為一個很小的 transformer（通常 1 層），吃進 verifier 最後一層的 hidden state。因為 draft 看得到 verifier 的特徵表示，它的預測與 verifier 的輸出分布高度相關。接受率從約 0.6（vanilla）攀升到 0.85 以上。這些是在聊天基準上、搭配以相符聊天資料訓練的 draft 所得到的每個 token 比率。在混合的正式流量上，Phase 17 · 05 對 EAGLE-3 的規劃是 0.6 到 0.8，而特定領域的流量可能更低。

EAGLE-3（2025）加入了對候選延伸的 tree search。vLLM 與 SGLang 把 EAGLE-2/3 作為 Llama 3/4 與 Qwen 3 的預設 spec 路徑。

### KV cache 之舞 (The KV cache dance)

驗證時會一次 forward pass 把 $N$ 個 draft token 餵進 verifier。這會讓 verifier 的 KV cache 增加 $N$ 筆項目。如果有些 draft 被拒絕，你必須把 cache 回滾到被接受的 prefix 長度。

正式的實作（vLLM 的 `--speculative-config`、TensorRT-LLM 的 LookaheadDecoder）用暫存的 KV buffer 處理這件事。先寫入，接受時才 commit。概念上不難，但很瑣碎。

```figure
draft-verify-tokens
```

## 動手做 (Build It)

請看 `code/main.py`。我們實作核心的 speculative-sampling 演算法（拒絕步驟 + residual 分布），包含：

- 一個「大模型」，是手寫分布上的確定性 softmax（這樣我們就能用解析方式驗證接受率的數學）。
- 一個「draft 模型」，是大模型的擾動版本。
- 一個接受 / 拒絕迴圈，產生的邊際分布與直接取樣相同。

### 步驟 1：拒絕步驟 (the rejection step)

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` 是均勻隨機數。`q_prob` 是 verifier 對被 draft 的 token 的機率。`p_prob` 是 draft 模型的機率。Leviathan 定理是：這個 Bernoulli 決策，加上拒絕時從 residual 取樣，可以精確保持 verifier 的分布。

### 步驟 2：Residual 分布 (residual distribution)

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

逐元素地從 `q` 減去 `p`，把負值截斷為零，再重新正規化。任何拒絕發生時就從這個分布取樣。

### 步驟 3：一個 speculative 步驟 (one speculative step)

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

五個被接受 → 一個 bonus → 一次 verifier pass 產生六個 token。

### 步驟 4：測量接受率 (measure acceptance rate)

在不同的 draft 品質等級下，執行 10,000 個 speculative 步驟。畫出接受率對 draft 與 verifier 分布之間 KL divergence 的圖。你應該會看到乾淨的單調關係。

### 步驟 5：驗證分布等價 (verify distribution equivalence)

實證上：speculative 迴圈產生的 token 直方圖，應該與直接從 verifier 取樣所產生的直方圖相符。這就是實務上的 Leviathan 定理。卡方檢定在取樣誤差範圍內確認了這點。

## 實際使用 (Use It)

正式環境：

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "eagle", "model": "/models/llama-3.1-eagle-70b", "draft_tensor_parallel_size": 1, "num_speculative_tokens": 5}'

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-config '{"method": "draft_model", "model": "meta-llama/Llama-3.2-1B-Instruct", "num_speculative_tokens": 5}'
```

截至 2026 年中，TensorRT-LLM 擁有最快的 Medusa 路徑。`faster-whisper` 為 Whisper-large 包裝了搭配小型 draft 的 speculative decoding。

**挑選 draft：**

| 策略 | 何時選擇 | 加速 |
|----------|--------------|---------|
| Vanilla draft（1B/3B Llama 家族） | 快速原型，不用訓練 | 1.8–2.3× |
| Medusa head | 你可以 fine-tune verifier | 2–3× |
| EAGLE-2 / 3 | 正式環境，追求最高速度 | 3–4× |
| Lookahead | 沒有 draft、不用訓練、沒有額外參數 | 1.3–1.6× |

**何時不要用 spec-decode：**

- 單一序列只生成 1 到 5 個 token。額外開銷會佔主導。
- 非常有創意 / 高 temperature 的取樣（$\alpha$ 會下降）。
- 記憶體受限的部署（draft 模型會增加 VRAM）。

## 交付 (Ship It)

請看 `outputs/skill-spec-decode-picker.md`。這個 skill 會為新的推論工作負載挑選 speculative decoding 策略（vanilla / Medusa / EAGLE / lookahead）與調校參數（N、draft temperature）。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。確認 speculative 的 token 分布，在 50,000 個 token 上與 verifier 直接取樣的分布相符，卡方檢定 p > 0.05。
2. **中等。** 對 $\alpha = 0.5, 0.7, 0.85$，畫出加速（每次大模型 forward 產生的 token 數）對 $N$ 的函式圖。找出每個 $\alpha$ 的最佳 $N$。（提示：每次 verify 呼叫的期望 token 數 = $(1 - \alpha^{N+1}) / (1 - \alpha)$。）
3. **困難。** 實作一個迷你 Medusa：拿第 14 課的總結專題 GPT，加上 3 個額外的 LM head 來預測位置 t+2、t+3、t+4。在 tinyshakespeare 上用聯合的 multi-head loss 訓練。與用同一模型截斷而成的 vanilla draft 比較接受率。
4. **困難。** 實作回滾：從 10 個 token 的 prefix KV cache 開始，餵入 5 個 draft token，模擬在位置 3 發生拒絕。驗證你在下一次迭代讀取的 cache，正確對應到「prefix + 前 2 個被接受的 draft」。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Draft model | 「便宜的那個」 | 提議候選 token 的較小模型；通常比 verifier 便宜 10 到 50 倍。 |
| Verifier | 「大的那個」 | 我們要保持其分布的目標模型；每個 speculative 步驟只執行一次。 |
| Acceptance rate ($\alpha$) | 「draft 猜對的頻率」 | Verifier 接受 draft 的每個 token 機率。典型為 0.7 到 0.9。 |
| Residual distribution | 「拒絕後的備案」 | 正規化後的 `(q - p)_+`；拒絕時從它取樣可以保持 verifier 的分布。 |
| Bonus token | 「免費的那個」 | 當全部 N 個 draft 都被接受時，再從 verifier 的下一步分布多取樣一個。 |
| Medusa | 「沒有 draft 的 speculative」 | Verifier 上的多個 LM head 平行預測位置 t+1..t+k。 |
| EAGLE | 「Hidden-state draft」 | 以 verifier 最後一層 hidden state 為條件的小型 transformer draft。 |
| Lookahead decoding | 「Jacobi 迭代」 | 使用固定點迭代的 self-speculation；沒有 draft 模型。 |
| Tree attention | 「一次驗證多個候選」 | 分支式驗證，同時考慮數個 draft 的後續延伸。 |
| KV rollback | 「復原被拒絕的 draft」 | 暫存的 KV buffer；接受時 commit，拒絕時丟棄。 |

## 延伸閱讀 (Further Reading)

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — 核心演算法與等價定理。
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — 同時期提出；乾淨的 Bernoulli 拒絕證明。
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — Medusa 論文；tree-attention 驗證。
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1；以 hidden state 為條件的 draft。
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2；動態的 tree 深度。
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3。
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead，不需要 draft 的做法。
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — 標準的正式環境參考，包含四種策略的完整接線。
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — EAGLE-1/2/3 的參考程式碼。
