# Speculative Decoding 與 EAGLE-3 (Speculative Decoding and EAGLE-3)

> Phase 7 · Lesson 16 證明了數學：Leviathan rejection rule 能完全保留 verifier 的分布。這堂課是 2026 年正式環境 speculative decoding 的訓練堆疊視角。EAGLE-3 把 draft model 從廉價的近似，變成一個專門打造、在 verifier 自己的 hidden state 上訓練的小型網路，再加上一個訓練時測試 (training-time test) 迴圈，讓它的訓練與推論分布對齊。結果：端到端加速 3 倍到 6.5 倍，聊天場景的單 token 接受率超過 0.9，而且在分布上沒有任何取捨。2026 年每個正式的推論堆疊都預設內建它。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 7 · 16 (speculative decoding math), Phase 10 · 12 (inference optimization)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 用一句話陳述 Leviathan 定理，並證明 speculative 迴圈產生的樣本，與 verifier 的分布完全相同。
- 走過從原始 spec-decoding (Leviathan 2023) 到 EAGLE、EAGLE-2、EAGLE-3 的兩年演進，並指出每一步移除的確切限制。
- 從接受率 `α` 與 draft 對 verifier 的成本比 `c`，計算預期加速，並為每種情況選出最佳的 draft 長度 `N`。
- 從零實作完整的 speculative 迴圈：draft、verify、從 residual 做 reject-sample、拒絕時回滾 KV cache、完全接受時輸出 bonus token。

## 問題 (The Problem)

70B 模型的自迴歸 decoding，在 H100 上大概只有每秒 35 個 token。GPU 遠遠沒有飽和。記憶體頻寬才是天花板：每個 token 都要從 HBM 載入 70B 的權重，做一步運算，然後產生一個 float。運算單元大部分時間都閒著。

Speculative decoding 把這個問題變成一個你真的能解決的吞吐量問題。一個廉價的 draft 用 `N` 次小型 forward pass 提出 `N` 個 token。Verifier 對前綴加上全部 `N` 個 draft，只執行一次。如果 verifier 在位置 `i` 的分布與 draft 一致（以我們即將精確定義的統計意義而言），我們就接受；如果不一致，就拒絕，並從 residual 分布中抽樣出修正。一次大模型的 forward，最多可以產生 `N+1` 個被接受的 token，而不是一個。

重要的定理是 Leviathan、Kalman、Matias (ICML 2023)：輸出的分布與直接從 verifier 抽樣所得到的完全相同。不是近似，是完全相同。這就是 speculative decoding 能被接受用於正式環境的全部原因，它是純粹的延遲最佳化，沒有品質上的取捨。

Phase 7 · Lesson 16 給你的是數學。這堂課給你的是訓練堆疊。一個好的 draft，比一個廉價的 draft 值 2 倍以上的加速。EAGLE、EAGLE-2 和 EAGLE-3 (Li et al., 2024–2025) 把「draft = 同一個模型的縮小版」變成一門精確的工程學。2026 年的正式推論伺服器預設使用 EAGLE-3。

## 核心概念 (The Concept)

### 不變式：Leviathan rejection sampling (The invariant: Leviathan rejection sampling)

令 `p(t)` 為 draft 對給定前綴後下一個 token 的分布，`q(t)` 為 verifier 的分布。從 `p` 抽樣一個 draft token `d ~ p`。以 `min(1, q(d) / p(d))` 的機率接受。若拒絕，則從 residual 分布 `(q - p)_+ / ||(q - p)_+||_1` 抽樣。得到的樣本服從 `q` 分布。不論 `p` 有多糟，這都成立：它越糟，你拒絕的次數越多，但輸出仍然是精確的。

用一次 verifier 對 `prefix + d_1 + ... + d_N` 的 forward pass，把 `N` 次這樣的呼叫連續堆疊起來。Verifier 同時回傳 `q_1, q_2, ..., q_{N+1}`。由左至右走。在位置 `j` 第一次被拒絕時，從 `residual(q_j, p_j)` 抽樣並停止。若完全接受，則從 `q_{N+1}` 抽樣一個 bonus token。

### 什麼決定加速幅度 (What determines speedup)

令 `α` 為每個 draft token 的預期接受率。令 `c = cost(draft) / cost(verifier)` 為成本比。每次 verifier forward 預期被接受的 token 數為：

```math
\mathbb{E}[\text{accepted}] = \frac{1 - \alpha^{N+1}}{1 - \alpha}
```

每個被接受 token 的預期總實際耗時是 `(N * c + 1) / E[accepted]`。針對 `N` 最小化它，你就得到最佳點。對於 `α = 0.8, c = 0.05`：最佳 `N` 大約是 5–7，加速是 3.2 倍。對於 `α = 0.95, c = 0.02`：最佳 `N` 大約是 8–10，加速逼近 5 倍。

最大的單一槓桿是 `α`。在固定 `N = 5` 下，從 `α = 0.6`（原始 draft）到 `α = 0.9`（EAGLE-3），每次 verifier forward 預期被接受的 token 數從 2.2 增加到 4.1。用同一個 verifier，吞吐量接近 2 倍。

### 兩年的演進 (The two-year progression)

**原始 speculative (Leviathan, 2023)。** Draft model 是來自同一家族、獨立訓練的較小 LLM。容易接線，`α ≈ 0.6`，最多約 2 倍加速。

**EAGLE-1 (Li et al., 2024)。** Draft 是一個小型 transformer（通常一到兩層），它以 verifier 的最後一層 hidden state 為輸入，直接預測下一個 token。因為 draft 看得到 verifier 的特徵表示，它的分布更接近 verifier 的。`α` 攀升到 0.7–0.8。

**EAGLE-2 (Li et al., 2024)。** 加入動態 draft tree：不是提出單一條 `N` 個 token 的序列，而是提出一棵小小的候選樹，用 verifier 在一次 forward pass 中（tree attention）為每個候選評分，並走過機率最高的路徑。Draft 長度在每一步變成自適應。被接受路徑上每個 token 的 `α` 攀升到 0.85 以上。

**EAGLE-3 (Li et al., 2025, NeurIPS)。** 再加兩項改變。第一，完全拿掉 feature-prediction loss：EAGLE-1/2 訓練 draft 去匹配 verifier 的 hidden state，這限制了資料量所能帶來的幫助。EAGLE-3 直接用 token prediction 訓練。第二，training-time test (TTT)：在 draft 訓練期間，跨多個步驟把 draft 自己先前的預測，當作輸入餵回去，就像它在推論時的運作方式。這讓訓練與測試的分布對齊，並阻止誤差累積。實測加速：聊天場景最高 6.5 倍，在 H100 上的 SGLang 中，batch 64 的吞吐量提升 38%。

### KV cache 回滾 (KV cache rollback)

Verification 一次 pass 就會把 verifier 的 KV cache 延伸 `N` 個項目。如果在位置 `j` 發生拒絕，cache 中位置 `j-1` 之後的內容就是錯的。兩種常見的實作：寫入暫存 buffer，並在接受時提交 (commit)（vLLM、TensorRT-LLM），或是保留一個實體 KV cache 加上一個邏輯長度，並在拒絕時截斷。不論哪一種，回滾的成本都是每層每個 head 的 byte 數，相較於 forward pass 的成本可以忽略。

對於 EAGLE-2 的樹狀搜尋，verifier 以尊重樹拓撲的非因果 mask 執行 attention。工程上很繁瑣，但運算本身就是一次帶自訂 mask 的標準 flash-attention 呼叫。

### 2026 年的 draft 架構 (Draft architectures in 2026)

| 策略 | Draft 類型 | `α` | 加速 | 訓練成本 |
|----------|-----------|-----|---------|---------------|
| Vanilla | 獨立的小型 LLM | 0.55-0.70 | 1.8-2.3× | 無（重複使用現有小模型） |
| Medusa | verifier 上額外的 LM head | 0.65-0.75 | 2-3× | 約 1B SFT token |
| EAGLE-1 | hidden state 上的 1 層 transformer | 0.70-0.80 | 2.5-3× | 約 60B token |
| EAGLE-2 | EAGLE-1 + 動態 draft tree | 0.80-0.88 | 3-4× | 約 60B token |
| EAGLE-3 | 多層特徵融合 + TTT | 0.88-0.92 | 3.5-6.5× | 約 60-200B token |
| Lookahead | 無 draft（Jacobi 迭代） | N/A | 1.3-1.6× | 無 |

`α` 這一欄是在聊天 benchmark 上、使用以相符聊天資料訓練的 draft，沿著被接受路徑的每 token 接受率。它是上限區間。Phase 10 · 25 引用了來自不同設定的 EAGLE-3 的 0.82，而 Phase 17 · 05 規劃正式上線時，在一般聊天上採用 0.6 到 0.8。在選擇 `N` 之前，先在你自己的流量上量測 `α`。

2026 年的正式環境：vLLM 與 SGLang 在可用時預設使用 EAGLE-3，否則用 EAGLE-2。TensorRT-LLM 針對 Meta 與 NVIDIA 的公開模型，有最快的 Medusa 路徑。llama.cpp 為 CPU 部署內建原始 draft。

```figure
l5-spec-decode-eagle
```

## 動手做 (Build It)

請見 `code/main.py`。這是完整的 Leviathan speculative 迴圈，包含所有部件：N 個 token 的 draft、verifier 平行 pass、每個位置的拒絕、residual 抽樣、bonus token、KV 回滾，以及經驗上驗證輸出分布與直接從 `q` 抽樣相符。

### 步驟 1：拒絕規則 (Step 1: the rejection rule)

```python
def accept(q_prob, p_prob, u):
    if p_prob <= 0:
        return True
    return u < min(1.0, q_prob / p_prob)
```

### 步驟 2：Residual 分布 (Step 2: residual distribution)

```python
def residual(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    if s == 0:
        return list(q)
    return [r / s for r in raw]
```

### 步驟 3：完整的 speculative 步驟 (Step 3: a full speculative step)

`spec_step` 函式從 `p` 草擬 `N` 個 token，然後用一次平行的 `q` 評估驗證全部的 token。對每個被草擬的 token，它套用拒絕規則，並在第一次拒絕時，從 residual 抽樣出修正。如果全部都接受，它會從 `q_{N+1}` 輸出一個 bonus token。

### 步驟 4：KV 回滾的記帳 (Step 4: KV rollback bookkeeping)

模擬器為每個 worker 追蹤一個邏輯的 `kv_length`。接受 `k` 個 draft 時，`kv_length += k`。在位置 `j` 被拒絕時，cache 其實已經寫到 `j` 之後了，但邏輯長度被設為 `prefix_length + j + 1`，也就是修正 token 之後一格。之後的讀取會截斷到邏輯長度。

### 步驟 5：Leviathan 檢驗 (Step 5: the Leviathan check)

執行 50,000 次 speculative 步驟。計算被接受 token 的經驗分布。與直接從 `q` 抽樣的 50,000 個樣本比較。Chi-square 統計量應遠低於臨界值。這個定理在實務上通過了。

### 步驟 6：加速對 α (Step 6: speedup vs. α)

以不同的幅度，把 `p` 從 `q` 擾動開來，掃過 draft 品質。量測 `α`，然後畫出每次 verifier 呼叫的預期 token 數，作為 `α` 與 `N` 的函數。程式碼會印出一張表，顯示 EAGLE-3 等級的 draft 品質 (`α ≈ 0.9`) 如何解鎖每次 verifier 呼叫 4–5 個 token。

## 實際使用 (Use It)

正式等級、搭配 EAGLE-3 的 `vllm serve`：

```bash
vllm serve meta-llama/Llama-3.3-70B-Instruct \
  --speculative-config '{
    "model": "yuhuili/EAGLE3-LLaMA3.3-Instruct-70B",
    "num_speculative_tokens": 5,
    "method": "eagle3"
  }'
```

在 H100 上以 batch 64 使用 SGLang 搭配 EAGLE-3：根據 EAGLE-3 論文，吞吐量約為 batch 64 原始 decoding 的 1.38 倍。

何時該使用 speculative decoding：

- 任何 p50 延遲比峰值吞吐量更重要的互動式聊天工作負載。
- 程式碼生成與結構化輸出（JSON、SQL）。`α` 高於 0.9，因為目標分布高度可預測。
- 長篇生成（數千個 token）。攤提後的加速會持續回報。

何時不該用：

- 非常小的模型（< 3B）。Draft 並沒有比 verifier 便宜多少。
- 極小的 batch-1 CPU 部署。Draft model 的記憶體開銷可能不值得。
- `α` 崩潰的極高 temperature 創意取樣。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-eagle3-tuner.md`。給定一個推論工作負載（模型、batch size、目標延遲、任務輪廓），它會推薦 speculative-decoding 策略與調整參數（draft 家族、`N`、樹深度、依 temperature 切換）。

## 練習 (Exercises)

1. 執行 `code/main.py`。確認 Leviathan 分布檢查的 chi-square 統計量，在 50,000 個樣本上維持在 95% 臨界值之下。

2. 在 `α` 固定為 0.9、`c` 固定為 0.04 的情況下，掃過 `N` 從 1 到 10。畫出每次 verifier 呼叫的預期 token 數，以及每個 token 實際的耗時。找出使耗時最小的 `N`。解釋這條曲線的形狀。

3. 修改程式碼，模擬 EAGLE-2 的樹狀搜尋：在每一步，draft 提出形狀為 `[2, 2, 2]` 的樹（八條候選路徑）。Verifier 執行一次，機率最高且被接受的路徑勝出。計算每個葉節點的 `α` 以及每次 verifier 呼叫的總 token 數。在等同運算量下，與線性鏈的 spec-decoding 比較。

4. 為兩條並行序列實作批次 KV 回滾模擬器。序列 A 的所有 draft 都被接受；序列 B 在位置 2 被拒絕。展示每條序列正確更新了 `kv_length`，且沒有浪費任何工作。

5. 閱讀 EAGLE-3 論文的第 4 節 (Training-Time Test)。用兩句話解釋，為什麼沒有 TTT 的單純 draft 訓練會有 exposure bias，以及為什麼在訓練期間餵給 draft 它自己的預測可以修正這個問題。把這與 seq2seq 中 scheduled-sampling 的文獻連結起來。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Leviathan rule | 「min(1, q over p)」 | 以 `min(1, q(d)/p(d))` 的機率做 Bernoulli 接受/拒絕，當你在拒絕時從 residual 抽樣，就能完全保留 verifier 的分布 |
| Residual distribution | 「(q minus p) plus，再正規化」 | 把 `(q - p)_+` 在零處截斷並重新正規化，是拒絕時該從中抽樣的正確分布 |
| Acceptance rate α | 「draft 猜對的頻率」 | 在拒絕規則下，每個 token 的預期 Bernoulli 成功機率，主導所有加速的數學 |
| EAGLE-1 | 「hidden-state draft」 | 以 verifier 最後一層 hidden state 為條件的小型 transformer draft (Li et al., 2024) |
| EAGLE-2 | 「動態 draft tree」 | EAGLE-1 加上一棵候選接續的樹，在一次 verifier pass 中以 tree attention 評分 |
| EAGLE-3 | 「training-time test」 | 拿掉 feature-prediction loss，以直接的 token prediction 訓練，並在訓練期間讓 draft 吃它自己的輸出 |
| Training-time test (TTT) | 「exposure bias 的修正」 | 在訓練期間以自迴歸方式執行 draft，使訓練與測試的輸入分布相符，是 scheduled sampling 的直接類比 |
| KV rollback | 「撤銷被拒絕的 draft」 | 在發生拒絕後，將 verifier 的 KV cache 重設到被接受前綴長度的記帳機制 |
| Bonus token | 「免費的那個」 | 當全部 `N` 個 draft 都被接受時，從 `q_{N+1}` 額外抽樣一個，不需額外的 verifier 成本 |
| Tree attention | 「一次驗證許多候選」 | 帶有尊重 draft tree 拓撲的非因果 mask 的 attention，在一次 forward pass 中為樹裡每個節點算出 `q_i` |

## 延伸閱讀 (Further Reading)

- [Leviathan, Kalman, Matias — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192, ICML 2023)](https://arxiv.org/abs/2211.17192)：奠基性論文與等價定理
- [Chen et al. — Accelerating Large Language Model Decoding with Speculative Sampling (arXiv:2302.01318)](https://arxiv.org/abs/2302.01318)：同時期獨立提出，附有簡潔的證明
- [Li et al. — EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077)：EAGLE-1，以 hidden state 為條件的 draft
- [Li et al. — EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858)：動態樹狀搜尋
- [Li et al. — EAGLE-3: Scaling up Inference Acceleration via Training-Time Test (arXiv:2503.01840, NeurIPS 2025)](https://arxiv.org/abs/2503.01840)：2026 年的正式環境預設
- [Cai et al. — Medusa: Multiple Decoding Heads (arXiv:2401.10774)](https://arxiv.org/abs/2401.10774)：另一種不需 draft 的做法
- [vLLM Speculative Decoding documentation](https://docs.vllm.ai/en/latest/features/spec_decode.html)：所有策略都已接好的標準正式環境參考資料
