# Mixture of Experts (MoE)

> Dense 70B 的 transformer 每個 token 都會啟動所有參數。671B 的 MoE 每個 token 只啟動 37B，卻在每個基準上勝過它。Sparsity 是這十年來最重要的 scaling 概念。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 05 (Full Transformer), Phase 7 · 07 (GPT)
**Time:** ~45 分鐘

## 問題 (The Problem)

Dense transformer 在推論時的 FLOPs 等於它的參數量（forward pass 要乘以 2）。把 dense 模型放大，每個 token 都要付全額帳單。到了 2024 年，前沿撞上了算力牆：想要顯著更聰明，每個 token 需要指數級更多的 FLOPs。

Mixture of Experts 打斷了這個連結。把每個 FFN 換成 $E$ 個獨立的 expert，加上一個為每個 token 挑出 $k$ 個 expert 的 router。總參數量 = $E \times \text{FFN size}$。每個 token 的啟動參數量 = $k \times \text{FFN size}$。2026 年的典型設定：$E=256$、$k=8$。儲存量隨 $E$ 成長，運算量隨 $k$ 成長。

2026 年的前沿幾乎全是 MoE：DeepSeek-V3（總共 671B / 啟動 37B）、Mixtral 8×22B、Qwen2.5-MoE、Llama 4、Kimi K2、gpt-oss。在 Artificial Analysis 的獨立排行榜上，前 10 名開源模型全是 MoE。

## 核心概念 (The Concept)

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### FFN 的替換 (The FFN swap)

Dense transformer block：

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block：

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

每個 expert 都是獨立的 FFN（通常是 SwiGLU）。Router 是單一線性層。每個 token 挑選自己的 `k` 個 expert，並得到它們輸出的 gated 混合。

### 負載平衡問題 (The load-balancing problem)

如果 router 把 90% 的 token 都送進 expert 3，其他 expert 就會吃不到東西。有人試過三種修正：

1. **Auxiliary load-balancing loss**（Switch Transformer、Mixtral）。加上一個與 expert 使用量變異數成正比的懲罰。有效，但多了一個超參數和第二個 gradient 訊號。
2. **Expert capacity + token dropping**（早期的 Switch）。每個 expert 最多處理 $C \times N/E$ 個 token；溢出的 token 跳過這一層。會傷害品質。
3. **Auxiliary-loss-free balancing**（DeepSeek-V3）。加上一個學習得來的、每個 expert 專屬的 bias，來移動 router 的 top-k 選擇。Bias 是在訓練 loss 之外更新的。對主要目標沒有懲罰。2024 年的重大突破。

DeepSeek-V3 的做法：每個訓練步驟之後，對每個 expert 檢查它的使用量高於還是低於目標。把 bias 微調 `±γ`。選擇時使用 `scores + bias`。用於 gating 的 expert 機率則是未經改動的原始 `scores`。這把 routing 與表達能力解耦了。

### Shared expert (Shared experts)

DeepSeek-V2/V3 還把 expert 分成 *shared* 與 *routed*。每個 token 都會經過所有 shared expert。Routed expert 則透過 top-k 挑選。Shared expert 捕捉共通知識；routed expert 負責專精。V3 執行 1 個 shared expert 加上 256 個 routed expert 中的 top-8。

### 細粒度 expert (Fine-grained experts)

經典 MoE（GShard、Switch）：每個 expert 的寬度跟完整的 FFN 一樣。`E` 很小（8 到 64），`k` 也很小（1 到 2）。

現代的細粒度 MoE（DeepSeek-V3、Qwen-MoE）：每個 expert 更窄（FFN 大小的 1/8）。$E$ 很大（256 以上），$k$ 也更大（8 以上）。總參數量相同，但組合數成長得快得多。每個 token 有 $\binom{256}{8} \approx 400$ 兆種可能的「expert」。品質提高，延遲不變。

### 成本概況 (The cost profile)

每個 token、每一層：

| 設定 | 每個 token 的啟動參數 | 總參數 |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 在幾乎每個基準上都勝過 Llama 3 70B（dense），而且**每個 token 的啟動 FLOPs 更少**。更多參數 = 更多知識。更多啟動 FLOPs = 每個 token 更多運算。MoE 把兩者解耦了。

### 陷阱：記憶體 (The catch: memory)

不論哪些 expert 被觸發，所有 expert 都放在 GPU 上。671B 的模型以 fp16 存權重，需要約 1.3 TB 的 VRAM。前沿 MoE 的部署需要 expert parallelism：把 expert 分片到各個 GPU，並在網路上路由 token。延遲主要由 all-to-all 通訊決定，而不是 matmul。

```figure
expert-routing
```

## 動手做 (Build It)

請看 `code/main.py`。一個用純標準函式庫寫的精簡 MoE 層，包含：

- `n_experts=8` 個類 SwiGLU 的 expert（為了說明，各只有一個線性層）
- top-k=2 的 routing
- 經 softmax 正規化的 gating 權重
- 透過每個 expert 專屬 bias 達成的 auxiliary-loss-free balancing

### 步驟 1：Router (the router)

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias 影響的是選擇，不是 gate 權重。這就是 DeepSeek-V3 的招數：bias 修正負載不平衡，卻不會左右模型的預測。

### 步驟 2：讓 100 個 token 通過 router (run 100 tokens through the router)

追蹤哪些 expert 被觸發了多少次。沒有 bias 的話，使用量會有偏斜。加上 bias 更新迴圈（過度使用的 expert 用 `-γ`，使用不足的用 `+γ`），使用量會在幾次迭代內收斂成均勻分布。

### 步驟 3：參數量比較 (param count comparison)

印出 MoE 設定的「dense 等效」。仿 DeepSeek-V3 的形狀：256 個 routed + 1 個 shared、啟動 8 個、d_model=7168。總參數量大得驚人。啟動的數量則是 dense Llama 3 70B 的七分之一。

## 實際使用 (Use It)

HuggingFace 載入：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

2026 年的正式推論：vLLM 原生支援 MoE routing。SGLang 擁有最快的 expert-parallel 路徑。兩者都會自動處理 top-k 選擇與 expert parallelism。

**何時選 MoE：**
- 你想用較低的每 token 推論成本得到前沿品質。
- 你有 VRAM / expert-parallel 的基礎設施。
- 你的工作負載偏重 token（聊天、程式碼），而不是偏重 context（長文件）。

**何時不要選 MoE：**
- 邊緣部署：不管啟動多少 FLOP，你都要付全額儲存空間。
- 對延遲要求嚴格的單一使用者服務：expert routing 會增加額外開銷。
- 小模型（<7B）：MoE 的品質優勢只在超過某個算力門檻（約 6B 啟動參數）之後才會出現。

## 交付 (Ship It)

請看 `outputs/skill-moe-configurator.md`。這個 skill 會根據參數預算、訓練 token 數與部署目標，為新的 MoE 挑選 E、k 與 shared-expert 的配置。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。觀察 auxiliary-loss-free 的 bias 更新如何在 50 次迭代內讓 expert 使用量趨於平均。
2. **中等。** 把學習得來的 router 換成基於 hash 的 router（確定性、不需學習）。比較品質與平衡程度。為什麼學習得來的 router 比較好？
3. **困難。** 實作 GRPO 風格的「rollout-matched routing」（DeepSeek-V3.2 的招數）：記錄推論期間哪些 expert 被觸發，在計算 gradient 時強制使用相同的 routing。在玩具 policy-gradient 設定上測量其效果。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Expert | 「眾多 FFN 之一」 | 獨立的 feed-forward network；其參數專門負責 FFN 運算中的一個稀疏切片。 |
| Router | 「那個閘門」 | 一個小型線性層，替每個 token 對每個 expert 打分；做 top-k 選擇。 |
| Top-k routing | 「每個 token 有 k 個啟動的 expert」 | 每個 token 的 FFN 運算恰好經過 k 個 expert，並以 gate 加權。 |
| Auxiliary loss | 「負載平衡懲罰」 | 額外的 loss 項，用來懲罰 expert 使用量的偏斜。 |
| Auxiliary-loss-free | 「DeepSeek-V3 的招數」 | 只透過 router 選擇上的每個 expert 專屬 bias 來平衡；沒有額外的 gradient。 |
| Shared expert | 「永遠啟動」 | 每個 token 都會經過的額外 expert；捕捉共通知識。 |
| Expert parallelism | 「依 expert 分片」 | 把不同的 expert 分配到不同的 GPU；在網路上路由 token。 |
| Sparsity | 「啟動參數 < 總參數」 | 比例 $k \times \text{expert size} / (E \times \text{expert size})$；DeepSeek-V3 是 37/671 ≈ 5.5%。 |

## 延伸閱讀 (Further Reading)

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — 這個概念的起源。
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch，經典的 MoE。
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B。
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP。
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — 基於 bias 的平衡論文。
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — 本課 router 所使用的細粒度 + shared-expert 切分。
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — 最初的 shared-expert 論文。
