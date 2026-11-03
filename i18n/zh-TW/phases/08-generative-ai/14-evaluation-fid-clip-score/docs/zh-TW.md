# 評估：FID、CLIP Score、人類偏好 (Evaluation — FID, CLIP Score, Human Preference)

> 每個生成模型的排行榜，引用的都是 FID、CLIP score，以及來自人類偏好競技場的勝率。每個數字都有一個失效模式，是決心夠強的研究者可以拿來作弊的。如果你不知道這些失效模式，你就分不出真正的進步和刷分的結果。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 8 · 01 (Taxonomy), Phase 2 · 04 (Evaluation Metrics)
**Time:** ~45 分鐘

## 問題 (The Problem)

生成模型是以 *樣本品質* 和 *條件遵循度* 來評判的。兩者都沒有封閉形式的量測方式。你的模型必須渲染 10,000 張圖像；必須有東西替它們打分數；而你得相信這些分數能跨模型家族、跨解析度、跨架構。在 2014-2026 年的考驗中，存活下來的有三個指標：

- **FID (Fréchet Inception Distance)。** 在 Inception 網路的特徵空間中，真實與生成兩個分佈之間的距離。越低越好。
- **CLIP score。** 生成圖像的 CLIP-image embedding 與 prompt 的 CLIP-text embedding 之間的 cosine similarity。越高越好。量測 prompt 遵循度。
- **人類偏好。** 讓兩個模型對同一個 prompt 正面對決，由人類（或 GPT-4 等級的模型）選出較好的一個，再彙整成 Elo 分數。

你還會看到：IS（inception score，已大致被淘汰）、KID、CMMD、ImageReward、PickScore、HPSv2、MJHQ-30k。每一個都修正了前一個的某個失效。

## 核心概念 (The Concept)

![FID、CLIP 與偏好：三個軸，不同的失效模式](../assets/evaluation.svg)

### FID：樣本品質

Heusel et al. (2017)。步驟：

1. 為 N 張真實圖像和 N 張生成圖像擷取 Inception-v3 特徵（2048-D）。
2. 對每一池擬合一個 Gaussian：計算平均值 $\mu_r, \mu_g$ 和共變異數 $\Sigma_r, \Sigma_g$。
3. $\text{FID} = \lVert \mu_r - \mu_g \rVert^2 + \mathop{\mathrm{Tr}}\left(\Sigma_r + \Sigma_g - 2 \cdot (\Sigma_r \cdot \Sigma_g)^{0.5}\right)$。

詮釋：特徵空間中兩個多變量 Gaussian 之間的 Fréchet distance。越低 = 分佈越相似。

失效模式：
- **小 N 時有偏誤。** FID 是對特徵分佈做均方，N 小會低估共變異數，給出偽低的 FID。一律使用 $N \ge 10{,}000$。
- **依賴 Inception。** Inception-v3 是在 ImageNet 上訓練的。離 ImageNet 很遠的領域（人臉、藝術、文字圖像）會產生沒有意義的 FID。請使用該領域專用的特徵擷取器。
- **刷分。** 過度擬合 Inception 的先驗，可以在視覺品質沒有改善的情況下得到低 FID。用 CMMD（見下）來破解。

### CLIP score：prompt 遵循度

Radford et al. (2021)。對一張生成圖像 + prompt：

```
clip_score = cos_sim( CLIP_image(x_gen), CLIP_text(prompt) )
```

對 30k 張生成圖像取平均 → 得到一個可在模型間比較的純量。

失效模式：
- **CLIP 本身的盲點。** CLIP 的組合推理能力很弱（「紅色立方體放在藍色球體上」常常失敗）。模型可能在 CLIP score 上排名很好，卻沒有真的遵循複雜的 prompt。
- **偏好短 prompt。** 短 prompt 在真實世界中有更多的 CLIP-image 匹配。較長的 prompt 在機制上會有較低的 CLIP score。
- **玩弄 prompt。** 在 prompt 中加入「high quality, 4k, masterpiece」，會灌高 CLIP score，卻沒有改善圖像與文字的綁定。

CMMD (Jayasumana et al., 2024) 修正了其中一些問題：用 CLIP 特徵取代 Inception，用 maximum-mean discrepancy 取代 Fréchet。更擅長偵測細微的品質差異。

### 人類偏好：真正的基準

挑一組 prompt。用模型 A 和模型 B 生成。把成對的結果給人類（或強大的 LLM 評審）看。把勝負彙整成 Elo 或 Bradley-Terry 分數。基準：

- **PartiPrompts (Google)**：1,600 個多元的 prompt，12 個類別。
- **HPSv2**：10.7 萬筆人類標註，被廣泛用作自動化的代理指標。
- **ImageReward**：13.7 萬組 prompt-圖像偏好配對，MIT 授權。
- **PickScore**：在 Pick-a-Pic 的 260 萬筆偏好上訓練。
- **Chatbot-Arena 風格的圖像競技場**：https://imagearena.ai/ 等。

失效模式：
- **評審的變異。** 非專家與專家的偏好不同。兩者都要用。
- **Prompt 分佈。** 刻意挑選的 prompt 會偏袒某一個家族。一定要記錄下來。
- **LLM 評審的 reward hacking。** GPT-4 評審會被漂亮但錯誤的輸出騙過。要與人類互相對照驗證。

## 合併使用 (Use together)

一份正式環境的評估報告應該包含：

1. 對保留的真實分佈，在 10-30k 個樣本上的 FID（樣本品質）。
2. 在相同樣本與其 prompt 上的 CLIP score / CMMD（遵循度）。
3. 與前一個模型在盲測競技場中的勝率（整體偏好）。
4. 失效模式分析：隨機抽 50 個輸出，標記已知問題（手部解剖、文字渲染、物件數量一致性）。

任何單一指標都是謊言。三個互相印證的指標 + 質性檢視，才算一個主張。

```figure
gx-fid-distributions
```

## 動手做 (Build It)

`code/main.py` 在合成的「特徵向量」上實作 FID、類 CLIP score，以及 Elo 彙整（我們用 4-D 向量來代替 Inception 特徵）。你會看到：

- 在小 N 與大 N 下計算 FID，看到偏誤。
- 把「CLIP score」做成特徵池之間的 cosine similarity。
- 從合成的偏好串流做 Elo 更新規則。

### 步驟 1：四行搞定 FID

```python
def fid(real_features, gen_features):
    mu_r, cov_r = mean_and_cov(real_features)
    mu_g, cov_g = mean_and_cov(gen_features)
    mean_diff = sum((a - b) ** 2 for a, b in zip(mu_r, mu_g))
    trace_term = trace(cov_r) + trace(cov_g) - 2 * sqrt_cov_product(cov_r, cov_g)
    return mean_diff + trace_term
```

### 步驟 2：CLIP 風格的 cosine similarity

```python
def clip_like(image_feat, text_feat):
    dot = sum(a * b for a, b in zip(image_feat, text_feat))
    norm = math.sqrt(dot_self(image_feat) * dot_self(text_feat))
    return dot / max(norm, 1e-8)
```

### 步驟 3：Elo 彙整

```python
def elo_update(r_a, r_b, winner, k=32):
    expected_a = 1 / (1 + 10 ** ((r_b - r_a) / 400))
    actual_a = 1.0 if winner == "a" else 0.0
    r_a_new = r_a + k * (actual_a - expected_a)
    r_b_new = r_b - k * (actual_a - expected_a)
    return r_a_new, r_b_new
```

## 常見陷阱 (Pitfalls)

- **N=1000 的 FID。** 在 N=10k 以下，這個啟發式方法並不可靠。回報低 N 的 FID 的論文就是在刷分。
- **跨解析度比較 FID。** Inception 的 299×299 縮放會改變特徵分佈。只在相同解析度下比較。
- **只回報一個 seed。** 至少跑 3 個 seed。回報標準差。
- **透過 negative prompt 灌高 CLIP score。** 有些管線透過過度擬合 prompt 來提高 CLIP。要檢查視覺是否飽和。
- **Prompt 重疊造成的 Elo 偏誤。** 如果兩個模型在訓練期間都看過某個基準的 prompt，Elo 就沒有意義。使用保留的 prompt 集合。
- **付費群眾的人類評估偏差。** Prolific、MTurk 的標註者偏年輕、偏愛科技。要混入招募來的藝術/設計專家。

## 實際使用 (Use It)

2026 年的正式環境評估流程：

| 支柱 | 最低要求 | 建議做法 |
|------|---------|---------|
| 樣本品質 | 對保留的真實資料在 10k 上算 FID | + 5k 上的 CMMD + 每個類別子集的 FID |
| Prompt 遵循度 | 30k 上的 CLIP score | + HPSv2 + ImageReward + VQA 風格的問答 |
| 偏好 | 對基準線的 200 組盲測配對 | + 2000 組人類配對 + LLM 評審 + Chatbot Arena |
| 失效分析 | 50 個手動標記 | 500 個手動標記 + 自動化安全分類器 |

四大支柱都在同一份報告中 = 主張。任何一個單獨出現 = 行銷。

## 交付 (Ship It)

存成 `outputs/skill-eval-report.md`。這個 skill 接收新的模型 checkpoint + 基準線，輸出完整的評估計畫：樣本數量、指標、失效模式探測、核可標準。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。在相同的合成分佈上，比較 N=100 與 N=1000 的 FID。回報偏誤的大小。
2. **中等。** 從合成的 CLIP 風格特徵實作 CMMD（公式見 Jayasumana et al., 2024）。與 FID 比較對品質差異的敏感度。
3. **困難。** 重現 HPSv2 的設定：從 Pick-a-Pic 的子集取 1000 組圖像-prompt 配對，在這些偏好上 fine-tune 一個小型的 CLIP 評分器，並量測它與保留集合的一致程度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|---------|---------|
| FID | 「Fréchet Inception Distance」 | 對真實與生成的 Inception 特徵各做 Gaussian 擬合後的 Fréchet distance。 |
| CLIP score | 「文字-圖像相似度」 | CLIP 圖像與文字 embedding 之間的 cosine similarity。 |
| CMMD | 「FID 的替代品」 | 基於 CLIP 特徵的 MMD；偏誤較小，不需要 Gaussian 假設。 |
| IS | 「Inception score」 | Exp KL(p(y|x) || p(y))；在現代模型上相關性很差，已被淘汰。 |
| HPSv2 / ImageReward / PickScore | 「學出來的偏好代理指標」 | 以人類偏好訓練的小型模型；用作自動評審。 |
| Elo | 「棋類評級」 | 對成對勝負做 Bradley-Terry 彙整。 |
| PartiPrompts | 「基準 prompt 集」 | Google 精選的 1,600 個 prompt，涵蓋 12 個類別。 |
| FD-DINO | 「自監督的替代品」 | 使用 DINOv2 特徵的 FD；對 ImageNet 之外的領域更好。 |

## 實務筆記 (Production note)：評估本身也是推論工作負載

對 10k 個樣本跑 FID，代表要生成 10k 張圖像。對單張 L4 上 1024² 的 50 步 SDXL base 來說，單一請求的推論大約要 11 小時。評估預算是真實存在的，而且框架正好就是離線推論的情境（最大化吞吐量，忽略 TTFT）：

- **用力 batch，忘掉延遲。** 離線評估 = 用記憶體放得下的最大尺寸做靜態 batching。在 80GB H100 上使用 `num_images_per_prompt=8` 的 `pipe(...).images`，實際耗時比單一請求快 4 到 6 倍。
- **快取真實資料的特徵。** 對真實參考集做 Inception（FID）或 CLIP（CLIP-score、CMMD）的特徵擷取，只要 *跑一次*，存成 `.npz`。不要每次評估都重算。

對於 CI / 回歸關卡：每個 PR 在 500 個樣本的子集上跑 FID + CLIP score（約 30 分鐘）；每晚則跑完整的 10k FID + HPSv2 + Elo。

## 延伸閱讀 (Further Reading)

- [Heusel et al. (2017). GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium (FID)](https://arxiv.org/abs/1706.08500) — FID 論文。
- [Jayasumana et al. (2024). Rethinking FID: Towards a Better Evaluation Metric for Image Generation (CMMD)](https://arxiv.org/abs/2401.09603) — CMMD。
- [Radford et al. (2021). Learning Transferable Visual Models from Natural Language Supervision (CLIP)](https://arxiv.org/abs/2103.00020) — CLIP。
- [Wu et al. (2023). HPSv2: A Comprehensive Human Preference Score](https://arxiv.org/abs/2306.09341) — HPSv2。
- [Xu et al. (2023). ImageReward: Learning and Evaluating Human Preferences for Text-to-Image Generation](https://arxiv.org/abs/2304.05977) — ImageReward。
- [Yu et al. (2023). Scaling Autoregressive Models for Content-Rich Text-to-Image Generation (Parti + PartiPrompts)](https://arxiv.org/abs/2206.10789) — PartiPrompts。
- [Stein et al. (2023). Exposing flaws of generative model evaluation metrics](https://arxiv.org/abs/2306.04675) — 失效模式調查。
