# Qwen-VL 家族與 Dynamic-FPS 影片 (Qwen-VL Family and Dynamic-FPS Video)

> Qwen-VL 家族（Qwen-VL (2023)、Qwen2-VL (2024)、Qwen2.5-VL (2025)、Qwen3-VL (2025)）是 2026 年最具影響力的開源 vision-language 模型系譜。每一代都下了一個決定性的架構賭注，其餘開源生態系在十二個月內紛紛跟進：透過 M-RoPE 的原生動態解析度、搭配絕對時間對齊的 dynamic-FPS 取樣、ViT 中的 window attention，以及結構化的 agent 輸出格式。到了 Qwen3-VL，配方已經穩定：使用原生長寬比輸入的 2D-RoPE-ViT encoder、接入大型 Qwen3 語言基底的 MLP projector，以及把 OCR、grounding 和 agent 行為視為第一級目標的訓練階段。這堂課依時間順序讀完整個家族，讓你理解每個旋鈕為什麼在現在的位置。

**Type:** Learn
**Languages:** Python (stdlib, M-RoPE encoder + dynamic-FPS sampler)
**Prerequisites:** Phase 12 · 06 (patch-n'-pack)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 計算 M-RoPE 的三軸旋轉（時間、高度、寬度），並解釋為什麼三者都需要。
- 為一段影片挑選 dynamic-FPS 取樣策略，並推論每秒 token 數與事件偵測準確率之間的取捨。
- 依序說出 Qwen-VL 的四次世代升級，以及每次解鎖了什麼。
- 串接 Qwen2.5-VL 風格的 JSON agent 輸出格式，並從 VLM 回應中解析結構化的 tool call。

## 問題 (The Problem)

Qwen-VL 在 2023 年 8 月推出，是對 LLaVA-1.5 和 BLIP-2 的直接回應。Qwen 團隊瞄準的差距有三方面：解析度、影片與結構化輸出。

解析度：LLaVA-1.5 以 336x336 運行。拍照片還行，對中文發票或密集的試算表截圖就沒用了。Qwen-VL 的第一個創新是 448x448 和 grounded 的 bounding box 輸出，讓模型可以指出東西的位置。

影片：Video-LLaMA 把逐影格的 encoder 堆疊起來餵給 LLM。對短片段有效，對時間軸本身就是訊號的多分鐘影片則不行。Qwen 團隊想要一個理解時間的單一 encoder。

結構化輸出：LLaVA 輸出的是自由格式文字。agent 需要 JSON。Qwen-VL 用明確的 JSON 輸出格式訓練，包括以文字表示的 bounding box 座標。

每一代 Qwen-VL 都在延伸這三條軸線之一。

## 核心概念 (The Concept)

### Qwen-VL (2023 年 8 月)

第一代：以 OpenCLIP ViT-bigG/14 作為 encoder（25 億參數）、與 LLama 相容的 Q-Former（單步、256 個 query）、Qwen-7B 基底。貢獻：

- 448x448 解析度（當時開源 VLM 的 SOTA）。
- Grounding：用帶有明確座標 token 輸出的圖文配對訓練。「The cat is at <box>(112, 204), (280, 344)</box>」。
- 從一開始就進行中英文多語言訓練。

當時的 benchmark：英文上與 GPT-4V 相當，中文上居於領先。grounding 監督才是真正的亮點。

### Qwen2-VL (2024 年 9 月)：M-RoPE 與原生解析度

Qwen2-VL 用原生動態解析度的 ViT encoder，取代了固定解析度加 Q-Former 的堆疊。主要改變：

- 原生動態解析度。ViT 接受任何可被 28 整除的 HxW（patch 14 搭配 2 倍空間合併）。1120x672 的圖片（40x24 個合併後 patch）產生 960 個 visual token。沒有縮放、沒有 tiling、沒有縮圖。
- M-RoPE (Multimodal RoPE)。每個 token 帶有 3D 位置 (t, h, w)，而不是 1D。對圖片 t=0，對影片 t = frame_index。RoPE 以每個軸各自的頻率旋轉 query/key 向量。沒有位置 embedding 表。
- MLP projector。捨棄 Q-Former；在合併後的 patch token 上使用 2 層 MLP。
- 帶 dynamic FPS 的影片。影片預設以 1-2 FPS 取樣，但模型接受任意的影格數。

結果：Qwen2-VL-7B 在數個 multimodal benchmark 上追平 GPT-4o，並在 DocVQA 上勝出（94.5 對 88.4）。架構改變是決定性的一步。

### Qwen2.5-VL (2025 年 2 月)：dynamic FPS + 絕對時間

Qwen2.5-VL 的重大轉變在影片。Dynamic FPS 不只是「需要時多取樣一些影格」。論文把以下內容正式化：

- Position ID 中的絕對時間。M-RoPE 的時間軸不再用影格索引（影格 0、1、2...），而是依真實時間戳，因此兩個影格 ID 的差距對應它們之間的秒數。「在 0:04，貓跳起來。」沒有額外的文字 token 攜帶時間，所以這個編碼不增加序列長度。
- Dynamic FPS。慢速畫面以 1 FPS 取樣，動作畫面用 4+ FPS。由使用者或訓練者選擇；M-RoPE 自行適應。
- ViT 中的 window attention。空間 attention 採用視窗化（在區塊內做局部 attention）以提升吞吐量；每隔幾層加入全域 attention。
- 明確的 JSON 輸出格式。以 tool-call 資料訓練：`"{\"tool\": \"click\", \"coords\": [380, 220]}"`。開箱即用就適合 agent。
- MRoPE-v2 縮放。位置隨最大輸入大小縮放，所以 10 分鐘的影片不會用完頻率範圍。

Benchmark：Qwen2.5-VL-72B 在多數影片 benchmark 上勝過 GPT-4o，在文件上追平 Gemini 2.0，並為 GUI grounding 創下開源模型 SOTA（ScreenSpot：準確率 84%，GPT-4o 為 38%）。

### Qwen3-VL (2025 年 11 月)

Qwen3-VL 是漸進式升級，偏向整合而非重新發明：更大的 LLM backbone (Qwen3-72B)、擴充的訓練資料、改善的 OCR，以及透過 Qwen3「thinking mode」增強的推理。ViT 和 M-RoPE 維持不變。論文把重點放在資料與訓練的改進，而非架構。

這條系譜的重點：到 2025 年，Qwen-VL 架構已經穩定。後續世代擴大的是運算和資料，而不是基本單元。

### M-RoPE 的數學 (M-RoPE mathematically)

傳統 RoPE 使用成對座標，依位置 `m` 旋轉維度 `d` 的 query `q`：

```math
\begin{aligned}
q_{\text{rot}}[2i] &= q[2i] \cos(m \theta_i) - q[2i+1] \sin(m \theta_i) \\
q_{\text{rot}}[2i+1] &= q[2i] \sin(m \theta_i) + q[2i+1] \cos(m \theta_i) \\
\theta_i &= 10000^{-2i/d}
\end{aligned}
```

M-RoPE 把 hidden 維度分成三個頻帶。假設 `d = 96`。分配 32 維給時間、32 維給高度、32 維給寬度。每個頻帶依自己軸上的位置旋轉。位於 (t=5, h=10, w=20) 的 patch，會在它的三個頻帶上分別套用旋轉 `R_t(5)`、`R_h(10)`、`R_w(20)`。

文字 token 使用 `t = text_index, h = 0, w = 0`（或某個正規化的選擇），以保持相容性。影片影格使用 `t = frame_time, h = row, w = col`。單張圖片使用 `t = 0`。

好處是：一套位置編碼就能處理文字、圖片與影片，不需要分支程式碼或不同的位置表。

### Dynamic-FPS 取樣邏輯 (Dynamic-FPS sampling logic)

給定長度 `T` 秒的影片與目標 token 預算 `B`：

1. 計算你負擔得起的最大 FPS：`fps_max = B / (T * tokens_per_frame)`。
2. 從 `{1, 2, 4, 8}` 中挑一個滿足 `fps <= fps_max` 的目標 FPS。
3. 如果動作多（光流啟發式或使用者明確要求），選較高的 FPS。如果動作少，選較低的。
4. 以選定的 FPS 均勻取樣；依每個影格以秒為單位的時間戳，設定它的 M-RoPE 時間 ID。

Qwen2.5-VL 隱含地訓練了這套邏輯；推論時使用者透過 `fps` 參數控制。60 秒的動作序列，以 4 FPS、每影格 81 個 token，共 19440 個 token，在 32k context 中綽綽有餘。

### 結構化的 agent 輸出 (Structured agent output)

Qwen2.5-VL 的 agent 訓練明確以結構化的 tool call 為目標：

```
{
  "tool": "mouse_click",
  "coords": [1024, 512],
  "button": "left",
  "modifier": null
}
```

解析是確定性的：對模型輸出做 JSON.parse。相較之下，自由格式的「click at (1024, 512)」需要 regex 和處理歧義。這個轉變就是 Qwen2.5-VL 的 ScreenSpot 分數從 Qwen2-VL 的 55% 躍升到 84% 的原因。

```figure
mm-mrope-axes
```

## 實際使用 (Use It)

`code/main.py` 實作了：

- 混合文字、圖片 patch 與影片影格的打包序列的 M-RoPE 位置計算。
- Dynamic-FPS 取樣器：給定 (duration, budget, motion_level)，挑選 FPS 並輸出影格時間戳。
- 一個玩具版的 Qwen2.5-VL JSON 輸出解析器，處理帶座標欄位的 tool-call 回應。

執行它，然後在 5 分鐘的影片上把固定 FPS 換成 dynamic-FPS，感受差別。

## 交付 (Ship It)

這堂課產出 `outputs/skill-qwen-vl-pipeline-designer.md`。給定一個影片任務（監控、agent、動作辨識、無障礙），它會輸出 Qwen2.5-VL 的設定（影格預算、FPS 策略、window-attention 旗標、agent 輸出模式）以及延遲估算。每當你為影片產品部署 Qwen-VL 家族的模型時就用它。

## 練習 (Exercises)

1. 計算位於 (t=3, h=5, w=7)、hidden 為 48（每個頻帶 16、base theta 10000）的 patch 的 M-RoPE 旋轉。列出每個頻帶前三對的旋轉角度。

2. 10 分鐘的監視器錄影以 1 FPS 產生多少影格？在 384 解析度、3 倍 pool 下，總共有多少 token？Qwen2.5-VL 預設的 32k context 撐得住嗎？

3. 為 30 秒的網球對打、30 秒的食譜示範與 30 秒的 UI-agent 錄影挑選 FPS。用 dynamic-FPS 邏輯為各個選擇辯護。

4. Qwen2.5-VL 完全捨棄了 Q-Former。為什麼簡單的 MLP 在 2025 年行得通，在 2023 年卻不行？（提示：資料規模與 encoder 品質。）

5. 把三段 Qwen2.5-VL 的 JSON tool-call 輸出解析成 Python dict。格式錯誤的 JSON 會在哪裡失敗？Qwen cookbook 建議什麼復原策略？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| M-RoPE | "Multimodal RoPE" | 3D 旋轉位置 embedding，在 hidden 維度中有時間、高度與寬度三個頻帶 |
| Dynamic FPS | "Smart sampling" | 依動作量、長度與 token 預算，為每段影片選擇影格取樣率 |
| Absolute time encoding | "Time-aligned M-RoPE" | 時間位置 ID 依每個影格的時間戳設定，所以模型看到的是實際秒數而非影格索引，且不需額外 token |
| Window attention | "Local attention" | 把空間 self-attention 限制在小視窗內以提升速度；定期加入全域 attention |
| Structured agent output | "JSON mode" | 用訓練資料監督，教 VLM 輸出可解析、帶座標與工具名稱的 JSON |
| min_pixels / max_pixels | "Resolution bounds" | Qwen2.5-VL 的每請求控制項，限制總像素數，進而限制 token 數 |
| Grounding | "Point-at-it" | 以文字 token 輸出 bounding box 座標；從 Qwen-VL v1 起就在使用 |

## 延伸閱讀 (Further Reading)

- [Bai et al. — Qwen-VL (arXiv:2308.12966)](https://arxiv.org/abs/2308.12966)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
- [Qwen Team — Qwen2.5-VL Technical Report (arXiv:2502.13923)](https://arxiv.org/abs/2502.13923)
- [Qwen Team — Qwen3-VL (arXiv:2511.21631)](https://arxiv.org/abs/2511.21631)
- [Zhu et al. — InternVL3 (arXiv:2504.10479)](https://arxiv.org/abs/2504.10479)
