# Emu3：用 Next-Token 預測生成圖片與影片 (Emu3: Next-Token Prediction for Image and Video Generation)

> BAAI 的 Emu3 (Wang et al., 2024 年 9 月) 是本該終結 diffusion 與 autoregressive 之爭的 2024 年成果。單一 Llama 風格的 decoder-only transformer，只用 next-token 預測目標訓練，跨越文字 + VQ 圖片 token + 3D VQ 影片 token 的統一詞彙表，在圖片生成上勝過 SDXL，在感知上勝過 LLaVA-1.6。沒有 CLIP loss，沒有 diffusion 排程。推論時為了品質會使用 classifier-free guidance，但核心訓練目標是搭配 teacher forcing 的 next-token 預測。已發表於 Nature。這堂課讀 Emu3 的論點，也就是為什麼更好的 tokenizer 加上規模就是你所需要的一切，並與 diffusion 做法對照。

**Type:** Learn
**Languages:** Python (stdlib, 3D video tokenizer math + autoregressive sampler skeleton)
**Prerequisites:** Phase 12 · 11 (Chameleon)
**Time:** ~120 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼 Emu3 的單一 loss next-token 目標行得通，儘管長期以來大家都假設圖片品質需要 diffusion。
- 描述 3D 影片 tokenizer：時空 VQ codebook 長什麼樣子，為什麼 patch 要橫跨時間。
- 在（訓練運算、推論成本、品質天花板）上比較 Emu3 與 Stable Diffusion XL。
- 說出同一個 Emu3 模型扮演的三種角色：Emu3-Gen（圖片生成）、Emu3-Chat（感知）、Emu3-Stage2（影片生成）。

## 問題 (The Problem)

直到 2024 年的傳統看法：圖片生成需要 diffusion。論點是：離散的圖片 token 會損失太多資訊而無法重建細節，而且 autoregressive 取樣會在數千個 token 之間累積誤差。Stable Diffusion、DALL-E 3、Imagen、Midjourney 全都使用某種形式的 diffusion。Chameleon（課程 12.11）在小規模上部分推翻了這個說法，但品質沒有追上 SDXL。

Emu3 正面攻擊這個論點。主張是：更好的 visual tokenizer + 足夠的規模 + next-token loss = 在同時也做感知的同一個模型中，達到勝過 diffusion 的圖片生成。

這個賭注在發表時很有爭議。兩年後，開源的統一生成家族 (Emu3、Show-o、Janus-Pro、Transfusion) 成為研究的預設路徑；production 前沿模型似乎也使用某種變體。

## 核心概念 (The Concept)

### Emu3 的 tokenizer

關鍵成分是 visual tokenizer。Emu3 訓練了客製的 IBQ 等級 tokenizer（Inverse Bottleneck Quantizer，SBER-MoVQGAN 家族），每個 token 的解析度縮減為 8x8。一張 512x512 的圖片變成 64x64 = 4096 個 token，codebook 大小為 32768。

這比 Chameleon 在 K=8192 下每張 512x512 的 1024 個 token 更大，兩個軸都是：token 多了四倍，codebook 也大四倍。額外的容量換來重建品質。關鍵指標：重建 PSNR 為 30.5 dB，與 Stable Diffusion 連續 latent 空間的 32 dB 不相上下。

對影片：3D VQ tokenizer 把一個時空 patch（4 個影格 x 8x8 像素）編碼成一個整數。8 FPS 的 4 秒片段有 32 個影格；在 256x256、空間縮減 8 倍、時間縮減 4 倍下，token 數為 (256/8) * (256/8) * (32/4) = 32 * 32 * 8 = 8,192 個 token。

tokenizer 的品質是天花板。Emu3 的貢獻有一部分是「我們訓練了一個非常好的 tokenizer」。

### 單一 loss 訓練 (Single-loss training)

Emu3 使用一個目標：在跨文字 token、2D 圖片 token 與 3D 影片 token 的共用詞彙表上做 next-token 預測。訓練時權重會乘上各 modality 專屬的係數來平衡貢獻，但 loss 函式本身相同。

訓練的混合資料：
- 圖片生成：`<text caption> <image> image_tokens </image>`
- 圖片感知：`<image> image_tokens </image> <question> text_tokens`
- 影片生成：`<text caption> <video> video_tokens </video>`
- 影片感知：類似。
- 純文字：標準 NTP。

模型從資料分布中學會何時該輸出圖片 token、何時輸出文字 token。生成能力來自模型在 `<image>` 標籤之後預測圖片 token。

### Classifier-free guidance 與 temperature

autoregressive 圖片生成在推論時搭配 classifier-free guidance (CFG) 會好很多。Emu3 使用它：生成兩次，一次用完整 caption，一次用空 caption，再以 guidance 權重混合 logits（典型值 3.0-7.0）。這就是 diffusion 使用的同一個 CFG 技巧，借用到 autoregressive 的設定。

Temperature 很重要：太高會有 artifact，太低會 mode collapse。Emu3 建議感知用 1.0，圖片生成用 0.8。

### 三種角色，一個模型 (Three roles, one model)

Emu3 以三個功能不同的 API 推出，但底層是同一組權重：

- Emu3-Gen。圖片生成。輸入文字，輸出圖片 token。
- Emu3-Chat。VQA 與 captioning。輸入圖片（token），輸出文字。
- Emu3-Stage2。影片生成與影片 VQA。輸入文字或影片，輸出文字或影片。

沒有任務專屬的 head，只是不同的 prompt 模板。同一個 checkpoint。

### Benchmark

來自 Emu3 論文（2024 年 9 月）：

- 圖片生成：在 MJHQ-30K FID 上勝過 SDXL（5.4 對 5.6），GenEval 整體（0.54 對 0.55，統計上持平），Deep-Eval 的綜合分數相當。
- 圖片感知：在 VQAv2（75.1 對 81.8）和 MMMU（31.6 對 35.1）上落後 LLaVA-1.6，但在 SEED-Bench、OCRBench 和 ChartQA 上勝出。
- 影片生成：4 秒片段的品質，FVD 與公開 benchmark 的 Sora 時代模型相當。

這些數字並非總是勝出，Emu3 這裡丟一分、那裡拿一分，但「next-token 預測就是你所需要的一切」這個主張，在各 modality 間站得住腳。

### 運算成本 (Compute cost)

Emu3 是 80 億參數的模型。論文沒有公布 GPU 小時總數；作為規模參考，Llama-2-7B 的 pretraining 用了約 18.4 萬 A100 GPU 小時，大約 21 個 GPU 年。像 Stable Diffusion 3 這樣的 diffusion 模型訓練預算相近，但需要獨立的文字 encoder 和更複雜的管線。

推論時，Emu3 每張圖片比 SDXL 慢：4096 個圖片 token 以 30 tok/s 約 2 分鐘生成一張 512x512，而 SDXL 只要 2-5 秒。Speculative decoding 和 KV-cache 最佳化縮小了差距，但沒有消除。Autoregressive 圖片生成運算量很大；這是長期存在的取捨。

### 為什麼重要 (Why it matters)

Emu3 更深層的貢獻是概念上的。如果 next-token 預測能擴展到在圖片生成上追平 diffusion，統一模型的路徑（一個 loss、一個 backbone、任何 modality）就是可行的。未來的模型不需要獨立的文字 encoder、獨立的 diffusion 排程器、獨立的 VAE。一個 transformer，每個 modality 一個 tokenizer，加上規模。

Show-o、Janus-Pro 和 InternVL-U 都是建立在這個論點之上，或對它提出挑戰。到 2025 年為止，中國的實驗室（BAAI、DeepSeek）在這個方向上的發表比美國實驗室更積極。

```figure
l5-emu3-next-token
```

## 實際使用 (Use It)

`code/main.py` 建立兩個玩具元件：

- 一個 2D 對 3D VQ tokenizer 的數量計算器：給定 (resolution, patch, clip_length, FPS)，計算圖片與影片的 token 數。
- 一個帶 classifier-free guidance 與 temperature 的 autoregressive 圖片 token 取樣器。

CFG 的實作與 Emu3 的配方一致：以 guidance 權重混合條件與無條件的 logits。

## 交付 (Ship It)

這堂課產出 `outputs/skill-token-gen-cost-analyzer.md`。給定一份生成產品規格（圖片或影片、目標解析度、品質等級、延遲預算），它會計算 token 數、推論成本，並在 Emu3 家族與 diffusion 之間做選擇。

## 練習 (Exercises)

1. Emu3 在 8x8 縮減下，每張 512x512 圖片產生 4096 個 token。計算 1024x1024 和 2048x2048 的對應數字。推論延遲會怎樣？

2. 閱讀 Emu3 第 3.3 節關於影片 tokenizer 的內容。描述 3D VQ patch 的形狀，以及為什麼它橫跨 4 個影格 x 8x8 像素，而不是單一影格。

3. Classifier-free guidance 權重 5.0 對 3.0：視覺效果有何不同？追蹤 `code/main.py` 中的數學。

4. 假設 3000 億個 token，計算 Emu3 (8B) 的訓練 FLOPs，並與 Stable Diffusion 3 比較。哪一個訓練起來更昂貴？

5. Emu3 在 FID 上勝過 SDXL，但在 VQAv2 上不如專門的 VLM。解釋為什麼統一 loss 的做法在不同 benchmark 上，相對於專家模型顯現出不同的強項。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| Next-token prediction | "NTP" | 標準 autoregressive loss：給定 token[0..i] 預測 token[i+1]；只要 token 化，對每種 modality 都適用 |
| IBQ tokenizer | "Inverse bottleneck quantizer" | 一類 VQ-VAE，codebook 更大 (32768+)，重建比 Chameleon 的更好 |
| 3D VQ | "Spatiotemporal quantizer" | 以 (time, row, col) 索引的 codebook；一個 token 涵蓋 4 個影格 x 8x8 像素 |
| Classifier-free guidance | "CFG" | 以權重 gamma 混合條件與無條件的 logits；在推論時提升圖片品質 |
| Unified vocabulary | "Shared tokens" | 文字 + 圖片 + 影片都取自同一個整數空間；模型預測接下來出現的是哪種 modality |
| MJHQ-30K | "Image gen benchmark" | 具有 3 萬個 prompt、Midjourney 品質的 benchmark；Emu3 在此回報 FID |

## 延伸閱讀 (Further Reading)

- [Wang et al. — Emu3: Next-Token Prediction is All You Need (arXiv:2409.18869)](https://arxiv.org/abs/2409.18869)
- [Sun et al. — Emu: Generative Pretraining in Multimodality (arXiv:2307.05222)](https://arxiv.org/abs/2307.05222)
- [Liu et al. — LWM (arXiv:2402.08268)](https://arxiv.org/abs/2402.08268)
- [Yu et al. — MAGVIT-v2 (arXiv:2310.05737)](https://arxiv.org/abs/2310.05737)
- [Tian et al. — VAR (arXiv:2404.02905)](https://arxiv.org/abs/2404.02905)
