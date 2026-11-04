# 硬體專用的 Inference 編譯 (Hardware-Specialized Inference Compilation)：Blackwell 上的 FP8 與 NVFP4

> 硬體專用的 inference 編譯用可攜性換取 throughput，而 TensorRT-LLM（僅限 NVIDIA、針對 Blackwell 調校）是這個取捨回報最明顯的例子。在搭配 Dynamo 編排的 GB200 NVL72 上，SemiAnalysis InferenceX 於 2026 年 Q1-Q2 量測到 120B 模型每百萬 tokens $0.012，而 H100 + vLLM 為 $0.09/M，經濟差距達 7x。這個 stack 把兩種浮點格式和兩種 serving 技術疊加：FP8 對 KV cache 與 attention kernels 仍然關鍵，因為它具備所需的動態範圍；NVFP4（4-bit microscaling）處理權重與 activations；multi-token prediction (MTP) 與 disaggregated prefill/decode 再額外帶來 2-3x。Day-0 模型支援可直接載入 FP4 權重，不需 post-training 轉換。2026 年工程團隊要注意的是：TRT-LLM 雖然開源，卻是 NVIDIA 專用（針對 CUDA 與 Blackwell 特化），所以採用它是用可攜性換 throughput。在承諾之前，請對你的模型與硬體組合算清楚。

**Type:** Learn
**Languages:** Python (stdlib, toy FP8/NVFP4 memory and cost calculator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 10 · 13 (Quantization)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼即使權重用 NVFP4，FP8 對 KV cache 與 attention 仍然關鍵。
- 計算前沿模型在 BF16、FP8、NVFP4 下的 HBM 占用，並推論節省來自哪裡。
- 說出 TRT-LLM 所利用的 Blackwell 專屬功能（day-0 FP4、MTP、disaggregated serving、all-to-all primitives）。
- 判斷何時 TRT-LLM 的 NVIDIA 綁定值得換取相對 Hopper 上 vLLM 的 7x 成本差距。

## 問題 (The Problem)

2026 年 inference 經濟學的前沿是「每塊錢能換多少 tokens」。答案取決於四個疊加的選擇：硬體世代（Hopper H100/H200 vs Blackwell B200/GB200）、精度（BF16 → FP8 → NVFP4）、serving engine（vLLM vs SGLang vs TRT-LLM），以及編排（單純 vs disaggregated vs Dynamo）。

在 Hopper 加 vLLM 上，120B MoE 的成本約 ~$0.09 每百萬 tokens。在 Blackwell 加 TRT-LLM + Dynamo 上，同一個模型約 ~$0.012，便宜 7x。這個差距有一部分來自硬體（Blackwell 的每 GPU LLM throughput 是 Hopper 的 11-15x）。另一部分來自 stack：FP4 權重、MTP draft、disaggregated prefill/decode，以及用於 MoE expert 通訊的 NVLink 5 all-to-all。

你無法在 NVIDIA 的 stack 之外複製這些。這就是取捨：用可攜性換經濟效益。這堂課的重點，就是了解 stack 中哪些選擇貢獻了差距的哪一部分。

## 核心概念 (The Concept)

### 為什麼 FP8 仍是 KV cache 的底線

2026 年常見的錯誤：以為 NVFP4 適用於所有地方。其實不然。KV cache 需要 FP8（8-bit 浮點），因為它儲存的 attention keys 與 values 跨越很寬的動態範圍。把 KV 量化成 FP4 會造成災難性的準確度損失：分布的尾端被截掉，attention 分數崩潰。FP8 的指數位元給了 KV cache 所需的範圍。

NVFP4 (2025-2026) 套用於權重與 activations。Microscaling：每一個權重 block 有自己的 scale factor，所以小 block 能涵蓋不同的動態範圍，而不會有 per-tensor scale 的損失。對 activations 來說，FP4 撐得住，因為同一層內 activations 的範圍很小。

典型的 Blackwell 設定：

- 權重：NVFP4（4-bit microscaling）。
- Activations：NVFP4。
- KV cache：FP8。
- Attention accumulator：FP32（softmax 穩定性）。

### TRT-LLM 使用的 Blackwell 專屬 primitives

- **Day-0 FP4 權重**：模型供應商直接提供 FP4 權重；TRT-LLM 不需 post-training 轉換就能載入。FP4 沒有 AWQ / GPTQ 這個步驟。
- **Multi-token prediction (MTP)**：和 EAGLE 同樣的概念 (Phase 17 · 05)，但整合進 TRT-LLM 的 build。
- **Disaggregated serving**：prefill 與 decode 放在不同的 GPU pool，KV cache 透過 NVLink 或 InfiniBand 傳輸。與 Dynamo 同樣的概念 (Phase 17 · 20)。
- **All-to-all 通訊 primitives**：NVLink 5 讓 MoE expert 通訊 latency 比 Hopper 降低 3x。TRT-LLM 的 MoE kernels 就是針對此調校。
- **NVFP4 + MXFP8 microscaling**：Blackwell Tensor Cores 上由硬體加速的 scale-factor 處理。

### 你該背起來的數字

- 透過 TRT-LLM 在 GPT-OSS-120B 上，HGX B200 為 $0.02/M tokens。
- 透過 Dynamo（編排 TRT-LLM）的 GB200 NVL72 為 $0.012/M tokens。
- 在可比較的工作負載上，H100 + vLLM ≈ $0.09/M tokens。
- TRT-LLM 三個月的更新帶來 2.8x throughput 提升 (2026)。
- 每 GPU LLM throughput 11-15x，Blackwell 對 Hopper。
- MLPerf Inference v6.0 (2026 年 4 月)：Blackwell 在每個提交的任務上都領先。

### FP4 實際上要付出多少品質代價

NVFP4 很激進。在重推理的工作負載（chain-of-thought、數學、長 context 的程式碼生成）上，FP4 權重會明顯退化。Per-block 校準能緩解但無法消除。交付推理模型的團隊常常採用 FP8 權重 + FP4 activations 作為折衷，或是全程留在 H200 的 FP8。

規則：在承諾使用 NVFP4 權重之前，一定要在你的 eval set 上驗證任務品質。

### 為什麼這是 NVIDIA 綁定的決策

TRT-LLM 是 C++ + CUDA + 閉源 kernels。模型需要針對特定 GPU SKU 編譯。不支援 AMD、Intel、ARM。如果你的基礎設施策略是多廠商，那由 TRT-LLM 服務的那一層就不適合用 TRT-LLM，你仍可在混合硬體上用 vLLM 服務。如果你只用 NVIDIA，7x 的差距足以抵過綁定。

### 2026 年的實務配方

對於年度 inference 帳單 $100M+ 的情況，停在 Hopper + vLLM 等於放掉 7-10x。把成本占比最高的工作負載遷移到 Blackwell + TRT-LLM + Dynamo。實驗層保留在 H100 + vLLM，以維持模型迭代速度。在上 production 前，驗證每個轉成 NVFP4 的模型品質。

### Disaggregation 的額外紅利

TRT-LLM 的 disaggregated serving（獨立的 prefill 與 decode pool）在 Phase 17 · 20 有深入介紹。在 Blackwell 上，倍數會疊加：FP4 權重 × MTP 加速 × disaggregated 放置 × cache-aware 路由。7x 這個數字假設的是完整的 stack。

```figure
pipeline-parallel
```

## 實際使用 (Use It)

`code/main.py` 針對三種 stack 計算模型的 HBM 占用、decode throughput（memory-bound 區間）與 $/M-tokens：H100 + BF16 + vLLM、H100 + FP8 + vLLM、B200 + NVFP4/FP8 + TRT-LLM。執行它，看看疊加效應，以及每項改變貢獻了差距的多大比例。

## 交付 (Ship It)

這堂課產出 `outputs/skill-trtllm-blackwell-advisor.md`。給定工作負載、模型大小與年度 token 量，它會判斷 Blackwell + TRT-LLM stack 是否值得 NVIDIA 綁定。

## 練習 (Exercises)

1. 執行 `code/main.py`。對一個 30% 活躍參數的 120B MoE，計算 H100 BF16、H100 FP8 與 B200 NVFP4/FP8 上受 memory 頻寬限制的 decode throughput。最大的躍升來自哪裡？
2. 一位客戶每年在 H100 + vLLM 上花 $2M。考慮 7x 的經濟差距，他們需要買多少張 Blackwell GPU，才能在 12 個月內攤提遷移到 TRT-LLM 的成本（損益平衡數量）？
3. 你看到在 NVFP4 權重轉換後，MATH 準確度下降 3 分。說出兩條恢復路徑：一條品質優先（保留 FP8 權重），一條成本優先（用領域內資料校準）。
4. 閱讀 MLPerf v6.0 的 inference 結果。哪個任務的 Blackwell 對 Hopper 差距最小，為什麼？
5. 計算 405B 模型在 NVFP4 權重 + FP8 KV cache、128k context 下所需的 HBM。能放進單一 GB200 NVL72 節點嗎？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| FP8 | 「八位元浮點」 | 8-bit 浮點；因動態範圍而用於 KV cache 與 attention |
| NVFP4 | 「四位元 micro」 | NVIDIA 的 4-bit microscaling FP 格式；Blackwell 上用於權重與 activations |
| MXFP8 | 「MX eight」 | Microscaling FP8 變體；在 Blackwell Tensor Cores 上由硬體加速 |
| Day-0 FP4 | 「直接提供 FP4 權重」 | 模型供應商發布已是 FP4 的權重；沒有 post-train 轉換步驟 |
| MTP | 「multi-token prediction」 | TRT-LLM 整合的 speculative-decoding draft (Phase 17 · 05) |
| Disaggregated serving | 「拆分 prefill/decode」 | Prefill 與 decode 放在不同 GPU pool；KV 透過 NVLink/IB 傳輸 |
| All-to-all | 「MoE expert 通訊」 | 把 tokens 路由到 expert GPU 的通訊模式；NVLink 5 降低 3x |
| InferenceX | 「SemiAnalysis inference bench」 | 2026 年業界公認的每 token 成本 benchmark |

## 延伸閱讀 (Further Reading)

- [NVIDIA — Blackwell Ultra MLPerf Inference v6.0](https://developer.nvidia.com/blog/nvidia-blackwell-ultra-sets-new-inference-records-in-mlperf-debut/)：2026 年 4 月的 MLPerf 結果。
- [NVIDIA — MoE Inference on Blackwell](https://developer.nvidia.com/blog/delivering-massive-performance-leaps-for-mixture-of-experts-inference-on-nvidia-blackwell/)：NVLink 5 all-to-all 與 MoE kernels。
- [TensorRT-LLM Overview](https://nvidia.github.io/TensorRT-LLM/overview.html)：官方 engine 文件。
- [NVIDIA — Introducing Dynamo](https://developer.nvidia.com/blog/introducing-nvidia-dynamo-a-low-latency-distributed-inference-framework-for-scaling-reasoning-ai-models/)：TRT-LLM 之上的 disaggregated 編排。
- [MLPerf Inference](https://mlcommons.org/benchmarks/inference-datacenter/)：發布 Blackwell 數字的 benchmark 套件。
