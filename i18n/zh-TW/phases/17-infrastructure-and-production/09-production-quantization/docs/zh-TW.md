# Production Quantization：AWQ、GPTQ、GGUF K-quants、FP8、MXFP4/NVFP4

> Quantization 格式不是萬用的選擇，而是硬體、serving engine 與工作負載的函數。GGUF Q4_K_M 或 Q5_K_M 稱霸 CPU 與 edge，透過 llama.cpp 和 Ollama 提供。當你需要在同一個 base 上跑 multi-LoRA 時，GPTQ 在 vLLM 內勝出。搭配 Marlin-AWQ kernels 的 AWQ 在 7B 等級模型上可達 ~741 tok/s，並在 INT4 中有最佳的 Pass@1，是 2026 年資料中心 production 的預設。FP8 在 Hopper、Ada 與 Blackwell 上仍是折衷選擇，近乎無損且支援廣泛。NVFP4 與 MXFP4（Blackwell microscaling）很激進，需要逐 block 驗證。有兩個陷阱會咬到團隊：校準資料集必須符合部署領域，而且 KV cache 與權重量化是分開的，「我的模型現在只有 4 GB」這種 AWQ 說法忘了 production batch size 下 10-30 GB 的 KV cache。

**Type:** Learn
**Languages:** Python (stdlib, toy memory and throughput comparison across formats)
**Prerequisites:** Phase 10 · 13 (Quantization foundations), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出六種 production quantization 格式及其在 2026 年的最佳使用情境。
- 依據硬體（CPU vs GPU、Hopper vs Blackwell）、engine（vLLM、TRT-LLM、llama.cpp）與工作負載（一般聊天、推理、multi-LoRA）挑選格式。
- 計算所選格式節省的權重記憶體，以及維持不變的 KV cache。
- 說出會讓量化模型在領域流量上退化的校準資料集陷阱。

## 問題 (The Problem)

Quantization 降低記憶體與 HBM 頻寬，而這正是 decode 所需要的。FP16 的 70B 模型有 140 GB 權重。把權重量化成 INT4（AWQ 或 GPTQ），模型就是 35 GB，能放進一張 H100，還留有 KV cache 的空間。這很重要，因為在 128 個並行 sequence、2k context 下，光是 KV cache 就有 20-30 GB。

但 quantization 不是免費的。激進的 quantization 會降低品質，尤其在重推理的任務上。不同格式搭配不同 engine。不同硬體原生支援不同精度。2026 年的格式動物園是真實存在的，你無法照抄別人的選擇，必須依你的 stack 來挑。

## 核心概念 (The Concept)

### 六種格式

| 格式 | Bits | 最佳情境 | Engines |
|--------|------|-----------|---------|
| GGUF Q4_K_M / Q5_K_M | 4-5 | CPU、edge、筆電 | llama.cpp, Ollama |
| GPTQ | 4-8 | vLLM 上的 Multi-LoRA | vLLM, TGI |
| AWQ | 4 | 資料中心 GPU production | vLLM (Marlin-AWQ), TGI |
| FP8 | 8 | Hopper/Ada/Blackwell 資料中心 | vLLM, TRT-LLM, SGLang |
| MXFP4 | 4 | Blackwell 多使用者 | TRT-LLM |
| NVFP4 | 4 | Blackwell 多使用者 | TRT-LLM |

### GGUF：CPU/edge 的預設

GGUF 是檔案格式，本身並非一種 quantization 方案。它在一個容器中打包 K-quant 變體（Q2_K、Q3_K_M、Q4_K_M、Q5_K_M、Q6_K、Q8_0）。Q4_K_M 與 Q5_K_M 是 production 預設，在 4-5 bits 就有接近 BF16 的品質。對 CPU 或 edge serving 是最佳選擇，因為 llama.cpp 是目前最快的 CPU inference engine。

在 vLLM 中的 throughput 損失：7B 約 ~93 tok/s，這個格式並未針對 GPU kernels 優化。部署目標是 CPU/edge 時才用 GGUF，否則不要。

### GPTQ：vLLM 中的 multi-LoRA

GPTQ 是帶有校準 pass 的 post-training quantization 演算法。Marlin kernels 讓它在 GPU 上很快（比非 Marlin 的 GPTQ 快 2.6x）。7B 上約 ~712 tok/s。

獨特的優勢：GPTQ-Int4 在 vLLM 中支援 LoRA adapters。如果你要服務一個 base 模型加上 10-50 個 fine-tuned 變體（各是一個 LoRA），GPTQ 就是你的路。截至 2026 年初，NVFP4 還不支援 LoRA。

### AWQ：資料中心 GPU 的預設

Activation-aware Weight Quantization。在量化期間保護最顯著的約 ~1% 權重。Marlin-AWQ kernels：比單純實作快 10.9x。7B 上約 ~741 tok/s，在 INT4 格式中有最佳 Pass@1。

新的 GPU serving 就選 AWQ，除非你需要 multi-LoRA (GPTQ) 或激進的 Blackwell FP4 (NVFP4)。

### FP8：可靠的折衷

8-bit 浮點。近乎無損。支援廣泛。Hopper Tensor Cores 原生加速 FP8，Blackwell 承襲。當品質不能妥協（推理、醫療、程式碼生成）時，FP8 是 2026 年安全的預設。記憶體節省只有 INT4 的一半，但品質風險低很多。

### MXFP4 / NVFP4：Blackwell 的激進選項

Microscaling FP4。每個權重 block 有自己的 scale factor。激進，但在 Blackwell Tensor Cores 上有硬體加速。每個 token 的位元組數比 FP8 少一半，也就是 Phase 17 · 07 的經濟效益。

注意事項：
- 目前還不支援 LoRA（2026 年初）。
- 在重推理的工作負載上，品質下降明顯。
- 要針對每個模型在你的 eval set 上驗證。

### 校準陷阱

AWQ 與 GPTQ 需要校準資料集，通常是 C4 或 WikiText。對領域模型（程式碼、醫療、法律）而言，用通用網頁文字校準，會讓演算法對該保護哪些權重做出錯誤決定。HumanEval 的 Pass@1 可能掉好幾個百分點。

修法：用領域內的資料校準。通常數百筆領域樣本就夠了。上線前在 eval set 上測試。

### KV cache 陷阱

AWQ 把權重縮到 4 bits。KV cache 是分開的，維持在 FP16/FP8。以使用 AWQ 的 70B 模型為例：

- 權重：~35 GB（從 140 GB 變成 INT4）。
- 128 並行 × 2k context 的 KV cache：~20 GB。
- Activations：~5 GB。
- 總計：~60 GB，放得進 H100 80GB。

天真地說「我把模型量化到 4 GB」，會忘了另外的 30-50 GB。請整體規劃 HBM 預算。

另外，KV cache quantization（FP8 KV 或 INT8 KV）是另一個選擇，有它自己的取捨：它直接影響 attention 的準確度，並不是免費的好處。

### AWQ INT4 對推理有風險

Chain-of-thought、數學、長 context 的程式碼生成，這些會明顯受到激進 quantization 的影響。AWQ INT4 在 MATH 上約損失 ~3-5 分。對重推理的工作負載，請使用 FP8 或 BF16，接受記憶體成本。

### 2026 年挑選指南

- CPU/edge serving：GGUF Q4_K_M。結束。
- GPU serving、一般聊天、無 LoRA：AWQ。
- GPU serving、multi-LoRA：搭配 Marlin 的 GPTQ。
- 推理工作負載：FP8。
- Blackwell 資料中心、已驗證品質：NVFP4 + FP8 KV。
- 不確定：對每個候選格式跑 1,000 筆樣本的 eval。

```figure
gpu-memory-breakdown
```

## 實際使用 (Use It)

`code/main.py` 針對一系列模型大小，計算六種格式的記憶體占用（權重 + KV + activations）與相對 throughput。展示 KV cache 在哪裡占主導、權重壓縮在哪裡划算，以及 FP8 在哪裡是安全的選擇。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?dtype=fp8&users=64) 這個機器展示了 NVIDIA L4 上的 FP8。權重從 15.28 GB 降到 8.81 GB，KV pool 從 24,094 成長到 142,369 tokens，而 64 位使用者下的實測輸出從 250.85 提升到 735.91 tokens per second。

## 交付 (Ship It)

這堂課產出 `outputs/skill-quantization-picker.md`。給定硬體、模型大小、工作負載類型與品質容忍度，挑出格式並產出校準/驗證計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。對 128 並行、2k context 的 70B 模型，計算每種格式的總 HBM。哪種格式能讓你放進單張 H100 80GB？
2. 你有一個 7B 的程式碼模型。挑一種格式並說明理由。如果你對品質容忍度判斷錯了，復原路徑是什麼？
3. 計算為醫療領域模型校準 AWQ 所需的校準資料集大小。為什麼資料越多不一定越好？
4. 閱讀 Marlin-AWQ kernel 論文或 release notes。用三句話解釋為什麼 AWQ 在 7B 上達到 741 tok/s，而原始 GPTQ 約 ~712。
5. 什麼時候把 AWQ 權重搭配 FP8 KV cache 是合理的，而不是把 KV 維持在 BF16？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| GGUF | 「llama.cpp 格式」 | 打包 K-quant 變體的檔案格式；CPU/edge 預設 |
| Q4_K_M | 「Q4 K M」 | 4-bit K-quant medium；production GGUF 預設 |
| GPTQ | 「gee pee tee q」 | 帶校準的 post-train INT4；在 vLLM 中支援 LoRA |
| AWQ | 「a w q」 | Activation-aware INT4；Marlin kernels；INT4 中最佳 Pass@1 |
| Marlin kernels | 「快速 INT4 kernels」 | Hopper 上 INT4 的客製 CUDA kernels；約 10x 加速 |
| FP8 | 「八位元浮點」 | Hopper/Ada/Blackwell 上安全的精度預設 |
| MXFP4 / NVFP4 | 「microscaling four」 | 帶有逐 block scale factor 的 Blackwell 4-bit FP |
| Calibration dataset | 「校準資料」 | 用來挑選 quantization 參數的輸入文字；必須符合領域 |
| KV cache quantization | 「KV INT8」 | 與權重分開的選擇；影響 attention 準確度 |

## 延伸閱讀 (Further Reading)

- [VRLA Tech — LLM Quantization 2026](https://vrlatech.com/llm-quantization-explained-int4-int8-fp8-awq-and-gptq-in-2026/)：比較性 benchmark。
- [Jarvis Labs — vLLM Quantization Complete Guide](https://jarvislabs.ai/blog/vllm-quantization-complete-guide-benchmarks)：各格式的 throughput 數字。
- [PremAI — GGUF vs AWQ vs GPTQ vs bitsandbytes 2026](https://blog.premai.io/llm-quantization-guide-gguf-vs-awq-vs-gptq-vs-bitsandbytes-compared-2026/)：逐格式挑選。
- [vLLM docs — Quantization](https://docs.vllm.ai/en/latest/features/quantization/index.html)：支援的格式與旗標。
- [AWQ paper (arXiv:2306.00978)](https://arxiv.org/abs/2306.00978)：原始 AWQ 公式化。
- [GPTQ paper (arXiv:2210.17323)](https://arxiv.org/abs/2210.17323)：原始 GPTQ 公式化。
