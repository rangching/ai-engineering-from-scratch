# 自架 Serving 選型：讓 Engine 對上硬體與規模 (Self-Hosted Serving Selection)

> Engine 的選擇是硬體、規模與生態系的函數，不是讀排行榜。2026 年主導自架 inference 的有四個 engine：llama.cpp、Ollama、vLLM、SGLang，TGI 則處於維護模式、逐漸落後。**llama.cpp** 在 CPU 上最快，模型支援最廣，對 quantization 與 threading 有完整控制。**Ollama** 是開發筆電上的一行指令安裝，比 llama.cpp 慢約 ~15-30%（Go + CGo + HTTP 序列化），在接近 prod 的負載下 throughput 有 3x 差距。**TGI 於 2025 年 12 月 11 日進入維護模式**，只修 bug，raw throughput 比 vLLM 慢約 ~10%，但向來擁有最佳的 observability 以及與 HF 生態系的整合。這個維護狀態讓它成為有風險的長期押注，新專案用 SGLang 或 vLLM 是更安全的預設。**vLLM** 是通用的 production 預設：v0.15.1（2026 年 2 月）新增 PyTorch 2.10、RTX Blackwell SM120、H200 優化。**SGLang** 是 agentic 多輪 / 重度前綴的專家：production 中有 400,000+ 張 GPU（xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS）。硬體限制：以 CPU 為主 → llama.cpp。AMD / 非 NVIDIA → vLLM 是支援最強的路徑（TRT-LLM 綁定 NVIDIA）。2026 年的 pipeline 模式：dev = Ollama，staging = llama.cpp，prod = vLLM 或 SGLang。這些 engine 使用不同的權重格式，llama.cpp 家族用 GGUF，GPU engine 用 HF safetensors，所以階段之間可能需要格式轉換。

**Type:** Learn
**Languages:** Python (stdlib, engine-decision tree walker)
**Prerequisites:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 給定硬體（CPU / AMD / NVIDIA Hopper / Blackwell）、規模（1 位使用者 / 100 / 10,000）與工作負載（一般聊天 / agent / 長 context），挑選 engine。
- 說出 TGI 2026 年的維護模式狀態（2025 年 12 月 11 日），以及為什麼它讓新專案偏向 vLLM 或 SGLang。
- 描述 dev/staging/prod pipeline，包括 GGUF 轉 safetensors 的格式轉換落在階段之間的哪個位置。
- 解釋為什麼「以 CPU 為主」指向 llama.cpp，而「AMD」排除 TRT-LLM。

## 問題 (The Problem)

你的團隊開始一個新的自架 LLM 專案。一位工程師說 Ollama，另一位說 vLLM，第三位說「TGI 不是開箱即用嗎？」三者在不同情境下都是對的，但沒有任何一個適用於所有情況。

在 2026 年，選擇樹很重要：先看硬體，再看規模，最後看工作負載。而且有一個 2025 年的特定事件，TGI 在 12 月 11 日進入維護模式，改變了新專案的預設。

## 核心概念 (The Concept)

### 五個 engine

| Engine | 最適合 | 備註 |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / 最少相依 / 最廣的模型支援 | CPU 上最快，完整控制 |
| **Ollama** | 開發筆電、單一使用者、一行指令安裝 | 比 llama.cpp 慢 15-30%；prod throughput 有 3x 差距 |
| **TGI** | HF 生態系、受監管產業 | **2025 年 12 月 11 日進入維護模式** |
| **vLLM** | 通用 production、100+ 位使用者 | 廣泛的 production 預設；v0.15.1 2026 年 2 月 |
| **SGLang** | Agentic 多輪、重度前綴的工作負載 | production 中有 400,000+ 張 GPU |

### 先看硬體的決策

**以 CPU 為主** → llama.cpp。Ollama 也行，但比較慢。沒有其他 engine 在 CPU 上有競爭力。

**AMD GPU** → vLLM 是支援最強的路徑（AMD ROCm 支援）。SGLang 也行。TRT-LLM 綁定 NVIDIA，所以出局。

**NVIDIA Hopper (H100 / H200)** → vLLM 或 SGLang 或 TRT-LLM。三者都是頂級。

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM 是 throughput 的領先者 (Phase 17 · 07)。vLLM 和 SGLang 緊追在後。

**Apple Silicon (M 系列)** → llama.cpp (Metal)。Ollama 包裝了它。

### 其次看規模的決策

**1 位使用者 / 本機開發** → Ollama。一行指令，幾秒內出第一個 token。

**10-100 位使用者 / 小團隊** → 單 GPU 的 vLLM。

**100-10k 位使用者 / production** → vLLM production-stack (Phase 17 · 18) 或 SGLang。

**10k+ 位使用者 / 企業** → vLLM production-stack + disaggregated (Phase 17 · 17) + LMCache (Phase 17 · 18)。

### 最後看工作負載的決策

**一般聊天 / Q&A** → vLLM 以廣泛的預設勝出。

**Agentic 多輪（tools、規劃、memory）** → SGLang 的 RadixAttention (Phase 17 · 06) 稱霸。

**大量重用前綴的 RAG** → SGLang。

**程式碼生成** → vLLM 沒問題；SGLang 在 cache 上略好。

**長 context (128K+)** → vLLM + chunked prefill；SGLang + 分層 KV。

### TGI 維護陷阱

Hugging Face TGI 在 2025 年 12 月 11 日進入維護模式，之後只會修 bug。歷史上：頂級的 observability、同類最佳的 HF 生態系整合（model cards、安全工具），raw throughput 略落後 vLLM。

2026 年的新專案：預設避開 TGI。既有的 TGI 部署可以繼續，但最終應遷移。SGLang 與 vLLM 是更安全的預設。

### Pipeline 模式

Dev (Ollama) → staging (llama.cpp) → prod (vLLM)。這些 engine 使用不同的權重格式，llama.cpp 家族用 GGUF，GPU engine 用 HF safetensors，所以階段之間可能需要格式轉換。工程師在筆電上快速迭代；staging 對應 production 的 quantization；prod 是 serving 目標。

### Ollama 的注意事項

Ollama 很適合開發。但不適合共用的 production：Go 的 HTTP 序列化增加開銷，並行管理比 vLLM 簡單，OpenTelemetry 支援落後。把 Ollama 用在它擅長的地方，也就是一位使用者、一行指令，共用時切換到 vLLM。

### 自架 vs 託管是另一個獨立的決策

Phase 17 · 01（託管 hyperscaler）、· 02（inference 平台）涵蓋託管。這堂課假設你已經決定要自架。自架的理由：資料落地、客製 fine-tune、規模化後的總擁有成本、託管服務上沒有的領域模型。

### 該記住的數字

- TGI 維護模式：2025 年 12 月 11 日。
- vLLM v0.15.1：2026 年 2 月；PyTorch 2.10；Blackwell SM120 支援。
- SGLang production 規模：400,000+ 張 GPU。
- Ollama 與 llama.cpp 的 throughput 差距：慢 15-30%；prod 負載下 3x。

```figure
data-parallel
```

## 實際使用 (Use It)

`code/main.py` 是一個決策樹走訪器：給定硬體 + 規模 + 工作負載，挑選 engine 並解釋原因。

## 交付 (Ship It)

這堂課產出 `outputs/skill-engine-picker.md`。給定限制條件，挑選 engine 並寫出遷移計畫。

## 練習 (Exercises)

1. 用你的硬體 / 規模 / 工作負載執行 `code/main.py`。輸出符合你的直覺嗎？
2. 你的基礎設施是 12 張 H100 和 8 張 MI300X AMD。用哪個 engine？為什麼 TRT-LLM 不在考慮範圍？
3. 某團隊想在 2026 年使用 TGI，因為「這是我們熟悉的」。請論證遷移的理由。
4. 從 Ollama dev 到 vLLM prod：quantization、設定與 observability 有什麼改變？
5. 一個 RAG 產品，P99 前綴長度 8K，且跨 tenant 重用率高。挑選 engine，並與 Phase 17 · 11 + 18 疊加。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| llama.cpp | 「CPU 那個」 | 模型支援最廣，CPU 上最快 |
| Ollama | 「筆電那個」 | 一行指令安裝，開發等級的 throughput |
| TGI | 「HF 的 serving」 | 自 2025 年 12 月起進入維護模式 |
| vLLM | 「預設選項」 | 2026 年廣泛的 production 基準 |
| SGLang | 「agentic 那個」 | 重度前綴、RadixAttention |
| TRT-LLM | 「綁定 NVIDIA」 | Blackwell throughput 領先者，僅限 NVIDIA |
| GGUF | 「llama.cpp 格式」 | 打包好的 K-quant 變體 |
| Production-stack | 「vLLM K8s」 | Phase 17 · 18 的參考部署 |
| Pipeline pattern | 「dev→stage→prod」 | Ollama → llama.cpp → vLLM；各 engine 的權重格式不同 |

## 延伸閱讀 (Further Reading)

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference)：release notes。
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
