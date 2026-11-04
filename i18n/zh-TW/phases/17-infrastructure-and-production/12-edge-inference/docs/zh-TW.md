# Edge Inference：Apple Neural Engine、Qualcomm Hexagon、WebGPU/WebLLM、Jetson

> Edge 的核心限制是記憶體頻寬，不是運算。行動裝置的 DRAM 約 50-90 GB/s；資料中心的 HBM3 超過 2-3 TB/s，差距 30-50x。Decode 是 memory-bound，所以這個差距是決定性的。2026 年的版圖分成四塊。Apple M4/A18 Neural Engine 在統一記憶體（CPU↔NPU 之間沒有複製）下峰值 38 TOPS。Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon 達到 45 TOPS。WebGPU + WebLLM 在 M3 Max 上以 ~41 tok/s 執行 Llama 3.1 8B (Q4)（約為原生的 70-80%）；擁有 17.6k GitHub stars、相容 OpenAI 的 API、約 ~70-75% 的行動裝置覆蓋率。NVIDIA Jetson Orin Nano Super (8GB) 放得下 Llama 3.2 3B / Phi-3；AGX Orin 透過 vLLM 以約 ~40 tok/s 執行 gpt-oss-20b；Jetson T4000 (JetPack 7.1) 是 AGX Orin 的 2x。TensorRT Edge-LLM 支援 EAGLE-3、NVFP4、chunked prefill，Bosch、ThunderSoft、MediaTek 在 CES 2026 展示過。

**Type:** Learn
**Languages:** Python (stdlib, toy bandwidth-bound decode simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 09 (Production Quantization)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋為什麼行動裝置的 LLM inference 受限於記憶體頻寬，運算只是次要。
- 列舉四個 edge 目標（Apple ANE、Qualcomm Hexagon、WebGPU/WebLLM、NVIDIA Jetson），並把每個對應到使用情境。
- 說出 2026 年 WebGPU 的覆蓋缺口（Firefox Android 還在追趕）以及 Safari iOS 26 的上線。
- 為每個目標挑選 quantization 格式（ANE 用 Core ML INT4 + FP16、Hexagon 用 QNN INT8/INT4、瀏覽器用 WebGPU Q4、Jetson Thor 用 NVFP4）。

## 問題 (The Problem)

客戶想要一個裝置端的聊天機器人：語音優先、預設保護隱私、可離線運作。在 MacBook Pro M3 Max 上，Llama 3.1 8B Q4 以 ~55 tok/s 執行，沒問題。在 iPhone 16 Pro 上，同一個模型只有 3 tok/s，不行。在搭載 Snapdragon 8 Gen 3 的中階 Android 上是 7 tok/s。在 Chrome Android v121+ 的瀏覽器中透過 WebGPU，依裝置而定是 4-8 tok/s。

Throughput 的差異不是移植問題。它是頻寬差距，乘上 quantization 格式，再乘上 NPU 是否能從 user-space 存取。2026 年的 edge inference 是四個不同的問題，有四種不同的解法。

## 核心概念 (The Concept)

### 頻寬才是真正的天花板

Decode 每個 token 都要讀取完整的權重。一個 Q4 的 7B 模型是 3.5 GB。以 50 GB/s 讀取 3.5 GB 要 70 ms，理論天花板約 ~14 tok/s。在 90 GB/s（高階行動 DRAM）下，天花板移到約 ~25 tok/s。低於這個數字，再多的運算都沒用。

3 TB/s 的資料中心 HBM3 讀同樣的 3.5 GB 只要 1.2 ms，天花板是 830 tok/s。同一個模型、同樣的權重，只是記憶體子系統不同。

### Apple Neural Engine (M4 / A18)

- 最高 38 TOPS。統一記憶體（CPU 與 ANE 共用同一個 pool），沒有複製開銷。
- 透過 Core ML + `.mlmodel` 編譯後的模型存取，或透過 PyTorch 使用 Metal Performance Shaders (MPS)。
- Llama.cpp 的 Metal backend 使用 MPS，而不是直接用 ANE；原生 ANE 需要 Core ML 轉換。
- 2026 年 iOS app 最佳的實務路徑：Core ML 搭配 INT4 權重 + FP16 activations。

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- 最高 45 TOPS。與 CPU、GPU 整合在 SoC 中，但屬於獨立的記憶體域。
- QNN (Qualcomm Neural Network) SDK 與 AI Hub 提供從 PyTorch/ONNX 的轉換。
- Chat templates、Llama 3.2、Phi-3 都以一等公民的 artifact 形式放在 AI Hub 上。

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS。軟體落後於 Apple/Qualcomm；OpenVINO 持續進步但屬小眾。
- 最適合 Windows ARM 的 copilot app；在 AMD/Intel 桌機上原生支援 local-first。

### WebGPU + WebLLM

- 透過 WebGPU compute shaders 在瀏覽器中執行模型；免安裝。
- Llama 3.1 8B Q4 在 M3 Max 上約 ~41 tok/s，約為同一 backend 原生效能的 70-80%。
- WebLLM 有 17.6k GitHub stars；相容 OpenAI 的 JS API；Apache 2.0。
- 2026 年覆蓋率：Chrome Android v121+、Safari iOS 26 GA、Firefox Android 仍在追趕。整體行動裝置覆蓋率約 ~70-75%。

### NVIDIA Jetson 系列

- Orin Nano Super (8GB)：放得下 Llama 3.2 3B、Phi-3，tok/s 不錯。
- AGX Orin：透過 vLLM 以約 ~40 tok/s 執行 gpt-oss-20b。
- Thor / T4000 (JetPack 7.1)：效能為 AGX Orin 的 2x，支援 EAGLE-3 與 NVFP4。
- TensorRT Edge-LLM (2026) 支援 EAGLE-3 speculative decoding、NVFP4 權重、chunked prefill，也就是把資料中心的最佳化移植到 edge。

### 各目標的 quantization 選擇

| 目標 | 格式 | 備註 |
|--------|--------|-------|
| Apple ANE | INT4 權重 + FP16 activations | Core ML 轉換路徑 |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub 轉換器 |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | 使用 `mlc_llm convert_weight` + 編譯後的 `.wasm`；不支援 GGUF |
| Jetson Orin Nano | Q4 GGUF 或 TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM 路徑 |

### Edge 上的長 context 陷阱

Llama 3.1 的 128K context 是資料中心功能。在 8 GB RAM 的手機上，4 GB 模型 + 32K tokens 的 2 GB KV cache + OS 開銷 = OOM。除非接受激進的 KV quantization (Q4 KV)，否則 edge 部署會把 context 維持在 4K-8K。

### 語音是殺手級應用

語音 agent 對 latency 敏感（first token < 500 ms）。本機 inference 完全消除網路 latency。搭配 speech-to-text（Whisper Turbo 變體可在 edge 執行），edge inference 就成了 production 品質的語音迴圈。

### 該記住的數字

- Apple M4 / A18 ANE：38 TOPS。
- Qualcomm Hexagon SD X Elite：45 TOPS。
- WebLLM M3 Max：Llama 3.1 8B Q4 約 ~41 tok/s。
- AGX Orin：透過 vLLM 執行 gpt-oss-20b 約 ~40 tok/s。
- 資料中心與 edge 的頻寬差距：30-50x。
- WebGPU 行動裝置覆蓋率：約 ~70-75%（Firefox Android 落後）。

```figure
edge-bandwidth-pipe
```

## 實際使用 (Use It)

`code/main.py` 依 bandwidth-bound 的算術，計算各 edge 目標的理論 decode throughput 天花板。與實測 benchmark 比較，並標出瓶頸是頻寬而非運算的地方。

## 交付 (Ship It)

這堂課產出 `outputs/skill-edge-target-picker.md`。給定平台（iOS/Android/瀏覽器/Jetson）、模型，以及 latency/記憶體預算，挑選 quantization 格式與轉換 pipeline。

## 練習 (Exercises)

1. 執行 `code/main.py`。對 Snapdragon 8 Gen 3（~77 GB/s 頻寬）上的 Q4 7B 模型，計算 decode 天花板。與實測的 6-8 tok/s 比較：runtime 有效率嗎？
2. Android 上的 WebGPU 需要 Chrome v121+。為舊版瀏覽器設計 fallback：透過同一個相容 OpenAI 的 API 走伺服器端。
3. 你的 iOS app 需要 4K context 的串流。哪種模型/格式組合能讓你在 iPhone 16 上維持低於 4 GB 的活躍記憶體？
4. Jetson AGX Orin 以 40 tok/s 執行 gpt-oss-20b。Jetson Nano 只放得下 3B。如果你的產品兩者都要支援，如何統一 inference stack？
5. 辯論「WebLLM 在 2026 年已可用於 production」。引用覆蓋率、效能，以及 Firefox Android 的缺口。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| ANE | 「Apple neural engine」 | M 系列與 A 系列中的裝置端 NPU；統一記憶體 |
| Hexagon | 「Qualcomm NPU」 | Snapdragon NPU；透過 QNN SDK 存取 |
| WebGPU | 「瀏覽器 GPU」 | W3C 標準化的瀏覽器 GPU API；2026 年 Chrome/Safari 支援 |
| WebLLM | 「瀏覽器 LLM runtime」 | MLC-LLM 專案；Apache 2.0；相容 OpenAI 的 JS |
| Jetson | 「NVIDIA edge」 | Orin Nano / AGX / Thor / T4000 系列 |
| TRT Edge-LLM | 「edge TensorRT」 | TensorRT-LLM 在 2026 年的 edge 移植版；EAGLE-3 + NVFP4 |
| Unified memory | 「共用 pool」 | CPU 與 NPU 看到同一塊 RAM；沒有複製開銷 |
| Bandwidth-bound | 「受記憶體限制」 | Decode 受限於讀取權重的 bytes/sec |
| Core ML | 「Apple 轉換」 | 用於 ANE 原生模型的 Apple framework |
| QNN | 「Qualcomm stack」 | Qualcomm Neural Network SDK |

## 延伸閱讀 (Further Reading)

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/)：版圖與 benchmark。
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/)：Orin / AGX / Thor。
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/)：2026 年 edge 移植版發布公告。
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2)：設計與 benchmark。
- [Apple Core ML](https://developer.apple.com/documentation/coreml)：ANE 原生轉換。
- [Qualcomm AI Hub](https://aihub.qualcomm.com/)：為 Hexagon 預先轉換好的模型。
