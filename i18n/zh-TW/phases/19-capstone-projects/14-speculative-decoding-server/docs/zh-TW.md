# Capstone 14 — Speculative-Decoding 推論伺服器 (Speculative-Decoding Inference Server)

> Speculative decoding：由便宜的 draft 提出 token，目標模型一次性驗證，如今已是可上線的最佳化，不再是研究技巧。vLLM 0.7 中的 EAGLE-3 在真實流量上帶來 2.5-3 倍的吞吐量。P-EAGLE (AWS 2026) 把平行 speculation 推得更遠。SGLang 的 SpecForge 大規模訓練了 draft heads。Red Hat 的 Speculators hub 發佈了常見開源模型的對齊 draft。TensorRT-LLM 讓 speculative decoding 在 NVIDIA 上成為一等公民。2026 年的正式服務堆疊是 vLLM 或 SGLang 搭配 EAGLE 系列 draft、FP8 或 INT4 量化，以及依佇列等待時間做 HPA。這個 capstone 要你以基準線 2.5 倍以上的吞吐量服務兩個開源模型，並附上完整的尾端延遲報告。

**Type:** Capstone
**Languages:** Python (serving), C++ / CUDA (kernel inspection), YAML (configs)
**Prerequisites:** Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 17 (infrastructure)
**Phases exercised:** P3 · P7 · P10 · P17
**Time:** 30 小時

## 問題 (Problem)

Speculative decoding 在 2026 年成為日用品。EAGLE-3 draft heads 以目標模型的 hidden states 訓練，往前預測 N 個 token；目標模型在單次 pass 中驗證。60-80% 的接受率可轉換為 2-3 倍的端到端吞吐量。vLLM 0.7 原生整合了這個功能。SGLang + SpecForge 給你訓練管線。Red Hat 的 Speculators 發佈了 Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B 的對齊 draft。

功夫在服務營運，不在模型。接受率會隨流量分布漂移 (ShareGPT、程式碼、領域資料)。被拒絕時的尾端延遲比沒有 speculation 時更糟，所以你必須回報多個 batch size 下的 p99，而不只是穩態的 tokens/sec。相對 Anthropic / OpenAI API 的每 1M tokens 成本，是建立可信度的槓桿。

## 核心概念 (Concept)

Speculative decoding 有兩層。**Draft** 模型 (EAGLE-3 head、ngram，或較小的、與目標對齊的模型) 每一步提出 k 個候選 token。**目標** 模型一次 pass 驗證全部 k 個；任何被接受的前綴會取代 greedy 路徑。接受率取決於 draft 與目標的對齊程度，以及輸入分布。

EAGLE-3 在大多數流量上勝過 ngram draft。P-EAGLE 執行平行 speculation，產生更深的 draft 樹。取捨是：被拒絕時 P99 延遲較高，因為驗證 pass 較大。服務設定必須回報依 batch size 分桶的延遲，才能把這點顯現出來。

部署用 Kubernetes。vLLM 0.7 每個 GPU 或 tensor-parallel 分片跑一個副本。HPA 依佇列等待時間而不是 CPU 自動擴縮。FP8 (Marlin) 與 INT4 (AWQ) 量化讓 GPU 記憶體維持在 H100 / H200 的範圍內。端到端報告包含吞吐量、接受率、batch 1/8/32 的 p50/p99，以及 $/1M tokens。

## 架構 (Architecture)

```
request ingress
    |
    v
vLLM server (0.7) or SGLang (0.4)
    |
    +-- draft: EAGLE-3 heads | P-EAGLE parallel | ngram fallback
    +-- target: Llama 3.3 70B | Qwen3-Coder-30B | GPT-OSS-120B
    |     quantized FP8-Marlin or INT4-AWQ
    |
    v
verify pass: batch k draft tokens through target
    |
    v (accept prefix; resample for rejected suffix)
    v
token stream back to client
    |
    v
Prometheus metrics: throughput, acceptance rate, queue wait, latency p50/p99
    |
    v
HPA on queue-wait metric
```

## 技術堆疊 (Stack)

- 服務：vLLM 0.7 或 SGLang 0.4
- Speculative 方法：EAGLE-3 draft heads、P-EAGLE 平行 speculation、ngram 備援
- Draft 訓練：SpecForge (SGLang) 或 Red Hat Speculators
- 目標模型：Llama 3.3 70B、Qwen3-Coder-30B MoE、GPT-OSS-120B
- 量化：FP8 (Marlin)、INT4 AWQ
- 部署：Kubernetes + NVIDIA device plugin；依佇列等待指標做 HPA
- 評測：ShareGPT、MT-Bench-v2、GSM8K、HumanEval，用於跨領域的接受率量測
- 參考：TensorRT-LLM speculative decoding，作為廠商基準線

```figure
cf-spec-decode
```

## 動手做 (Build It)

1. **目標模型準備。** 選 Llama 3.3 70B。透過 Marlin 量化為 FP8。在 1xH100 (或 2x tensor-parallel) 的 vLLM 0.7 上部署。

2. **Draft 來源。** 從 Red Hat Speculators 取得對齊的 EAGLE-3 draft head (或用 SpecForge 自己訓練)。載入 vLLM 的 speculative-decoding 設定。

3. **基準線數字。** 啟用 speculation 之前：batch 1/8/32 的 tokens/s、p50/p99 延遲、GPU 使用率。發佈。

4. **啟用 EAGLE-3。** 切換設定；重跑同一個 benchmark。回報加速比、接受率、p99 尾端延遲的差異。

5. **P-EAGLE。** 啟用平行 speculation；衡量更深的 draft 樹與序列式 EAGLE-3 的差別。回報 P-EAGLE 從有幫助變成有害的轉折點。

6. **領域流量。** 讓 ShareGPT、HumanEval 與領域專屬流量通過同一台伺服器。衡量各分布的接受率。找出 draft 何時漂移。

7. **第二個目標模型。** 對 Qwen3-Coder-30B MoE 執行同樣的管線。Draft 比較棘手 (MoE 路由雜訊)。回報結果。

8. **K8s HPA。** 在 K8s 下部署，HPA 追蹤 `queue_wait_ms`。展示負載增為三倍時的橫向擴展。

9. **成本比較。** 在相同評測下，計算與 Anthropic Claude Sonnet 4.7 和 OpenAI GPT-5.4 比較的 $/1M tokens。發佈。

## 實際使用 (Use It)

```
$ curl https://infer.example.com/v1/chat/completions -d '{"messages":[...]}'
[serve]     vLLM 0.7, Llama 3.3 70B FP8, EAGLE-3 active
[decode]    bs=8, accepted_tokens_per_step=3.2, acceptance_rate=0.76
[latency]   first-token 42ms, full-response 980ms (620 tokens)
[cost]      $0.34 per 1M output tokens at sustained throughput
```

## 交付 (Ship It)

`outputs/skill-inference-server.md` 描述交付成果。一個經過量測、具備 speculative decoding 的服務堆疊，附上完整的 benchmark 報告與 K8s 部署。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 相對基準線的實測加速 | 在兩個模型上、品質相同的條件下吞吐量 2.5 倍以上 |
| 20 | 真實流量的接受率 | 各分布的接受率報告 |
| 20 | P99 尾端延遲紀律 | 有無 speculation 時 batch 1/8/32 的 p99 |
| 20 | 營運 | K8s 部署、依佇列等待做 HPA、發佈順暢 |
| 15 | 報告與方法論 | 清楚說明改了什麼、為什麼 |
| **100** | | |

## 練習 (Exercises)

1. 衡量 draft 比目標落後一個版本時的接受率退化 (例如 Llama 3.3 -> 3.4 的漂移)。建立監控警示。

2. 實作 ngram 備援：如果 EAGLE-3 接受率掉到閾值以下，就切換到 ngram draft。回報可靠度的改善。

3. 執行對照式 MoE 實驗：同一個 Qwen3-Coder-30B，注入路由雜訊與不注入。衡量 draft 接受率的敏感度。

4. 擴充到 H200 (141 GB)。回報每個副本能容納的模型大小增加了多少，以及是否能服務未量化的 Llama 3.3 70B。

5. 在相同的 H100 硬體上對 TensorRT-LLM speculative decoding 做 benchmark。回報它在哪裡勝過 vLLM。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Draft model | 「Speculator」 | 為目標模型提出 N 個 token 供驗證的小模型 |
| EAGLE-3 | 「2026 draft 架構」 | 以目標模型 hidden states 訓練的 draft head；約 75% 接受率 |
| P-EAGLE | 「平行 speculation」 | 在一次目標 pass 中驗證的 draft 分支樹 |
| Acceptance rate | 「命中率」 | 不需重新取樣就被接受的 draft token 比例 |
| Quantization | 「FP8 / INT4」 | 用較低精度的權重，在 GPU 記憶體中放進更多模型 |
| Queue wait | 「HPA 指標」 | 請求在待處理佇列中等待、直到推論開始的時間 |
| Speculators hub | 「對齊的 draft」 | Red Hat Neural Magic 提供的常見開源模型 EAGLE draft hub |

## 延伸閱讀 (Further Reading)

- [vLLM EAGLE and P-EAGLE documentation](https://docs.vllm.ai) — 參考的服務堆疊
- [P-EAGLE (AWS 2026)](https://aws.amazon.com/blogs/machine-learning/p-eagle-faster-llm-inference-with-parallel-speculative-decoding-in-vllm/) — 平行 speculative decoding 論文 + 整合
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — draft-head 訓練管線
- [Red Hat Speculators](https://github.com/neuralmagic/speculators) — 對齊 draft hub
- [TensorRT-LLM speculative decoding](https://nvidia.github.io/TensorRT-LLM/) — 廠商替代方案
- [Fireworks.ai serving architecture](https://fireworks.ai/blog) — 商用參考
- [EAGLE-3 paper (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — 方法論文
- [vLLM repository](https://github.com/vllm-project/vllm) — 程式碼與 benchmark
