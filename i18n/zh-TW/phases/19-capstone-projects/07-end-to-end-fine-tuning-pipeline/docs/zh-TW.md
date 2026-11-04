# Capstone 07 — 端到端 Fine-Tuning 管線 (End-to-End Fine-Tuning Pipeline，資料到 SFT 到 DPO 到部署)

> 用你自己的資料訓練 8B 等級的模型，用你自己的偏好做 DPO 對齊，量化、套用 speculative decoding，並以可衡量的 $/1M tokens 提供服務。2026 年的開源堆疊是：Axolotl v0.8、TRL 0.15、用於快速迭代的 Unsloth、用於量化的 GPTQ/AWQ/GGUF，以及搭配 EAGLE-3 做服務的 vLLM 0.7。這個 capstone 要你可重現地跑完整條管線 (YAML 進，上線的 endpoint 出)，並依 2026 Model Openness Framework 發佈 model card。

**Type:** Capstone
**Languages:** Python (pipeline), YAML (configs), Bash (scripts)
**Prerequisites:** Phase 2 (ML), Phase 3 (DL), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 11 (LLM engineering), Phase 17 (infrastructure), Phase 18 (safety)
**Phases exercised:** P2 · P3 · P7 · P10 · P11 · P17 · P18
**Time:** 35 小時

## 問題 (Problem)

2026 年每個認真的 AI 團隊都隨時備有 fine-tuning 管線。不是因為他們要發佈前沿的 base model，而是因為下游適配 (領域 SFT、針對標註偏好的 DPO、用於 speculative decoding 的蒸餾 draft、搭配 EAGLE-3 服務) 才是可衡量收益所在。Axolotl v0.8 處理多 GPU 的 SFT 設定。TRL 0.15 處理 DPO 和 GRPO。Unsloth 讓你在單 GPU 上快速迭代。vLLM 0.7 搭配 EAGLE-3，在品質不損失的情況下把 decode 吞吐量提高 2-3 倍。工具都能用；功夫在 YAML、資料衛生與評測紀律。

你要拿 8B 到 14B 的 base (Llama 3.1 8B、Qwen3 14B 或 Gemma 3 12B)，用任務專屬資料先做 SFT 再做 DPO，量化後提供服務，並對照 lm-evaluation-harness、RewardBench-2、MT-Bench-v2 與 MMLU-Pro 衡量增益。你要依 2026 Model Openness Framework 產出 model card。重點是可重現性：一行指令就能端到端重跑整條管線。

## 核心概念 (Concept)

管線有五個階段。**資料**：去重 (MinHash / Datatrove)、品質過濾 (Nemotron-CC 風格分類器)、PII 清除，以及針對公開 benchmark 污染的 split 衛生檢查。**SFT**：Axolotl YAML、8xH100 上的 ZeRO-3、cosine schedule、packed sequences、2-3 個 epoch。**DPO 或 GRPO**：TRL 設定、1 個 epoch、偏好配對可以是人工標註或模型評判、調整 beta。**量化**：GPTQ + AWQ + GGUF，讓部署更有彈性。**服務**：vLLM 0.7 搭配 EAGLE-3 speculative heads (或 SGLang 搭配 SpecForge)、K8s deployment、依佇列等待時間做 HPA。

Ablation 就是交付成果：在三個任務專屬 benchmark 上比較 SFT-only、SFT+DPO 與 SFT+GRPO。服務指標：batch 1 / 8 / 32 的 tokens/s、EAGLE-3 接受率、$/1M tokens。安全評測：Llama Guard 4 通過率。Model card：偏誤評估、可重現的 seed、資料授權。

## 架構 (Architecture)

```
raw data (HF datasets + internal)
    |
    v
Datatrove dedup + Nemotron-CC quality filter + PII scrub
    |
    v
split hygiene (MMLU-Pro contamination check)
    |
    v
Axolotl SFT config (YAML)  ---> 8xH100, ZeRO-3
    |
    v
TRL DPO / GRPO config       ---> 4xH100, 1 epoch
    |
    v
GPTQ + AWQ + GGUF quantize
    |
    v
vLLM 0.7 + EAGLE-3 speculative decoding
    |
    v
K8s deployment, HPA on queue-wait
    |
    v
lm-eval-harness + RewardBench-2 + MT-Bench-v2 + MMLU-Pro
    |
    v
model card (2026 MOF) + safety eval (Llama Guard 4)
```

## 技術堆疊 (Stack)

- 資料：去重用 Datatrove，品質用 Nemotron-CC 分類器，PII 用 Presidio
- Base：Llama 3.1 8B、Qwen3 14B 或 Gemma 3 12B
- SFT：Axolotl v0.8，搭配 ZeRO-3、Flash Attention 3、packed sequences
- 偏好調校：DPO 或 GRPO 用 TRL 0.15；單 GPU 迭代用 Unsloth
- 量化：GPTQ (Marlin)、AWQ、透過 llama.cpp 的 GGUF
- 服務：vLLM 0.7 搭配 EAGLE-3 speculative decoding (或 SGLang 0.4 + SpecForge)
- 評測：lm-evaluation-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro
- 安全評測：Llama Guard 4、ShieldGemma-2
- 基礎設施：Kubernetes + NVIDIA device plugin，依佇列等待指標做 HPA
- 可觀測性：訓練用 W&B，推論用 Langfuse

```figure
ce-finetune-stages
```

## 動手做 (Build It)

1. **資料管線。** 對原始資料集執行 Datatrove 去重。套用 Nemotron-CC 風格的品質分類器。Presidio 清除 PII。以明確的 seed 寫出 train/val 切分。

2. **污染檢查。** 對每個驗證切分，與 MMLU-Pro、MT-Bench-v2、RewardBench-2 測試集計算 MinHash。拒絕任何重疊。

3. **Axolotl SFT。** 使用 ZeRO-3、FA3、sequence packing 的 YAML。在 8xH100 上跑 2-3 個 epoch。記錄到 W&B。

4. **TRL DPO / GRPO。** 取 SFT checkpoint，對偏好配對跑一個 epoch 的 DPO (或在數學/程式碼上以可驗證獎勵做 GRPO)。掃描 beta。

5. **量化。** 產出三種量化：GPTQ-INT4-Marlin、AWQ-INT4、給 llama.cpp 的 GGUF-Q4_K_M。記錄大小與名義吞吐量。

6. **搭配 speculative decoding 服務。** vLLM 0.7 設定，使用透過 Red Hat Speculators 訓練的 EAGLE-3 draft heads。衡量 batch 1 / 8 / 32 的接受率與尾端延遲。在相同評測下，回報 $/1M tokens 與 Anthropic / OpenAI 的比較。

7. **評測矩陣。** 對 base、SFT-only、SFT+DPO、SFT+GRPO 執行 lm-eval-harness、RewardBench-2、MT-Bench-v2、MMLU-Pro。產出表格。

8. **安全評測。** 在 dev 集上計算 Llama Guard 4 通過率。加上 ShieldGemma-2 輸出過濾器。

9. **Model card。** MOF 2026 範本：資料、訓練、評測、安全、授權，以及附有 YAML 與 commit SHA 的可重現性章節。

## 實際使用 (Use It)

```
$ ./pipeline.sh config/llama3.3-8b-domainX.yaml
[data]    300k deduped, 12k filtered, 280k accepted (seed=7)
[SFT]     3 epochs, 8xH100, 6h12m, val loss 1.42 -> 1.03
[DPO]     1 epoch, beta=0.08, 4xH100, 1h40m
[quant]   GPTQ-INT4 4.6 GB, AWQ-INT4 4.8 GB, GGUF-Q4_K_M 5.1 GB
[serve]   vLLM 0.7, EAGLE-3 acceptance 0.74, p99 126ms @ bs=8
[eval]    MMLU-Pro +3.2, MT-Bench-v2 +0.41, RewardBench-2 +0.08
[card]    model-card.md generated under 2026 MOF
```

## 交付 (Ship It)

`outputs/skill-finetuning-pipeline.md` 描述交付成果。一行指令就能讓資料經過 SFT、DPO、量化、服務、評測，並輸出 model card + 上線的 endpoint。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 相對 base 的評測差異 | 目標任務 (MMLU-Pro、MT-Bench-v2、任務專屬) 上的實測增益 |
| 20 | 管線可重現性 | 一行指令以相同 seed 端到端重跑 |
| 20 | 資料衛生 | 去重率、PII 清除涵蓋率、污染檢查通過 |
| 20 | 服務效率 | bs=1/8/32 的 tokens/s、EAGLE-3 接受率、$/1M tokens |
| 15 | Model card + 安全評測 | 2026 MOF 完整度 + Llama Guard 4 通過率 |
| **100** | | |

## 練習 (Exercises)

1. 在相同的任務專屬 benchmark 上執行 SFT-only、SFT+DPO 與 SFT+GRPO。回報哪種偏好方法勝出、差多少。

2. 把 Llama 3.1 8B 換成 Qwen3 14B。在相同品質下衡量 $/1M tokens。

3. 衡量 EAGLE-3 在領域資料與通用 ShareGPT 上的接受率。回報差異，以及它對延遲預算的意義。

4. 注入 1% 的污染 (把 MMLU-Pro 答案洩漏到訓練資料)，重跑評測。觀察 MMLU-Pro 準確度不合理地飆升。建立能抓到這種狀況的污染檢查 CI 關卡。

5. 加入 LoRA SFT 作為完整 fine-tune 的替代方案。在記憶體少 10 倍的情況下衡量品質差距。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Axolotl | 「SFT trainer」 | 以 YAML 驅動、統一處理 SFT、DPO 與蒸餾的 trainer |
| TRL | 「偏好調校器」 | Hugging Face 用於 LLM 的 DPO、GRPO、PPO 函式庫 |
| GRPO | 「Group-relative policy optimization」 | DeepSeek R1 以可驗證獎勵進行的 RL 配方 |
| EAGLE-3 | 「Speculative decoding draft」 | 預測往後 N 個 token 的 draft heads；vLLM 用目標模型驗證 |
| MOF | 「Model Openness Framework」 | 2026 年依資料、程式碼、授權為模型釋出評分的標準 |
| Contamination check | 「Split 衛生」 | 以 MinHash 偵測測試集洩漏到訓練資料 |
| Acceptance rate | 「EAGLE / MTP 指標」 | 目標模型接受的 draft token 比例 |

## 延伸閱讀 (Further Reading)

- [Axolotl documentation](https://axolotl-ai-cloud.github.io/axolotl/) — 參考用的 SFT / DPO trainer
- [TRL documentation](https://huggingface.co/docs/trl) — DPO 與 GRPO 的參考實作
- [Unsloth](https://github.com/unslothai/unsloth) — 單 GPU 迭代的參考
- [DeepSeek R1 paper (arXiv:2501.12948)](https://arxiv.org/abs/2501.12948) — GRPO 方法論
- [vLLM + EAGLE-3 documentation](https://docs.vllm.ai) — 參考的服務堆疊
- [SGLang SpecForge](https://github.com/sgl-project/SpecForge) — 另一個 speculative-decoding trainer
- [Model Openness Framework (arXiv:2403.13784)](https://arxiv.org/abs/2403.13784)：開放釋出的評分標準
- [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness) — 標準評測執行器
