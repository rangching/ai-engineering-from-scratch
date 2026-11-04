# Inference 平台經濟學 (Inference Platform Economics)：Fireworks、Together、Baseten、Modal、Replicate、Anyscale

> 2026 年的 inference 市場已經不只是出租 GPU 時間。它分成三條路：客製晶片 (Groq、Cerebras、SambaNova)、GPU 平台 (Baseten、Together、Fireworks、Modal)，以及 API 優先的市集 (Replicate、DeepInfra)。Fireworks 在 2026 年 5 月 1 日把每個 GPU 的價格調漲 $1/hr，並以每天 10T+ tokens 取得 $4B 估值，顯示以量取勝的模式行得通。Baseten 在 2026 年 1 月完成 $300M 的 Series E，估值 $5B。競爭定位的規則很簡單：Fireworks 優化 latency，Together 優化目錄廣度，Baseten 優化企業級完整度，Modal 優化 Python 原生的 DX，Replicate 優化 multimodal 覆蓋，Anyscale 優化分散式 Python。這堂課給你一份可以直接交給創辦人的比較矩陣。

**Type:** Learn
**Languages:** Python (stdlib, toy per-call economics comparator)
**Prerequisites:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (Serving Engine Internals)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出三個市場區隔（客製晶片、GPU 平台、API 優先），並把每家廠商對應到一個區隔。
- 解釋為什麼「per-token」API 定價會向 serving engine 的成本曲線收斂，而不是向硬體的成本曲線收斂。
- 計算至少三家廠商的每個 request 有效成本，並解釋什麼時候 per-minute (Baseten、Modal) 勝過 per-token。
- 針對特定工作負載（serverless 突發、穩定高 throughput、fine-tuned 變體、multimodal），找出合適的預設平台。

## 問題 (The Problem)

你已經評估過託管的 hyperscaler 平台，結論是需要更專注、更快的供應商：要 latency 選 Fireworks，要廣度選 Together，要 fine-tuned 的客製模型選 Baseten。現在你有六個實際的選項，但定價頁面對不起來。Fireworks 顯示 $/M tokens；Baseten 顯示 $/minute；Modal 顯示 $/second；Replicate 顯示 $/prediction。不替工作負載建模，就無法直接比較。

更糟的是，每個定價頁面背後的商業模式都不同。Fireworks 在共用 GPU 上執行自家的客製 engine (FireAttention)，per-token 費率反映的是它們的使用率曲線。Baseten 給你 Truss 加專屬 GPU，per-minute 反映的是獨占性。Modal 是真正的 Python serverless，按秒計費，cold start 不到一秒。同樣的輸出（一則 LLM 回應），卻有三種不同的成本函數。

這堂課替這六家建模，並告訴你各自在什麼情況下勝出。

## 核心概念 (The Concept)

### 三個區隔

**客製晶片**：Groq (LPU)、Cerebras (WSE)、SambaNova (RDU)。在同一個模型上，decode 通常比 GPU 叢集快 5-10 倍。per-token 價格較高（2025 年底 Groq 在 Llama-70B 約 ~$0.99/M），但在對 latency 敏感的情境中無可匹敵。Groq 是語音 agent 與即時翻譯的 production 首選。

**GPU 平台**：Baseten、Together、Fireworks、Modal、Anyscale。執行在 NVIDIA（2026 年是 H100、H200、B200）或偶爾 AMD 上。它們是介於「純 GPU 出租」(RunPod、Lambda) 與「hyperscaler 託管服務」(Bedrock) 之間的經濟層。

**API 優先的市集**：Replicate、DeepInfra、OpenRouter、Fal。目錄廣、按 prediction 或按秒計費，強調第一次呼叫的上手時間。

### Fireworks：優化 latency 的 GPU 平台

- FireAttention engine（客製）；行銷宣稱在同等設定下 latency 比 vLLM 低 4 倍。
- 針對非互動式工作負載的 batch tier，約為 serverless 費率的 ~50%。
- Fine-tuned 模型以與 base 模型相同的費率提供服務，這與對你的 LoRA 收取溢價的供應商相比是真正的差異。
- 2026 年中：自 2026 年 5 月 1 日起，on-demand GPU 出租價格調漲 $1/hour。大量使用可議價。
- 財務訊號：$4B 估值，每天處理 10T+ tokens。

### Together：優化廣度

- 200+ 個模型，包括在上游發布後幾天內就上架的開源模型。
- 在同等 LLM 模型上比 Replicate 便宜 50-70%。「AI Native Cloud」的定位就是量與目錄。
- Inference + fine-tuning + training 在同一個 API 中。

### Baseten：優化企業級完整度

- Truss framework：把相依套件、secrets、serving 設定包在同一個 manifest 的模型封裝。
- GPU 從 T4 到 B200。按分鐘計費，並有合理的 cold-start 緩解。
- SOC 2 Type II、HIPAA-ready。是 fintech 與醫療常見的選擇。
- $5B 估值，2026 年 1 月 Series E（$300M，來自 CapitalG、IVP、NVIDIA）。

### Modal：優化 Python 原生體驗

- 以純 Python 寫的 infrastructure-as-code。用 `@modal.function(gpu="A100")` 裝飾一個函式，一個指令就能部署。
- 按秒計費。預先 warm 時 cold start 2-4 秒；小模型 <1 秒。
- $87M Series B，估值 $1.1B (2025)。在獨立調查中開發者體驗評分最高。

### Replicate：multimodal 廣度

- 按 prediction 計費。是影像、影片、音訊模型的預設平台。
- 整合生態系 (Zapier、Vercel、CMS 外掛)。
- 在 LLM per-token 費率上較沒競爭力，但勝在 multimodal 的多樣性。

### Anyscale：Ray 原生

- 建立在 Ray 之上；RayTurbo 是 Anyscale 專有的 inference engine（與 vLLM 競爭）。
- 最適合 inference 只是大型運算圖中一個節點的分散式 Python 工作負載。
- 託管的 Ray 叢集；與 Ray AIR 和 Ray Serve 緊密整合。

### Per-token 與 per-minute：各自何時勝出

當工作負載對 latency 不敏感且有突發性時，per-token 比較合理，因為你只為用到的部分付費。當使用率高且可預測時，per-minute 比較合理，一旦把 GPU 吃滿，就會勝過 per-token。

粗略規則：對專屬 GPU 持續使用率超過 ~30% 的工作負載，per-minute (Baseten、Modal) 開始勝過 per-token (Fireworks、Together)。低於這個值，per-token 勝出，因為你不用為閒置付費。

### 客製 engine 才是真正的護城河

每個位於 vLLM 和 SGLang 之上的平台都宣稱擁有客製 engine：FireAttention、RayTurbo、Baseten 的 inference stack。客製 engine 的說法帶有行銷色彩。誠實的說法是：vLLM + SGLang 約占 production 開源 inference 的 80%，平台層的差異在於 DX、attribution 與 SLAs。

### 該記住的數字

- Fireworks GPU 出租：自 2026 年 5 月 1 日起調漲 $1/hr。
- Fireworks 宣稱：同等設定下 latency 比 vLLM 低 4 倍。
- Together：LLM 比 Replicate 便宜 50-70%。
- Baseten 估值：$5B (Series E，2026 年 1 月，$300M 募資)。
- Modal 估值：$1.1B (Series B，2025)。
- 持續使用率超過 ~30% 時，per-minute 勝過 per-token。

```figure
cost-per-token
```

## 實際使用 (Use It)

`code/main.py` 在一個合成工作負載上，跨定價模型比較這六家廠商，回報 $/day 與有效 $/M tokens。執行它，找出 per-token 與 per-minute 的損益平衡點。

## 交付 (Ship It)

這堂課產出 `outputs/skill-inference-platform-picker.md`。給定工作負載描述、SLA 與預算，選出主要 inference 平台並指出次選。

## 練習 (Exercises)

1. 執行 `code/main.py`。對單張 H100 上的 70B 模型，持續使用率到多少時 Baseten (per-minute) 勝過 Fireworks (per-token)？自行推導交叉點，並與經驗法則比較。
2. 你的產品同時提供影像生成、聊天和 speech-to-text。為每種 modality 選平台，並說出能統一它們的 gateway 模式。
3. Fireworks 把你主要模型的價格調漲 $1/hr。如果 40% 的流量改走 batch tier（便宜 50%），建模計算混合成本的影響。
4. 一位受監管的客戶要求 SOC 2 Type II + HIPAA + 專屬 GPU。哪三個平台可行？哪一個在 FinOps 上勝出？
5. 比較 Llama 3.1 70B 在 Fireworks serverless、Together on-demand、Baseten 專屬，以及 Replicate API 上每 1,000 次 prediction 的成本。每天 10 次 prediction 時哪個最便宜？每天 10,000 次呢？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Custom silicon | 「非 GPU 晶片」 | Groq LPU、Cerebras WSE、SambaNova RDU，針對 decode 優化 |
| FireAttention | 「Fireworks engine」 | 客製 attention kernel；行銷宣稱 latency 比 vLLM 低 4 倍 |
| Truss | 「Baseten 的格式」 | 模型封裝 manifest；相依套件 + secrets + serving 設定 |
| Per-token | 「API 定價」 | 依消耗的 tokens 收費；不用為閒置付費 |
| Per-minute | 「專屬定價」 | 依實際經過的 GPU 時間收費；高使用率時勝出 |
| Per-prediction | 「Replicate 定價」 | 每次模型呼叫收費；常見於影像/影片 |
| RayTurbo | 「Anyscale engine」 | 在 Ray 上的專有 inference；在 Ray 叢集上與 vLLM 競爭 |
| Batch tier | 「五折」 | 以較低費率處理的非互動式佇列；常見於 Fireworks、OpenAI |
| Fine-tuned at base rate | 「Fireworks LoRA」 | LoRA 服務的 request 以 base 模型費率計費（差異點） |

## 延伸閱讀 (Further Reading)

- [Fireworks Pricing](https://fireworks.ai/pricing)：per-token 費率、batch tier、GPU 出租。
- [Baseten Pricing](https://www.baseten.co/pricing/)：per-minute 費率、承諾容量、企業方案。
- [Modal Pricing](https://modal.com/pricing)：按秒計費的 GPU 費率與免費額度。
- [Together AI Pricing](https://www.together.ai/pricing)：模型目錄與 per-token 費率。
- [Anyscale Pricing](https://www.anyscale.com/pricing)：RayTurbo 與託管 Ray 定價。
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference)：比較評估。
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared)：廠商全景。
