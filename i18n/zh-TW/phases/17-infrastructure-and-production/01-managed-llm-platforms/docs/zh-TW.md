# 託管 LLM 平台 (Managed LLM Platforms)：Bedrock、Vertex AI、Azure OpenAI

> 三家 hyperscaler，三種不同策略。AWS Bedrock 是模型市集：Claude、Llama、Titan、Stability、Cohere 都在同一個 API 後面。Azure OpenAI 是與 OpenAI 的獨家合作，再加上提供專屬容量的 Provisioned Throughput Units (PTUs)。Vertex AI 以 Gemini 為先，擁有最強的 long-context 與 multimodal 表現。2026 年 Artificial Analysis 測得 Azure OpenAI 的 median 約 ~50 ms，Bedrock 在 Llama 3.1 405B 等級約 ~75 ms。這個差距來自 PTUs，因為專屬容量勝過共用的 on-demand。決策規則不是「哪個最快」，而是「哪個模型目錄與 FinOps 介面符合我的產品」。這堂課教你把取捨寫下來再做選擇，而不是憑感覺。

**Type:** Learn
**Languages:** Python (stdlib, toy cost-and-latency comparator)
**Prerequisites:** Phase 11 (LLM Engineering), Phase 13 (Tools & Protocols)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 說出三種平台策略（市集 vs 獨家 vs Gemini 為先），並將每種對應到產品使用情境。
- 解釋 Azure OpenAI 中 Provisioned Throughput Units (PTUs) 買到什麼，以及為什麼 on-demand 的 Bedrock 在 405B 規模通常慢約 ~25 ms。
- 畫出各平台的 FinOps attribution 介面（Bedrock Application Inference Profiles vs Vertex 每團隊一個 project vs Azure scopes + PTU reservations）。
- 寫下「至少兩家供應商」的政策，並解釋為什麼單一廠商 lock-in 是 2026 年代價最高的錯誤。

## 問題 (The Problem)

你的產品選了 Claude 3.7 Sonnet，現在要把它跑起來。你可以直接呼叫 Anthropic API，也可以透過 AWS Bedrock，或是經過 gateway。直接呼叫 API 最簡單；Bedrock 多了 BAAs、VPC endpoints、IAM 與 CloudWatch attribution。Gateway 則提供 failover、統一計費，以及跨供應商的 rate limits。

更深一層的問題是目錄。如果同一個產品要同時用 Claude、Llama 和 Gemini，你無法從單一地方全部買齊，除非同時使用 Bedrock、Vertex 和 Azure OpenAI。hyperscaler 並不能互相取代，因為它們對「誰掌控模型層」押了不同的賭注。

這堂課整理三種賭注、latency 差距、FinOps 差距，以及 lock-in 風險。

## 核心概念 (The Concept)

### 三種策略

**AWS Bedrock**：模型市集。Claude (Anthropic)、Llama (Meta)、Titan (AWS 自家)、Stability (影像)、Cohere (embeddings)、Mistral，另外還有影像與 embedding 的子目錄。一個 API、一個 IAM 介面、一個 CloudWatch export。Bedrock 的賭注是：客戶想要的是選擇彈性，而不是單一模型。

**Azure OpenAI**：獨家合作。你能在 Azure 資料中心使用 GPT-4 / 4o / 5 / o-series、DALL·E、Whisper，以及 OpenAI 模型的 fine-tuning。「Azure OpenAI Service」目錄裡沒有非 OpenAI 的模型，那些在 Azure AI Foundry（另一個產品）。Azure 的賭注是：OpenAI 持續站在最前線，而客戶想在這段特定關係上取得企業級控管。

**Vertex AI**：Gemini 第一，其他第二。Gemini 1.5 / 2.0 / 2.5 Flash 與 Pro，加上 Model Garden（第三方）。Vertex 的賭注是 multimodal long-context，1M-token 的 Gemini context 是它的差異化優勢。

### 規模化下的 latency 差距

Artificial Analysis 持續執行 benchmark。在同等的 Llama 3.1 405B 部署上，Azure 搭配 PTUs 的 median first-token latency 約 50 ms；Bedrock 的共用 on-demand 約 75 ms。這個差距不是 AWS 的失誤，而是容量模型不同。Azure 賣 PTUs (Provisioned Throughput Units)，替你的 tenant 保留 GPU 容量。Bedrock 也有對應的 Provisioned Throughput，但每個單位約從 $21/小時起跳，多數客戶仍使用共用的 on-demand。

On-demand 的共用容量要和其他所有客戶的流量競爭，專屬容量則不用。如果你的產品 SLA 是 P99 下 TTFT < 100 ms，你要嘛在 Azure 買 PTUs，要嘛買 Bedrock Provisioned Throughput，要嘛接受預設的波動。

### Provisioned Throughput 的經濟學

Azure PTUs：一塊保留的 inference 運算資源。對可預測的工作負載，相較 on-demand 最多可省約 ~70%。不論流量多少，每小時成本固定，閒置時也要為保留付費。損益平衡點通常在持續使用率 40-60% 左右。

Bedrock Provisioned Throughput：依模型和區域而定，每小時 $21-$50。算法類似，損益平衡點約在尖峰使用率的一半。需要簽訂月度承諾。

Vertex 的 provisioned capacity 依 Gemini SKU 販售，價格隨模型與區域而異，公開資訊也較少。

### FinOps 介面：真正的差異

**Bedrock Application Inference Profiles** 是市集中最乾淨的 attribution。為 profile 加上 `team`、`product`、`feature` 標籤，讓所有模型呼叫都經過它，CloudWatch 不需後處理就能拆出每個 profile 的成本。2025 年加入，至今仍是 hyperscaler 原生功能中最細緻的。

**Vertex** 的 attribution 是每個團隊一個 project，再加上到處貼 label。你把每個團隊建模成一個 GCP project，在每個資源上加 label，再用 BigQuery Billing Export + DataStudio 做彙整。比較費工，但 BigQuery 讓你能對成本資料執行任意 SQL。

**Azure** 依賴 subscription/resource-group scopes 加上 tags，並把 PTU reservations 當成一等公民的成本物件。Tags 是從 resource group 繼承，而不是從 request 繼承，所以要做到每個 request 的 attribution，需要 Application Insights 自訂 metrics，或是會蓋上 header 的 gateway。

規律是：Bedrock 的原生功能最乾淨，Vertex 透過 BigQuery 最有彈性，Azure 除非你自己加 instrumentation，否則最不透明。

### Lock-in 是 2026 年的風險

當單一模型獨大時，只承諾一家 hyperscaler 沒問題。2026 年前沿每個月都在移動：這一季是 Claude 3.7，下一季是 Gemini 2.5，再下一季是 GPT-5。綁死一個平台，就等於被擋在三分之二的前沿之外。

有效的團隊採用的做法：任何產品關鍵的 LLM 呼叫，至少用兩家供應商。常見的組合是 Bedrock 加 Azure OpenAI：Claude 從一邊來，GPT 從另一邊來，兩者之間做 failover，共用同一個 gateway。成本增加可以忽略，因為 gateway 會挑最佳路由；而在停機期間（例如 2025 年 1 月的 Azure OpenAI 事件、AWS us-east-1 停機），可用性的提升是決定性的。

### 資料落地、BAAs 與受監管產業

Bedrock：多數區域提供 BAAs；有 VPC endpoints；有 guardrails。是 fintech 常見的預設選擇。
Azure OpenAI：HIPAA、SOC 2、ISO 27001；有歐盟資料落地；是受監管企業的預設選擇。
Vertex：HIPAA、GDPR、依區域的資料落地；屬於 Google Cloud 的合規體系。

三者都符合基本的勾選項目。差異在於資料保留政策、log 的處理方式，以及 abuse-monitoring 是否會讀取你的流量（多數預設為 opt-in；企業可 opt-out）。

### 該記住的數字

- Azure OpenAI 在 Llama 3.1 405B 等級的 median TTFT：~50 ms（搭配 PTUs）。
- Bedrock on-demand 的 median TTFT：~75 ms。
- Bedrock Provisioned Throughput：每單位 $21-$50/hr。
- Azure PTU 損益平衡點：持續使用率約 ~40-60%。
- 高使用率下，PTU 相較 on-demand 的節省：最多 70%。

```figure
i4-platform-lanes
```

## 實際使用 (Use It)

`code/main.py` 在一個合成工作負載上比較三個平台。它模擬 on-demand vs PTU 的經濟學、TTFT 波動，以及成本 attribution 的精確度。執行它，看看 PTUs 在哪裡划算，以及市集的模型廣度何時勝過 TTFT 差距。

## 交付 (Ship It)

這堂課產出 `outputs/skill-managed-platform-picker.md`。給定工作負載描述（所需模型、TTFT SLA、每日量、合規需求），它會推薦主要平台、備援平台，以及 FinOps instrumentation 計畫。

## 練習 (Exercises)

1. 執行 `code/main.py`。對 70B 等級的模型，持續使用率到多少時 Azure PTU 勝過 on-demand？計算損益平衡點，並與宣稱的 40-60% 區間比較。
2. 你的產品需要 Claude 3.7 Sonnet 和 GPT-4o。設計一個雙供應商部署：哪個放哪個 hyperscaler、前面放什麼 gateway、failover 政策是什麼？
3. 一位受監管的醫療客戶要求 BAAs、US-East 資料落地，以及 P99 TTFT 低於 100ms。選一個平台，並用三項具體功能說明理由。
4. 你發現這個月 Bedrock 帳單暴增 4 倍，但流量沒有變化。沒有 Application Inference Profiles 時，你要怎麼找出元凶？有 profiles 時，要花多久？
5. 閱讀 Anthropic API 與 Bedrock 的定價頁面。對每月 100M-token 的 Claude 工作負載，哪個比較便宜：直接用 Anthropic API、Bedrock on-demand，還是 Bedrock Provisioned Throughput？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Bedrock | 「AWS LLM 服務」 | 涵蓋 Claude、Llama、Titan、Mistral、Cohere 的模型市集 |
| Azure OpenAI | 「Azure 的 ChatGPT」 | 在 Azure 資料中心提供的獨家 OpenAI 模型，附企業級控管 |
| Vertex AI | 「Google 的 LLM」 | 以 Gemini 為先的平台，Model Garden 提供第三方模型 |
| PTU | 「專屬容量」 | Provisioned Throughput Unit：保留的 inference GPU，按小時計價 |
| Application Inference Profile | 「Bedrock 標籤」 | 每個產品的成本/用量 profile，附 tags，CloudWatch 原生支援 |
| Model Garden | 「Vertex 目錄」 | Vertex AI 的第三方模型區，與 Gemini 分開 |
| Two-provider minimum | 「LLM 備援」 | 讓每條關鍵 LLM 路徑跨 ≥2 家 hyperscaler 執行的政策 |
| BAA | 「HIPAA 文件」 | Business Associate Agreement；處理 PHI 時必須具備；三家都提供 |
| Abuse monitoring | 「log 監看員」 | 供應商端對 prompts/outputs 的安全掃描；企業可 opt-out |

## 延伸閱讀 (Further Reading)

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/)：權威的費率表與 Provisioned Throughput 定價。
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/azure-openai/)：PTU 經濟學與費率表。
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing)：Gemini 層級與 Model Garden 附加費。
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/)：跨供應商的持續 latency 與 throughput benchmark。
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/)：企業決策框架。
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend)：並排比較 attribution 機制。
