# Prefix-Cache Serving：RadixAttention 與 KV 重用 (RadixAttention and KV Reuse)

> 把 KV cache 當成一等公民、可重用的資源，存放在 radix tree 裡，排程也會隨之改變：不再像 vLLM 那樣用 FCFS (first-come, first-served)，cache-aware scheduler 會優先處理共用前綴較長的 request，實質上是 depth-first 的 radix 走訪，讓熱門分支留在 HBM 中。SGLang 就是圍繞這個概念建立 serving 的 engine。在 Llama 3.1 8B 加上類似 ShareGPT 的 1K prompts 上，SGLang 達到 ~16,200 tok/s，vLLM 為 ~12,500，領先約 ~29%。在大量使用前綴的 RAG 工作負載上，優勢可達 6.4x。在語音克隆類型的工作負載上，cache hit rate 超過 86%。2026 年已部署在 400,000+ 張 GPU 上，客戶包括 xAI、LinkedIn、Cursor、Oracle、GCP、Azure、AWS。陷阱是：當前綴順序不一致時，6.4x 這個數字就會消失，順序是工程師可以掌握的槓桿。

**Type:** Learn
**Languages:** Python (stdlib, toy radix-tree cache + cache-aware scheduler)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 14 (Agentic RAG)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 畫出 RadixAttention：前綴如何存放在 radix tree 中，以及源自同一分支的 sequence 如何共用 KV blocks。
- 解釋 cache-aware scheduling，以及為什麼 FCFS 不適合大量使用前綴的流量。
- 給定 prefix-cache hit rate 與 prompt 長度分布，計算工作負載的預期加速。
- 說出能讓 6.4x 這個數字成真（而不是白白損失）的 prompt 排序紀律。

## 問題 (The Problem)

傳統 serving 把每個 request 的 prompt 當成不透明的東西。即使 5,000 個 RAG request 都以同一個 2,000-token 的 system prompt 加上同樣的檢索開頭起頭，vLLM 還是會把那 2,000-token 的前綴 prefill 5,000 次。GPU 一遍又一遍地做同樣的工作。

觀察：agentic 與 RAG 工作負載中的 prompt 幾乎總是共用很長的前綴。System prompt、tool schemas、few-shot 範例、檢索標頭、對話歷史，全都在 request 之間重複。如果你把那段前綴的 KV cache 存一次並重用，就不需要再 prefill 它。

RadixAttention 正是這麼做。Tokens 被索引在 radix tree 中；每個節點擁有從根到它那條路徑上 token 序列的 KV blocks。新 request 走訪這棵樹：任何 token 相符的節點，就重用該節點的 KV blocks。Prefill 成本變成與「新的」後綴成正比，而不是整個 prompt。

挑戰在於排程。如果兩個 request 共用 2,000-token 的前綴，而第三個只與該前綴共用 200 tokens，你會希望把兩個長共用的 request 一起服務，讓長前綴留在 HBM 中。FCFS 剛好相反：它服務最先到的人，可能在下一個長前綴 request 命中之前，就把熱門分支驅逐掉。

## 核心概念 (The Concept)

### 把 radix tree 當作 KV 索引

Radix tree（compact trie）儲存 token 序列。每個節點擁有一段 token 範圍，以及為該範圍計算出的 KV blocks。子節點則延伸序列一個或多個 tokens。

```
root
 |- "You are a helpful assistant..."  (2,000 tokens, 124 KV blocks)
      |- "Context: <doc A>..."        (500 tokens, 31 blocks)
           |- "Question: Alice..."    (80 tokens, 5 blocks)
           |- "Question: Bob..."      (95 tokens, 6 blocks)
      |- "Context: <doc B>..."        (520 tokens, 33 blocks)
```

新的 request 帶著 system prompt + "Context: <doc A>" + "Question: Carol" 進來。Scheduler 走訪：system 前綴相符（重用 124 個 blocks），doc-A 分支相符（重用 31 個 blocks），然後只為 "Question: Carol" 配置新的 blocks（4 個 blocks）。Prefill 成本：4 個新 token 的 blocks。沒有這棵樹則是 160 個 blocks。Prefill 約省下 ~40x。

### Cache-aware scheduling

如果 cache 一直在抖動，radix-tree 支撐的重用就沒有意義。兩個關鍵政策：

1. **Depth-first dispatch**。從佇列挑下一個 request 時，優先選擇與目前執行集合同一分支的 request。這讓熱門分支保持釘住。
2. **以分支為單位的 LRU，而不是以 block 為單位**。驅逐整個分支（從最少使用的葉節點開始），而不是個別 blocks，讓 cache 的形狀與 radix 的形狀一致。

FCFS 兩者都違反。共用 2,000 tokens 的 request 排在只共用 50 tokens 的 request 後面，然後 2,000-token 的分支為了接納那個 50-token 的 request 而被驅逐。

### 該背起來的 benchmark 數字

- Llama 3.1 8B、H100、ShareGPT 1K prompts：SGLang ~16,200 tok/s vs vLLM ~12,500（領先約 ~29%）。
- 大量前綴的 RAG（相同 system + 相同文件，問題不同）：SGLang 最高 6.4x。
- 語音克隆工作負載：86.4% prefix-cache hit rate。
- SGLang 客戶的 production hit rate：50-99%，視 prompt 紀律而定。
- 2026 年已部署在 400,000+ 張 GPU 上。

### 排序的陷阱

6.4x 這個數字仰賴一致的 prompt-template 順序。如果你的 client 在某些 request 中把 prompt 組成 `[system, tools, context, history, question]`，在其他 request 中組成 `[system, context, tools, history, question]`，那棵樹就找不到共用前綴。人看起來是共用前綴的東西，對 radix tree 來說是兩個不同的 sequence。

工程師的槓桿：你的 prompt template 就是 cache key。固定順序。把所有不可變的內容（system、tools、schemas）放最前面。接著放檢索 context。使用者問題放最後。不要把動態內容穿插進前綴。

研究中的真實案例：把動態內容移出可快取的前綴後，一個部署一次改動就把 cache hit rate 從 7% 提升到 74%。

### RadixAttention 在哪裡贏、在哪裡輸

贏：
- RAG（相同檢索開頭，問題不同）。
- Agents（相同 tool schemas，查詢不同）。
- 有長 system prompt 的聊天。
- 有重複開頭的語音/視覺工作負載。

輸（回到 vLLM 等級的 throughput）：
- Prompt 各不相同的單次生成（程式碼補全、沒有 system prompt 的開放式聊天）。
- 每個 request 都把獨特內容穿插進前綴的動態 prompt。

### 為什麼這是 scheduler 問題，而不只是 kernel 問題

你可以把 KV 重用實作成 kernel 技巧。SGLang 的洞見是：只有當 scheduler 讓熱門分支持續留在記憶體裡，重用才划算。單純的「有就重用」政策在混合負載下會讓 cache 抖動。以 radix tree 為索引的 scheduler，才是把 kernel 技巧變成 29% production 優勢的關鍵。

### 與 vLLM 的互動

這兩個系統並非嚴格的競爭者。2026 年 vLLM 加入了 prefix caching (`--enable-prefix-caching`) 與 cache-aware router（以 Rust 寫的 vLLM Router）。差距縮小了，但沒有完全消失：SGLang 整個 stack 以 radix 為先，vLLM 是後來嫁接上去的。對以前綴重用為主的工作負載，SGLang 仍是預設選擇。對沒有明顯前綴模式的通用 serving，vLLM 仍然持平或更好。

```figure
roofline
```

## 實際使用 (Use It)

`code/main.py` 實作一個玩具 radix-tree KV cache，加上兩種政策的 scheduler：FCFS 與 cache-aware。讓相同的工作負載跑過兩者，回報 prefix-cache hit rate 與 throughput 差異。接著執行「順序被打亂」的工作負載，展示 6.4x 的崩潰。

[Inside an LLM Serving GPU](https://aiengineeringfromscratch.com/inside-an-llm-serving-gpu?workload=shared&shared=4096) 這個機器展示了真實 L4 執行下的 radix cache。四個群組各有自己的 4,096-token system prompt。群組的第一個 request 會計算它的 prompt。只要 prompt 還在 cache 中，該群組後續的 request 只需計算 128-token 的問題。把 radix cache 關掉，就能看到 median first token 從 479 ms 變成 14,706 ms。

## 交付 (Ship It)

這堂課產出 `outputs/skill-radix-scheduler-advisor.md`。給定工作負載描述（prompt-template 形狀、檢索模式、並行 tenant 數量），它會產出 prompt 排序建議，以及是否採用 SGLang 的 go/no-go 判斷。

## 練習 (Exercises)

1. 執行 `code/main.py`。在相同工作負載上比較 FCFS 與 cache-aware。差異來自哪裡：prefill 節省、decode 節省，還是佇列延遲？
2. 修改工作負載，讓 prompt 隨機排列 `[system, tools, context]`。重新執行。hit rate 會怎樣？為什麼？
3. 計算在 Llama 3.1 8B 上，把 2,000-token system prompt 作為一個 radix 分支常駐的 HBM 成本。與沒有前綴重用的 16-sequence batch 的成本比較。
4. 閱讀 SGLang RadixAttention 論文。用三句話解釋為什麼在大量前綴的負載下，樹狀 LRU 驅逐勝過以 block 為單位的 LRU。
5. 一位客戶回報 cache hit rate 只有 8%。說出三個可能原因，以及你會對每一個執行的診斷。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| RadixAttention | 「SGLang 的那個東西」 | 以 radix tree 索引的 KV cache，讓共用前綴重用 blocks |
| Radix tree | 「compact trie」 | 每個節點擁有一段 token 範圍及其 KV blocks 的樹 |
| Cache-aware scheduler | 「熱門分支優先」 | 偏好與常駐分支共用的 request 的 scheduler |
| Prefix-cache hit rate | 「你的 prompt 有多少是免費的」 | 由重用的 KV blocks 提供的 prompt tokens 比例 |
| FCFS | 「先到先服務」 | 會破壞前綴局部性的預設排程 |
| Branch-level LRU | 「驅逐葉節點」 | 與 radix 形狀相符的驅逐政策 |
| Prompt template ordering | 「cache key」 | prompt 各元件的順序決定了樹能共用什麼 |
| System prompt pinning | 「常駐前綴」 | 把不可變的 system 部分釘住，避免驅逐抖動 |

## 延伸閱讀 (Further Reading)

- [SGLang GitHub](https://github.com/sgl-project/sglang)：原始碼與文件。
- [SGLang documentation](https://sgl-project.github.io/)：RadixAttention 與排程細節。
- [SGLang paper — Efficiently Programming Large Language Models (arXiv:2312.07104)](https://arxiv.org/abs/2312.07104)：設計參考。
- [LMSYS blog — SGLang with RadixAttention](https://www.lmsys.org/blog/2024-01-17-sglang/)：benchmark 數字與 scheduler 的設計理由。
- [vLLM — Prefix Caching](https://docs.vllm.ai/en/latest/features/automatic_prefix_caching/)：vLLM 自己類似 radix 的實作，供比較。
