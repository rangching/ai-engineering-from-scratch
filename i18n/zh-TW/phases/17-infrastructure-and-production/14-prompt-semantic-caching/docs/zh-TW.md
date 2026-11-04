# Prompt Caching 與 Semantic Caching 的經濟學 (Prompt Caching and Semantic Caching Economics)

> **定價快照日期為 2026-04。** 以下數字反映本課程發布時擷取的供應商費率表；在向下游引用之前，請對照連結的文件驗證。

> Caching 發生在兩個層次。L2（供應商層級）的 prompt/prefix caching 會為重複的前綴重用 attention KV：Anthropic 的 prompt-caching 文件宣稱，長 prompt 最多可降低 90% 成本與 85% latency；以 Claude 3.5 Sonnet 為例，cache 讀取為 $0.30/M，全新輸入為 $3.00/M，TTL 為 5 分鐘，若選 1 小時 TTL 則寫入有 2x 溢價 (docs.anthropic.com, 2026-04)。OpenAI 的 prompt caching 對 ≥1024 tokens 的 prompt 自動套用，快取輸入的價格大約比全新輸入便宜 90% (platform.openai.com, 2026-04)；各模型實際的快取費率取決於即時費率表。L1（應用層級）的 semantic caching 在 embedding 相似度命中時完全跳過 LLM。供應商宣稱的「95% 準確度」指的是比對的正確性，不是命中率：回報的 production 命中率從 10%（開放式聊天）到 70%（結構化 FAQ）不等，固定 prompt 集合的語音 agent 回報為 50-80%；兩家供應商都沒有公布官方基準，所以請把這些視為社群的 telemetry，而非保證。Production 的陷阱：平行化會殺死 caching（在第一次 cache 寫入之前就發出 N 個平行 request，可能讓花費膨脹數倍），而前綴內的動態內容會讓 cache 完全無法命中。ProjectDiscovery 回報，把動態文字移出可快取的前綴後，命中率從 7% 提升到 74% (2025-11)。

**Type:** Learn
**Languages:** Python (stdlib, toy two-layer cache simulator)
**Prerequisites:** Phase 17 · 04 (Serving Engine Internals), Phase 17 · 06 (SGLang RadixAttention)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 區分 L2 prompt/prefix caching（供應商端的 KV 重用）與 L1 semantic caching（相似 prompt 時繞過 LLM）。
- 解釋 Anthropic 的 `cache_control` 明確標記，以及兩種 TTL 選項（5 分鐘 vs 1 小時）和它們的價格倍數。
- 給定命中率、prompt/回應的組成與 token 價格，計算預期每月節省。
- 說出會讓帳單膨脹 5-10x 的平行化反模式，以及會讓命中率崩潰的動態內容反模式。

## 問題 (The Problem)

你替 RAG 服務加上 prompt caching。帳單卻沒變。你量測命中率：7%。你的 prompt 看起來是靜態的，其實不是：system prompt 包含精確到分鐘的目前日期、一個 request ID，以及為了多樣性而隨機重新排序的範例。每個 request 都寫入一筆新的 cache 項目，卻讀取不到任何東西。

另外，你的 agent 每個使用者問題會平行執行十個 tool 呼叫。這十個在第一次 cache 寫入完成之前就全部抵達供應商。十次寫入，零次讀取。你的帳單是「有 caching」本應花費的 5-10 倍。

Caching 是一個協定，不是一個旗標。兩個層次，兩種不同的失敗模式。

## 核心概念 (The Concept)

### L2：供應商 prompt/prefix caching

供應商為可快取的前綴儲存 attention KV，並在下一個符合該前綴的 request 重用。你只付一次寫入成本，讀取幾乎免費。

**Anthropic (Claude 3.5 / 3.7 / 4 系列)**：request 中有明確的 `cache_control` 標記。你標記哪些 block 可快取。TTL：5 分鐘（寫入成本為 base 的 1.25x）或 1 小時（寫入成本為 base 的 2x）。Cache 讀取：Claude 3.5 Sonnet 為 $0.30/M，全新為 $3.00/M，便宜 10x (docs.anthropic.com，截至 2026-04)。各模型費率不同（Opus/Haiku 另外公布）；務必對照即時定價頁面。

**OpenAI**：對 ≥1024 tokens 的 prompt 自動 caching (platform.openai.com, 2026-04)。沒有明確旗標。在目前 gpt-4o/gpt-5 的費率表上，快取輸入大約比全新便宜 10x。文件與 release notes 都沒有公布官方的命中率基準；社群回報在精心設計 prompt 時約落在 30-60%。請監看 `usage.cached_tokens` 來量測你自己的數字。

**Google (Gemini)**：透過明確 API 做 context caching；1M-token 的 context 讓 caching 更划算。

**自架 (vLLM, SGLang)**：Phase 17 · 06 介紹 RadixAttention，相同模式，只是用在你自己的運算上。

### L1：應用層級的 semantic caching

在呼叫 LLM 之前，先對 prompt 做 hash、做 embedding，並尋找相似的已快取 request（cosine 相似度高於門檻，通常 0.95+）。命中時，回傳快取的回應。未命中時，呼叫 LLM 並快取結果。

開源：Redis Vector Similarity、GPTCache、Qdrant。商業：Portkey Cache、Helicone Cache。

供應商的準確度宣稱，指的是回傳的快取回應在語意上恰當的頻率，不是你命中的頻率。Production 命中率：

- 開放式聊天：10-15%。
- 結構化 FAQ / 客服：40-70%。
- 程式碼問題：20-30%（小變體就會讓命中失敗）。
- 重複 prompt 的語音 agent：50-80%（語音正規化後的固定集合）。

### 平行化反模式

你的 agent 平行發出 10 個 tool 呼叫。這 10 個都有相同的 4K-token system prompt。Anthropic 的 cache 寫入是以 request 為單位；第一次 cache 寫入大約在供應商看到 prompt 後 300 ms 完成。第 2-10 個 request 在同一毫秒的視窗內抵達，各自看到 cache miss。你付了 10 次寫入溢價，0 次讀取折扣。

修法：以「先循序」的方式批次處理：先單獨發出 request 1，等它的 cache 填好後再發出 2-10。第一個 tool 呼叫多 300 ms；但避免帳單膨脹 5-10x。

### 動態內容反模式

你的 system prompt 長這樣：

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

每個 request 都是獨一無二的。每個 request 都寫入。零命中。

修法：把真正靜態的內容全部移到可快取的前綴；把動態內容附加在 cache 邊界之後：

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery 就是這樣把 cache 命中率從 7% 提升到 74%，並公布了詳細剖析。

### 為夜間工作負載疊加 batch + cache

Batch API (Phase 17 · 15) 在 24 小時內交付的前提下提供 50% 折扣。再疊加快取輸入，又能再多約 ~10x。夜間的分類、標註與報告生成工作負載，疊加之後可降到同步未快取成本的約 ~10%。

### 該記住的數字

定價是 2026-04 從連結的供應商文件擷取，每隔幾個月就會變動，使用前請重新確認。

- Anthropic 快取讀取：Claude 3.5 Sonnet 為 $0.30/M，比全新輸入便宜約 10x (docs.anthropic.com)。
- Anthropic cache 寫入溢價：1.25x（5 分鐘 TTL）或 2x（1 小時 TTL）。
- OpenAI 自動 cache：適用於 ≥1024 tokens 的 prompt；在目前費率表上，快取輸入的價格約為全新輸入的 10% (platform.openai.com)。
- Semantic cache 命中率（社群回報）：開放聊天約 ~10%；結構化 FAQ 最高約 ~70%；固定 prompt 集合的語音 agent 為 50-80%。不是供應商文件載明的基準。
- ProjectDiscovery：把動態內容移出前綴，命中率 7% → 74%（專案部落格，2025-11）。
- 平行化反模式：當 N 個平行 request 都錯過第一次 cache 寫入時，常見的回報是帳單膨脹 5–10x。

```figure
semantic-cache-hit
```

## 實際使用 (Use It)

`code/main.py` 在混合工作負載上模擬 L1 + L2 caching。回報命中率、帳單，並展示平行化的損失。

## 交付 (Ship It)

這堂課產出 `outputs/skill-cache-auditor.md`。給定 prompt template 與流量，稽核可快取性並建議重構方式。

## 練習 (Exercises)

1. 執行 `code/main.py`。切換平行化旗標。帳單變化多少？
2. 你的 system prompt 含有日期。把它移出去。展示前後的命中率計算。
3. 依你的 request 抵達速率，計算 1 小時 TTL（2x 寫入）與 5 分鐘 TTL（1.25x 寫入）的損益平衡。
4. Semantic cache 在 0.95 門檻下命中 20%。在 0.85 下命中 50%，但你看到不正確的快取回應。挑選正確的門檻並說明理由。
5. 你的每個使用者問題批次處理 10 個平行子查詢。在不增加端到端 latency 的前提下，改寫成對 cache 友善的形式。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| L2 prompt cache | 「prefix cache」 | 供應商為重複的前綴儲存 KV |
| `cache_control` | 「Anthropic cache 標記」 | 標記可快取 block 的明確屬性 |
| Cache write premium | 「寫入稅」 | 第一次 miss 轉為 cache 的額外成本 (1.25x 或 2x) |
| L1 semantic cache | 「embedding cache」 | 呼叫 LLM 之前，在應用層做 hash 與 embedding |
| GPTCache | 「LLM caching 函式庫」 | 熱門的開源 L1 cache 函式庫 |
| Cache hit rate | 「命中 / 總數」 | 由 cache 提供服務的 request 比例 |
| Parallelization anti-pattern | 「N 次寫入陷阱」 | N 個平行 request 讓 cache miss N 次 |
| Dynamic content trap | 「prompt 裡放時間的陷阱」 | 前綴中的動態位元組會殺死命中率 |
| RadixAttention | 「replica 內 cache」 | SGLang 的 prefix-cache 實作 |

## 延伸閱讀 (Further Reading)

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)：官方 `cache_control` 語意與 TTL。
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching)：自動 caching 的行為與資格。
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
