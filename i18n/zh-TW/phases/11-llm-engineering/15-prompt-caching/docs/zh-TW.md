# Prompt Caching 與 Context Caching

> 你的 system prompt 有 4,000 個 token。你的 RAG context 有 20,000 個 token。你每次請求都把兩者一起送出，也每次都要為兩者付費。Prompt caching 讓供應商在他們那邊保持這段 prefix 的「暖機」狀態，重複使用時只收你正常費率的 10%。用得正確，可以把推論成本降低 50-90%，並把首個 token 的延遲降低 40-85%。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 11 · 01 (Prompt Engineering), Phase 11 · 05 (Context Engineering), Phase 11 · 11 (Caching and Cost)
**Time:** ~60 分鐘

## 問題 (The Problem)

一個 coding agent 在對話的每一輪，都把同一個 15,000 token 的 system prompt 送給 Claude。以 $3/M input token 計算，20 輪光是 input 成本就要 $0.90 — 這還沒算使用者實際的訊息。乘上每天 10,000 場對話，這些永遠不會改變的文字，每天的帳單就高達 $9,000。

你無法在不損害品質的情況下縮減 prompt。你也無法避免送出它 — 模型每一輪都需要它。唯一的辦法，是不要再為供應商已經看過的 prefix 付全額的價錢。

這個辦法就是 prompt caching。Anthropic 在 2024 年 8 月推出（2025 年加入 1 小時延長 TTL 的變體），OpenAI 在同年稍後將它自動化，Google 在 Gemini 1.5 推出時一併提供明確的 context caching，現在三家都把它作為頂級模型的一等功能。

## 核心概念 (The Concept)

![Prompt caching: write once, read cheap](../assets/prompt-caching.svg)

**機制。**當請求的 prefix 與近期請求相符時，供應商會提供前一次執行留下的 KV-cache，而不是重新編碼這些 token。你在第一次付一小筆寫入加價，之後每次都享有大幅的讀取折扣。

**2026 年的三種供應商風格。**

| 供應商 | API 風格 | 命中折扣 | 寫入加價 | 預設 TTL | 最小可快取量 |
|---------|-----------|--------------|---------------|-------------|---------------|
| Anthropic | 在 content block 上加入明確的 `cache_control` 標記 | input 折 90% | 加價 25% | 5 分鐘（可延長至 1 小時） | 1,024 tokens（Sonnet/Opus）、2,048（Haiku） |
| OpenAI | 自動偵測 prefix | input 折 50% | 無 | 最長 1 小時（盡力而為） | 1,024 tokens |
| Google (Gemini) | 明確的 `CachedContent` API | 依儲存計費；讀取約為正常的 25% | 每 token·小時的儲存費 | 使用者設定（預設 1 小時） | 4,096 tokens（Flash）、32,768（Pro） |

**不變的原則。**三家都只快取 prefix。只要請求之間有任何一個 token 不同，從第一個不同的 token 之後的所有內容都是 miss。請把*穩定*的部分放在最上面，把*會變動*的部分放在最下面。

### 對快取友善的版面配置 (The cache-friendly layout)

```
[system prompt]          <-- cache this
[tool definitions]       <-- cache this
[few-shot examples]      <-- cache this
[retrieved documents]    <-- cache if reused, else don't
[conversation history]   <-- cache up to last turn
[current user message]   <-- never cache (different every time)
```

違反這個順序 — 把使用者訊息放在 system prompt 之上，或是在 few-shot 之間穿插動態檢索的內容 — 快取就永遠不會命中。

### 損益平衡計算 (The break-even calculation)

Anthropic 的 25% 寫入加價（5 分鐘 TTL）在第一次讀取時就回本，因為快取讀取的成本是 0.1 倍。1 次寫入加 1 次讀取，平均每個請求的成本是 0.675 倍（省 32%）；1 次寫入加 10 次讀取，平均是 0.205 倍（省 80%）。1 小時 TTL 的寫入要收 2 倍，所以那個區塊需要兩次讀取才能回本。經驗法則：快取任何你預期在 TTL 內至少會重複使用 3 次的東西；這在任何一種 TTL 上都能越過損益平衡點。

```figure
prompt-cache-hit
```

## 動手做 (Build It)

### 步驟 1：用明確標記做 Anthropic prompt caching

```python
import anthropic

client = anthropic.Anthropic()

SYSTEM = [
    {
        "type": "text",
        "text": "You are a senior Python reviewer. Follow the rubric exactly.\n\n" + RUBRIC_15K_TOKENS,
        "cache_control": {"type": "ephemeral"},
    }
]

def review(code: str):
    return client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=SYSTEM,
        messages=[{"role": "user", "content": code}],
    )
```

`cache_control` 標記告訴 Anthropic 把這個區塊儲存 5 分鐘。在這個時間窗口內重複使用會命中；過期後再使用則會重新寫入。

**回應的 usage 欄位：**

```python
response = review(code_a)
response.usage
# InputTokensUsage(
#     input_tokens=120,
#     cache_creation_input_tokens=15023,   # paid at 1.25x
#     cache_read_input_tokens=0,
#     output_tokens=340,
# )

response_b = review(code_b)
response_b.usage
# cache_creation_input_tokens=0
# cache_read_input_tokens=15023           # paid at 0.1x
```

請在 CI 中檢查這兩個欄位 — 如果 `cache_read_input_tokens` 在各個請求之間一直維持為零，代表你的 cache key 在漂移。

### 步驟 2：一小時的延長 TTL

對於長時間執行的批次工作，5 分鐘的預設值會在工作之間過期。設定 `ttl`：

```python
{"type": "text", "text": RUBRIC, "cache_control": {"type": "ephemeral", "ttl": "1h"}}
```

1 小時 TTL 的寫入加價是 2 倍（比基準高 50%，而不是 25%），但對任何重複使用該 prefix 超過 5 次的批次來說，很快就能回本。

### 步驟 3：OpenAI 自動快取

OpenAI 不需要你設定任何東西。任何超過 1,024 token、且與近期請求相符的 prefix，都會自動享有 50% 折扣。

```python
from openai import OpenAI
client = OpenAI()

resp = client.chat.completions.create(
    model="gpt-5",
    messages=[
        {"role": "system", "content": SYSTEM_PROMPT},   # long and stable
        {"role": "user", "content": user_msg},
    ],
)
resp.usage.prompt_tokens_details.cached_tokens  # the discounted portion
```

同樣適用對快取友善的版面配置規則。有兩件事會破壞 OpenAI 的快取，卻不會破壞 Anthropic 的：變更 `user` 欄位（它是 cache key 的組成之一），以及重新排列 tool 的順序。

### 步驟 4：Gemini 的明確 context caching

Gemini 把快取視為你建立並命名的一等物件：

```python
from google import genai
from google.genai import types

client = genai.Client()

cache = client.caches.create(
    model="gemini-3.8-flash",
    config=types.CreateCachedContentConfig(
        display_name="rubric-v3",
        system_instruction=RUBRIC,
        contents=[FEW_SHOT_EXAMPLES],
        ttl="3600s",
    ),
)

resp = client.models.generate_content(
    model="gemini-3.8-flash",
    contents=["Review this code:\n" + code],
    config=types.GenerateContentConfig(cached_content=cache.name),
)
```

只要快取還存在，Gemini 就按每 token·小時收取儲存費，讀取則約為正常 input 費率的 25%。當你要在數天內、跨許多 session 重複使用同一個龐大的 prompt 時，這是正確的形式。

### 步驟 5：在 production 中量測命中率

請見 `code/main.py`，那裡有一個模擬的三供應商帳務程式，追蹤寫入/讀取/未命中的次數，並計算每 1K 次請求的混合成本。請依目標命中率來把關部署 — 多數 production 的 Anthropic 設定在暖機後，讀取比例應該超過 80%。

## 2026 年仍然會上線的陷阱 (Pitfalls that still ship in 2026)

- **最上方有動態時間戳記。**system prompt 最上面寫著 `"Current time: 2026-04-22 15:30:02"`。每個請求都 miss。把時間戳記移到快取斷點之下。
- **Tool 重新排序。**請以穩定的順序序列化 tool — 部署之間字典被重新洗牌，會破壞每一次命中。
- **自由文字的近似重複。**「You are helpful.」對比「You are a helpful assistant.」— 差一個位元組就是完全 miss。
- **區塊太小。**Anthropic 強制規定 1,024 token 的下限（Haiku 為 2,048）。更小的區塊會無聲地不被快取。
- **看不見的成本儀表板。**把「input token」拆成有快取與未快取。否則流量下降看起來會像是快取的功勞。

## 實際使用 (Use It)

2026 年的快取技術堆疊：

| 情況 | 選擇 |
|-----------|------|
| 有穩定 10k 以上 system prompt、多輪對話的 agent | 搭配 5 分鐘 TTL 的 Anthropic `cache_control` |
| 重複使用同一個 prefix 達 30 分鐘以上的批次工作 | 搭配 `ttl: "1h"` 的 Anthropic |
| 在 GPT-5 上的 serverless endpoint，沒有自訂基礎設施 | OpenAI 自動快取（只要讓你的 prefix 保持穩定且夠長） |
| 跨數天重複使用龐大的程式碼/文件語料 | Gemini 明確的 `CachedContent` |
| 跨供應商的備援 | 讓可快取 prefix 的版面配置在各供應商之間保持一致，這樣任何一邊命中都有效 |

與 semantic caching（Phase 11 · 11）搭配使用於使用者訊息層：prompt caching 處理*token 完全相同*的重複使用，semantic caching 處理*語意相同*的重複使用。

## 交付 (Ship It)

儲存 `outputs/skill-prompt-caching-planner.md`：

```markdown
---
name: prompt-caching-planner
description: Design a cache-friendly prompt layout and pick the right provider caching mode.
version: 1.0.0
phase: 11
lesson: 15
tags: [llm-engineering, caching, cost]
---

Given a prompt (system + tools + few-shot + retrieval + history + user) and a usage profile (requests per hour, TTL needed, provider), output:

1. Layout. Reordered sections with a single cache breakpoint marked; explain which sections are stable, which are volatile.
2. Provider mode. Anthropic cache_control, OpenAI automatic, or Gemini CachedContent. Justify from TTL and reuse pattern.
3. Break-even. Expected reads per write within TTL; net cost vs no-cache with math.
4. Verification plan. CI assertion that cache_read_input_tokens > 0 on the second identical request; dashboard split by cached vs uncached tokens.
5. Failure modes. List the three most likely reasons the cache will miss in this setup (dynamic timestamp, tool reorder, near-duplicate text) and how you will prevent each.

Refuse to ship a cache plan that places a dynamic field above the breakpoint. Refuse to enable 1h TTL without a reuse count that makes the 2x write premium pay back.
```

## 練習 (Exercises)

1. **簡單。**拿一段 10 輪的對話，搭配 5,000 token 的 system prompt，對 Claude 執行。先不加 `cache_control` 執行一次，再加上之後執行一次。回報兩者各自的 input-token 帳單。
2. **中等。**撰寫測試框架：給定 prompt template 與請求 log，計算每個供應商（Anthropic 5m、Anthropic 1h、OpenAI 自動、Gemini 明確）預期的命中率與節省的金額。
3. **困難。**建立版面配置最佳化器：給定一個 prompt 與一份標記為 `stable=True/False` 的欄位清單，重寫 prompt，在不損失資訊的前提下，把單一快取斷點放在最大的對快取友善位置。在真實的 Anthropic endpoint 上驗證。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Prompt caching | 「讓長 prompt 變便宜」 | 對相符的 prefix 重複使用供應商端的 KV-cache；重複的 input token 享有 50-90% 折扣。 |
| `cache_control` | 「Anthropic 的標記」 | Content block 的屬性，宣告「到這裡為止的所有內容都可快取」；`{"type": "ephemeral"}`。 |
| Cache write | 「付加價」 | 填入快取的第一個請求；在 Anthropic 以約 1.25 倍的 input 費率計費，在 OpenAI 則免費。 |
| Cache read | 「折扣」 | 之後符合 prefix 的請求；計費為 10%（Anthropic）、50%（OpenAI）、約 25%（Gemini）。 |
| TTL | 「它能活多久」 | 快取保持暖機狀態的秒數；Anthropic 預設 5 分鐘（可延長至 1 小時）、OpenAI 盡力而為最長 1 小時、Gemini 由使用者設定。 |
| Extended TTL | 「1 小時的 Anthropic 快取」 | `{"type": "ephemeral", "ttl": "1h"}`；寫入加價 2 倍，但對批次重複使用很值得。 |
| Prefix match | 「為什麼我的快取 miss」 | 只有當從開頭到斷點的每個 token 都逐位元組相同時，快取才會命中。 |
| Context caching (Gemini) | 「明確版」 | Google 具名、依儲存計費的快取物件；最適合跨數天重複使用大型語料。 |

## 延伸閱讀 (Further Reading)

- [Anthropic — Prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — `cache_control`、1 小時 TTL、損益平衡表。
- [OpenAI — Prompt caching](https://platform.openai.com/docs/guides/prompt-caching) — 自動 prefix 比對。
- [Google — Context caching](https://ai.google.dev/gemini-api/docs/caching) — `CachedContent` API 與儲存定價。
- [Anthropic engineering — Prompt caching for long-context workloads](https://www.anthropic.com/news/prompt-caching) — 原始的發表文章，附有延遲數字。
- Phase 11 · 05 (Context Engineering) — 在哪裡切分 prompt，快取才能落地。
- Phase 11 · 11 (Caching and Cost) — 把 prompt caching 與針對使用者訊息的 semantic cache 搭配使用。
- [Pope et al., "Efficiently Scaling Transformer Inference" (2022)](https://arxiv.org/abs/2211.05102) — prompt caching 向使用者揭露的 KV-cache 記憶體模型；解釋為什麼重讀快取的 prefix 比重新計算便宜約 10 倍。
- [Agrawal et al., "SARATHI: Efficient LLM Inference by Piggybacking Decodes with Chunked Prefills" (2023)](https://arxiv.org/abs/2308.16369) — prefill 是 prompt caching 所省略的階段；這篇論文解釋為什麼快取命中時 TTFT 會大幅下降，而 TPOT 不受影響。
- [Leviathan et al., "Fast Inference from Transformers via Speculative Decoding" (2023)](https://arxiv.org/abs/2211.17192) — prompt caching 與 speculative decoding、Flash Attention 以及 MQA/GQA 並列，都是彎曲推論成本曲線的槓桿；其餘三者請讀這篇。
