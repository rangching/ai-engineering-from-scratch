# 平行 Tool Call 與搭配 Tool 的串流 (Parallel Tool Calls and Streaming with Tools)

> 三個獨立的天氣查詢如果串行執行，就是三次往返。改成平行執行，總時間會縮到只剩最慢的那一個 call。現在每家前沿 provider 都能在單一 turn 輸出多個 tool call。好處是實在的，但接線的細節很微妙。這堂課會走過兩個部分：平行 fan-out，以及串流參數的重組，重點放在 id 對應的陷阱。

**Type:** Build
**Languages:** Python (stdlib, thread pool + streaming harness)
**Prerequisites:** Phase 13 · 02 (function calling deep dive)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說明 `parallel_tool_calls: true` 為什麼存在，以及什麼時候該關掉。
- 在平行 fan-out 期間，把串流的參數 chunk 對應到正確的 tool-call id。
- 把零碎的 `arguments` 字串重組成完整的 JSON，而且不提早解析。
- 執行三個城市的天氣 benchmark，示範循序與平行的延遲差異。

## 問題 (The Problem)

沒有平行 call 的話，agent 回答「班加羅爾、東京和蘇黎世的天氣如何」時會這樣做：

```
user -> LLM
LLM -> call get_weather(Bengaluru)
host -> run executor, reply with result
LLM -> call get_weather(Tokyo)
host -> run executor, reply with result
LLM -> call get_weather(Zurich)
host -> run executor, reply with result
LLM -> final text answer
```

三次 LLM 往返，而且每次還要再加上 executor 的延遲。大約是理想 wall-clock 時間的 4 倍。

有了平行 call：

```
user -> LLM
LLM -> call get_weather(Bengaluru); call get_weather(Tokyo); call get_weather(Zurich)
host -> run all three executors concurrently, reply with three results
LLM -> final text answer
```

只有一次 LLM 往返。Executor 的時間是三者中的最大值，而不是總和。在 OpenAI、Anthropic 和 Gemini 上的 production benchmark 顯示，fan-out 的工作負載可以減少 60% 到 70% 的 wall-clock 時間。

代價是對應上的複雜度。當三個 call 亂序完成時，你的結果必須帶著對應的 `tool_call_id`，模型才能把它們對上。當結果以串流方式到達時，你必須先把零碎的參數片段組裝成完整的 JSON，才能執行。Gemini 3 加入唯一 id，部分原因就是為了解決一個真實世界的問題：對同一個 tool 的兩個平行 call 無法區分。

## 核心概念 (The Concept)

### 啟用平行 (Enabling parallel)

- **OpenAI。** `parallel_tool_calls: true` 預設開啟。設成 `false` 可強制串行。
- **Anthropic。** 透過 `disable_parallel_tool_use: false` 啟用平行（Claude 3.5 以上預設如此）。設成 `true` 則為串行。
- **Gemini。** 一律具備平行能力；`tool_config.function_calling_config.mode = "AUTO"` 讓模型自行決定。

當 tool 之間有順序相依（先 `create_file` 再 `write_file`）、某個 call 的輸出會影響另一個 call 的輸入，或是 rate limiter 無法承受 fan-out 時，就要關閉平行。

### Id 對應 (Id correlation)

模型輸出的每個 call 都有一個 `id`。Host 回傳的每個結果都必須帶著同一個 id。沒有它，結果就會有歧義。

- **OpenAI。** 每則 tool-role 訊息上的 `tool_call_id`。
- **Anthropic。** 每個 `tool_result` block 上的 `tool_use_id`。
- **Gemini。** 每個 `functionResponse` 上的 `id`（Gemini 3 以上；Gemini 2 是用名稱對應，遇到同名的平行 call 就會壞掉）。

### 並行執行 call (Running calls concurrently)

Host 會讓每個 call 的 executor 在自己的 thread、coroutine 或遠端 worker 上執行。最簡單的 harness 用 thread pool；production 則用 asyncio 搭配 `asyncio.gather` 或 structured concurrency。完成的順序無法預測，id 才是識別依據。

一個常見的 bug：依照 call 清單的順序回覆結果，而不是依完成順序。這通常沒問題，因為模型只在意 `tool_call_id`，但如果某個結果被丟掉或重複了，與結果實際到達順序不同的回覆清單，會讓你更難在 log 裡找出缺口。建議依完成順序回覆，並附上明確的 id。

### 串流 tool call (Streaming tool calls)

當模型以串流輸出時，`arguments` 是一段一段到達的。三個平行 call 的三條 chunk 串流，會在 wire 上交錯。你需要為每個 id 準備一個 accumulator。

各 provider 的形狀：

- **OpenAI。** 每個 chunk 是 `choices[0].delta.tool_calls[i].function.arguments`（部分字串）。chunk 帶有 `index`（在 call 清單中的位置）。你依 index 累積，在 `id` 第一次出現時讀取它，並在 `finish_reason = "tool_calls"` 時解析 JSON。
- **Anthropic。** 串流事件先是 `message_start`，接著每個 block 有一個 `content_block_start`，型別為 `tool_use`（包含 id、name、空的 input）。`content_block_delta` 事件攜帶 `input_json_delta` chunk。`content_block_stop` 關閉每個 block。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 以上）輸出帶有 `functionCallId` 的 chunk，讓 call 能乾淨地交錯。在 Gemini 3 之前，串流一次只回傳一個完整的 call。

### 部分 JSON 與提早解析的陷阱 (Partial JSON and the parse-early trap)

在 `arguments` 完整之前，你無法解析它。像 `{"city": "Beng` 這樣的部分 JSON 是無效的，會丟出例外。正確的閘門是 provider 的 call 結束訊號：OpenAI 的 `finish_reason = "tool_calls"`、Anthropic 的 `content_block_stop`，或 Gemini 的 stream-end 事件。只有到那時才嘗試 `json.loads`。更穩健的做法是使用 incremental JSON parser，在結構完成時產生事件；OpenAI 的串流指南建議在需要顯示即時「thinking」指示器的 UX 中這麼做。用大括號計數來判斷完整性並不可靠（引號字串或跳脫內容裡的大括號會造成誤判），只應當作非正式的除錯啟發法。

### 亂序完成 (Out-of-order completion)

```
call_A: fast API, returns first
call_B: slow API, returns second
call_C: median API, returns third
```

Host 的回覆仍然必須引用這些 id：

```
[{role: "tool", tool_call_id: "call_A", content: ...},
 {role: "tool", tool_call_id: "call_B", content: ...},
 {role: "tool", tool_call_id: "call_C", content: ...}]
```

在 OpenAI 或 Anthropic 上，回覆的順序不影響正確性。Gemini 只要 id 對得上，任何順序都接受。

### Benchmark：循序 vs 平行

`code/main.py` 裡的 harness 模擬三個延遲分別為 400、600 和 800 ms 的 executor。循序執行總共需要 1800 ms。平行執行則是 max(400, 600, 800) = 800 ms。差異是固定的，而不是成比例的，所以 tool 越多，省下的時間越多。

真實世界的注意事項：平行 call 會對下游 API 造成壓力。對有 rate limit 的服務做 10 路 fan-out 會失敗。Phase 13 · 17 涵蓋 gateway 層級的 backpressure；retry 語意則規劃在未來的 phase。

### 串流 fan-out 的 wall-clock

如果模型本身是串流輸出，你可以在某個 call 的參數一完成就開始執行，而不用等所有 call 都結束。這是 OpenAI 有文件說明、但不是所有 SDK 都公開的最佳化。這堂課的 harness 就是這樣做：模擬的串流一產生完整的參數物件，host 立刻啟動該 call。

```figure
tp-parallel-fanout
```

## 實際使用 (Use It)

`code/main.py` 分成兩半。前半段用 `concurrent.futures.ThreadPoolExecutor`，以循序和平行兩種方式執行三個模擬的天氣 call，並印出 wall-clock 時間。後半段重播一段假的串流 response：三個平行 call 的 `arguments` chunk 交錯在同一條串流上，再用 `StreamAccumulator` 依 id 重組。沒有 LLM、沒有網路，只有重組邏輯。

觀察重點：

- 循序計時器跑到 1.8 秒。同樣的假延遲下，平行計時器只有 0.8 秒。
- Accumulator 處理亂序到達的 chunk 的方式，是依 id 做緩衝，只在每個 call 的 JSON 完整時才解析。
- Executor 在某個 id 的參數一完成就啟動，而不是等所有串流結束。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-parallel-call-safety-check.md`。給定一份 tool registry，這個 skill 會稽核哪些 tool 可以安全地平行化、哪些有順序相依、哪些會壓垮下游的 rate limit，並回傳修訂後、每個 tool 都帶有 `parallel_safe` 旗標的 registry。

## 練習 (Exercises)

1. 執行 `code/main.py` 並調整模擬的延遲。確認平行與循序的比率大約是 `max/sum`（實際執行會因為 thread 排程、序列化和 harness 開銷，而略微偏離理想值）。延遲分布變成什麼樣子時，平行就不再重要了？

2. 擴充 accumulator，處理「call 在串流中途被取消」的情況：丟掉它的 buffer，並輸出 `cancelled` 事件。哪家 provider 有明確記載這種情況？請查看 Anthropic 的 `content_block_stop` 語意，以及 OpenAI 的 `finish_reason: "length"` 行為。

3. 把 thread pool 換成 `asyncio.gather`，並對兩者做 benchmark。你應該會看到 async 因為 context-switch 成本較低而有小幅優勢，但前提是 executor 真的在做 I/O。

4. 挑出兩個不該平行化的 tool（例如 `create_file` 再 `write_file`）。在 registry 加入 `ordering_dependency` 圖，並以這張圖作為平行 fan-out 的閘門。這是相依感知排程的最小機制，未來的 agent-engineering phase 會把它正式化。

5. 閱讀 OpenAI 的 parallel-function-calling 章節和 Anthropic 的 `disable_parallel_tool_use` 文件。找出 Anthropic 建議關閉平行的那一種真實世界 tool 類型。（提示：對同一個資源的 consequential 變更。）

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Parallel tool calls | 「一個 turn 內 fan-out」 | 模型在單一 assistant 訊息中輸出多個 tool call |
| `parallel_tool_calls` | 「OpenAI 的旗標」 | 啟用或停用多重 call 輸出 |
| `disable_parallel_tool_use` | 「Anthropic 的反向旗標」 | 手動關閉的旗標；預設為啟用平行 |
| Tool call id | 「對應用的把手」 | 每個 call 的識別碼，result 訊息必須原樣回傳 |
| Accumulator | 「串流 buffer」 | 為部分 `arguments` chunk 準備、依 id 區分的字串 buffer |
| Out-of-order completion | 「最快的先到」 | 平行 call 以無法預測的順序完成，id 就是黏著劑 |
| Dependency graph | 「順序限制」 | 輸出會成為其他 tool 輸入的 tool，不能平行化 |
| Parse-early trap | 「JSON.parse 炸掉了」 | 嘗試解析不完整的 `arguments` 字串 |
| `streamFunctionCallArguments` | 「Gemini 3 的功能」 | 每個 call 帶有唯一 id 的串流參數 chunk |
| Completion-order reply | 「不要等全部」 | 結果一到就回覆，並以 id 為鍵 |

## 延伸閱讀 (Further Reading)

- [OpenAI — Parallel function calling](https://platform.openai.com/docs/guides/function-calling#parallel-function-calling) — 預設行為與關閉旗標
- [Anthropic — Parallel tool use](https://platform.claude.com/docs/en/agents-and-tools/tool-use/parallel-tool-use) — `disable_parallel_tool_use` 與結果批次處理
- [Google — Gemini function calling parallel section](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 3 起以 id 對應的平行 call
- [OpenAI — Streaming responses with tools](https://platform.openai.com/docs/api-reference/responses-streaming) — OpenAI 串流的 chunked 參數重組
- [Anthropic — Streaming messages](https://docs.anthropic.com/en/api/messages-streaming) — 帶有 `input_json_delta` 的 `content_block_delta`
