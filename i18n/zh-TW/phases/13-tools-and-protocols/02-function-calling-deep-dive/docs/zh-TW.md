# Function Calling 深入解析：OpenAI、Anthropic、Gemini (Function Calling Deep Dive — OpenAI, Anthropic, Gemini)

> 三家前沿 provider 在 2024 年收斂到同一個 tool-call 迴圈，之後其他地方卻各走各的。OpenAI 用 `tools` 和 `tool_calls`，Anthropic 用 `tool_use` 和 `tool_result` blocks，Gemini 用 `functionDeclarations` 和唯一 id 對應。這堂課把三者並排比較，讓你在某個 provider 上線的程式碼，移植時不會壞掉。

**Type:** Build
**Languages:** Python (stdlib, schema translators)
**Prerequisites:** Phase 13 · 01 (the tool interface)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 OpenAI、Anthropic、Gemini function-calling payload 在形狀上的三個差異（declaration、call、result）。
- 把一個 tool declaration 翻譯成三家 provider 的格式，並預測 strict-mode 限制會在哪裡不同。
- 在各個 provider 中使用 `tool_choice` 來強制、禁止或自動選擇 tool call。
- 了解各 provider 的硬性限制（tool 數量、schema 深度、參數長度），以及違反限制時各自會丟出的錯誤特徵。

## 問題 (The Problem)

Function-calling request 的形狀，每個 provider 都不一樣。以下是 2026 年 production 堆疊中的三個具體例子：

**OpenAI Chat Completions / Responses API。** 你傳入 `tools: [{type: "function", function: {name, description, parameters, strict}}]`。模型的回應包含 `choices[0].message.tool_calls: [{id, type: "function", function: {name, arguments}}]`，其中 `arguments` 是你必須自己解析的 JSON 字串。Strict mode（`strict: true`）透過 constrained decoding 強制 schema 合規。

**Anthropic Messages API。** 你傳入 `tools: [{name, description, input_schema}]`。回應以 `content: [{type: "text"}, {type: "tool_use", id, name, input}]` 的形式回來。`input` 已經解析好了（是物件，不是字串）。你要用一則新的 `user` 訊息回覆，裡面包含 `{type: "tool_result", tool_use_id, content}` block。

**Google Gemini API。** 你傳入 `tools: [{functionDeclarations: [{name, description, parameters}]}]`（巢狀在 `functionDeclarations` 之下）。回應以 `candidates[0].content.parts: [{functionCall: {name, args, id}}]` 的形式送達，其中 `id` 從 Gemini 3 起是唯一的，用於平行 call 的對應。你用 `{functionResponse: {name, id, response}}` 回覆。

同樣的迴圈，卻有不同的欄位名稱、不同的巢狀方式、不同的字串或物件慣例，以及不同的對應機制。一個團隊在 OpenAI 上寫好天氣 agent，光是接線就要花兩天移植到 Anthropic，再花一天移植到 Gemini。

這堂課會建立一個 translator，把三種格式統一成一份標準的 tool declaration，並在邊緣做路由。Phase 13 · 17 會把同一個模式一般化成 LLM gateway。

## 核心概念 (The Concept)

### 共通結構 (The common structure)

每個 provider 都需要五樣東西：

1. **Tool 清單。** 每個 tool 的 name、description 與 input schema。
2. **Tool choice。** 強制指定某個 tool、禁止 tool，或讓模型自行決定。
3. **Call 輸出。** 指名 tool 與參數的結構化輸出。
4. **Call id。** 把回應對應到正確的 call（平行時特別重要）。
5. **Result 注入。** 把結果接回 call 的訊息或 block。

### 形狀差異，逐欄比較 (Shape diffs, field by field)

| 面向 | OpenAI | Anthropic | Gemini |
|--------|--------|-----------|--------|
| Declaration 外層 | `{type: "function", function: {...}}` | `{name, description, input_schema}` | `{functionDeclarations: [{...}]}` |
| Schema 欄位 | `parameters` | `input_schema` | `parameters` |
| Response 容器 | assistant 訊息上的 `tool_calls[]` | 型別為 `tool_use` 的 `content[]` | 型別為 `functionCall` 的 `parts[]` |
| Arguments 型別 | 字串化的 JSON | 已解析的物件 | 已解析的物件 |
| Id 格式 | `call_...`（OpenAI 產生） | `toolu_...`（Anthropic） | UUID（Gemini 3+） |
| Result block | role `tool`、`tool_call_id` | 帶有 `tool_result` 的 `user`、`tool_use_id` | 帶有對應 `id` 的 `functionResponse` |
| 強制某個 tool | `tool_choice: {type: "function", function: {name}}` | `tool_choice: {type: "tool", name}` | `tool_config: {function_calling_config: {mode: "ANY"}}` |
| 禁止 tool | `tool_choice: "none"` | `tool_choice: {type: "none"}` | `mode: "NONE"` |
| Strict schema | `strict: true` | 每個 tool 各自 `strict: true`（需手動啟用） | 請求層級的 `responseSchema` |

### 你真的會碰到的限制 (Limits you will actually hit)

- **OpenAI。** 每個 request 128 個 tool。Schema 深度 5。參數字串 <= 8192 bytes。Strict mode 要求不能有 `$ref`，不能有互相重疊的 `oneOf`/`anyOf`/`allOf`，而且每個 property 都要列在 `required` 裡。
- **Anthropic。** 每個 request 64 個 tool。Schema 深度實質上沒有上限，但實務上限是 10。Strict tool use 需要手動啟用，在 tool definition 上設 `strict: true`。沒有它的話，schema 只是模型傾向遵守的合約，不是保證。
- **Gemini。** 每個 request 64 個 function。Schema 型別是 OpenAPI 3.0 的子集（與 JSON Schema 2020-12 略有出入）。從 Gemini 3 起，平行 call 有唯一 id。

### `tool_choice` 的行為

三種模式大家都支援，只是名稱不同。

- **Auto。** 模型自己選擇 tool 或文字。這是預設值。
- **Required / Any。** 模型必須至少呼叫一個 tool。
- **None。** 模型不得呼叫 tool。

另外每個 provider 各有一種獨有的模式：

- **OpenAI。** 用名稱強制指定某個 tool。
- **Anthropic。** 用名稱強制指定某個 tool；`disable_parallel_tool_use` 旗標用來區分單一與多重 call。
- **Gemini。** `mode: "VALIDATED"` 不管模型的意圖，把每個回應都送進 schema validator。

### 平行 call (Parallel calls)

OpenAI 的 `parallel_tool_calls: true`（預設）會在一則 assistant 訊息中輸出多個 call。你把它們全部執行，再用一則批次的 tool-role 訊息回覆，每個 `tool_call_id` 各有一筆。Anthropic 過去是單一 call；`disable_parallel_tool_use: false`（自 Claude 3.5 起為預設）則啟用多重 call。Gemini 2 允許平行 call，但沒有提供穩定的 id；Gemini 3 加入 UUID，讓亂序回來的回應也能乾淨地對應。

### 串流 (Streaming)

三家都支援串流 tool call，但 wire format 不同：

- **OpenAI。** `tool_calls[i].function.arguments` 的 delta chunk 會逐步到達。你持續累積，直到 `finish_reason: "tool_calls"`。
- **Anthropic。** 有 block-start / block-delta / block-stop 事件。`input_json_delta` chunk 攜帶部分參數。
- **Gemini。** `streamFunctionCallArguments`（Gemini 3 新增）輸出帶有 `functionCallId` 的 chunk，讓多個平行 call 可以交錯。

Phase 13 · 03 會深入探討平行加串流的重組。這堂課聚焦在 declaration 與單一 call 的形狀。

### 錯誤與修復 (Errors and repair)

無效參數的錯誤看起來也不一樣。

- **OpenAI（non-strict）。** 模型回傳 `arguments: "{bad json}"`，你的 JSON 解析失敗，你注入錯誤訊息並重新呼叫。
- **OpenAI（strict）。** 驗證發生在 decoding 期間；不可能出現無效 JSON，但可能出現 `refusal`。
- **Anthropic（non-strict）。** `input` 可能包含非預期的欄位；schema 只是建議。請在 server 端驗證。使用 `strict: true` 時，sampling 會被限制在 schema 之內。
- **Gemini。** OpenAPI 3.0 的怪癖：物件欄位上的 `enum` 會被默默忽略，要自己驗證。

### Translator 模式 (The translator pattern)

在你的程式碼裡，標準的 tool declaration 長這樣（形狀由你決定）：

```python
Tool(
    name="get_weather",
    description="Use when ...",
    input_schema={"type": "object", "properties": {...}, "required": [...]},
    strict=True,
)
```

三個小函式把它翻譯成三家 provider 的形狀。`code/main.py` 裡的 harness 正是這麼做，然後把一個假的 tool call 送過每家 provider 的 response 形狀來回轉換一遍。不需要網路，這堂課教的是形狀，不是 HTTP。

Production 團隊會把這個 translator 包進 `AbstractToolset`（Pydantic AI）、`UniversalToolNode`（LangGraph）或 `BaseTool`（LlamaIndex）。Phase 13 · 17 會交付一個 gateway，在三家之中任何一家前面，提供 OpenAI 形狀的 API。

```figure
function-call-args
```

## 實際使用 (Use It)

`code/main.py` 定義了一個標準的 `Tool` dataclass，以及三個 translator，分別輸出 OpenAI、Anthropic 與 Gemini 的 declaration JSON。接著它把每種形狀的手工 provider response，解析成同一個標準 call 物件，證明在表面之下語意其實完全相同。執行它，並把三份 declaration 並排比對。

觀察重點：

- 三個 declaration block 只有外層和欄位名稱不同。
- 三個 response block 的差別在 call 所在的位置（頂層 `tool_calls`、`content[]` block、`parts[]` 項目）。
- 一個 `canonical_call()` 函式能從三種 response 形狀中取出 `{id, name, args}`。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-provider-portability-audit.md`。給定針對某一家 provider 的 function-calling 整合，這個 skill 會產出可攜性稽核：它依賴哪些 provider 限制、哪些欄位需要改名，以及移植到其他每家 provider 時會壞掉什麼。

## 練習 (Exercises)

1. 執行 `code/main.py`，確認三份 provider declaration JSON 都是序列化同一個底層的 `Tool` 物件。修改標準 tool，加入一個 enum 參數，確認只有 Gemini translator 需要處理 OpenAPI 的怪癖。

2. 為每家 provider 新增 `ListToolsResponse` parser，用來取出模型在 `list_tools` 或 discovery call 之後回傳的 tool 清單。OpenAI 原生沒有這個功能，請記下這個不對稱。

3. 實作 `tool_choice` 轉換：把標準的 `ToolChoice(mode="force", tool_name="x")` 對應到三家 provider 的形狀。接著對應 `mode="any"` 和 `mode="none"`。請對照本課的差異表。

4. 從三家 provider 中挑一家，從頭到尾讀完它的 function-calling 指南。找出它的 schema 規格中，另外兩家都不支援的一個欄位。候選項：OpenAI 的 `strict`、Anthropic 的 `disable_parallel_tool_use`、Gemini 的 `function_calling_config.allowed_function_names`。

5. 寫一組測試向量：一個參數違反宣告 schema 的 tool call。讓它通過每家 provider 的 validator（課程 01 的 stdlib 版本可以當代理），記錄會觸發哪些錯誤。並記下 production 中你會選哪家 provider 來取得最嚴格的檢查。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Function calling | 「Tool use」 | Provider 層級、用來輸出結構化 tool call 的 API |
| Tool declaration | 「Tool 規格」 | Name + description + JSON Schema input payload |
| `tool_choice` | 「強制 / 禁止」 | Auto / required / none / 指定名稱 等模式 |
| Strict mode | 「Schema 強制」 | OpenAI 的旗標，把 decoding 限制在符合 schema 的範圍內 |
| `tool_use` block | 「Anthropic 的 call 形狀」 | 內含 id、name、input 的 inline content block |
| `functionCall` part | 「Gemini 的 call 形狀」 | `parts[]` 中包含 name、args 與 id 的項目 |
| Arguments-as-string | 「字串化 JSON」 | OpenAI 把 args 以 JSON 字串回傳，而不是物件 |
| Parallel tool calls | 「一個 turn 內 fan-out」 | 一則 assistant 訊息中有多個 tool call |
| Refusal | 「模型拒絕」 | 只有 strict mode 才會出現、用來取代 call 的 refusal block |
| OpenAPI 3.0 subset | 「Gemini schema 怪癖」 | Gemini 使用的是類似 JSON-Schema 但有細微差異的方言 |

## 延伸閱讀 (Further Reading)

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — 標準參考，包含 strict mode 與平行 call
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — `tool_use` 與 `tool_result` block 的語意
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — 平行 call、唯一 id 與 OpenAPI 子集
- [Vertex AI — Function calling reference](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/multimodal/function-calling) — Gemini 的企業版介面
- [OpenAI — Structured outputs](https://platform.openai.com/docs/guides/structured-outputs) — strict-mode schema 強制的細節
