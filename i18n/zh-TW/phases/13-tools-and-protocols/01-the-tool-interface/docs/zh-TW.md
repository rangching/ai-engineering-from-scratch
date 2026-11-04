# Tool Interface：為什麼 Agent 需要結構化 I/O (The Tool Interface — Why Agents Need Structured I/O)

> 語言模型產生 token，程式則採取行動。兩者之間的落差，就是 tool interface：一份讓模型提出行動請求、再由 host 去執行的合約。2026 年所有的技術堆疊，包括 OpenAI、Anthropic、Gemini 的 function calling、MCP 的 `tools/call`、A2A 的 task parts，都是同一個四步驟迴圈的不同編碼方式。這堂課會幫這個迴圈命名，並示範執行它所需的最小機制。

**Type:** Learn
**Languages:** Python (stdlib, no LLM)
**Prerequisites:** Phase 11 (LLM completion APIs)
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 說明為什麼只能產生文字的 LLM，無法單靠自己對真實世界採取行動。
- 畫出四步驟的 tool-call 迴圈（describe → decide → execute → observe），並指出每個步驟由誰負責。
- 把 tool definition 寫成四個部分：name、description、JSON Schema input，以及確定性的 executor 函式。
- 區分 pure tool 與有副作用的 tool，並說明這個區分為什麼對安全很重要。

## 問題 (The Problem)

LLM 輸出的是下一個 token 的機率分布，輸出介面就只有這樣。如果你問聊天模型「班加羅爾現在天氣如何」，它可以寫出一句聽起來合理的話，但它沒辦法連上天氣 API。這句話可能剛好是對的，也可能已經過期三天了。

Tool interface 的目的，就是補上這個落差。Host 程式（你的 agent runtime、Claude Desktop、ChatGPT、Cursor，或自訂腳本）會向模型公布一份可呼叫的 tool 清單。模型判斷需要採取行動時，會輸出一個結構化的 payload，指名要用哪個 tool 以及它的參數。Host 解析這個 payload，真正執行 tool，再把結果餵回去。迴圈會一直持續，直到模型判斷不需要再呼叫為止。

這份合約的第一版在 2023 年 6 月隨 OpenAI 的 "functions" 參數問世。Anthropic 接著在 Claude 2.1 推出 `tool_use` blocks，Gemini 幾個月後加入 `functionDeclarations`。現在每個 provider 都是同一種形狀：輸入是 JSON-Schema 型別的 tool 清單，輸出是 JSON-payload 的 tool call。Model Context Protocol（2024 年 11 月）把這份合約一般化，讓一個 tool registry 就能服務所有模型。A2A（2026 年 4 月，v1.0）則把同一個基本單元疊到 agent 之間的委派上。

四步驟迴圈是這一切底下不變的結構。Phase 13 其餘的內容，都是它的延伸。

## 核心概念 (The Concept)

### 步驟一：describe

Host 用三個欄位向模型宣告每個 tool。第四個部分是 executor，它留在 host 端，不會傳給模型。

- **Name。** 穩定、機器可讀的識別名稱。用 `get_weather`，不要用「weather thing」。
- **Description。** 一段自然語言的簡介。「當使用者詢問特定城市的目前天氣狀況時使用。不要用於歷史資料。」
- **Input schema。** 一個 JSON Schema 物件（draft 2020-12），描述 tool 的參數。

模型會收到這份清單。現代 provider 會用各自的 template 把這些宣告序列化進 system prompt，所以身為呼叫端的你，只需要處理結構化的形式。

### 步驟二：decide

根據使用者的訊息和可用的 tool，模型會選擇三種行為之一。

1. **直接用文字回答。** 不呼叫 tool。
2. **呼叫一個或多個 tool。** 輸出結構化的 call 物件。在 `parallel_tool_calls: true` 之下（OpenAI 與 Gemini 預設開啟，Anthropic 需要手動啟用），模型可以在同一個 turn 輸出多個 call。
3. **拒絕。** Strict-mode 的 structured output 可以產生帶型別的 `refusal` block，而不是 call。

Tool call 的 payload 有三個穩定欄位：call `id`、tool `name`，以及 JSON `arguments` 物件。id 的用途是讓 host 能把之後的結果對應到特定的 call，在平行 call 以亂序回來時，這點很重要。

### 步驟三：execute

Host 收到 call 之後，會依照宣告的 schema 驗證參數，然後執行 executor。參數無效，代表模型幻想出不存在的欄位，或用了錯誤的型別，這在較弱的模型上非常常見。Production 的 host 遇到無效參數時，會採取三種做法之一：快速失敗並把錯誤回報給模型、用 constrained parser 修復 JSON，或是把驗證錯誤放進 prompt，讓模型重試。

Executor 本身就是普通的程式碼：Python、TypeScript、shell 指令、資料庫查詢都可以。它會產生一個結果，通常是字串，但也可以是任何 JSON 值，或結構化的 content block（MCP 裡的 text、image 或 resource reference）。結果必須能被序列化。

### 步驟四：observe

Host 把 tool 的結果附加到對話中（以帶有對應 `id` 的 `tool` role 訊息形式），然後重新呼叫模型。此時模型的 context 裡已經有 tool 的輸出，可以產生最終答案，或要求更多 call。這個過程會持續到模型不再輸出 call，或是 host 碰到迭代次數的安全上限。

### 信任分界 (The trust split)

Tool 分成兩種，對安全而言差別很大。

- **Pure。** 唯讀、確定性、沒有副作用。例如 `get_weather`、`search_docs`、`get_current_time`。可以放心地預先呼叫。
- **Consequential。** 會改變狀態、花錢，或碰觸使用者資料。例如 `send_email`、`delete_file`、`execute_trade`。必須加上管控。

Meta 在 2026 年提出的 agent 安全「Rule of Two」指出，單一 turn 最多只能同時具備以下三者中的兩個：不受信任的輸入、敏感資料、會造成後果的行動。Tool interface 正是執行這條規則的地方，做法包括拒絕 call、要求使用者確認，或是提升 scope。完整的安全章節請見 Phase 13 · 15，agent 層級的權限政策請見 Phase 14 · 09。

### 迴圈在哪裡運作 (Where the loop lives)

| Context | 誰負責 describe | 誰負責 decide | 誰負責 execute |
|---------|---------------|-------------|--------------|
| 單輪 function calling (OpenAI/Anthropic/Gemini) | 應用程式開發者 | LLM | 應用程式開發者 |
| MCP | MCP server | 透過 MCP client 的 LLM | MCP server |
| A2A | Agent Card 發布者 | 發起呼叫的 agent | 被呼叫的 agent |
| Web browser (function-calling agent) | Browser extension / WebMCP | LLM | Browser runtime |

不管在哪裡，都是同樣的四個步驟。欄位名稱會變，結構不會變。

### 為什麼不直接要求模型輸出 JSON？

「請模型用 JSON 回覆」是 function calling 出現之前的做法。在前沿模型上，它大約有 5% 到 15% 的失敗率，在較小的模型上則更高。常見的失敗包括少了大括號、多餘的逗號、幻想出來的欄位和錯誤的型別。接著你還得加一道 JSON 修復、重試，或 constrained decoder。

原生 function calling 比較好，原因有三。第一，provider 會針對確切的 call 形狀對模型做端對端訓練，所以在 strict mode 下，合法 JSON 的比率可以升到 98% 到 99%。第二，call payload 放在專屬的 protocol 欄位，而不是夾在自由文字裡，所以 tool call 絕不會漏進使用者看得到的回覆。第三，provider 會用 constrained decoding 強制 schema 合規（OpenAI 的 strict mode、Anthropic 的 `tool_use`、Gemini 的 `responseSchema`），輸出保證能通過驗證。

Phase 13 · 02 會並排走過三家 provider 的 API，Phase 13 · 04 則深入 structured output。

### 斷路器 (Circuit breakers)

迴圈會在模型不再輸出 call，或 host 達到最大 turn 數時結束。Production 的 host 通常把上限設在 5 到 20 個 turn。超過這個數字，你幾乎可以確定是掉進模型出不來的迴圈。Claude Code 預設是 20，OpenAI Assistants 是 10，Cursor 的 agent mode 是 25。

另一種做法是不設上限的迴圈，每半年就會冒出一篇「agent 一夜之間燒掉 400 美元 API 費用」的事後檢討。沒設上限，就不要上線。

Phase 14 · 12 會深入談錯誤復原與自我修復，Phase 17 則談 production 的 rate limit。

### Phase 13 接下來的走向 (Where Phase 13 goes from here)

- 課程 02 到 05 打磨 provider 層級的 tool-call 介面。
- 課程 06 到 14 把這個迴圈一般化成 MCP。
- 課程 15 到 18 保護這個迴圈，抵禦惡意 server、敵對使用者，以及未經驗證的遠端 auth 介面。
- 課程 19 到 22 把這個模式延伸到 agent 之間的協作、observability、routing 與封裝。
- 課程 23 用上所有基本單元，交付一個完整的生態系。

之後的每一堂課，都是這個四步驟迴圈的延伸。請把它記在心裡，當作不變的基礎。

```figure
tp-tool-loop
```

## 實際使用 (Use It)

`code/main.py` 在沒有 LLM 的情況下執行這個四步驟迴圈。一個假的「decider」函式會用比對使用者訊息的方式模擬模型，而 executor、schema validator 和 observe 步驟的 harness 都是真的。執行它，就能看到完整的 request/response 編排，以及可列印的中間狀態，之後的課程再把假的 decider 換成任何真正的 provider。

觀察重點：

- Tool registry 為每個 tool 保存四個欄位：name、description、schema 和 executor 參照。只有前三個會送給模型。
- Validator 是 JSON Schema 的最小子集（types、required、enum、min/max），只用 stdlib 寫成。Phase 13 · 04 會提供更完整的版本。
- 迴圈把迭代次數限制在五次。Production 的 agent 正需要這種斷路器。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-tool-interface-reviewer.md`。給定一份 tool definition 草稿（name + description + schema + executor 大綱），這個 skill 會從迴圈適用性的角度稽核它：name 是否機器穩定、description 是否是完整的使用簡介、schema 是否正確使用 JSON Schema 2020-12，以及 pure 與 consequential 的分類是否明確。

## 練習 (Exercises)

1. 在 `code/main.py` 新增第四個 tool，叫做 `get_stock_price(ticker)`。把它的 description 寫成「當使用者依 ticker 詢問目前股價時使用。不要用於歷史股價或市場摘要。」執行 harness，確認假的 decider 會把提到 ticker 的查詢導向新 tool。

2. 破壞 schema validator。傳入一個 `arguments` 物件缺少必要欄位的 call，確認 host 在執行前就拒絕它。接著傳入一個多了未知欄位的 call。請決定：host 應該拒絕還是忽略？用安全面的論點說明你的選擇。

3. 把 harness 中的每個 tool 分類為 pure 或 consequential。在需要的 registry 項目加上 `consequential: true` 旗標，並修改迴圈，每當選到 consequential tool 時，就印出一行「would confirm with user」。這就是每個 production host 都需要的確認閘門的樣子。

4. 在紙上畫出四步驟迴圈，並把上面的 provider 欄位表，填成你最愛的 client（Claude Desktop、Cursor、ChatGPT 或自訂的堆疊）。再對照 Phase 13 · 06 中 MCP 專屬的變體。

5. 從頭到尾讀完 OpenAI 的 function-calling 指南。找出那個出現在 request 裡、但不在這裡所介紹的四步驟迴圈中的欄位。說明它多帶來了什麼，以及為什麼它只是方便而不是必要。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Tool | 「模型可以呼叫的東西」 | 由 name + JSON-Schema 型別的 input + executor 函式組成的三元組 |
| Function calling | 「原生 tool use」 | Provider 層級的 API 支援，用來輸出結構化的 tool call，而不是散文 |
| Tool call | 「模型要求採取行動」 | 由模型輸出、包含 `id`、`name`、`arguments` 的 JSON payload |
| Tool result | 「tool 回傳的內容」 | Executor 的輸出，包在帶有對應 id 的 `tool` role 訊息裡 |
| Parallel tool calls | 「一次多個 call」 | 同一個模型 turn 中的多個 call 物件，彼此獨立，可依 id 排序 |
| Strict mode | 「保證是 JSON」 | 用 constrained decoding 強制模型的輸出符合宣告的 schema |
| Pure tool | 「唯讀 tool」 | 沒有副作用，可以放心重跑 |
| Consequential tool | 「行動型 tool」 | 會改變外部狀態，需要閘門、稽核或使用者確認 |
| Four-step loop | 「tool-call 週期」 | describe → decide → execute → observe |
| Host | 「Agent runtime」 | 持有 tool registry、呼叫模型並執行 executor 的程式 |

## 延伸閱讀 (Further Reading)

- [OpenAI — Function calling guide](https://platform.openai.com/docs/guides/function-calling) — OpenAI 風格 tool 宣告與 call 形狀的標準參考
- [Anthropic — Tool use overview](https://docs.anthropic.com/en/docs/agents-and-tools/tool-use/overview) — Claude 的 `tool_use` / `tool_result` block 格式
- [Google — Gemini function calling](https://ai.google.dev/gemini-api/docs/function-calling) — Gemini 中的 `functionDeclarations` 與平行 call 語意
- [Model Context Protocol — Specification 2026-07-28](https://modelcontextprotocol.io/specification/2026-07-28) — 目前無狀態、與 provider 無關的 tool interface 一般化版本
- [JSON Schema — 2020-12 release notes](https://json-schema.org/draft/2020-12/release-notes) — 所有現代 tool API 使用的 schema 方言
