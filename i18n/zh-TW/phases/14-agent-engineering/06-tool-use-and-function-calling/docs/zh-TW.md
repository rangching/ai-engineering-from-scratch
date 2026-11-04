# Tool Use 與 Function Calling (Tool Use and Function Calling)

> Toolformer（Schick et al., 2023）開創了自監督的 tool 標註。Berkeley Function Calling Leaderboard V4（Patil et al., 2025）訂下 2026 年的標準：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination。單輪已經解決了。Memory、動態決策和長時程的 tool chain 還沒有。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 13 · 01 (Function Calling Deep Dive)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 解釋 Toolformer 的自監督訓練訊號：只有在執行結果能降低 next-token loss 時，才保留 tool 標註。
- 說出 BFCL V4 的五個評估類別，以及各自衡量什麼。
- 用 stdlib 實作 tool registry，包含 schema 驗證、參數轉型和執行 sandbox。
- 診斷 2026 年的三個未解問題：長時程 tool chaining、動態決策和 memory。

## 問題 (The Problem)

早期的 tool use 問的是：模型能預測出正確的 function call 嗎？現代的 tool use 問的是：模型能不能串起 40 步的 tool，帶著 memory、在部分可觀測的情況下、從 tool 失敗中恢復，而且不 hallucinate 不存在的 tool？

Toolformer 建立了 baseline：模型可以透過自監督學會何時呼叫 tool。BFCL V4 定義了 2026 年的評估目標。兩者之間的差距，就是正式環境 agent 所處的空間。

## 核心概念 (The Concept)

### Toolformer (Schick et al., NeurIPS 2023)

想法：讓模型替自己的 pretraining 語料標註候選的 API call。對每個候選都執行一次。只有在納入 tool 結果能降低下一個 token 的 loss 時，才保留該標註。再用過濾後的語料做 fine-tune。

涵蓋的 tool：計算機、QA 系統、搜尋引擎、翻譯器、行事曆。自監督訊號純粹看 tool 是否有助於預測文字，不需要人工標籤。

規模結果：tool use 在規模夠大時才會出現。較小的模型會被 tool 標註傷害；較大的模型則受益。這就是為什麼 2026 年的前沿模型天生就有很強的 tool use，而大多數 7B 模型需要明確的 tool-use fine-tuning 才可靠。

### Berkeley Function Calling Leaderboard V4 (Patil et al., ICML 2025)

BFCL 是 2026 年事實上的評估標準。V4 的組成：

- **Agentic (40%)**：完整的 agent trajectory：memory、multi-turn、動態決策。
- **Multi-Turn (30%)**：帶有 tool chain 的互動對話。
- **Live (10%)**：使用者提交的真實 prompt（分布更難）。
- **Non-Live (10%)**：合成的測試案例。
- **Hallucination (10%)**：偵測何時不應該呼叫任何 tool。

V3 引入了以狀態為基礎的評估：在一連串 tool 執行之後，檢查 API 的實際狀態（例如「檔案建立了嗎？」），而不是比對 tool call 的 AST。V4 加入了 web search、memory 和格式敏感度類別。

2026 年的關鍵發現：單輪 function calling 已接近解決。失敗集中在 memory（跨 turn 攜帶 context）、動態決策（依先前結果選 tool）、長時程 chain（20 步以上後漂移），以及 hallucination 偵測（沒有合適 tool 時拒絕呼叫）。

### Tool schema

每個 provider 都有 schema。細節不同，但形狀相同：

```
name: string
description: string (what it does, when to use it)
input_schema: JSON Schema (properties, required, types, enums)
```

Anthropic 直接用 `input_schema`。OpenAI 用 `function.parameters`。兩者都接受 JSON Schema。Description 是承重的，因為模型靠讀它來挑對 tool。糟糕的 tool description 是挑錯 tool 失敗的第一名根本原因。

### 參數驗證 (Argument validation)

不要信任任何 tool call。要驗證：

1. **型別轉換 (Type coercion)。** 模型可能在 schema 要求 int 的地方回傳字串 "5"。能明確判斷就轉；不能就拒絕。
2. **Enum 驗證。** 如果 schema 說 `status in {"open", "closed"}`，模型卻輸出 `"in_progress"`，就用描述性的錯誤拒絕。
3. **必填欄位。** 缺少必填欄位 -> 立即把錯誤 observation 回傳給模型，而不是當機。
4. **格式驗證。** 日期、email、URL：用具體的 parser 驗證，不要用 regex。

每次驗證失敗都應該回傳結構化的 observation，讓模型能用正確的形狀重試。

### 平行 tool call (Parallel tool calls)

現代 provider 支援在一個 assistant turn 中做平行 tool call。流程：

1. 模型輸出 3 個 tool call，各有不同的 `tool_use_id`。
2. Runtime 執行它們（互相獨立的話就平行執行）。
3. 每個結果以 `tool_result` block 送回，並用 `tool_use_id` 對應。

工程規則：把 correlation ID 當作承重的東西。把它們換掉，就會把結果送到錯的 tool。

### Sandboxing

Tool 執行是 sandbox 邊界。細節見 Phase 13 · 26。簡短版本：每個 tool 都應該指定讀寫範圍、網路存取、timeout、記憶體上限。通用的 `run_shell(cmd)` 是警訊；具體的 `git_status()` 比較安全。

```figure
tool-routing
```

## 動手做 (Build It)

`code/main.py` 實作一個接近正式環境形狀的 tool registry：

- JSON Schema 子集的 validator（只用 stdlib）。
- Tool 註冊，包含 description、輸入 schema、timeout 和 executor。
- 參數轉型與 enum 驗證。
- 帶 correlation ID 的平行 tool 分派。
- 以結構化字串表示的錯誤 observation。

執行：

```
python3 code/main.py
```

Trace 顯示一個迷你 agent 在一個 turn 中呼叫三個 tool，其中一個刻意格式錯誤的 call 被拒絕，並回傳模型可以據以行動的描述性錯誤。

## 實際使用 (Use It)

每個 provider 都有自己的 tool schema：Anthropic、OpenAI、Gemini、Bedrock。如果需要支援多個 provider，就用轉換層（OpenAI Agents SDK、Vercel AI SDK、LangChain tool adapter）。BFCL 是參考 benchmark：如果 tool use 是產品的核心，出貨前先用它測你的 agent。

## 交付 (Ship It)

`outputs/skill-tool-registry.md` 為特定任務領域產生 tool catalog、schema 和 registry。包含 description 品質檢查（每個 tool 的 description 有沒有告訴模型何時該用它？）。

## 練習 (Exercises)

1. 加一個「no-op」tool，讓模型可以明確拒絕使用任何其他 tool。在類似 BFCL 的 hallucination 測試上量測。
2. 實作 int-as-string 和 float-as-string 的參數轉型。轉型從哪裡開始會掩蓋真正的 bug？
3. 加上每個 tool 的 timeout 和 circuit breaker（連續失敗 3 次後，60 秒內拒絕該 tool）。這會如何改變模型的恢復方式？
4. 讀 BFCL V4 的說明。挑一個類別（例如「multi-turn」），用你的 agent 跑 10 個範例 prompt。回報通過率。
5. 把 stdlib validator 移植到 Pydantic 或 Zod。Pydantic/Zod 抓到了 toy 漏掉的什麼？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Function calling | 「Tool use」 | 以經過驗證的 schema 做結構化輸出的 tool 呼叫 |
| Toolformer | 「自監督 tool 標註」 | Schick 2023：保留結果能降低 next-token loss 的 tool call |
| BFCL | 「Berkeley Function Calling Leaderboard」 | 2026 年 benchmark：40% agentic、30% multi-turn、10% live、10% non-live、10% hallucination |
| Tool schema | 「給模型看的 function signature」 | name、description、參數的 JSON Schema |
| tool_use_id | 「Correlation ID」 | 把 tool call 與其結果綁在一起；平行分派不可或缺 |
| Hallucination detection | 「知道何時不要呼叫」 | V4 類別：沒有合適 tool 時拒絕呼叫 |
| Argument coercion | 「字串轉 int 的修補」 | 針對可預期的 schema 不符所做的窄範圍修正；含糊時就拒絕 |
| Sandboxing | 「Tool 執行邊界」 | 每個 tool 的讀寫範圍、網路、timeout、記憶體上限 |

## 延伸閱讀 (Further Reading)

- [Schick et al., Toolformer (arXiv:2302.04761)](https://arxiv.org/abs/2302.04761)：自監督 tool 標註
- [Berkeley Function Calling Leaderboard (V4)](https://gorilla.cs.berkeley.edu/leaderboard.html)：2026 年的評估 benchmark
- [Anthropic, Tool use documentation](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude Agent SDK 中正式環境的 tool schema
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/)：function tool 型別與 Guardrails
