# A2A：Agent-to-Agent Protocol (A2A — Agent-to-Agent Protocol)

> MCP 是 agent 對 tool。A2A (Agent2Agent) 是 agent 對 agent，是一個開放的 protocol，讓建立在不同框架上、內部不透明的 agent 能夠協作。由 Google 在 2025 年 4 月發布，2025 年 6 月捐贈給 Linux Foundation，2026 年 4 月達到 v1.0，有超過 150 個支持者，包括 AWS、Cisco、Microsoft、Salesforce、SAP 和 ServiceNow。它吸收了 IBM 的 ACP，並加入 AP2 付款 extension。這堂課會走過 Agent Card、Task 生命週期，以及三種 protocol binding，使用 A2A 1.0.1 的 wire 名稱。

**Type:** Build
**Languages:** Python (stdlib, Agent Card + Task harness)
**Prerequisites:** Phase 13 · 06 (MCP fundamentals), Phase 13 · 08 (MCP client)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 區分 agent 對 tool (MCP) 與 agent 對 agent (A2A) 的使用情境。
- 在 `/.well-known/agent-card.json` 發布帶有 skill 與 `supportedInterfaces` metadata 的 Agent Card。
- 走過 Task 生命週期：`TASK_STATE_SUBMITTED`、`TASK_STATE_WORKING`、`TASK_STATE_INPUT_REQUIRED`，以及終結狀態 `TASK_STATE_COMPLETED`、`TASK_STATE_FAILED`、`TASK_STATE_CANCELED`、`TASK_STATE_REJECTED`。
- 使用 Message，其每個 Part 各自持有 `text`、`raw`、`url` 或 `data` 其中之一，並以 Artifact 作為輸出。

## 問題 (The Problem)

客服 agent 需要把撰寫報告的工作，委派給專門的寫作 agent。A2A 出現之前的選項：

- 自訂 REST API。可行，但每一組配對都是一次性的。
- 共用程式碼庫。要求兩個 agent 執行相同的框架。
- MCP。不適合：MCP 是用來呼叫 tool，不是讓兩個 agent 在保留各自不透明內部推理的情況下協作。

A2A 填補了這個缺口。它把互動建模為一個 agent 將 Task 送給另一個 agent，帶有生命週期、訊息與 artifact。被呼叫的 agent 內部狀態保持不透明，呼叫者只看得到 task 的狀態轉換與最終輸出。

A2A 是「讓跨框架的 agent 彼此對話」的 protocol。它不取代 MCP；兩者是互補的。

## 核心概念 (The Concept)

### Agent Card

每個符合 A2A 的 agent 都在 `/.well-known/agent-card.json` 發布一張 card：

```json
{
  "name": "research-agent",
  "description": "Summarizes academic papers and drafts citations.",
  "version": "1.2.0",
  "supportedInterfaces": [
    {
      "url": "https://research.example.com/a2a",
      "protocolBinding": "JSONRPC",
      "protocolVersion": "1.0"
    }
  ],
  "capabilities": {"streaming": true, "pushNotifications": true},
  "securitySchemes": {
    "bearer": {"httpAuthSecurityScheme": {"scheme": "Bearer"}}
  },
  "securityRequirements": [{"schemes": {"bearer": {"list": []}}}],
  "defaultInputModes": ["text/plain"],
  "defaultOutputModes": ["text/markdown"],
  "skills": [
    {
      "id": "summarize_paper",
      "name": "Summarize a paper",
      "description": "Read a paper PDF and produce a 3-paragraph summary.",
      "tags": ["research", "summarization"],
      "inputModes": ["text/plain", "application/pdf"],
      "outputModes": ["text/markdown"]
    }
  ]
}
```

Discovery 是以 URL 為基礎：取得 card，挑選第一個 `protocolBinding` 是你的 client 能使用的 `supportedInterfaces` 項目，並列舉 skill。輸入與輸出模式是 media type。

### 簽章的 Agent Card

Card 可以帶有 `signatures` 陣列。每個項目都是一個 JWS (RFC 7515)，對 card 的 RFC 8785 canonical JSON 計算而得，並排除 `signatures` 欄位。使用者以相同的方式正規化 card 並驗證。可防止冒充。

### Task 生命週期

```text
TASK_STATE_SUBMITTED
  -> TASK_STATE_WORKING
  -> TASK_STATE_COMPLETED | TASK_STATE_FAILED | TASK_STATE_CANCELED | TASK_STATE_REJECTED

TASK_STATE_WORKING
  -> TASK_STATE_INPUT_REQUIRED
  -> TASK_STATE_WORKING (the client sends a message with the same taskId)
```

Client 以 `SendMessage` 發起，由 server 建立 Task。被呼叫的 agent 逐一經過各個狀態；client 用 `GetTask` 輪詢，或用 `SendStreamingMessage` 和 `SubscribeToTask` 透過 SSE 串流。串流攜帶 `statusUpdate` 與 `artifactUpdate` 事件，並在 task 到達終結狀態時關閉。沒有 `final` 旗標。

### Message 與 Part

Message 有 `messageId`、`role`（`ROLE_USER` 或 `ROLE_AGENT`）以及一個或多個 Part。每個 Part 恰好持有一個內容欄位，而該欄位名稱就是型別。沒有 `kind` 欄位。

- `text`：純文字內容。
- `raw`：檔案 bytes，在 JSON 中以 base64 表示，通常帶有 `filename` 與 `mediaType`。
- `url`：指向檔案內容的連結。
- `data`：結構化的 JSON payload（給被呼叫 agent 的結構化輸入）。

範例：

```json
{
  "messageId": "msg-001",
  "role": "ROLE_USER",
  "parts": [
    {"text": "Summarize this paper."},
    {"raw": "...", "filename": "paper.pdf", "mediaType": "application/pdf"},
    {"data": {"targetLength": "3 paragraphs"}, "mediaType": "application/json"}
  ]
}
```

### Artifact

輸出是 Artifact，不是原始字串。Artifact 是有名稱、有型別的輸出：

```json
{
  "artifactId": "art-001",
  "name": "summary",
  "parts": [{"text": "...", "mediaType": "text/markdown"}]
}
```

Artifact 可以分塊串流。每個 `artifactUpdate` 事件攜帶 artifact 以及 `append` 與 `lastChunk`。呼叫者負責累積。

### 三種 protocol binding

1. **JSON-RPC 2.0 over HTTP** (`JSONRPC`)。請求使用 POST，串流使用 SSE。Method 採用 PascalCase：`SendMessage`、`SendStreamingMessage`、`GetTask`、`ListTasks`、`CancelTask`、`SubscribeToTask`、`CreateTaskPushNotificationConfig`、`GetTaskPushNotificationConfig`、`ListTaskPushNotificationConfigs`、`DeleteTaskPushNotificationConfig`，以及 `GetExtendedAgentCard`。
2. **gRPC** (`GRPC`)。適用於 gRPC 為原生的企業環境。Method 名稱相同。
3. **HTTP+JSON/REST** (`HTTP+JSON`)。資源 URL，例如 `POST /message:send` 和 `GET /tasks/{id}`。

三種 binding 都承載相同的資料模型。每個 `supportedInterfaces` 項目指名一種 binding 與它的 `protocolVersion`。Client 在每個請求上送出 header `A2A-Version: 1.0`，因為 server 會把沒有這個 header 的請求讀成 version 0.3。

```http
POST /a2a HTTP/1.1
Host: research.example.com
Content-Type: application/json
A2A-Version: 1.0

{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "SendMessage",
  "params": {
    "message": {
      "messageId": "msg-001",
      "role": "ROLE_USER",
      "parts": [{"text": "Summarize this paper."}]
    }
  }
}
```

### 保持不透明 (Opacity preservation)

一個關鍵的設計原則：被呼叫 agent 的內部狀態是不透明的。呼叫者看到 task 狀態與 artifact。被呼叫 agent 的 chain-of-thought、它的 tool call、它的子 agent 委派，全都看不到。這與 MCP 不同，在 MCP 中 tool call 是透明的。

理由：A2A 讓競爭對手在不揭露內部細節的情況下協作。A2A 可以做到「呼叫這個客服 agent」，而呼叫者不必得知該 agent 是如何實作這項服務的。

### 時間線 (Timeline)

- **2025-04-09。** Google 宣布 A2A。
- **2025-06-23。** 捐贈給 Linux Foundation。
- **2025-08。** 吸收 IBM 的 ACP。
- **2025-09。** AP2 extension (Agent Payments) 發布。
- **2026-04。** v1.0 發布，有超過 150 個支持組織。

### 與 MCP 的關係

| 面向 | MCP | A2A |
|-----------|-----|-----|
| 使用情境 | Agent 對 tool | Agent 對 agent |
| 不透明性 | 透明的 tool call | 不透明的內部推理 |
| 典型呼叫者 | Agent runtime | 另一個 agent |
| 狀態 | Tool-call 結果 | 具生命週期的 Task |
| 授權 | OAuth 2.1 (Phase 13 · 16) | Agent Card 的 `securitySchemes` + `securityRequirements` |
| Transport | Stdio / Streamable HTTP | JSON-RPC / gRPC / HTTP+JSON |

當你想呼叫特定的 tool 時，使用 MCP。當你想把整個 task 委派給另一個 agent 時，使用 A2A。許多 production 系統兩者並用：agent 的 tool 層使用 MCP，協作層使用 A2A。

```figure
a2a-task-lifecycle
```

## 實際使用 (Use It)

`code/main.py` 實作一個最小的 A2A harness：寫作 agent 發布它的 card，研究 agent 對它送出帶有 PDF part 與文字指令的 `SendMessage` 請求，task 依序經過 `TASK_STATE_WORKING` → `TASK_STATE_INPUT_REQUIRED` → `TASK_STATE_WORKING` → `TASK_STATE_COMPLETED`，最後回傳文字 artifact。全部使用 stdlib；採用 in-memory transport，專注在訊息的形狀。

觀察重點：

- Agent Card 的 JSON 形狀。
- Server 端指派 task id 與狀態轉換。
- 以出現哪個內容欄位來決定型別的 Part。
- Task 進行中的 `TASK_STATE_INPUT_REQUIRED` 分支。
- 完成時回傳 Artifact。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-a2a-agent-spec.md`。給定一個應該可被其他 agent 呼叫的新 agent，這個 skill 會產出 Agent Card JSON、skill schema，以及 endpoint 藍圖。

## 練習 (Exercises)

1. 執行 `code/main.py`。追蹤完整的 Task 生命週期，包括被呼叫的 agent 要求釐清的 `TASK_STATE_INPUT_REQUIRED` 暫停。

2. 加入簽章的 Agent Card。在 `signatures` 中放一個 `alg` 設為 `HS256` 的 JWS 項目，對不含 `signatures` 欄位的 card canonical JSON 簽章。寫一個 verifier，並確認它在 card 被修改後會失敗。

3. 用 `SendStreamingMessage` 實作 task 串流：寫作 agent 送出 `task`、三個 `artifactUpdate` chunk，以及帶有 `TASK_STATE_COMPLETED` 的 `statusUpdate`，然後關閉串流。呼叫者累積這些 chunk。

4. 設計一個包裝 MCP server 的 A2A agent。把每個 MCP tool 對應到一個 A2A skill。注意取捨，會失去哪些不透明性？

5. 閱讀 A2A v1.0 的公告，找出截至 2026 年 4 月，還沒有任何框架實作的那一項功能。（提示：它與多跳 task 委派有關。）

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| A2A | 「Agent-to-Agent protocol」 | 用於不透明 agent 協作的開放 protocol |
| Agent Card | 「`/.well-known/agent-card.json`」 | 描述 agent 的 skill 與 `supportedInterfaces` 的已發布 metadata |
| Skill | 「可呼叫的單元」 | agent 支援的具名操作（類似 MCP tool） |
| Task | 「委派的單位」 | 帶有生命週期與最終 artifact 的工作項目 |
| Message | 「Task 輸入」 | 攜帶 Part（`text`、`raw`、`url`、`data`） |
| Part | 「有型別的區塊」 | `text` / `raw` / `url` / `data` 其中恰好一個，加上選擇性的 `mediaType`；沒有 `kind` 欄位 |
| Artifact | 「Task 輸出」 | 完成時回傳、有名稱且有型別的輸出 |
| AP2 | 「Agent Payments Protocol」 | 建立在 A2A 之上的付款 extension；card 簽章屬於 A2A 核心 (`signatures`) |
| Opacity | 「黑盒協作」 | 被呼叫 agent 的內部細節對呼叫者隱藏 |
| `TASK_STATE_INPUT_REQUIRED` | 「Task 暫停」 | agent 需要更多資訊時的中斷狀態 |

## 延伸閱讀 (Further Reading)

- [a2a-protocol.org](https://a2a-protocol.org/latest/) — A2A 的標準規格
- [a2aproject/A2A — GitHub](https://github.com/a2aproject/A2A) — 參考實作與 SDK
- [A2A v1.0.1 release](https://github.com/a2aproject/A2A/tree/v1.0.1)：這堂課所遵循、已打 tag 的 `docs/specification.md` 與規範性的 `specification/a2a.proto`
- [Linux Foundation — A2A launch press release](https://www.linuxfoundation.org/press/linux-foundation-launches-the-agent2agent-protocol-project-to-enable-secure-intelligent-communication-between-ai-agents) — 2025 年 6 月的治理權轉移
- [Google Cloud — A2A protocol upgrade](https://cloud.google.com/blog/products/ai-machine-learning/agent2agent-protocol-is-getting-an-upgrade) — 路線圖與合作夥伴動能
- [Google Dev — A2A 1.0 milestone](https://discuss.google.dev/t/the-a2a-1-0-milestone-ensuring-and-testing-backward-compatibility/352258) — v1.0 發布說明與向後相容指引
