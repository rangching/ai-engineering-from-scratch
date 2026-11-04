# 作為函式庫的 Harness：Subagent 與 Session Store (The Harness as a Library — Subagents and Session Store)

> 一個可以 import 的 harness：內建 tool、用於 context 隔離的 subagent、hook、W3C trace 傳播、session 持久化。Claude Agent SDK 是參考範例，也就是 Claude Code harness 的函式庫形式，而 Claude Managed Agents 則是長時間非同步工作的託管替代方案。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 01 (Agent Loop), Phase 14 · 10 (Skill Libraries)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 解釋 Anthropic Client SDK（原始 API）與 Claude Agent SDK（harness 形態）的差異。
- 描述 subagent（平行化與 context 隔離）以及何時該使用它們。
- 說出 Python SDK 的 session store 介面（`append`、`load`、`list_sessions`、`delete`、`list_subkeys`）以及 `--session-mirror` 的角色。
- 用 stdlib 實作 harness，包含內建 tool、具隔離 context 的 subagent 產生、生命週期 hook 和 session store。

## 問題 (The Problem)

原始的 LLM API 只給你一次來回。正式環境的 agent 需要 tool 執行、MCP server、生命週期 hook、subagent 產生、session 持久化、trace 傳播。Claude Agent SDK 把這個形態當成函式庫提供，也就是 Claude Code 使用的同一個 harness，開放給自訂 agent 使用。

## 核心概念 (The Concept)

### Client SDK 對 Agent SDK (Client SDK vs Agent SDK)

- **Client SDK（`anthropic`）。** 原始的 Messages API。Loop、tool、state 都由你負責。
- **Agent SDK（`claude-agent-sdk`）。** 內建 tool 執行、MCP 連線、hook、subagent 產生、session store。以函式庫形式提供的 Claude Code loop。

### 內建 tool (Built-in tools)

SDK 開箱即用地提供 10 個以上的 tool：檔案讀寫、shell、grep、glob、web fetch 等等。自訂 tool 透過標準的 tool-schema 介面註冊。

### Subagent

Anthropic 文件記載了兩個目的：

1. **平行化。** 並行執行互相獨立的工作。「為這 20 個模組各找出測試檔」就是 20 個平行的 subagent 任務。
2. **Context 隔離。** Subagent 使用自己的 context window；只有結果會回到 orchestrator。Orchestrator 的預算得以保留。

Python SDK 近期新增：`list_subagents()`、`get_subagent_messages()`，用於讀取 subagent 的 transcript。

### Session store

與 TypeScript 的協定對等：

- `append(session_id, message)`：新增一個 turn。
- `load(session_id)`：還原對話。
- `list_sessions()`：列舉。
- `delete(session_id)`：連同 subagent session 一併串聯刪除。
- `list_subkeys(session_id)`：列出 subagent 的 key。

`--session-mirror`（CLI flag）會在 transcript 串流時，將其鏡像到外部檔案，供 debug 使用。

### Hook

你可以註冊的生命週期 hook：

- `PreToolUse`、`PostToolUse`：把關或稽核 tool call。
- `SessionStart`、`SessionEnd`：設定與拆除。
- `UserPromptSubmit`：在模型看到使用者輸入之前處理它。
- `PreCompact`：在 context 壓縮之前執行。
- `Stop`：agent 結束時的清理。
- `Notification`：旁路通知。

Hook 是 pro-workflow（Phase 14 課程參考）與類似系統加入橫切行為的方式。

### W3C trace context

呼叫端上啟用的 OTel span，會透過 W3C trace context header 傳播到 CLI 子行程。整個跨多行程的 trace 在你的後端中會顯示為單一個 trace。

### Claude Managed Agents

託管的替代方案（beta header `managed-agents-2026-04-01`）。長時間非同步工作、內建 prompt caching、內建壓縮。用控制權換取受管理的基礎設施。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Subagent 過度產生 (Subagent over-spawn)。** 為 100 個微小任務產生 100 個 subagent。額外開銷反而佔大宗。改用批次處理。
- **Hook 蔓延 (Hook creep)。** 每個團隊都加 hook；啟動時間膨脹。每季審查 hook。
- **Session 膨脹 (Session bloat)。** Session 持續累積；大小增長。使用 `list_sessions` 加上過期政策。

```figure
ae-subagent-isolation
```

## 動手做 (Build It)

`code/main.py` 用 stdlib 實作 SDK 的形態：

- `Tool`、`ToolRegistry`，內建 `read_file`、`write_file`、`list_dir`。
- `Subagent`：私有 context、隔離執行、回傳結果。
- `SessionStore`：append、load、list、delete、list_subkeys。
- `Hooks`：`pre_tool_use`、`post_tool_use`、`session_start`、`session_end`。
- 示範：主 agent 平行產生 3 個 subagent（各自隔離），彙整結果，持久化 session。

執行：

```
python3 code/main.py
```

Trace 顯示 subagent 的 context 隔離（orchestrator 的 context 大小維持在有界範圍）、hook 執行，以及 session 持久化。

## 實際使用 (Use It)

- **Claude Agent SDK** 用於想要 Claude Code harness 形態、以 Claude 為優先的產品。
- **Claude Managed Agents** 用於託管的長時間非同步工作。
- **OpenAI Agents SDK**（Lesson 16）用於以 OpenAI 為優先的對應方案。
- **LangGraph + 自訂 tool**：如果你想要 graph 形狀的狀態機。

## 交付 (Ship It)

`outputs/skill-claude-agent-scaffold.md` 建立 Claude Agent SDK 應用程式的骨架，包含 subagent、hook、session store、MCP server 掛載和 W3C trace 傳播。

## 練習 (Exercises)

1. 加一個 subagent 產生器，把 20 個任務分批成每組 5 個平行 subagent。量測 orchestrator 的 context 大小，並與每個任務一個 subagent 的做法比較。
2. 實作 `PreToolUse` hook，對 `write_file` 呼叫做速率限制（每個 session 每分鐘 5 次）。追蹤這個行為。
3. 把 `list_subkeys` 接起來，繪製 subagent 樹。深層巢狀看起來是什麼樣子？
4. 把 toy 移植到真正的 `claude-agent-sdk` Python 套件。tool 註冊方式有什麼改變？
5. 讀 Claude Managed Agents 文件。你什麼時候會從自架換成託管？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Agent SDK | 「函式庫形式的 Claude Code」 | Harness 形態：tool、MCP、hook、subagent、session store |
| Subagent | 「子 agent」 | 獨立的 context、自己的預算；結果向上傳回 |
| Session store | 「對話資料庫」 | 持久化、載入、列出、刪除 turn，並串聯 subagent |
| Hook | 「生命週期 callback」 | Tool 前後、session、prompt 送出、壓縮、停止 |
| W3C trace context | 「跨行程 trace」 | 父 span 傳播到 CLI 子行程 |
| Managed Agents | 「託管 harness」 | Anthropic 託管的長時間非同步工作 |
| `--session-mirror` | 「Transcript 鏡像」 | 在 session turn 串流時，將其寫入外部檔案 |
| MCP server | 「Tool 介面」 | 掛載到 agent 的外部 tool/resource 來源 |

## 延伸閱讀 (Further Reading)

- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：Claude Code 的函式庫形式
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)：正式環境模式
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview)：託管替代方案
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/)：對應方案
