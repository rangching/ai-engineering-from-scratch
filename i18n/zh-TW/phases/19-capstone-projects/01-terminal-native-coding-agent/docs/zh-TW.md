# Capstone 01 — 終端機原生 Coding Agent (Terminal-Native Coding Agent)

> 到了 2026 年，coding agent 的樣貌已經底定：一個 TUI harness、一份有狀態的 plan、一組放在 sandbox 裡的工具、一個負責規劃、行動、觀察、復原的迴圈。Claude Code、Cursor 3、OpenCode 從遠處看都長得一樣。這個 capstone 要你從頭到尾做一個：CLI 進，pull request 出，然後在 SWE-bench Pro 上跟 mini-swe-agent 和 Live-SWE-agent 比較。你會學到，難的部分不是呼叫模型，而是工具迴圈、sandbox，以及 50 回合執行的成本上限。

**Type:** Capstone
**Languages:** TypeScript / Bun (harness), Python (eval scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools and protocols), Phase 14 (agents), Phase 15 (autonomous systems), Phase 17 (infrastructure)
**Phases exercised:** P0 · P5 · P7 · P10 · P11 · P13 · P14 · P15 · P17 · P18
**Time:** 35 小時

## 問題 (Problem)

2026 年，coding agent 成為最主流的 AI 應用類別。Claude Code (Anthropic)、搭載 Composer 2 與 Agent Tabs 的 Cursor 3 (Cursor)、Amp (Sourcegraph)、OpenCode (112k stars)、Factory Droids 和 Google Jules，全都是同一種架構的變體：終端機 harness、有權限控管的工具、sandbox，以及圍繞前沿模型打造的 plan-act-observe 迴圈。前沿很窄，Live-SWE-agent 用 Opus 4.5 在 SWE-bench Verified 上達到 79.2%，但工程功夫的範圍很廣。大多數失敗不是模型的錯，而是工具迴圈不穩定、context 被污染、token 成本失控，以及破壞性的檔案系統操作。

從外面看，你沒辦法推理這些 agent。你得自己做一個，看著迴圈在第 47 回合因為 ripgrep 回傳 8MB 的比對結果而當掉，再重建截斷層。這就是這個 capstone 的重點。

## 核心概念 (Concept)

Harness 有四個面向。**Plan** 維護一個 TodoWrite 風格的狀態物件，模型每回合都會重寫它。**Act** 派送工具呼叫 (read、edit、run、search、git)。**Observe** 擷取 stdout / stderr / exit code，截斷後把摘要回饋給模型。**Recover** 處理工具錯誤，不讓 context window 爆掉，也不會無限循環。2026 年的樣貌多了一樣東西：**hooks**。`PreToolUse`、`PostToolUse`、`SessionStart`、`SessionEnd`、`UserPromptSubmit`、`Notification`、`Stop` 和 `PreCompact` 都是可設定的擴充點，操作者可以在這裡注入政策、遙測與護欄。

Sandbox 用 E2B 或 Daytona。每個任務都在全新的 devcontainer 裡執行，並以可讀寫方式掛載一個 git worktree。Harness 永遠不碰主機的檔案系統。無論成功或失敗，worktree 都會被拆除。成本控制分三層：每回合的 token 上限、每個 session 的美元預算，以及硬性的回合數上限 (通常是 50)。可觀測性層使用符合 GenAI semantic conventions 的 OpenTelemetry spans，送到自行架設的 Langfuse。

## 架構 (Architecture)

```
  user CLI  ->  harness (Bun + Ink TUI)
                  |
                  v
           plan / act / observe loop  <--->  Claude Sonnet 4.7 / GPT-5.4-Codex / Gemini 3 Pro
                  |                          (via OpenRouter, model-agnostic)
                  v
           tool dispatcher (MCP StreamableHTTP client)
                  |
     +------------+------------+----------+
     v            v            v          v
  read/edit    ripgrep     tree-sitter   git/run
     |            |            |          |
     +------------+------------+----------+
                  |
                  v
           E2B / Daytona sandbox  (worktree isolated)
                  |
                  v
           hooks: Pre/Post, Session, Prompt, Compact
                  |
                  v
           OpenTelemetry -> Langfuse (spans, tokens, $)
                  |
                  v
           PR via GitHub app
```

## 技術堆疊 (Stack)

- Harness runtime：Bun 1.2 + Ink 5 (React-in-terminal)
- 模型存取：OpenRouter 統一 API，搭配 Claude Sonnet 4.7、GPT-5.4-Codex、Gemini 3 Pro、Opus 4.5 (用於最困難的任務)
- 工具傳輸：Model Context Protocol StreamableHTTP (MCP 2026 修訂版)
- Sandbox：E2B sandboxes (JS SDK) 或 Daytona devcontainers
- 程式碼搜尋：ripgrep subprocess、支援 17 種語言的 tree-sitter parsers (預先編譯)
- 隔離：每個任務一個 `git worktree add`，成功或失敗都會清除
- 評測 harness：SWE-bench Pro (verified 子集) + Terminal-Bench 2.0 + 你自己的 30 題 holdout
- 可觀測性：OpenTelemetry SDK 搭配 `gen_ai.*` semconv → 自行架設的 Langfuse
- PR 發佈：使用 fine-grained token 的 GitHub App，權限範圍限定在目標 repo

```figure
ce-agent-loop
```

## 動手做 (Build It)

1. **TUI 與指令迴圈。** 用 Ink 建立 Bun 專案。接受 `agent run <repo> "<task>"`。畫面分割顯示：plan 窗格 (上)、工具呼叫串流 (中)、token 預算 (下)。加入 Ctrl-C 取消功能，結束前會觸發 `SessionEnd` hook。

2. **Plan 狀態。** 定義有型別的 TodoWrite schema (pending / in_progress / done 項目，附註記)。模型每回合以一次工具呼叫重寫完整狀態，不要讓它逐步修改。把 plan 存到 `.agent/state.json`，這樣當機後可以接著跑。

3. **工具介面。** 定義六個工具：`read_file`、`edit_file` (附 diff 預覽)、`ripgrep`、`tree_sitter_symbols`、`run_shell` (附 timeout)、`git` (status / diff / commit / push)。透過 MCP StreamableHTTP 對外提供，讓 harness 不受傳輸方式綁定。每個工具回傳的輸出都要截斷 (每次呼叫上限 4k tokens)。

4. **Sandbox 包裝。** 每個任務啟動一個 E2B sandbox。用 `git worktree add -b agent/$TASK_ID` 建立全新分支。所有工具呼叫都在 sandbox 內執行，主機檔案系統無法存取。

5. **Hooks。** 實作全部八種 2026 hook 類型。至少接上四個使用者自寫的 hook：(a) `PreToolUse` 破壞性指令防護，阻擋 worktree 以外的 `rm -rf`，(b) `PostToolUse` token 計帳，(c) `SessionStart` 預算初始化，(d) `Stop` 寫出最終的 trace bundle。

6. **評測迴圈。** Clone SWE-bench Pro Python 的 30 題子集，用你的 harness 逐題執行。在 pass@1、每任務回合數和每任務美元成本上，與 mini-swe-agent (最小基準線) 比較。把結果寫到 `eval/results.jsonl`。

7. **成本控制。** 硬性上限：50 回合、200k context、每任務 $5。`PreCompact` hook 在 150k 時把較早的回合摘要成一個先前狀態區塊，為新的觀察結果騰出空間，同時不遺失 plan。

8. **PR 發佈。** 成功時，最後一步是 `git push`，加上一次 GitHub API 呼叫，開出 PR，內文包含 plan 與 diff 摘要。

## 實際使用 (Use It)

```
$ agent run ./my-repo "Fix the race condition in worker.rs"
[plan]  1 locate worker.rs and enumerate mutex uses
        2 identify shared state under contention
        3 propose fix, verify tests
[tool]  ripgrep mutex.*lock -t rust           (44 matches, truncated)
[tool]  read_file src/worker.rs 120..180
[tool]  edit_file src/worker.rs (+8 -3)
[tool]  run_shell cargo test worker::          (passed)
[plan]  1 done · 2 done · 3 done
[done]  PR opened: #482   turns=9   tokens=38k   cost=$0.41
```

## 交付 (Ship It)

交付的 skill 放在 `outputs/skill-terminal-coding-agent.md`。給定 repo 路徑和任務描述，它會在 sandbox 中跑完整的 plan-act-observe 迴圈，回傳 PR URL 與 trace bundle。這個 capstone 的評分標準：

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 與基準線比較 | 你的 harness 對上 mini-swe-agent，在 30 題配對的 Python 任務上比較 |
| 20 | 架構清晰度 | Plan/act/observe 的分離、hook 介面、工具 schema，對照 Live-SWE-agent 的架構審查 |
| 20 | 安全性 | Sandbox 逃逸測試、權限提示、破壞性指令防護通過 red-team |
| 20 | 可觀測性 | Trace 完整度 (100% 的工具呼叫都有 span)、每回合 token 計帳 |
| 15 | 開發者體驗 | Cold-start < 2s、當機後可接續 plan、Ctrl-C 能在工具執行中乾淨地取消 |
| **100** | | |

## 練習 (Exercises)

1. 把底層模型從 Claude Sonnet 4.7 換成在 vLLM 上服務的 Qwen3-Coder-30B。比較 pass@1 與每任務美元成本，回報開源模型在哪裡表現較差。

2. 加入 `reviewer` sub-agent，在發佈 PR 前讀取 diff，並可要求進入修訂迴圈。衡量誤報的 review 是否會讓 SWE-bench 通過率低於單一 agent 基準線 (提示：通常會)。

3. 對 sandbox 做壓力測試：寫一個嘗試 `curl` 外部 URL 的任務，以及一個寫入 worktree 之外的任務。確認兩者都被 PreToolUse hook 阻擋，並記錄這些嘗試。

4. 用較小的模型 (Haiku 4.5) 實作 `PreCompact` 摘要。衡量在 3 倍壓縮時 plan 的保真度損失多少。

5. 把 MCP StreamableHTTP 傳輸換成 stdio。對 cold-start 和每次呼叫的延遲做 benchmark，選出純本機使用時的贏家。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Harness | 「agent 迴圈」 | 包在模型外面的程式碼，負責派送工具、維護 plan 狀態、執行預算限制 |
| Hook | 「Agent 事件監聽器」 | 由 harness 在八種生命週期事件之一觸發的使用者自寫腳本 |
| Worktree | 「Git sandbox」 | 位於獨立路徑的連結 git checkout；可隨時丟棄，不影響主要 clone |
| TodoWrite | 「Plan 狀態」 | 一份有型別的 pending/in-progress/done 項目清單，模型每回合重寫一次 |
| StreamableHTTP | 「MCP 傳輸」 | 2026 MCP 修訂版：支援雙向串流的長連線 HTTP，取代 SSE |
| Token ceiling | 「Context 預算」 | 每回合或每 session 的輸入+輸出 token 上限；觸發壓縮或終止 |
| pass@1 | 「單次嘗試通過率」 | 不重試、不偷看測試集，第一次執行就解出的 SWE-bench 任務比例 |

## 延伸閱讀 (Further Reading)

- [Claude Code documentation](https://docs.anthropic.com/en/docs/claude-code) — Anthropic 的參考 harness
- [Cursor 3 changelog](https://cursor.com/changelog) — Agent Tabs 與 Composer 2 的產品說明
- [mini-swe-agent](https://github.com/SWE-agent/mini-swe-agent) — SWE-bench harness 比較用的最小基準線
- [Live-SWE-agent](https://github.com/OpenAutoCoder/live-swe-agent) — 用 Opus 4.5 在 SWE-bench Verified 達到 79.2%
- [OpenCode](https://opencode.ai) — 開放 harness，112k stars
- [SWE-bench Pro leaderboard](https://www.swebench.com) — 這個 capstone 要挑戰的評測
- [Model Context Protocol 2026 roadmap](https://blog.modelcontextprotocol.io/posts/2026-mcp-roadmap/) — StreamableHTTP、capability metadata
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — 工具呼叫與 token 用量的 span schema
