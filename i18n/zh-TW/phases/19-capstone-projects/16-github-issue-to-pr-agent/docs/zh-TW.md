# Capstone 16 — GitHub Issue-to-PR 自主 Agent (GitHub Issue-to-PR Autonomous Agent)

> 貼上標籤，就得到 PR。這是 2026 年自主 coding agent 的產品形態：在雲端 sandbox 中執行 agent、驗證測試通過，並貼出附有理由、可供審查的 PR。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud 和 Google Jules 都有提供。困難的部分是自動重現 repo 的建置環境、防止憑證洩漏、對每個 repo 強制執行預算，以及確保 agent 不能 force-push。這個 capstone 要打造自行架設的版本，並在成本與通過率上與託管方案比較。

**Type:** Capstone
**Languages:** Python (agent), TypeScript (GitHub App), YAML (Actions)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P17
**Time:** 30 小時

## 問題 (Problem)

非同步的雲端 coding agent，是和互動式 coding agent (capstone 01) 不同的產品類別。UX 就是一個 GitHub 標籤。你在 issue 上貼 `@agent fix this` 標籤，worker 就在雲端 sandbox 中啟動、clone repo、執行測試、編輯檔案、驗證，然後開出 PR，並在內文附上 agent 的理由。沒有互動迴圈，也沒有終端機。AWS Remote SWE Agents、Cursor Background Agents、OpenAI Codex cloud、Google Jules 和 Factory Droids 全都收斂到這裡。

工程上的挑戰很具體：環境重現 (agent 必須在沒有快取開發映像檔的情況下，從頭建置 repo)、不穩定的測試 (必須重跑或隔離)、憑證範圍 (權限最小、細粒度的 GitHub App)、每個 repo 每天的預算強制執行，以及禁止 force-push 的政策。這個 capstone 衡量通過率、成本與安全性，並與託管方案比較。

## 核心概念 (Concept)

觸發點是 GitHub webhook (issue 標籤或 PR 留言)。Dispatcher 把工作排入 ECS Fargate 或 Lambda。Worker 把 repo 拉進 Daytona 或 E2B sandbox，並使用從 repo 推斷出的通用 Dockerfile (語言、框架)。Agent 對 Claude Opus 4.7 或 GPT-5.4-Codex 執行 mini-swe-agent 或 SWE-agent v2 迴圈。它反覆迭代：讀程式碼、提出修正、套用 patch、執行測試。

驗證是把關步驟。完整的 CI 必須在 sandbox 中通過，PR 才會開出。覆蓋率差異會被計算；如果負向超過閾值，PR 仍會開出，但會被標上 `needs-review`。Agent 把理由貼成 PR 描述，外加一個 `@agent` 討論串，審查者可以 ping 它來追問後續。

安全性透過兩個不同的 GitHub 層面來界定範圍：App 提供短效的 installation token，具備 `workflows: read` 與狹窄的 repo contents/PR 範圍；branch protection (而不是 app 權限) 強制執行「不得直接寫入 `main`」與「不得 force-push」，而且 app 永遠不會被加進 bypass 清單。對 `.github/workflows` 做路徑範圍的唯讀存取，並不是真正的 GitHub App 原語，所以必須由 worker 上針對檔案編輯的 agent allow-list 來強制執行。每個 repo 每天的預算上限在 dispatcher 強制執行 (例如每個 repo 每天最多 5 個 PR、每個 PR $20)。

## 架構 (Architecture)

```
GitHub issue labeled `@agent fix` or PR comment
            |
            v
    GitHub App webhook -> AWS Lambda dispatcher
            |
            v
    ECS Fargate task (or GitHub Actions self-hosted runner)
       - pull repo
       - infer Dockerfile (language, package manager)
       - Daytona / E2B sandbox with target runtime
       - clone -> git worktree -> agent branch
            |
            v
    mini-swe-agent / SWE-agent v2 loop
       Claude Opus 4.7 or GPT-5.4-Codex
       tools: ripgrep, tree-sitter, read/edit, run_tests, git
            |
            v
    verify CI passes in-sandbox + coverage delta check
            |
            v (verified)
    git push + open PR via GitHub App
       PR body = rationale + diff summary + trace URL
       label: needs-review
            |
            v
    operator reviews; can @-mention agent for follow-ups
```

## 技術堆疊 (Stack)

- 觸發：使用 fine-grained token 的 GitHub App；透過 Lambda 或 Fly.io 的 webhook 接收器
- Worker：ECS Fargate task (或 GitHub Actions 自行架設的 runner)
- Sandbox：每個任務一個 Daytona devcontainer 或 E2B sandbox
- Agent 迴圈：在 Claude Opus 4.7 / GPT-5.4-Codex 之上的 mini-swe-agent 基準線或 SWE-agent v2
- 檢索：tree-sitter repo-map + ripgrep
- 驗證：sandbox 內的完整 CI + 覆蓋率差異關卡
- 可觀測性：Langfuse，每個 PR 的 trace 封存連結放在 PR 內文
- 預算：每個 repo 每日的美元上限；每個 repo 每日 PR 數上限

```figure
cf-issue-to-pr
```

## 動手做 (Build It)

1. **GitHub App。** Fine-grained installation token：issues 讀寫、pull_requests 寫入、contents 讀寫、workflows 讀取。Branch protection (唯一能做到這件事的層面) 強制執行「不得直接 push 到 `main`」與「不得 force-push」；app 不在 bypass 清單中。因為 GitHub App 權限沒有路徑範圍，worker 要對提議的 diff 以 allow-list 檢查，強制執行「不得寫入 `.github/workflows` 底下」。

2. **Webhook 接收器。** Lambda 函式接受 issue 標籤 / PR 留言的 webhook。依標籤 `@agent fix this` 過濾。排入 SQS。

3. **Dispatcher。** 從 SQS 取出任務。強制執行每個 repo 每天的預算。啟動 ECS Fargate task，帶入 repo URL、issue 內文，以及全新的 Daytona sandbox。

4. **環境推斷。** 偵測語言 (Python、Node、Go、Rust) 與套件管理工具 (uv、pnpm、go mod、cargo)。若不存在 Dockerfile，就即時產生一份。

5. **Agent 迴圈。** 使用 Claude Opus 4.7 的 mini-swe-agent 或 SWE-agent v2。工具：ripgrep、tree-sitter repo-map、read_file、edit_file、run_tests、git。硬性限制：$20 成本、30 分鐘牆鐘時間、30 個 agent 回合。

6. **驗證。** 迴圈結束後，在 sandbox 內執行完整測試套件。透過 jacoco / coverage.py 計算覆蓋率差異。如果 CI 是紅的：中止，不開 PR。如果覆蓋率下降超過 2%：開 PR 並加上 `needs-review` 標籤。

7. **發佈 PR。** 推送 agent 分支。透過 GitHub API 開 PR，內含：標題、理由、diff 摘要、trace URL、成本、回合數。

8. **憑證衛生。** Worker 以短效的 GitHub App installation token 執行。日誌在封存前會清除 secret。

9. **評測。** 30 個種子內部 issue，難度不一。衡量通過率、PR 品質 (diff 大小、風格、覆蓋率)、成本、延遲。在相同的 issue 上，與 Cursor Background Agents 和 AWS Remote SWE Agents 比較。

## 實際使用 (Use It)

```
# on github.com
  - user labels issue #842 with `@agent fix this`
  - PR #1903 appears 14 minutes later
  - body:
    > Fixed NPE in widget.dedupe() caused by null comparator entry.
    > Added regression test widget_test.go::TestDedupeNullComparator.
    > Coverage delta: +0.12%
    > Turns: 7  Cost: $1.80  Trace: langfuse:...
    > Label: needs-review
```

## 交付 (Ship It)

`outputs/skill-issue-to-pr.md` 是交付成果。一個 GitHub App + 非同步雲端 worker，把被貼上標籤的 issue 轉成可供審查的 PR，成本有界、憑證範圍受限。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | 30 個 issue 上的通過率 | 端到端成功 (CI 綠燈 + 覆蓋率 OK) |
| 20 | PR 品質 | Diff 大小、覆蓋率差異、風格符合度 |
| 20 | 每個已解決 issue 的成本與延遲 | 每個 PR 的 $ 與牆鐘時間 |
| 20 | 安全性 | 受限範圍的 token、每個 repo 的預算、禁止 force-push、憑證衛生 |
| 15 | 操作者 UX | 理由留言、重試的操作入口、@ 提及後續追問 |
| **100** | | |

## 練習 (Exercises)

1. 加入「修復不穩定測試」模式：標籤 `@agent stabilize-flake TestX` 會在 sandbox 中把該測試跑 50 次，並提出讓它穩定的最小變更。

2. 在三個共同的 issue 上，與 Cursor Background Agents 比較成本。回報哪個工具在哪方面勝出。

3. 實作預算 dashboard：每個 repo 每日成本、每位使用者成本。異常時發出警示。

4. 建立「dry-run」模式，不跑 CI 就開出 draft PR，讓審查者能以低成本檢視計畫。

5. 加入保留政策：超過 7 天沒有合併的 PR 分支自動刪除。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| GitHub App | 「限定範圍的 bot 身分」 | 具備細粒度權限 + 短效 installation token 的 App |
| Async cloud agent | 「背景 agent」 | 在雲端 sandbox 而非終端機中執行的非互動式 worker |
| Environment inference | 「Dockerfile 合成」 | 偵測語言 + 套件管理工具，缺少時產生 Dockerfile |
| Verification | 「Sandbox 內的 CI」 | 開 PR 之前，在 worker 內執行完整測試套件 |
| Coverage delta | 「覆蓋率保留」 | 從基底到 agent 分支的測試覆蓋率 % 變化 |
| Per-repo budget | 「每日上限」 | 在 dispatcher 強制執行的美元與 PR 數上限 |
| Rationale | 「PR 內文說明」 | Agent 對改了什麼、為什麼改的摘要；PR 內文必備 |

## 延伸閱讀 (Further Reading)

- [AWS Remote SWE Agents](https://github.com/aws-samples/remote-swe-agents) — 標準的非同步雲端 agent 參考
- [SWE-agent](https://github.com/SWE-agent/SWE-agent) — CLI 參考
- [Cursor Background Agents](https://docs.cursor.com/background-agent) — 商用替代方案
- [OpenAI Codex (cloud)](https://openai.com/codex) — 託管的競爭者
- [Google Jules](https://jules.google) — Google 的託管版本
- [Factory Droids](https://www.factory.ai) — 另一個商用參考
- [GitHub App documentation](https://docs.github.com/en/apps) — 限定範圍的 bot 身分
- [Daytona cloud sandboxes](https://daytona.io) — 參考 sandbox
