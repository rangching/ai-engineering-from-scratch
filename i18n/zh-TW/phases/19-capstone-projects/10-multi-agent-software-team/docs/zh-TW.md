# Capstone 10 — 多 Agent 軟體工程團隊 (Multi-Agent Software Engineering Team)

> 2026 年多 agent 工程團隊的樣貌已經收斂：architect 規劃，N 個 coder 在平行的 worktree 中工作，reviewer 把關，tester 驗證。SWE-AF 的 factory 架構、MetaGPT 的角色式 prompting、AutoGen 0.4 的具型別 actor graph、Cognition 的 Devin，以及 Factory 的 Droids，都各自獨立地走到這裡。平行 worktree 把牆鐘時間換成吞吐量。共享狀態與交接協定則成為失敗面。這個 capstone 要你打造這個團隊，在 SWE-bench Pro 上評測，並回報哪些交接會壞、頻率多高。

**Type:** Capstone
**Languages:** Python / TypeScript (agents), Shell (worktree scripts)
**Prerequisites:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 17 (infrastructure)
**Phases exercised:** P11 · P13 · P14 · P15 · P16 · P17
**Time:** 40 小時

## 問題 (Problem)

單一 agent 的 coding harness 在大型任務上會撞到天花板。不是因為任何單一 agent 太弱，而是 200k-token 的 context 裝不下架構計畫、四個平行的程式碼切片、reviewer 意見，加上測試輸出。多 agent factory 把問題拆開：architect 擁有計畫，coder 在平行 worktree 中擁有實作，reviewer 把關，tester 驗證。SWE-AF 的「factory」架構、MetaGPT 的角色、AutoGen 的具型別 actor graph，三種說法描述的是同一個形狀。

失敗面在交接。Architect 規劃了 coder 實作不出來的東西。Coder 產生互相衝突的 diff。Reviewer 核准了幻覺出來的修正。Tester 跟還在寫程式的 coder 搶時間。你要打造這樣一個團隊，在 50 個 SWE-bench Pro issue 上執行，追蹤每一次交接，並發佈事後檢討。

## 核心概念 (Concept)

角色是具型別的 agent。**Architect** (Claude Opus 4.7) 讀取 issue、撰寫計畫，並拆成附有明確介面的子任務。**Coder** (Claude Sonnet 4.7，N 個平行實例，各自在 `git worktree` + Daytona sandbox 中) 各自獨立實作子任務。**Reviewer** (GPT-5.4) 讀取合併後的 diff，核准或要求具體修改。**Tester** (Gemini 2.5 Pro) 在隔離環境中執行測試套件，並回報通過/失敗與產出物。

溝通透過共享的任務看板 (檔案或 Redis 支撐)。每個角色只消費自己有權處理的任務。交接是 A2A 協定具型別的訊息。協調上的考量：合併衝突解決 (協調者角色或自動三方合併)、共享狀態同步 (coder 一開始工作計畫就凍結；重新規劃是獨立事件)，以及 reviewer 把關 (reviewer 不能核准自己的變更，或自己提議的變更)。

Token 放大是隱藏成本。每個角色邊界都會增加摘要 prompt 與交接 context。單一 agent 的 40 回合執行，在四個角色間會變成共 160 回合。評分標準特別衡量相對單一 agent 基準線的 token 效率，因為問題不是「多 agent 行不行」，而是「每塊錢划不划算」。

## 架構 (Architecture)

```
GitHub issue URL
      |
      v
Architect (Opus 4.7)
   reads issue, produces plan with subtasks + interfaces
      |
      v
Task board (file / Redis)
      |
   +-- subtask 1 ---+-- subtask 2 ---+-- subtask 3 ---+-- subtask 4 ---+
   v                v                v                v                v
Coder A          Coder B          Coder C          Coder D          (4 parallel)
 (Sonnet)         (Sonnet)         (Sonnet)         (Sonnet)
 worktree A       worktree B       worktree C       worktree D
 Daytona          Daytona          Daytona          Daytona
      |                |                |                |
      +--------+-------+-------+--------+
               v
           merge coordinator  (three-way merge + conflict resolution)
               |
               v
           Reviewer (GPT-5.4)
               |
               v
           Tester  (Gemini 2.5 Pro)  -> passes? -> open PR
                                     -> fails?  -> route back to coder
```

## 技術堆疊 (Stack)

- 編排：LangGraph，搭配共享狀態 + 每個 agent 的子圖
- 訊息傳遞：A2A 協定 (Google 2025)，用於具型別的 agent 間訊息
- 模型：Opus 4.7 (architect)、Sonnet 4.7 (coders)、GPT-5.4 (reviewer)、Gemini 2.5 Pro (tester)
- Worktree 隔離：每個 coder 一個 `git worktree add` + Daytona sandbox
- 合併協調者：自訂的三方合併 + LLM 居中處理的衝突解決
- 評測：SWE-bench Pro (50 個 issue)、SWE-AF 情境、單元測試用 HumanEval++
- 可觀測性：Langfuse，帶角色標記的 span 與每個 agent 的 token 計帳
- 部署：K8s，每個角色一個獨立 Deployment，依積壓量做 HPA

```figure
ce-team-handoff
```

## 動手做 (Build It)

1. **任務看板。** 以檔案為基礎的 JSONL，包含具型別訊息：`plan_request`、`subtask`、`diff_ready`、`review_needed`、`test_needed`、`approved`、`rejected`、`replan_needed`。Agent 訂閱標籤。

2. **Architect。** 讀取 GitHub issue，以要求明確子任務介面 (涉及的檔案、公開函式、測試影響) 的計畫範本執行 Opus 4.7。輸出一個 `plan_request`，內含子任務的 DAG。

3. **Coders。** N 個平行 worker，各自從看板認領一個子任務。每個都產生全新的 `git worktree add` 分支加上 Daytona sandbox。實作子任務。輸出 `diff_ready`，附上 patch + 測試差異。

4. **合併協調者。** 所有 coder 完成後，把 N 個分支三方合併到 staging 分支。只有在檔案層級有重疊時，才使用 LLM 居中處理衝突解決。

5. **Reviewer。** GPT-5.4 讀取合併後的 diff。不能核准自己撰寫的 diff。輸出 `approved` (無動作) 或 `review_feedback`，附上具體修改要求，路由回相關的 coder。

6. **Tester。** Gemini 2.5 Pro 在乾淨的 sandbox 中執行測試套件。擷取產出物。輸出 `test_passed` 或 `test_failed`，附上 stacktrace。失敗的測試會回到擁有該失敗子任務的 coder。

7. **交接計帳。** 每則跨越角色邊界的訊息，都在 Langfuse 中取得一個 span，帶有 payload 大小與所用模型。計算每個子任務的 token 放大 (coder_tokens + reviewer_tokens + tester_tokens + architect_share / coder_tokens)。

8. **評測。** 在 50 個 SWE-bench Pro issue 上執行。與單一 agent 基準線 (單一 worktree 中的一個 Sonnet 4.7) 比較 pass@1 與每解決一個 issue 的美元成本。

9. **事後檢討。** 對每個失敗的 issue，找出壞掉的交接 (計畫太模糊、合併衝突、reviewer 誤核准、tester 不穩定)。產出交接失敗直方圖。

## 實際使用 (Use It)

```
$ team run --issue https://github.com/acme/widget/issues/842
[architect] plan: 4 subtasks (parser, cache, api, migration)
[board]     dispatched to 4 coders in parallel worktrees
[coder-A]   subtask parser  -> 42 lines, tests pass locally
[coder-B]   subtask cache   -> 88 lines, tests pass locally
[coder-C]   subtask api     -> 31 lines, tests pass locally
[coder-D]   subtask migration -> 19 lines, tests pass locally
[merge]     3-way merge: 0 conflicts
[reviewer]  comments on cache (thread pool sizing); routed to coder-B
[coder-B]   revision: 92 lines; submits
[reviewer]  approved
[tester]    all 412 tests pass
[pr]        opened #3382   4 coders, 1 revision, $4.90, 18m
```

## 交付 (Ship It)

`outputs/skill-multi-agent-team.md` 是交付成果。給定 issue URL 與平行度，團隊會產出可合併的 PR，並附上每個角色的 token 計帳。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | SWE-bench Pro pass@1 | 配對的 50 個 issue 子集，pass@1 |
| 20 | 平行加速 | 牆鐘時間與單一 agent 基準線比較 |
| 20 | Review 品質 | 在注入錯誤探測上的誤核准率 |
| 20 | Token 效率 | 每解決一個 issue 的總 token 數，與單一 agent 比較 |
| 15 | 協調工程 | 合併衝突解決、交接失敗直方圖 |
| **100** | | |

## 練習 (Exercises)

1. 在執行中途對 diff 注入一個明顯的 bug (在主體之前多一個 `return None`)。衡量 reviewer 的誤核准率。調整 reviewer prompt，直到誤核准率低於 5%。

2. 縮減為兩個 coder (architect + 兩個 coder + reviewer + tester，每個 coder 依序執行兩個子任務)。比較牆鐘時間與通過率。

3. 把合併協調者換成單一寫入者限制 (子任務動到互不相交的檔案集合)。衡量 architect 的規劃負擔。

4. 把 reviewer 從 GPT-5.4 換成 Claude Opus 4.7。衡量誤核准率與 token 成本的差異。

5. 加入第五個角色：documenter (Haiku 4.5)。review 後，它產出一則 changelog 條目。衡量文件品質是否值得額外的 token 花費。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Parallel worktree | 「隔離的分支」 | `git worktree add` 為每個 coder 產生一個全新的工作樹 |
| Task board | 「共享訊息匯流排」 | agent 訂閱的具型別訊息的檔案或 Redis 儲存 |
| Handoff | 「角色邊界」 | 任何從一個角色 context 傳到另一個角色 context 的訊息 |
| Token amplification | 「多 agent 額外負擔」 | 同一任務下，各角色總 token 數 / 單一 agent token 數 |
| A2A protocol | 「Agent 對 agent」 | Google 2025 年針對具型別 agent 間訊息的規格 |
| Merge coordinator | 「整合者」 | 執行三方合併並居中處理衝突的元件 |
| False approval | 「Reviewer 幻覺」 | Reviewer 核准了含已知 bug 的 diff |

## 延伸閱讀 (Further Reading)

- [SWE-AF factory architecture](https://github.com/Agent-Field/SWE-AF) — 2026 年多 agent factory 的參考
- [MetaGPT](https://github.com/FoundationAgents/MetaGPT) — 角色式多 agent 框架
- [AutoGen v0.4](https://github.com/microsoft/autogen) — Microsoft 的具型別 actor 框架
- [Cognition AI (Devin)](https://cognition.ai) — 參考產品
- [Factory Droids](https://www.factory.ai) — 另一個參考產品
- [Google A2A protocol](https://a2a-protocol.org/latest/) — agent 間訊息規格
- [git worktree documentation](https://git-scm.com/docs/git-worktree) — 隔離的基礎
- [SWE-bench Pro](https://www.swebench.com) — 評測目標
