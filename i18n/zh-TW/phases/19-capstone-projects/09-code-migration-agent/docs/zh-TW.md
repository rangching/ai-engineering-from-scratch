# Capstone 09 — 程式碼遷移 Agent (Code Migration Agent，Repo 層級的語言 / Runtime 升級)

> Amazon 的 MigrationBench (Java 8 到 17) 與 Google 的 App Engine Py2-to-Py3 遷移工具，訂下了 2026 年的標準。Moderne 的 OpenRewrite 大規模執行確定性的 AST 改寫。Grit 以 codemod 風格的 DSL 處理同樣的問題。正式環境的模式兩者並用：以確定性基底負責安全改寫，以 agent 層處理模糊案例，每個分支有 sandbox 做建置，並在開 PR 前讓測試套件轉綠。這個 capstone 要遷移 50 個真實 repo，並發佈通過率與失敗分類。

**Type:** Capstone
**Languages:** Python (agent), Java / Python (targets), TypeScript (dashboard)
**Prerequisites:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 13 (tools), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure)
**Phases exercised:** P5 · P7 · P11 · P13 · P14 · P15 · P17
**Time:** 30 小時

## 問題 (Problem)

大規模程式碼遷移，是 2026 年 coding agent 最乾淨的正式應用之一。真實基準很明確 (遷移後測試套件有通過嗎？)，回報是實在的 (Java 8 整個機群的遷移，是以人力編制計算的專案)，而且 benchmark 是公開的 (MigrationBench 50 個 repo 的子集)。Moderne 的 OpenRewrite 處理確定性的部分。Agent 層處理 OpenRewrite recipe 做不到的一切：模糊的改寫、建置系統漂移、長尾語法、傳遞依賴的破壞。

你要打造一個 agent，輸入 Java 8 repo (或 Python 2 repo)，產出 CI 全綠、已遷移的分支。你要衡量通過率、測試覆蓋率的保留、每個 repo 的成本，並建立失敗分類。與僅用確定性方法的基準線並排比較，就能看出 agent 的價值真正落在哪裡。

## 核心概念 (Concept)

管線有兩層。**確定性基底** (Java 用 OpenRewrite，Python 用 libcst) 安全地執行大部分機械式改寫：imports、方法簽章、null-safety 修改、try-with-resources、已棄用 API 的替換。它很快，且產出可稽核的 diff。**Agent 層** (在 Claude Opus 4.7 與 GPT-5.4-Codex 之上，使用 OpenAI Agents SDK 或 LangGraph) 處理 recipe 做不到的案例：建置檔升級 (Maven/Gradle/pyproject)、傳遞依賴衝突、測試不穩定、自訂 annotation。

每個 repo 都有預先安裝目標 runtime 的 Daytona sandbox。Agent 迭代：執行建置、分類失敗、套用修正、重跑。硬性限制：每個 repo 30 分鐘、$8、20 個 agent 回合。如果所有測試通過，且覆蓋率下降不超過 2 個百分點，該分支就開 PR。否則，這個 repo 會連同證據歸入某個失敗類別。

失敗分類就是交付成果。在 50 個 repo 之中，什麼壞了？傳遞依賴？自訂 annotation？建置工具版本？與遷移無關的測試不穩定？每個類別都有計數和一個範例 diff。未來的 recipe 作者可以針對前三名下手。

## 架構 (Architecture)

```
target repo
      |
      v
OpenRewrite / libcst deterministic recipes
   (safe, fast, auditable, ~70-80% of fixes)
      |
      v
Daytona sandbox per branch
      |
      v
agent loop (Claude Opus 4.7 / GPT-5.4-Codex):
   - run build -> capture failures
   - classify failures (build, test, lint)
   - apply fix (patch or retry recipe)
   - rerun
   - budget: 30 min, $8, 20 turns
      |
      v
test + coverage delta gate
      |
      v (passed)
open PR
      |
      v (failed)
file under failure class + attach repro
```

## 技術堆疊 (Stack)

- 確定性基底：OpenRewrite (Java) 或 libcst (Python)
- Agent：在 Claude Opus 4.7 + GPT-5.4-Codex 之上使用 OpenAI Agents SDK 或 LangGraph
- Sandbox：每個分支一個 Daytona devcontainer，預先安裝目標 runtime (Java 17 / Python 3.12)
- 建置系統：Maven、Gradle、uv (Python)
- Benchmark：Amazon MigrationBench 50 個 repo 的子集 (Java 8 到 17)、Google App Engine Py2-to-Py3 repos
- 測試 harness：平行執行器，覆蓋率用 Jacoco (Java) 或 coverage.py (Python)
- 可觀測性：Langfuse + 每個 repo 一份含每個 diff chunk 的 trace bundle
- Dashboard：失敗分類 dashboard，含各類別計數與範例 diff

```figure
ce-migration-funnel
```

## 動手做 (Build It)

1. **Recipe 階段。** 先執行 OpenRewrite (Java) 或 libcst (Python) 的 recipe。攔下 70-80% 屬於機械式的遷移。以「recipe」commit 提交。

2. **建置試跑。** Daytona sandbox：安裝目標 runtime，執行建置。如果是綠的，直接跳到測試。如果是紅的，交給 agent。

3. **Agent 迴圈。** 具備這些工具的 LangGraph：`run_build`、`read_file`、`edit_file`、`run_test`、`git_diff`。Agent 將失敗分類 (dep、syntax、test、build-tool) 並套用針對性的修正。重跑。

4. **預算上限。** 每個 repo 30 分鐘牆鐘時間、$8 成本、20 個 agent 回合。任何一項超出就中止，連同目前的 diff 歸入 "budget_exhausted"。

5. **測試 + 覆蓋率關卡。** 建置轉綠後，執行測試套件。與基底 repo 比較覆蓋率。如果覆蓋率下降超過 2 個百分點，就歸入 "coverage_regression"。

6. **開 PR。** 成功時，推送分支，開 PR，附上 diff，以及哪些 recipe 被套用、哪些 commit 由 agent 撰寫的摘要。

7. **失敗分類。** 對每個失敗的 repo，標記一個類別：`dep_upgrade_required`、`build_tool_drift`、`custom_annotation`、`test_flake`、`syntax_edge_case`、`budget_exhausted`。建立 dashboard。

8. **50 個 repo 的執行。** 在 MigrationBench 子集上執行。回報各類別的通過率、每個 repo 的成本、覆蓋率保留，以及與僅用確定性方法基準線的比較。

## 實際使用 (Use It)

```
$ migrate legacy-java-service --target java17
[recipe]   27 rewrites applied (JUnit 4->5, HashMap initializer, try-with-resources)
[build]    FAIL: cannot find symbol sun.misc.BASE64Encoder
[agent]    turn 1 classify: removed_jdk_api
[agent]    turn 2 apply: sun.misc.BASE64Encoder -> java.util.Base64
[build]    OK
[tests]    412/412 passing; coverage 84.1% -> 84.3%
[pr]       opened #1841  cost=$3.20  turns=4
```

## 交付 (Ship It)

`outputs/skill-migration-agent.md` 是交付成果。給定一個 repo，它先執行確定性 recipe，再跑 agent 迴圈，產出全綠的遷移分支，或把 repo 歸入某個分類類別。

| 權重 | 評分項目 | 衡量方式 |
|:-:|---|---|
| 25 | MigrationBench 通過率 | 50 個 repo 子集的 pass@1 |
| 20 | 測試覆蓋率保留 | 相對基底的平均覆蓋率差異 |
| 20 | 每個遷移 repo 的成本 | 通過執行的 $/repo |
| 20 | Agent / 確定性工具整合 | OpenRewrite 處理的修正與 agent 撰寫的修正比例 |
| 15 | 失敗分析報告 | 分類的完整度與範例 |
| **100** | | |

## 練習 (Exercises)

1. 只用 OpenRewrite (不用 agent) 執行遷移管線。與完整管線比較通過率，找出僅靠 agent 才有差別的案例。

2. 實作「lint-clean」檢查：遷移後執行風格 linter (Java 用 spotless，Python 用 ruff)。如果出現新的 lint 錯誤，就讓 PR 失敗。衡量「覆蓋率保留但風格退步」的比率。

3. 加入「最小 diff」最佳化器：agent 的分支通過測試後，用第二輪修剪不必要的變更。回報 diff 大小的縮減。

4. 擴充到第三種遷移：Node 18 到 Node 22。重用 sandbox 包裝；把 recipe 層換成自訂 codemod。

5. 衡量首次建置轉綠時間 (TTFGB) 作為 UX 指標。目標：p50 低於 10 分鐘。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Deterministic substrate | 「Recipe 引擎」 | OpenRewrite / libcst：附有安全保證的宣告式 AST 改寫 |
| Codemod | 「修改程式碼的程式」 | 以機械方式改變原始碼的改寫規則 |
| Build drift | 「工具版本偏移」 | 主要版本之間 Maven / Gradle / uv 行為的細微變化 |
| Failure class | 「分類桶」 | repo 沒有遷移成功的標記原因：dep、syntax、test、build-tool、budget |
| Coverage delta | 「覆蓋率保留」 | 從基底到遷移分支的測試覆蓋率 % 變化 |
| Agent turn | 「工具呼叫回合」 | agent 迴圈中的一次 plan -> act -> observe 週期 |
| Budget exhaustion | 「撞到上限」 | repo 用完 30 分鐘 / $8 / 20 回合的限制，仍未通過 |

## 延伸閱讀 (Further Reading)

- [Amazon MigrationBench](https://aws.amazon.com/blogs/devops/amazon-introduces-two-benchmark-datasets-for-evaluating-ai-agents-ability-on-code-migration/) — 2026 年的標準 benchmark
- [Moderne.io OpenRewrite platform](https://www.moderne.io) — 確定性基底的參考
- [OpenRewrite documentation](https://docs.openrewrite.org) — recipe 撰寫
- [Grit.io](https://www.grit.io) — 另一種 codemod DSL
- [OpenAI sandboxed migration cookbook](https://developers.openai.com/cookbook/examples/agents_sdk/sandboxed-code-migration/sandboxed_code_migration_agent) — Agents SDK 參考
- [Google App Engine Py2 to Py3 migrator](https://cloud.google.com/appengine) — 另一個遷移 benchmark
- [libcst](https://github.com/Instagram/LibCST) — Python 確定性基底
- [Daytona sandboxes](https://daytona.io) — 參考的每分支 sandbox
