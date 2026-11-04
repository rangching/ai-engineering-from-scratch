# Skill Library 與終身學習 (Voyager) (Skill Libraries and Lifelong Learning (Voyager))

> Voyager（Wang et al., TMLR 2024）把可執行的程式碼視為 skill。Skill 有名稱、可取回、可組合，並由環境回饋來精進。這是 Claude Agent SDK skill、skillkit，以及 2026 年 skill-library 模式的參考架構。

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 07 (MemGPT), Phase 14 · 08 (Letta Blocks)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 說出 Voyager 的三個元件：automatic curriculum、skill library、iterative prompting，以及各自的角色。
- 解釋為什麼 Voyager 讓 action space 是程式碼，而不是原始指令。
- 用 stdlib 實作 skill library，包含註冊、取回、組合，以及由失敗驅動的精進。
- 把 Voyager 的模式對應到 2026 年的 Claude Agent SDK skill 與 skillkit 生態系。

## 問題 (The Problem)

每個 session 都從零重建每項能力的 agent，會有三件事做錯：

1. **浪費 token。** 每個任務都重新引出相同的推理。
2. **進度遺失。** 在 session A 學到的修正，無法轉移到 session B。
3. **在長時程組合上失敗。** 複雜任務需要能力階層；一次性的 prompt 無法表達。

Voyager 的答案：把每個可重複使用的能力視為存在 library 裡、有名稱的一段程式碼，可依相似度取回、與其他 skill 組合，並由執行回饋精進。

## 核心概念 (The Concept)

### 三個元件 (Three components)

Voyager（arXiv:2305.16291）把 agent 圍繞以下結構組織：

1. **Automatic curriculum。** 由好奇心驅動的提案者，依 agent 目前的 skill 集合與環境狀態挑選下一個任務。探索是由下而上的。
2. **Skill library。** 每個 skill 都是可執行的程式碼。任務成功時加入新 skill。Skill 依查詢與描述的相似度取回。
3. **Iterative prompting 機制。** 失敗時，agent 收到執行錯誤、環境回饋和自我驗證輸出，然後精進 skill。

Minecraft 評估（Wang et al., 2024）：相較於 baseline，獨特物品多 3.3 倍、石製工具快 8.5 倍、鐵製工具快 6.4 倍、地圖行走距離長 2.3 倍。這些數字是 Minecraft 特有的，但模式可以轉移。

### Action space = 程式碼 (Action space = code)

大多數 agent 輸出原始指令。Voyager 輸出 JavaScript 函式。一個 skill 是：

```
async function craftIronPickaxe(bot) {
  await mineIron(bot, 3);
  await mineStick(bot, 2);
  await placeCraftingTable(bot);
  await craft(bot, 'iron_pickaxe');
}
```

由子 skill 組合而成。以描述和 embedding 為 key 儲存。以程式（而不是 prompt）的形式取回。

這就是 2026 年的 Claude Agent SDK skill：一段有名稱、可取回的程式碼，加上 agent 按需載入的指令。

### Skill 取回 (Skill retrieval)

新任務「做一把鑽石鎬」。Agent：

1. 對任務描述做 embedding。
2. 在 skill library 中查詢最相似的 top-k skill。
3. 取回 `craftIronPickaxe`、`mineDiamond`、`placeCraftingTable` 等。
4. 用取回的原語加上新邏輯，組合出新 skill。

這就是 MCP resource（Phase 13）和 Agent SDK skill 實作的模式：針對知識/程式碼的範圍做 retrieval，並限縮在目前任務。

### 迭代精進 (Iterative refinement)

Voyager 的回饋 loop：

1. Agent 寫一個 skill。
2. Skill 在環境中執行。
3. 回傳三種訊號之一：`success`、`error`（附 stack trace）、`self-verification failure`。
4. Agent 以該訊號為 context 重寫 skill。
5. 重複直到成功或達到最大回合數。

這是 Self-Refine（Lesson 05）套用在程式碼生成上，並以環境為依據做驗證。CRITIC（Lesson 05）是同樣的模式，只是以外部 tool 作為 verifier。

### Curriculum 與探索 (Curriculum and exploration)

Voyager 的 curriculum 模組會根據 agent 擁有什麼、還沒做過什麼，提出像「在湖邊蓋一個避難所」這樣的任務。提案者用環境狀態 + skill 清單，挑出一個剛好略高於目前能力的任務，也就是探索的甜蜜點。

對正式環境的 agent 來說，這會轉成一個「缺了什麼」的運算：給定目前的 skill library 和一個領域，我們還沒涵蓋哪些 skill？團隊通常以手動的 curriculum 審查來實作。

### 這個模式哪裡會出錯 (Where this pattern goes wrong)

- **Skill library 腐化 (Skill library rot)。** 同一個 skill 用略微不同的描述加了 10 次。寫入時做去重；retrieval 只回傳一個。
- **組合 skill 漂移 (Composed-skill drift)。** 父 skill 依賴一個被精進過的子 skill。要為 skill 做版本管理；釘在 v1 的父 skill 不會神奇地升到 v3。
- **Retrieval 品質。** 針對 skill 描述的 vector retrieval，在 library 超過幾百個之後會退化。要補上 tag 過濾和硬性限制（「只取 `category=tooling` 的 skill」）。

```figure
voyager-skills
```

## 動手做 (Build It)

`code/main.py` 實作一個 stdlib skill library：

- `Skill`：name、description、code（字串）、version、tags、dependencies。
- `SkillLibrary`：register、search（token 重疊）、compose（依賴的拓撲排序），以及 refine（更新時版本號加一）。
- 一個腳本化的 agent：註冊三個原始 skill、組合出第四個、遇到失敗，然後精進。

執行：

```
python3 code/main.py
```

Trace 顯示 library 寫入、取回、組合、一次失敗的執行，以及 v2 精進，也就是 Voyager 完整的 loop。

## 實際使用 (Use It)

- **Claude Agent SDK skills**（Anthropic）：2026 年的參考實作：每個 skill 有 description、程式碼和指令；在 agent session 中按需載入。
- **skillkit**（npm: skillkit）：跨 agent 的 skill 管理，支援 32 個以上的 AI coding agent。
- **自訂 skill library**：特定領域（資料 agent 的 SQL skill、infra agent 的 Terraform skill）。Voyager 模式可以縮小規模使用。
- **OpenAI Agents SDK `tools`**：在低階端；每個 tool 就是輕量的 skill。

## 交付 (Ship It)

`outputs/skill-skill-library.md` 為任何目標 runtime 產生 Voyager 形狀的 skill library，內含註冊、取回、版本管理與精進。

## 練習 (Exercises)

1. 在 `compose()` 加入依賴循環偵測器。當 skill A 依賴 B、B 又依賴 A 時會發生什麼？該報錯還是警告？
2. 實作每個 skill 的版本固定。當父 skill 組合子 skill `crafting@1` 時，對 `crafting@2` 的精進不得悄悄升級父 skill。
3. 把 token 重疊的 retrieval 換成 sentence-transformers embedding（或 stdlib 的 BM25 實作）。在 50 個 skill 的 toy library 上量測 retrieval@5。
4. 加一個「curriculum」agent：給定目前的 library 和一個領域描述，提出 5 個缺少的 skill。每週呼叫一次。
5. 讀 Anthropic 的 Claude Agent SDK skill 文件。把 toy library 移植到 SDK 的 skill schema。可發現性（discoverability）有什麼改變？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Skill | 「可重複使用的能力」 | 有名稱的一段程式碼 + 描述，可依相似度取回 |
| Skill library | 「Agent 的 how-to 記憶」 | 持久的 skill 儲存，可搜尋、可組合 |
| Curriculum | 「任務提案者」 | 由目前能力缺口驅動、由下而上的目標產生器 |
| Composition | 「Skill DAG」 | Skill 呼叫 skill；執行時做拓撲排序 |
| Iterative refinement | 「自我修正 loop」 | 環境回饋 + 錯誤 + 自我驗證折回成下一個版本 |
| Action-space-as-code | 「程式化的 action」 | 針對時間上延伸的行為，輸出函式而不是原始指令 |
| Dedup on write | 「Skill 合併」 | 近乎重複的描述合併成一個標準 skill |

## 延伸閱讀 (Further Reading)

- [Wang et al., Voyager (arXiv:2305.16291)](https://arxiv.org/abs/2305.16291)：最早的 skill-library 論文
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview)：skill 在 2026 年的產品化
- [Anthropic, Building agents with the Claude Agent SDK](https://www.anthropic.com/engineering/building-agents-with-the-claude-agent-sdk)：skill 與 subagent 的實務
- [Madaan et al., Self-Refine (arXiv:2303.17651)](https://arxiv.org/abs/2303.17651)：Voyager 底下的精進 loop
