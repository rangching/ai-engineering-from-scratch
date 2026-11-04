# 為什麼要 Multi-Agent？(Why Multi-Agent?)

> 一個 agent 會撞牆。聰明的做法不是做更大的 agent，而是用更多的 agent。

**Type:** Learn
**Languages:** TypeScript
**Prerequisites:** Phase 14 (Agent Engineering)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 辨識 single-agent 的天花板（context 溢位、專業混雜、循序瓶頸），並說明什麼時候拆成多個 agent 才是對的選擇
- 比較 orchestration pattern（pipeline、parallel fan-out、supervisor、hierarchical），並依任務結構選出合適的一種
- 設計一個 multi-agent 系統，包含清楚的角色邊界、共享狀態和溝通契約
- 分析 multi-agent 複雜度（延遲、成本、除錯難度）與 single-agent 簡單性之間的取捨

## 問題 (The Problem)

你在 Phase 14 做了一個 single agent。它能用。它會讀檔案、執行指令、呼叫 API，還能針對結果推理。然後你把它丟進一個真實的 codebase：200 個檔案、三種語言、依賴基礎設施的測試，還要求在寫程式前先研究外部 API。

這個 agent 卡住了。不是因為 LLM 笨，而是任務超出一個 agent loop 能處理的範圍。Context window 被檔案內容塞滿。Agent 忘了 40 次 tool call 之前讀過什麼。它想同時當研究員、工程師和 reviewer，結果三件事都做不好。

這就是 single-agent 的天花板。只要任務需要下面任何一項，你每次都會撞到：

- **超過一個 window 放得下的 context** - 讀 50 個檔案就會超過 200k token
- **不同階段需要不同專業** - 研究和產生程式碼需要不同的 prompting
- **可以平行進行的工作** - 三個檔案可以同時讀，為什麼要一個一個讀？

## 核心概念 (The Concept)

### Single-Agent 的天花板 (The Single-Agent Ceiling)

Single agent 就是一個 loop、一個 context window、一個 system prompt。想像一下：

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

有三件事會壞掉：

1. **Context 飽和 (Context saturation)** - tool 結果越堆越多。到第 30 輪，agent 已經消耗了 150k token 的檔案內容、指令輸出和先前的推理。第 5 輪的關鍵細節就丟了。

2. **角色混淆 (Role confusion)** - system prompt 寫著「你是研究員、工程師、reviewer 兼測試員」，做出來的 agent 研究做一半、寫程式寫一半，而且永遠 review 不完。

3. **循序瓶頸 (Sequential bottleneck)** - agent 先讀檔案 A，再讀 B，再讀 C。三次串行的 LLM 呼叫，三次串行的 tool 執行。沒有任何平行。

### Multi-Agent 的解法 (The Multi-Agent Solution)

把工作拆開。每個 agent 只做一件事，有自己的 context window，以及針對該工作調整過的 system prompt：

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

每個 agent 都有：
- 聚焦的 system prompt（「你是 code reviewer。你唯一的工作是找 bug。」）
- 自己的 context window（不會被其他 agent 的工作汙染）
- 清楚的輸入／輸出契約（收到研究筆記，輸出程式碼）

### 實際採用這種做法的系統 (Real Systems That Do This)

**Claude Code subagents** - Claude Code 用 `Task` 產生 subagent 時，會建立一個有範圍限定任務的子 agent。父 agent 的 context 保持乾淨。子 agent 專注做事，然後回傳摘要。

**Devin** - 執行一個 planner agent、一個 coder agent 和一個 browser agent。Planner 把工作拆成步驟。Coder 寫程式。Browser 查文件。各自有獨立的 context。

**Multi-agent 程式開發團隊 (SWE-bench)** - SWE-bench 上表現最好的系統，會用一個讀 codebase 的 researcher、一個設計修正方式的 planner，以及一個實作的 coder。Single-agent 系統的分數比較低。

**ChatGPT Deep Research** - 平行產生多個搜尋 agent，各自探索不同角度，最後再綜合結果。

### 光譜 (The Spectrum)

Multi-agent 不是二選一，而是一條光譜：

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - 一個 loop、一個 prompt。適合簡單任務。

**Subagents** - 父 agent 為聚焦的子任務產生子 agent。父 agent 維持計畫，子 agent 回報結果。Claude Code 就是這樣做的。

**Pipeline** - agent 依序執行。Agent A 的輸出變成 Agent B 的輸入。適合分階段的工作流程：研究 -> 寫程式 -> review -> 測試。

**Team** - agent 透過共享的 message bus 平行執行。每個有自己的角色，由一個 orchestrator 協調。適合同時需要不同技能的情況。

**Swarm** - 許多相同或近乎相同的 agent 共享狀態。沒有固定的 orchestrator。Agent 從 queue 取工作。適合高吞吐量的平行任務。

### 四種 Multi-Agent Pattern (The Four Multi-Agent Patterns)

#### Pattern 1：Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

每個 agent 轉換資料，再往前傳。容易推理。某一階段失敗，後面都會被擋住。

#### Pattern 2：Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

把工作分給平行的 agent，再合併結果。適合可以拆成獨立子任務的工作。

#### Pattern 3：Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

一個聰明的 orchestrator 決定要做什麼、委派給 worker，再綜合結果。Orchestrator 本身也是 agent，擁有產生 worker 的 tool。

#### Pattern 4：Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

沒有中央 orchestrator。Agent 之間點對點溝通，決策從互動中浮現。比較難除錯，但可以擴展到很多 agent。

### 什麼時候不該用 Multi-Agent (When NOT to Use Multi-Agent)

Multi-agent 會增加複雜度。Agent 之間的每一則訊息都可能出錯。除錯會從「讀一段對話」變成「追蹤五個 agent 之間的訊息」。

**這些情況請維持 single-agent：**
- 任務放得進一個 context window（工作資料在 ~100k token 以內）
- 不同階段不需要不同的 system prompt
- 循序執行已經夠快
- 任務夠簡單，拆開的額外負擔比好處還大

**複雜度的成本：**
- 每個 agent 邊界都是一次有損壓縮：agent A 的完整 context 會被濃縮成給 agent B 的一則訊息
- 協調邏輯（誰做什麼、何時做、什麼順序）本身就是 bug 的來源
- 延遲增加：N 個 agent 至少是 N 次串行的 LLM 呼叫，來回溝通的話更多
- 成本倍增：每個 agent 各自消耗 token

經驗法則：如果任務的 tool call 少於 20 次，而且放得進 100k token，就維持 single-agent。

```figure
swarm-messages
```

## 動手做 (Build It)

### 步驟 1：負擔過重的 Single Agent (The Overloaded Single Agent)

這是一個想包辦一切的 single agent。它只有一個巨大的 system prompt 和一個 context window，裡面同時放著研究、程式碼和 review：

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

這種做法的問題：
- Context window 每個階段都在變大。到了 review 步驟，裡面已經有研究筆記、程式碼和先前的推理。
- System prompt 太通用，無法針對每個階段調整。
- 沒有任何東西平行執行。

### 步驟 2：專家 Agent (Specialist Agents)

現在把它拆開。每個 agent 只做一件事：

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

每個專家都有聚焦的 prompt，也各自拿到乾淨的 context window，裡面只有它需要的輸入。

### 步驟 3：透過訊息協調 (Coordinate Through Messages)

用明確的 message passing 把專家串起來：

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

每個 agent 只會收到寄給它的訊息。沒有 context 汙染。Researcher 讀文件用掉的 50k token，永遠不會進到 reviewer 的 context。

### 步驟 4：比較 (Compare)

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Multi-agent 版本用掉的 token 總量比較多（三個 agent、三次獨立的 LLM 呼叫），但每個 agent 的 context 都保持乾淨。因為 system prompt 是專門化的，每個階段的品質也會提升。

## 實際使用 (Use It)

這堂課會產出一個可重複使用的 prompt，用來判斷何時該改用 multi-agent。請見 `outputs/prompt-multi-agent-decision.md`。

## 練習 (Exercises)

1. 加入第四個專家：「tester」agent，它從 coder 接收程式碼、從 reviewer 接收 review 意見，然後撰寫測試
2. 修改 pipeline，讓 reviewer 可以把意見送回 coder 做修訂迴圈（最多 2 輪）
3. 把循序的 pipeline 改成 fan-out：讓 researcher 和一個「requirements analyzer」agent 平行執行，合併它們的輸出後再交給 coder

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|----------------------|
| Swarm | 「AI agent 的蜂巢意識」 | 一組共享狀態、沒有固定領導者的 peer agent。行為從局部互動中浮現。 |
| Orchestrator | 「老闆 agent」 | tool 包含產生與管理其他 agent 的 agent。它負責規劃與委派，但不一定自己做實際的工作。 |
| Coordinator | 「交通警察」 | 非 agent 的元件（通常只是程式，不是 LLM），依規則在 agent 之間轉送訊息。 |
| Consensus | 「agent 們達成共識」 | 多個 agent 必須先取得一致才能繼續的協定。用在需要解決輸出衝突的時候。 |
| Emergent behavior | 「agent 自己想通了」 | 從 agent 互動中產生、但沒有被明確寫進程式的系統層級模式。可能有用，也可能有害。 |
| Fan-out / fan-in | 「給 agent 用的 map-reduce」 | 把任務分給平行的 agent（fan-out），再合併它們的結果（fan-in）。 |
| Message passing | 「agent 彼此對話」 | agent 之間的溝通機制：一個 agent 把結構化資料傳給另一個，取代共用的 context window。 |

## 延伸閱讀 (Further Reading)

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - multi-agent pattern 的綜述
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft 的 multi-agent 對話框架
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - Claude Code 如何用 Task 委派工作
- [CrewAI documentation](https://docs.crewai.com/) - 以角色為基礎的 multi-agent 框架
