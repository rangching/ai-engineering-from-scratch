# Hierarchical 架構與它的失敗模式 (Hierarchical Architecture and Its Failure Mode)

> Hierarchical 就是巢狀的 supervisor。Manager agent 在 sub-manager 之上，sub-manager 在 worker 之上。CrewAI 的 `Process.hierarchical` 是教科書版本：由 `manager_llm` 動態委派任務並驗證輸出。LangGraph 的對應寫法是 `create_supervisor(create_supervisor(...))`。當任務真的就是一張組織圖時，這是很自然的 pattern。但它也是最容易崩潰成管理迴圈的 pattern：manager agent 分配工作做得差、誤解子輸出，或無法達成 consensus。Sequential 往往比它更好。

**Type:** Learn + Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 16 · 05 (Supervisor Pattern)
**Time:** ~60 分鐘

## 問題 (Problem)

Supervisor pattern 一旦搞懂，很自然的下一步就是：「如果 worker 本身也是 supervisor 呢？」團隊有子團隊；公司有部門下的部門。Hierarchical 架構就是在映射這件事。

問題在於：LLM manager 和人類 manager 不一樣。人類 manager 對自己的部屬知道什麼，有穩定的先驗。LLM manager 每一輪都根據 context 裡有的東西重新推理整個組織。Context 只要有一點點漂移，整棵樹就會錯配工作。

## 核心概念 (Concept)

### 形狀 (The shape)

```
                 Manager
                 ┌─────┐
                 └──┬──┘
           ┌────────┴────────┐
           ▼                 ▼
       Sub-Mgr A         Sub-Mgr B
       ┌─────┐           ┌─────┐
       └──┬──┘           └──┬──┘
         ┌┴──┬──┐          ┌┴──┐
         ▼   ▼  ▼          ▼   ▼
       W1  W2  W3         W4  W5
```

每個內部節點都負責規劃、委派和綜合。只有葉節點做實際的工作。

### 它擅長的地方 (Where it shines)

- **清楚的組織對應。** 如果真正的任務是依部門劃分的（「法務審文件、財務審文件、工程審文件，然後為主管做摘要」），階層就是明確的。
- **局部摘要。** 每個 sub-manager 在上層 manager 看到之前，先綜合自己團隊的輸出。上層 manager 看到的是三份 sub-manager 的摘要，而不是十五份 worker 的輸出。

### 它壞掉的地方 (Where it breaks)

2026 年的事後檢討不斷發現三種失敗模式：

1. **任務分配錯誤 (Task assignment error)。** Manager 讀了目標、幻想出一個拆解方式，然後委派給錯誤的 sub-manager。因為 sub-manager 很聽話地做被交付的事，錯誤只會在最上層綜合時才浮現，離人類能攔截的地方已經隔了一層。
2. **輸出誤讀 (Output misinterpretation)。** Sub-manager 回傳「無法驗證主張 X」。上層 manager 摘要成「主張 X 未獲證實」。意義在每一層都會漂移。
3. **Consensus 迴圈 (Consensus loops)。** 兩個 sub-manager 意見不同；上層 manager 要它們調和；它們往下重新委派；worker 重新執行；sub-manager 回傳稍有不同的答案；迴圈。CrewAI 的 `Process.hierarchical` 用步數上限來防範，但這個上限本身現在就成了超參數。

### 決定性的問題 (The deciding question)

Sequential（線性 pipeline）還是 hierarchical：你的任務真的有獨立的子團隊，還是只是假裝成樹的一條線性流程？如果是後者，用 sequential。如果是前者，用 hierarchical，但要預留明確的調和規則預算。

### Role 框架的實作 (Role-framework implementation)

CrewAI 的 `Process.hierarchical` 在專家 crew 之上接一個 manager LLM。這個 manager：

- 接收頂層任務，
- 把子任務分配給各個 crew，
- 評估 crew 的輸出，
- 決定要接受、重新委派，還是迭代。

文件：https://docs.crewai.com/en/introduction（在 Core Concepts 下找「Hierarchical Process」）。

### Graph 框架的實作 (Graph-framework implementation)

LangGraph 使用巢狀的 `create_supervisor` 呼叫。內層 supervisor 有自己的圖；外層 supervisor 把內層的圖視為不透明的 node。這在除錯上比 CrewAI 乾淨（你可以分別單步執行每張圖），但比較難表達動態重塑這棵樹。

參考：https://reference.langchain.com/python/langgraph-supervisor。

```figure
swarm-hierarchy-token
```

## 動手做 (Build It)

`code/main.py` 執行一個 3 層的階層：

- 頂層 manager：把任務拆成「engineering」與「legal」兩個分支，
- engineering sub-manager：拆成「frontend」與「backend」worker，
- legal sub-manager：一個 worker。

Demo 把順利路徑（大家都同意）和 **擾動路徑** 做對照。在擾動路徑中，頂層 manager 的拆解把「legal」誤標成「finance」，然後觀察錯誤如何層層擴散：sub-manager 很聽話地做財務工作，頂層綜合者回報財務發現，原本的法務問題則無人回答。

執行：

```
python3 code/main.py
```

輸出會顯示兩條路徑，並清楚並排「被要求的內容」與「實際交付的內容」。

## 實際使用 (Use It)

`outputs/skill-hierarchy-fitness.md` 評估某個任務該用 hierarchical、sequential，還是扁平的 supervisor。輸入：任務描述、組織結構、調和預算。輸出：pattern 建議，以及需要防範的具體失敗模式。

## 交付 (Ship It)

如果你要上線 hierarchical：

- **把樹的深度上限設為 2。** 三層就已經讓大多數錯誤從可觀測性中消失。
- **明確的調和預算。** 設定頂層 manager 必須做出決定之前的最大輪數。通常是 2。
- **每次綜合都附上 provenance。** 每個節點的摘要必須引用是哪些葉節點的輸出產生了它。
- **對拆解漂移發出警報。** 記錄 manager 每一步的拆解；與使用者的查詢做 diff。如果拆解不再涵蓋該查詢，就觸發警報。

## 練習 (Exercises)

1. 執行 `code/main.py`，比較順利路徑與擾動路徑。要經過幾層 manager 的 hand-off，頂層輸出才會完全偏離使用者的問題？
2. 加入第三層（top → sub → sub-sub → worker）。隨著深度增加，測量擾動路徑自我修正與完全偏離的頻率。
3. 在每個 sub-manager 實作一個「canary」worker，永遠被問到未經修改的原始使用者問題。用 canary 的答案偵測拆解漂移。當 canary 與綜合出的答案不一致時，manager 應該如何反應？
4. 閱讀 CrewAI 的 `Process.hierarchical` 文件。找出 CrewAI 採用的一項具體護欄（步數上限、manager_llm 限制），並描述它針對的是哪種失敗模式。
5. 比較巢狀的 LangGraph supervisor 和 CrewAI hierarchical。哪一個讓調和迴圈更容易被偵測？

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Hierarchical | 「組織圖 pattern」 | Supervisor 在 supervisor 之上；只有葉節點做實際工作。 |
| Manager LLM | 「老闆」 | 在內部節點負責拆解、分配和驗證的 LLM。 |
| Decomposition drift | 「老闆搞不清楚狀況了」 | 頂層 manager 的拆分不再涵蓋原本的問題。 |
| Reconciliation loop | 「沒完沒了的會議」 | Sub-manager 意見不一；上層重新委派；worker 重新執行；迴圈直到預算用盡。 |
| Depth-2 ceiling | 「不要超過 2 層」 | 經驗上的護欄：3 層以上會讓可觀測性崩潰。 |
| Canary question | 「每一層的 ground truth」 | 一個永遠被問到未經修改的原始查詢的 worker，用來偵測漂移。 |
| Provenance chain | 「誰說了什麼」 | 從每次綜合回溯到產生它的葉節點輸出的追蹤。 |

## 延伸閱讀 (Further Reading)

- [CrewAI introduction — Process.hierarchical](https://docs.crewai.com/en/introduction) — 帶 manager LLM 的教科書式 hierarchical
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — 透過 `create_supervisor` 的巢狀 supervisor
- [Anthropic engineering — Research system](https://www.anthropic.com/engineering/multi-agent-research-system) — 為什麼 Anthropic 刻意選擇扁平 supervisor 而不是 hierarchical
- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST 分類法；協調失敗那一節記錄了 decomposition drift
