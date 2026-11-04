# Multimodal Agent 與 Computer-Use（總結專題）(Multimodal Agents and Computer-Use (Capstone))

> 2026 年的前沿產品是一個 multimodal agent：它讀螢幕截圖、點按鈕、瀏覽網頁 UI、填表單，並端到端完成工作流程。SeeClick 和 CogAgent (2024) 證明了 GUI grounding 這個基本單元。Ferret-UI 加入了行動裝置。ChartAgent 為圖表引入了視覺工具使用。VisualWebArena 與 AgentVista (2026) 是前沿正在追逐的 benchmark，即使是 Gemini 3 Pro 和 Claude Opus 4.7，在 AgentVista 的困難任務上也只得到約 30%。這個總結專題把 Phase 12 的每一條線索拉在一起：感知（高解析度 VLM）、推理（具備工具使用的 LLM）、grounding（座標輸出）、長時間範圍的記憶，以及評估。

**Type:** Capstone
**Languages:** Python (stdlib, action schema + agent loop skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 09 (Qwen-VL JSON), Phase 14 (Agent Engineering)
**Time:** ~240 分鐘

## 學習目標 (Learning Objectives)

- 設計 multimodal agent 迴圈：感知 → 推理 → 行動 → 觀察 → 重複。
- 建立 VLM 可以用 JSON 輸出的 GUI grounding 輸出 schema（點擊座標、輸入文字、捲動、拖曳）。
- 比較僅用螢幕截圖的 agent、accessibility-tree agent 與混合型 agent。
- 在一小段 VisualWebArena 切片上設定 multimodal agent 的 benchmark 評估。

## 問題 (The Problem)

一個訂票網站的工作流程：「幫我找 4 月 15 日飛東京、走道座位、低於 $800 的班機，並訂票。」

Multimodal agent 需要：

1. 對瀏覽器截圖。
2. 把截圖 + URL + 目標解析成計畫。
3. 輸出結構化動作：點擊 (x,y)、輸入 "Tokyo"（在元素 E）、向下捲動、選取（radio button）。
4. 把動作套用到瀏覽器。
5. 觀察新的狀態（下一張截圖）。
6. 重複直到任務完成。

每一步都是一次 multimodal VLM 呼叫。VLM 的輸出必須是可解析的 JSON。錯誤會在各步驟間累積，所以復原很重要。

## 核心概念 (The Concept)

### GUI grounding：基本單元 (GUI grounding — the primitive)

GUI grounding 是：給定螢幕截圖與自然語言指令，輸出要點擊（或其他動作）的 (x, y) 座標。

SeeClick (arXiv:2401.10935) 是第一個大規模的開源成果：用合成 + 真實的 GUI 資料 fine-tune VLM，把座標以純文字 token 輸出。有效。

CogAgent (arXiv:2312.08914) 為密集 UI 加入了 1120x1120 的高解析度編碼。分數：網頁導覽約 84%。

Ferret-UI (arXiv:2404.05719) 專注於行動 UI，與 iOS accessibility 資料整合。

輸出格式通常是 JSON：

```json
{"action": "click", "x": 384, "y": 220, "element_desc": "Search button"}
```

`element_desc` 有助於復原：如果座標在截圖之間漂移，語意提示讓系統能重新 grounding。

### 動作 schema (Action schemas)

典型的動作 schema 有 6-10 種動作類型：

- `click`：(x, y)
- `type`：(text, x?, y?)
- `scroll`：(direction, amount)
- `drag`：(x0, y0, x1, y1)
- `select`：(option_index)
- `hover`：(x, y)
- `navigate`：(url)
- `wait`：(ms)
- `done`：(success, explanation)

agent 每一步輸出一個動作。瀏覽器包裝層執行並回傳新的狀態。

### 僅用截圖與 accessibility-tree 的比較 (Screenshot-only vs accessibility-tree)

兩種輸入模式：

- 僅用截圖：完整圖片，沒有結構資訊。最通用；適用於任何 app。
- Accessibility tree：結構化的 DOM / iOS accessibility 資訊。grounding 可靠得多；用於有 tree 可用的地方。
- 混合：兩者並用，以 tree 作為原子動作的可靠 grounding 依據，以截圖提供語意 context。

Production agent 盡可能使用混合型。瀏覽器自動化 (Selenium + accessibility) 永遠有 tree；桌面 app 有時有。

### 長時間範圍的記憶 (Long-horizon memory)

20 步的工作流程會產生 20 張截圖。VLM 的 context 很快就會被塞滿。三種壓縮策略：

- Summary-chain：每 5 步摘要一次發生了什麼，丟掉舊截圖。
- Skip-frame：保留第一張、最後一張，以及每隔 3 張的截圖。
- 工具記錄的日誌：執行動作，保留做過什麼的文字日誌；不再回頭看舊截圖。

Claude 的 computer-use API 使用日誌模式。更簡單，也更可靠。

### 視覺工具使用 (Visual tool use)

ChartAgent (arXiv:2510.04514) 為圖表理解引入視覺工具使用：裁切、縮放、OCR、呼叫外部偵測。agent 可以把「裁切到區域 (100, 200, 300, 400) 然後呼叫 OCR」當作 tool call 輸出。工具回傳文字；VLM 繼續推理。

這個模式可以推廣：set-of-mark prompting、區域註記與外部偵測工具，都符合相同的「輸出 tool call、接收結構化回應」schema。

### 2026 年的 benchmark (The 2026 benchmarks)

- ScreenSpot-Pro。約 1000 張網頁截圖上的 GUI grounding。開源 SOTA Qwen2.5-VL-72B 約 85%。前沿約 90%。
- VisualWebArena。端到端網頁任務（購物、論壇、分類廣告）。開源 SOTA 約 20%。Gemini 3 Pro 約 27%。
- AgentVista (arXiv:2602.23166)。2026 年最困難的 benchmark。涵蓋 12 個領域的真實工作流程。前沿模型得分 27-40%；開源模型 10-20%。
- WebArena / WebShop。較舊的 benchmark；已被前沿模型飽和。

### 為什麼仍然困難 (Why it's still hard)

agent 表現的瓶頸：

1. 細微尺度的視覺 grounding。「點擊小小的 X」在行動裝置解析度下常常失敗。
2. 長時間範圍規劃。經過 10 個動作後，agent 就偏離了目標。
3. 錯誤復原。當點擊失敗（按錯按鈕）時，偵測與復原很少有訓練資料。
4. 跨頁面 context。在分頁之間跳轉或填寫長表單會遺失狀態。

研究方向：記憶架構、明確的重新規劃、multimodal 驗證（以截圖比對確認動作成功）。

### 總結專題實作 (The capstone build-it)

總結專題任務：建立一個 computer-use agent，它要：

1. 讀取訂票網站模擬頁面的 HTML + 截圖。
2. 規劃多步驟序列：搜尋 → 選擇 → 填表單 → 送出。
3. 輸出符合動作 schema 的 JSON 動作。
4. 在固定的 10 個任務切片上評估。

這堂課提供容易擴充成真實瀏覽器的骨架程式碼。

```figure
mm-agent-loop
```

## 實際使用 (Use It)

`code/main.py` 是總結專題的骨架：

- 動作 schema 的 JSON 定義（10 種動作：上面列出的九種，加上 `screenshot_region`，一個裁切 + 縮放工具）。
- 以 dict 表示的模擬瀏覽器狀態。
- Agent 迴圈骨架：接收狀態、輸出動作、套用、迴圈。
- 10 個任務的迷你 benchmark（合成頁面），用來量測端到端成功率。
- 動作失敗時的錯誤復原 hook。

## 交付 (Ship It)

這堂課產出 `outputs/skill-multimodal-agent-designer.md`。給定 computer-use 產品（領域、動作集、評估目標），它會設計完整的 agent 迴圈、記憶策略、grounding 模式與預期的 benchmark 分數。

## 練習 (Exercises)

1. 程式碼的 schema 在九個核心動作之外，加了 `screenshot_region` 工具（裁切 + 縮放）。哪些任務會受益？

2. 閱讀 AgentVista (arXiv:2602.23166)。描述最困難的任務類別，以及前沿模型為什麼仍然失敗。

3. 長時間範圍記憶壓縮：設計一個 summary-chain，最多保留 ≤4 張即時截圖，記錄數量不限。

4. 建立錯誤復原 hook：動作失敗時（找不到按鈕），agent 接下來做什麼？

5. 在 10 個網頁任務上，比較僅用截圖的 Claude 4.7 與混合截圖 + accessibility-tree 的 Qwen2.5-VL。哪個在哪些任務上勝出？

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| GUI grounding | "Click coordinates" | 模型為截圖上某個指令的目標輸出 (x,y) |
| Action schema | "Tool definitions" | 有效動作（click、type、scroll、drag）的 JSON 描述 |
| Accessibility tree | "Structured DOM" | 來自瀏覽器/iOS API、機器可讀的 UI 階層 |
| Hybrid agent | "Screenshot + tree" | 同時使用圖片與結構化資訊；比單獨任一種更可靠 |
| Visual tool use | "Zoom/crop/detect" | agent 在規劃中途呼叫外部視覺工具（OCR、偵測） |
| Summary-chain | "Memory compression" | 以定期的文字摘要取代冗長的截圖歷史 |
| VisualWebArena | "E2E web bench" | 2024 年的端到端網頁任務 benchmark |
| AgentVista | "2026 hard bench" | 12 個領域的真實工作流程；即使 Gemini 3 Pro 也只得到約 30% |

## 延伸閱讀 (Further Reading)

- [Cheng et al. — SeeClick (arXiv:2401.10935)](https://arxiv.org/abs/2401.10935)
- [Hong et al. — CogAgent (arXiv:2312.08914)](https://arxiv.org/abs/2312.08914)
- [You et al. — Ferret-UI (arXiv:2404.05719)](https://arxiv.org/abs/2404.05719)
- [ChartAgent (arXiv:2510.04514)](https://arxiv.org/abs/2510.04514)
- [Koh et al. — VisualWebArena (arXiv:2401.13649)](https://arxiv.org/abs/2401.13649)
- [AgentVista (arXiv:2602.23166)](https://arxiv.org/abs/2602.23166)
