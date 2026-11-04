# Benchmark：WebArena 與 OSWorld (Benchmarks: WebArena and OSWorld)

> WebArena 在四個自架的應用程式上測試 web agent 的能力。OSWorld 在 Ubuntu、Windows、macOS 上測試桌面 agent 的能力。發表時（2023–2024），兩者都顯示出最佳 agent 與人類之間的巨大差距。差距正在縮小；失敗模式卻沒有改變。

**Type:** Learn
**Languages:** Python (stdlib)
**Prerequisites:** Phase 14 · 19 (SWE-bench, GAIA)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 描述 WebArena 的四個自架應用程式，以及為什麼以執行結果為基礎的評估很重要。
- 解釋為什麼 OSWorld 使用真實的 OS 截圖，而不是 accessibility API。
- 說出 OSWorld 的兩個主要失敗模式：GUI grounding 和操作知識。
- 總結 OSWorld-G 與 OSWorld-Human 在基礎 benchmark 之上增加了什麼。

## 問題 (The Problem)

通才 agent 可以呼叫 tool。它們能不能操作瀏覽器，點 20 下完成一次購物結帳？它們能不能只用鍵盤和滑鼠設定一台 Linux 機器？這些就是 WebArena 和 OSWorld 要回答的問題。

## 核心概念 (The Concept)

### WebArena (Zhou et al., ICLR 2024)

- 橫跨四個自架 web 應用程式的 812 個長時程 task：購物網站、論壇、類似 GitLab 的開發工具、商業 CMS。
- 外加工具：地圖、計算機、草稿本。
- 評估以執行結果為基礎，透過 gym API 進行：訂單下了嗎、issue 關閉了嗎、CMS 頁面更新了嗎？
- 發表時：最好的 GPT-4 agent 成功率 14.41%，人類為 78.24%。

自架這個框架很重要，因為目標應用程式是固定且可重現的，所以這個 benchmark 不會不穩定。

### 延伸 (Extensions)

- **VisualWebArena**：視覺上有依據的 task，成功與否取決於詮釋圖片（截圖是一級 observation）。
- **TheAgentCompany**（2024 年 12 月）：加入 terminal + 寫程式；更像真實的遠端工作環境。

### OSWorld (Xie et al., NeurIPS 2024)

- 橫跨 Ubuntu、Windows、macOS 的 369 個真實電腦 task。
- 對真實應用程式進行自由形式的鍵盤與滑鼠控制。
- 以 1920×1080 截圖作為 observation。
- 發表時：最佳模型 12.24%，人類 72.36%。

### 主要失敗模式 (Primary failure modes)

1. **GUI grounding。** 像素 → 元素的對應。模型難以在 1920×1080 中可靠地定位 UI 元素。
2. **操作知識 (Operational knowledge)。** 哪個選單有那個設定、哪個鍵盤快捷鍵、哪個偏好設定面板。人類花多年累積的知識長尾。

### 後續工作 (Follow-ups)

- **OSWorld-G**：564 個樣本的 grounding 套件 + Jedi 訓練集。把 grounding 與 planning 拆開，讓你可以分別衡量。
- **OSWorld-Human**：人工整理的黃金 action trajectory。顯示頂尖 agent 用的步數比必要的多 1.4-2.7 倍（trajectory 效率差距）。

### 為什麼這很重要 (Why this matters)

Claude computer use、OpenAI CUA、Gemini 2.5 Computer Use（Lesson 21）都是用 WebArena 和 OSWorld 形狀的工作負載訓練的。Benchmark 是目標；正式環境的模型是出貨的答案。

### Benchmark 哪裡會出錯 (Where benchmarking goes wrong)

- **只靠截圖的 eval。** OSWorld 是以截圖驅動的；在 OSWorld 上評估使用 DOM 或 accessibility API 的 agent，會漏掉 grounding 這個挑戰。
- **忽略 trajectory 長度。** 只計算成功率，會漏掉 OSWorld-Human 揭露的 1.4-2.7 倍步數低效率。
- **過時的自架應用程式。** WebArena 的應用程式固定在特定版本；未重新整理就更新，會破壞可比較性。

```figure
ae-agent-human-gap
```

## 動手做 (Build It)

`code/main.py` 實作一個 toy web-agent harness：

- 一個最小的「購物應用程式」狀態機：list_items、add_to_cart、checkout。
- 3 個 task 的黃金 trajectory。
- 一個嘗試每個 task 的腳本化 agent。
- 以執行結果為基礎的 evaluator（狀態檢查）與 trajectory 效率指標（步數對黃金）。

執行：

```
python3 code/main.py
```

輸出：每個 task 的成功率與 trajectory 效率，對應 OSWorld-Human 的方法論。

## 實際使用 (Use It)

- **WebArena Verified** 自架在內部 cluster 上，做持續評估。
- **OSWorld** 在 VM 機群中，用於桌面 agent。
- **Computer-use agent**（Lesson 21）：Claude、OpenAI CUA、Gemini 都是用類似這些的工作負載訓練的。
- **你自己的產品流程**：為你最常見的 20 個 task 擷取黃金 trajectory；每週用 agent 對它們執行一次。

## 交付 (Ship It)

`outputs/skill-web-desktop-harness.md` 建立 web/desktop agent harness，包含以執行結果為基礎的 eval 和 trajectory 效率指標。

## 練習 (Exercises)

1. 用第二個應用程式（論壇）擴充 toy harness。寫 3 個 task 以及黃金 trajectory。
2. 加上每個 task 的 trajectory 效率回報。在你的 toy 上，agent 是黃金的 1 倍、2 倍還是 3 倍？
3. 實作一個「干擾」tool，也就是黃金 trajectory 從不使用的 tool。腳本化的 agent 會被誘惑嗎？
4. 讀 OSWorld-G。在你自己的 eval 中，你要如何把 grounding 失敗與 planning 失敗分開？
5. 讀 WebArena 的應用程式 README。當你升級其中一個固定版本的應用程式時，什麼會壞掉？

## 關鍵術語 (Key Terms)

| Term | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| WebArena | 「Web agent benchmark」 | 橫跨 4 個自架應用程式的 812 個 task；gym 風格評估 |
| VisualWebArena | 「視覺版 WebArena」 | 視覺上有依據的 WebArena；截圖是 observation |
| OSWorld | 「桌面 agent benchmark」 | 真實 Ubuntu/Windows/macOS 上的 369 個 task |
| GUI grounding | 「像素到元素的對應」 | 模型在 1920x1080 中定位 UI 元素 |
| Operational knowledge | 「OS 的竅門」 | 哪個選單、哪個快捷鍵、哪個偏好設定面板 |
| OSWorld-G | 「Grounding 套件」 | 564 個僅限 grounding 的樣本 + 訓練集 |
| OSWorld-Human | 「黃金 trajectory」 | 人工專家的 action 序列，用來衡量效率 |
| Trajectory efficiency | 「超出黃金的步數」 | Agent 步數除以人類最少步數 |

## 延伸閱讀 (Further Reading)

- [Zhou et al., WebArena (arXiv:2307.13854)](https://arxiv.org/abs/2307.13854)：四個應用程式的 web benchmark
- [Xie et al., OSWorld (arXiv:2404.07972)](https://arxiv.org/abs/2404.07972)：跨 OS 的桌面 benchmark
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use)：Claude 以 benchmark 為形狀的能力
- [OpenAI, Computer-Using Agent](https://openai.com/index/computer-using-agent/)：OSWorld 與 WebArena 的數字
