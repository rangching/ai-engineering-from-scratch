# AI Scientist v2：Workshop 等級的自主研究 (Workshop-Level Autonomous Research)

> Sakana 的 AI Scientist v2 (Yamada et al., arXiv:2504.08066) 跑完整個研究 loop：假說、程式、實驗、圖表、撰寫、投稿。它是第一個產出的論文通過 ICLR 2025 workshop 同行審查的系統。獨立評估 (Beel et al.) 發現，42% 的實驗因為程式錯誤而失敗，文獻回顧也經常把已確立的概念誤標為新穎。Sakana 自己的文件警告，這個程式碼庫會執行 LLM 寫的程式碼，並建議使用 Docker 隔離。這幅畫面的兩半都是重點。

**Type:** Learn
**Languages:** Python (stdlib, research-loop state-machine toy)
**Prerequisites:** Phase 15 · 03 (AlphaEvolve), Phase 15 · 04 (DGM)
**Time:** ~60 分鐘

## 問題 (The Problem)

研究是一種開放式的任務。不像 AlphaEvolve 的演算法搜尋，或 DGM 受 benchmark 限定的自我修改，研究成果沒有機器可檢查的正確性標準。一篇論文是由審稿人評斷的，不是 unit test。這讓 loop 更難閉合，但若能閉合，價值更高，因為複利式的進展就發生在研究裡。

AI Scientist v1 (Sakana, 2024) 從人類撰寫的模板出發，閉合了這個 loop。LLM 在固定的 scaffolding 裡填入實驗。AI Scientist v2 (Yamada et al., 2025) 改用帶有 vision-language model 評論 loop 的 agentic tree search，不再需要模板。系統會產生點子、實作實驗、產出圖表、寫論文，並根據審稿意見迭代。

同行審查的結論：一篇 v2 產生的論文被 ICLR 2025 workshop 接受（並已揭露來源）。獨立評估的結論：這個系統離可靠還很遠。兩者都是真的。

## 核心概念 (The Concept)

### 架構

1. **點子產生。** LLM 根據主題和既有文獻提出研究點子。v1 用模板；v2 在假說空間上做 agentic search。
2. **新穎性檢查。** 文獻檢索步驟會檢查這個點子是否已被發表過。Beel et al. 的評估就是在這一步發現誤標：已確立的方法經常被歸類為新穎。
3. **實驗計畫。** Agent 擬定實驗協定並撰寫程式。
4. **執行。** 程式在 sandbox 中執行。失敗會回饋進重試 loop。在 Beel et al. 的量測中，這個階段有 42% 的實驗因為程式錯誤而失敗。
5. **圖表產生。** Vision-language model 讀取產生的圖表，並為了清晰度重寫它們。這是 v2 的關鍵技術新增項目。
6. **撰寫。** LLM 草擬論文，並與內部審稿人迭代。
7. **選用：投稿。** 論文被投到某個會議或期刊。

### Workshop 錄取結果的意義

一篇 v2 產生的論文通過了 ICLR 2025 workshop 的同行審查。作者向程式委員會揭露了論文的來源。這次錄取是一個資料點，並不是宣稱這個系統「會做研究」的許可證。

重要的背景：workshop 論文的門檻比主會議論文低。同行審查有雜訊；在任何一天，只有一小部分投稿會被接受。一次成功是概念驗證，不是可靠度的主張。2026 年的 Nature 論文記錄了端到端的 loop，而它本身是由人類研究者共同撰寫的；它不是「系統寫了一篇 Nature 論文」。

### 獨立評估發現了什麼

Beel et al. (arXiv:2502.14297) 做了外部評估。主要發現：

- **實驗失敗。** 42% 的實驗因為程式錯誤而失敗（錯誤的 import、shape 不吻合、未定義的變數）。重試 loop 抓到了一些，但不是全部。
- **新穎性誤標。** 文獻檢索步驟經常把已確立的概念標成新穎。這是研究版的 hallucination。
- **呈現品質落差。** Vision-language 的圖表評論產生了可發表等級的視覺效果，掩蓋了底層實驗的弱點。

最後一項發現對本階段最重要。一個產生出令人信服的輸出、卻沒有做出令人信服的研究的系統，比明顯失敗的系統更危險，而不是更安全。評估必須深入底層的主張，不能停在圖表上。

### Sandbox 逃脫的疑慮

Sakana 自己的 repository README 警告：

> Due to the nature of this software, which executes LLM-generated code, we cannot guarantee safety. There are risks of dangerous packages, uncontrolled web access, and spawning of unintended processes. Use at your own risk and consider Docker isolation.

這就是在未經驗證的領域中，自主性的實際樣貌。LLM 寫程式；程式被執行；程式可以做行程被允許做的任何事。沒有一個能硬性限制檔案系統、網路與行程動作的 sandbox，任何自我導向的研究 agent 都可能外洩資料、燒掉運算資源，或改寫自己。

AlphaEvolve 的 sandbox 比較好處理，因為它的 evaluator 很緊。AI Scientist v2 的 loop 則以開放式目標執行開放式程式碼。這就是為什麼它需要更強的隔離（Docker 是最低限度；最好是 seccomp / gVisor），以及在每份投稿離開系統之前進行人工審查。

### v2 在 frontier 技術棧中的位置

| 系統 | 目標 | 輸出類型 | Evaluator | 已知失敗 |
|---|---|---|---|---|
| AlphaEvolve | 演算法 | 程式碼 | unit + benchmark | 受限於 evaluator 的嚴謹程度 |
| DGM | agent scaffolding | 程式碼 | SWE-bench | reward hacking |
| AI Scientist v2 | 研究論文 | 文字 + 程式碼 + 圖表 | 同行審查（弱） | 實驗失敗、誤標、用包裝掩蓋弱點 |

三者之中，v2 的自動 evaluator 最弱、輸出面最廣，到公開成品的路徑也最短。運作上的控制措施（sandbox、審查、揭露）承擔了大部分的安全工作。

```figure
mx-research-loop
```

## 實際使用 (Use It)

`code/main.py` 把 v2 loop 模擬成一個狀態機：點子 → 新穎性檢查 → 實驗 → 圖表 → 撰寫 → 審查 → 接受或迭代。每個狀態都有可設定的失敗機率，取自 Beel et al. 的發現。執行 simulator N 次 loop，並計算：

- 有多少點子走到投稿。
- 有多少投稿帶有被包裝過的論文所隱藏的關鍵實驗缺陷。
- 重試預算如何在品質與產出量之間取捨。

## 交付 (Ship It)

`outputs/skill-ai-scientist-sandbox-review.md` 是一份雙關卡的審查清單，用於檢查研究 loop agent 產出的任何東西，在它離開 sandbox 之前。

## 練習 (Exercises)

1. 用預設參數執行 `code/main.py`。有多少比例的 loop 會產出「乾淨」的論文？有多少比例產出帶有實驗失敗缺陷、但被圖表評論包裝掉的論文？

2. 預設值已經使用 Beel et al. 的 42% / 25%。用 `--experiment-failure 0.20 --novelty-mislabel 0.10` 再執行一次，接著用 `--experiment-failure 0.60 --novelty-mislabel 0.40` 執行。「包裝漂亮但有缺陷」的比例在兩次之間如何變化？

3. 閱讀 Sakana 的 AI Scientist v2 repo README 中關於 sandbox 需求的部分。說出你在多日自主執行時，除了 Docker 之外會再加上的兩項限制。

4. 閱讀 Beel et al. 第 4 節關於呈現品質落差的部分。設計一個額外的 evaluator，能抓到看起來漂亮、但實驗上有缺陷的論文。

5. 為研究 agent 的輸出提出一個比「博士逐篇閱讀」更能擴展的人工審查協定。找出瓶頸，並圍繞它設計。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|---|---|---|
| AI Scientist v1 | 「Sakana 的模板式研究 agent」 | 把實驗填進固定的 scaffold |
| AI Scientist v2 | 「無模板的研究 agent」 | 帶有 VLM 圖表評論的 agentic tree search |
| Agentic tree search | 「分支式研究 agent」 | 平行展開多個實驗計畫；由內部 critic 修剪 |
| Vision-language critique | 「用 VLM 修飾圖表」 | 多模態模型讀取圖表，並為了清晰度重寫 |
| Literature retrieval | 「新穎性檢查」 | 搜尋先前的研究來確認點子的新穎性，已有記錄顯示會誤標 |
| Polish masking | 「漂亮的論文，壞掉的研究」 | 呈現品質高於實驗品質；隱藏了弱點 |
| Sandbox escape | 「LLM 的程式碼突破限制」 | Agent 執行的程式碼做了 loop 設計者沒預期的事 |

## 延伸閱讀 (Further Reading)

- [Yamada et al. (2025). The AI Scientist-v2](https://arxiv.org/abs/2504.08066) — 論文。
- [Sakana blog on the Nature 2026 publication](https://sakana.ai/ai-scientist-nature/) — 廠商摘要，附同行審查背景。
- [Beel et al. (2025). Independent evaluation of The AI Scientist](https://arxiv.org/abs/2502.14297) — 外部評估的數字。
- [Sakana AI Scientist v1 paper](https://arxiv.org/abs/2408.06292) — 模板式的前身。
- [Anthropic — Measuring AI agent autonomy](https://www.anthropic.com/research/measuring-agent-autonomy) — 對開放式研究 agent 更廣的框架。
