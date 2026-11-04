# Scalable Oversight 與 Weak-to-Strong Generalization (Scalable Oversight and Weak-to-Strong Generalization)

> Burns et al.（OpenAI Superalignment，「Weak-to-Strong Generalization」，2023）為 superalignment 問題提出一個 proxy：用較弱模型產生的標籤去 fine-tune 強模型。如果強模型能從不完美的弱監督中正確泛化，那麼目前人類尺度的 alignment 方法或許能延伸到超人類系統。Scalable oversight 與 W2SG 互補。Scalable oversight（debate、recursive reward modeling、任務分解）提高監督者的有效能力，使它跟得上被監督的模型。W2SG 則確保強模型能從監督者提供的任何不完美監督中正確泛化。Debate Helps W2SG (arXiv:2501.13124, January 2025) 把兩者結合。

**Type:** Learn
**Languages:** Python (stdlib, W2SG gap simulator)
**Prerequisites:** Phase 18 · 01 (instruction-following), Phase 18 · 10 (AI Control), Phase 09 (RL foundations)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 定義 scalable oversight 與 weak-to-strong generalization，並解釋兩者如何互補。
- 描述 Burns et al. 2023 的實驗設置：用 GPT-2 的標籤去 fine-tune GPT-4。
- 解釋 performance gap recovered (PGR) 指標，以及它測量什麼。
- 說出三個主要的 scalable-oversight 機制（debate、recursive reward modeling、任務分解），以及各自的一項優點。

## 問題 (The Problem)

Phase 18 到目前為止的每一種 alignment 技術，都假設監督者能評估模型的行為。當模型是超人類時，監督者就是最弱的一環。Superalignment 的問題是：較弱的監督者能不能可靠地產生一個更強而且 aligned 的模型？

Burns et al. 把這個問題化簡為操作化的實證設置：用弱模型監督強模型，測量強模型的能力有多少能撐過弱監督。這不是 superalignment 的解答，而是讓我們能測量進展的 proxy。

## 核心概念 (The Concept)

### W2SG：Burns et al. 的設置

- 弱模型：GPT-2 等級。
- 強模型：GPT-4 等級。
- 目標：強 GPT-4 在該任務上的天花板。

流程：
1. 取得弱模型在任務上的 zero-shot 預測。
2. 在弱標籤資料上 fine-tune 強模型。
3. 測量 fine-tune 後強模型的準確率。

基準線：弱模型自己的準確率、強模型在 gold 標籤上的監督式天花板。

差距指標：Performance Gap Recovered (PGR) = (fine-tuned - weak) / (ceiling - weak)。PGR 為 1.0 表示弱監督完全填補差距；PGR 為 0 表示弱監督毫無幫助。

### Burns et al. 的實證發現

在 NLP、西洋棋謎題與 reward modeling 任務中，PGR 一致為正：強模型的泛化超越了弱監督者的錯誤。各任務的 PGR 從約 20% 到約 80% 不等。強模型從不完美的標籤中「理解」了預期的任務，並用它預訓練的先驗補足。

Burns et al. 指出的限制：
- 這裡的 weak-to-strong 差距是能力，不是 alignment。弱模型是 aligned 的；真正的超人類模型可能由人類對齊，但能力差距要大得多。
- 強模型的泛化可能更多來自任務先驗，而不是恢復 ground-truth 的意圖。
- 引出強模型的潛在知識才是真正的問題；PGR 測量的是一個特定的操作化。

### Scalable oversight：三個機制

- **Debate (Irving et al. 2018)。** U 的兩個實例為相反立場辯論；弱裁判做決定。假設：找到令人信服的真答案，比找到令人信服的假答案容易。2024-2025 年的近期研究（Khan et al.、Michael et al.）顯示 debate 有時有幫助、有時沒有，是否有幫助取決於任務結構。
- **Recursive Reward Modeling (Leike et al. 2018)。** U 協助人類訓練 U+1 的 reward model。監督者的有效能力隨模型一起成長。
- **Task Decomposition (Christiano, Shlegeris, Amodei 2018)。** 把困難任務遞迴分解成人類可以檢查的子任務。假設任務是可分解的。

每個機制都對任務結構或中間元件的 alignment 做了某些假設。

### 為什麼 scalable oversight 與 W2SG 互補

Scalable oversight 提高監督者的有效訊號品質。
W2SG 從監督者所能提供的任何不完美訊號中縮小差距。

Lang et al. 的 Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124) 把兩者結合：debate 協定提供更好的弱標籤，再用這些標籤訓練強模型。回報了在 NLP 任務上的 PGR 提升。

### 組織上的戲劇

OpenAI 的 Superalignment 團隊在 2024 年 5 月 Jan Leike 離職轉往 Anthropic 後解散。這項議程（scalable oversight、W2SG、自動化 alignment 研究）在 Anthropic 與學術實驗室持續進行，如 MATS（第 28 課）、Redwood（第 10 課）、Apollo（第 8 課）、METR（第 28 課）。組織結構變了，研究問題沒變。

### 這在 Phase 18 中的位置

第 6 到 10 課描述在 U 不值得信任的假設下的威脅與防禦典範。第 11 課是進攻典範：讓監督者強到足以驗證 U 的 alignment。第 12 到 16 課接著轉向 adversarial 評估的實務工具。

```figure
scalable-oversight
```

## 實際使用 (Use It)

`code/main.py` 在一個合成任務上模擬 W2SG fine-tune。弱標註者準確率 70%，帶有結構化錯誤；強模型在 gold 標籤上的天花板為 95%。你用弱標籤 fine-tune 強模型，測量 PGR，並與強模型用 gold 訓練、以及弱模型單獨表現相比較。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-w2sg-pgr.md`。給它一份監督設置的描述，它會找出弱監督者、強模型、監督品質，並計算（或要求提供）PGR。它會標示該主張是「弱模型能監督強模型」還是「弱模型加監督機制能監督強模型」。

## 練習 (Exercises)

1. 執行 `code/main.py`。回報 weak_accuracy = 0.60、0.70、0.80 時的 PGR。解釋 PGR 曲線的形狀。

2. 修改弱標註者，使其有結構化錯誤（例如對特定輸入類別總是答錯）。PGR 會增加、減少還是不變？請解釋。

3. 閱讀 Burns et al. 2023 第 4.3 節（NLP 任務）。重現「confidence auxiliary loss」的直覺：當強模型比弱標籤更有信心時，誰會勝出？

4. 為軟體工程任務設計一個結合 debate 與任務分解的 scalable-oversight 協定。說出每個元件的一種失敗模式，並解釋這個組合如何處理或無法處理每一種。

5. 闡述什麼會推翻「weak-to-strong generalization 是通往 superalignment 的可行路徑」這個主張。具體說明你需要看到什麼樣的實證特徵。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| Scalable oversight | 「making the overseer stronger」 | 提高監督者評估更強大模型之能力的機制 |
| W2SG | 「weak supervises strong」 | 在弱標籤上 fine-tune 強模型，並測量恢復的能力 |
| PGR | 「performance gap recovered」 | (fine-tuned - weak) / (ceiling - weak)；1.0 = 完全填補，0 = 毫無幫助 |
| Debate | 「two U instances argue」 | Scalable oversight 機制，由弱裁判在 U 的兩位辯護者之間做選擇 |
| RRM | 「recursive reward modeling」 | U 協助訓練 U+1 的 reward model；監督者能力跟著 U 走 |
| Task decomposition | 「sub-tasks the human checks」 | 把困難任務遞迴拆成人類可驗證的子任務 |
| Superalignment | 「aligning superhuman AI」 | 關於對齊人類無法直接評估之模型的研究議程 |

## 延伸閱讀 (Further Reading)

- [Burns et al. — Weak-to-Strong Generalization (OpenAI 2023)](https://openai.com/index/weak-to-strong-generalization/)：W2SG 論文
- [Irving, Christiano, Amodei — AI safety via debate (arXiv:1805.00899)](https://arxiv.org/abs/1805.00899)：debate 機制
- [Leike et al. — Scalable agent alignment via reward modeling (arXiv:1811.07871)](https://arxiv.org/abs/1811.07871)：recursive reward modeling
- [Khan et al. — Debating with More Persuasive LLMs Leads to More Truthful Answers (arXiv:2402.06782)](https://arxiv.org/abs/2402.06782)：2024 年用更強辯論者做 debate 的實證研究
- [Lang et al. — Debate Helps Weak-to-Strong Generalization (arXiv:2501.13124)](https://arxiv.org/abs/2501.13124)：2025 年 debate 與 W2SG 的結合
