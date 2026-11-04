# LLaVA-OneVision：單圖、多圖、影片合一的模型 (LLaVA-OneVision: Single-Image, Multi-Image, Video in One Model)

> 在 LLaVA-OneVision (Li et al., 2024 年 8 月) 之前，open-VLM 世界有各自獨立的系譜：LLaVA-1.5 處理單張圖片，Mantis 和 VILA 等處理多圖，Video-LLaVA 和 Video-LLaMA 處理影片。每個都在自己的 benchmark 上贏，在其他的上輸。LLaVA-OneVision 主張，單一 curriculum 就能訓練出一個在三種情境都稱霸的模型，而且湧現出的任務遷移效果（單圖技能輸出到影片、多圖推理輸出到單圖）勝過各個專家模型的總和。配方看似簡單：跨情境平衡的 visual-token 預算，加上一個明確的 curriculum，從單圖移動到 OneVision（多圖），再到影片。這堂課會讀預算、curriculum 與湧現行為。

**Type:** Build
**Languages:** Python (stdlib, token budget solver + curriculum planner)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 12 · 06 (any-resolution)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 設計一個在單圖、多圖與影片輸入之間保持平衡的 visual-token 預算。
- 安排訓練 curriculum 的順序，讓技能從單圖遷移到影片，而不發生 catastrophic forgetting。
- 解釋為什麼 curriculum 做對時，單一模型在相同參數量下勝過各個專家。
- 說出 LLaVA-OneVision 回報的三種湧現能力：multi-camera 推理、set-of-mark prompting、iPhone 截圖 agent。

## 問題 (The Problem)

單圖、多圖與影片，各自以不同方式壓迫模型。

單圖需要高解析度 token（AnyRes，約 2880 個 visual token）來抓住 OCR 與細節。每個樣本的預算：一張圖片，2880 個 token。

多圖需要幾張中等解析度（各約 576 個 token）的圖片，讓跨圖片推理能塞進 context。每個樣本的預算：4-8 張圖片，各 576 個，共 2300-4600 個 token。

影片需要許多低解析度的影格（pooling 後每影格約 196 個 token）來捕捉時間動態。每個樣本的預算：8-32 個影格，各 196 個，共 1600-6200 個 token。

如果你訓練分開的模型，你只挑一種預算。如果你訓練單一模型，就需要預算能在各情境間合理伸縮，又不撐爆 context。

在 OneVision 之前，預設答案是「訓練一種情境，忽略其他的。」Video-LLaVA 靠額外的訓練階段把影片改裝到圖片模型上。LLaVA-NeXT 用 tiling 加入多圖支援。沒有一個能乾淨處理全部三種。

## 核心概念 (The Concept)

### OneVision 的 token 預算 (The OneVision token budget)

LLaVA-OneVision 並不固定某個精確的 token 數。它平衡每個樣本的最大 visual-token 數，讓三種情境落在相同範圍，約 6,000 到 9,000 個 token，並在各情境以不同方式分配：

- 單圖：AnyRes-Max-9（最多 9 個 tile 加一張基底視圖），每個 384x384 視圖從 SigLIP SO400M 得到 729 個 token。最大值：729 * (1 + 9) = 7290 個 token。
- 多圖：每張圖片 384、不做 tiling，729 個 token、不做 pooling。最多 12 張圖片 → 8748 個 token。
- 影片：最多 32 個影格，384 並做 2x2 雙線性 pooling → 每影格 196 個 token。總計：32 * 196 = 6272 個 token。

這個分配讓最大總量保持在同一個範圍，所以沒有任何情境主導 context 或梯度。encoder 在每種情境產生不同的幾何形狀，但 LLM 消耗的是相近的預算。下面 `code/main.py` 的規劃器使用較小的 4096 玩具預算，展示相同的平衡邏輯。

### 三階段 curriculum (The three-stage curriculum)

LLaVA-OneVision 分三個階段訓練：

1. 單圖 SFT（階段 SI）。所有資料都是單圖加文字。在高解析度 AnyRes 輸入上訓練。這教會感知、OCR 與細緻理解。使用 LLaVA-NeXT 資料加上 OneVision 專屬的單圖資料。
2. OneVision SFT（階段 OV）。混合單圖 + 多圖 + 影片（均勻取樣的影格）。在統一的 token 預算上訓練。這教模型處理異質的 batch 形狀。不重設權重，從階段 SI 接續。
3. 任務遷移（階段 TT）。以目標任務組合繼續訓練，通常依產品偏重多圖或影片。可選擇為部署做 fine-tune。

關鍵：curriculum 的順序很重要。先訓練影片或先訓練多圖，圖片表現會比先訓練單圖差，即使資料相同。論文明確做了這個 ablation。

### 為什麼 curriculum 有效 (Why curriculum works)

單圖訓練打下感知基礎。patch token 攜帶細緻的視覺特徵；LLM 學著把它們與文字整合。多圖與影片引入結構上的挑戰（哪張圖是哪張、什麼先發生），沒有強大的感知基礎很難學會。

如果你把所有情境從頭一起訓練，模型會感知欠擬合（每個 batch 的單圖資料有限）而結構過擬合（大量多圖/影片資料）。結果是一個遵循跨圖推理模式、但視覺上很淺的模型。

curriculum 排序讓你從階段 SI 得到感知強度，再從階段 OV 得到組合/時間推理，兩者都不流失。

### 湧現的跨情境技能 (Emergent cross-scenario skills)

LLaVA-OneVision 論文回報三種湧現能力：

1. Multi-camera 推理。多圖與影片分別訓練；推論時被要求推理多鏡頭的駕駛場景。模型正確整合各視角，儘管訓練中從未見過那種確切格式。
2. Set-of-mark prompting。使用者用編號標記標註圖片中的物體；模型推理「標記 3 相對於標記 7 在做什麼」。訓練中既沒有標記也沒有標註；是從空間 grounding 加多圖參照的組合中學到的。
3. iPhone 截圖 agent。使用者提供 iPhone 畫面截圖，要求規劃下一步點擊。訓練資料有 UI 截圖、使用者工作流程影片，以及多圖的前後配對。能泛化到 agent 使用情境。

這些不是被訓練的任務；它們從 curriculum 的組合結構中湧現。

### Visual-token pooling

token 預算需要 pooling。OneVision 在 2D patch 網格上使用雙線性插值：24x24 = 576 個 patch 變成 12x12 = 144（2 倍因子）或 8x8 = 64（3 倍因子）。pooling 在 patch 網格空間中進行，而不是 token 空間，以保留局部性。

每個情境的 pooling 因子選擇本身就是超參數。Pooling 越少 = token 越多 = 表示越豐富。Pooling 越多 = token 越少 = 能塞進更多影格/圖片。

### LLaVA-OneVision-1.5

2025 年的後續作品 (LLaVA-OneVision-1.5, arXiv 2509.23661) 在訓練資料、模型權重與程式碼上「完全開放」。在某些 benchmark 上追平了與專有模型的差距，並讓配方普及。相同的 curriculum、更多資料、更好的基礎 LLM。架構沒有改變。

### 與 Qwen2.5-VL 的對比 (Contrast with Qwen2.5-VL)

Qwen2.5-VL（課程 12.09）做了不同的選擇。它用 M-RoPE 和動態 FPS，而不是固定的 pooling。它的預算隨輸入伸縮，1 分鐘的影片用的 token 比 5 秒的影片多。LLaVA-OneVision 固定預算，改為伸縮 pooling。兩者都行得通；是在可設定性與可預測性之間取捨。

```figure
l5-onevision-budget
```

## 實際使用 (Use It)

`code/main.py` 是 OneVision 風格 VLM 的 curriculum 與預算規劃器。給定每個樣本的 token 預算與目標情境組合（例如 40% 單圖、30% 多圖、30% 影片），它會：

- 為每個情境分配解析度、pooling 因子與影格數。
- 檢查每個情境是否都在共用預算內。
- 回報預期的 token 數、LLM FLOPs，以及哪些情境的 token 不足。
- 印出逐階段的訓練排程。

用它來規劃 OneVision 的 fine-tune，或檢查 VLM 部署的每請求成本是否合理。

## 交付 (Ship It)

這堂課產出 `outputs/skill-onevision-budget-planner.md`。給定目標任務分布與每個樣本的預算，它會輸出 AnyRes 因子、每影格 pooling、影片影格數與 curriculum 各階段的權重。每當你訓練或 fine-tune 統一情境的 VLM 時就用它。

## 練習 (Exercises)

1. 你的產品支援 80% 單圖、10% 多圖（2-4 張圖片）、10% 影片（8-16 個影格）。設計 token 預算。因為不做大量多圖而省下的額外預算，你會用在哪裡？

2. 閱讀 LLaVA-OneVision 第 4.3 節（湧現能力）。提出第四種 curriculum 很可能解鎖、但論文沒有回報的湧現技能。

3. 交換 curriculum 的順序：先訓練多圖，再單圖，最後影片。預測哪些 benchmark 會退化，以及原因。

4. 論文回報的影片 benchmark 每個樣本只用 8 個影格訓練。這能在推論時泛化到 30 秒的影片嗎？什麼會先壞掉，是 token 預算還是時間推理？

5. 把 24x24 的 patch 雙線性 pooling 成 12x12，每個維度縮小 2 倍，所以 token 減少 4 倍。用標準函式庫的 Python 實作 pooling，並驗證每個 2x2 區塊的平均與雙線性輸出相符。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| OneVision scenario | "Single-image, multi-image, or video" | 統一 VLM 處理的三種輸入形狀之一；預算在三者之間保持平衡 |
| Token budget | "How many tokens per sample" | LLM 在每個訓練/推論樣本中看到的 visual token 總數；LLaVA-OneVision 中最多約 6k 到 9k |
| Curriculum | "Training order" | 為了湧現遷移而選定的階段順序（單圖 → 多圖 → 影片） |
| Bilinear pooling | "Token shrink" | 對 patch 網格 (2D) 套用雙線性插值，減少 token 數同時保留局部性 |
| Emergent skill | "Not trained, still works" | 沒有對應訓練資料、卻在推論時出現的能力，來自 curriculum 的組合 |
| AnyRes-k | "k-tile setup" | k 個固定解析度的子 tile 加一張縮圖，常見 $k \in \lbrace 4, 9 \rbrace$ |
| Task transfer | "Cross-scenario generalization" | 透過共用 backbone，在單圖學到的技能可套用到影片（反之亦然） |

## 延伸閱讀 (Further Reading)

- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326)
- [LLaVA-OneVision-1.5: Fully Open Framework (arXiv:2509.23661)](https://arxiv.org/abs/2509.23661)
- [Lin et al. — Video-LLaVA (arXiv:2311.10122)](https://arxiv.org/abs/2311.10122)
- [Lin et al. — VILA (arXiv:2312.07533)](https://arxiv.org/abs/2312.07533)
- [Wang et al. — Qwen2-VL (arXiv:2409.12191)](https://arxiv.org/abs/2409.12191)
