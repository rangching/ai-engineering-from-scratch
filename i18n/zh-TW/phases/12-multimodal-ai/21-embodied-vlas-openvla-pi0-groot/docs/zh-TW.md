# Embodied VLA：RT-2、OpenVLA、π0、GR00T (Embodied VLAs: RT-2, OpenVLA, π0, GR00T)

> 模型第一次從網站上讀到食譜，並在廚房機器人上執行，是 RT-2 (Google DeepMind, 2023 年 7 月)。RT-2 把動作離散化成文字 token，用網路資料加機器人動作資料對 VLM 做 co-fine-tune，證明了網路規模的 vision-language 知識可以遷移到機器人控制。OpenVLA (2024 年 6 月) 推出了開源 7B 參考模型。Physical Intelligence 的 π0 系列 (2024-2025) 加入了 flow-matching 動作專家。NVIDIA 的 GR00T N1 (2025 年 3 月) 為人形機器人大規模提供了雙系統 (System 1 / System 2) 控制。VLA 這個基本單元（vision-language-action，單一模型能看、能讀、能行動）是這個階段的理解模型與 Phase 15 自主系統之間的橋樑。

**Type:** Learn
**Languages:** Python (stdlib, action tokenizer + VLA inference skeleton)
**Prerequisites:** Phase 12 · 05 (LLaVA), Phase 15 (Autonomous Systems, referenced)
**Time:** ~180 分鐘

## 學習目標 (Learning Objectives)

- 描述動作 tokenization：離散 bin 編碼 (RT-2)、FAST 高效動作 token、連續 flow-matching 動作 (π0)。
- 解釋為什麼在網路 + 機器人資料上做 co-fine-tune，能保留對新任務的通用知識遷移。
- 在同一個機器人任務上比較 OpenVLA（開源 7B Llama+VLM）、π0（flow-matching）與 GR00T N1（雙系統）。
- 說出 Open X-Embodiment 資料集，以及它作為 RT-X 訓練語料庫的角色。

## 問題 (The Problem)

能依自然語言指令做家事的機器人，自 1970 年代起就是研究目標。2020 年代的答案：vision-language-action (VLA) 模型。使用與 VQA 相同的 VLM 架構，但輸出是動作（關節力矩、末端執行器姿態、離散指令），而不是文字。

VLA 特有的挑戰：

1. 動作空間是連續的（關節角度、力）且高維度（7 自由度手臂 + 3 自由度夾爪 = 30 Hz 下 10 維）。
2. 機器人專屬的訓練資料很稀少。Open X-Embodiment 約有 100 萬條軌跡；網路圖文則超過 50 億。
3. 控制頻率很重要。30 Hz 的控制迴圈代表每個動作有 33ms 的預算。
4. 安全。錯誤的動作會損壞硬體、傷害人或財物。

## 核心概念 (The Concept)

### 動作 tokenization (RT-2)

RT-2 的技巧：把每個關節目標表示成量化的文字 token。把正規化的 [-1, 1] 範圍離散成 256 個 bin，每個 bin 對應到一個詞彙表 ID。10 自由度的動作在每個控制步驟變成 10 個 token。

在混合資料上對 PaLM-X VLM 做 co-fine-tune：

- 網路圖文配對（captioning、VQA）。
- 機器人示範，動作以 token 表示。

模型看到「pick up the red cube」（語言）→ 圖片（視覺）→ 10 個 token 的動作序列（離散化的關節目標）。網路 pretraining 保留了通用知識的遷移：RT-2 能遵循「move towards the fast-moving object」，儘管「fast-moving」不在訓練資料中。

RT-2 論文中的推論頻率為 3-5 Hz，受 VLM autoregressive 解碼所限。

### OpenVLA：開源的 7B 參考模型 (OpenVLA — the open 7B reference)

OpenVLA (Kim et al., 2024 年 6 月) 是開放權重的 RT-2 對應版本。7B Llama backbone、DINOv2 + SigLIP 雙 vision encoder、256 個 bin 的動作 tokenization。

在 Open X-Embodiment（22 台機器人的 97 萬條軌跡）上訓練。內建支援 LoRA fine-tuning，用來適應新機器人。

推論：A100 上搭配量化為 4-5 Hz。對慢速操作夠快，對高頻控制則不夠。

### FAST tokenizer：更快的動作解碼

Pertsch 等人 (2024) 顯示離散 bin 的 tokenization 效率不彰，因為大多數動作都集中在 bin 空間的一小塊區域。FAST (Frequency-domain Action Sequence Tokenizer) 透過 DCT 壓縮動作序列，並量化係數。

30 步的動作軌跡變成約 10 個 FAST token，而不是 300 個離散 bin token。推論加速 3-5 倍，品質不損失。

### π0 與 flow-matching 動作 (π0 and flow-matching actions)

Physical Intelligence 的 π0 (Black et al., 2024 年 10 月) 用 flow-matching 動作專家取代離散動作 token：

- 小型動作 transformer 讀取 VLM 的 hidden state，透過 rectified flow 輸出連續的 50 步動作序列。
- 動作 head 以 flow-matching loss 訓練；VLM pretraining 保持不變。
- 推論：約 5 個去噪步驟就輸出完整動作序列，實質上相當於 50 Hz 控制。

π0 的主張：在廣泛的操作任務套件上勝過 OpenVLA 和 Octo。連續動作的形式保留了離散化會破壞的平滑性。

π0.5 與 π0-FAST 是漸進式升級。π0-FAST 結合了 FAST tokenization 與 flow matching。

### GR00T N1：人形機器人的雙系統 (GR00T N1 — dual-system for humanoids)

NVIDIA 的 GR00T N1 (2025 年 3 月) 是為人形機器人打造的（>30 自由度、全身）：

- System 2：大型 VLM 讀取場景 + 指令，以約 1 Hz 產生高層次的子目標。
- System 1：小型動作 head transformer，以子目標為條件，產生 50-100 Hz 的低層次關節指令。

這個拆分對應 Kahneman 的快與慢思考：System 2 規劃，System 1 行動。好處：VLM 等級的慢速規劃不會擋住快速控制；System 1 保持小型以維持低延遲。

GR00T N1.7（2025 年底）改善了資料 scaling。GR00T 以來自 Omniverse 的 sim-to-real 資料進行 fine-tune。

### Open X-Embodiment

訓練資料。RT-X (2023 年 10 月) 整合了 22 個資料集，涵蓋 22 台機器人、100 萬條軌跡。Open X-Embodiment 是大家使用的語料庫：

- ALOHA / Bridge V2 / Droid / RT-2 Kitchen / Language Table。
- 每個樣本：（機器人狀態、相機視角、指令、動作序列）。
- 訓練衛生：統一動作空間、正規化關節範圍、調整相機大小。

OpenVLA 和 π0 都在 Open X-Embodiment 上訓練。與任一特定機器人的領域差距，靠在 100-1000 筆任務專屬示範上做 LoRA fine-tuning 來彌補。

### Co-fine-tuning 對僅用機器人資料 (Co-fine-tuning vs robot-only)

Co-fine-tuning 把網路 VQA 資料與機器人軌跡混在一起。比例很重要：VQA 太多，模型會忘記動作；機器人資料太多，模型會失去通用知識。

RT-2 的比例：約 1:1。OpenVLA：網路對機器人約 0.5:1。π0：類似。確切比例是要依資料集大小調整的超參數。

僅用機器人資料訓練會產生任務專屬的模型，遇到分布外的指令就失敗。Co-fine-tuning 是「pick up the red cube（示範中的）」與「pick up the third largest object from the left（新穎的措辭）」之間的差別。

### 安全與動作限制 (Safety and action limits)

每個 production VLA 都附帶：

- 硬性關節限制（不能超出規格施加力矩）。
- 速度限制（軟性截斷）。
- 工作空間邊界（末端執行器不能離開桌面）。
- 對新任務需要人工介入核准。

這些都放在 VLA 之外，作為控制層的檢查。VLA 的輸出是建議，不是命令。

```figure
mm-action-tokens
```

## 實際使用 (Use It)

`code/main.py`：

- 實作 256-bin 的動作 tokenization 與反 tokenization。
- 草擬以 DCT + 量化為基礎的 FAST tokenizer。
- 比較 (離散 bin、FAST、連續 flow) 每個動作步驟的 token 數。
- 印出 RT-2 → OpenVLA → π0 → GR00T 的系譜摘要。

## 交付 (Ship It)

這堂課產出 `outputs/skill-vla-action-format-picker.md`。給定機器人任務（操作、導航、人形全身），它會在離散 bin + RT-2、FAST + OpenVLA、flow-matching + π0 或雙系統 + GR00T 之間做選擇。

## 練習 (Exercises)

1. 10 自由度的手臂，30 Hz 控制頻率。256 個 bin 的離散 bin tokenization 每秒輸出多少個 token？7B VLM 跟得上嗎？

2. FAST tokenization 把 30 步軌跡壓縮成約 10 個 token。如果軌跡有高頻動作（例如打鼓），使用者會損失什麼？

3. π0 的 flow-matching head 約用 5 步去噪。把吞吐量與 OpenVLA 4-5 Hz 的 autoregressive 解碼比較。

4. GR00T 的 System 1 / System 2 拆分對應到 Kahneman。提出一種不同的拆分（System 3？），可能對雙足行走有幫助。

5. 閱讀 Open X-Embodiment 第 4 節關於資料集整理的內容。說出防止領域洩漏的三條整理規則。

## 關鍵術語 (Key Terms)

| 術語 (Term) | 一般說法 (What people say) | 實際意義 (What it actually means) |
|------|-----------------|------------------------|
| VLA | "Vision-language-action" | 接收圖片 + 指令並輸出動作命令的模型 |
| Action tokenization | "Discrete bins" | 把連續的關節目標量化成每個維度 256 個 bin，每個 bin 是一個詞彙表 ID |
| FAST tokenizer | "Frequency action tokens" | DCT + 量化，把 30 步軌跡壓縮成約 10 個 token |
| Co-fine-tune | "Mix web + robot" | 在機器人示範之外一併訓練網路 VQA 資料，以保留通用知識 |
| Flow-matching action head | "π0 continuous output" | 透過 rectified flow 輸出 50 步動作序列的小型 transformer |
| System 1 / System 2 | "Dual-system control" | 大型 VLM 慢速規劃，小型動作 head 快速行動；GR00T 的模式 |
| Open X-Embodiment | "RT-X dataset" | 100 萬條軌跡的跨機器人資料集；訓練語料庫 |

## 延伸閱讀 (Further Reading)

- [Brohan et al. — RT-2 (arXiv:2307.15818)](https://arxiv.org/abs/2307.15818)
- [Kim et al. — OpenVLA (arXiv:2406.09246)](https://arxiv.org/abs/2406.09246)
- [Black et al. — π0 (arXiv:2410.24164)](https://arxiv.org/abs/2410.24164)
- [NVIDIA — GR00T N1 (arXiv:2503.14734)](https://arxiv.org/abs/2503.14734)
- [Open X-Embodiment Collab — RT-X (arXiv:2310.08864)](https://arxiv.org/abs/2310.08864)
