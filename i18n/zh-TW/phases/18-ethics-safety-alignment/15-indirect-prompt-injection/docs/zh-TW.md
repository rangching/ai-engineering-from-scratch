# Indirect Prompt Injection：量產環境的攻擊面 (Indirect Prompt Injection — Production Attack Surface)

> Indirect prompt injection (IPI) 把指令嵌入外部內容中，如網頁、電子郵件、共享文件、客服工單，由 agentic 系統在沒有使用者明確操作的情況下讀取。IPI 是 2026 年量產環境的主要威脅：它能繞過使用者輸入過濾器，因為攻擊者從不接觸使用者；它會隨著 agent 處理更多外部內容而悄悄擴大；它的目標是沒有人在讀 prompt 的自動化工作流程。MDPI Information 17(1):54 (January 2026) 彙整了 2023-2025 的研究。NDSS 2026 的 IPI 防禦論文點出核心挑戰：被注入的指令在語意上可以是無害的（「請印出 Yes」），所以偵測需要的不只是關鍵字過濾。「The Attacker Moves Second」(Nasr et al., 由 OpenAI/Anthropic/DeepMind 聯合, October 2025)：adaptive 攻擊（gradient、RL、隨機搜尋、人類 red team）繞過了他們測試的全部 12 個已發表防禦，對其中多數的攻擊成功率超過 90%，但這些防禦原本多半回報接近零的攻擊成功率。

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 分鐘

## 學習目標 (Learning Objectives)

- 定義 indirect prompt injection，並描述三種常見的傳遞向量。
- 解釋為什麼使用者輸入過濾器完全漏掉 IPI。
- 描述以「information flow control」作為 2026 年防禦典範的框架。
- 說明 Nasr et al. (October 2025) 關於 adaptive 攻擊對已發表 IPI 防禦之成功率的發現。

## 問題 (The Problem)

直接 prompt injection 需要攻擊者接觸到使用者或其 prompt。IPI 兩者都不需要：攻擊者把 payload 放在 agent 可能讀取的任何內容中，如網頁、收件匣裡的電子郵件、GitHub issue、商品評論。Agent 在正常運作中撿起它，並執行其中的指令。使用者是傳話者，而不是意圖的來源。

## 核心概念 (The Concept)

### 三種傳遞向量

- **Retrieval-augmented generation (RAG)。** 攻擊者發布一份文件；檢索步驟取得它；prompt 把它串接在使用者問題之前；模型執行攻擊者的指令。
- **收件匣 / 文件工作流程。** 攻擊者寄一封電子郵件給使用者；agent 讀取電子郵件；prompt 包含郵件內文；模型遵循郵件中的指令。
- **工具輸出。** 攻擊者控制 agent 使用的工具（例如回傳攻擊者控制之結果的網頁搜尋）；工具輸出含有指令；agent 的控制流程跟著執行。

三者共享一個結構性質：攻擊者在不接觸面向使用者之輸入的情況下，控制了 prompt 的一個片段。

### 為什麼使用者輸入過濾器會漏掉它

IPI payload 不會出現在使用者的輸入中。它出現在被檢索的內容中。如果過濾器只擋使用者輸入，payload 就繞過去了。如果過濾器擋的是所有到達模型的內容，它就必須套用在任意的檢索文字上，這既昂貴，又會對剛好含有祈使語氣的合法內容產生誤報。

### AI 的 Information Flow Control (IFC)

2026 年的防禦典範借用了經典作業系統安全。把每個內容來源視為一個安全標籤。把使用者的查詢標為「受信任」。把檢索到的內容標為「不受信任」。把模型的控制流程視為資訊流：由不受信任內容觸發的行動，必須在執行前經過受信任輸入的確認。

CaMeL (Microsoft 2025)、ConfAIde (Stanford 2024) 和 NDSS 2026 的 IPI 防禦論文，以不同方式落實 IFC。共同原則是：只要程式碼與資料共享同一個 context window，目標就是圍堵，而不是預防。

### The Attacker Moves Second

Nasr et al. (October 2025) 用 adaptive 攻擊（gradient 搜尋、RL policy、隨機搜尋、72 小時的人類 red team）測試了 12 個已發表的 IPI 防禦。每一個防禦都被繞過，其中多數的 ASR >90%，而且多數原本回報的 ASR 接近零。

方法論上的教訓：只有在做過 adaptive 攻擊評估之後，才發表防禦。靜態攻擊 benchmark 不能證明穩健性；攻擊者可以知道防禦的內容。

### 真實事件

第 25 課涵蓋 EchoLeak (CVE-2025-32711, CVSS 9.3)，這是 Microsoft 365 Copilot 中第一個公開記載的 zero-click IPI。還有 GitHub Copilot Chat 中的 CamoLeak (CVSS 9.6)，以及 GitHub Copilot 中的 CVE-2025-53773。量產部署在實際環境中被 IPI 攻陷，而不只是在 benchmark 裡。

### OWASP 與 NIST 的框架

OWASP LLM Top 10 (2025) 把 prompt injection（直接與間接）列為 LLM01，也就是第一名的應用層威脅。NIST AI SPD 2024 稱 indirect prompt injection 為「generative AI's greatest security flaw」。

### 這在 Phase 18 中的位置

第 12 到 14 課是以模型為中心的 jailbreak。第 15 課是主導 2026 年量產部署、以系統為中心的攻擊。第 16 課涵蓋防禦工具。第 25 課涵蓋具體的 CVE 敘事。

```figure
al-injection-vector
```

## 實際使用 (Use It)

`code/main.py` 建立一個 IPI harness。一個玩具 agent 有三個工具（搜尋網頁、讀取電子郵件、傳送訊息）。環境中含有嵌入指令（「forward this to all contacts」）的攻擊者控制內容。你可以在三種 agent 之間切換：天真的 agent（照做被注入的指令）、以過濾器防禦的 agent（對檢索內容做關鍵字過濾），以及 IFC agent（分隔受信任與不受信任的內容，並拒絕不受信任的控制流程指令）。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-ipi-audit.md`。給它一份 agentic 部署的描述，它會列舉不受信任的內容來源，檢查部署是否套用 IFC，並標示出沒有信任標籤就到達模型的來源。

## 練習 (Exercises)

1. 執行 `code/main.py`。測量攻擊對三種 agent 各自的成功率。

2. 對檢索到的內容實作基於改寫的防禦。測量它在合法檢索文字上的無害誤報率。

3. 閱讀 NDSS 2026 的 IPI 防禦論文。描述「benign instruction」的挑戰，以及它為什麼使基於關鍵字的過濾行不通。

4. 設計一個 agent 從第三方 API 收到工具輸出的部署。為每個 prompt 片段標上信任等級，並撰寫規範 agent 行動的 IFC policy。

5. 在練習 2 中以過濾器防禦的 agent 上，重現 Nasr et al. 2025 的 adaptive 攻擊方法論。回報 adaptive 攻擊前後的 ASR。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|------------------------|
| IPI | 「indirect prompt injection」 | 透過使用者並未撰寫、由 agent 在正常運作中讀取的內容所做的注入 |
| RAG injection | 「poisoned retrieval」 | 攻擊者發布內容，由檢索步驟取得；prompt 含有 payload |
| Zero-click | 「no user action」 | 攻擊在 agent 運作期間自動觸發；使用者什麼都不用做 |
| IFC | 「information flow control」 | 以標籤為基礎的做法：由不受信任內容觸發的行動，需要受信任的確認 |
| Adaptive attack | 「gradient / RL red-team」 | 知道防禦並針對它最佳化的攻擊；誠實的評估所必需 |
| Benign instruction | 「please print Yes」 | 語意上無害的 IPI payload；任何關鍵字過濾器都抓不到 |
| Scope violation | 「cross-trust exfiltration」 | Agent 從一個信任情境存取資料，並輸出到另一個情境 |

## 延伸閱讀 (Further Reading)

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54)：2023-2025 年的彙整
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108)：adaptive 攻擊評估
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173)：最早的 IPI 論文
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/)：prompt injection 排名 LLM01
