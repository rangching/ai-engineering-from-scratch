# 安全性：Secrets、API Key 輪替、Audit Logs、Guardrails (Security)

> 透過集中式 vault（HashiCorp Vault、AWS Secrets Manager、Azure Key Vault）消除 secret 散落。絕不把憑證存在設定檔、版本控制中的 env 檔、試算表裡。優先使用 IAM roles 而非靜態 key；CI/CD 使用 OIDC。AI-gateway 模式是 2026 年的解法：apps → gateway → 模型供應商，由 gateway 在 runtime 從 vault 拉取憑證。在 vault 中輪替，所有 app 幾分鐘內就會取得新值，不需要重新部署，也不會有 Slack 上「誰有新的 key」的訊息。輪替政策 ≤90 天；每次 commit 都用 TruffleHog / GitGuardian / Gitleaks 掃描。Zero-trust：MFA、SSO、RBAC/ABAC、短效 token、裝置狀態。PII 清除使用實體辨識，在轉發前遮罩 PHI/PII；一致的 tokenization（Mesh 做法）把敏感值對應到穩定的佔位符，讓 LLM 保留程式碼/關係的語意。網路 egress：LLM 服務放在專用的 VPC/VNet 子網路，僅允許 `api.openai.com`、`api.anthropic.com` 等；封鎖所有其他對外流量。2026 年的事件推手：Vercel 供應鏈攻擊，被入侵的 CI/CD 憑證讓數千個客戶部署的 env 變數外洩。

**Type:** Learn
**Languages:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Prerequisites:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Time:** ~60 分鐘

## 學習目標 (Learning Objectives)

- 列舉四個 secret 管理的反模式（版本控制中的設定檔、寫死的 env、試算表、靜態 key），並說出它們的替代方案。
- 解釋 AI-gateway 從 vault 拉取的模式，作為 2026 年的 production 標準。
- 實作具備一致 tokenization（相同的值 → 相同的佔位符）的 PII scrubber，使語意得以保留。
- 說出 2026 年 Vercel 供應鏈事件，以及它對 CI/CD 憑證衛生的啟示。

## 問題 (The Problem)

一位實習生 commit 了含有 API keys 的 `.env`。他們很快刪掉。這些 key 已經在 git 歷史裡了，GitGuardian 掃描抓到它，而你的輪替流程是「在 Slack 通知團隊、更新 40 個設定檔、重新部署所有服務」。8 小時後，你的服務一半已上線，一半還在等部署視窗。

另外，使用者 prompt 含有「我的 SSN 是 123-45-6789。」Prompt 送到 OpenAI。你有 BAA，但你內部的政策是在轉發前遮罩 PII。你沒做到。

還有，你的 EKS 叢集的 LLM pod 可以連到任何網際網路主機。有人透過對攻擊者控制網域的 DNS 查詢外洩資料。沒有任何東西擋住它。

LLM 服務的安全性必須處理這三個向量，並在事後證明做到了。由 vault 支撐的憑證。PII 清除。網路 egress 過濾。以 audit logs 作為證據軌跡。

## 核心概念 (The Concept)

### 集中式 vault + IAM-role 拉取

**Vault**：HashiCorp Vault、AWS Secrets Manager、Azure Key Vault、GCP Secret Manager。單一事實來源。

**IAM role**：app/gateway 透過其 IAM 身分驗證，而不是靜態 key。Vault 在 token 的有效期間內回傳 secret。

**AI-gateway 模式**：gateway 在 request 時從 vault 拉取 `OPENAI_API_KEY`。在 vault 輪替；下一個 request 就拿到新的 key。不需重新部署。

### 輪替政策 ≤ 90 天

所有 API keys、vault root tokens、CI/CD 憑證。盡可能自動輪替。手動輪替要記錄並追蹤。

### Secret 掃描

- **TruffleHog**：對 commit 做 regex + 熵值檢查。
- **GitGuardian**：商業，準確度高。
- **Gitleaks**：OSS，在 CI 中執行。

每次 commit 都執行。偵測到新 secret 時封鎖 PR。

### Zero-trust 立場

- 所有帳號強制 MFA。
- 透過 SAML/OIDC 的 SSO。
- 用 RBAC（角色式）或 ABAC（屬性式）做細粒度存取。
- 短效 token（數小時，不是數天）。
- 裝置狀態：只允許有磁碟加密的公司裝置。

### PII / PHI 清除

在 prompt 離開你的基礎設施之前：

1. 實體辨識（spaCy NER、Presidio、商業工具）。
2. 遮罩相符的實體：`"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`。
3. 一致的 tokenization（Mesh 做法）：相同的值對應到相同的佔位符，讓 LLM 保留關係。
4. 可選：對 LLM 回應做反向對映。

靜態 regex 過濾器抓基本模式；NER 抓更多。兩者並用。

### 輸入 + 輸出 guardrails

輸入：封鎖已知的 jailbreak、禁止的主題；每個使用者的 rate limit。

輸出：用 regex 清除洩漏的 secrets（API key 模式、拒絕情境中的 email 模式），並以分類器偵測政策違規。

### 網路 egress 白名單

LLM 服務放在專用子網路：
- 白名單：`api.openai.com`、`api.anthropic.com`、向量 DB endpoints、vault endpoints。
- 其他一律丟棄。
- DNS 透過僅允許清單的 resolver（避免 DNS tunneling 外洩）。

### Audit log

每次 LLM 呼叫的不可變 log，包含：
- 時間戳記。
- 使用者 / tenant。
- Prompt hash（為隱私不存原始 prompt）。
- 模型 + 版本。
- Token 數量。
- 成本。
- 回應 hash。
- 任何 guardrail 觸發。

依法規要求保留（SOC 2 一年、HIPAA 六年）。

### 2026 年的 Vercel 事件

供應鏈攻擊：被入侵的 CI/CD 憑證讓數千個客戶部署的 env 變數外洩。教訓：CI/CD 憑證等同於 prod。存在 vault 中。範圍要窄。積極輪替。

### 該記住的數字

- 輪替政策：≤ 90 天。
- 每次 commit 掃描：TruffleHog / GitGuardian / Gitleaks。
- Vercel 2026：CI/CD 憑證遭入侵 → 數千個客戶 env 變數外洩。
- Audit log 保留：SOC 2 = 1 年，HIPAA = 6 年。

```figure
i4-vault-rotation
```

## 實際使用 (Use It)

`code/main.py` 實作一個具備一致 tokenization 的玩具 PII scrubber，以及一個只能附加的 audit log。

## 交付 (Ship It)

這堂課產出 `outputs/skill-llm-security-plan.md`。給定法規範圍與現況，規劃 vault 遷移、scrubber、egress、audit log。

## 練習 (Exercises)

1. 執行 `code/main.py`。送出兩個參照同一個 SSN 的 prompt。確認兩者得到相同的佔位符。
2. 為呼叫 OpenAI + Anthropic + Weaviate 的 vLLM-on-EKS 部署設計網路 egress 政策。
3. 你在 git 歷史中發現一個 key（兩年前的）。正確的回應是什麼：輪替 key、清除歷史，還是兩者都做？說明理由。
4. 你的 audit log 每天成長 10 GB。設計保留分層（熱 30 天、溫 12 個月、冷 6 年）。
5. 辯論反向 tokenization（把真實的值替換回 LLM 回應中）相較於保持佔位符可見，是否值得這份複雜度。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|----------------|------------------------|
| Vault | 「secrets 儲存庫」 | 集中式憑證管理服務 |
| IAM role | 「身分式驗證」 | 由 app 擔任的角色；回傳短效憑證 |
| OIDC for CI/CD | 「雲端簽發的 token」 | CI 中沒有靜態 key，透過 OIDC 取得身分 |
| TruffleHog / GitGuardian / Gitleaks | 「secret 掃描器」 | 在 commit 時偵測 secret |
| RBAC / ABAC | 「存取控制」 | 角色式 vs 屬性式 |
| PII scrubbing | 「資料遮罩」 | 移除或 tokenize 敏感實體 |
| Consistent tokenization | 「穩定的佔位符」 | 相同的值 → 每次都是相同的 token |
| Mesh approach | 「Mesh tokenization」 | 保留語意的 tokenization 模式 |
| Egress whitelist | 「對外允許清單」 | 只有被允許的網域可以連到 |
| Audit log | 「不可變歷史」 | 供合規使用、只能附加的紀錄 |

## 延伸閱讀 (Further Reading)

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio)：PII 偵測與匿名化。
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
