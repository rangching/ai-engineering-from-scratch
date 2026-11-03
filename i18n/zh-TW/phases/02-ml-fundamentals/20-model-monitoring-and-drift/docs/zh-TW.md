# 監控正式環境中的模型 — Data Drift 與 Concept Drift (Monitoring Models in Production — Data Drift and Concept Drift)

> 已部署的模型失敗時不會有錯誤訊息。當世界逐漸偏離它的訓練資料時，它仍然持續回傳分數，只有監控才會顯示出這個落差。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 2, Lessons 09 (Model Evaluation), 16 (Anomaly Detection), and 19 (ML System Design)
**Time:** ~110 分鐘

## 學習目標 (Learning Objectives)

- 在四個層次上選擇監控訊號：系統健康、輸入、預測與結果
- 依聯合分布中哪一部分改變，區分 covariate drift、prior drift 與 concept drift
- 從零實作 Population Stability Index、two-sample Kolmogorov-Smirnov 統計量與 chi-square 檢定
- 比較固定的 reference window 與滑動的 current window，並說明視窗大小如何改變雜訊與延遲
- 設計警報策略，利用 effect size、多重檢定校正與持續性規則來限制誤報
- 判斷哪種 drift 需要重新訓練、哪種需要新標籤、哪種不需要採取行動

## 問題 (The Problem)

一個付款風險模型上線時有 95% 的 accuracy。兩個月後，一場行銷活動帶來了購物籃更大的 app 使用者。四個月後，詐欺者改變了手法，原本安全的付款開始變成 chargeback。

在這整個過程中，模型服務對每個請求都回傳 HTTP 200。延遲正常，日誌裡也沒有錯誤。第一個出問題的跡象，是一份關於 chargeback 的財務報告，而那時損害已經發生數週了。

軟體監控檢查服務是否有回應。模型監控檢查回答是否仍然正確，以及資料是否仍然與訓練模型的資料相似。這堂課為第二項工作建立統計方法、視窗與警報策略。你會在一個注入了兩種變化的合成資料流上執行它們，並觀察哪個 monitor 對哪種變化有反應。

## 核心概念 (The Concept)

### 要監控什麼 (What to monitor)

監控四個層次。前三層提供早期預警。只有第四層衡量正確性，而且它來得晚。

| 層次 | 訊號 | 何時可得 | 它回答的問題 |
|---|---|---|---|
| 系統健康 | p50 與 p99 延遲、錯誤率、吞吐量、記憶體 | 立即 | 服務有回應嗎？ |
| 輸入 | 每個 feature 的分布、缺失值比率、schema、新類別 | 立即 | 即時資料看起來像訓練資料嗎？ |
| 預測 | 分數分布、正例比率、預設答案的比例 | 立即 | 模型的行為和以前一樣嗎？ |
| 結果 | Accuracy、precision、校準、business metric | 標籤延遲之後 | 模型還正確嗎？ |

輸入監控是這堂課的主要主題，因為它在標籤到達之前就能運作。它有一個 demo 會顯示的限制：有些變化不會動到輸入。

### 三種 Drift (Three kinds of drift)

用兩種方式寫出輸入 `x` 與標籤 `y` 的聯合分布：

```math
P(x, y) = P(x) \cdot P(y \mid x) = P(y) \cdot P(x \mid y)
```

每種 drift 改變的因子不同：

| Drift | 什麼改變 | 什麼不變 | 付款範例 | 哪個 monitor 看得到 |
|---|---|---|---|---|
| Covariate drift | `P(x)` | `P(y \| x)` | 更多購物籃更大的 app 使用者 | 輸入 monitor |
| Prior drift | `P(y)` | `P(x \| y)` | 攻擊期間詐欺率加倍，而詐欺看起來一樣 | 預測與結果 monitor |
| Concept drift | `P(y \| x)` | 可以不變 | 同樣的購物籃現在有不同的詐欺風險 | 只有結果 monitor |

Gama 等人把 `P(y | x)` 的改變稱為 real concept drift。他們把 `P(x)` 改變、但 `P(y | x)` 不動的情況稱為 virtual drift。他們的綜述也依 drift 在時間上如何發生來分類：突然 (sudden)、漸進 (incremental)、逐步 (gradual)，或是一個會回來的概念。

Covariate drift 不一定有害。如果模型學好了 `P(y | x)`，它對每個輸入仍然是正確的。但當流量移入訓練資料很少的區域或靠近 decision boundary 的區域時，它仍然可能損失 accuracy。

### 延遲的標籤 (Delayed labels)

結果在預測之後才到達。延遲取決於問題：

| 問題 | 標籤何時到達 |
|---|---|
| 廣告點擊 | 數秒 |
| 未來 30 天的 churn | 30 天 |
| 信用卡詐欺 | 當 chargeback 到達時，通常是數週之後 |
| 貸款違約 | 數月或數年 |

在延遲期間，你有輸入與預測，但沒有結果。監控輸入與預測，等標籤到達時再 join 上去，並只在有標籤的部分計算結果指標。這個 demo 使用 500 個事件的標籤延遲，所以結果 monitor 永遠是在回頭看 500 個事件之前。

### Reference 與 Current Window (Reference and current windows)

每個 drift 檢定都比較兩個樣本：

- **Reference window。**訓練資料，或是你手動檢查過的一段正式環境期間。它保持固定，直到你重新訓練。
- **Current window。**最近的事件，可以是最後 `N` 個事件或最後 `T` 小時。

Sliding window 以小於其大小的步長向前移動，所以連續的視窗會重疊。Tumbling window 以其完整大小向前移動。視窗大小是一種取捨。小視窗反應快，但會把雜訊當成 drift。大視窗穩定，但報告變化的時間較晚。

Seasonality 會破壞單純的視窗。許多產品中，週一的流量和週六的流量不同。讓 reference 涵蓋一個完整的週期，或是將每個視窗與較早週期中的同一段期間比較。

### Population Stability Index

PSI 來自信用評分。把 reference 的值切成 bin，通常是在 reference 的十分位數處切成十個 bin，使每個 bin 包含 reference 的 10%。計算 current 值在每個 bin 中的比例。然後每個 bin 加總一項：

```math
\text{PSI} = \sum_{\text{bins}} (\text{current\_share} - \text{reference\_share}) \cdot \ln\left(\frac{\text{current\_share}}{\text{reference\_share}}\right)
```

比例為零的 bin 會讓 log 變成無限大，所以把每個比例截斷到一個很小的值，例如 0.0001。PSI 是對稱的。它等於從 current 到 reference 的 KL divergence 加上從 reference 到 current 的 KL divergence。

實務者用固定的區間解讀 PSI：低於 0.1 是穩定，0.1 到 0.25 是中度偏移，高於 0.25 是顯著偏移。把這些區間當成慣例，而不是檢定。PSI 隨 bin 的數量增加而變大，隨樣本大小增加而變小。在 demo 中，沒有 drift 的 500 個事件視窗，PSI 分數介於 0.01 和 0.04 之間。

### Kolmogorov-Smirnov 統計量 (Kolmogorov-Smirnov statistic)

Two-sample KS 統計量 `D` 是兩個經驗累積分布函數之間最大的垂直落差：

```math
D = \max_x \left\lvert F_{\text{reference}}(x) - F_{\text{current}}(x) \right\rvert
```

相同的樣本 `D` 為 0，完全不重疊的樣本 `D` 為 1。它不需要 bin。p-value 來自 Kolmogorov 分布，使用有效樣本大小 `n * m / (n + m)`。

樣本很大時，對一個沒人在乎的落差，p-value 也會趨近零。當每個視窗有一百萬個事件時，0.02 個標準差的偏移都是「顯著」的。把 `D` 本身當成 effect size，並只有在 `D` 與 p-value 都通過各自 threshold 時才發出警報。

### 類別型 Feature 的 Chi-square 檢定 (Chi-square test for categorical features)

對於類別型 feature（例如付款管道），建立一個有兩列（reference、current）、每個類別一欄的表格。每個格子的期望次數來自列與欄的總和：

```math
\begin{aligned}
\text{expected}[r][c] &= \text{row\_total}[r] \cdot \text{column\_total}[c] / \text{grand\_total} \\
\text{chi2} &= \sum_{\text{cells}} (\text{observed} - \text{expected})^2 / \text{expected}
\end{aligned}
```

對於 `k` 個類別，檢定有 `k - 1` 個自由度。只出現在 current window 中的類別會加入很大的一項，所以這個檢定對新值有反應。Cramér's V 是 effect size。對於兩列的表格，它等於 `chi2 / grand_total` 的平方根。

### 警報 Threshold 與誤報 (Alert thresholds and false alarms)

當沒有任何改變時，每個檢定都有一個誤報率 `alpha`。監控會執行許多檢定。在 `alpha = 0.05` 下，如果每小時檢查 50 個 feature，即使什麼都沒變，你預期每天也會有大約 60 次誤報。人們遠在那之前就不會再去讀警報了。

四條規則讓警報率保持有用：

1. **校正多重檢定。**Bonferroni 校正把 `alpha` 除以檢定的數量。這個 demo 對兩個檢定使用 0.01 的整體 `alpha`，所以每個檢定使用 0.005。
2. **要求 effect size。**只有在 PSI、`D` 或 Cramér's V 也通過 threshold 時才發出警報。
3. **要求持續性。**只有在連續 `k` 個 drift 視窗之後才發出警報。這個 demo 使用 `k = 2`。
4. **區分警告與警報。**警告送到 dashboard。警報送到人員手上。

這個 demo 展示了規則 2 的作用。有一個乾淨的視窗，chi-square 的 p-value 是 0.0009，低於每個檢定的 `alpha`。它的 Cramér's V 低於 0.1，所以策略記錄一個警告，而不是警報。

### 重新訓練的觸發條件 (Retraining triggers)

重新訓練有三種常見的觸發條件：

- **排程。**每週或每月重新訓練。簡單，但在沒有改變時重新訓練，在確實改變時卻在等待。
- **Drift。**當重要 feature 的輸入 drift 持續高於 threshold 時重新訓練。
- **效能。**當結果指標低於 reference 值減去容忍度時重新訓練。

Drift 的種類決定了重新訓練能做什麼：

| Drift | 有用的行動 |
|---|---|
| Covariate drift，accuracy 穩定 | 通常什麼都不用做。持續觀察結果指標。 |
| Covariate drift 進入資料很少的區域 | 用包含這些區域的近期資料重新訓練 |
| Prior drift | 重新校準分數，或移動決策 threshold |
| Concept drift | 先收集新標籤。在舊標籤上重新訓練只會再次教它舊的概念。 |

重新訓練之後，在最近的已標記資料上測試新模型。使用系統設計那堂課的 shadow 與 canary 階段來 rollout。然後把 reference window 重設為新的訓練資料。

```figure
mlprod-drift-psi
```

## 動手做 (Build It)

`code/main.py` 中的程式碼只使用標準函式庫。

### 步驟 1：Quantile bin 與 PSI

Bin 的邊界是 reference 的十分位數。`bisect_right` 找出每個值所屬的 bin。

```python
def quantile_edges(reference: list[float], bins: int = 10) -> list[float]:
    ordered = sorted(reference)
    n = len(ordered)
    edges = [ordered[min(n - 1, (n * i) // bins)] for i in range(1, bins)]
    return sorted(set(edges))


def psi_from_fractions(expected: list[float], actual: list[float], eps: float = 1e-4) -> float:
    total = 0.0
    for e, a in zip(expected, actual):
        e, a = max(e, eps), max(a, eps)
        total += (a - e) * math.log(a / e)
    return total
```

### 步驟 2：KS 統計量與它的 p-value

同時走訪兩個已排序的樣本。在每個相異的值，跳過它在兩個樣本中的所有副本，再比較兩個累積比例。這樣能正確處理相同值 (ties)。

```python
def ks_statistic(sample_a: list[float], sample_b: list[float]) -> float:
    a, b = sorted(sample_a), sorted(sample_b)
    n, m = len(a), len(b)
    i = j = 0
    d = 0.0
    while i < n and j < m:
        x = min(a[i], b[j])
        while i < n and a[i] <= x:
            i += 1
        while j < m and b[j] <= x:
            j += 1
        d = max(d, abs(i / n - j / m))
    return d
```

p-value 使用 Kolmogorov 分布的級數 `2 * sum((-1)^(k-1) * exp(-2 k^2 lambda^2))`，其中 `lambda = sqrt(n * m / (n + m)) * D`。

### 步驟 3：Chi-square 檢定

`chi2_sf` 透過正規化的上不完全 gamma 函數計算 chi-square 的 survival function。小的引數用級數，大的引數用連分數。`chi_square_test` 建立兩列的表格、計算統計量，並回傳 p-value 與 Cramér's V。

```python
def chi_square_test(reference_counts: dict[str, int], current_counts: dict[str, int]) -> ChiSquareResult:
    categories = [c for c in sorted(set(reference_counts) | set(current_counts))
                  if reference_counts.get(c, 0) + current_counts.get(c, 0) > 0]
    rows = [[reference_counts.get(c, 0) for c in categories], [current_counts.get(c, 0) for c in categories]]
    row_totals = [sum(r) for r in rows]
    grand = sum(row_totals)
    if len(categories) < 2 or 0 in row_totals:
        return ChiSquareResult(0.0, 0, 1.0, 0.0)
    statistic = 0.0
    for col, _ in enumerate(categories):
        col_total = rows[0][col] + rows[1][col]
        for r in range(2):
            expected = row_totals[r] * col_total / grand
            statistic += (rows[r][col] - expected) ** 2 / expected
    dof = len(categories) - 1
    return ChiSquareResult(statistic, dof, chi2_sf(statistic, dof), math.sqrt(statistic / grand))
```

### 步驟 4：視窗與 drift monitor

`SlidingWindow` 在 `deque` 中保留最後 `size` 個事件。`DriftMonitor` 只計算一次 reference 的 bin 與計數。它的 `check` 方法對每個數值型 feature 回傳 PSI、`D` 與 KS p-value，對每個類別型 feature 回傳 chi-square 結果。

### 步驟 5：警報策略

策略把每組指標轉成一個等級，並對每個訊號計算連續的 drift 視窗數：

```python
def level(self, metrics: dict[str, float]) -> str:
    if "psi" in metrics:
        if metrics["psi"] >= self.psi_drift or (metrics["ks"] >= self.ks_drift and metrics["ks_p"] < self.alpha):
            return "drift"
        return "warn" if metrics["psi"] >= self.psi_warn else "ok"
    if metrics["p"] < self.alpha and metrics["v"] >= self.min_cramers_v:
        return "drift"
    return "warn" if metrics["p"] < self.alpha else "ok"
```

`update` 在完成連續條件的那個視窗回傳 `ALERT`，並在警報之後的第一個乾淨視窗回傳 `resolved`。

### 步驟 6：注入 drift 的資料流

這個資料流有 9000 個付款事件，含有 `amount`、`channel` 與詐欺 `label`。由四個區段定義：

| 事件 | Amount | 管道比例 (app, store, web) | 標籤邊界 | 變化 |
|---|---|---|---|---|
| 0 到 2999 | mean 100, sd 20 | 35, 15, 50 percent | 120 | Reference 期間 |
| 3000 到 4999 | mean 118, sd 24 | 60, 10, 30 percent | 120 | Covariate drift |
| 5000 到 6499 | mean 100, sd 20 | 35, 15, 50 percent | 120 | 回到正常 |
| 6500 到 8999 | mean 100, sd 20 | 35, 15, 50 percent | 105 | Concept drift |

模型是對 `amount` 的 threshold，在前 2000 個事件上 fit。視窗包含 500 個事件，每次移動 250。標籤晚 500 個事件到達。

```bash
python3 code/main.py
```

```console
  end    PSI   KS D   chi2 p    acc  levels (amount/channel/accuracy)  events
 3000  0.010  0.041  3.0e-01  0.932  ok/ok/ok
 3250  0.152  0.163  7.6e-06  0.924  drift/warn/ok
 3500  0.463  0.298  2.8e-26  0.936  drift/drift/ok                     ALERT amount
 3750  0.609  0.343  3.5e-35  0.942  drift/drift/ok                     ALERT channel
 ...
 5500  0.026  0.044  2.6e-01  0.898  ok/ok/ok                           resolved amount, resolved channel
 ...
 7250  0.040  0.048  7.1e-01  0.848  ok/ok/drift
 7500  0.022  0.039  6.1e-01  0.778  ok/ok/drift                        ALERT accuracy
```

分三部分解讀結果：

- **Covariate drift。**Drift 從事件 3000 開始，`amount` 的警報在 3500 跟上。Accuracy 維持在 0.89 到 0.95 之間，因為標籤規則沒有改變。
- **恢復。**輸入在 5000 回到正常，兩個警報在 5500 解除。
- **Concept drift。**Drift 從 6500 開始，輸入 monitor 保持安靜，因為輸入沒有改變。Accuracy 從 0.946 掉到 0.76 到 0.79 之間。Accuracy 警報在 7500 出現，比變化晚了 1000 個事件，其中 500 個是標籤延遲造成的。

## 實際使用 (Use It)

**scipy。**`scipy.stats` 實作了這兩種檢定。統計量與從零實作的程式碼一致。執行這堂課不需要 scipy。

```python
from scipy import stats

result = stats.ks_2samp(reference_amounts, current_amounts)
print(result.statistic, result.pvalue)

table = [[1000, 700, 300], [140, 290, 70]]
chi2 = stats.chi2_contingency(table)
print(chi2.statistic, chi2.pvalue, chi2.dof)
```

我們在隨機樣本上，將 `ks_statistic` 和 `chi_square_test` 與 scipy 1.18.1 比對。KS 統計量、chi-square 統計量與 chi-square p-value 在浮點數精度內一致。在 200 組隨機樣本對中，KS 的 p-value 與 scipy 的差異最多達到 scipy 值的 18%。最大的落差出現在非常小的 p-value。

這個落差來自方法。我們的程式碼使用極限的 Kolmogorov 分布。預設情況下，當兩個樣本都最多有 10,000 個值時，`ks_2samp` 使用精確方法。它的 asymptotic 模式使用帶有四捨五入後有效樣本大小的有限樣本分布。在只有兩個類別時，`chi2_contingency` 預設會套用 Yates 連續性校正，所以統計量在那裡有所不同。

**監控函式庫。**幾個開源函式庫把這些檢定與報告和 dashboard 打包在一起：

- **Evidently** 為每個欄位挑選 drift 檢定。對於 1000 列以內的 reference，它對數值欄位使用 KS 檢定，對類別欄位使用 chi-square 檢定。對於更大的資料，它使用 Wasserstein 距離與 Jensen-Shannon divergence，threshold 為 0.1。PSI 是你可以選擇的其他方法之一。
- **NannyML** 在標籤到達之前估計模型效能。它的 Confidence-based Performance Estimation (CBPE) 使用模型預測的機率，而這些機率必須校準得很好。NannyML 文件指出，CBPE 能處理 covariate shift，但無法應付 concept drift。這個 demo 資料流顯示了原因：只有標籤才揭露了 concept drift。
- **TensorFlow Data Validation** 比較連續的資料區段 (span)。它對類別型 feature 使用 L-infinity 距離，對數值型 feature 使用近似的 Jensen-Shannon divergence。它也會比較 training 資料與 serving 資料，這在 feature store 那堂課中會涵蓋。

從零實作的程式碼顯示了這些工具各自計算的內容。這些工具加上了儲存、排程、dashboard 與更多的檢定。

對於 LLM 應用，基礎設施階段中的 [LLM observability lesson](../../../17-infrastructure-and-production/13-llm-observability/docs/en.md) 涵蓋了 tracing 與評估工具。這堂課的想法仍然適用於那些系統的輸入與輸出。

## 交付 (Ship It)

這堂課會產出 `outputs/skill-drift-monitoring.md`。這個 skill 為新模型規劃 monitor，並對 drift 警報做分流 (triage)。它會問哪個層次觸發了、哪種 drift 符合證據，以及正確的行動是等待、重新校準、收集標籤，還是重新訓練。

## 練習 (Exercises)

1. 用 200 和 2000 個事件的視窗執行這個資料流。記錄事件 3000 之前的誤警告與警報延遲。
2. 設定 `n_tests=1`、`patience=1` 與 `min_cramers_v=0`。計算 5000 到 6500 這段乾淨期間內的警報數量。
3. 加入一個 feature，其平均值每 100 個事件成長 0.01。找出哪個訊號（PSI 或 KS）先有反應。
4. 在相同的十分位 bin 上實作 Jensen-Shannon 距離。在每個視窗上將它與 PSI 比較。
5. 為模型的正向預測率加入一個 monitor。解釋為什麼它在 concept drift 期間保持安靜。

## 關鍵術語 (Key Terms)

| 術語 | 大家怎麼說 | 實際意義 |
|------|----------------|----------------------|
| Data drift | 「資料變了」 | 與 reference window 相比，輸入、預測或標籤的分布發生改變 |
| Covariate drift | 「新型態的使用者」 | `P(x)` 改變，而 `P(y \| x)` 保持不變 |
| Prior drift | 「正例比以前多」 | `P(y)` 改變，而 `P(x \| y)` 保持不變，也稱為 label shift |
| Concept drift | 「遊戲規則變了」 | `P(y \| x)` 改變，所以同樣的輸入現在有不同的正確答案 |
| Reference window | 「基準資料」 | 固定的樣本，通常是訓練資料，每個 current window 都與它比較 |
| PSI | 「穩定度分數」 | 對各 bin 的（current 比例減 reference 比例）乘以兩者比值的 log 所做的對稱加總 |
| KS statistic | 「KS 檢定」 | 兩個經驗累積分布函數之間最大的落差 |
| Cramér's V | 「類別的 effect size」 | 縮放到 0 到 1 範圍的 chi-square 統計量 |
| Label delay | 「真實答案晚點才來」 | 從預測到其真實結果到達之間的時間 |
| Alert fatigue | 「通知太多」 | 人們忽略警報，因為其中大多數都是誤報 |

## 延伸閱讀 (Further Reading)

- [Gama et al., A Survey on Concept Drift Adaptation (ACM Computing Surveys, 2014)](https://research.tue.nl/en/publications/a-survey-on-concept-drift-adaptation)：real 與 virtual drift、drift 模式與適應方法
- [Rabanser, Günnemann, and Lipton, Failing Loudly: An Empirical Study of Methods for Detecting Dataset Shift (NeurIPS 2019)](https://arxiv.org/abs/1810.11953)：用於偵測 shift 的 two-sample 檢定
- [Lu et al., Learning under Concept Drift: A Review (2020)](https://arxiv.org/abs/2004.05785)：drift 的偵測、理解與適應方法
- [Yurdakul, Statistical Properties of Population Stability Index (Western Michigan University, 2018)](https://scholarworks.wmich.edu/dissertations/3208/)：PSI 如何隨樣本大小與 bin 數量而表現
- [scipy.stats.ks_2samp documentation](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.ks_2samp.html)：two-sample KS 檢定的精確與 asymptotic p-value 方法
- [Evidently, How data drift detection works](https://docs.evidentlyai.com/metrics/explainer_drift)：依欄位類型與資料大小的預設檢定選擇
- [NannyML, Performance estimation](https://nannyml.readthedocs.io/en/stable/how_it_works/performance_estimation.html)：CBPE 及其假設
