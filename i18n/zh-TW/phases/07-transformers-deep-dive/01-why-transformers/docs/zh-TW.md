# 為什麼需要 Transformer：RNN 的問題 (Why Transformers — The Problems with RNNs)

> RNN 一次處理一個 token。Transformer 一次處理所有 token。就是這個架構上的賭注，改變了 2017 年之後深度學習所有的 scaling 曲線。

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 · 09 (Sequence-to-Sequence), Phase 5 · 10 (Attention Mechanism)
**Time:** ~45 分鐘

## 問題 (The Problem)

2017 年以前，全世界最先進的序列模型，不管是語言、翻譯還是語音，全都是 recurrent neural network。LSTM 和 GRU 在翻譯基準上稱霸了五年，因為當時大家手上只有這個工具。

它們有三個致命弱點。第一，循序計算代表你無法沿著時間軸平行化：token `t+1` 需要 token `t` 的 hidden state。1,024 個 token 的序列，就是在每個 cycle 可以做 1,000,000 次浮點運算的 GPU 上，跑 1,024 個串行步驟。在專為平行化設計的硬體上，訓練的實際耗時卻隨序列長度線性成長。

第二，vanishing gradient 代表 50 個 token 之前的資訊，已經被 50 層非線性壓縮過了。Gated recurrent unit（LSTM、GRU）減輕了這種壓縮，卻從未消除它。長距離依賴，例如「去年夏天我在飛往京都的飛機上讀的那本書是……」，經常失敗。

第三，固定寬度的 hidden state 代表 encoder 要把整個來源序列擠進單一向量，decoder 才看得到東西。來源是 5 個 token 或 500 個 token 都沒差，瓶頸的形狀是一樣的。

2017 年的論文 "Attention Is All You Need" 提出了一個激進的做法：完全拿掉 recurrence。讓每個位置平行地 attend 到其他所有位置。用一次大型矩陣乘法來訓練，取代 1,024 次循序運算。

到了 2026 年，這個做法已經主宰所有模態。語言（GPT-5、Claude 4、Llama 4）、視覺（ViT、DINOv2、SAM 3）、音訊（Whisper）、生物（AlphaFold 3）、機器人（RT-2）。同一種 block，不同的輸入。

## 核心概念 (The Concept)

![RNN sequential compute vs Transformer parallel attention](../assets/rnn-vs-transformer.svg)

**Recurrence 是瓶頸。** RNN 計算 $h_t = f(h_{t-1}, x_t)$。每一步都依賴前一步。你沒辦法在 $h_4$ 之前算出 $h_5$。在擁有 10,000 個以上平行核心的現代 GPU 上，長序列會浪費 99% 的矽晶片。

**Attention 是廣播。** Self-attention 對每一對 $(i, j)$ 同時計算 `output_i = sum_j(a_ij * v_j)`。整個 N×N 的 attention 矩陣用一次 batched matmul 就填滿。沒有任何一步依賴另一步。GPU 超愛這種做法。

**加速不是常數。** 差別在於 $O(N)$ 的串行深度與 $O(1)$ 的串行深度。實務上，在同等硬體、N=512 的條件下，transformer 每個 epoch 的訓練速度快 5 到 10 倍，而且序列越長差距越大，直到撞上 attention 的 $O(N^2)$ 記憶體牆（後來 Flash Attention 解決了這個問題，見第 12 課）。

**Transformer 的代價。** Attention 的記憶體用量以 $O(N^2)$ 成長。2K 的 context 沒問題；128K 的 context 就需要 sliding window、RoPE extrapolation、Flash Attention tiling，或是 linear attention 的變體。Recurrence 在時間與記憶體上都是 $O(N)$；transformer 用時間換記憶體，再靠平行化把時間贏回來。

**Inductive bias 的轉變。** RNN 假設局部性與近期性。Transformer 什麼都不假設，每一對位置都是 attention 的候選。這就是為什麼 transformer 需要更多資料才能訓練得好，但一旦資料足夠，它能 scale 得更遠。Chinchilla（2022）把這點形式化了：只要 token 夠多，transformer 永遠勝過參數量相同的 RNN。

```figure
rnn-vs-parallel
```

## 動手做 (Build It)

這裡沒有神經網路。我們用數值方式模擬核心瓶頸，讓你在自己的筆電上就能感受到差距。

### 步驟 1：測量串行深度 (measure serial depth)

請看 `code/main.py`。我們建立兩個函式。一個把序列編碼成加法鏈（串行，像 RNN）。另一個把它編碼成平行 reduction（廣播，像 attention）。數學一樣，相依圖不同。

```python
def rnn_style(xs):
    h = 0.0
    for x in xs:
        h = 0.9 * h + x   # can't parallelize: h depends on previous h
    return h

def attention_style(xs):
    return sum(xs) / len(xs)  # every x is independent
```

我們在最長 100,000 個元素的序列上，替兩者計時。RNN 版本是 $O(N)$，而且只用單一 CPU pipeline。即使是純 Python，attention 風格的 reduction 在長度 ≥ 1,000 時也會勝出，因為 Python 的 `sum()` 是用 C 實作的，每一步都沒有直譯器的額外開銷。

### 步驟 2：計算理論運算量 (count theoretical operations)

兩種演算法都做 N 次加法。差別在於 *dependency depth*：下一個運算開始之前，必須先循序完成多少個運算。RNN 的深度 = N。Attention 的深度，用 tree reduction 是 log(N)，用 parallel scan 則是 1。決定 GPU 時間的是深度，不是運算次數。

### 步驟 3：長序列的實測 scaling (empirical scaling on long sequences)

我們會印出一張計時表，讓 $O(N)$ 的差距一目了然。在 2026 年的 Mac 筆電上，少於 1,000 個元素的序列快到量不出來。100,000 個元素的序列則顯示出乾淨的線性掃描。把它放大到 16,384 個 token 的 transformer，對上 12 層 LSTM 的等效模型，你就會明白為什麼 2016 年訓練的實際耗時是個阻礙。

## 實際使用 (Use It)

2026 年還值得選 RNN 的情況：

| 情境 | 選擇 |
|-----------|------|
| 串流推論，一次一個 token，記憶體固定 | RNN 或 state-space model（Mamba、RWKV） |
| 超長序列（>1M token），attention 記憶體會爆掉 | Linear attention、Mamba 2、Hyena |
| 沒有 matmul 加速器的邊緣裝置 | Depthwise-separable RNN 在每瓦 FLOPs 上仍然勝出 |
| 其他所有情況（訓練、batched 推論、最高 128K 的 context） | Transformer |

像 Mamba 這類 state-space model（SSM），本質上是帶有結構化參數化的 RNN，兩邊的優點都拿到：$O(N)$ 的 scan 記憶體，加上透過 selective scan 達成的平行訓練。它們能達到 transformer 九成的品質，而且長 context 的 scaling 更好。2026 年多數前沿實驗室都在訓練 SSM+transformer 的混合模型（例如 Jamba、Samba）。Recurrence 沒有死，它只是變成了一個元件。

## 交付 (Ship It)

請看 `outputs/skill-architecture-picker.md`。這個 skill 會根據長度、吞吐量與訓練預算的限制，為新的序列問題挑選架構。對於超過 1B token 的訓練，在沒有說明取捨的情況下，它一律不該建議使用純 RNN。

## 練習 (Exercises)

1. **簡單。** 把 `code/main.py` 裡的 `rnn_style` 的純量 hidden state，換成長度 64 的 hidden state 向量。重新測量。串行的額外開銷會隨 hidden state 維度成長多少？
2. **中等。** 用純 Python 實作 parallel prefix-sum（Hillis-Steele scan）。驗證它在長度 1024 時，數值輸出與串行 scan 相同。算出它的深度。
3. **困難。** 把 attention 風格的 reduction 移植到 GPU 上的 PyTorch。在序列長度從 64 掃到 65,536 的過程中，替兩者計時。畫出圖並解釋曲線的形狀。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| Recurrence | 「RNN 是循序的」 | 第 `t` 步依賴第 `t-1` 步的計算，迫使沿著時間軸串行執行。 |
| Serial depth | 「圖有多深」 | 相依運算的最長鏈；即使硬體無限多，也決定了實際耗時的下限。 |
| Attention | 「讓 token 互相看見彼此」 | 加權和 $\sum_j a_{ij} v_j$，其中 $a_{ij}$ 來自位置 $i$ 與 $j$ 之間的相似度分數。 |
| Context window | 「模型能看到多少」 | Attention 層可以當作輸入的位置數量；二次方的記憶體成本就落在這裡。 |
| Inductive bias | 「內建在架構裡的假設」 | 對資料長相的先驗；CNN 假設平移不變性，RNN 假設近期性。 |
| State-space model | 「背後有代數的 RNN」 | 透過結構化 state-space 矩陣，為平行訓練而參數化的 recurrence。 |
| Quadratic bottleneck | 「為什麼 context 這麼貴」 | Attention 記憶體 = 序列長度的 $O(N^2)$；Flash Attention 藏起了常數，沒有改變 scaling。 |

## 延伸閱讀 (Further Reading)

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — 在主流 NLP 中終結 recurrence 的那篇論文。
- [Bahdanau, Cho, Bengio (2014). Neural MT by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — attention 誕生的地方，當時是接在 RNN 上。
- [Hochreiter, Schmidhuber (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — 最初的 LSTM 論文，留個紀錄。
- [Gu, Dao (2023). Mamba: Linear-Time Sequence Modeling with Selective State Spaces](https://arxiv.org/abs/2312.00752) — 對 transformer 的現代 recurrent 回應。
