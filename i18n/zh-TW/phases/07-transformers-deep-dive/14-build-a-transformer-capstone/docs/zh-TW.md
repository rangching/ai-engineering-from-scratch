# 從零打造 Transformer：總結專題 (Build a Transformer from Scratch — The Capstone)

> 十三堂課。一個模型。沒有捷徑。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 01 through 13. Don't skip.
**Time:** ~120 分鐘

## 問題 (The Problem)

你讀過每一篇論文。你實作過 attention、multi-head 拆分、positional encoding、encoder 與 decoder block、BERT 與 GPT 的 loss、MoE、KV cache。現在要讓它們在真實任務上一起運作。

總結專題：在字元層級的語言模型任務上，端到端訓練一個小型 decoder-only transformer。它讀莎士比亞，也生成新的莎士比亞。它小到可以在筆電上 10 分鐘內訓練完。它又正確到只要換上更大的資料集、訓練更久，就能得到一個真正的 LM。

這是這門課的「nanoGPT」。它不是原創的：Karpathy 2023 年的 nanoGPT 教學是每個學生至少都會寫一次的參考實作。我們沿用它的形狀，並依照我們學過的內容重新打造。

## 核心概念 (The Concept)

![Transformer-from-scratch block diagram](../assets/capstone.svg)

附註解的架構：

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### 我們會交付什麼 (What we ship)

- `GPTConfig`：一個設定所有超參數的地方。
- `MultiHeadAttention`：causal、batched，並有選用的 Flash 風格路徑（PyTorch 的 `scaled_dot_product_attention`）。
- `SwiGLUFFN`：現代 FFN。
- `Block`：pre-norm、以 residual 包起來的 attention + FFN。
- `GPT`：embedding、堆疊的 block、LM head、generate()。
- 訓練迴圈，使用 AdamW、cosine LR、gradient clipping。
- 針對莎士比亞文字的字元層級 tokenizer。

### 我們不交付什麼 (What we don't ship)

- RoPE：在第 04 課以概念方式實作。這裡為了簡單，我們使用學習式 positional embedding。練習會要求你換成 RoPE。
- 生成期間的 KV cache：每個生成步驟都會對整個 prefix 重新計算 attention。比較慢，但比較簡單。練習會要求你加上 KV cache。
- Flash Attention：PyTorch 2.0+ 在輸入符合時會自動分派；我們使用 `F.scaled_dot_product_attention`。
- MoE：每個 block 只有單一 FFN。你在第 11 課看過 MoE。

### 目標指標 (Target metrics)

在 Mac M2 筆電上，4 層、4 個 head、d_model=128 的 GPT，在 `tinyshakespeare.txt` 上訓練 2,000 步：

- 訓練 loss 在約 6 分鐘內，從約 4.2（隨機）收斂到約 1.5。
- 取樣的輸出看起來有莎士比亞的樣子：古字、換行，以及像 "ROMEO:" 這樣的專有名稱會浮現。
- Val loss（保留的最後 10% 文字）與訓練 loss 貼近；在這個規模 / 預算下沒有 overfit。

```figure
n5-block-stack
```

## 動手做 (Build It)

這堂課使用 PyTorch。請安裝 `torch`（CPU 版本即可）。請看 `code/main.py`。這個腳本負責：

- 若 `tinyshakespeare.txt` 不存在就下載（或讀取本機副本）。
- Byte-level 的字元 tokenizer。
- 以 90/10 切分 train/val。
- 在支援的硬體上使用 bf16 autocast 的訓練迴圈。
- 訓練完成後取樣。

### 步驟 1：資料 (data)

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 個不重複字元。很小的詞彙。放得進 4 byte 的 vocab_size。沒有 BPE，也沒有 tokenizer 的麻煩。

### 步驟 2：模型 (model)

請看 `code/main.py`。這個 block 就是第 05 課的教科書版本：pre-norm、RMSNorm、SwiGLU、causal MHA。4/4/128 的參數量：約 800K。

### 步驟 3：訓練迴圈 (training loop)

取得一個隨機 batch，內容是長度 256 的 token 視窗。Forward。位移一格的 cross-entropy。Backward。AdamW 更新。記錄。重複。

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### 步驟 4：取樣 (sample)

給定 prompt，反覆 forward、從 top-p logit 取樣、附加上去，然後繼續。生成 500 個 token 後停止。

### 步驟 5：讀輸出 (read the output)

2,000 步之後：

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

不是莎士比亞，但有莎士比亞的樣子。對約 800K 的參數和筆電上的 6 分鐘來說，這是明確的勝利。

## 實際使用 (Use It)

這個總結專題是個參考架構。要把它變成真正可用的東西，有三個延伸：

1. **換 tokenizer。** 使用 BPE（例如 `tiktoken.get_encoding("cl100k_base")`）。詞彙大小從 65 跳到約 50,000。模型容量需要擴大來補償。
2. **在更大的語料庫上訓練。** 使用 `OpenWebText` 或 `fineweb-edu`（HuggingFace）。在單張 A100 上，為 125M 參數的 GPT 訓練 10B 個 token 約需 24 小時。
3. **加入 RoPE + KV cache + Flash Attention。** 下面的練習會帶你逐一完成。

最終你會得到一個 125M 參數、能生成流暢英文的 GPT。不是前沿模型。但相同的程式碼路徑（只是更大）正是 Karpathy、EleutherAI 和 Allen Institute 在 2026 年用來訓練研究用 checkpoint 的方法。

## 交付 (Ship It)

請看 `outputs/skill-transformer-review.md`。這個 skill 會針對前面 13 堂課，審查從零打造的 transformer 實作是否正確。

## 練習 (Exercises)

1. **簡單。** 執行 `code/main.py`。驗證你訓練出的模型在最後一步的驗證 loss 低於 2.0。把 `max_steps` 從 2,000 改成 5,000，val loss 還會持續改善嗎？
2. **中等。** 把學習式 positional embedding 換成 RoPE。在 `MultiHeadAttention` 內對 Q 和 K 套用旋轉。訓練並驗證 val loss 至少一樣低。
3. **中等。** 在取樣迴圈中實作 KV cache。分別在有與沒有 cache 的情況下生成 500 個 token。在筆電上，實際耗時應該改善 5 到 20 倍。
4. **困難。** 為模型加上第二個 head，預測下一個再下一個 token（MTP，來自 DeepSeek-V3 的 Multi-Token Prediction）。一起訓練。有幫助嗎？
5. **困難。** 把每個 block 的單一 FFN 換成 4 個 expert 的 MoE。Router + top-2 routing。觀察在啟動參數相同的情況下，val loss 如何變化。

## 關鍵術語 (Key Terms)

| 術語 | 一般說法 | 實際意義 |
|------|-----------------|-----------------------|
| nanoGPT | 「Karpathy 的教學 repo」 | 精簡的 decoder-only transformer 訓練程式碼，約 300 行；標準參考實作。 |
| tinyshakespeare | 「標準的玩具語料庫」 | 約 1.1 MB 的文字；2015 年以來每個字元層級 LM 教學都會用到。 |
| Tied embeddings | 「共用輸入 / 輸出矩陣」 | LM head 權重 = token embedding 矩陣的轉置；省參數，提升品質。 |
| bf16 autocast | 「訓練精度的技巧」 | forward/backward 用 bf16 執行，optimizer 狀態維持 fp32；2021 年以來的標準做法。 |
| Gradient clipping | 「防止尖峰」 | 把全域 gradient norm 限制在 1.0；避免訓練爆掉。 |
| Cosine LR schedule | 「2020 年之後的預設」 | LR 先線性上升（warmup），再以 cosine 形狀衰減到峰值的 10%。 |
| MFU | 「Model FLOP Utilization」 | 實際達到的 FLOPs / 理論峰值；2026 年 dense 達 40%、MoE 達 30% 算是很強。 |
| Val loss | 「保留資料的 loss」 | 在模型從未見過的資料上的 cross-entropy；用來偵測 overfit。 |

## 延伸閱讀 (Further Reading)

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — 經典的附註解實作。
