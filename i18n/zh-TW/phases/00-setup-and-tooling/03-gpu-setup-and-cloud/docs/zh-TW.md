# GPU 設定與雲端 (GPU Setup & Cloud)

> 學習時用 CPU 訓練就夠了。真正的訓練需要 GPU。

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 0, Lesson 01
**Time:** ~45 分鐘

## 學習目標 (Learning Objectives)

- 用 `nvidia-smi` 和 PyTorch 的 CUDA API 驗證本機 GPU 是否可用
- 設定 Google Colab 的 T4 GPU，免費做雲端實驗
- 對 CPU 與 GPU 的矩陣乘法做 benchmark，並量測加速倍數
- 用 fp16 的經驗法則，估算 VRAM 放得下多大的模型

## 問題 (The Problem)

Phase 1 到 3 的大多數課程用 CPU 就能跑。但開始訓練 CNN、transformer 或 LLM（phase 4 以後）時，就需要 GPU 加速。一次在 CPU 上要跑 8 小時的訓練，在 GPU 上只要 10 分鐘。

你有三個選擇：本機 GPU、雲端 GPU，或 Google Colab（免費）。

## 核心概念 (The Concept)

```
Your options:

1. Local NVIDIA GPU
   Cost: $0 (you already have it)
   Setup: Install CUDA + cuDNN
   Best for: Regular use, large datasets

2. Google Colab (free tier)
   Cost: $0
   Setup: None
   Best for: Quick experiments, no GPU at home

3. Cloud GPU (Lambda, RunPod, Vast.ai)
   Cost: $0.20-2.00/hr
   Setup: SSH + install
   Best for: Serious training, large models
```

```figure
s0-gpu-dispatch
```

## 動手做 (Build It)

### 選項 1：本機 NVIDIA GPU

先確認你有沒有：

```bash
nvidia-smi
```

安裝支援 CUDA 的 PyTorch。在 Linux 上，預設的 wheel 就內建 CUDA。在 Windows 上，請從 pytorch.org 的選擇器複製 CUDA 指令。

```bash
pip install torch torchvision torchaudio
```

接著確認 PyTorch 看得到 GPU：

```python
import torch

print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
```

### 選項 2：Google Colab

1. 前往 [colab.research.google.com](https://colab.research.google.com)
2. Runtime > Change runtime type > T4 GPU
3. 執行 `!nvidia-smi` 來驗證

可以直接把這門課的 notebook 上傳到 Colab。

### 選項 3：雲端 GPU

使用 Lambda Labs、RunPod 或 Vast.ai 時：

```bash
ssh user@your-gpu-instance

pip install torch torchvision torchaudio
python -c "import torch; print(torch.cuda.get_device_name(0))"
```

### 沒有 GPU？沒問題。

大多數課程用 CPU 就能跑。需要 GPU 的課程會特別註明，並附上 Colab 連結。

```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using: {device}")
```

## 動手做 (Build It)：GPU 與 CPU 的 benchmark

```python
import torch
import time

size = 5000

a_cpu = torch.randn(size, size)
b_cpu = torch.randn(size, size)

start = time.time()
c_cpu = a_cpu @ b_cpu
cpu_time = time.time() - start
print(f"CPU: {cpu_time:.3f}s")

if torch.cuda.is_available():
    a_gpu = a_cpu.to("cuda")
    b_gpu = b_cpu.to("cuda")

    torch.cuda.synchronize()
    start = time.time()
    c_gpu = a_gpu @ b_gpu
    torch.cuda.synchronize()
    gpu_time = time.time() - start
    print(f"GPU: {gpu_time:.3f}s")
    print(f"Speedup: {cpu_time / gpu_time:.0f}x")
```

## 練習 (Exercises)

1. 執行上面的 benchmark，比較 CPU 與 GPU 的時間
2. 如果你沒有 GPU，改在 Google Colab 上執行並比較
3. 查看你的 GPU 記憶體有多少，並估算最大能放下的模型（經驗法則：fp16 每個參數 2 bytes）

### 提示 (Hints)

**練習 1。** 原封不動地執行「動手做」的程式碼，記下 CPU 時間、GPU 時間和 speedup 那一行。再跑一次；第一次的 GPU 執行包含啟動成本，所以要比較的是第二次的數字。

**練習 2。** 在 Colab 選擇 Runtime > Change runtime type > GPU，再把 benchmark 貼進一個 cell。先在另一個 cell 執行 `!nvidia-smi`；它必須顯示 GPU 名稱，benchmark 的結果才有意義。

**練習 3。** 用 `torch.cuda.get_device_properties(0).total_memory / 1e9` 讀出總記憶體，就像 `code/gpu_check.py` 那樣。把以 bytes 計的記憶體除以 2，就是 fp16 能放的參數量；還要留一些餘裕，因為 activation 和 CUDA context 也會佔用記憶體。

## 關鍵術語 (Key Terms)

| 術語 | 大家怎麼說 | 實際意思 |
|------|----------------|----------------------|
| CUDA | 「GPU 程式設計」 | NVIDIA 的平行運算平台，讓你在 GPU 上執行程式 |
| VRAM | 「GPU 記憶體」 | GPU 上的 Video RAM，與系統 RAM 分開，會限制模型大小。 |
| fp16 | 「半精度」 | 16 位元浮點數，記憶體用量只有 fp32 的一半，準確度損失很小 |
| Tensor Core | 「快速矩陣硬體」 | 專門做矩陣乘法的 GPU 核心，比一般核心快 4 到 8 倍 |
