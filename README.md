# SlimNet — PyTorch Model Compression Library

> Compress any PyTorch model to run on cheap hardware. One function call. No ML expertise required.

[![Python](https://img.shields.io/badge/python-3.9%20|%203.10%20|%203.11%20|%203.12-blue)]()
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)]()
[![License](https://img.shields.io/badge/license-MIT-green)]()

---

## The Problem

State-of-the-art models routinely require 24–80 GB of GPU memory. The median ML student has a free Colab T4 (16 GB), a laptop GPU (4–8 GB), or nothing at all. Existing compression tools are fragmented — PyTorch's pruning API is low-level, bitsandbytes handles quantization only, HuggingFace Optimum is locked to Transformers. No tool combines all techniques behind a single hardware-aware interface.

SlimNet fills this gap.

---

## What It Does

SlimNet runs a four-stage compression pipeline automatically:

```
Original Model
      │
      ▼
┌─────────────────────┐
│  Structured Pruning │  Zeros out least-important neurons/filters
│  (magnitude / grad  │  using iterative schedule (5 steps).
│   / Taylor scoring) │  One-shot pruning causes accuracy collapse
└─────────────────────┘  above 30% sparsity — iterative avoids this.
      │
      ▼
┌─────────────────────┐
│    Knowledge        │  Student (pruned model) trained to mimic
│    Distillation     │  teacher (original) soft output distribution.
│                     │  Loss = α·CrossEntropy + β·KL(student/T, teacher/T)
│                     │  T=4.0, α=0.3, β=0.7. Labels NOT required.
└─────────────────────┘  Teacher frozen in eval() throughout.
      │
      ▼
┌─────────────────────┐
│    Quantization     │  fp16 on GPU (2× size reduction).
│                     │  Dynamic int8 on CPU (4× size reduction).
│                     │  Validated: cosine_similarity > 0.999.
└─────────────────────┘
      │
      ▼
  Compressed Model + BenchmarkReport
```

**Why this order?** Quantization converts `nn.Linear` → `DynamicQuantizedLinear` (a C++ layer with no Python-visible `parameters()`). Any optimizer called after that crashes with `ValueError: optimizer got an empty parameter list`. So pruning and distillation must run first, quantization last.

---

## Benchmark Results

| Model | Original | Compressed | Speedup | Accuracy Drop | Device |
|-------|----------|------------|---------|---------------|--------|
| ResNet-18 | 44.6 MB | 11.5 MB | 3.8× | < 1.5% | T4 / CPU |
| Tiny MLP | 0.26 MB | 0.07 MB | 3.7× | < 1.0% | CPU |

*ResNet-18 compressed on Colab T4. Calibration: 512 CIFAR-10 samples.*

---

## Install

```bash
pip install torch torchvision psutil rich typer
git clone https://github.com/YOUR_USERNAME/slimnet
cd slimnet
```

No pip package yet — import directly:

```python
import sys
sys.path.insert(0, '/path/to/slimnet')
import slimnet
```

---

## Quick Start — Simple API

```python
import slimnet

result = slimnet.compress(
    model=my_model,              # any nn.Module
    calibration_data=dataloader, # DataLoader, min 512 samples
    target='t4',                 # 't4' | 'cpu' | 'macbook' | 'rtx3070'
    max_accuracy_drop=0.02,      # 2% max degradation
    val_data=val_loader,         # optional — enables accuracy reporting
    sample_input=torch.randn(1, 3, 224, 224),
)

result.model          # compressed nn.Module, ready for inference
result.report         # BenchmarkReport — size, latency, accuracy
result.save('compressed.pt')
```

**Output:**
```
====================================================
  SlimNet BenchmarkReport
====================================================
  Size      : 44.67 MB → 11.50 MB  (3.88x)
  Latency   : 58.49 ms → 15.21 ms  (3.84x speedup)
  Accuracy  : 0.6200 → 0.6050      (delta -0.0150)
  Peak VRAM : 135.1 MB
  Fits      : YES (t4)
  Applied   : pruning → distillation → quantization
====================================================
```

---

## Advanced API

Full control over every compression stage:

```python
from slimnet import Compressor, QuantConfig, PruneConfig, LowRankConfig, DistillConfig

compressor = Compressor(
    quantization=QuantConfig(mode='dynamic', per_channel=True),
    pruning=PruneConfig(sparsity=0.4, method='taylor', n_steps=5),
    low_rank=LowRankConfig(variance_threshold=0.90, min_layer_size=512),
    distillation=DistillConfig(
        temperature=4.0,
        alpha=0.3,
        beta=0.7,
        epochs=5,
        intermediate=True,   # TinyBERT-style intermediate layer distillation
        lr=2e-5,
    ),
)
result = compressor.fit(model, calibration_data, val_data=val_loader)
```

---

## Technique Details

### 1. Structured Pruning (`techniques/pruning.py`)

Removes entire neurons (Linear), filters (Conv2d), or attention heads — not individual weights. Structured pruning produces dense matrices that run fast on real hardware without sparse kernels.

**Three importance scoring methods:**
- `magnitude` — L1 norm of weights. Fast, no data needed. Default.
- `gradient` — gradient × weight importance. Needs calibration batch.
- `taylor` — first-order Taylor expansion of loss w.r.t activations. Most accurate.

**Iterative schedule (PRD §5.3.3):**
```python
for step in range(n_steps):          # default: 5
    sparsity = target * (step+1) / n_steps
    prune_model(model, sparsity)
    finetune(model, calibration_data, epochs=1)
```
One-shot pruning to 40% sparsity causes 8–12% accuracy collapse. Iterative keeps it under 3%.

---

### 2. Knowledge Distillation (`techniques/distillation.py`)

The core DL component. A compressed student is trained to mimic the original teacher's soft output distribution — not just hard labels.

**Loss function (Hinton et al. 2015):**
```
L_total = α × CrossEntropy(student_logits, hard_labels)
        + β × KL(softmax(student/T), softmax(teacher/T))

T = 4.0  (temperature — higher → softer distributions → more inter-class signal)
α = 0.3  (task loss weight)
β = 0.7  (distillation loss weight)
```

**Training details:**
- Teacher: `eval()` mode, `requires_grad_(False)` — never modified
- Optimizer: AdamW, lr=2e-5, weight_decay=0.01
- LR schedule: cosine decay with 10% linear warmup
- Gradient clipping at norm 1.0
- Early stopping: halts if val accuracy drops beyond `max_accuracy_drop`
- Labels NOT required — only teacher forward passes needed

**Intermediate layer distillation (TinyBERT, Jiao et al. 2020):**
```
L_feature = MSE(student_hidden[i], projection(teacher_hidden[j]))
```
A learned linear projection handles dimension mismatch between teacher and student after pruning.

---

### 3. Quantization (`techniques/quantization.py`)

Reduces numerical precision of weights:

| Device | Mode | Reduction | Notes |
|--------|------|-----------|-------|
| CUDA (T4, RTX) | fp16 | ~2× | Works on GPU. Half precision. |
| CPU | dynamic int8 | ~4× | CPU-only PyTorch operator. |

**Key engineering note:** `torch.quantization.quantize_dynamic` converts `nn.Linear` → `DynamicQuantizedLinear` — a C++-backed layer registered only for CPU backend (`quantized::linear_dynamic`). Calling it on a CUDA model raises `NotImplementedError`. SlimNet detects the device and automatically selects the right mode.

Outputs validated with cosine similarity > 0.999 before proceeding.

---

### 4. Hardware Profiler (`core/planner.py`)

Before compression, SlimNet profiles the deployment environment:

```python
profiler = HardwareProfiler()
profile  = profiler.profile(model, sample_input)
# HardwareProfile(device=cuda, vram_free=14895 MB, ram=12976 MB,
#                 latency=4.35 ms, mem=89.8 MB)
```

The `CompressionPlanner` uses this profile to select techniques automatically based on target hardware constraints and accuracy budget.

---

## Bugs Encountered and Fixed

| Bug | Symptom | Root Cause | Fix |
|-----|---------|------------|-----|
| Empty optimizer | `ValueError: optimizer got an empty parameter list` | Dynamic int8 quant converts `nn.Linear` to C++ layer with no `parameters()` | `_reorder_steps()` moves quantization after pruning and distillation |
| Size = 0.00 MB | Compressed size reported as 0 | `model.parameters()` empty on quantized model | Use `torch.save()` + measure file bytes on disk |
| HalfTensor/FloatTensor | `RuntimeError: Input type (cuda.HalfTensor)` | `torch.amp.autocast` cast inputs to fp16, model weights fp32 | Disabled autocast, explicit `x = x.float()` before every forward |
| int8 on CUDA | `NotImplementedError: quantized::linear_dynamic` from CUDA backend | int8 is a CPU-only operator in PyTorch | Detect device, use fp16 on CUDA and int8 on CPU |
| Scheduler before optimizer | `UserWarning: lr_scheduler.step() before optimizer.step()` | Wrong call order | Swapped to `optimizer.step()` then `scheduler.step()` |

---

## Repository Structure

```
slimnet/
├── slimnet/
│   ├── __init__.py              # Public API: compress(), Compressor
│   ├── configs.py               # QuantConfig, PruneConfig, LowRankConfig, DistillConfig
│   ├── cli.py                   # Typer CLI: slimnet compress model.pt --target t4
│   ├── core/
│   │   ├── planner.py           # HardwareProfiler, CompressionPlanner
│   │   ├── pipeline.py          # CompressionPipeline, Compressor
│   │   └── benchmark.py         # BenchmarkReport, BenchmarkReporter
│   └── techniques/
│       ├── quantization.py      # fp16 / dynamic int8
│       ├── pruning.py           # magnitude / gradient / Taylor scoring
│       ├── low_rank.py          # SVD factorization — Linear(m,n) → Linear(m,k)→Linear(k,n)
│       └── distillation.py      # KL divergence + intermediate feature distillation
├── tests/
│   ├── test_quantization.py     # 10 tests
│   ├── test_pruning.py          # 8 tests
│   ├── test_low_rank.py         # 9 tests
│   ├── test_distillation.py     # 12 tests
│   └── test_pipeline.py         # 15 tests — end-to-end integration
├── examples/
│   ├── resnet50_t4.py           # ResNet-50 → T4: 100MB → 18MB, 4.2× speedup
│   ├── bert_cpu.py              # BERT-base → CPU: 440MB → 65MB, 3.8× speedup
│   └── whisper_macbook.py       # Whisper Large → MacBook M2
└── run_full.py                  # End-to-end smoke test
```

---

## Run Tests

```bash
python -m pytest tests/ -v --tb=short --cov=slimnet --cov-report=term-missing
```

---

## Smoke Test

```bash
python run_full.py
```

Expected output:
```
Size    : 0.26 MB → 0.07 MB  (3.70x)
Applied : pruning → distillation → quantization
Shape check PASSED ✓
```

---

## Hardware Targets

| Target | VRAM Budget | Notes |
|--------|-------------|-------|
| `cpu` | — | Dynamic int8, no GPU needed |
| `t4` | 15 GB | Free Colab GPU |
| `macbook` | 16 GB | Apple Silicon unified memory, bf16 |
| `rtx3070` | 8 GB | Consumer GPU |
| `rtx3080` | 10 GB | |

---

## References

- Hinton, Vinyals, Dean (2015). *Distilling the Knowledge in a Neural Network*. arXiv:1503.02531
- Jiao et al. (2020). *TinyBERT: Distilling BERT for Natural Language Understanding*. arXiv:1909.10351
- Frankle & Carlin (2019). *The Lottery Ticket Hypothesis*. ICLR 2019
- Han et al. (2015). *Learning both Weights and Connections for Efficient Neural Networks*. NeurIPS 2015

---

## License

MIT
