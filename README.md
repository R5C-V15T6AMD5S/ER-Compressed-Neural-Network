# CNN Model Compression — Group Project

A pure NumPy implementation of a Convolutional Neural Network trained on the **Fruits-360** dataset, with model compression techniques applied and compared.

Based on the survey paper:
> Li, Z.; Li, H.; Meng, L. *Model Compression for Deep Neural Networks: A Survey.* Computers 2023, 12, 60.

---

## Table of Contents

1. [What This Project Does](#what-this-project-does)
2. [Project Structure](#project-structure)
3. [Requirements](#requirements)
4. [Dataset Setup](#dataset-setup)
5. [Quick Start — Full Pipeline](#quick-start--full-pipeline)
6. [Per-Person Tutorials](#per-person-tutorials)
   - [Person 1 — CNN Architecture](#person-1--cnn-architecture)
   - [Person 1 V2 — Improved Architecture (Milestone 2)](#person-1-v2--improved-architecture-milestone-2)
   - [Person 2 — Training](#person-2--training)
   - [Person 2 V2 — Quantization-Aware Training (Milestone 2)](#person-2-v2--quantization-aware-training-milestone-2)
   - [Person 3 — Pruning](#person-3--pruning)
   - [Person 3 V2 — Dynamic Pruning on SimpleCNNv2 (Milestone 2)](#person-3-v2--dynamic-pruning-on-simplecnnv2-milestone-2)
   - [Person 4 — Quantization](#person-4--quantization)
7. [Output and Logs](#output-and-logs)
8. [Architecture Details](#architecture--simplecnn)
9. [Compression Techniques](#compression-techniques)
10. [Hardware Notes](#hardware-notes)
11. [References](#references)

---

## What This Project Does

Trains a CNN from scratch (no PyTorch, no TensorFlow — only NumPy) on the Fruits-360 image classification dataset, then applies two model compression techniques and compares the results:

| Step | Who | File | What |
|------|-----|------|------|
| 1 | Person 1 | `person1_model.py` | CNN architecture — layers, forward, backward |
| 1b | Person 1 (KT2) | `person1_model_v2.py` | Improved V2 architecture — BatchNorm + Dropout + 3rd Conv |
| 2 | Person 2 | `person2_train.py` | Training loop, loss function, optimizer, save/load |
| 2b | Person 2 (KT2) | `person2_train_v2.py` | Training + Quantization-Aware Training (QAT) |
| 3 | Person 3 | `person3_pruning.py` | Weight pruning — removes 50% of weights (post-training) |
| 3b | Person 3 (KT2) | `person3_dynamic_pruning.py` | Dynamic pruning during training on V2 model |
| 4 | Person 4 | `person4_quantization.py` | Post-training quantization — FP32 → INT8 |

The full pipeline runs in one command via `main.py` and automatically saves a timestamped log to the `logs/` folder.

**Dependency chain:**
```
Person 1 (model) → Person 2 (train) → Person 3 (pruning)
                                     → Person 4 (quantization)
```
Person 1 and Person 2 are fully independent. Person 3 and 4 need a trained model from Person 2.

---

## Project Structure

```
project-root/
│
├── person1_model.py        ← CNN architecture V1 (Person 1)
├── person1_model_v2.py     ← CNN architecture V2 with BatchNorm + Dropout (Person 1, KT2)
├── compare_models.py       ← Compare V1 vs V2 side-by-side (Person 1, KT2)
├── person2_train.py        ← Training loop + optimizer (Person 2)
├── person2_train_v2.py     ← Training + Quantization-Aware Training (Person 2, KT2)
├── person3_pruning.py            ← Weight pruning V1 — train→prune→fine-tune (Person 3)
├── person3_dynamic_pruning.py    ← Dynamic pruning during training on V2 (Person 3, KT2)
├── person4_quantization.py       ← Quantization (Person 4)
├── main.py                 ← Full pipeline — runs everything in order
├── logger.py               ← Shared logging utility (saves to logs/)
├── README.md               ← This file
│
├── fruits-360-100x100/     ← Dataset folder (you provide this)
│   ├── Training/
│   │   ├── Apple Braeburn/
│   │   ├── Banana/
│   │   └── ...
│   └── Test/
│       ├── Apple Braeburn/
│       └── ...
│
├── logs/                   ← Auto-created on first run
│   ├── main_20260522_1420.txt
│   ├── person2_20260522_1435.txt
│   └── ...
│
└── models/                 ← Auto-created on first run, all .npz files saved here
    ├── cnn_fruits.npz          ← trained model (Person 2)
    ├── cnn_pruned.npz           ← pruned model (Person 3)
    └── cnn_quantized.npz       ← quantized model (Person 4)
```

---

## Requirements

Only standard Python libraries are needed:

```bash
pip install numpy pillow
```

- Python 3.8+
- NumPy
- Pillow (PIL) — for loading images

No PyTorch, no TensorFlow, no sklearn.

---

## Dataset Setup

This project uses **Fruits-360** (100×100 px version).

Download it from Kaggle:
[https://www.kaggle.com/datasets/moltean/fruits](https://www.kaggle.com/datasets/moltean/fruits)

After downloading and extracting, your folder should look like this:
```
fruits-360-100x100/
    Training/
        Apple Braeburn/
        Apple Granny Smith/
        Banana/
        ...
    Test/
        Apple Braeburn/
        Apple Granny Smith/
        Banana/
        ...
```

Place the `fruits-360-100x100/` folder directly in the project root. All scripts expect it there by default.

---

## Quick Start — Full Pipeline

If you just want to run everything at once (all 4 steps in order):

**Quick test — confirms everything works, finishes in ~30–50 min:**
```bash
python main.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**Full training run — better accuracy, ~1–2 hours:**
```bash
python main.py --data_dir ./fruits-360-100x100 --epochs 10 --max_per_class 100
```

All results are printed to the terminal and saved automatically to `logs/main_<timestamp>.txt`.

---

## Per-Person Tutorials

Each person can test their own part independently without running the full pipeline. Follow the section for your person number below.

---

### Person 1 — CNN Architecture

**File:** `person1_model.py`  
**Depends on:** Nothing — no dataset, no other files needed  
**What it does:** Defines all the neural network layers (Conv, ReLU, MaxPool, FC, Softmax) and assembles them into `SimpleCNN`

#### How to test your part

No dataset needed. The script generates fake random images internally and runs a mini training loop to confirm the architecture is working:

```bash
python person1_model.py
```

Expected output:
```
==========================================================
  Person 1 — SimpleCNN self-test
  (fake random data, no dataset needed)
==========================================================

  Parameters : 1,282,810
  Trainable layers (4):
    [0] Conv  shape=(8, 3, 3, 3)   params=224
    [1] Conv  shape=(16, 8, 3, 3)  params=1,168
    [2] FC    shape=(10000, 128)   params=1,280,128
    [3] FC    shape=(128, 10)      params=1,290

  [Check 1] Forward pass ... OK — output shape (2, 10), probs sum to 1.0  ✓
  [Check 2] Backward pass ... OK — all gradient shapes correct  ✓

  [Check 3] Mini training loop
  Fake dataset: 40 images | 10 classes | 3 epochs | batch=4

    Epoch [1/3]  Loss: 5.4560  |  Accuracy: 47.5%
    Epoch [2/3]  Loss: 2.0103  |  Accuracy: 67.5%
    Epoch [3/3]  Loss: 1.4582  |  Accuracy: 100.0%

  First epoch loss : 5.4560
  Last  epoch loss : 1.4582  ✓ Loss decreased as expected

  ✓ All checks passed — person1_model.py is working correctly.
  Person 2 can now use this model for real training.
==========================================================
```

**What to look for:**
- All 3 checks should show ✓
- Loss should go down each epoch (e.g. 5.4 → 2.0 → 1.4)
- Final line should say "All checks passed"

**Optional — run more epochs or larger fake dataset:**
```bash
python person1_model.py --epochs 5 --num_samples 80 --num_classes 20
```

| Option | Default | What it does |
|--------|---------|-------------|
| `--epochs` | 3 | How many training epochs to run |
| `--batch_size` | 4 | Images per batch |
| `--num_classes` | 10 | Number of fake output classes |
| `--num_samples` | 40 | Number of fake training images |

---

### Person 1 V2 — Improved Architecture (Milestone 2)

**Files:** `person1_model_v2.py`, `compare_models.py`
**Depends on:** `person1_model.py` (reuses ConvLayer, ReLULayer, MaxPoolLayer, FlattenLayer, FCLayer, SoftmaxLayer), `person2_train.py` (for training utilities), dataset
**What it does:** Introduces an improved `SimpleCNNv2` architecture and a script to compare it against the original V1 baseline under identical conditions.

#### What changed vs V1

Three concrete improvements designed to reduce overfitting (V1's training accuracy was ~99% while test accuracy plateaued at ~70%):

1. **BatchNorm after every Conv layer** — stabilizes activations during training, acts as mild regularization
2. **Dropout (p=0.3) after FC1** — randomly drops 30% of FC1 outputs during training, prevents memorization
3. **Third Conv block (16→32 filters)** with extra MaxPool — reduces spatial dimensions from 25→12, shrinking FC1 input from 10 000 to 4 608 features

**Architecture comparison:**

```
V1 (original)                          V2 (improved)
─────────────                          ──────────────
Conv(3→8) → ReLU → MaxPool             Conv(3→8)  → BN → ReLU → MaxPool
Conv(8→16) → ReLU → MaxPool            Conv(8→16) → BN → ReLU → MaxPool
                                       Conv(16→32) → BN → ReLU → MaxPool  ← NEW
Flatten (10 000)                       Flatten (4 608)
FC(10000→128) → ReLU                   FC(4608→128) → ReLU → Dropout(0.3)
FC(128→261)                            FC(128→261)
Softmax                                Softmax

~1.32M parameters                      ~630K parameters (~52% smaller)
```

#### Compatibility with the rest of the team — IMPORTANT

V2 keeps the **exact same public interface as V1**:

- `model.forward(X)` → softmax probabilities
- `model.backward(d_probs)` → populates gradients
- `model.get_trainable_layers()` → list of ConvLayer / FCLayer only
- `model.get_param_count()` → integer

This means **Person 2's training loop, Person 3's pruning, and Person 4's quantization work on V2 without any code changes**. You only need to swap the import:

```python
# Old (V1)
from person1_model import SimpleCNN
model = SimpleCNN(num_classes=261)

# New (V2)
from person1_model_v2 import SimpleCNNv2
model = SimpleCNNv2(num_classes=261)
```

Everything else stays identical — your optimizer iterates over `get_trainable_layers()`, your pruning checks `hasattr(layer, "filters")`, your quantization reads `layer.filters`/`layer.weights`. None of that changes.

#### BatchNorm parameter handling (additional info)

V2 has BatchNorm gamma/beta parameters that are NOT returned by `get_trainable_layers()` — this is intentional, otherwise Person 2's optimizer would try to treat them like Conv/FC weights and crash. Instead, V2 manages them internally with its own SGD-with-momentum step inside `model.backward()`.

These BN parameters need to be saved/loaded separately from the main `.npz`:

```python
# Saving — call BOTH
save_model(model, "models/v2_best", classes=classes)   # Person 2's save_model
model.save_bn_params("models/v2_best")                 # writes models/v2_best_bn.npz

# Loading — call BOTH
load_model(model, "models/v2_best")
model.load_bn_params("models/v2_best")
model.eval()   # important — switches BN to use running stats, Dropout to identity
```

If Person 3 or Person 4 wants to prune/quantize a V2 model, they need to call `model.load_bn_params(...)` and `model.eval()` after loading. Otherwise BatchNorm uses default identity stats and accuracy will be near-random.

#### How to test your part — standalone (no dataset needed)

```bash
python person1_model_v2.py
```

Generates fake data, runs forward + backward, trains for 3 epochs, confirms loss decreases. Takes ~90 seconds.

#### How to run the V1 vs V2 comparison

The `compare_models.py` script trains BOTH architectures under identical conditions (same seed, same data, same hyperparameters) and prints a side-by-side comparison.

**Quick smoke test (~25 minutes total):**

```bash
python compare_models.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 30
```

**Full run (~5-6 hours total — V1 ≈ 2.5h, V2 ≈ 3h):**

```bash
python compare_models.py --data_dir ./fruits-360-100x100 --epochs 10 --max_per_class 100
```

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360-100x100` | Path to dataset |
| `--max_per_class` | 100 | Limit images per class |
| `--epochs` | 10 | Training epochs for both V1 and V2 |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Learning rate (used for both) |
| `--seed` | 42 | Random seed (used for both — guarantees same data shuffle) |
| `--dropout_p` | 0.3 | Dropout probability for V2 only |

#### Output files after a comparison run

```
logs/
└── compare_20260619_HHMM.txt        ← side-by-side comparison log

models/
├── v1_best.npz                       ← best V1 weights
├── v2_best.npz                       ← best V2 weights (Conv/FC)
└── v2_best_bn.npz                    ← V2 BatchNorm parameters
```

#### Using V2 with Person 3 and Person 4 (for KT2 combined pipeline)

If a teammate wants to apply pruning or quantization to V2 instead of V1, the minimal change is the import + BN load/eval calls. For example, Person 3's main code would become:

```python
from person1_model_v2 import SimpleCNNv2

# Instead of: model = SimpleCNN(num_classes=num_classes)
model = SimpleCNNv2(num_classes=num_classes)
load_model(model, "models/v2_best")
model.load_bn_params("models/v2_best")
model.eval()

# Everything else stays the same — global_prune() and apply_masks() iterate
# over get_trainable_layers() which only returns Conv/FC layers, same as V1.
masks = prune_model(model, "global", 0.5)
```

Same idea for Person 4 — only the model creation and BN-load lines change.

#### Why these specific changes (and not others)

The V1 baseline showed clear overfitting (~30 percentage point train-test gap). The standard solutions are:

- **More data / augmentation** — out of scope (no time for new dataset work)
- **More regularization** — what we did (Dropout + BatchNorm)
- **Smaller model** — partial side effect (FC1 shrunk because of extra pool)

We did **not** simply add more filters or more layers because that would make overfitting worse, not better.

#### What to expect from the comparison

Based on the seminar paper [1], BatchNorm + Dropout typically need more epochs to converge than a vanilla CNN, because they slow down memorization by design. In our 3-epoch smoke test V2 was still catching up to V1; in the full 10-epoch run V2 should close most of the gap while remaining ~52% smaller.

Even if V2's final accuracy is slightly below V1's, the comparison itself is the contribution — V2 is a methodologically clean experiment in the direction the seminar paper recommends (combining architectural design with model compression).

---

### Person 2 — Training

**File:** `person2_train.py`  
**Depends on:** `person1_model.py`, dataset  
**What it does:** Loads the dataset, trains `SimpleCNN` using SGD with momentum and cross-entropy loss, saves the best model to `models/cnn_fruits.npz`

#### How to test your part

**Quick test (~15–25 min):**
```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**More thorough test (~1 hour):**
```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 10 --max_per_class 100
```

Expected output:
```
[Data] Loading 'Training' split (261 classes)...
  Loaded 13016 images. Shape: (13016, 3, 100, 100)
[Data] Loading 'Test' split (261 classes)...
  Loaded 12999 images. Shape: (12999, 3, 100, 100)

[Model] SimpleCNN — 1,315,189 parameters
[Train] 13016 samples | 3 epochs | lr=0.01 | batch=32

Epoch [  1/3]  Loss: 2.0516  |  Test Accuracy: 63.30%
  ↑ New best! (63.30%)
Epoch [  2/3]  Loss: 0.0835  |  Test Accuracy: 65.61%
  ↑ New best! (65.61%)
Epoch [  3/3]  Loss: 0.0158  |  Test Accuracy: 70.81%
  ↑ New best! (70.81%)

────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────
Best accuracy : 70.81%
Model saved to: models/cnn_fruits.npz
```

**What to look for:**
- Loss should decrease each epoch
- Accuracy should increase each epoch
- `models/cnn_fruits.npz` should appear in the `models/` folder when done

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 32 | Images per batch |
| `--lr` | 0.01 | Learning rate |
| `--save_path` | `models/cnn_fruits` | Filename for saved model |
| `--max_per_class` | None (all) | Limit images per class |

> **Note:** Once this finishes and `models/cnn_fruits.npz` exists, Person 3 and Person 4 can run their parts.

---


### Person 2 V2 — Quantization-Aware Training (Milestone 2)

**File:** `person2_train_v2.py`
**Depends on:** `person1_model.py` (V1), `logger.py`, dataset
**What it does:** Same training pipeline as the original `person2_train.py`, plus an **optional QAT (Quantization-Aware Training)** mode. During training, weights are periodically "fake quantized" — rounded to the INT8 grid and immediately dequantized back to FP32. The model learns to be robust to the precision loss that will happen later when Person 4 applies real INT8 quantization.

#### Why QAT is better than plain PTQ

| Pipeline | Accuracy after INT8 quantization |
|---|---|
| Train FP32 → PTQ (Person 4 only) | Some accuracy drop |
| Train FP32 with QAT → PTQ (Person 2 V2 + Person 4) | **Much smaller drop** |

The model "knows" during training that its weights will eventually be discretized to 256 INT8 levels, so it learns weight values that survive that rounding well. This is the approach described in **Section 3.1 of the seminar paper** [1].

#### How it works (under the hood)

For each Conv/FC layer, on every Nth batch:
1. Find the min/max of the weight tensor (or use a symmetric scheme)
2. Compute scale `S` and zero-point `Z` (same math as Person 4's PTQ — equations 1–4 in the seminar)
3. Round the weights to the INT8 grid: `q = round(w/S + Z)`, clipped to [-128, 127]
4. Dequantize back: `w_fake = (q - Z) * S`
5. Replace the layer's weights with `w_fake`

The result has dtype float32 but its values can only land on the discrete INT8 grid. The rest of the training continues as normal — backprop, optimizer step, etc.

#### How to test your part — quick technical sanity check

Baseline (no QAT, like the original training):

```bash
python person2_train_v2.py --data_dir ./fruits-360-100x100 \
                              --epochs 1 \
                              --max_per_class 5 \
                              --batch_size 4
```

With QAT enabled:

```bash
python person2_train_v2.py --data_dir ./fruits-360-100x100 \
                              --epochs 1 \
                              --max_per_class 5 \
                              --batch_size 4 \
                              --qat
```

#### Full QAT training run

```bash
python person2_train_v2.py --data_dir ./fruits-360-100x100 \
                              --epochs 10 \
                              --max_per_class 100 \
                              --qat \
                              --qat_warmup_epochs 1 \
                              --qat_frequency 1
```

#### QAT-specific options

| Option | Default | What it does |
|--------|---------|-------------|
| `--qat` | off | Enable fake quantization during training |
| `--qat_strategy` | asymmetric | `symmetric` or `asymmetric` quantization |
| `--qat_bits` | 8 | Bits for fake quantization (8 = INT8) |
| `--qat_warmup_epochs` | 0 | Train initial epochs as FP32 only (helps stability) |
| `--qat_frequency` | 1 | Fake-quantize every Nth batch (1 = every batch) |

#### Output files

```
models/
└── cnn_fruits_qat.npz        ← model trained with QAT (when --qat is on)

logs/
└── person2_TIMESTAMP.txt      ← training log with per-epoch QAT error
```

#### Using a QAT-trained model with Person 4's quantization

The QAT model is saved like any other model — Person 4's existing quantization script can load it directly:

```bash
python person4_quantization.py --data_dir ./fruits-360-100x100 \
                                  --model_path models/cnn_fruits_qat \
                                  --max_per_class 100
```

The interesting comparison is **two PTQ results side-by-side**:
- Person 4 quantizing `models/cnn_fruits` (baseline FP32 training) → some accuracy drop
- Person 4 quantizing `models/cnn_fruits_qat` (QAT-trained) → smaller accuracy drop

That comparison is a strong KT2 result: it directly demonstrates the benefit of QAT predicted by the seminar.

#### Note on V1 vs V2 compatibility

The current `person2_train_v2.py` imports V1 (`from person1_model import SimpleCNN`). To use Borna's QAT on the V2 architecture, the import would need to change to `from person1_model_v2 import SimpleCNNv2` plus a `model.load_bn_params(...)` / `model.eval()` call where appropriate. This is a small change (a few lines) and can be done if the team decides to combine V2 + QAT for the final pipeline.

---
### Person 3 — Pruning

**File:** `person3_pruning.py`  
**Depends on:** `person1_model.py`, `person2_train.py`, a saved `models/cnn_fruits.npz` from Person 2  
**What it does:** Loads the trained model, removes the 50% lowest-magnitude weights (sets them to zero), fine-tunes for a few epochs to recover accuracy, then compares results

#### Before you start

You need `models/cnn_fruits.npz` to exist. If it doesn't exist yet, run Person 2's script first:

```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

This takes ~15–25 min and creates `models/cnn_fruits.npz`.

#### How to test your part

```bash
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits --max_per_class 50
```

Expected output:
```
────────────────────────────────────────────────────
  BASELINE
────────────────────────────────────────────────────
  Accuracy : 70.81%
  Size     : 10275.4 KB
  Inference: 82.66 ms

────────────────────────────────────────────────────
  PRUNING
────────────────────────────────────────────────────
[Pruning] Strategy: 'global' | Amount: 50%
  [Global Pruning] Threshold: 0.010731
  [Global Pruning] 657,388 weights zeroed. Sparsity: 50.0%
  Accuracy after pruning (before fine-tune): 70.59%

  [Fine-tuning] 3 epoch(s) at lr=0.001...
    Fine-tune [1/3]  Loss: 0.0063  |  Accuracy: 71.60%
    Fine-tune [2/3]  Loss: 0.0017  |  Accuracy: 71.64%
    Fine-tune [3/3]  Loss: 0.0012  |  Accuracy: 71.55%

────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────
  Metric                   Original        Pruned
  Accuracy (%)               70.81%         71.55%
  Model Size (KB)          10275.4        10275.4
  Inference (ms)             82.66          82.63
  Sparsity                    0.0%           50.0%
  Accuracy Drop                             -0.75%
  Speedup                                    1.00x
```

**What to look for:**
- Sparsity should reach ~50% after pruning
- Accuracy after fine-tuning should be close to (or better than) the original
- A negative accuracy drop means pruning actually improved accuracy slightly — this is normal, it acts like regularization

**If the model file is missing**, the script will tell you exactly what to do:
```
  ERROR — No trained model found!
  Expected file: models/cnn_fruits.npz

  Person 3 needs a trained model from Person 2 first.
  Run this command to train one:

    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--model_path` | `models/cnn_fruits` | Trained model to load |
| `--save_path` | `models/cnn_pruned` | Where to save pruned model |
| `--strategy` | `global` | `global` or `per_layer` |
| `--amount` | 0.5 | Fraction of weights to prune |
| `--finetune_epochs` | 3 | Epochs to fine-tune after pruning |
| `--finetune_lr` | 0.001 | Learning rate during fine-tuning |
| `--max_per_class` | None (all) | Limit images per class |

**Try a different pruning amount:**
```bash
# More aggressive — 70% of weights removed
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits --amount 0.7 --max_per_class 50

# Less aggressive — 30% of weights removed
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits --amount 0.3 --max_per_class 50
```

**Try per-layer strategy instead of global:**
```bash
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits --strategy per_layer --max_per_class 50
```

---


### Person 3 V2 — Dynamic Pruning on SimpleCNNv2 (Milestone 2)

**File:** `person3_dynamic_pruning.py`
**Depends on:** `person1_model_v2.py` (V2 architecture), `logger.py`, dataset
**What it does:** Trains Person 1's V2 model **while gradually pruning weights at the same time** — a single integrated training+pruning pipeline, instead of the original "train fully, then prune, then fine-tune" approach.

#### How it differs from the original `person3_pruning.py`

| Aspect | Original (`person3_pruning.py`) | V2 (`person3_dynamic_pruning.py`) |
|---|---|---|
| Pruning timing | After full training | **Gradually during training** |
| Sparsity schedule | One step, 0% → 50% | **Smooth ramp** from epoch 2 to last epoch |
| Model architecture | V1 (SimpleCNN) | **V2 (SimpleCNNv2)** with BatchNorm + Dropout |
| Requires Borna's pre-trained model | Yes | **No** — runs end-to-end on its own |
| Output | Pruned model + masks | Pruned V2 model + BN params + masks |

#### Why dynamic pruning is better

The model has more chances to **adapt to missing weights** as they get pruned, instead of suddenly losing 50% of weights all at once and trying to recover. This typically gives **better accuracy at the same sparsity level**.

#### How to test your part — standalone

```bash
python person3_dynamic_pruning.py --data_dir ./fruits-360-100x100 \
                                    --epochs 5 \
                                    --max_per_class 30 \
                                    --amount 0.5
```

#### Full run

```bash
python person3_dynamic_pruning.py --data_dir ./fruits-360-100x100 \
                                    --epochs 15 \
                                    --max_per_class 100 \
                                    --amount 0.5 \
                                    --strategy global \
                                    --prune_start_epoch 2 \
                                    --prune_frequency 10
```

#### All options

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360-100x100` | Path to dataset |
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 32 | Batch size |
| `--lr` | 0.01 | Learning rate |
| `--momentum` | 0.9 | SGD momentum |
| `--amount` | 0.5 | Final target sparsity (0.5 = 50%) |
| `--strategy` | global | `global` or `per_layer` |
| `--prune_start_epoch` | 2 | Epoch where pruning begins |
| `--prune_end_epoch` | None | Last pruning epoch (defaults to `--epochs`) |
| `--prune_frequency` | 10 | Update masks every N batches |
| `--dropout_p` | 0.3 | Dropout for V2 |
| `--max_per_class` | None | Image limit per class (quick tests) |
| `--save_path` | `models/cnn_dynamic_pruned` | Where to save the model |
| `--mask_path` | `models/dynamic_pruning_masks` | Where to save masks |

#### Output files

```
models/
├── cnn_dynamic_pruned.npz          ← Conv/FC weights (best epoch)
├── cnn_dynamic_pruned_bn.npz       ← BatchNorm gamma/beta + running stats
└── dynamic_pruning_masks.npz       ← Binary pruning masks per layer

logs/
└── person3_dynamic_pruning_TIMESTAMP.txt    ← full run log
```

#### Using the dynamically pruned V2 model with Person 4's quantization

This is the **combined KT2 pipeline** — V2 architecture, dynamically pruned, then quantized. The pruned model is saved exactly like any other V2 model, so Person 4 can load it with just the standard V2 loading sequence:

```python
# Inside Person 4's quantization script
from person1_model_v2 import SimpleCNNv2
from person2_train import load_model, compute_accuracy

# Load the dynamically pruned V2 model (not the original V2)
model = SimpleCNNv2(num_classes=num_classes)
load_model(model, "models/cnn_dynamic_pruned")
model.load_bn_params("models/cnn_dynamic_pruned")
model.eval()   # IMPORTANT: switches BatchNorm to running stats, Dropout off

# Now apply existing quantization — no changes needed
model, metadata = quantize_model(model, strategy="asymmetric")
acc = compute_accuracy(model, X_test, y_test, batch_size=32)
```

**Important:** Person 4 must call `model.load_bn_params(...)` AND `model.eval()` — otherwise BatchNorm uses default identity statistics and accuracy drops to near-random.

The end-to-end combined pipeline would then look like:

```
V2 architecture (Person 1)
        ↓
Dynamic pruning during training (Person 3 V2) → cnn_dynamic_pruned.npz (sparse + V2)
        ↓
Post-training INT8 quantization (Person 4)    → fully compressed model
```

Expected stacked compression: **~52% from V2** × **~50% from pruning** × **~75% from quantization (INT8 vs FP32)** = a model many times smaller than the original V1 baseline.

---
### Person 4 — Quantization

**File:** `person4_quantization.py`  
**Depends on:** `person1_model.py`, `person2_train.py`, a saved `models/cnn_fruits.npz` from Person 2  
**What it does:** Loads the trained model, converts all weights from FP32 to INT8 using scale `S` and zero-point `Z` (Equations 1–5 from the survey), dequantizes back to FP32, then measures the accuracy impact and theoretical size savings

#### Before you start

You need `models/cnn_fruits.npz` to exist. If it doesn't, run Person 2's script first:

```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

#### How to test your part

```bash
python person4_quantization.py --data_dir ./fruits-360-100x100 --model_path models/cnn_fruits --max_per_class 50
```

Expected output:
```
────────────────────────────────────────────────────
  BASELINE
────────────────────────────────────────────────────
  Accuracy : 70.81%
  Size     : 10275.4 KB
  Inference: 82.66 ms

────────────────────────────────────────────────────
  QUANTIZATION
────────────────────────────────────────────────────
[Quantization] Applying Algorithm 1 from the survey paper...
  Steps: min/max → S and Z → Q = R/S + Z → R = (Q-Z)*S

  [Quantization] Quantizing layers FP32 → INT8 → FP32 (dequant)...
  Layer    Shape                    S          Z     Avg Error     INT8 Range
  --------------------------------------------------------------------------
  0   Conv  (8, 3, 3, 3)      0.010009       9      0.002431    [-128, 127]
  1   Conv  (16, 8, 3, 3)     0.006665      -8      0.001643    [-128, 127]
  2   FC    (10000, 128)       0.001761     -16      0.000441    [-128, 127]
  3   FC    (128, 261)         0.004027       3      0.001005    [-128, 127]

────────────────────────────────────────────────────
  RESULTS
────────────────────────────────────────────────────
  Metric                         Original    Quantized
  Accuracy (%)                     70.81%       70.84%
  Theoretical FP32 Size (KB)      5135.8
  Theoretical INT8 Size (KB)                   1284.0
  Theoretical Compression                           4x
  Inference (ms)                    82.66        70.24
  Accuracy Drop                                 -0.04%
  Speedup                                        1.18x

  Per-layer quantization error summary:
    Layer 0: avg weight error = 0.002431, S=0.010009, Z=9
    Layer 1: avg weight error = 0.001643, S=0.006665, Z=-8
    Layer 2: avg weight error = 0.000441, S=0.001761, Z=-16
    Layer 3: avg weight error = 0.001005, S=0.004027, Z=3
```

**What to look for:**
- Accuracy after quantization should be almost identical to before (typically < 1% drop)
- `Avg Error` per layer shows how much the weight values changed — lower is better
- Theoretical INT8 size should be exactly 4× smaller than FP32 (this is the main result)
- Inference time may improve slightly due to simpler arithmetic

**If the model file is missing**, the script will tell you exactly what to do:
```
  ERROR — No trained model found!
  Expected file: models/cnn_fruits.npz

  Person 4 needs a trained model from Person 2 first.
  Run this command to train one:

    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--model_path` | `models/cnn_fruits` | Trained model to load |
| `--save_path` | `models/cnn_quantized` | Where to save quantized model |
| `--batch_size` | 32 | Batch size for accuracy evaluation |
| `--max_per_class` | None (all) | Limit images per class |

---

## Output and Logs

Every run saves a timestamped `.txt` log file to the `logs/` folder automatically:

```
logs/
    main_20260522_1420.txt          ← full pipeline (main.py)
    person2_20260522_1435.txt       ← training only
    person3_20260522_1450.txt       ← pruning only
    person4_20260522_1505.txt       ← quantization only
```

The log contains everything printed to the terminal — every epoch, every accuracy reading, and the final comparison table. Open any log in a text editor (VS Code recommended) to review a past run.

**Tip:** If the terminal shows garbled characters (missing letters), this is a Windows encoding display issue. The saved `.txt` log file will be correct — open it in VS Code to read the real output.

---

## Architecture — SimpleCNN

```
Input  (N, 3, 100, 100)          ← batch of N RGB images, 100×100 px
  │
  ├─ Conv(3→8,  3×3, pad=1)  → ReLU → MaxPool(2×2)   → (N,  8, 50, 50)
  ├─ Conv(8→16, 3×3, pad=1)  → ReLU → MaxPool(2×2)   → (N, 16, 25, 25)
  ├─ Flatten                                           → (N, 10000)
  ├─ FC(10000 → 128)         → ReLU
  ├─ FC(128 → num_classes)
  └─ Softmax                                           → (N, num_classes)
```

| Layer | Type | Output shape | Parameters |
|-------|------|-------------|-----------|
| Conv1 | Convolution | (N, 8, 100, 100) | 224 |
| Pool1 | MaxPool 2×2 | (N, 8, 50, 50) | 0 |
| Conv2 | Convolution | (N, 16, 50, 50) | 1,168 |
| Pool2 | MaxPool 2×2 | (N, 16, 25, 25) | 0 |
| FC1 | Fully connected | (N, 128) | 1,280,128 |
| FC2 | Fully connected | (N, num_classes) | varies |
| Softmax | Activation | (N, num_classes) | 0 |

Total: ~1.3 million parameters (varies slightly by number of classes in the dataset).

---

## Compression Techniques

### Pruning (Person 3) — Survey Section 2

Global unstructured L1 pruning (Han et al.). Collects all weight magnitudes across every layer, finds the value at the 50th percentile, and zeroes out everything below it. After pruning, the model is fine-tuned for a few epochs with those weights frozen at zero.

- Default: removes **50%** of all weights globally
- Sparsity = percentage of weights equal to exactly zero
- Fine-tuning with frozen mask ensures pruned connections stay pruned

```
threshold = percentile(|all_weights|, 50)
mask      = |w| >= threshold
w         = w * mask          ← zeroes out low-magnitude weights
```

### Quantization (Person 4) — Survey Section 3, Algorithm 1

Post-training quantization (PTQ). Converts each layer's FP32 weights to INT8 using a scale factor `S` and zero-point `Z`, then dequantizes back to FP32 for inference. Demonstrates the theoretical **4× size reduction** of INT8 vs FP32 storage.

The math directly from the paper (Equations 1–5):

```
S = (R_max - R_min) / (Q_max - Q_min)     ← scale factor
Z = Q_max - R_max / S                      ← zero-point
Q = round(R / S + Z)                       ← FP32 → INT8
R = (Q - Z) * S                            ← INT8 → FP32 (dequantize)
```

---

## Hardware Notes

This runs on **CPU only** — no GPU needed. Tested on a laptop with 16 GB RAM.

| Setting | Images loaded | Approx. time per epoch |
|---------|--------------|------------------------|
| `--max_per_class 50` | ~13k | 15–25 min |
| `--max_per_class 100` | ~26k | 30–50 min |
| No limit (full dataset) | ~90k | several hours |

**Recommended settings for each situation:**

| Situation | Command |
|-----------|---------|
| Just checking if it works | `--epochs 3 --max_per_class 50` |
| Getting decent results | `--epochs 10 --max_per_class 100` |
| Best possible results | `--epochs 15` (no limit, overnight) |

If Python gets killed mid-run (out of memory), reduce `--batch_size` to `16` or lower `--max_per_class`.

---

## References

Li, Z.; Li, H.; Meng, L. Model Compression for Deep Neural Networks: A Survey. *Computers* **2023**, *12*, 60. https://doi.org/10.3390/computers12030060
