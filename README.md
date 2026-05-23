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
   - [Person 2 — Training](#person-2--training)
   - [Person 3 — Pruning](#person-3--pruning)
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
| 2 | Person 2 | `person2_train.py` | Training loop, loss function, optimizer, save/load |
| 3 | Person 3 | `person3_pruning.py` | Weight pruning — removes 50% of weights |
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
├── person1_model.py        ← CNN architecture (Person 1)
├── person2_train.py        ← Training loop + optimizer (Person 2)
├── person3_pruning.py      ← Weight pruning (Person 3)
├── person4_quantization.py ← Quantization (Person 4)
├── main.py                 ← Full pipeline — runs everything in order
├── logger.py               ← Shared logging utility (saves to logs/)
├── prepare_dataset.py      ← Optional: create a smaller dataset subset
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
└── cnn_fruits.npz          ← Saved model weights (created after training)
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

### Person 2 — Training

**File:** `person2_train.py`  
**Depends on:** `person1_model.py`, dataset  
**What it does:** Loads the dataset, trains `SimpleCNN` using SGD with momentum and cross-entropy loss, saves the best model to `cnn_fruits.npz`

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
Model saved to: cnn_fruits.npz
```

**What to look for:**
- Loss should decrease each epoch
- Accuracy should increase each epoch
- `cnn_fruits.npz` should appear in the project folder when done

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--epochs` | 15 | Training epochs |
| `--batch_size` | 32 | Images per batch |
| `--lr` | 0.01 | Learning rate |
| `--save_path` | `cnn_fruits` | Filename for saved model |
| `--max_per_class` | None (all) | Limit images per class |

> **Note:** Once this finishes and `cnn_fruits.npz` exists, Person 3 and Person 4 can run their parts.

---

### Person 3 — Pruning

**File:** `person3_pruning.py`  
**Depends on:** `person1_model.py`, `person2_train.py`, a saved `cnn_fruits.npz` from Person 2  
**What it does:** Loads the trained model, removes the 50% lowest-magnitude weights (sets them to zero), fine-tunes for a few epochs to recover accuracy, then compares results

#### Before you start

You need `cnn_fruits.npz` to exist in the project folder. If it doesn't exist yet, run Person 2's script first:

```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

This takes ~15–25 min and creates `cnn_fruits.npz`.

#### How to test your part

```bash
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path cnn_fruits --max_per_class 50
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
  Expected file: cnn_fruits.npz

  Person 3 needs a trained model from Person 2 first.
  Run this command to train one:

    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--model_path` | `cnn_fruits` | Trained model to load |
| `--save_path` | `cnn_pruned` | Where to save pruned model |
| `--strategy` | `global` | `global` or `per_layer` |
| `--amount` | 0.5 | Fraction of weights to prune |
| `--finetune_epochs` | 3 | Epochs to fine-tune after pruning |
| `--finetune_lr` | 0.001 | Learning rate during fine-tuning |
| `--max_per_class` | None (all) | Limit images per class |

**Try a different pruning amount:**
```bash
# More aggressive — 70% of weights removed
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path cnn_fruits --amount 0.7 --max_per_class 50

# Less aggressive — 30% of weights removed
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path cnn_fruits --amount 0.3 --max_per_class 50
```

**Try per-layer strategy instead of global:**
```bash
python person3_pruning.py --data_dir ./fruits-360-100x100 --model_path cnn_fruits --strategy per_layer --max_per_class 50
```

---

### Person 4 — Quantization

**File:** `person4_quantization.py`  
**Depends on:** `person1_model.py`, `person2_train.py`, a saved `cnn_fruits.npz` from Person 2  
**What it does:** Loads the trained model, converts all weights from FP32 to INT8 using scale `S` and zero-point `Z` (Equations 1–5 from the survey), dequantizes back to FP32, then measures the accuracy impact and theoretical size savings

#### Before you start

You need `cnn_fruits.npz` to exist in the project folder. If it doesn't, run Person 2's script first:

```bash
python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

#### How to test your part

```bash
python person4_quantization.py --data_dir ./fruits-360-100x100 --model_path cnn_fruits --max_per_class 50
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
  Expected file: cnn_fruits.npz

  Person 4 needs a trained model from Person 2 first.
  Run this command to train one:

    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
```

**All options:**

| Option | Default | What it does |
|--------|---------|-------------|
| `--data_dir` | `./fruits-360` | Path to dataset |
| `--model_path` | `cnn_fruits` | Trained model to load |
| `--save_path` | `cnn_quantized` | Where to save quantized model |
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
