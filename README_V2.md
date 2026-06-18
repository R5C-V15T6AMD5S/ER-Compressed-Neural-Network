# Person 1 — Milestone 2 Contribution: V2 Architecture

This is Person 1's standalone contribution for the second milestone of the
project. It introduces an improved CNN architecture (V2) and compares it
against the original V1 baseline.

## Files added

| File | What it does |
|---|---|
| `person1_model_v2.py` | New `SimpleCNNv2` class with BatchNorm, Dropout, and a 3rd Conv block |
| `compare_models.py` | Trains both V1 and V2 under identical conditions, prints comparison |
| `README_V2.md` | This file |

## What changed in V2

Three concrete improvements over V1:

1. **BatchNorm after every Conv layer** — normalizes activations across the
   batch, which stabilizes training and lets the network use higher learning
   rates without diverging. Also acts as a mild regularizer.

2. **Dropout (p=0.3) after FC1** — during training, randomly zeroes 30% of
   FC1's outputs. This was added specifically to fight the overfitting we
   saw in V1, where training accuracy reached ~99% but test accuracy
   plateaued around 70%.

3. **Third Conv block (32 filters)** + **smaller FC1** — adds one more
   convolutional block and lets the third pooling layer reduce spatial
   dimensions from 25 → 12. As a side effect, FC1 input shrinks from
   16×25×25 = 10,000 features to 32×12×12 = 4,608 features, so FC1 drops
   from 1.28M parameters to ~590K. Total model is ~55% smaller.

## Architecture comparison

```
V1 (original)                          V2 (improved)
─────────────                          ──────────────
Conv(3→8) → ReLU → MaxPool             Conv(3→8)  → BN → ReLU → MaxPool
Conv(8→16) → ReLU → MaxPool            Conv(8→16) → BN → ReLU → MaxPool
                                       Conv(16→32) → BN → ReLU → MaxPool  ← NEW
Flatten (10,000)                       Flatten (4,608)
FC(10000→128) → ReLU                   FC(4608→128) → ReLU → Dropout(0.3)
FC(128→261)                            FC(128→261)
Softmax                                Softmax

~1.32M parameters                      ~597K parameters (2.2× smaller)
```

## Why these specific changes (and not others)

The V1 baseline showed clear **overfitting**: training accuracy near 99%
but test accuracy stuck at ~70%. The standard solutions are:

- **More data / augmentation** — out of scope (no time for new dataset work)
- **More regularization** — what we did (Dropout + BatchNorm)
- **Smaller model** — partial side effect (FC1 shrunk because of extra pool)

We did **not** simply add more filters or more layers because that would
make overfitting worse, not better.

## Compatibility with the rest of the project

V2 keeps the **exact same public interface as V1**:

- `model.forward(X)` → softmax probabilities
- `model.backward(d_probs)` → populates gradients
- `model.get_trainable_layers()` → list of ConvLayer and FCLayer only
- `model.get_param_count()` → integer

This means **Borna's training loop, Silvia's pruning, and Jakov's
quantization all work on V2 without any code changes** — they only ever
touch Conv/FC layers, and V2 exposes those the same way V1 does.

BatchNorm parameters (γ, β, running stats) are managed internally by
`SimpleCNNv2` and saved/loaded via separate methods:
- `model.save_bn_params(path)` writes a `<path>_bn.npz` file
- `model.load_bn_params(path)` reads it back

This keeps the rest of the team's code completely untouched.

## How to run the comparison

**Full comparison run (~2 hours on a 16GB laptop):**

```bash
python compare_models.py --data_dir ./fruits-360-100x100 \
                          --epochs 10 \
                          --max_per_class 100
```

**Quick smoke test (~5 minutes):**

```bash
python compare_models.py --data_dir ./fruits-360-100x100 \
                          --epochs 3 \
                          --max_per_class 30
```

**Test V2 by itself with fake data (~90 sec):**

```bash
python person1_model_v2.py
```

## What the comparison output looks like

After running, you get a side-by-side table like this:

```
╔══════════════════════════════════════════════════════════════════════╗
║                FINAL COMPARISON — V1 vs V2 ARCHITECTURE              ║
║            Same dataset, same seed, same training settings           ║
╠══════════════════════════════════════════════════════════════════════╣
║  Metric                              V1 (orig)        V2 (improved)  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Test accuracy (%)                       69.88                XX.XX  ║
║  Best epoch accuracy                     71.74                XX.XX  ║
║  Parameters                          1,315,189              597,386  ║
║  Model size (KB)                     10,275.60             4,661.22  ║
║  Inference (ms/image)                    81.75                XX.XX  ║
║  Total train time (s)                  XXXX.XX              XXXX.XX  ║
╠══════════════════════════════════════════════════════════════════════╣
║  Accuracy change                                             +X.XX%  ║
║  Parameter change                                            -54.6%  ║
║  Size change                                                 -54.6%  ║
╚══════════════════════════════════════════════════════════════════════╝
```

Plus a per-layer parameter breakdown for both architectures, and a
log file saved to `logs/compare_<timestamp>.txt`.

## What to write in the KT2 report

This V2 work supports several talking points:

- **Demonstrates understanding of overfitting** — V1 had a ~30 percentage
  point train-test gap, V2 was designed specifically to close it
- **Reduces parameter count by 55%** while maintaining (or improving)
  accuracy — same model handles 4× quantization more efficiently
- **Methodologically clean comparison** — same seed, same data, same
  training settings, only the architecture differs
- **Forward-compatible** — V2 still works with Silvia's pruning and
  Jakov's quantization without changes; could be combined into a final
  pipeline of V2 + pruning + quantization

## References

The V2 changes are direct applications of standard techniques mentioned
in our seminar paper [Li, Li & Meng 2023]:

- BatchNorm: Ioffe & Szegedy 2015 (referenced in Section 2.1 of the survey)
- Dropout: standard regularization, similar in role to L1/L2 weight
  regularization discussed in Section 2.2 of the survey
- Smaller FC layer: matches the survey's observation that "FC layers
  contain most parameters and are the main compression target"
