"""
compare_models.py — Trains BOTH SimpleCNN (V1) and SimpleCNNv2 under
identical conditions and prints a side-by-side comparison.

This is Person 1's contribution to Milestone 2 — demonstrates whether
the architectural improvements in V2 (BatchNorm, Dropout, extra Conv,
smaller FC1) actually help vs the V1 baseline.

Same seed, same dataset slice, same epochs, same optimizer settings —
the only difference is the architecture.

Usage:
    python compare_models.py --data_dir ./fruits-360-100x100 --epochs 10 --max_per_class 100

Quick test:
    python compare_models.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 30
"""

import argparse
import os
import time
import numpy as np

from logger import Logger
from person1_model    import SimpleCNN      # V1
from person1_model_v2 import SimpleCNNv2    # V2

from person2_train import (
    load_dataset, get_batches, cross_entropy_loss,
    SGDMomentum, compute_accuracy,
    save_model, load_model,
)


# ─── HELPERS ──────────────────────────────────────────────────────────────────

def measure_inference_ms(model, runs=50, img_size=100, is_v2=False):
    """Average inference time per single-image forward pass."""
    if is_v2:
        model.eval()
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(3):
        model.forward(dummy)
    start = time.time()
    for _ in range(runs):
        model.forward(dummy)
    return (time.time() - start) / runs * 1000


def model_size_kb(model, path):
    """Save model to disk and report .npz size."""
    save_model(model, path, classes=None)
    size = os.path.getsize(path + ".npz") / 1024
    return size


def get_param_breakdown(model):
    """Print per-layer parameter counts."""
    rows = []
    for i, l in enumerate(model.get_trainable_layers()):
        kind = "Conv" if hasattr(l, 'filters') else "FC"
        w    = l.filters if hasattr(l, 'filters') else l.weights
        params = w.size + l.biases.size
        rows.append((i, kind, str(w.shape), params))
    return rows


# ─── TRAINING LOOPS ───────────────────────────────────────────────────────────

def train_v1(args, X_train, y_train, X_test, y_test, num_classes, classes, log, seed):
    """Train original V1 architecture."""
    np.random.seed(seed)
    log("")
    log("=" * 64)
    log("  TRAINING V1 — SimpleCNN (original)")
    log("=" * 64)

    model = SimpleCNN(num_classes=num_classes)
    opt   = SGDMomentum(model, lr=args.lr, momentum=0.9)

    log(f"  Parameters : {model.get_param_count():,}")
    log(f"  Epochs     : {args.epochs}  |  lr={args.lr}  |  batch={args.batch_size}")
    log("")

    best_acc = 0.0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        ep_start = time.time()

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            probs = model.forward(X_batch)
            loss, d_probs = cross_entropy_loss(probs, y_batch)
            total_loss   += loss
            n_batches    += 1
            model.backward(d_probs)
            opt.step(model)

        avg_loss = total_loss / n_batches
        test_acc = compute_accuracy(model, X_test, y_test, args.batch_size)
        ep_time  = time.time() - ep_start
        log(f"  V1 Epoch [{epoch:>2}/{args.epochs}]  Loss: {avg_loss:.4f}  "
            f"|  Acc: {test_acc:.2f}%  |  {ep_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, "models/v1_best", classes=classes)

    total_time = time.time() - train_start

    # Reload best
    model = load_model(SimpleCNN(num_classes=num_classes), "models/v1_best")
    final_acc = compute_accuracy(model, X_test, y_test, args.batch_size)

    return {
        "model":      model,
        "best_acc":   best_acc,
        "final_acc":  final_acc,
        "params":     model.get_param_count(),
        "size_kb":    model_size_kb(model, "models/v1_size_tmp"),
        "inf_ms":     measure_inference_ms(model, is_v2=False),
        "train_time": total_time,
        "breakdown":  get_param_breakdown(model),
    }


def train_v2(args, X_train, y_train, X_test, y_test, num_classes, classes, log, seed):
    """Train V2 architecture (BatchNorm + Dropout + extra Conv)."""
    np.random.seed(seed)
    log("")
    log("=" * 64)
    log("  TRAINING V2 — SimpleCNNv2 (BatchNorm + Dropout + 3rd Conv)")
    log("=" * 64)

    model = SimpleCNNv2(num_classes=num_classes, dropout_p=args.dropout_p)
    # Keep BN learning rate in sync with main optimizer
    model.set_bn_lr(args.lr)
    opt = SGDMomentum(model, lr=args.lr, momentum=0.9)

    log(f"  Parameters : {model.get_param_count():,}")
    log(f"  Dropout p  : {args.dropout_p}")
    log(f"  Epochs     : {args.epochs}  |  lr={args.lr}  |  batch={args.batch_size}")
    log("")

    best_acc = 0.0
    train_start = time.time()

    for epoch in range(1, args.epochs + 1):
        total_loss, n_batches = 0.0, 0
        ep_start = time.time()

        # Training mode
        model.train()

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            probs = model.forward(X_batch)
            loss, d_probs = cross_entropy_loss(probs, y_batch)
            total_loss   += loss
            n_batches    += 1
            model.backward(d_probs)   # also updates BN gamma/beta
            opt.step(model)           # updates Conv/FC

        avg_loss = total_loss / n_batches

        # Eval mode for accuracy
        model.eval()
        test_acc = compute_accuracy(model, X_test, y_test, args.batch_size)
        ep_time  = time.time() - ep_start

        log(f"  V2 Epoch [{epoch:>2}/{args.epochs}]  Loss: {avg_loss:.4f}  "
            f"|  Acc: {test_acc:.2f}%  |  {ep_time:.1f}s")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, "models/v2_best", classes=classes)
            model.save_bn_params("models/v2_best")

    total_time = time.time() - train_start

    # Reload best
    model = SimpleCNNv2(num_classes=num_classes, dropout_p=args.dropout_p)
    load_model(model, "models/v2_best")
    model.load_bn_params("models/v2_best")
    model.eval()
    final_acc = compute_accuracy(model, X_test, y_test, args.batch_size)

    return {
        "model":      model,
        "best_acc":   best_acc,
        "final_acc":  final_acc,
        "params":     model.get_param_count(),
        "size_kb":    model_size_kb(model, "models/v2_size_tmp"),
        "inf_ms":     measure_inference_ms(model, is_v2=True),
        "train_time": total_time,
        "breakdown":  get_param_breakdown(model),
    }


# ─── COMPARISON TABLE ─────────────────────────────────────────────────────────

def print_comparison(v1, v2, log):
    log("")
    log("╔" + "═" * 70 + "╗")
    log("║" + "  FINAL COMPARISON — V1 vs V2 ARCHITECTURE".center(70) + "║")
    log("║" + "  Same dataset, same seed, same training settings".center(70) + "║")
    log("╠" + "═" * 70 + "╣")
    log(f"║  {'Metric':<28} {'V1 (orig)':>16} {'V2 (improved)':>20}  ║")
    log("╠" + "═" * 70 + "╣")

    def row(label, v1v, v2v, fmt="{:,.2f}", unit=""):
        s1 = (fmt.format(v1v) + unit) if not isinstance(v1v, str) else v1v
        s2 = (fmt.format(v2v) + unit) if not isinstance(v2v, str) else v2v
        log(f"║  {label:<28} {s1:>16} {s2:>20}  ║")

    row("Test accuracy (%)",     v1['final_acc'],   v2['final_acc'])
    row("Best epoch accuracy",   v1['best_acc'],    v2['best_acc'])
    row("Parameters",            v1['params'],      v2['params'], fmt="{:,d}")
    row("Model size (KB)",       v1['size_kb'],     v2['size_kb'])
    row("Inference (ms/image)",  v1['inf_ms'],      v2['inf_ms'])
    row("Total train time (s)",  v1['train_time'],  v2['train_time'])

    log("╠" + "═" * 70 + "╣")

    acc_diff   = v2['final_acc'] - v1['final_acc']
    param_diff = (v2['params'] - v1['params']) / v1['params'] * 100
    size_diff  = (v2['size_kb'] - v1['size_kb']) / v1['size_kb'] * 100
    inf_diff   = (v2['inf_ms']  - v1['inf_ms'])  / v1['inf_ms']  * 100
    time_diff  = (v2['train_time'] - v1['train_time']) / v1['train_time'] * 100

    log(f"║  {'Accuracy change':<28} {'':<16} {acc_diff:>+19.2f}%  ║")
    log(f"║  {'Parameter change':<28} {'':<16} {param_diff:>+19.1f}%  ║")
    log(f"║  {'Size change':<28} {'':<16} {size_diff:>+19.1f}%  ║")
    log(f"║  {'Inference change':<28} {'':<16} {inf_diff:>+19.1f}%  ║")
    log(f"║  {'Train time change':<28} {'':<16} {time_diff:>+19.1f}%  ║")

    log("╚" + "═" * 70 + "╝")
    log("")

    # Verdict
    log("─" * 70)
    log("  CONCLUSION")
    log("─" * 70)
    if acc_diff > 0.5:
        log(f"  ✓ V2 IS BETTER on test accuracy (+{acc_diff:.2f}%)")
    elif acc_diff > -0.5:
        log(f"  = V2 is EQUIVALENT on test accuracy ({acc_diff:+.2f}%)")
    else:
        log(f"  ✗ V2 is WORSE on test accuracy ({acc_diff:+.2f}%)")

    if v2['params'] < v1['params']:
        log(f"  ✓ V2 has FEWER parameters ({param_diff:+.1f}%)")
    else:
        log(f"  → V2 has MORE parameters ({param_diff:+.1f}%)")

    log("")


def print_breakdowns(v1, v2, log):
    log("─" * 70)
    log("  LAYER-BY-LAYER PARAMETER BREAKDOWN")
    log("─" * 70)
    log(f"\n  V1 — SimpleCNN ({v1['params']:,} params)")
    log(f"    {'idx':<4} {'type':<6} {'shape':<22} {'params':>14}")
    for idx, kind, shape, params in v1['breakdown']:
        log(f"    {idx:<4} {kind:<6} {shape:<22} {params:>14,}")

    log(f"\n  V2 — SimpleCNNv2 ({v2['params']:,} params total, "
        f"incl. BatchNorm)")
    log(f"    {'idx':<4} {'type':<6} {'shape':<22} {'params':>14}")
    for idx, kind, shape, params in v2['breakdown']:
        log(f"    {idx:<4} {kind:<6} {shape:<22} {params:>14,}")
    log("")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(args):
    log = Logger("compare")

    log("=" * 64)
    log("  V1 vs V2 ARCHITECTURE COMPARISON")
    log("  Person 1 — Milestone 2 contribution")
    log("=" * 64)
    log(f"  Dataset       : {args.data_dir}")
    log(f"  Max per class : {args.max_per_class}")
    log(f"  Epochs        : {args.epochs}")
    log(f"  Batch size    : {args.batch_size}")
    log(f"  Learning rate : {args.lr}")
    log(f"  Seed          : {args.seed}")
    log(f"  Dropout p     : {args.dropout_p}")

    os.makedirs("models", exist_ok=True)

    # ── Load data once, share between both runs ──────────────────────────────
    log("\n[Data] Loading dataset...")
    X_train, y_train, classes = load_dataset(
        args.data_dir, "Training", max_per_class=args.max_per_class
    )
    X_test, y_test, _ = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)
    log(f"  Train: {len(X_train)} images  |  Test: {len(X_test)} images  "
        f"|  Classes: {num_classes}\n")

    # ── Train V1 ─────────────────────────────────────────────────────────────
    v1_results = train_v1(args, X_train, y_train, X_test, y_test,
                          num_classes, classes, log, seed=args.seed)

    # ── Train V2 (re-seed to identical state) ────────────────────────────────
    v2_results = train_v2(args, X_train, y_train, X_test, y_test,
                          num_classes, classes, log, seed=args.seed)

    # ── Print comparison ─────────────────────────────────────────────────────
    print_comparison(v1_results, v2_results, log)
    print_breakdowns(v1_results, v2_results, log)

    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Compare V1 SimpleCNN vs V2 SimpleCNNv2 under identical conditions"
    )
    parser.add_argument("--data_dir",      type=str,   default="./fruits-360-100x100")
    parser.add_argument("--max_per_class", type=int,   default=100)
    parser.add_argument("--epochs",        type=int,   default=10)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=0.01)
    parser.add_argument("--seed",          type=int,   default=42)
    parser.add_argument("--dropout_p",     type=float, default=0.3)
    args = parser.parse_args()
    main(args)
