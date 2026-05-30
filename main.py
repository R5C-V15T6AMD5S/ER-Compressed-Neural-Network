"""
main.py — Main pipeline that connects all 4 parts of the project.

Runs the full experiment in order:
    1. Train the original CNN           (Person 1 + 2)
    2. Apply pruning + compare          (Person 3)
    3. Apply quantization + compare     (Person 4)
    4. Print final summary table        (all together)

This main file includes small compatibility shims so it works even if the
4 person files use slightly different function names or signatures.

Usage:
    python main.py --data_dir ./fruits-360-100x100 --epochs 10 --max_per_class 100

Quick test (small data, fast):
    python main.py --data_dir ./fruits-360-100x100 --epochs 3 --max_per_class 50
"""

import argparse
import os
import numpy as np

from logger import Logger
from person1_model import SimpleCNN

# ─── PERSON 2 — TRAINING ──────────────────────────────────────────────────────
from person2_train import (
    load_dataset, get_batches, cross_entropy_loss,
    SGDMomentum, compute_accuracy,
    save_model, load_model,
)

# ─── PERSON 3 — PRUNING ───────────────────────────────────────────────────────
# Person 3 named their functions a bit differently — alias them here so the
# rest of this file stays readable and main.py doesn't depend on naming choices.
from person3_pruning import (
    prune_model,                            # used as `prune` below
    fine_tune as p3_fine_tune,
    count_nonzero_weights,
    apply_masks,
)

# ─── PERSON 4 — QUANTIZATION ──────────────────────────────────────────────────
from person4_quantization import (
    quantize_model,
    estimate_int8_size_kb,
    get_model_size_kb,        # reused by all 3 steps
    measure_inference_ms,     # reused by all 3 steps
)


# ─── COMPATIBILITY SHIMS ──────────────────────────────────────────────────────
# main.py and person 3 have slightly different APIs, so wrap them here.

def count_sparsity(model):
    """Return % of weights that are exactly zero (uses Person 3's helper)."""
    return count_nonzero_weights(model)["sparsity"]


def run_finetune(model, masks, X_train, y_train, X_test, y_test,
                 epochs, lr, batch_size, momentum=0.9):
    """
    Wrap Person 3's fine_tune so it has the same signature main.py expects.
    Person 3's fine_tune needs the pruning masks as well.
    """
    return p3_fine_tune(
        model=model,
        X_train=X_train, y_train=y_train,
        X_test=X_test, y_test=y_test,
        masks=masks,
        epochs=epochs, batch_size=batch_size,
        lr=lr, momentum=momentum,
    )


# ─── STEP 1 & 2: TRAIN ────────────────────────────────────────────────────────

def run_training(args, X_train, y_train, X_test, y_test, num_classes, classes, log):
    log("")
    log("=" * 60)
    log("  STEP 1 — Training original CNN (Person 1 + Person 2)")
    log("=" * 60)

    os.makedirs(os.path.dirname(args.save_path) or "models", exist_ok=True)
    model     = SimpleCNN(num_classes=num_classes)
    optimizer = SGDMomentum(model, lr=args.lr, momentum=0.9)

    log(f"[Model] SimpleCNN — {model.get_param_count():,} parameters")
    log(f"[Train] {len(X_train)} samples | {args.epochs} epochs | "
        f"lr={args.lr} | batch={args.batch_size}")
    log("")

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        total_loss, num_batches = 0.0, 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            probs = model.forward(X_batch)
            loss, d_probs = cross_entropy_loss(probs, y_batch)
            total_loss   += loss
            num_batches  += 1
            model.backward(d_probs)
            optimizer.step(model)

        avg_loss = total_loss / num_batches
        test_acc = compute_accuracy(model, X_test, y_test, args.batch_size)
        log(f"  Epoch [{epoch:>3}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} | Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, args.save_path, classes=classes)
            log(f"    ↑ Best model saved ({best_acc:.2f}%)")

    # Reload best model
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    acc  = compute_accuracy(model, X_test, y_test, args.batch_size)
    size = get_model_size_kb(model)
    inf  = measure_inference_ms(model)

    log(f"\n  ✓ Training done. Best accuracy: {acc:.2f}%")
    return model, {"acc": acc, "size": size, "inf": inf, "sparsity": 0.0}


# ─── STEP 3: PRUNING ──────────────────────────────────────────────────────────

def run_pruning(args, X_train, y_train, X_test, y_test, num_classes, classes, log):
    log("")
    log("=" * 60)
    log("  STEP 2 — Pruning (Person 3)")
    log("=" * 60)

    # Always reload the best trained model, never the in-memory one
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    log(f"[Pruning] Strategy: {args.prune_strategy} | "
        f"Amount: {args.prune_amount*100:.0f}%")

    # Person 3's prune_model returns the masks dict — we need it for fine-tuning
    masks = prune_model(model, args.prune_strategy, args.prune_amount)

    acc_before = compute_accuracy(model, X_test, y_test, args.batch_size)
    log(f"  Accuracy after pruning (before fine-tune): {acc_before:.2f}%")

    # Fine-tune so pruned weights stay zero but live ones can recover
    if args.finetune_epochs > 0:
        log(f"\n  [Fine-tuning] {args.finetune_epochs} epoch(s) "
            f"at lr={args.finetune_lr}")
        run_finetune(
            model, masks,
            X_train, y_train, X_test, y_test,
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            batch_size=args.batch_size,
        )

    os.makedirs("models", exist_ok=True)
    save_model(model, "models/cnn_pruned", classes=classes)

    acc      = compute_accuracy(model, X_test, y_test, args.batch_size)
    size     = get_model_size_kb(model)
    inf      = measure_inference_ms(model)
    sparsity = count_sparsity(model)

    log(f"\n  ✓ Pruning done. Accuracy: {acc:.2f}% | Sparsity: {sparsity:.1f}%")
    return {"acc": acc, "size": size, "inf": inf, "sparsity": sparsity}


# ─── STEP 4: QUANTIZATION ─────────────────────────────────────────────────────

def run_quantization(args, X_test, y_test, num_classes, classes, log):
    log("")
    log("=" * 60)
    log("  STEP 3 — Quantization (Person 4)")
    log("=" * 60)

    # Reload the ORIGINAL trained model, not the pruned one,
    # so the two compression techniques are compared independently
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    log(f"[Quantization] PTQ ({args.quant_strategy}): FP32 → INT8 → FP32 (dequant)")
    model, metadata = quantize_model(model, strategy=args.quant_strategy)

    os.makedirs("models", exist_ok=True)
    save_model(model, "models/cnn_quantized", classes=classes)

    acc              = compute_accuracy(model, X_test, y_test, args.batch_size)
    size             = get_model_size_kb(model)
    inf              = measure_inference_ms(model)
    fp32_kb, int8_kb = estimate_int8_size_kb(model)

    log(f"\n  ✓ Quantization done. Accuracy: {acc:.2f}% | "
        f"Theoretical INT8 size: {int8_kb:.1f} KB (vs {fp32_kb:.1f} KB FP32)")

    return {"acc": acc, "size": size, "inf": inf, "sparsity": 0.0,
            "int8_kb": int8_kb, "fp32_kb": fp32_kb}


# ─── FINAL SUMMARY TABLE ──────────────────────────────────────────────────────

def print_final_summary(orig, pruned, quant, log):
    log("")
    log("╔" + "═" * 70 + "╗")
    log("║" + "  FINAL COMPARISON SUMMARY".center(70) + "║")
    log("║" + "  SimpleCNN on Fruits-360 — Original vs Compressed".center(70) + "║")
    log("╠" + "═" * 70 + "╣")
    log(f"║  {'Metric':<26} {'Original':>12} {'Pruned (50%)':>14} {'Quantized':>12}  ║")
    log("╠" + "═" * 70 + "╣")

    def row(label, o, p, q, fmt="{:.2f}"):
        log(f"║  {label:<26} {fmt.format(o):>12} {fmt.format(p):>14} {fmt.format(q):>12}  ║")

    row("Accuracy (%)",    orig['acc'],      pruned['acc'],      quant['acc'])
    row("Model Size (KB)", orig['size'],     pruned['size'],     quant['size'])
    row("Inference (ms)",  orig['inf'],      pruned['inf'],      quant['inf'])
    row("Sparsity (%)",    orig['sparsity'], pruned['sparsity'], quant['sparsity'])

    log("╠" + "═" * 70 + "╣")

    acc_drop_p = orig['acc'] - pruned['acc']
    acc_drop_q = orig['acc'] - quant['acc']
    speedup_p  = orig['inf'] / pruned['inf'] if pruned['inf'] > 0 else 0
    speedup_q  = orig['inf'] / quant['inf']  if quant['inf']  > 0 else 0

    log(f"║  {'Accuracy Drop':<26} {'—':>12} {acc_drop_p:>13.2f}% {acc_drop_q:>11.2f}%  ║")
    log(f"║  {'Speedup':<26} {'—':>12} {speedup_p:>13.2f}x {speedup_q:>11.2f}x  ║")

    if 'int8_kb' in quant:
        size_reduction = (1 - quant['int8_kb'] / quant['fp32_kb']) * 100
        log(f"║  {'Theoretical Size Reduction':<26} {'—':>12} {'—':>14} {size_reduction:>11.1f}%  ║")

    log("╚" + "═" * 70 + "╝")
    log("")
    log("  Techniques from: 'Model Compression for Deep Neural Networks'")
    log("  Li, Li & Meng, Computers 2023 — Section 2 (Pruning), Section 3 (Quantization)")
    log("")


# ─── MAIN ─────────────────────────────────────────────────────────────────────

def main(args):
    log = Logger("main")

    log("=" * 60)
    log("  CNN Model Compression — Group Project")
    log("  Fruits-360 Dataset | Pure NumPy Implementation")
    log("=" * 60)

    log("\n[Data] Loading dataset...")
    X_train, y_train, classes = load_dataset(
        args.data_dir, "Training", max_per_class=args.max_per_class
    )
    X_test, y_test, _ = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)

    trained_model, orig_stats = run_training(
        args, X_train, y_train, X_test, y_test, num_classes, classes, log
    )
    pruned_stats = run_pruning(
        args, X_train, y_train, X_test, y_test, num_classes, classes, log
    )
    quant_stats = run_quantization(
        args, X_test, y_test, num_classes, classes, log
    )

    log.section("FINAL SUMMARY")
    print_final_summary(orig_stats, pruned_stats, quant_stats, log)
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full CNN compression pipeline on Fruits-360"
    )

    # ── Data ──────────────────────────────────────────────────────────────────
    parser.add_argument("--data_dir",        type=str,   default="./fruits-360-100x100")
    parser.add_argument("--max_per_class",   type=int,   default=None)

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--epochs",          type=int,   default=10)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=0.01)
    parser.add_argument("--save_path",       type=str,   default="models/cnn_fruits")

    # ── Pruning ───────────────────────────────────────────────────────────────
    parser.add_argument("--prune_amount",    type=float, default=0.5)
    parser.add_argument("--prune_strategy",  type=str,   default="global",
                        choices=["global", "per_layer"])
    parser.add_argument("--finetune_epochs", type=int,   default=3)
    parser.add_argument("--finetune_lr",     type=float, default=0.001)

    # ── Quantization ──────────────────────────────────────────────────────────
    parser.add_argument("--quant_strategy",  type=str,   default="asymmetric",
                        choices=["symmetric", "asymmetric"])

    args = parser.parse_args()
    main(args)
