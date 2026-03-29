"""
main.py — Main pipeline connecting all 4 parts of the project.

Runs the full experiment in order:
    1. Train the original CNN           (Person 1 + 2)
    2. Apply pruning + compare          (Person 3)
    3. Apply quantization + compare     (Person 4)
    4. Print final summary table        (all together)

Usage:
    python main.py --data_dir ./fruits-360 --epochs 15

Quick test (small data, fast):
    python main.py --data_dir ./fruits-360 --epochs 3 --max_per_class 50
"""

import argparse
import numpy as np

from person1_model import SimpleCNN
from person2_train import (load_dataset, get_batches, cross_entropy_loss,
                            SGDMomentum, compute_accuracy,
                            save_model, load_model)
from person3_pruning  import (global_pruning, fine_tune as prune_finetune,
                               count_sparsity, get_model_size_kb,
                               measure_inference_ms)
from person4_quantization import (quantize_model, estimate_int8_size_kb)


# ─── STEP 1 & 2: TRAIN ───────────────────────────────────────────────────────

def run_training(args, X_train, y_train, X_test, y_test, num_classes):
    print("\n" + "=" * 60)
    print("  STEP 1 — Training original CNN (Person 1 + Person 2)")
    print("=" * 60)

    model     = SimpleCNN(num_classes=num_classes)
    optimizer = SGDMomentum(model, lr=args.lr, momentum=0.9)

    print(f"[Model] SimpleCNN — {model.get_param_count():,} parameters")
    print(f"[Train] {len(X_train)} samples | {args.epochs} epochs | "
          f"lr={args.lr} | batch={args.batch_size}\n")

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
        print(f"  Epoch [{epoch:>3}/{args.epochs}] "
              f"Loss: {avg_loss:.4f} | Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, args.save_path)
            print(f"    ↑ Best model saved ({best_acc:.2f}%)")

    # Reload best
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    acc  = compute_accuracy(model, X_test, y_test, args.batch_size)
    size = get_model_size_kb(model)
    inf  = measure_inference_ms(model)

    print(f"\n  ✓ Training done. Best accuracy: {acc:.2f}%")
    return model, {"acc": acc, "size": size, "inf": inf, "sparsity": 0.0}


# ─── STEP 3: PRUNING ─────────────────────────────────────────────────────────

def run_pruning(args, trained_model, X_train, y_train, X_test, y_test, num_classes):
    print("\n" + "=" * 60)
    print("  STEP 2 — Pruning (Person 3)")
    print("=" * 60)

    # Work on a fresh copy so we don't modify the original
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    print(f"[Pruning] Strategy: global | Amount: {args.prune_amount*100:.0f}%")
    model = global_pruning(model, args.prune_amount)

    acc_before = compute_accuracy(model, X_test, y_test, args.batch_size)
    print(f"  Accuracy before fine-tune: {acc_before:.2f}%")

    model = prune_finetune(model, X_train, y_train, X_test, y_test,
                           epochs=args.finetune_epochs,
                           lr=args.finetune_lr,
                           batch_size=args.batch_size)

    save_model(model, "cnn_pruned")

    acc      = compute_accuracy(model, X_test, y_test, args.batch_size)
    size     = get_model_size_kb(model)
    inf      = measure_inference_ms(model)
    sparsity = count_sparsity(model)

    print(f"\n  ✓ Pruning done. Accuracy: {acc:.2f}% | Sparsity: {sparsity:.1f}%")
    return {"acc": acc, "size": size, "inf": inf, "sparsity": sparsity}


# ─── STEP 4: QUANTIZATION ────────────────────────────────────────────────────

def run_quantization(args, X_test, y_test, num_classes):
    print("\n" + "=" * 60)
    print("  STEP 3 — Quantization (Person 4)")
    print("=" * 60)

    # Work on a fresh copy
    model = load_model(SimpleCNN(num_classes=num_classes), args.save_path)

    print("[Quantization] Applying PTQ: FP32 → INT8 → FP32 (dequant)...")
    model, metadata = quantize_model(model)

    save_model(model, "cnn_quantized")

    acc            = compute_accuracy(model, X_test, y_test, args.batch_size)
    size           = get_model_size_kb(model)
    inf            = measure_inference_ms(model)
    fp32_kb, int8_kb = estimate_int8_size_kb(model)

    print(f"\n  ✓ Quantization done. Accuracy: {acc:.2f}% | "
          f"Theoretical INT8 size: {int8_kb:.1f} KB (vs {fp32_kb:.1f} KB FP32)")
    return {"acc": acc, "size": size, "inf": inf,
            "sparsity": 0.0, "int8_kb": int8_kb, "fp32_kb": fp32_kb}


# ─── FINAL SUMMARY TABLE ─────────────────────────────────────────────────────

def print_final_summary(orig, pruned, quant):
    print("\n")
    print("╔" + "═" * 70 + "╗")
    print("║" + "  FINAL COMPARISON SUMMARY".center(70) + "║")
    print("║" + "  SimpleCNN on Fruits-360 — Original vs Compressed".center(70) + "║")
    print("╠" + "═" * 70 + "╣")
    print(f"║  {'Metric':<26} {'Original':>12} {'Pruned (50%)':>14} {'Quantized':>12}  ║")
    print("╠" + "═" * 70 + "╣")

    def row(label, o, p, q, fmt="{:.2f}"):
        print(f"║  {label:<26} {fmt.format(o):>12} {fmt.format(p):>14} {fmt.format(q):>12}  ║")

    row("Accuracy (%)",    orig['acc'],  pruned['acc'],  quant['acc'])
    row("Model Size (KB)", orig['size'], pruned['size'], quant['size'])
    row("Inference (ms)",  orig['inf'],  pruned['inf'],  quant['inf'])
    row("Sparsity (%)",    orig['sparsity'], pruned['sparsity'], quant['sparsity'])

    print("╠" + "═" * 70 + "╣")

    acc_drop_p  = orig['acc']  - pruned['acc']
    acc_drop_q  = orig['acc']  - quant['acc']
    speedup_p   = orig['inf']  / pruned['inf']  if pruned['inf']  > 0 else 0
    speedup_q   = orig['inf']  / quant['inf']   if quant['inf']   > 0 else 0

    print(f"║  {'Accuracy Drop':<26} {'—':>12} {acc_drop_p:>13.2f}% {acc_drop_q:>11.2f}%  ║")
    print(f"║  {'Speedup':<26} {'—':>12} {speedup_p:>13.2f}x {speedup_q:>11.2f}x  ║")

    if 'int8_kb' in quant:
        size_reduction = (1 - quant['int8_kb'] / quant['fp32_kb']) * 100
        print(f"║  {'Theoretical Size Reduction':<26} {'—':>12} {'—':>14} {size_reduction:>11.1f}%  ║")

    print("╚" + "═" * 70 + "╝")
    print()
    print("  Techniques from: 'Model Compression for Deep Neural Networks'")
    print("  Li, Li & Meng, Computers 2023 — Section 2 (Pruning), Section 3 (Quantization)\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(args):
    print("\n" + "=" * 60)
    print("  CNN Model Compression — Group Project")
    print("  Fruits-360 Dataset | Pure NumPy Implementation")
    print("=" * 60)

    # ── Load data once, share across all steps ────────────────────────────────
    print("\n[Data] Loading dataset...")
    X_train, y_train, classes = load_dataset(
        args.data_dir, "Training", max_per_class=args.max_per_class
    )
    X_test, y_test, _ = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)

    # ── Run all steps ─────────────────────────────────────────────────────────
    trained_model, orig_stats   = run_training(args, X_train, y_train,
                                               X_test, y_test, num_classes)
    pruned_stats                = run_pruning(args, trained_model,
                                              X_train, y_train,
                                              X_test, y_test, num_classes)
    quant_stats                 = run_quantization(args, X_test, y_test,
                                                   num_classes)

    # ── Final summary ─────────────────────────────────────────────────────────
    print_final_summary(orig_stats, pruned_stats, quant_stats)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Full CNN compression pipeline on Fruits-360"
    )
    parser.add_argument("--data_dir",        type=str,   default="./fruits-360",
                        help="Path to Fruits-360 dataset")
    parser.add_argument("--epochs",          type=int,   default=15,
                        help="Training epochs")
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--lr",              type=float, default=0.01,
                        help="Learning rate for training")
    parser.add_argument("--save_path",       type=str,   default="cnn_fruits",
                        help="Base path to save trained model")
    parser.add_argument("--prune_amount",    type=float, default=0.5,
                        help="Fraction of weights to prune (default: 0.5 = 50%%)")
    parser.add_argument("--finetune_epochs", type=int,   default=3,
                        help="Fine-tune epochs after pruning")
    parser.add_argument("--finetune_lr",     type=float, default=0.001)
    parser.add_argument("--max_per_class",   type=int,   default=None,
                        help="Limit images per class — use 50 for quick testing")
    args = parser.parse_args()
    main(args)
