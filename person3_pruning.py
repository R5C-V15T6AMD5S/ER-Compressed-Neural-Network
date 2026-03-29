"""
person3_pruning.py — Manual weight pruning implemented from scratch with NumPy.
PERSON 3's responsibility.

Implements the train → prune → fine-tune pipeline described in the survey
(Han et al., Section 2.2). No torch.nn.utils.prune — everything is manual.

Two pruning strategies:
    1. Global pruning  — removes the lowest-magnitude weights across ALL layers
    2. Per-layer pruning — removes a fixed % of weights within each layer

Usage:
    python person3_pruning.py --data_dir ./fruits-360 --model_path cnn_fruits
                              --strategy global --amount 0.5
"""

import os
import argparse
import numpy as np

from person1_model import SimpleCNN, ConvLayer, FCLayer
from person2_train import (load_dataset, get_batches, cross_entropy_loss,
                            SGDMomentum, compute_accuracy, save_model, load_model)


# ─── PRUNING UTILITIES ────────────────────────────────────────────────────────

def get_all_weights(model):
    """
    Collect all weight values (flattened) from all trainable layers.
    Used to find the global pruning threshold.
    Returns a single 1D array of all weight magnitudes.
    """
    all_weights = []
    for layer in model.get_trainable_layers():
        if hasattr(layer, 'filters'):
            all_weights.append(np.abs(layer.filters).flatten())
        else:
            all_weights.append(np.abs(layer.weights).flatten())
    return np.concatenate(all_weights)


def count_sparsity(model):
    """Count % of weights that are exactly zero (pruned)."""
    total, zeros = 0, 0
    for layer in model.get_trainable_layers():
        w = layer.filters if hasattr(layer, 'filters') else layer.weights
        total += w.size
        zeros += np.sum(w == 0)
    return 100.0 * zeros / total


# ─── PRUNING STRATEGIES ───────────────────────────────────────────────────────

def global_pruning(model, amount):
    """
    Global unstructured L1 pruning.

    Steps:
    1. Collect ALL weights across all layers into one array
    2. Find the threshold = amount-th percentile of absolute values
    3. Zero out any weight whose |value| < threshold

    This matches the approach from Han et al. described in the survey —
    removing globally unimportant connections regardless of which layer
    they belong to.

    amount: fraction of total weights to prune (e.g. 0.5 = 50%)
    """
    # Step 1: Find global threshold
    all_magnitudes = get_all_weights(model)
    threshold = np.percentile(all_magnitudes, amount * 100)
    print(f"  [Global Pruning] Threshold: {threshold:.6f} "
          f"(pruning weights with |w| < threshold)")

    # Step 2: Apply mask — zero out weights below threshold
    total_pruned = 0
    for layer in model.get_trainable_layers():
        if hasattr(layer, 'filters'):
            mask           = np.abs(layer.filters) >= threshold
            total_pruned  += np.sum(~mask)
            layer.filters *= mask
        else:
            mask           = np.abs(layer.weights) >= threshold
            total_pruned  += np.sum(~mask)
            layer.weights *= mask

    sparsity = count_sparsity(model)
    print(f"  [Global Pruning] {total_pruned:,} weights zeroed. "
          f"Sparsity: {sparsity:.1f}%")
    return model


def per_layer_pruning(model, amount):
    """
    Per-layer unstructured L1 pruning.

    Prunes 'amount' fraction of the lowest-magnitude weights within
    each layer independently. This gives more control than global pruning
    but may over-prune small layers.

    amount: fraction of weights to prune per layer (e.g. 0.4 = 40%)
    """
    print(f"  [Per-Layer Pruning] Pruning {amount*100:.0f}% of each layer...")
    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, 'filters'):
            w         = layer.filters
            threshold = np.percentile(np.abs(w), amount * 100)
            mask      = np.abs(w) >= threshold
            pruned    = np.sum(~mask)
            layer.filters *= mask
            print(f"    Layer {i} (Conv):  {pruned:>6,} weights pruned, "
                  f"threshold={threshold:.6f}")
        else:
            w         = layer.weights
            threshold = np.percentile(np.abs(w), amount * 100)
            mask      = np.abs(w) >= threshold
            pruned    = np.sum(~mask)
            layer.weights *= mask
            print(f"    Layer {i} (FC):    {pruned:>6,} weights pruned, "
                  f"threshold={threshold:.6f}")

    sparsity = count_sparsity(model)
    print(f"  [Per-Layer Pruning] Overall sparsity: {sparsity:.1f}%")
    return model


# ─── FINE-TUNING ─────────────────────────────────────────────────────────────

def fine_tune(model, X_train, y_train, X_test, y_test,
              epochs, lr, batch_size):
    """
    Fine-tune the pruned model to recover lost accuracy.
    Uses a lower learning rate than initial training.
    Crucially: after each gradient update, re-zeroes the pruned weights
    so that pruned connections stay pruned (frozen mask).
    """
    print(f"\n  [Fine-tuning] {epochs} epoch(s) at lr={lr}...")

    # Record which weights are zero — these stay zero (the pruning mask)
    masks = {}
    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, 'filters'):
            masks[i] = (layer.filters != 0)
        else:
            masks[i] = (layer.weights != 0)

    optimizer = SGDMomentum(model, lr=lr, momentum=0.9)

    for epoch in range(1, epochs + 1):
        total_loss  = 0.0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, batch_size):
            probs = model.forward(X_batch)
            loss, d_probs = cross_entropy_loss(probs, y_batch)
            total_loss   += loss
            num_batches  += 1
            model.backward(d_probs)
            optimizer.step(model)

            # Re-apply pruning mask — keep zeroed weights at zero
            for j, layer in enumerate(model.get_trainable_layers()):
                if hasattr(layer, 'filters'):
                    layer.filters *= masks[j]
                else:
                    layer.weights *= masks[j]

        avg_loss = total_loss / num_batches
        acc      = compute_accuracy(model, X_test, y_test, batch_size)
        print(f"    Fine-tune [{epoch}/{epochs}] Loss: {avg_loss:.4f} "
              f"| Accuracy: {acc:.2f}%")

    return model


# ─── MODEL SIZE ───────────────────────────────────────────────────────────────

def get_model_size_kb(model, path="tmp_model"):
    """Save model and return file size in KB."""
    save_model(model, path)
    size = os.path.getsize(path + ".npz") / 1024
    os.remove(path + ".npz")
    return size


def measure_inference_ms(model, runs=100, batch_size=1, img_size=100):
    """Average inference time in ms over multiple runs."""
    import time
    dummy = np.random.randn(batch_size, 3, img_size, img_size).astype(np.float32)
    for _ in range(5):  # warm up
        model.forward(dummy)
    start = time.time()
    for _ in range(runs):
        model.forward(dummy)
    return (time.time() - start) / runs * 1000


# ─── COMPARISON TABLE ────────────────────────────────────────────────────────

def print_comparison(orig, comp, strategy, amount):
    print("\n" + "=" * 57)
    print(f"  COMPARISON: Original vs Pruning ({strategy}, {amount*100:.0f}%)")
    print("=" * 57)
    print(f"{'Metric':<24} {'Original':>14} {'Pruned':>14}")
    print("-" * 57)
    print(f"{'Accuracy (%)':<24} {orig['acc']:>13.2f}% {comp['acc']:>13.2f}%")
    print(f"{'Model Size (KB)':<24} {orig['size']:>14.1f} {comp['size']:>14.1f}")
    print(f"{'Inference (ms)':<24} {orig['inf']:>14.2f} {comp['inf']:>14.2f}")
    print(f"{'Sparsity':<24} {'0.0%':>14} {comp['sparsity']:>13.1f}%")
    print("-" * 57)
    print(f"{'Accuracy Drop':<24} {orig['acc'] - comp['acc']:>13.2f}%")
    print(f"{'Speedup':<24} {orig['inf'] / comp['inf']:>13.2f}x")
    print("=" * 57 + "\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    X_train, y_train, classes = load_dataset(
        args.data_dir, "Training", max_per_class=args.max_per_class
    )
    X_test, y_test, _ = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)

    # ── Load trained model ────────────────────────────────────────────────────
    print(f"\n[Model] Loading from '{args.model_path}'...")
    model = SimpleCNN(num_classes=num_classes)
    model = load_model(model, args.model_path)

    # ── Baseline stats ────────────────────────────────────────────────────────
    print("\n[Baseline] Evaluating original model...")
    orig_acc  = compute_accuracy(model, X_test, y_test, args.batch_size)
    orig_size = get_model_size_kb(model)
    orig_inf  = measure_inference_ms(model)
    print(f"  Accuracy : {orig_acc:.2f}%")
    print(f"  Size     : {orig_size:.1f} KB")
    print(f"  Inference: {orig_inf:.2f} ms")

    original_stats = {"acc": orig_acc, "size": orig_size,
                      "inf": orig_inf, "sparsity": 0.0}

    # ── Apply pruning ─────────────────────────────────────────────────────────
    print(f"\n[Pruning] Strategy: '{args.strategy}' | Amount: {args.amount*100:.0f}%")
    if args.strategy == "global":
        model = global_pruning(model, args.amount)
    elif args.strategy == "per_layer":
        model = per_layer_pruning(model, args.amount)
    else:
        raise ValueError(f"Unknown strategy '{args.strategy}'. Use 'global' or 'per_layer'.")

    # Accuracy immediately after pruning (before fine-tuning)
    acc_after_prune = compute_accuracy(model, X_test, y_test, args.batch_size)
    print(f"\n  Accuracy after pruning (before fine-tune): {acc_after_prune:.2f}%")

    # ── Fine-tune ─────────────────────────────────────────────────────────────
    model = fine_tune(model, X_train, y_train, X_test, y_test,
                      epochs=args.finetune_epochs,
                      lr=args.finetune_lr,
                      batch_size=args.batch_size)

    # ── Pruned model stats ────────────────────────────────────────────────────
    print("\n[Evaluation] Final pruned model stats...")
    pruned_acc      = compute_accuracy(model, X_test, y_test, args.batch_size)
    pruned_size     = get_model_size_kb(model)
    pruned_inf      = measure_inference_ms(model)
    pruned_sparsity = count_sparsity(model)

    pruned_stats = {"acc": pruned_acc, "size": pruned_size,
                    "inf": pruned_inf, "sparsity": pruned_sparsity}

    save_model(model, args.save_path)

    # ── Comparison ────────────────────────────────────────────────────────────
    print_comparison(original_stats, pruned_stats, args.strategy, args.amount)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Prune SimpleCNN (from scratch)")
    parser.add_argument("--data_dir",        type=str,   default="./fruits-360")
    parser.add_argument("--model_path",      type=str,   default="cnn_fruits")
    parser.add_argument("--save_path",       type=str,   default="cnn_pruned")
    parser.add_argument("--strategy",        type=str,   default="global",
                        choices=["global", "per_layer"])
    parser.add_argument("--amount",          type=float, default=0.5,
                        help="Fraction of weights to prune (default: 0.5 = 50%%)")
    parser.add_argument("--finetune_epochs", type=int,   default=3)
    parser.add_argument("--finetune_lr",     type=float, default=0.001)
    parser.add_argument("--batch_size",      type=int,   default=32)
    parser.add_argument("--max_per_class",   type=int,   default=None)
    args = parser.parse_args()
    main(args)
