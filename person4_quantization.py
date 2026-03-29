"""
person4_quantization.py — Post-training quantization implemented from scratch.
PERSON 4's responsibility.

Implements the exact algorithm from the survey paper (Algorithm 1, Section 3):
    Step 1: Find min/max values in weights
    Step 2: Choose quantization type (symmetric INT8)
    Step 3: Calculate scale S and zero-point Z
    Step 4: Quantize FP32 weights → INT8
    Step 5: Dequantize back → FP32 for inference, measure accuracy loss

The math directly mirrors Equations (1)–(5) from the survey.

Usage:
    python person4_quantization.py --data_dir ./fruits-360 --model_path cnn_fruits
"""

import os
import argparse
import numpy as np

from person1_model import SimpleCNN, ConvLayer, FCLayer
from person2_train import (load_dataset, compute_accuracy,
                            save_model, load_model)


# ─── QUANTIZATION MATH (directly from the survey, Equations 1–5) ─────────────

def compute_scale_and_zero(r_min, r_max, q_min=-128, q_max=127):
    """
    Compute quantization parameters S (scale) and Z (zero-point).

    From the survey paper:
        S = (R_max - R_min) / (Q_max - Q_min)    [Equation 1]
        Z = Q_max - R_max / S                     [Equation 2]

    R = real floating-point range
    Q = quantized integer range
    S = scale factor (FP32)
    Z = zero-point (INT8) — the quantized value corresponding to 0.0
    """
    S = (r_max - r_min) / (q_max - q_min)

    # Avoid division by zero for constant layers
    if S == 0:
        S = 1e-8

    Z = q_max - r_max / S
    Z = int(np.round(np.clip(Z, q_min, q_max)))  # must be integer

    return float(S), Z


def quantize(weights, S, Z, q_min=-128, q_max=127):
    """
    Quantize FP32 weights → INT8.

    From the survey paper:
        Q = R / S + Z    [Equation 3]

    Clips to valid INT8 range and rounds to nearest integer.
    """
    Q = weights / S + Z
    Q = np.round(Q)
    Q = np.clip(Q, q_min, q_max)
    return Q.astype(np.int8)


def dequantize(Q, S, Z):
    """
    Dequantize INT8 back to FP32 for inference.

    From the survey paper:
        R = (Q - Z) * S    [Equation 4]

    This is used because hardware may not support INT8 matmul in all cases.
    The round-trip FP32 → INT8 → FP32 shows the accuracy impact of quantization.
    """
    return (Q.astype(np.float32) - Z) * S


# ─── QUANTIZE A FULL LAYER ────────────────────────────────────────────────────

def quantize_layer(layer):
    """
    Quantize a single Conv or FC layer's weights.
    Returns quantization metadata for analysis.
    """
    if hasattr(layer, 'filters'):
        w = layer.filters
    else:
        w = layer.weights

    r_min = float(w.min())
    r_max = float(w.max())

    S, Z = compute_scale_and_zero(r_min, r_max)

    # Quantize to INT8
    w_int8 = quantize(w, S, Z)

    # Dequantize back to FP32 for inference
    w_dequant = dequantize(w_int8, S, Z)

    # Apply dequantized weights back to layer
    if hasattr(layer, 'filters'):
        layer.filters = w_dequant
    else:
        layer.weights = w_dequant

    # Calculate quantization error
    error = np.mean(np.abs(w - w_dequant))

    return {
        "r_min":    r_min,
        "r_max":    r_max,
        "S":        S,
        "Z":        Z,
        "error":    error,
        "shape":    w.shape,
        "int8_min": int(w_int8.min()),
        "int8_max": int(w_int8.max()),
    }


def quantize_model(model):
    """
    Apply post-training quantization to all trainable layers.
    Returns the quantized model and per-layer metadata.
    """
    print("\n  [Quantization] Quantizing layers FP32 → INT8 → FP32 (dequant)...")
    print(f"  {'Layer':<8} {'Shape':<22} {'S':>10} {'Z':>6} "
          f"{'Avg Error':>12} {'INT8 Range':>14}")
    print("  " + "-" * 76)

    metadata = {}
    for i, layer in enumerate(model.get_trainable_layers()):
        layer_type = "Conv" if hasattr(layer, 'filters') else "FC"
        meta = quantize_layer(layer)
        metadata[i] = meta

        shape_str    = str(meta['shape'])
        int8_range   = f"[{meta['int8_min']}, {meta['int8_max']}]"
        print(f"  {i:<3} {layer_type:<5} {shape_str:<22} "
              f"{meta['S']:>10.6f} {meta['Z']:>6} "
              f"{meta['error']:>12.6f} {int8_range:>14}")

    print()
    return model, metadata


# ─── SIZE COMPARISON ─────────────────────────────────────────────────────────

def estimate_int8_size_kb(model):
    """
    Estimate the size if weights were stored as INT8 (1 byte per weight)
    vs FP32 (4 bytes per weight). Shows theoretical compression ratio.
    """
    total_weights = 0
    for layer in model.get_trainable_layers():
        w = layer.filters if hasattr(layer, 'filters') else layer.weights
        total_weights += w.size

    fp32_kb = total_weights * 4 / 1024   # 4 bytes per float32
    int8_kb = total_weights * 1 / 1024   # 1 byte per int8
    return fp32_kb, int8_kb


def get_model_size_kb(model, path="tmp_quant"):
    save_model(model, path)
    size = os.path.getsize(path + ".npz") / 1024
    os.remove(path + ".npz")
    return size


def measure_inference_ms(model, runs=100, img_size=100):
    import time
    dummy = np.random.randn(1, 3, img_size, img_size).astype(np.float32)
    for _ in range(5):
        model.forward(dummy)
    start = time.time()
    for _ in range(runs):
        model.forward(dummy)
    return (time.time() - start) / runs * 1000


# ─── COMPARISON TABLE ─────────────────────────────────────────────────────────

def print_comparison(orig, quant, fp32_kb, int8_kb):
    print("\n" + "=" * 60)
    print("  COMPARISON: Original (FP32) vs Post-Training Quantization (INT8)")
    print("=" * 60)
    print(f"{'Metric':<30} {'Original':>14} {'Quantized':>12}")
    print("-" * 60)
    print(f"{'Accuracy (%)':<30} {orig['acc']:>13.2f}% {quant['acc']:>11.2f}%")
    print(f"{'Saved Size (KB)':<30} {orig['size']:>14.1f} {quant['size']:>12.1f}")
    print(f"{'Theoretical FP32 Size (KB)':<30} {fp32_kb:>14.1f} {'':>12}")
    print(f"{'Theoretical INT8 Size (KB)':<30} {'':>14} {int8_kb:>12.1f}")
    print(f"{'Theoretical Compression':<30} {'':>14} {'4x':>12}")
    print(f"{'Inference (ms)':<30} {orig['inf']:>14.2f} {quant['inf']:>12.2f}")
    print("-" * 60)
    print(f"{'Accuracy Drop':<30} {orig['acc'] - quant['acc']:>13.2f}%")
    print(f"{'Speedup':<30} {orig['inf'] / quant['inf']:>13.2f}x")
    print("=" * 60)
    print()
    print("  NOTE: 'Saved Size' uses .npz (FP32 storage for both).")
    print("  'Theoretical INT8 Size' shows what a real INT8 deployment")
    print("  would save — a 4x reduction over FP32 storage.\n")


# ─── MAIN ────────────────────────────────────────────────────────────────────

def main(args):
    # ── Load data ─────────────────────────────────────────────────────────────
    X_test, y_test, classes = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)

    # ── Load original model ───────────────────────────────────────────────────
    print(f"\n[Model] Loading from '{args.model_path}'...")
    model = SimpleCNN(num_classes=num_classes)
    model = load_model(model, args.model_path)

    # ── Baseline stats ────────────────────────────────────────────────────────
    print("\n[Baseline] Evaluating original FP32 model...")
    orig_acc  = compute_accuracy(model, X_test, y_test, args.batch_size)
    orig_size = get_model_size_kb(model)
    orig_inf  = measure_inference_ms(model)
    print(f"  Accuracy : {orig_acc:.2f}%")
    print(f"  Size     : {orig_size:.1f} KB")
    print(f"  Inference: {orig_inf:.2f} ms")

    original_stats = {"acc": orig_acc, "size": orig_size, "inf": orig_inf}

    # ── Apply quantization ────────────────────────────────────────────────────
    print("\n[Quantization] Applying Algorithm 1 from the survey paper...")
    print("  Steps: min/max → S and Z → Q = R/S + Z → R = (Q-Z)*S")

    model, metadata = quantize_model(model)

    # ── Quantized model stats ─────────────────────────────────────────────────
    print("[Evaluation] Evaluating quantized model (dequantized weights)...")
    quant_acc  = compute_accuracy(model, X_test, y_test, args.batch_size)
    quant_size = get_model_size_kb(model)
    quant_inf  = measure_inference_ms(model)
    fp32_kb, int8_kb = estimate_int8_size_kb(model)

    print(f"  Accuracy : {quant_acc:.2f}%")
    print(f"  Size     : {quant_size:.1f} KB")
    print(f"  Inference: {quant_inf:.2f} ms")

    quantized_stats = {"acc": quant_acc, "size": quant_size, "inf": quant_inf}

    save_model(model, args.save_path)
    print(f"\n  ✓ Quantized model saved to '{args.save_path}.npz'")

    # ── Final comparison ──────────────────────────────────────────────────────
    print_comparison(original_stats, quantized_stats, fp32_kb, int8_kb)

    # ── Per-layer quantization summary ───────────────────────────────────────
    print("  Per-layer quantization error summary:")
    for i, meta in metadata.items():
        print(f"    Layer {i}: avg weight error = {meta['error']:.6f}, "
              f"S={meta['S']:.6f}, Z={meta['Z']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Post-training quantization of SimpleCNN (from scratch)"
    )
    parser.add_argument("--data_dir",      type=str,  default="./fruits-360")
    parser.add_argument("--model_path",    type=str,  default="cnn_fruits")
    parser.add_argument("--save_path",     type=str,  default="cnn_quantized")
    parser.add_argument("--batch_size",    type=int,  default=32)
    parser.add_argument("--max_per_class", type=int,  default=None)
    args = parser.parse_args()
    main(args)
