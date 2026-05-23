"""
person2_train.py — Training loop, loss function, and optimizer built from scratch.
PERSON 2's responsibility.

Implements:
    - Cross-entropy loss (manually, from scratch)
    - SGD with momentum optimizer (manually, from scratch)
    - Full training loop with accuracy tracking
    - Model saving/loading via numpy .npz files

Usage:
    python person2_train.py --data_dir ./fruits-360 --epochs 15
"""

import os
import argparse
import numpy as np
from PIL import Image

from logger import Logger
from person1_model import SimpleCNN


# ─── DATA LOADING ─────────────────────────────────────────────────────────────

def load_dataset(data_dir, split="Training", img_size=100, max_per_class=None):
    """
    Loads Fruits-360 images from disk into NumPy arrays.
    Returns X (N, 3, H, W) float32 normalised [0,1] and y (N,) int labels.
    """
    split_dir = os.path.join(data_dir, split)
    classes   = sorted(os.listdir(split_dir))
    class_map = {cls: i for i, cls in enumerate(classes)}

    X, y = [], []
    print(f"[Data] Loading '{split}' split ({len(classes)} classes)...")

    for cls in classes:
        cls_dir = os.path.join(split_dir, cls)
        if not os.path.isdir(cls_dir):
            continue
        files = os.listdir(cls_dir)
        if max_per_class:
            files = files[:max_per_class]

        for fname in files:
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue
            img_path = os.path.join(cls_dir, fname)
            try:
                img = Image.open(img_path).convert("RGB")
                img = img.resize((img_size, img_size))
                arr = np.array(img, dtype=np.float32) / 255.0
                arr = arr.transpose(2, 0, 1)
                X.append(arr)
                y.append(class_map[cls])
            except Exception:
                pass

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  Loaded {len(X)} images. Shape: {X.shape}")
    return X, y, classes


def get_batches(X, y, batch_size, shuffle=True):
    """Yields (X_batch, y_batch) tuples."""
    N = len(X)
    idx = np.random.permutation(N) if shuffle else np.arange(N)
    for start in range(0, N, batch_size):
        batch_idx = idx[start:start + batch_size]
        yield X[batch_idx], y[batch_idx]


# ─── LOSS FUNCTION ────────────────────────────────────────────────────────────

def cross_entropy_loss(probs, labels):
    """
    Cross-entropy loss for multi-class classification.

    Formula: L = -1/N * sum( log(p[correct_class]) )

    probs:  (batch, num_classes) — softmax probabilities from model
    labels: (batch,)             — integer class indices
    Returns: scalar loss, gradient w.r.t. probs (batch, num_classes)
    """
    batch_size = len(labels)
    eps = 1e-9

    correct_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_probs + eps))

    d_probs = probs.copy()
    d_probs[np.arange(batch_size), labels] -= 1
    d_probs /= batch_size

    return loss, d_probs


# ─── OPTIMIZER ────────────────────────────────────────────────────────────────

class SGDMomentum:
    """
    Stochastic Gradient Descent with Momentum.

    v = momentum * v - lr * gradient
    weight += v
    """

    def __init__(self, model, lr=0.01, momentum=0.9):
        self.lr       = lr
        self.momentum = momentum
        self.velocity = {}

        for i, layer in enumerate(model.get_trainable_layers()):
            if hasattr(layer, 'filters'):
                self.velocity[f"layer{i}_filters"] = np.zeros_like(layer.filters)
                self.velocity[f"layer{i}_biases"]  = np.zeros_like(layer.biases)
            else:
                self.velocity[f"layer{i}_weights"] = np.zeros_like(layer.weights)
                self.velocity[f"layer{i}_biases"]  = np.zeros_like(layer.biases)

    def step(self, model):
        """Apply one gradient update step to all trainable layers."""
        for i, layer in enumerate(model.get_trainable_layers()):
            if hasattr(layer, 'filters'):
                v_f = self.velocity[f"layer{i}_filters"]
                v_b = self.velocity[f"layer{i}_biases"]

                v_f = self.momentum * v_f - self.lr * layer.d_filters
                v_b = self.momentum * v_b - self.lr * layer.d_biases

                layer.filters += v_f
                layer.biases  += v_b

                self.velocity[f"layer{i}_filters"] = v_f
                self.velocity[f"layer{i}_biases"]  = v_b
            else:
                v_w = self.velocity[f"layer{i}_weights"]
                v_b = self.velocity[f"layer{i}_biases"]

                v_w = self.momentum * v_w - self.lr * layer.d_weights
                v_b = self.momentum * v_b - self.lr * layer.d_biases

                layer.weights += v_w
                layer.biases  += v_b

                self.velocity[f"layer{i}_weights"] = v_w
                self.velocity[f"layer{i}_biases"]  = v_b


# ─── MODEL SAVE / LOAD ────────────────────────────────────────────────────────

def save_model(model, path):
    """Save all model weights to a .npz file."""
    params = {}
    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, 'filters'):
            params[f"layer{i}_filters"] = layer.filters
            params[f"layer{i}_biases"]  = layer.biases
        else:
            params[f"layer{i}_weights"] = layer.weights
            params[f"layer{i}_biases"]  = layer.biases
    np.savez(path, **params)
    print(f"  ✓ Model saved to '{path}.npz'")


def load_model(model, path):
    """Load weights from a .npz file into the model."""
    if not path.endswith(".npz"):
        path += ".npz"
    data = np.load(path)
    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, 'filters'):
            layer.filters = data[f"layer{i}_filters"]
            layer.biases  = data[f"layer{i}_biases"]
        else:
            layer.weights = data[f"layer{i}_weights"]
            layer.biases  = data[f"layer{i}_biases"]
    print(f"  ✓ Model loaded from '{path}'")
    return model


# ─── ACCURACY ─────────────────────────────────────────────────────────────────

def compute_accuracy(model, X, y, batch_size=32):
    """Evaluate accuracy over a dataset without computing gradients."""
    correct = 0
    total   = len(y)

    for X_batch, y_batch in get_batches(X, y, batch_size, shuffle=False):
        probs    = model.forward(X_batch)
        preds    = np.argmax(probs, axis=1)
        correct += np.sum(preds == y_batch)

    return 100.0 * correct / total


# ─── TRAINING LOOP ────────────────────────────────────────────────────────────

def train(args):
    log = Logger("person2")

    X_train, y_train, classes = load_dataset(
        args.data_dir, "Training", max_per_class=args.max_per_class
    )
    X_test, y_test, _ = load_dataset(
        args.data_dir, "Test", max_per_class=args.max_per_class
    )
    num_classes = len(classes)

    model     = SimpleCNN(num_classes=num_classes)
    optimizer = SGDMomentum(model, lr=args.lr, momentum=0.9)

    log(f"[Model] SimpleCNN — {model.get_param_count():,} parameters")
    log(f"[Train] {len(X_train)} samples | {args.epochs} epochs | "
        f"lr={args.lr} | batch={args.batch_size}")
    log("")

    best_acc = 0.0

    for epoch in range(1, args.epochs + 1):
        total_loss  = 0.0
        num_batches = 0

        for X_batch, y_batch in get_batches(X_train, y_train, args.batch_size):
            probs = model.forward(X_batch)
            loss, d_probs = cross_entropy_loss(probs, y_batch)
            total_loss   += loss
            num_batches  += 1
            model.backward(d_probs)
            optimizer.step(model)

        avg_loss = total_loss / num_batches
        test_acc = compute_accuracy(model, X_test, y_test, args.batch_size)

        log(f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Loss: {avg_loss:.4f} | Test Accuracy: {test_acc:.2f}%")

        if test_acc > best_acc:
            best_acc = test_acc
            save_model(model, args.save_path)
            log(f"  ↑ New best! ({best_acc:.2f}%)")

    log.section("RESULTS")
    log(f"Best accuracy : {best_acc:.2f}%")
    log(f"Model saved to: {args.save_path}.npz")
    log("Run person3_pruning.py or person4_quantization.py next.")
    log.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train SimpleCNN on Fruits-360")
    parser.add_argument("--data_dir",      type=str,   default="./fruits-360")
    parser.add_argument("--epochs",        type=int,   default=15)
    parser.add_argument("--batch_size",    type=int,   default=32)
    parser.add_argument("--lr",            type=float, default=0.01)
    parser.add_argument("--save_path",     type=str,   default="cnn_fruits")
    parser.add_argument("--max_per_class", type=int,   default=None)
    args = parser.parse_args()
    train(args)
