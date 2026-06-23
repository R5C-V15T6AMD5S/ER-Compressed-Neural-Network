"""
person2_train.py — Training loop, data loader, loss function, optimizer,
and model save/load for the Fruits-360 CNN project.

Main responsibilities:
    - discover Fruits-360 classes from Training/ and Test/ folders
    - load RGB 100x100 images and normalize them to [0, 1]
    - train Person 1's SimpleCNN using cross-entropy loss
    - update parameters using SGD with momentum
    - track running training loss and running training accuracy during each epoch
    - evaluate test accuracy after each epoch
    - save the best trained model to .npz
    - optionally run Quantization-Aware Training (QAT) style fake INT8
      weight quantization during training
    - expose helper functions used later by Person 3 and Person 4

Usage:
    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 15

Baseline quick technical test:
    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 1 --max_per_class 5 --batch_size 4

QAT quick technical test:
    python person2_train.py --data_dir ./fruits-360-100x100 --epochs 1 --max_per_class 5 --batch_size 4 --qat

Spyder example:
    runfile('person2_train.py', args='--data_dir ./fruits-360-100x100 --epochs 1 --max_per_class 5 --batch_size 4 --qat')
"""

import os
import argparse
import numpy as np
from PIL import Image

from logger import Logger
from person1_model import SimpleCNN


IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")


# -----------------------------------------------------------------------------
# DATA LOADING
# -----------------------------------------------------------------------------

def discover_classes(data_dir, split="Training"):
    """
    Read class names from a Fruits-360 split folder.

    Fruits-360 is organized as:
        fruits-360-100x100/
            Training/
                Apple 5/
                Banana/
                ...
            Test/
                Apple 5/
                Banana/
                ...

    The folder names are the class names. Sorting them gives a stable mapping:
        class name -> integer label.
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Could not find '{split}' folder at: {split_dir}\n"
            f"Expected structure: {data_dir}/{split}/<class_name>/<image>.jpg"
        )

    classes = sorted(
        name for name in os.listdir(split_dir)
        if os.path.isdir(os.path.join(split_dir, name))
    )

    if not classes:
        raise ValueError(
            f"No class folders found in: {split_dir}\n"
            "Make sure the dataset folder is not empty."
        )

    return classes


def load_image_paths(data_dir, split="Training", max_per_class=None, classes=None):
    """
    Collect image paths and labels without loading all pixel data into RAM.

    This is the preferred loader for training because the full Fruits-360 dataset
    is large. We store only file paths in memory and load images batch-by-batch.

    Parameters
    ----------
    data_dir : str
        Dataset root folder, usually './fruits-360-100x100'.
    split : str
        'Training' or 'Test'.
    max_per_class : int or None
        Optional limit for quick tests. None means use all images.
    classes : list[str] or None
        If provided, this exact class order is reused. This is important so that
        Training and Test labels match.

    Returns
    -------
    paths : np.ndarray of object/string paths
    labels : np.ndarray of int labels
    classes : list[str]
    """
    split_dir = os.path.join(data_dir, split)
    if not os.path.isdir(split_dir):
        raise FileNotFoundError(
            f"Could not find '{split}' folder at: {split_dir}"
        )

    if classes is None:
        classes = discover_classes(data_dir, split)

    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    paths = []
    labels = []

    print(f"[Data] Scanning '{split}' split ({len(classes)} classes)...")

    for class_name in classes:
        class_dir = os.path.join(split_dir, class_name)
        if not os.path.isdir(class_dir):
            # Robustness: if a class is missing from Test, skip it instead of
            # destroying the mapping. Standard Fruits-360 normally has matches.
            continue

        image_files = sorted(
            fname for fname in os.listdir(class_dir)
            if fname.lower().endswith(IMAGE_EXTENSIONS)
        )

        if max_per_class is not None:
            image_files = image_files[:max_per_class]

        label = class_to_idx[class_name]
        for fname in image_files:
            paths.append(os.path.join(class_dir, fname))
            labels.append(label)

    if not paths:
        raise ValueError(
            f"No image files found in '{split_dir}'. "
            f"Supported extensions: {IMAGE_EXTENSIONS}"
        )

    paths = np.array(paths, dtype=object)
    labels = np.array(labels, dtype=np.int32)

    print(f"  Found {len(paths)} images.")
    return paths, labels, classes


def load_image(path, img_size=100):
    """
    Load one image as a normalized NumPy tensor.

    Output shape is (3, img_size, img_size), because Person 1's CNN expects
    channel-first input: (N, C, H, W).
    """
    with Image.open(path) as img:
        img = img.convert("RGB")
        if img.size != (img_size, img_size):
            img = img.resize((img_size, img_size))
        arr = np.asarray(img, dtype=np.float32) / 255.0

    return arr.transpose(2, 0, 1)


def iter_path_batches(paths, labels, batch_size, img_size=100, shuffle=True):
    """
    Yield mini-batches by loading images from disk only when needed.

    This behaves like a simple DataLoader, but it is implemented manually with
    NumPy and Pillow. It avoids keeping the full dataset as one huge array in RAM.
    """
    n = len(paths)
    indices = np.random.permutation(n) if shuffle else np.arange(n)

    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        x_batch = []
        y_batch = []

        for idx in batch_indices:
            try:
                x_batch.append(load_image(paths[idx], img_size=img_size))
                y_batch.append(labels[idx])
            except Exception as exc:
                print(f"  [Warning] Skipping unreadable image: {paths[idx]} ({exc})")

        if not x_batch:
            continue

        yield np.stack(x_batch).astype(np.float32), np.array(y_batch, dtype=np.int32)


# Backward-compatible eager loader used by Person 3 / Person 4 in the current repo.
# It is useful for smaller subsets. For full Fruits-360 training, train() below uses
# the streaming path-based loader instead.
def load_dataset(data_dir, split="Training", img_size=100, max_per_class=None):
    """
    Load a Fruits-360 split fully into memory.

    Returns
    -------
    X : np.ndarray, shape (N, 3, H, W), float32 in [0, 1]
    y : np.ndarray, shape (N,), int32 labels
    classes : list[str]

    Note
    ----
    This function keeps compatibility with Person 3 and Person 4 scripts, which
    currently expect arrays. For very large runs, use max_per_class or switch
    those scripts to the streaming loader too.
    """
    paths, labels, classes = load_image_paths(
        data_dir=data_dir,
        split=split,
        max_per_class=max_per_class,
        classes=None,
    )

    X = []
    y = []
    print(f"[Data] Loading '{split}' images into memory...")
    for path, label in zip(paths, labels):
        try:
            X.append(load_image(path, img_size=img_size))
            y.append(label)
        except Exception as exc:
            print(f"  [Warning] Skipping unreadable image: {path} ({exc})")

    if not X:
        raise ValueError(f"No valid images could be loaded from split '{split}'.")

    X = np.stack(X).astype(np.float32)
    y = np.array(y, dtype=np.int32)
    print(f"  Loaded {len(X)} images. Shape: {X.shape}")
    return X, y, classes


def get_batches(X, y, batch_size, shuffle=True):
    """
    Yield mini-batches from arrays already loaded into memory.

    Kept intentionally simple because Person 3 uses it for pruning fine-tuning.
    """
    n = len(X)
    indices = np.random.permutation(n) if shuffle else np.arange(n)
    for start in range(0, n, batch_size):
        batch_indices = indices[start:start + batch_size]
        yield X[batch_indices], y[batch_indices]


# -----------------------------------------------------------------------------
# LOSS FUNCTION
# -----------------------------------------------------------------------------

def cross_entropy_loss(probs, labels):
    """
    Multi-class cross-entropy loss.

    Person 1's model already returns softmax probabilities, so the loss is:
        L = -mean(log(probability assigned to the correct class))

    The returned gradient is the standard softmax + cross-entropy gradient:
        d_logits = (probs - one_hot(labels)) / batch_size

    In this project it is passed to model.backward().
    """
    batch_size = len(labels)
    eps = 1e-9

    correct_class_probs = probs[np.arange(batch_size), labels]
    loss = -np.mean(np.log(correct_class_probs + eps))

    d_probs = probs.copy()
    d_probs[np.arange(batch_size), labels] -= 1.0
    d_probs /= batch_size

    return float(loss), d_probs


# -----------------------------------------------------------------------------
# OPTIMIZER
# -----------------------------------------------------------------------------

class SGDMomentum:
    """
    Stochastic Gradient Descent with momentum.

    Plain SGD update:
        weight = weight - lr * gradient

    Momentum version:
        velocity = momentum * velocity - lr * gradient
        weight   = weight + velocity

    Momentum helps because the update direction is smoothed over multiple
    batches instead of reacting only to the current noisy mini-batch.
    """

    def __init__(self, model, lr=0.01, momentum=0.9):
        self.lr = lr
        self.momentum = momentum
        self.velocity = {}

        for i, layer in enumerate(model.get_trainable_layers()):
            if hasattr(layer, "filters"):
                self.velocity[f"layer{i}_filters"] = np.zeros_like(layer.filters)
                self.velocity[f"layer{i}_biases"] = np.zeros_like(layer.biases)
            else:
                self.velocity[f"layer{i}_weights"] = np.zeros_like(layer.weights)
                self.velocity[f"layer{i}_biases"] = np.zeros_like(layer.biases)

    def step(self, model):
        """Apply one optimizer update to all trainable layers."""
        for i, layer in enumerate(model.get_trainable_layers()):
            if hasattr(layer, "filters"):
                v_filters = self.velocity[f"layer{i}_filters"]
                v_biases = self.velocity[f"layer{i}_biases"]

                v_filters = self.momentum * v_filters - self.lr * layer.d_filters
                v_biases = self.momentum * v_biases - self.lr * layer.d_biases

                layer.filters += v_filters
                layer.biases += v_biases

                self.velocity[f"layer{i}_filters"] = v_filters
                self.velocity[f"layer{i}_biases"] = v_biases
            else:
                v_weights = self.velocity[f"layer{i}_weights"]
                v_biases = self.velocity[f"layer{i}_biases"]

                v_weights = self.momentum * v_weights - self.lr * layer.d_weights
                v_biases = self.momentum * v_biases - self.lr * layer.d_biases

                layer.weights += v_weights
                layer.biases += v_biases

                self.velocity[f"layer{i}_weights"] = v_weights
                self.velocity[f"layer{i}_biases"] = v_biases




# -----------------------------------------------------------------------------
# QUANTIZATION-AWARE TRAINING (QAT) HELPERS
# -----------------------------------------------------------------------------

def compute_qparams(r_min, r_max, strategy="asymmetric", num_bits=8):
    """
    Compute scale S and zero-point Z for fake quantization.

    This mirrors the quantization idea used later by Person 4, but it is used
    during training instead of only after training. The weights are rounded to
    an INT8 grid and immediately dequantized back to float32 so the rest of the
    pure NumPy pipeline can keep using normal floating-point operations.

    Parameters
    ----------
    r_min, r_max : float
        Minimum and maximum real FP32 values in the weight tensor.
    strategy : {'symmetric', 'asymmetric'}
        symmetric keeps Z=0 and is common for weights;
        asymmetric follows the min/max formula with a zero-point.
    num_bits : int
        Number of quantization bits. The project uses 8 to simulate INT8.
    """
    if num_bits < 2:
        raise ValueError("num_bits must be at least 2")

    q_min = -(2 ** (num_bits - 1))
    q_max = (2 ** (num_bits - 1)) - 1

    if strategy == "symmetric":
        max_abs = max(abs(float(r_min)), abs(float(r_max)))
        scale = max_abs / q_max if max_abs != 0 else 1e-8
        zero_point = 0
    elif strategy == "asymmetric":
        scale = (float(r_max) - float(r_min)) / (q_max - q_min)
        if scale == 0:
            scale = 1e-8
        zero_point = int(np.round(q_max - float(r_max) / scale))
        zero_point = int(np.clip(zero_point, q_min, q_max))
    else:
        raise ValueError(f"Unknown QAT strategy: {strategy}")

    return float(scale), int(zero_point), int(q_min), int(q_max)


def fake_quantize_array(values, strategy="asymmetric", num_bits=8):
    """
    Simulate INT8 quantization while returning float32 values.

    Steps:
        1) find min/max of a FP32 tensor
        2) compute scale and zero-point
        3) round to the INT8 grid
        4) dequantize back to FP32

    The output still has dtype float32, but the values can only lie on the same
    discrete grid that INT8 quantization would allow. This is why the operation
    is called fake quantization: it simulates quantization noise during training
    without requiring a separate INT8 inference engine.
    """
    r_min = float(np.min(values))
    r_max = float(np.max(values))
    scale, zero_point, q_min, q_max = compute_qparams(
        r_min, r_max, strategy=strategy, num_bits=num_bits
    )

    q = np.round(values / scale + zero_point)
    q = np.clip(q, q_min, q_max)
    dequant = (q.astype(np.float32) - zero_point) * scale

    avg_error = float(np.mean(np.abs(values - dequant)))
    return dequant.astype(np.float32), {
        "scale": scale,
        "zero_point": zero_point,
        "q_min": q_min,
        "q_max": q_max,
        "avg_error": avg_error,
        "r_min": r_min,
        "r_max": r_max,
    }


def fake_quantize_model_weights(model, strategy="asymmetric", num_bits=8):
    """
    Apply fake quantization to every trainable weight tensor in-place.

    Biases are intentionally not quantized here. Person 4's current PTQ module
    quantizes weights, so Person 2's QAT mode prepares the same part of the
    model for later PTQ comparison while keeping the interface unchanged.

    Returns a small summary that can be logged for the final report.
    """
    layer_errors = []

    for layer in model.get_trainable_layers():
        if hasattr(layer, "filters"):
            q_weights, meta = fake_quantize_array(
                layer.filters, strategy=strategy, num_bits=num_bits
            )
            layer.filters = q_weights
        else:
            q_weights, meta = fake_quantize_array(
                layer.weights, strategy=strategy, num_bits=num_bits
            )
            layer.weights = q_weights

        layer_errors.append(meta["avg_error"])

    if not layer_errors:
        return {"avg_error": 0.0, "max_error": 0.0, "layers": 0}

    return {
        "avg_error": float(np.mean(layer_errors)),
        "max_error": float(np.max(layer_errors)),
        "layers": len(layer_errors),
    }


def should_apply_qat(epoch, batch_index, args):
    """
    Decide whether fake quantization should be applied on this optimizer step.

    epoch is 1-based, batch_index is 1-based. Warmup lets the model learn a
    rough FP32 solution first, and frequency can reduce overhead on slow CPUs.
    """
    if not getattr(args, "qat", False):
        return False
    if epoch <= getattr(args, "qat_warmup_epochs", 0):
        return False
    frequency = max(1, int(getattr(args, "qat_frequency", 1)))
    return batch_index % frequency == 0


# -----------------------------------------------------------------------------
# MODEL SAVE / LOAD
# -----------------------------------------------------------------------------

def _npz_path(path):
    """Ensure that the saved model path ends with .npz."""
    return path if path.endswith(".npz") else path + ".npz"


def save_model(model, path, classes=None):
    """
    Save all trainable parameters to a NumPy .npz file.

    The naming convention layer0_filters, layer0_biases, ... is intentionally
    stable because Person 3 and Person 4 load the model by the same keys.
    """
    file_path = _npz_path(path)
    parent = os.path.dirname(file_path)
    if parent:
        os.makedirs(parent, exist_ok=True)

    params = {}
    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, "filters"):
            params[f"layer{i}_filters"] = layer.filters
            params[f"layer{i}_biases"] = layer.biases
        else:
            params[f"layer{i}_weights"] = layer.weights
            params[f"layer{i}_biases"] = layer.biases

    # Extra metadata is harmless for the current loaders and useful for review.
    params["num_classes"] = np.array([model.num_classes], dtype=np.int32)
    if classes is not None:
        params["classes"] = np.array(classes, dtype=object)

    np.savez(file_path, **params)
    print(f"  Model saved to '{file_path}'")


def load_model(model, path):
    """Load model weights from a .npz file into an existing SimpleCNN instance."""
    file_path = _npz_path(path)
    data = np.load(file_path, allow_pickle=True)

    for i, layer in enumerate(model.get_trainable_layers()):
        if hasattr(layer, "filters"):
            layer.filters = data[f"layer{i}_filters"]
            layer.biases = data[f"layer{i}_biases"]
        else:
            layer.weights = data[f"layer{i}_weights"]
            layer.biases = data[f"layer{i}_biases"]

    print(f"  Model loaded from '{file_path}'")
    return model


# -----------------------------------------------------------------------------
# ACCURACY
# -----------------------------------------------------------------------------

def batch_accuracy(probs, labels):
    """
    Compute accuracy for one batch from probabilities already produced by forward().

    This function is the key part of the running training accuracy option: it does
    not perform an extra forward pass. It only reuses the predictions that were
    already needed for loss/backpropagation.
    """
    if len(labels) == 0:
        return 0, 0
    preds = np.argmax(probs, axis=1)
    correct = int(np.sum(preds == labels))
    total = int(len(labels))
    return correct, total


def compute_accuracy(model, X, y, batch_size=32):
    """Evaluate accuracy for arrays already loaded into memory."""
    if len(y) == 0:
        return 0.0

    correct = 0
    total = 0

    for X_batch, y_batch in get_batches(X, y, batch_size, shuffle=False):
        probs = model.forward(X_batch)
        batch_correct, batch_total = batch_accuracy(probs, y_batch)
        correct += batch_correct
        total += batch_total

    return 100.0 * correct / total if total > 0 else 0.0


def compute_accuracy_from_paths(model, paths, labels, batch_size=32,
                                img_size=100, max_batches=None):
    """
    Evaluate accuracy using the streaming path-based loader.

    max_batches is optional and useful only for quick technical checks. By
    default it is None, meaning the full evaluation split is used.
    """
    correct = 0
    total = 0
    batches_done = 0

    for X_batch, y_batch in iter_path_batches(
        paths, labels, batch_size=batch_size, img_size=img_size, shuffle=False
    ):
        probs = model.forward(X_batch)
        batch_correct, batch_total = batch_accuracy(probs, y_batch)
        correct += batch_correct
        total += batch_total
        batches_done += 1

        if max_batches is not None and batches_done >= max_batches:
            break

    return 100.0 * correct / total if total > 0 else 0.0


# -----------------------------------------------------------------------------
# TRAINING LOOP
# -----------------------------------------------------------------------------

def train(args):
    """Run my complete training pipeline."""
    np.random.seed(args.seed)
    log = Logger("person2")

    log.section("DATA")
    train_paths, y_train, classes = load_image_paths(
        args.data_dir,
        split="Training",
        max_per_class=args.max_per_class,
        classes=None,
    )
    test_paths, y_test, _ = load_image_paths(
        args.data_dir,
        split="Test",
        max_per_class=args.max_per_class,
        classes=classes,
    )

    num_classes = len(classes)
    log(f"Dataset path : {args.data_dir}")
    log(f"Classes      : {num_classes}")
    log(f"Train images : {len(train_paths)}")
    log(f"Test images  : {len(test_paths)}")
    if args.max_per_class is not None:
        log(f"Limit        : max_per_class={args.max_per_class}")
    log("")

    # Default save path depends on the training mode. Baseline remains
    # models/cnn_fruits.npz so Person 3 and Person 4 stay compatible. QAT uses
    # a separate file by default so it does not overwrite the FP32 baseline.
    if args.save_path is None:
        args.save_path = "models/cnn_fruits_qat" if args.qat else "models/cnn_fruits"

    log.section("MODEL")
    model = SimpleCNN(num_classes=num_classes)
    optimizer = SGDMomentum(model, lr=args.lr, momentum=args.momentum)
    log(f"Architecture : SimpleCNN from person1_model.py")
    log(f"Parameters   : {model.get_param_count():,}")
    log(f"Optimizer    : SGD with momentum={args.momentum}")
    log(f"Loss         : cross-entropy")
    log("")

    log.section("TRAINING")
    log(f"Epochs       : {args.epochs}")
    log(f"Batch size   : {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log("Train metric : running train accuracy from the same forward pass used for loss")
    if args.qat:
        log("QAT mode     : ENABLED - fake INT8 weight quantization during training")
        log(f"QAT strategy : {args.qat_strategy}, {args.qat_bits}-bit")
        log(f"QAT warmup   : {args.qat_warmup_epochs} epoch(s)")
        log(f"QAT frequency: every {max(1, args.qat_frequency)} optimizer step(s)")
        log(f"Save path    : {_npz_path(args.save_path)}")
    else:
        log("QAT mode     : disabled - normal FP32 baseline training")
        log(f"Save path    : {_npz_path(args.save_path)}")
    if args.eval_max_batches is not None:
        log(f"Evaluation   : first {args.eval_max_batches} test batch(es) only")
    else:
        log("Evaluation   : full Test split")
    log("")

    best_acc = -1.0
    best_epoch = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        samples_seen = 0
        train_correct = 0
        train_total = 0

        qat_error_sum = 0.0
        qat_error_steps = 0

        for X_batch, y_batch in iter_path_batches(
            train_paths,
            y_train,
            batch_size=args.batch_size,
            img_size=args.img_size,
            shuffle=True,
        ):
            # 1) Forward pass: probabilities are needed for both loss and accuracy.
            probs = model.forward(X_batch)

            # 2) Running training accuracy: no extra model.forward() is performed.
            batch_correct, batch_total = batch_accuracy(probs, y_batch)
            train_correct += batch_correct
            train_total += batch_total

            # 3) Loss + gradient for backpropagation.
            loss, d_probs = cross_entropy_loss(probs, y_batch)

            # 4) Backward pass and optimizer update.
            model.backward(d_probs)
            optimizer.step(model)

            num_batches += 1

            # 5) Optional QAT: after the FP32 optimizer update, snap weights to
            # the simulated INT8 grid. The next forward pass therefore sees
            # quantization noise during training.
            if should_apply_qat(epoch, num_batches, args):
                q_summary = fake_quantize_model_weights(
                    model, strategy=args.qat_strategy, num_bits=args.qat_bits
                )
                qat_error_sum += q_summary["avg_error"]
                qat_error_steps += 1

            total_loss += loss
            samples_seen += len(y_batch)

        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0

        test_acc = compute_accuracy_from_paths(
            model,
            test_paths,
            y_test,
            batch_size=args.batch_size,
            img_size=args.img_size,
            max_batches=args.eval_max_batches,
        )

        qat_msg = ""
        if args.qat:
            avg_qat_error = qat_error_sum / qat_error_steps if qat_error_steps > 0 else 0.0
            qat_msg = f" | QAT avg weight error: {avg_qat_error:.6f}"

        log(f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}% | "
            f"Test Accuracy: {test_acc:.2f}% | "
            f"Samples: {samples_seen}"
            f"{qat_msg}")

        if test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(model, args.save_path, classes=classes)
            log(f"  New best model saved ({best_acc:.2f}%, epoch {best_epoch})")

    log.section("RESULTS")
    log(f"Best test accuracy : {best_acc:.2f}%")
    log(f"Best epoch         : {best_epoch}")
    log(f"Model saved to     : {_npz_path(args.save_path)}")
    log("Next step          : Person 3 and Person 4 can load this .npz model.")
    log.close()

    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Person 1's SimpleCNN on Fruits-360 using pure NumPy"
    )
    parser.add_argument("--data_dir", type=str, default="./fruits-360-100x100",
                        help="Path to Fruits-360 root folder")
    parser.add_argument("--epochs", type=int, default=15,
                        help="Number of training epochs")
    parser.add_argument("--batch_size", type=int, default=32,
                        help="Mini-batch size")
    parser.add_argument("--lr", type=float, default=0.01,
                        help="Learning rate for SGD")
    parser.add_argument("--momentum", type=float, default=0.9,
                        help="Momentum factor for SGD")
    parser.add_argument("--save_path", type=str, default=None,
                        help="Where to save the trained model (.npz is added automatically). "
                             "Default: models/cnn_fruits for FP32 baseline, "
                             "models/cnn_fruits_qat when --qat is enabled")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Optional image limit per class for quick tests")
    parser.add_argument("--img_size", type=int, default=100,
                        help="Image size expected by the CNN")
    parser.add_argument("--eval_max_batches", type=int, default=None,
                        help="Optional evaluation batch limit for quick tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")
    parser.add_argument("--qat", action="store_true",
                        help="Enable Quantization-Aware Training style fake INT8 weight quantization")
    parser.add_argument("--qat_strategy", type=str, default="asymmetric",
                        choices=["symmetric", "asymmetric"],
                        help="Quantization strategy used during fake quantization")
    parser.add_argument("--qat_bits", type=int, default=8,
                        help="Number of bits for fake quantization; 8 simulates INT8")
    parser.add_argument("--qat_warmup_epochs", type=int, default=0,
                        help="Number of initial epochs trained as FP32 before applying QAT")
    parser.add_argument("--qat_frequency", type=int, default=1,
                        help="Apply fake quantization every N optimizer steps; 1 means every step")

    train(parser.parse_args())
