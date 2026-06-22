"""
person3_dynamic_pruning.py — Dynamic magnitude pruning for SimpleCNNv2.

This script is based on the training pipeline, but adds gradual pruning:
training + pruning schedule + mask re-application + sparsity tracking.

It is separate from person2_train.py so the original training code stays unchanged.
"""

import os
import argparse
import numpy as np
from PIL import Image

from logger import Logger
from person1_model_v2 import SimpleCNNv2


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

    was_training = getattr(model, "dropout", None) is not None and model.dropout.training
    if hasattr(model, "eval"):
        model.eval()

    correct = 0
    total = 0

    for X_batch, y_batch in get_batches(X, y, batch_size, shuffle=False):
        probs = model.forward(X_batch)
        batch_correct, batch_total = batch_accuracy(probs, y_batch)
        correct += batch_correct
        total += batch_total

    if was_training and hasattr(model, "train"):
        model.train()

    return 100.0 * correct / total if total > 0 else 0.0


def compute_accuracy_from_paths(model, paths, labels, batch_size=32,
                                img_size=100, max_batches=None):
    """
    Evaluate accuracy using the streaming path-based loader.

    max_batches is optional and useful only for quick technical checks. By
    default it is None, meaning the full evaluation split is used.
    """
    was_training = getattr(model, "dropout", None) is not None and model.dropout.training
    if hasattr(model, "eval"):
        model.eval()

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

    if was_training and hasattr(model, "train"):
        model.train()

    return 100.0 * correct / total if total > 0 else 0.0


#DYNAMIC PRUNING
def get_weight_array(layer):
    """return the prunable weight tensor for convlayer or FCLayer"""
    if hasattr(layer, "filters"):
        return layer.filters
    return layer.weights


def create_all_one_masks(model):
    """create one binary mask for each prunable conv/FC weight tensor"""
    masks = {}
    for i, layer in enumerate(model.get_trainable_layers()):
        masks[i] = np.ones_like(get_weight_array(layer), dtype=np.float32)
    return masks


def apply_masks(model, masks):
    """force already-pruned weights to remain exactly zero """
    for i, layer in enumerate(model.get_trainable_layers()):
        get_weight_array(layer)[:] *= masks[i]


def count_nonzero_weights(model):
    """count active/pruned conv and FC weights, biases and batchnorm are not pruned"""
    total = 0
    nonzero = 0
    for layer in model.get_trainable_layers():
        weights = get_weight_array(layer)
        total += weights.size
        nonzero += int(np.count_nonzero(weights))

    zero = total - nonzero
    sparsity = 100.0 * zero / total if total > 0 else 0.0
    return {
        "total": total,
        "nonzero": nonzero,
        "zero": zero,
        "sparsity": sparsity,
    }


def target_sparsity_for_step(current_step, start_step, end_step, final_amount):
    """
    - cubic gradual pruning schedule based on training steps
    - sparsity increases smoothly from 0 to final_amount between start_step and end_step
    """
    if current_step < start_step:
        return 0.0

    if current_step >= end_step:
        return final_amount

    progress = (current_step - start_step + 1) / max(1, end_step - start_step)
    progress = min(max(progress, 0.0), 1.0)

    return final_amount * (1.0 - (1.0 - progress) ** 3)


def update_global_masks(model, masks, target_sparsity):
    active_abs_weights = []
    active_refs = []

    for i, layer in enumerate(model.get_trainable_layers()):
        weights = get_weight_array(layer)
        active = masks[i].reshape(-1) > 0
        if np.any(active):
            active_abs_weights.append(np.abs(weights).reshape(-1)[active])
            active_refs.append((i, active))

    if not active_abs_weights:
        return masks

    total_weights = sum(mask.size for mask in masks.values())
    target_pruned = int(total_weights * target_sparsity)
    current_pruned = sum(mask.size - int(np.count_nonzero(mask)) for mask in masks.values())
    prune_now = target_pruned - current_pruned

    if prune_now <= 0:
        return masks

    flat_active = np.concatenate(active_abs_weights)
    prune_now = min(prune_now, flat_active.size)
    threshold = np.partition(flat_active, prune_now - 1)[prune_now - 1]

    remaining_to_prune = prune_now
    for i, active in active_refs:
        flat_weights = np.abs(get_weight_array(model.get_trainable_layers()[i])).reshape(-1)
        flat_mask = masks[i].reshape(-1)
        candidates = np.where(active & (flat_weights <= threshold))[0]
        if candidates.size > remaining_to_prune:
            order = np.argsort(flat_weights[candidates])
            candidates = candidates[order[:remaining_to_prune]]

        flat_mask[candidates] = 0.0
        remaining_to_prune -= candidates.size
        if remaining_to_prune <= 0:
            break

    apply_masks(model, masks)
    return masks


def update_per_layer_masks(model, masks, target_sparsity):
    for i, layer in enumerate(model.get_trainable_layers()):
        weights = get_weight_array(layer)
        flat_weights = np.abs(weights).reshape(-1)
        flat_mask = masks[i].reshape(-1)

        target_pruned = int(flat_mask.size * target_sparsity)
        current_pruned = flat_mask.size - int(np.count_nonzero(flat_mask))
        prune_now = target_pruned - current_pruned

        if prune_now <= 0:
            continue

        active_indices = np.where(flat_mask > 0)[0]
        prune_now = min(prune_now, active_indices.size)
        active_values = flat_weights[active_indices]
        prune_order = np.argpartition(active_values, prune_now - 1)[:prune_now]
        flat_mask[active_indices[prune_order]] = 0.0

    apply_masks(model, masks)
    return masks


def update_pruning_masks(model, masks, strategy, target_sparsity):
    if not 0.0 <= target_sparsity < 1.0:
        raise ValueError("target_sparsity must be between 0.0 and 1.0")

    if strategy == "global":
        return update_global_masks(model, masks, target_sparsity)
    if strategy == "per_layer":
        return update_per_layer_masks(model, masks, target_sparsity)

    raise ValueError("strategy must be 'global' or 'per_layer'")



#TRAINING LOOP
def train(args):
    """ train simpleCNNv2 while progressively pruning weights """
    np.random.seed(args.seed)
    log = Logger("person3_dynamic_pruning")

    prune_end_epoch = args.prune_end_epoch or args.epochs
    if args.prune_start_epoch < 1:
        raise ValueError("--prune_start_epoch must be at least 1")
    if prune_end_epoch < args.prune_start_epoch:
        raise ValueError("--prune_end_epoch must be >= --prune_start_epoch")
    if not 0.0 <= args.amount < 1.0:
        raise ValueError("--amount must be between 0.0 and 1.0")
    if args.prune_frequency < 1:
        raise ValueError("--prune_frequency must be at least 1")

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

    log.section("MODEL")
    model = SimpleCNNv2(num_classes=num_classes, dropout_p=args.dropout_p)
    model.set_bn_lr(args.lr)
    optimizer = SGDMomentum(model, lr=args.lr, momentum=args.momentum)
    masks = create_all_one_masks(model)

    log(f"Architecture : SimpleCNNv2 from person1_model_v2.py")
    log(f"Parameters   : {model.get_param_count():,}")
    log(f"Optimizer    : SGD with momentum={args.momentum}")
    log(f"Loss         : cross-entropy")
    log(f"Dropout      : p={args.dropout_p}")
    log("")

    log.section("TRAINING")
    log(f"Epochs       : {args.epochs}")
    log(f"Batch size   : {args.batch_size}")
    log(f"Learning rate: {args.lr}")
    log("Train metric : running train accuracy from the same forward pass used for loss")
    log(f"Pruning      : {args.strategy}, target sparsity={args.amount * 100:.1f}%")
    log(f"Schedule     : epoch {args.prune_start_epoch} through {prune_end_epoch}")
    log(f"Frequency    : every {args.prune_frequency} batch(es)")
    if args.eval_max_batches is not None:
        log(f"Evaluation   : first {args.eval_max_batches} test batch(es) only")
    else:
        log("Evaluation   : full Test split")
    log("")

    best_acc = -1.0
    best_epoch = 0

    steps_per_epoch = int(np.ceil(len(train_paths) / args.batch_size))
    total_steps = args.epochs * steps_per_epoch

    prune_start_step = (args.prune_start_epoch - 1) * steps_per_epoch
    prune_end_epoch = args.prune_end_epoch or args.epochs
    prune_end_step = prune_end_epoch * steps_per_epoch

    global_step = 0

    for epoch in range(1, args.epochs + 1):
        total_loss = 0.0
        num_batches = 0
        samples_seen = 0
        train_correct = 0
        train_total = 0
        model.train()

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
            apply_masks(model, masks)

            global_step += 1

            if (
                global_step >= prune_start_step
                and global_step <= prune_end_step
                and global_step % args.prune_frequency == 0
            ):
                target_sparsity = target_sparsity_for_step(
                    global_step,
                    prune_start_step,
                    prune_end_step,
                    args.amount,
                )

                update_pruning_masks(model, masks, args.strategy, target_sparsity)

            total_loss += loss
            num_batches += 1
            samples_seen += len(y_batch)


        avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
        train_acc = 100.0 * train_correct / train_total if train_total > 0 else 0.0
        stats = count_nonzero_weights(model)

        test_acc = compute_accuracy_from_paths(
            model,
            test_paths,
            y_test,
            batch_size=args.batch_size,
            img_size=args.img_size,
            max_batches=args.eval_max_batches,
        )

        log(f"Epoch [{epoch:>3}/{args.epochs}] "
            f"Train Loss: {avg_loss:.4f} | "
            f"Train Accuracy: {train_acc:.2f}% | "
            f"Test Accuracy: {test_acc:.2f}% | "
            f"Sparsity: {stats['sparsity']:.2f}% | "
            f"Samples: {samples_seen}")

        if stats["sparsity"] > 0 and test_acc > best_acc:
            best_acc = test_acc
            best_epoch = epoch
            save_model(model, args.save_path, classes=classes)
            model.save_bn_params(args.save_path)
            mask_path = _npz_path(args.mask_path)
            parent = os.path.dirname(mask_path)
            if parent:
                os.makedirs(parent, exist_ok=True)
            np.savez(mask_path, **{f"mask_layer{i}": mask for i, mask in masks.items()})
            log(f"  New best model saved ({best_acc:.2f}%, epoch {best_epoch})")

    final_stats = count_nonzero_weights(model)
    log.section("RESULTS")
    log(f"Total steps        : {total_steps}")
    log(f"Seed               : {args.seed}")
    log(f"Best test accuracy : {best_acc:.2f}%")
    log(f"Best epoch         : {best_epoch}")
    log(f"Final sparsity     : {final_stats['sparsity']:.2f}%")
    log(f"Model saved to     : {_npz_path(args.save_path)}")
    log(f"BatchNorm saved to : {_npz_path(args.save_path).replace('.npz', '_bn.npz')}")
    log(f"Masks saved to     : {_npz_path(args.mask_path)}")
    log.close()

    return model, best_acc


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train Person 1's SimpleCNNv2 with dynamic magnitude pruning"
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
    parser.add_argument("--save_path", type=str, default="models/cnn_dynamic_pruned",
                        help="Where to save the best pruned model (.npz is added automatically)")
    parser.add_argument("--mask_path", type=str, default="models/dynamic_pruning_masks",
                        help="Where to save pruning masks (.npz is added automatically)")
    parser.add_argument("--strategy", type=str, default="global",
                        choices=["global", "per_layer"],
                        help="Dynamic pruning strategy")
    parser.add_argument("--amount", type=float, default=0.5,
                        help="Final target sparsity, e.g. 0.5 prunes 50%% of Conv/FC weights")
    parser.add_argument("--prune_start_epoch", type=int, default=2,
                        help="First epoch where pruning may update masks")
    parser.add_argument("--prune_end_epoch", type=int, default=None,
                        help="Last epoch for gradual pruning; defaults to --epochs")
    parser.add_argument("--prune_frequency", type=int, default=10,
                        help="Update pruning masks every N training batches")
    parser.add_argument("--dropout_p", type=float, default=0.3,
                        help="Dropout probability for SimpleCNNv2")
    parser.add_argument("--max_per_class", type=int, default=None,
                        help="Optional image limit per class for quick tests")
    parser.add_argument("--img_size", type=int, default=100,
                        help="Image size expected by the CNN")
    parser.add_argument("--eval_max_batches", type=int, default=None,
                        help="Optional evaluation batch limit for quick tests")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility")

    train(parser.parse_args())
