"""
person1_model_v2.py — Improved CNN architecture (V2).
PERSON 1's responsibility — improvement for KT2.

Improvements over V1 (person1_model.py):
    1. BatchNorm after each Conv layer  — stabilizes activations during training,
                                          allows higher learning rates, and acts
                                          as a mild regularizer.
    2. Dropout after FC1                — randomly drops 30% of features during
                                          training, which prevents FC1 from
                                          memorizing the training set (the main
                                          source of overfitting in V1).
    3. One additional Conv block (32 filters) — adds depth without exploding
                                                 parameter count. Spatial dims
                                                 go 100 -> 50 -> 25 -> 12.
    4. Smaller FC1                       — input is now 32*12*12 = 4608, not
                                            10000, so FC1 shrinks from ~1.28M
                                            to ~590K parameters (less overfit).

All new layers (BatchNormLayer, DropoutLayer) implement forward(training=True/False)
which matters: during inference (test accuracy / pruning / quantization)
BatchNorm uses running stats and Dropout becomes identity.

Public interface stays identical to V1:
    SimpleCNNv2(num_classes=261)
    .forward(X) -> probs
    .backward(d_probs)
    .get_trainable_layers()   -> only ConvLayer / FCLayer (same as V1)
    .get_param_count()        -> int

This means Borna's training loop, Silvia's pruning, and Jakov's quantization
work UNCHANGED on SimpleCNNv2 — they only ever touch ConvLayer / FCLayer.
"""

import numpy as np

# ── Reuse V1 layers as building blocks ────────────────────────────────────────
from person1_model import (
    ConvLayer, ReLULayer, MaxPoolLayer, FlattenLayer,
    FCLayer, SoftmaxLayer,
)


# ══════════════════════════════════════════════════════════════════════════════
#  NEW LAYER 1 — BatchNormLayer
# ══════════════════════════════════════════════════════════════════════════════

class BatchNormLayer:
    """
    Spatial (2D) batch normalization for convolutional outputs.

    For each of C channels, normalizes across the (N, H, W) dimensions:
        mean = mean over batch + spatial positions for this channel
        var  = variance over batch + spatial positions for this channel
        x_hat = (x - mean) / sqrt(var + eps)
        y = gamma * x_hat + beta

    gamma (scale) and beta (shift) are learned per-channel; running stats
    are tracked so test-time forward uses fixed mean/var instead of batch stats.

    NOTE — this layer has learnable gamma/beta but it is NOT exposed via
    get_trainable_layers() because Borna's save/load, Silvia's pruning, and
    Jakov's quantization assume layers expose either .filters or .weights.
    Adding BatchNorm to the trainable list would break their code.

    Instead, BatchNorm params are saved/loaded separately via gamma/beta
    attributes — this is documented and intentional.
    """

    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        self.num_features = num_features
        self.eps          = eps
        self.momentum     = momentum

        # Learnable parameters
        self.gamma = np.ones(num_features, dtype=np.float32)
        self.beta  = np.zeros(num_features, dtype=np.float32)

        # Running stats for inference
        self.running_mean = np.zeros(num_features, dtype=np.float32)
        self.running_var  = np.ones(num_features, dtype=np.float32)

        # Gradients (populated in backward)
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta  = np.zeros_like(self.beta)

        self.training = True

    def forward(self, x):
        """x : (N, C, H, W)"""
        N, C, H, W = x.shape

        if self.training:
            # Mean/var across (N, H, W), separately per channel
            mean = x.mean(axis=(0, 2, 3), keepdims=True)  # (1, C, 1, 1)
            var  = x.var(axis=(0, 2, 3), keepdims=True)   # (1, C, 1, 1)

            # Update running stats
            self.running_mean = (
                self.momentum * self.running_mean
                + (1 - self.momentum) * mean.reshape(-1)
            )
            self.running_var = (
                self.momentum * self.running_var
                + (1 - self.momentum) * var.reshape(-1)
            )
        else:
            mean = self.running_mean.reshape(1, C, 1, 1)
            var  = self.running_var.reshape(1, C, 1, 1)

        # Normalize
        x_hat = (x - mean) / np.sqrt(var + self.eps)

        # Cache for backward
        self._x_hat = x_hat
        self._var   = var
        self._mean  = mean
        self._x     = x

        # Scale and shift
        return self.gamma.reshape(1, C, 1, 1) * x_hat + self.beta.reshape(1, C, 1, 1)

    def backward(self, d_out):
        """
        Backward pass for BatchNorm.
        d_out : (N, C, H, W) — gradient from layer above
        returns: (N, C, H, W) — gradient w.r.t. input

        Reference: derivation in Ioffe & Szegedy 2015.
        """
        N, C, H, W = d_out.shape
        M = N * H * W            # total elements per channel

        x_hat = self._x_hat
        var   = self._var
        gamma_r = self.gamma.reshape(1, C, 1, 1)

        # Gradients of learnable params (sum over N, H, W)
        self.d_gamma = np.sum(d_out * x_hat, axis=(0, 2, 3))
        self.d_beta  = np.sum(d_out,         axis=(0, 2, 3))

        # Gradient w.r.t. input — standard BatchNorm derivative
        d_x_hat = d_out * gamma_r
        inv_std = 1.0 / np.sqrt(var + self.eps)

        sum_d_x_hat        = np.sum(d_x_hat,           axis=(0, 2, 3), keepdims=True)
        sum_d_x_hat_x_hat  = np.sum(d_x_hat * x_hat,   axis=(0, 2, 3), keepdims=True)

        d_x = (1.0 / M) * inv_std * (
            M * d_x_hat
            - sum_d_x_hat
            - x_hat * sum_d_x_hat_x_hat
        )
        return d_x

    def set_training(self, mode):
        self.training = mode


# ══════════════════════════════════════════════════════════════════════════════
#  NEW LAYER 2 — DropoutLayer
# ══════════════════════════════════════════════════════════════════════════════

class DropoutLayer:
    """
    Inverted dropout: during training, randomly zeroes p fraction of units
    and scales the rest by 1/(1-p) so expected sum stays the same.
    During inference, dropout is a no-op.

    Has no learnable parameters — not in get_trainable_layers().
    """

    def __init__(self, p=0.3):
        self.p        = p           # fraction to drop
        self.training = True
        self._mask    = None

    def forward(self, x):
        if not self.training or self.p == 0:
            return x

        # Binary mask, scaled by 1/(1-p) — "inverted dropout"
        self._mask = (np.random.rand(*x.shape) > self.p).astype(np.float32) / (1.0 - self.p)
        return x * self._mask

    def backward(self, d_out):
        if not self.training or self.p == 0:
            return d_out
        return d_out * self._mask

    def set_training(self, mode):
        self.training = mode


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLECNNv2 — the improved model
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCNNv2:
    """
    Improved CNN architecture (V2).

    Architecture
    ------------
    Conv(3 -> 8,   3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)   -> (N, 8,  50, 50)
    Conv(8 -> 16,  3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)   -> (N, 16, 25, 25)
    Conv(16 -> 32, 3x3, pad=1) -> BatchNorm -> ReLU -> MaxPool(2x2)   -> (N, 32, 12, 12)   [NEW]
    Flatten                                                            -> (N, 4608)
    FC(4608 -> 128) -> ReLU -> Dropout(p=0.3)                         -> (N, 128)
    FC(128 -> num_classes)
    Softmax

    Public interface — identical to V1 SimpleCNN
    --------------------------------------------
    forward(x)              -> (N, num_classes) probabilities
    backward(d_probs)       -> populates gradients on each ConvLayer / FCLayer
    get_trainable_layers()  -> [Conv1, Conv2, Conv3, FC1, FC2] (Conv/FC only)
    get_param_count()       -> int

    Why only Conv/FC in get_trainable_layers()?
    -------------------------------------------
    Borna's SGDMomentum, save_model, load_model all iterate over
    get_trainable_layers() and assume each item has .filters or .weights.
    Silvia's pruning and Jakov's quantization make the same assumption.

    BatchNorm gamma/beta and Dropout (no params) are intentionally hidden
    from this list so the rest of the pipeline keeps working.
    BatchNorm params ARE still updated during backward (via a custom
    optimizer step inside this class — see _optimizer_step_for_bn below)
    and saved/loaded separately.
    """

    def __init__(self, num_classes=261, dropout_p=0.3):
        self.num_classes = num_classes
        self.dropout_p   = dropout_p

        # ── Block 1 ──────────────────────────────────────────────────────────
        self.conv1 = ConvLayer(3, 8, kernel_size=3, padding=1)
        self.bn1   = BatchNormLayer(8)
        self.relu1 = ReLULayer()
        self.pool1 = MaxPoolLayer(2, 2)

        # ── Block 2 ──────────────────────────────────────────────────────────
        self.conv2 = ConvLayer(8, 16, kernel_size=3, padding=1)
        self.bn2   = BatchNormLayer(16)
        self.relu2 = ReLULayer()
        self.pool2 = MaxPoolLayer(2, 2)

        # ── Block 3 (NEW) ────────────────────────────────────────────────────
        self.conv3 = ConvLayer(16, 32, kernel_size=3, padding=1)
        self.bn3   = BatchNormLayer(32)
        self.relu3 = ReLULayer()
        self.pool3 = MaxPoolLayer(2, 2)

        # ── Head ─────────────────────────────────────────────────────────────
        self.flatten = FlattenLayer()
        # After 3 pools: 100 -> 50 -> 25 -> 12 (floor(25/2) = 12)
        # Flatten size: 32 * 12 * 12 = 4608
        self.fc1     = FCLayer(32 * 12 * 12, 128)
        self.relu4   = ReLULayer()
        self.dropout = DropoutLayer(p=dropout_p)

        self.fc2     = FCLayer(128, num_classes)
        self.softmax = SoftmaxLayer()

        # Ordered list (for forward/backward iteration)
        self._layers = [
            self.conv1, self.bn1, self.relu1, self.pool1,
            self.conv2, self.bn2, self.relu2, self.pool2,
            self.conv3, self.bn3, self.relu3, self.pool3,
            self.flatten,
            self.fc1, self.relu4, self.dropout,
            self.fc2, self.softmax,
        ]

        # Track BatchNorm layers separately — they need their own optimizer step
        self._bn_layers = [self.bn1, self.bn2, self.bn3]

        # Simple SGD state for BN params (separate from Borna's optimizer)
        self._bn_lr = 0.01
        self._bn_momentum = 0.9
        self._bn_velocity = {}
        for i, bn in enumerate(self._bn_layers):
            self._bn_velocity[f"bn{i}_gamma"] = np.zeros_like(bn.gamma)
            self._bn_velocity[f"bn{i}_beta"]  = np.zeros_like(bn.beta)

    # ── training mode toggle ─────────────────────────────────────────────────

    def train(self):
        """Switch to training mode (BN uses batch stats, Dropout drops)."""
        for layer in self._layers:
            if hasattr(layer, "set_training"):
                layer.set_training(True)

    def eval(self):
        """Switch to inference mode (BN uses running stats, Dropout off)."""
        for layer in self._layers:
            if hasattr(layer, "set_training"):
                layer.set_training(False)

    # ── forward / backward ───────────────────────────────────────────────────

    def forward(self, x):
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, d_probs):
        """
        Backprop through all layers. After this call:
            - ConvLayer / FCLayer have .d_filters / .d_weights / .d_biases set
              (Borna's optimizer will read these)
            - BatchNormLayer has .d_gamma / .d_beta set
              (we update these ourselves via _bn_step below)
        """
        grad = d_probs
        for layer in reversed(self._layers):
            grad = layer.backward(grad)

        # Borna's SGDMomentum will be called externally and updates Conv/FC.
        # BatchNorm params are not in his loop, so we update them here.
        self._bn_step()

    def _bn_step(self):
        """SGD-with-momentum update for BatchNorm gamma/beta after backward()."""
        for i, bn in enumerate(self._bn_layers):
            v_g = self._bn_velocity[f"bn{i}_gamma"]
            v_b = self._bn_velocity[f"bn{i}_beta"]

            v_g = self._bn_momentum * v_g - self._bn_lr * bn.d_gamma
            v_b = self._bn_momentum * v_b - self._bn_lr * bn.d_beta

            bn.gamma += v_g
            bn.beta  += v_b

            self._bn_velocity[f"bn{i}_gamma"] = v_g
            self._bn_velocity[f"bn{i}_beta"]  = v_b

    def set_bn_lr(self, lr):
        """Allow comparison script / Borna's loop to keep BN lr in sync."""
        self._bn_lr = lr

    # ── public interface (matches V1) ────────────────────────────────────────

    def get_trainable_layers(self):
        """
        Returns only Conv/FC layers — same convention as V1.
        BatchNorm params are managed internally and saved/loaded
        via save_bn_params() / load_bn_params() below.
        """
        return [l for l in self._layers if isinstance(l, (ConvLayer, FCLayer))]

    def get_param_count(self):
        """Total learnable parameters (includes BatchNorm gamma/beta)."""
        total = 0
        for layer in self.get_trainable_layers():
            if hasattr(layer, 'filters'):
                total += layer.filters.size + layer.biases.size
            else:
                total += layer.weights.size + layer.biases.size
        # Add BatchNorm parameters (gamma + beta per channel)
        for bn in self._bn_layers:
            total += bn.gamma.size + bn.beta.size
        return total

    # ── extra: save/load BatchNorm params alongside Borna's save_model ───────

    def save_bn_params(self, path):
        """Save BatchNorm gamma/beta/running_mean/running_var to .npz."""
        if not path.endswith(".npz"):
            path = path + "_bn.npz"
        else:
            path = path.replace(".npz", "_bn.npz")

        bn_data = {}
        for i, bn in enumerate(self._bn_layers):
            bn_data[f"bn{i}_gamma"]        = bn.gamma
            bn_data[f"bn{i}_beta"]         = bn.beta
            bn_data[f"bn{i}_running_mean"] = bn.running_mean
            bn_data[f"bn{i}_running_var"]  = bn.running_var
        np.savez(path, **bn_data)

    def load_bn_params(self, path):
        """Load BatchNorm params from .npz produced by save_bn_params."""
        if not path.endswith(".npz"):
            path = path + "_bn.npz"
        else:
            path = path.replace(".npz", "_bn.npz")

        data = np.load(path)
        for i, bn in enumerate(self._bn_layers):
            bn.gamma        = data[f"bn{i}_gamma"]
            bn.beta         = data[f"bn{i}_beta"]
            bn.running_mean = data[f"bn{i}_running_mean"]
            bn.running_var  = data[f"bn{i}_running_var"]


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST — runs without dataset (same approach as V1)
# ══════════════════════════════════════════════════════════════════════════════

def _cross_entropy(probs, labels):
    N   = len(labels)
    eps = 1e-9
    loss = -np.mean(np.log(probs[np.arange(N), labels] + eps))
    d    = probs.copy()
    d[np.arange(N), labels] -= 1
    d   /= N
    return loss, d


def _sgd_step(model, lr=0.01):
    for layer in model.get_trainable_layers():
        if hasattr(layer, 'filters'):
            layer.filters -= lr * layer.d_filters
            layer.biases  -= lr * layer.d_biases
        else:
            layer.weights -= lr * layer.d_weights
            layer.biases  -= lr * layer.d_biases


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="SimpleCNNv2 self-test — trains on fake data (no dataset needed)"
    )
    parser.add_argument("--epochs",      type=int, default=3)
    parser.add_argument("--batch_size",  type=int, default=4)
    parser.add_argument("--num_classes", type=int, default=10)
    parser.add_argument("--num_samples", type=int, default=40)
    args = parser.parse_args()

    print("=" * 60)
    print("  SimpleCNNv2 self-test (V2 = BatchNorm + Dropout + extra Conv)")
    print("=" * 60)

    model = SimpleCNNv2(num_classes=args.num_classes)
    print(f"\n  Parameters         : {model.get_param_count():,}")
    print(f"  Trainable Conv/FC  : {len(model.get_trainable_layers())}")
    print(f"  BatchNorm layers   : {len(model._bn_layers)}")
    print(f"  Dropout p          : {model.dropout_p}")

    for i, l in enumerate(model.get_trainable_layers()):
        kind = "Conv" if hasattr(l, 'filters') else "FC  "
        w    = l.filters if hasattr(l, 'filters') else l.weights
        print(f"    [{i}] {kind}  shape={w.shape}  params={w.size + l.biases.size:,}")

    # Smoke test — forward + backward shapes
    print(f"\n  [Check 1] Forward pass (training mode) ... ", end="", flush=True)
    model.train()
    dummy = np.random.randn(2, 3, 100, 100).astype(np.float32)
    probs = model.forward(dummy)
    assert probs.shape == (2, args.num_classes), \
        f"Bad shape: got {probs.shape}, expected (2, {args.num_classes})"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-4), \
        f"Softmax doesn't sum to 1: {probs.sum(axis=1)}"
    print(f"OK — {probs.shape}, sums to 1.0  ✓")

    print(f"  [Check 2] Forward pass (eval mode) ... ", end="", flush=True)
    model.eval()
    probs_eval = model.forward(dummy)
    assert probs_eval.shape == (2, args.num_classes)
    print("OK  ✓")

    print(f"  [Check 3] Backward pass ... ", end="", flush=True)
    model.train()
    probs = model.forward(dummy)
    fake_labels = np.array([0, 1])
    _, d_probs = _cross_entropy(probs, fake_labels)
    model.backward(d_probs)
    for i, l in enumerate(model.get_trainable_layers()):
        if hasattr(l, 'filters'):
            assert l.d_filters.shape == l.filters.shape
        else:
            assert l.d_weights.shape == l.weights.shape
    print("OK  ✓")

    # Mini training loop
    print(f"\n  [Check 4] Mini training loop ({args.num_samples} fake samples, "
          f"{args.epochs} epochs)\n")

    np.random.seed(42)
    X_fake = np.random.randn(args.num_samples, 3, 100, 100).astype(np.float32)
    y_fake = np.random.randint(0, args.num_classes, size=args.num_samples)

    model = SimpleCNNv2(num_classes=args.num_classes)
    first_loss, last_loss = None, None

    for epoch in range(1, args.epochs + 1):
        model.train()
        epoch_loss = 0.0
        num_batches = 0
        idx = np.random.permutation(args.num_samples)

        for start in range(0, args.num_samples, args.batch_size):
            batch_idx = idx[start:start + args.batch_size]
            probs        = model.forward(X_fake[batch_idx])
            loss, d_probs = _cross_entropy(probs, y_fake[batch_idx])
            epoch_loss  += loss
            num_batches += 1
            model.backward(d_probs)
            _sgd_step(model, lr=0.01)

        avg_loss = epoch_loss / num_batches

        model.eval()
        all_probs = model.forward(X_fake)
        acc = 100.0 * np.mean(np.argmax(all_probs, axis=1) == y_fake)

        print(f"    Epoch [{epoch}/{args.epochs}]  Loss: {avg_loss:.4f}  |  Accuracy: {acc:.1f}%")
        if first_loss is None:
            first_loss = avg_loss
        last_loss = avg_loss

    decreased = last_loss < first_loss
    print(f"\n  First epoch loss: {first_loss:.4f}")
    print(f"  Last  epoch loss: {last_loss:.4f}  "
          f"{'✓ Loss decreased' if decreased else '⚠ Loss did not decrease'}")
    print("\n  ✓ SimpleCNNv2 ready." if decreased else "\n  ✗ Something is off.")
    print("=" * 60)
