"""
person1_model.py — CNN architecture built from scratch using only NumPy.
PERSON 1's responsibility.

Defines the full SimpleCNN used by all other parts of the project:
    - ConvLayer     : 2D convolution + bias, stores gradients d_filters / d_biases
    - ReLULayer     : element-wise ReLU activation
    - MaxPoolLayer  : 2×2 max-pooling with stride 2
    - FCLayer       : fully-connected (dense) layer, stores gradients d_weights / d_biases
    - SoftmaxLayer  : softmax over class logits (forward only — gradient handled externally)
    - SimpleCNN     : assembles the above into a complete model

Architecture:
    Input  : (N, 3, 100, 100)   — RGB 100×100 images
    Conv1  : 8 filters, 3×3, pad=1  → (N,  8, 100, 100)
    ReLU1
    Pool1  : 2×2, stride 2          → (N,  8,  50,  50)
    Conv2  : 16 filters, 3×3, pad=1 → (N, 16,  50,  50)
    ReLU2
    Pool2  : 2×2, stride 2          → (N, 16,  25,  25)
    Flatten                          → (N, 16*25*25) = (N, 10000)
    FC1    : 10000 → 128
    ReLU3
    FC2    : 128   → num_classes
    Softmax

All layers implement:
    forward(x)  → output
    backward(d) → gradient w.r.t. input

ConvLayer and FCLayer also expose:
    .filters / .weights   — learnable parameters
    .biases
    .d_filters / .d_weights  — gradients (set during backward)
    .d_biases

Usage:
    from person1_model import SimpleCNN, ConvLayer, FCLayer
    model = SimpleCNN(num_classes=20)
    probs = model.forward(X_batch)   # (N, num_classes)
    model.backward(d_probs)
"""

import numpy as np


# ══════════════════════════════════════════════════════════════════════════════
#  LAYER IMPLEMENTATIONS
# ══════════════════════════════════════════════════════════════════════════════

class ConvLayer:
    """
    2D Convolutional layer — pure NumPy.

    Parameters
    ----------
    in_channels  : number of input feature maps
    out_channels : number of filters (output feature maps)
    kernel_size  : square kernel side length (default 3)
    padding      : zero-padding added to each spatial side (default 1)
                   padding=1 with kernel_size=3 keeps spatial size unchanged.

    Attributes set after forward()
    --------------------------------
    d_filters, d_biases : gradients ready for the optimizer
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.kernel_size  = kernel_size
        self.padding      = padding

        # He initialisation — good default for ReLU networks
        fan_in = in_channels * kernel_size * kernel_size
        self.filters = np.random.randn(
            out_channels, in_channels, kernel_size, kernel_size
        ).astype(np.float32) * np.sqrt(2.0 / fan_in)

        self.biases = np.zeros(out_channels, dtype=np.float32)

        # Gradient placeholders (set during backward)
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases  = np.zeros_like(self.biases)

    def forward(self, x):
        """
        x : (N, C_in, H, W)
        returns : (N, C_out, H_out, W_out)
        """
        self._x = x
        N, C, H, W = x.shape
        F, _, kH, kW = self.filters.shape
        p = self.padding

        H_out = H + 2 * p - kH + 1
        W_out = W + 2 * p - kW + 1

        # Zero-pad input
        x_pad = np.pad(x, ((0,0),(0,0),(p,p),(p,p)), mode='constant')
        self._x_pad = x_pad

        out = np.zeros((N, F, H_out, W_out), dtype=np.float32)

        # Use im2col-style vectorised approach for speed
        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, :, i:i+kH, j:j+kW]          # (N, C, kH, kW)
                patch_flat = patch.reshape(N, -1)              # (N, C*kH*kW)
                filt_flat  = self.filters.reshape(F, -1).T     # (C*kH*kW, F)
                out[:, :, i, j] = patch_flat @ filt_flat + self.biases  # (N, F)

        return out

    def backward(self, d_out):
        """
        d_out : gradient from the layer above, shape (N, F, H_out, W_out)
        returns: gradient w.r.t. input x, shape (N, C, H, W)
        """
        x_pad = self._x_pad
        N, F, H_out, W_out = d_out.shape
        _, C, kH, kW = self.filters.shape
        p = self.padding

        self.d_filters = np.zeros_like(self.filters)
        self.d_biases  = np.zeros_like(self.biases)
        d_x_pad        = np.zeros_like(x_pad)

        # Bias gradient: sum over N, H_out, W_out for each filter
        self.d_biases = np.sum(d_out, axis=(0, 2, 3))

        filt_flat = self.filters.reshape(F, -1)   # (F, C*kH*kW)

        for i in range(H_out):
            for j in range(W_out):
                patch = x_pad[:, :, i:i+kH, j:j+kW]   # (N, C, kH, kW)
                patch_flat = patch.reshape(N, -1)        # (N, C*kH*kW)
                g = d_out[:, :, i, j]                   # (N, F)

                # Filter gradient: sum over batch
                self.d_filters += (g.T @ patch_flat).reshape(self.filters.shape)

                # Input gradient
                d_patch = (g @ filt_flat).reshape(N, C, kH, kW)  # (N,C,kH,kW)
                d_x_pad[:, :, i:i+kH, j:j+kW] += d_patch

        # Remove padding to recover gradient w.r.t. original x
        if p > 0:
            d_x = d_x_pad[:, :, p:-p, p:-p]
        else:
            d_x = d_x_pad

        return d_x


class ReLULayer:
    """
    Element-wise ReLU: f(x) = max(0, x)
    Gradient: 1 where x > 0, else 0.
    """

    def forward(self, x):
        self._mask = (x > 0)
        return x * self._mask

    def backward(self, d_out):
        return d_out * self._mask


class MaxPoolLayer:
    """
    2×2 max-pooling with stride 2.
    Halves both spatial dimensions.
    During backward, gradient flows only to the max element in each window.
    """

    def __init__(self, pool_size=2, stride=2):
        self.pool_size = pool_size
        self.stride    = stride

    def forward(self, x):
        """x : (N, C, H, W)  →  (N, C, H//2, W//2)"""
        self._x = x
        N, C, H, W = x.shape
        p, s = self.pool_size, self.stride
        H_out = (H - p) // s + 1
        W_out = (W - p) // s + 1

        out = np.zeros((N, C, H_out, W_out), dtype=np.float32)

        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * s, j * s
                window = x[:, :, h_s:h_s+p, w_s:w_s+p]
                out[:, :, i, j] = np.max(window, axis=(2, 3))

        return out

    def backward(self, d_out):
        """Route gradient to the position of the max value."""
        x = self._x
        N, C, H, W = x.shape
        p, s = self.pool_size, self.stride
        H_out = (H - p) // s + 1
        W_out = (W - p) // s + 1

        d_x = np.zeros_like(x)

        for i in range(H_out):
            for j in range(W_out):
                h_s, w_s = i * s, j * s
                window = x[:, :, h_s:h_s+p, w_s:w_s+p]       # (N,C,p,p)
                max_v  = np.max(window, axis=(2, 3), keepdims=True)
                mask   = (window == max_v).astype(np.float32)
                # Normalise in case of ties
                mask  /= (np.sum(mask, axis=(2,3), keepdims=True) + 1e-8)
                d_x[:, :, h_s:h_s+p, w_s:w_s+p] += (
                    mask * d_out[:, :, i:i+1, j:j+1]
                )

        return d_x


class FlattenLayer:
    """Reshape (N, C, H, W) → (N, C*H*W) on forward, reverse on backward."""

    def forward(self, x):
        self._shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self._shape)


class FCLayer:
    """
    Fully-connected (dense) layer.

    Attributes
    ----------
    weights  : (in_features, out_features)
    biases   : (out_features,)
    d_weights, d_biases : gradients (set during backward)
    """

    def __init__(self, in_features, out_features):
        self.in_features  = in_features
        self.out_features = out_features

        # He initialisation
        self.weights = np.random.randn(
            in_features, out_features
        ).astype(np.float32) * np.sqrt(2.0 / in_features)

        self.biases   = np.zeros(out_features, dtype=np.float32)
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases  = np.zeros_like(self.biases)

    def forward(self, x):
        """x : (N, in_features)  →  (N, out_features)"""
        self._x = x
        return x @ self.weights + self.biases

    def backward(self, d_out):
        """
        d_out : (N, out_features)
        returns: (N, in_features)
        """
        self.d_weights = self._x.T @ d_out
        self.d_biases  = np.sum(d_out, axis=0)
        return d_out @ self.weights.T


class SoftmaxLayer:
    """
    Numerically stable Softmax.
    Note: backward is a pass-through — the combined cross-entropy + softmax
    gradient is computed externally in person2_train.py (cross_entropy_loss).
    """

    def forward(self, x):
        """x : (N, num_classes)  →  (N, num_classes) probabilities"""
        shifted = x - np.max(x, axis=1, keepdims=True)
        exp_x   = np.exp(shifted)
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)

    def backward(self, d_out):
        # Gradient handled externally via cross_entropy_loss
        return d_out


# ══════════════════════════════════════════════════════════════════════════════
#  SIMPLECNN — THE FULL MODEL
# ══════════════════════════════════════════════════════════════════════════════

class SimpleCNN:
    """
    A simple CNN for image classification on Fruits-360 (100×100 RGB).

    Architecture
    ------------
    Conv(3→8, 3×3, pad=1) → ReLU → MaxPool(2×2)   → spatial: 50×50
    Conv(8→16, 3×3, pad=1) → ReLU → MaxPool(2×2)  → spatial: 25×25
    Flatten                                          → 16*25*25 = 10 000
    FC(10000 → 128) → ReLU
    FC(128 → num_classes)
    Softmax

    Public interface (used by person2, person3, person4)
    -----------------------------------------------------
    model.forward(X)              — returns softmax probabilities (N, num_classes)
    model.backward(d_probs)       — runs backprop, stores gradients in each layer
    model.get_trainable_layers()  — list of ConvLayer / FCLayer objects
    model.get_param_count()       — total number of learnable parameters (int)
    """

    def __init__(self, num_classes=131):
        self.num_classes = num_classes

        self.conv1   = ConvLayer(in_channels=3,  out_channels=8,  kernel_size=3, padding=1)
        self.relu1   = ReLULayer()
        self.pool1   = MaxPoolLayer(pool_size=2, stride=2)

        self.conv2   = ConvLayer(in_channels=8,  out_channels=16, kernel_size=3, padding=1)
        self.relu2   = ReLULayer()
        self.pool2   = MaxPoolLayer(pool_size=2, stride=2)

        self.flatten = FlattenLayer()

        # After two 2×2 pools on 100×100: 100 → 50 → 25
        # Flattened size = 16 * 25 * 25 = 10000
        self.fc1     = FCLayer(in_features=16 * 25 * 25, out_features=128)
        self.relu3   = ReLULayer()

        self.fc2     = FCLayer(in_features=128, out_features=num_classes)
        self.softmax = SoftmaxLayer()

        self._layers = [
            self.conv1, self.relu1, self.pool1,
            self.conv2, self.relu2, self.pool2,
            self.flatten,
            self.fc1,   self.relu3,
            self.fc2,   self.softmax,
        ]

    def forward(self, x):
        """
        x      : (N, 3, 100, 100)  float32  — normalised [0, 1]
        returns: (N, num_classes)  float32  — softmax probabilities
        """
        out = x
        for layer in self._layers:
            out = layer.forward(out)
        return out

    def backward(self, d_probs):
        """
        d_probs : (N, num_classes) — gradient from cross_entropy_loss()
        Stores d_filters/d_biases in ConvLayers, d_weights/d_biases in FCLayers.
        """
        grad = d_probs
        for layer in reversed(self._layers):
            grad = layer.backward(grad)

    def get_trainable_layers(self):
        """
        Returns only layers with learnable parameters (ConvLayer, FCLayer).
        Used by person2 (save/load/optimizer), person3 (pruning), person4 (quantization).
        Ordered same as forward pass so index-based naming is consistent.
        """
        return [l for l in self._layers if isinstance(l, (ConvLayer, FCLayer))]

    def get_param_count(self):
        """Total number of learnable parameters across all layers."""
        total = 0
        for layer in self.get_trainable_layers():
            if hasattr(layer, 'filters'):
                total += layer.filters.size + layer.biases.size
            else:
                total += layer.weights.size + layer.biases.size
        return total


# ══════════════════════════════════════════════════════════════════════════════
#  SMOKE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("=" * 55)
    print("  SimpleCNN — smoke test")
    print("=" * 55)

    NUM_CLASSES = 20
    BATCH       = 2    # keep small so the naive loops finish quickly

    model = SimpleCNN(num_classes=NUM_CLASSES)
    print(f"  Parameters : {model.get_param_count():,}")
    print(f"  Trainable layers ({len(model.get_trainable_layers())}):")
    for i, l in enumerate(model.get_trainable_layers()):
        kind = "Conv" if hasattr(l, 'filters') else "FC  "
        w    = l.filters if hasattr(l, 'filters') else l.weights
        print(f"    [{i}] {kind}  shape={w.shape}  params={w.size + l.biases.size:,}")

    # Forward
    dummy_X = np.random.randn(BATCH, 3, 100, 100).astype(np.float32)
    print(f"\n  Running forward pass (batch={BATCH}) ... ", end="", flush=True)
    probs = model.forward(dummy_X)
    print("done")
    assert probs.shape == (BATCH, NUM_CLASSES), f"Expected ({BATCH},{NUM_CLASSES}), got {probs.shape}"
    assert np.allclose(probs.sum(axis=1), 1.0, atol=1e-5), "Softmax rows don't sum to 1"
    print(f"  Output shape : {probs.shape}  ✓")
    print(f"  Prob sums    : {probs.sum(axis=1)}  ✓")

    # Backward
    print("  Running backward pass ... ", end="", flush=True)
    d_probs = probs.copy()
    fake_labels = np.random.randint(0, NUM_CLASSES, size=BATCH)
    d_probs[np.arange(BATCH), fake_labels] -= 1
    d_probs /= BATCH
    model.backward(d_probs)
    print("done")

    for i, l in enumerate(model.get_trainable_layers()):
        if hasattr(l, 'filters'):
            assert l.d_filters.shape == l.filters.shape, f"Layer {i} d_filters shape mismatch"
        else:
            assert l.d_weights.shape == l.weights.shape, f"Layer {i} d_weights shape mismatch"
    print("  Gradients    : all correct shapes  ✓")

    print("\n  ✓ All checks passed. person1_model.py is ready.")
    print("=" * 55)