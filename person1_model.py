"""
person1_model.py — Custom CNN architecture built from scratch using NumPy only.
PERSON 1's responsibility.

Architecture (Improved Baseline):
    Input (3x100x100)
        → Conv1 (8 filters, 3x3)  + BN + ReLU
        → MaxPool (2x2)
        → Conv2 (16 filters, 3x3) + BN + ReLU
        → MaxPool (2x2)
        → Conv3 (32 filters, 3x3) + BN + ReLU
        → MaxPool (2x2)
        → Flatten
        → FC1 (256 neurons)       + BN + ReLU + Dropout
        → FC2 (num_classes)       + Softmax

All operations are implemented manually — no PyTorch, no TensorFlow.
"""

import numpy as np


# ─── ACTIVATION FUNCTIONS ────────────────────────────────────────────────────

def relu(x):
    """ReLU activation: max(0, x)"""
    return np.maximum(0, x)

def relu_derivative(x):
    """Derivative of ReLU — 1 where x > 0, else 0."""
    return (x > 0).astype(float)

def softmax(x):
    """
    Numerically stable softmax over last axis.
    Subtracts max to prevent overflow.
    """
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / np.sum(e, axis=-1, keepdims=True)


# ─── BATCH NORMALIZATION LAYER ────────────────────────────────────────────────

class BatchNormLayer:
    """
    Batch Normalization layer implemented from scratch.
    
    Normalizes activations to have zero mean and unit variance,
    then applies learnable scale (gamma) and shift (beta).
    
    This is a standard component in modern CNNs that helps with:
    - Faster convergence
    - Better gradient flow
    - Reduced sensitivity to initialization
    """
    
    def __init__(self, num_features, momentum=0.9, eps=1e-5):
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        
        # Learnable parameters
        self.gamma = np.ones(num_features)
        self.beta = np.zeros(num_features)
        
        # Running statistics for inference
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
        # Cache for backward pass
        self.last_input = None
        self.last_normalized = None
        self.last_mean = None
        self.last_var = None
        
        # Gradients
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.beta)
        
        self.training = True
    
    def forward(self, x, training=True):
        """
        x shape: (batch, channels, H, W) for conv layers OR
                 (batch, features) for FC layers
        Returns normalized output with same shape.
        """
        self.training = training
        self.last_input = x
        
        # Flatten spatial dimensions for conv layers to treat each channel separately
        if len(x.shape) == 4:  # Conv layer output (N, C, H, W)
            N, C, H, W = x.shape
            x_flat = x.transpose(0, 2, 3, 1).reshape(-1, C)
            spatial_shape = (N, H, W)
        else:  # FC layer output (N, features)
            x_flat = x
            C = self.num_features
            spatial_shape = None
        
        if training:
            # Compute mean and variance
            mean = np.mean(x_flat, axis=0)
            var = np.var(x_flat, axis=0)
            
            # Update running statistics
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
            
            self.last_mean = mean
            self.last_var = var
        else:
            mean = self.running_mean
            var = self.running_var
        
        # Normalize
        x_normalized = (x_flat - mean) / np.sqrt(var + self.eps)
        self.last_normalized = x_normalized
        
        # Scale and shift
        out = self.gamma * x_normalized + self.beta
        
        # Reshape back if needed
        if spatial_shape is not None:
            N, H, W = spatial_shape
            out = out.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        return out
    
    def backward(self, d_out):
        """
        Backward pass for Batch Normalization.
        d_out shape matches input x shape.
        """
        if len(d_out.shape) == 4:
            N, C, H, W = d_out.shape
            d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, C)
            x_flat = self.last_input.transpose(0, 2, 3, 1).reshape(-1, C)
            N_total = N * H * W
        else:
            d_out_flat = d_out
            x_flat = self.last_input
            N_total = len(x_flat)
        
        # Gradients for gamma and beta
        self.d_gamma = np.sum(d_out_flat * self.last_normalized, axis=0)
        self.d_beta = np.sum(d_out_flat, axis=0)
        
        # Gradient through the normalized input
        d_normalized = d_out_flat * self.gamma
        
        # Gradient through mean and variance
        var_inv = 1.0 / np.sqrt(self.last_var + self.eps)
        
        d_var = np.sum(d_normalized * (x_flat - self.last_mean), axis=0) * -0.5 * var_inv**3
        d_mean = np.sum(d_normalized * -var_inv, axis=0) + d_var * np.mean(-2.0 * (x_flat - self.last_mean), axis=0)
        
        # Gradient through input
        d_x = d_normalized * var_inv + d_var * (2.0 * (x_flat - self.last_mean) / N_total) + d_mean / N_total
        
        # Reshape back if needed
        if len(d_out.shape) == 4:
            d_x = d_x.reshape(N, H, W, C).transpose(0, 3, 1, 2)
        
        return d_x


# ─── DROPOUT LAYER ───────────────────────────────────────────────────────────

class DropoutLayer:
    """
    Dropout layer for regularization.
    Randomly zeros out a fraction of activations during training.
    """
    
    def __init__(self, p=0.5):
        self.p = p  # Dropout probability
        self.mask = None
        self.training = True
    
    def forward(self, x, training=True):
        self.training = training
        
        if training and self.p > 0:
            self.mask = (np.random.rand(*x.shape) > self.p).astype(float)
            # Scale by 1/(1-p) to maintain expected sum
            return x * self.mask / (1 - self.p)
        return x
    
    def backward(self, d_out):
        if self.training and self.p > 0:
            return d_out * self.mask / (1 - self.p)
        return d_out


# ─── CONVOLUTION LAYER ────────────────────────────────────────────────────────

class ConvLayer:
    """
    2D Convolutional layer with 'same' padding.
    
    Attributes:
        filters  : (num_filters, in_channels, kH, kW)
        biases   : (num_filters,)
    """

    def __init__(self, in_channels, num_filters, kernel_size=3, stride=1, padding='same'):
        self.num_filters = num_filters
        self.in_channels = in_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding if padding != 'same' else kernel_size // 2
        
        # He initialization — good default for ReLU networks
        # Improved: use 2.0 factor and consider kernel size properly
        fan_in = in_channels * kernel_size * kernel_size
        scale = np.sqrt(2.0 / fan_in)
        self.filters = np.random.randn(
            num_filters, in_channels, kernel_size, kernel_size
        ) * scale
        self.biases = np.zeros(num_filters)
        
        # For BatchNorm integration: store output channel dimension
        self.out_channels = num_filters
        
        # Cached values needed for backprop
        self.last_input = None
        self.last_output = None
        
        # Gradients
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)
        
        # Padding amount
        self.pad_amount = self.padding if self.padding != 'same' else kernel_size // 2

    def _pad(self, x):
        """Apply zero-padding to spatial dimensions."""
        if self.pad_amount == 0:
            return x
        return np.pad(x, ((0, 0), (0, 0), 
                          (self.pad_amount, self.pad_amount),
                          (self.pad_amount, self.pad_amount)), 
                      mode='constant')

    def forward(self, x):
        """
        x shape: (batch, in_channels, H, W)
        Returns: (batch, num_filters, out_H, out_W)
        """
        self.last_input = x
        x_pad = self._pad(x)

        batch, _, H, W = x_pad.shape
        kH = kW = self.kernel_size
        out_H = (H - kH) // self.stride + 1
        out_W = (W - kW) // self.stride + 1
        
        # Pre-allocate output for better performance
        out = np.zeros((batch, self.num_filters, out_H, out_W))

        for f in range(self.num_filters):
            for i in range(out_H):
                for j in range(out_W):
                    si = i * self.stride
                    sj = j * self.stride
                    patch = x_pad[:, :, si:si+kH, sj:sj+kW]
                    out[:, f, i, j] = (
                        np.sum(patch * self.filters[f], axis=(1, 2, 3))
                        + self.biases[f]
                    )

        self.last_output = out
        return out

    def backward(self, d_out):
        """
        d_out shape: (batch, num_filters, out_H, out_W)
        Returns d_input: (batch, in_channels, H, W)
        """
        x = self.last_input
        x_pad = self._pad(x)

        batch, _, H_pad, W_pad = x_pad.shape
        kH = kW = self.kernel_size
        _, _, out_H, out_W = d_out.shape

        d_x_pad = np.zeros_like(x_pad)
        self.d_filters = np.zeros_like(self.filters)
        self.d_biases = np.zeros_like(self.biases)

        for f in range(self.num_filters):
            self.d_biases[f] = np.sum(d_out[:, f, :, :])
            for i in range(out_H):
                for j in range(out_W):
                    si = i * self.stride
                    sj = j * self.stride
                    patch = x_pad[:, :, si:si+kH, sj:sj+kW]
                    
                    # Gradient w.r.t. filter weights
                    self.d_filters[f] += np.sum(
                        patch * d_out[:, f, i, j][:, None, None, None],
                        axis=0
                    )
                    
                    # Gradient w.r.t. input
                    d_x_pad[:, :, si:si+kH, sj:sj+kW] += (
                        self.filters[f] * d_out[:, f, i, j][:, None, None, None]
                    )

        # Remove padding from gradient
        if self.pad_amount > 0:
            p = self.pad_amount
            d_x = d_x_pad[:, :, p:-p, p:-p]
        else:
            d_x = d_x_pad

        return d_x


# ─── MAX POOLING LAYER ────────────────────────────────────────────────────────

class MaxPoolLayer:
    """2x2 Max Pooling with stride 2."""

    def __init__(self, pool_size=2):
        self.pool_size = pool_size
        self.last_input = None
        self.last_mask = None

    def forward(self, x):
        """x: (batch, channels, H, W) → (batch, channels, H//2, W//2)"""
        self.last_input = x
        batch, C, H, W = x.shape
        p = self.pool_size
        out_H, out_W = H // p, W // p

        out = np.zeros((batch, C, out_H, out_W))
        mask = np.zeros_like(x, dtype=bool)

        for i in range(out_H):
            for j in range(out_W):
                patch = x[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                max_vals = np.max(patch, axis=(2, 3), keepdims=True)
                out[:, :, i, j] = max_vals[:, :, 0, 0]
                mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p] = (patch == max_vals)

        self.last_mask = mask
        return out

    def backward(self, d_out):
        """Route gradient only through the max positions."""
        batch, C, H, W = self.last_input.shape
        p = self.pool_size
        out_H, out_W = H // p, W // p

        d_x = np.zeros_like(self.last_input)
        for i in range(out_H):
            for j in range(out_W):
                d_x[:, :, i*p:(i+1)*p, j*p:(j+1)*p] += (
                    self.last_mask[:, :, i*p:(i+1)*p, j*p:(j+1)*p]
                    * d_out[:, :, i, j][:, :, None, None]
                )
        return d_x


# ─── FULLY CONNECTED LAYER ────────────────────────────────────────────────────

class FCLayer:
    """Standard fully-connected (dense) layer."""

    def __init__(self, in_features, out_features):
        scale = np.sqrt(2.0 / in_features)
        self.weights = np.random.randn(in_features, out_features) * scale
        self.biases = np.zeros(out_features)
        
        self.out_features = out_features  # For BatchNorm compatibility

        self.last_input = None
        self.d_weights = np.zeros_like(self.weights)
        self.d_biases = np.zeros_like(self.biases)

    def forward(self, x):
        """x: (batch, in_features) → (batch, out_features)"""
        self.last_input = x
        return x @ self.weights + self.biases

    def backward(self, d_out):
        """d_out: (batch, out_features) → d_input: (batch, in_features)"""
        self.d_weights = self.last_input.T @ d_out
        self.d_biases = np.sum(d_out, axis=0)
        return d_out @ self.weights.T


# ─── FLATTEN LAYER ────────────────────────────────────────────────────────────

class FlattenLayer:
    """Flattens (batch, C, H, W) → (batch, C*H*W)."""

    def __init__(self):
        self.last_shape = None

    def forward(self, x):
        self.last_shape = x.shape
        return x.reshape(x.shape[0], -1)

    def backward(self, d_out):
        return d_out.reshape(self.last_shape)


# ─── FULL CNN MODEL ───────────────────────────────────────────────────────────

class SimpleCNN:
    """
    3-conv-layer CNN built entirely from scratch with NumPy.
    
    Improved Architecture (Modern Baseline):
        Conv1(3→8)   + BN + ReLU + MaxPool
        Conv2(8→16)  + BN + ReLU + MaxPool
        Conv3(16→32) + BN + ReLU + MaxPool
        Flatten
        FC1(32*12*12→256) + BN + ReLU + Dropout
        FC2(256→num_classes) + Softmax
    """

    def __init__(self, num_classes=131, input_size=100, dropout_p=0.5):
        # Calculate feature map size after 3 MaxPool layers
        # Input: 100 → after 1st pool: 50 → after 2nd: 25 → after 3rd: 12
        fc_input_size = 32 * 12 * 12  # 32 channels * 12 * 12 = 4608
        
        # Build layers in order
        self.layers = []
        
        # Convolutional blocks with BatchNorm
        # Block 1
        self.layers.append(ConvLayer(in_channels=3, num_filters=8, kernel_size=3))
        self.layers.append(BatchNormLayer(8))
        # ReLU will be applied separately
        self.layers.append(MaxPoolLayer(pool_size=2))
        
        # Block 2
        self.layers.append(ConvLayer(in_channels=8, num_filters=16, kernel_size=3))
        self.layers.append(BatchNormLayer(16))
        self.layers.append(MaxPoolLayer(pool_size=2))
        
        # Block 3
        self.layers.append(ConvLayer(in_channels=16, num_filters=32, kernel_size=3))
        self.layers.append(BatchNormLayer(32))
        self.layers.append(MaxPoolLayer(pool_size=2))
        
        # Flatten and FC layers
        self.layers.append(FlattenLayer())
        self.layers.append(FCLayer(fc_input_size, 256))
        self.layers.append(BatchNormLayer(256))
        self.layers.append(DropoutLayer(dropout_p))
        self.layers.append(FCLayer(256, num_classes))
        
        # Track which layers have ReLU after them
        self._relu_after = {1, 4, 7, 11}  # Indices of BatchNorm layers
        self._relu_cache = {}
        
        # Track layer types for optimizer
        self._bn_layers = [i for i, l in enumerate(self.layers) 
                          if isinstance(l, BatchNormLayer)]
        self._dropout_layers = [i for i, l in enumerate(self.layers) 
                               if isinstance(l, DropoutLayer)]
        
        # Store input size for inference time measurement
        self.input_size = input_size

    def forward(self, x, training=True):
        """
        Full forward pass.
        x: (batch, 3, 100, 100) — pixel values normalised to [0, 1]
        training: whether to use training mode (enables dropout and BN statistics)
        Returns: (batch, num_classes) probabilities
        """
        out = x
        for i, layer in enumerate(self.layers):
            if isinstance(layer, (BatchNormLayer, DropoutLayer)):
                out = layer.forward(out, training=training)
            else:
                out = layer.forward(out)
            
            if i in self._relu_after:
                self._relu_cache[i] = out.copy()
                out = relu(out)
        
        # Final softmax
        return softmax(out)

    def backward(self, d_out):
        """
        Full backward pass. Propagates gradient through all layers.
        d_out: gradient of loss w.r.t. softmax output
        """
        grad = d_out
        for i in reversed(range(len(self.layers))):
            if i in self._relu_after:
                grad = grad * relu_derivative(self._relu_cache[i])
            grad = self.layers[i].backward(grad)
        return grad

    def get_trainable_layers(self):
        """Returns only layers that have trainable parameters."""
        trainable = []
        for layer in self.layers:
            if isinstance(layer, (ConvLayer, FCLayer, BatchNormLayer)):
                trainable.append(layer)
        return trainable

    def get_param_count(self):
        """Total number of trainable parameters."""
        total = 0
        for layer in self.get_trainable_layers():
            if isinstance(layer, ConvLayer):
                total += layer.filters.size + layer.biases.size
            elif isinstance(layer, FCLayer):
                total += layer.weights.size + layer.biases.size
            elif isinstance(layer, BatchNormLayer):
                total += layer.gamma.size + layer.beta.size
        return total
    
    def eval(self):
        """Set model to evaluation mode (disables dropout, uses running stats for BN)."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = False
    
    def train(self):
        """Set model to training mode."""
        for layer in self.layers:
            if hasattr(layer, 'training'):
                layer.training = True
