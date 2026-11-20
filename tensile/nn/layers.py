# tensile/nn/layers.py
"""
Neural network layers, activations, convolutional modules, and sequential containers for Tensile.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
from tensile.tensor import Tensor, unbroadcast
from tensile.nn.module import Module


class Linear(Module):
    """
    Affine linear transformation layer: y = x @ W + b.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        name_prefix: str = "",
    ) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        scale = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        prefix = f"{name_prefix}." if name_prefix else ""
        self.weight = Tensor(weight_data, requires_grad=True, name=f"{prefix}weight")

        if bias:
            self.bias: Optional[Tensor] = Tensor(
                np.zeros((out_features,), dtype=np.float64),
                requires_grad=True,
                name=f"{prefix}bias",
            )
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        out = x @ self.weight
        if self.bias is not None:
            out = out + self.bias
        return out

    def __repr__(self) -> str:
        return (
            f"Linear(in_features={self.in_features}, "
            f"out_features={self.out_features}, bias={self.bias is not None})"
        )


class ReLU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.relu()

    def __repr__(self) -> str:
        return "ReLU()"


class LeakyReLU(Module):
    def __init__(self, negative_slope: float = 0.01) -> None:
        super().__init__()
        self.negative_slope = float(negative_slope)

    def forward(self, x: Tensor) -> Tensor:
        mask = x.data > 0
        slope = self.negative_slope
        out_data = np.where(mask, x.data, slope * x.data)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _prev=(x,) if x.requires_grad else (),
            _op="LeakyReLU",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return
            grad = np.where(mask, out.grad, slope * out.grad)
            x.grad += grad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"LeakyReLU(negative_slope={self.negative_slope})"


class Sigmoid(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

    def __repr__(self) -> str:
        return "Sigmoid()"


class Tanh(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()

    def __repr__(self) -> str:
        return "Tanh()"


class GELU(Module):
    def forward(self, x: Tensor) -> Tensor:
        return x.gelu()

    def __repr__(self) -> str:
        return "GELU()"


class Dropout(Module):
    """
    Randomly zeroes some elements of the input tensor with probability `p`
    using samples from a Bernoulli distribution during training.
    """

    def __init__(self, p: float = 0.5) -> None:
        super().__init__()
        if p < 0.0 or p >= 1.0:
            raise ValueError(f"Dropout probability has to be between 0 and 1, got {p}")
        self.p = float(p)

    def forward(self, x: Tensor) -> Tensor:
        if not self.training or self.p == 0.0:
            return x

        keep_p = 1.0 - self.p
        mask = (np.random.rand(*x.shape) < keep_p).astype(np.float64) / keep_p
        out_data = x.data * mask

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _prev=(x,) if x.requires_grad else (),
            _op="Dropout",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return
            x.grad += out.grad * mask

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return f"Dropout(p={self.p})"


class LayerNorm(Module):
    """
    Applies Layer Normalization over a mini-batch of inputs:
    y = ((x - E[x]) / sqrt(Var[x] + eps)) * gamma + beta
    """

    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int]],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
    ) -> None:
        super().__init__()
        if isinstance(normalized_shape, int):
            self.normalized_shape = (normalized_shape,)
        else:
            self.normalized_shape = tuple(normalized_shape)

        self.eps = float(eps)
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.weight = Tensor(
                np.ones(self.normalized_shape, dtype=np.float64),
                requires_grad=True,
                name="weight",
            )
            self.bias = Tensor(
                np.zeros(self.normalized_shape, dtype=np.float64),
                requires_grad=True,
                name="bias",
            )
        else:
            self.weight = None
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        norm_dims = len(self.normalized_shape)
        axis = tuple(range(x.ndim - norm_dims, x.ndim))
        m = int(np.prod(self.normalized_shape))

        mean = np.mean(x.data, axis=axis, keepdims=True)
        var = np.var(x.data, axis=axis, keepdims=True)
        std_inv = 1.0 / np.sqrt(var + self.eps)
        x_hat = (x.data - mean) * std_inv

        out_data = x_hat
        if self.elementwise_affine and self.weight is not None:
            out_data = out_data * self.weight.data
            if self.bias is not None:
                out_data = out_data + self.bias.data

        parents = [x]
        if self.elementwise_affine:
            if self.weight is not None and self.weight.requires_grad:
                parents.append(self.weight)
            if self.bias is not None and self.bias.requires_grad:
                parents.append(self.bias)

        out = Tensor(
            out_data,
            requires_grad=any(p.requires_grad for p in parents),
            _prev=parents,
            _op="LayerNorm",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            if self.elementwise_affine and self.weight is not None:
                if self.bias is not None and self.bias.requires_grad:
                    db = unbroadcast(out.grad, self.bias.shape)
                    self.bias.grad += db
                if self.weight.requires_grad:
                    dw = unbroadcast(out.grad * x_hat, self.weight.shape)
                    self.weight.grad += dw
                dy = out.grad * self.weight.data
            else:
                dy = out.grad

            if x.requires_grad:
                dx_hat = dy
                sum_dx_hat = np.sum(dx_hat, axis=axis, keepdims=True)
                sum_dx_hat_x_hat = np.sum(dx_hat * x_hat, axis=axis, keepdims=True)
                dx = (1.0 / m) * std_inv * (m * dx_hat - sum_dx_hat - x_hat * sum_dx_hat_x_hat)
                x.grad += dx

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"LayerNorm({self.normalized_shape}, "
            f"eps={self.eps}, elementwise_affine={self.elementwise_affine})"
        )


class Flatten(Module):
    """
    Flattens contiguous dimensions into a tensor of shape (N, -1).
    """

    def __init__(self, start_dim: int = 1, end_dim: int = -1) -> None:
        super().__init__()
        self.start_dim = start_dim
        self.end_dim = end_dim

    def forward(self, x: Tensor) -> Tensor:
        start = self.start_dim if self.start_dim >= 0 else x.ndim + self.start_dim
        end = self.end_dim if self.end_dim >= 0 else x.ndim + self.end_dim
        if end < start:
            raise ValueError(f"Flatten end_dim ({end}) < start_dim ({start})")

        new_shape = (
            x.shape[:start]
            + (int(np.prod(x.shape[start : end + 1])),)
            + x.shape[end + 1 :]
        )
        return x.reshape(new_shape)

    def __repr__(self) -> str:
        return f"Flatten(start_dim={self.start_dim}, end_dim={self.end_dim})"


# =========================================================================
# Convolution & Pooling Modules
# =========================================================================

def _im2col(
    x: np.ndarray,
    kh: int,
    kw: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
) -> tuple[np.ndarray, int, int]:
    """
    Extracts sliding 2D spatial patches into a 2D matrix of shape
    (C_in * kh * kw, N * H_out * W_out).
    """
    n, c, h, w = x.shape
    sh, sw = stride
    ph, pw = padding

    if ph > 0 or pw > 0:
        x_padded = np.pad(x, ((0, 0), (0, 0), (ph, ph), (pw, pw)), mode="constant")
    else:
        x_padded = x

    h_pad, w_pad = x_padded.shape[2], x_padded.shape[3]
    h_out = (h_pad - kh) // sh + 1
    w_out = (w_pad - kw) // sw + 1

    shape = (n, c, kh, kw, h_out, w_out)
    strides = (
        x_padded.strides[0],
        x_padded.strides[1],
        x_padded.strides[2],
        x_padded.strides[3],
        x_padded.strides[2] * sh,
        x_padded.strides[3] * sw,
    )
    patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
    cols = patches.transpose(1, 2, 3, 0, 4, 5).reshape(c * kh * kw, n * h_out * w_out)
    return cols, h_out, w_out


def _col2im(
    cols: np.ndarray,
    x_shape: tuple[int, int, int, int],
    kh: int,
    kw: int,
    stride: tuple[int, int],
    padding: tuple[int, int],
    h_out: int,
    w_out: int,
) -> np.ndarray:
    """
    Accumulates gradients from 2D column format back into input image dimensions (N, C, H, W).
    """
    n, c, h, w = x_shape
    sh, sw = stride
    ph, pw = padding
    h_pad = h + 2 * ph
    w_pad = w + 2 * pw

    cols_reshaped = cols.reshape(c, kh, kw, n, h_out, w_out).transpose(3, 0, 1, 2, 4, 5)
    dx_pad = np.zeros((n, c, h_pad, w_pad), dtype=np.float64)

    for i in range(kh):
        for j in range(kw):
            dx_pad[:, :, i : i + h_out * sh : sh, j : j + w_out * sw : sw] += (
                cols_reshaped[:, :, i, j, :, :]
            )

    if ph > 0 or pw > 0:
        h_slice = slice(ph, h_pad - ph) if ph > 0 else slice(None)
        w_slice = slice(pw, w_pad - pw) if pw > 0 else slice(None)
        return dx_pad[:, :, h_slice, w_slice]
    return dx_pad


class Conv2d(Module):
    """
    Applies a 2D convolution over an input signal composed of several input planes.
    Input shape: (N, C_in, H_in, W_in)
    Output shape: (N, C_out, H_out, W_out)
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, tuple[int, int]],
        stride: Union[int, tuple[int, int]] = 1,
        padding: Union[int, tuple[int, int]] = 0,
        bias: bool = True,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

        kh, kw = self.kernel_size
        k = 1.0 / np.sqrt(in_channels * kh * kw)
        weight_data = np.random.uniform(-k, k, size=(out_channels, in_channels, kh, kw))
        self.weight = Tensor(weight_data, requires_grad=True, name="weight")

        if bias:
            bias_data = np.random.uniform(-k, k, size=(out_channels,))
            self.bias: Optional[Tensor] = Tensor(bias_data, requires_grad=True, name="bias")
        else:
            self.bias = None

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got shape {x.shape}")
        if x.shape[1] != self.in_channels:
            raise ValueError(
                f"Expected {self.in_channels} input channels, but got {x.shape[1]}"
            )

        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        cols, h_out, w_out = _im2col(x.data, kh, kw, self.stride, self.padding)

        w_row = self.weight.data.reshape(self.out_channels, -1)
        out_row = w_row @ cols  # (C_out, N * H_out * W_out)

        if self.bias is not None:
            out_row = out_row + self.bias.data.reshape(-1, 1)

        out_data = out_row.reshape(self.out_channels, n, h_out, w_out).transpose(1, 0, 2, 3)

        parents = [x, self.weight]
        if self.bias is not None:
            parents.append(self.bias)

        out = Tensor(
            out_data,
            requires_grad=any(p.requires_grad for p in parents),
            _prev=parents,
            _op="Conv2d",
        )

        def _backward() -> None:
            if out.grad is None:
                return

            # dOut in row format: (C_out, N * H_out * W_out)
            dout_row = out.grad.transpose(1, 0, 2, 3).reshape(self.out_channels, -1)

            if self.bias is not None and self.bias.requires_grad:
                self.bias.grad += np.sum(out.grad, axis=(0, 2, 3))

            if self.weight.requires_grad:
                dw_row = dout_row @ cols.T
                self.weight.grad += dw_row.reshape(self.weight.shape)

            if x.requires_grad:
                dcols = w_row.T @ dout_row
                dx = _col2im(
                    dcols,
                    x.shape,
                    kh,
                    kw,
                    self.stride,
                    self.padding,
                    h_out,
                    w_out,
                )
                x.grad += dx

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"Conv2d({self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}, "
            f"padding={self.padding}, bias={self.bias is not None})"
        )


class MaxPool2d(Module):
    """
    Applies a 2D max pooling over an input signal composed of several input planes.
    Input shape: (N, C, H, W)
    """

    def __init__(
        self,
        kernel_size: Union[int, tuple[int, int]],
        stride: Optional[Union[int, tuple[int, int]]] = None,
        padding: Union[int, tuple[int, int]] = 0,
    ) -> None:
        super().__init__()
        self.kernel_size = (
            (kernel_size, kernel_size) if isinstance(kernel_size, int) else tuple(kernel_size)
        )
        if stride is None:
            self.stride = self.kernel_size
        else:
            self.stride = (stride, stride) if isinstance(stride, int) else tuple(stride)
        self.padding = (padding, padding) if isinstance(padding, int) else tuple(padding)

    def forward(self, x: Tensor) -> Tensor:
        if x.ndim != 4:
            raise ValueError(f"Expected 4D input (N, C, H, W), got shape {x.shape}")

        n, c, h, w = x.shape
        kh, kw = self.kernel_size
        sh, sw = self.stride
        ph, pw = self.padding

        if ph > 0 or pw > 0:
            x_padded = np.pad(
                x.data,
                ((0, 0), (0, 0), (ph, ph), (pw, pw)),
                mode="constant",
                constant_values=-np.inf,
            )
        else:
            x_padded = x.data

        h_pad, w_pad = x_padded.shape[2], x_padded.shape[3]
        h_out = (h_pad - kh) // sh + 1
        w_out = (w_pad - kw) // sw + 1

        shape = (n, c, h_out, w_out, kh, kw)
        strides = (
            x_padded.strides[0],
            x_padded.strides[1],
            x_padded.strides[2] * sh,
            x_padded.strides[3] * sw,
            x_padded.strides[2],
            x_padded.strides[3],
        )
        patches = np.lib.stride_tricks.as_strided(x_padded, shape=shape, strides=strides)
        patches_flat = patches.reshape(n, c, h_out, w_out, kh * kw)

        argmax = patches_flat.argmax(axis=-1)
        out_data = patches_flat.max(axis=-1)

        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _prev=(x,) if x.requires_grad else (),
            _op="MaxPool2d",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return

            dx_pad = np.zeros_like(x_padded, dtype=np.float64)
            for idx in range(kh * kw):
                i = idx // kw
                j = idx % kw
                mask = (argmax == idx).astype(np.float64)
                dx_pad[:, :, i : i + h_out * sh : sh, j : j + w_out * sw : sw] += (
                    mask * out.grad
                )

            if ph > 0 or pw > 0:
                h_slice = slice(ph, h_pad - ph) if ph > 0 else slice(None)
                w_slice = slice(pw, w_pad - pw) if pw > 0 else slice(None)
                x.grad += dx_pad[:, :, h_slice, w_slice]
            else:
                x.grad += dx_pad

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return (
            f"MaxPool2d(kernel_size={self.kernel_size}, "
            f"stride={self.stride}, padding={self.padding})"
        )


class Softmax(Module):
    def __init__(self, dim: int = -1) -> None:
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        shifted = x.data - np.max(x.data, axis=self.dim, keepdims=True)
        exps = np.exp(shifted)
        probs = exps / np.sum(exps, axis=self.dim, keepdims=True)

        out = Tensor(
            probs,
            requires_grad=x.requires_grad,
            _prev=(x,) if x.requires_grad else (),
            _op="Softmax",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return
            dot = np.sum(out.grad * probs, axis=self.dim, keepdims=True)
            x.grad += probs * (out.grad - dot)

        out._backward = _backward
        return out


class Sequential(Module):
    def __init__(self, *layers: Union[Module, Sequence[Module]]) -> None:
        super().__init__()
        if len(layers) == 1 and isinstance(layers[0], (list, tuple)):
            self.layers: list[Module] = list(layers[0])
        else:
            self.layers = [layer for layer in layers if isinstance(layer, Module)]

    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x

    def named_modules(
        self, prefix: str = "", memo: Optional[set[int]] = None
    ):
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self
        for idx, layer in enumerate(self.layers):
            sub_prefix = f"{prefix}.{idx}" if prefix else str(idx)
            yield from layer.named_modules(sub_prefix, memo)

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        layer_lines = "\n".join(f"  ({i}): {layer}" for i, layer in enumerate(self.layers))
        return f"Sequential(\n{layer_lines}\n)"
