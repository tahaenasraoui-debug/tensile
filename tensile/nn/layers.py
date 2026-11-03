"""
Core neural network layers, activations, and containers for Tensile.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
from tensile.tensor import Tensor
from tensile.nn.module import Module


class Linear(Module):
    """
    Applies an affine linear transformation to incoming data: y = x @ W + b.
    Uses He / Kaiming normal initialization for weights and zero initialization for biases.
    """

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # He / Kaiming normal initialization
        scale = np.sqrt(2.0 / in_features)
        weight_data = np.random.randn(in_features, out_features) * scale
        self.weight = Tensor(weight_data, requires_grad=True)

        if bias:
            self.bias: Optional[Tensor] = Tensor(
                np.zeros((out_features,), dtype=np.float64), requires_grad=True
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
    """
    Rectified Linear Unit activation: ReLU(x) = max(0, x).
    """

    def forward(self, x: Tensor) -> Tensor:
        out_data = np.maximum(0.0, x.data)
        out = Tensor(
            out_data,
            requires_grad=x.requires_grad,
            _prev=(x,) if x.requires_grad else (),
            _op="ReLU",
        )

        def _backward() -> None:
            if out.grad is None or not x.requires_grad:
                return
            x.grad += out.grad * (x.data > 0.0)

        out._backward = _backward
        return out

    def __repr__(self) -> str:
        return "ReLU()"


class Softmax(Module):
    """
    Numerically stable Softmax activation along a specified dimension:
    Softmax(x_i) = exp(x_i - max(x)) / sum_j exp(x_j - max(x))
    """

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

    def __repr__(self) -> str:
        return f"Softmax(dim={self.dim})"


class Sequential(Module):
    """
    A sequential container that chains modules in the order they are passed.
    Accepts either variable positional arguments or a single list/tuple of modules.
    """

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

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)

    def __repr__(self) -> str:
        layer_strs = ",\n  ".join(repr(layer) for layer in self.layers)
        return f"Sequential(\n  {layer_strs}\n)"
