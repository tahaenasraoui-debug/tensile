# tensile/nn/layers.py
"""
Neural network layers, activations, and sequential containers for Tensile.
"""

from __future__ import annotations
from typing import Optional, Sequence, Union
import numpy as np
from tensile.tensor import Tensor
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
        self.weight = Tensor(weight_data, requires_grad=True, name=f"{prefix}W")

        if bias:
            self.bias: Optional[Tensor] = Tensor(
                np.zeros((out_features,), dtype=np.float64),
                requires_grad=True,
                name=f"{prefix}b",
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

    def __getitem__(self, idx: int) -> Module:
        return self.layers[idx]

    def __len__(self) -> int:
        return len(self.layers)
