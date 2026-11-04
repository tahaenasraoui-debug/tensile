# tensile/ops.py
"""
Functional operator interface for Tensile Tensors.
"""

from __future__ import annotations
from typing import Optional, Union
from tensile.tensor import ArrayLike, Tensor, unbroadcast


def add(a: Tensor, b: ArrayLike) -> Tensor:
    return a + b


def sub(a: Tensor, b: ArrayLike) -> Tensor:
    return a - b


def mul(a: Tensor, b: ArrayLike) -> Tensor:
    return a * b


def div(a: Tensor, b: ArrayLike) -> Tensor:
    return a / b


def matmul(a: Tensor, b: ArrayLike) -> Tensor:
    return a @ b


def relu(x: Tensor) -> Tensor:
    return x.relu()


def sigmoid(x: Tensor) -> Tensor:
    return x.sigmoid()


def tanh(x: Tensor) -> Tensor:
    return x.tanh()


def gelu(x: Tensor) -> Tensor:
    return x.gelu()


def sum(
    x: Tensor,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    return x.sum(axis=axis, keepdims=keepdims)


def mean(
    x: Tensor,
    axis: Optional[Union[int, tuple[int, ...]]] = None,
    keepdims: bool = False,
) -> Tensor:
    return x.mean(axis=axis, keepdims=keepdims)


__all__ = [
    "unbroadcast",
    "add",
    "sub",
    "mul",
    "div",
    "matmul",
    "relu",
    "sigmoid",
    "tanh",
    "gelu",
    "sum",
    "mean",
]
