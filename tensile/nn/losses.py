# tensile/nn/losses.py
"""
Numerically stable loss functions and Modules for Tensile.
"""

from __future__ import annotations
from typing import Union
import numpy as np
from tensile.tensor import Tensor
from tensile.nn.module import Module


def mse_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """Computes Mean Squared Error: mean((pred - target)^2)."""
    target_t = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - target_t
    return (diff ** 2).mean()


def cross_entropy(logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Numerically stable categorical cross-entropy loss using direct log-sum-exp:
        L = (1 / N) * sum_i ( log(sum_j exp(z_{i,j} - m_i)) + m_i - z_{i, y_i} )
    Backward pass computes:
        dL/dZ = (softmax(Z) - Y) / N
    Supports both 1D integer class indices `(N,)` and 2D one-hot/probability targets `(N, C)`.
    """
    target_arr = target.data if isinstance(target, Tensor) else np.asarray(target)

    is_1d = logits.ndim == 1
    z = logits.data[np.newaxis, :] if is_1d else logits.data
    n_samples, num_classes = z.shape

    # Log-Sum-Exp trick for numerical stability
    m = np.max(z, axis=-1, keepdims=True)
    shifted = z - m
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_sum_exp = m + np.log(sum_exp)  # shape: (N, 1)

    softmax_probs = exp_shifted / sum_exp  # shape: (N, C)

    # Construct one-hot target matrix Y of shape (N, C)
    if target_arr.ndim == z.ndim - 1:
        class_indices = target_arr.astype(np.int64).reshape(-1)
        y_one_hot = np.zeros_like(z, dtype=np.float64)
        y_one_hot[np.arange(n_samples), class_indices] = 1.0
        z_target = z[np.arange(n_samples), class_indices]
        loss_val = np.mean(log_sum_exp.squeeze(-1) - z_target)
    elif target_arr.shape == z.shape or (is_1d and target_arr.shape == logits.shape):
        y_one_hot = (target_arr[np.newaxis, :] if is_1d else target_arr).astype(np.float64)
        loss_val = np.mean(
            np.sum(y_one_hot * log_sum_exp, axis=-1) - np.sum(y_one_hot * z, axis=-1)
        )
    else:
        raise ValueError(
            f"Incompatible target shape {target_arr.shape} for logits shape {logits.shape}."
        )

    out = Tensor(
        loss_val,
        requires_grad=logits.requires_grad,
        _prev=(logits,) if logits.requires_grad else (),
        _op="CrossEntropyLoss",
        name="Loss",
    )

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return
        grad_z = (softmax_probs - y_one_hot) / float(n_samples)
        if is_1d:
            grad_z = grad_z.squeeze(0)
        logits.grad += out.grad * grad_z

    out._backward = _backward
    return out


class CrossEntropyLoss(Module):
    """Module wrapper for numerically stable CrossEntropyLoss."""

    def forward(self, logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        return cross_entropy(logits, target)

    def __repr__(self) -> str:
        return "CrossEntropyLoss()"


class MSELoss(Module):
    """Module wrapper for Mean Squared Error Loss."""

    def forward(self, pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
        return mse_loss(pred, target)

    def __repr__(self) -> str:
        return "MSELoss()"
