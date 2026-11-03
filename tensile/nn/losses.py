"""
Loss functions with numerically stable forward and backward passes.
"""

from __future__ import annotations
from typing import Union
import numpy as np
from tensile.tensor import Tensor


def mse_loss(pred: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes the Mean Squared Error (MSE) between predictions and targets:
    L = mean((pred - target)^2)
    """
    target_t = target if isinstance(target, Tensor) else Tensor(target, requires_grad=False)
    diff = pred - target_t
    return (diff ** 2).mean()


def cross_entropy(logits: Tensor, target: Union[Tensor, np.ndarray]) -> Tensor:
    """
    Computes numerically stable log-softmax cross-entropy loss directly from raw logits.
    Supports both 1D integer class indices of shape `(B,)` and 2D one-hot / probability
    targets of shape `(B, C)`.
    """
    target_arr = target.data if isinstance(target, Tensor) else np.asarray(target)

    is_1d = logits.ndim == 1
    x = logits.data[np.newaxis, :] if is_1d else logits.data
    batch_size = x.shape[0]

    max_logits = np.max(x, axis=-1, keepdims=True)
    shifted = x - max_logits
    exp_shifted = np.exp(shifted)
    sum_exp = np.sum(exp_shifted, axis=-1, keepdims=True)
    log_probs = shifted - np.log(sum_exp)
    probs = exp_shifted / sum_exp

    if target_arr.ndim == x.ndim - 1:
        class_indices = target_arr.astype(np.int64).reshape(-1)
        loss_val = -np.mean(log_probs[np.arange(batch_size), class_indices])
        is_index_target = True
    elif target_arr.shape == x.shape or (is_1d and target_arr.shape == logits.shape):
        t_2d = target_arr[np.newaxis, :] if is_1d else target_arr
        loss_val = -np.sum(t_2d * log_probs) / batch_size
        is_index_target = False
    else:
        raise ValueError(
            f"Incompatible target shape {target_arr.shape} for logits shape {logits.shape}."
        )

    out = Tensor(
        loss_val,
        requires_grad=logits.requires_grad,
        _prev=(logits,) if logits.requires_grad else (),
        _op="cross_entropy",
    )

    def _backward() -> None:
        if out.grad is None or not logits.requires_grad:
            return

        grad_x = probs.copy()
        if is_index_target:
            grad_x[np.arange(batch_size), class_indices] -= 1.0
        else:
            grad_x -= t_2d

        grad_x /= batch_size
        if is_1d:
            grad_x = grad_x.squeeze(0)

        logits.grad += out.grad * grad_x

    out._backward = _backward
    return out
