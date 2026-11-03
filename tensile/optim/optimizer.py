"""
Parameter optimization algorithms (SGD with Momentum and Adam).
"""

from __future__ import annotations
from typing import Iterable
import numpy as np
from tensile.tensor import Tensor


class Optimizer:
    """Base class for all optimizers."""

    def __init__(self, params: Iterable[Tensor], lr: float = 1e-3) -> None:
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        self.params: list[Tensor] = list(params)
        self.lr: float = float(lr)

    def zero_grad(self) -> None:
        """Clears the gradients of all optimized Tensor parameters."""
        for p in self.params:
            p.zero_grad()

    def step(self) -> None:
        """Performs a single optimization step."""
        raise NotImplementedError("Optimizer subclasses must implement step().")


class SGD(Optimizer):
    """
    Stochastic Gradient Descent optimizer with momentum.
    v_t = momentum * v_{t-1} + g_t
    p_t = p_{t-1} - lr * v_t
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.9,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum: float = float(momentum)
        self.velocities: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue
            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + p.grad
                update = self.velocities[i]
            else:
                update = p.grad
            p.data -= self.lr * update


class Adam(Optimizer):
    """
    Adaptive Moment Estimation (Adam) optimizer with bias-corrected first and second
    moment estimates.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps: float = float(eps)
        self.t: int = 0

        self.m: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]
        self.v: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]

    def step(self) -> None:
        self.t += 1
        bias_correction1 = 1.0 - (self.beta1 ** self.t)
        bias_correction2 = 1.0 - (self.beta2 ** self.t)

        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
