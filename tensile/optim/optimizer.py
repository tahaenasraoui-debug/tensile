"""
Parameter optimization algorithms (SGD with Momentum, Adam, AdamW, and RMSprop).
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
    Stochastic Gradient Descent optimizer with optional momentum and weight decay.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.momentum: float = float(momentum)
        self.weight_decay: float = float(weight_decay)
        self.velocities: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad.copy()
            if self.weight_decay != 0.0:
                g += self.weight_decay * p.data

            if self.momentum > 0.0:
                self.velocities[i] = self.momentum * self.velocities[i] + g
                update = self.velocities[i]
            else:
                update = g

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
        weight_decay: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps: float = float(eps)
        self.weight_decay: float = float(weight_decay)
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

            g = p.grad.copy()
            if self.weight_decay != 0.0:
                g += self.weight_decay * p.data

            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class AdamW(Optimizer):
    """
    Adam with decoupled weight decay (Loshchilov & Hutter, 2017).
    Standard in modern Transformers and deep ConvNets.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-3,
        betas: tuple[float, float] = (0.9, 0.999),
        eps: float = 1e-8,
        weight_decay: float = 1e-2,
    ) -> None:
        super().__init__(params, lr=lr)
        self.beta1, self.beta2 = float(betas[0]), float(betas[1])
        self.eps: float = float(eps)
        self.weight_decay: float = float(weight_decay)
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

            # Decoupled weight decay
            if self.weight_decay != 0.0:
                p.data -= self.lr * self.weight_decay * p.data

            g = p.grad
            self.m[i] = self.beta1 * self.m[i] + (1.0 - self.beta1) * g
            self.v[i] = self.beta2 * self.v[i] + (1.0 - self.beta2) * (g ** 2)

            m_hat = self.m[i] / bias_correction1
            v_hat = self.v[i] / bias_correction2

            p.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


class RMSprop(Optimizer):
    """
    RMSprop optimizer maintaining moving average of squared gradients.
    """

    def __init__(
        self,
        params: Iterable[Tensor],
        lr: float = 1e-2,
        alpha: float = 0.99,
        eps: float = 1e-8,
        weight_decay: float = 0.0,
        momentum: float = 0.0,
    ) -> None:
        super().__init__(params, lr=lr)
        self.alpha: float = float(alpha)
        self.eps: float = float(eps)
        self.weight_decay: float = float(weight_decay)
        self.momentum: float = float(momentum)

        self.square_avg: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]
        self.momentum_buffer: list[np.ndarray] = [
            np.zeros_like(p.data, dtype=np.float64) for p in self.params
        ]

    def step(self) -> None:
        for i, p in enumerate(self.params):
            if not p.requires_grad or p.grad is None:
                continue

            g = p.grad.copy()
            if self.weight_decay != 0.0:
                g += self.weight_decay * p.data

            self.square_avg[i] = self.alpha * self.square_avg[i] + (1.0 - self.alpha) * (g ** 2)
            avg = np.sqrt(self.square_avg[i]) + self.eps

            if self.momentum > 0.0:
                self.momentum_buffer[i] = self.momentum * self.momentum_buffer[i] + g / avg
                p.data -= self.lr * self.momentum_buffer[i]
            else:
                p.data -= self.lr * (g / avg)
