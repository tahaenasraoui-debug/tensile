"""
Learning rate schedulers for Tensile optimizers.
"""

from __future__ import annotations

import math

from tensile.optim.optimizer import Optimizer


class LRScheduler:
    """Base class for all learning rate schedulers."""

    def __init__(self, optimizer: Optimizer, last_epoch: int = -1) -> None:
        self.optimizer = optimizer
        self.base_lr = float(optimizer.lr)
        self.last_epoch = last_epoch
        self.step()

    def get_lr(self) -> float:
        raise NotImplementedError

    def step(self) -> None:
        self.last_epoch += 1
        new_lr = self.get_lr()
        self.optimizer.lr = float(new_lr)


class StepLR(LRScheduler):
    """
    Decays the learning rate of each parameter group by gamma every step_size epochs.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        step_size: int,
        gamma: float = 0.1,
        last_epoch: int = -1,
    ) -> None:
        if step_size <= 0:
            raise ValueError(f"step_size must be positive, got {step_size}")
        self.step_size = step_size
        self.gamma = float(gamma)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        exponent = self.last_epoch // self.step_size
        return self.base_lr * (self.gamma ** exponent)


class CosineAnnealingLR(LRScheduler):
    """
    Set the learning rate using a cosine annealing schedule.
    """

    def __init__(
        self,
        optimizer: Optimizer,
        T_max: int,
        eta_min: float = 0.0,
        last_epoch: int = -1,
    ) -> None:
        if T_max <= 0:
            raise ValueError(f"T_max must be positive, got {T_max}")
        self.T_max = T_max
        self.eta_min = float(eta_min)
        super().__init__(optimizer, last_epoch)

    def get_lr(self) -> float:
        if self.last_epoch == 0:
            return self.base_lr
        return (
            self.eta_min
            + (self.base_lr - self.eta_min)
            * (1.0 + math.cos(math.pi * self.last_epoch / self.T_max))
            / 2.0
        )
