"""
Base Module abstraction for neural network components in Tensile.
"""

from __future__ import annotations
from typing import Any
from tensile.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    Automatically discovers trainable `Tensor` attributes and sub-modules.
    """

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Defines the computation performed at every call."""
        raise NotImplementedError("Subclasses of Module must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def parameters(self) -> list[Tensor]:
        """
        Recursively collects all unique trainable Tensor parameters (`requires_grad=True`)
        belonging to this module and its child modules.
        """
        params: list[Tensor] = []
        visited: set[int] = set()

        def _collect(obj: Any) -> None:
            if isinstance(obj, Tensor):
                if obj.requires_grad and id(obj) not in visited:
                    visited.add(id(obj))
                    params.append(obj)
            elif isinstance(obj, Module):
                for val in obj.__dict__.values():
                    _collect(val)
            elif isinstance(obj, (list, tuple, set)):
                for item in obj:
                    _collect(item)
            elif isinstance(obj, dict):
                for item in obj.values():
                    _collect(item)

        _collect(self)
        return params

    def zero_grad(self) -> None:
        """Resets gradients of all parameters in the module to zero."""
        for param in self.parameters():
            param.zero_grad()
