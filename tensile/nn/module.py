"""
Base Module abstraction for neural network components in Tensile.
Supports training/eval modes, recursive parameter collection, and state_dict serialization.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import numpy as np

from tensile.tensor import Tensor


class Module:
    """
    Base class for all neural network modules.
    Automatically discovers trainable `Tensor` attributes, registered buffers, and sub-modules.
    """

    def __init__(self) -> None:
        self.training: bool = True
        self._buffers: dict[str, np.ndarray | Tensor | None] = {}

    def __setattr__(self, name: str, value: Any) -> None:
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            if isinstance(value, Tensor) and value.requires_grad:
                raise ValueError(f"Buffer '{name}' cannot have requires_grad=True.")
            if value is not None and not isinstance(value, (np.ndarray, Tensor)):
                value = np.asarray(value)
            buffers[name] = value
        object.__setattr__(self, name, value)

    def __delattr__(self, name: str) -> None:
        buffers = self.__dict__.get("_buffers")
        if buffers is not None and name in buffers:
            del buffers[name]
        super().__delattr__(name)

    def register_buffer(self, name: str, value: np.ndarray | Tensor | None) -> None:
        """
        Registers a non-trainable buffer (e.g., BatchNorm running statistics) on the module.
        Buffers are excluded from `parameters()` but included in `state_dict()` and `load_state_dict()`.
        """
        if not isinstance(name, str) or not name or "." in name:
            raise ValueError(f"Buffer name must be a non-empty string without '.', got {name!r}.")
        existing = self.__dict__.get(name)
        if isinstance(existing, Tensor) and existing.requires_grad:
            raise ValueError(f"Attribute '{name}' is already a trainable parameter.")
        if isinstance(value, Tensor) and value.requires_grad:
            raise ValueError(f"Buffer '{name}' cannot have requires_grad=True.")
        if value is not None and not isinstance(value, (np.ndarray, Tensor)):
            value = np.asarray(value)
        if "_buffers" not in self.__dict__:
            object.__setattr__(self, "_buffers", {})
        self.__dict__["_buffers"][name] = value
        object.__setattr__(self, name, value)

    def forward(self, *args: Any, **kwargs: Any) -> Tensor:
        """Defines the computation performed at every call."""
        raise NotImplementedError("Subclasses of Module must implement forward().")

    def __call__(self, *args: Any, **kwargs: Any) -> Tensor:
        return self.forward(*args, **kwargs)

    def train(self, mode: bool = True) -> Module:
        """Sets the module and all submodules in training mode."""
        self.training = bool(mode)
        for _, module in self.named_modules():
            if module is not self:
                module.training = bool(mode)
        return self

    def eval(self) -> Module:
        """Sets the module and all submodules in evaluation mode (inference)."""
        return self.train(False)

    def named_modules(
        self, prefix: str = "", memo: set[int] | None = None
    ) -> Iterator[tuple[str, Module]]:
        """Yields an iterator over all modules in the network, yielding both name and module."""
        if memo is None:
            memo = set()
        if id(self) in memo:
            return
        memo.add(id(self))
        yield prefix, self

        for name, value in self.__dict__.items():
            if isinstance(value, Module):
                sub_prefix = f"{prefix}.{name}" if prefix else name
                yield from value.named_modules(sub_prefix, memo)
            elif isinstance(value, (list, tuple)):
                for idx, item in enumerate(value):
                    if isinstance(item, Module):
                        sub_prefix = f"{prefix}.{name}.{idx}" if prefix else f"{name}.{idx}"
                        yield from item.named_modules(sub_prefix, memo)
            elif isinstance(value, dict):
                for key, item in value.items():
                    if isinstance(item, Module):
                        sub_prefix = f"{prefix}.{name}.{key}" if prefix else f"{name}.{key}"
                        yield from item.named_modules(sub_prefix, memo)

    def named_parameters(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, Tensor]]:
        """Yields an iterator over module parameters, yielding both the name and parameter Tensor."""
        memo: set[int] = set()
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for mod_prefix, mod in modules:
            buffers = mod.__dict__.get("_buffers", {})
            for name, value in mod.__dict__.items():
                if (
                    name not in buffers
                    and isinstance(value, Tensor)
                    and value.requires_grad
                    and id(value) not in memo
                ):
                    memo.add(id(value))
                    p_name = f"{mod_prefix}.{name}" if mod_prefix else name
                    yield p_name, value

    def parameters(self) -> list[Tensor]:
        """
        Recursively collects all unique trainable Tensor parameters (`requires_grad=True`)
        belonging to this module and its child modules.
        """
        return [p for _, p in self.named_parameters()]

    def named_buffers(
        self, prefix: str = "", recurse: bool = True
    ) -> Iterator[tuple[str, np.ndarray | Tensor]]:
        """Yields an iterator over module buffers, yielding both the name and buffer array/Tensor."""
        memo: set[int] = set()
        modules = self.named_modules(prefix) if recurse else [(prefix, self)]
        for mod_prefix, mod in modules:
            buffers = mod.__dict__.get("_buffers")
            if not buffers:
                continue
            for name, value in buffers.items():
                if value is not None and id(value) not in memo:
                    memo.add(id(value))
                    b_name = f"{mod_prefix}.{name}" if mod_prefix else name
                    yield b_name, value

    def buffers(self) -> list[np.ndarray | Tensor]:
        """
        Recursively collects all unique non-trainable buffers belonging to this module
        and its child modules.
        """
        return [b for _, b in self.named_buffers()]

    def state_dict(
        self,
        destination: dict[str, np.ndarray] | None = None,
        prefix: str = "",
    ) -> dict[str, np.ndarray]:
        """
        Returns a dictionary containing the full state of the module.
        Copies trainable parameter data arrays and non-trainable buffers.
        """
        if destination is None:
            destination = {}
        for name, param in self.named_parameters(prefix=prefix):
            destination[name] = param.data.copy()
        for name, buf in self.named_buffers(prefix=prefix):
            arr = buf.data if isinstance(buf, Tensor) else buf
            destination[name] = arr.copy()
        return destination

    def load_state_dict(self, state_dict: dict[str, np.ndarray], strict: bool = True) -> None:
        """
        Copies parameters and buffers from `state_dict` into this module and its descendants.
        """
        current_params = dict(self.named_parameters())
        current_buffers = dict(self.named_buffers())
        expected_keys = set(current_params) | set(current_buffers)
        missing_keys: list[str] = []
        unexpected_keys: list[str] = [k for k in state_dict if k not in expected_keys]

        for name, param in current_params.items():
            if name in state_dict:
                src_data = np.asarray(state_dict[name])
                if param.shape != src_data.shape:
                    raise ValueError(
                        f"Size mismatch for parameter '{name}': copying a param with shape {src_data.shape} "
                        f"from checkpoint, the shape in current model is {param.shape}."
                    )
                param.data[...] = src_data
            else:
                missing_keys.append(name)

        for name, buf in current_buffers.items():
            if name in state_dict:
                src_data = np.asarray(state_dict[name])
                target = buf.data if isinstance(buf, Tensor) else buf
                if target.shape != src_data.shape:
                    raise ValueError(
                        f"Size mismatch for buffer '{name}': copying a buffer with shape {src_data.shape} "
                        f"from checkpoint, the shape in current model is {target.shape}."
                    )
                target[...] = src_data
            else:
                missing_keys.append(name)

        if strict and (missing_keys or unexpected_keys):
            error_msgs = []
            if missing_keys:
                error_msgs.append(f"Missing key(s) in state_dict: {missing_keys}")
            if unexpected_keys:
                error_msgs.append(f"Unexpected key(s) in state_dict: {unexpected_keys}")
            raise KeyError("Error(s) in loading state_dict:\n\t" + "\n\t".join(error_msgs))

    def zero_grad(self) -> None:
        """Resets gradients of all parameters in the module to zero."""
        for param in self.parameters():
            param.zero_grad()
