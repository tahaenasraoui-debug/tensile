"""
Core Tensor and Reverse-Mode Autograd Engine for Tensile.
Relies strictly on pure NumPy.
"""

from __future__ import annotations
from typing import Callable, Iterable, Optional, Sequence, Union
import numpy as np

ArrayLike = Union[int, float, list, np.ndarray, "Tensor"]


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Sums `grad` along any dimensions that were broadcasted during the forward pass
    so that the resulting gradient matches `target_shape` exactly.

    Parameters
    ----------
    grad : np.ndarray
        Incoming upstream gradient array.
    target_shape : tuple[int, ...]
        The original shape of the tensor receiving the gradient.

    Returns
    -------
    np.ndarray
        Gradient reduced to `target_shape`.
    """
    if grad.shape == target_shape:
        return grad

    # 1. Sum out any leading dimensions prepended during broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # 2. Sum along dimensions that were size 1 in target_shape but stretched in grad
    axes_to_sum = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape))
        if t_dim == 1 and g_dim > 1
    )
    if axes_to_sum:
        grad = grad.sum(axis=axes_to_sum, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """
    Multi-dimensional array wrapper supporting reverse-mode automatic differentiation.
    """

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _prev: Iterable["Tensor"] = (),
        _op: str = "",
    ) -> None:
        if isinstance(data, Tensor):
            raw = data.data.copy()
        elif isinstance(data, np.ndarray):
            raw = data
        else:
            raw = np.asarray(data)

        # Differentiable tensors use float64 for numerical stability and accurate gradchecks.
        # Non-differentiable integer arrays (e.g., class labels) retain their integer dtype.
        if requires_grad or np.issubdtype(raw.dtype, np.floating):
            self.data: np.ndarray = raw.astype(np.float64, copy=False)
        elif np.issubdtype(raw.dtype, np.integer) or np.issubdtype(raw.dtype, np.bool_):
            self.data = raw
        else:
            self.data = raw.astype(np.float64)

        self.requires_grad: bool = bool(requires_grad)
        self.grad: Optional[np.ndarray] = (
            np.zeros_like(self.data, dtype=np.float64) if self.requires_grad else None
        )
        self._prev: set[Tensor] = set(_prev)
        self._backward: Callable[[], None] = lambda: None
        self._op: str = _op

    # -------------------------------------------------------------------------
    # Properties & Representations
    # -------------------------------------------------------------------------
    @property
    def shape(self) -> tuple[int, ...]:
        return self.data.shape

    @property
    def ndim(self) -> int:
        return self.data.ndim

    @property
    def size(self) -> int:
        return self.data.size

    @property
    def dtype(self) -> np.dtype:
        return self.data.dtype

    @property
    def T(self) -> Tensor:
        return self.transpose()

    def __repr__(self) -> str:
        grad_fn_str = f", op='{self._op}'" if self._op else ""
        return (
            f"Tensor({np.array2string(self.data, separator=', ', prefix='Tensor(')}, "
            f"requires_grad={self.requires_grad}{grad_fn_str})"
        )

    @staticmethod
    def _unbroadcast(grad: np.ndarray, original_shape: tuple[int, ...]) -> np.ndarray:
        """Internal static helper delegating to `unbroadcast`."""
        return unbroadcast(grad, original_shape)

    @staticmethod
    def _ensure_tensor(val: ArrayLike) -> Tensor:
        return val if isinstance(val, Tensor) else Tensor(val, requires_grad=False)

    # -------------------------------------------------------------------------
    # Autograd Engine
    # -------------------------------------------------------------------------
    def zero_grad(self) -> None:
        """Resets the gradient of this tensor to zeros."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self.grad = None

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Computes the gradient of this tensor with respect to all upstream leaves
        in the computational graph via reverse-mode autodiff (DFS topological sort).
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a Tensor with requires_grad=False.")

        # Seed gradient for the root node
        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "grad can be implicitly created only for scalar outputs (size == 1)."
                )
            seed_grad = np.ones_like(self.data, dtype=np.float64)
        else:
            seed_grad = np.asarray(grad, dtype=np.float64)
            if seed_grad.shape != self.data.shape:
                raise ValueError(
                    f"Seed grad shape {seed_grad.shape} does not match Tensor shape {self.data.shape}."
                )

        # Build topological ordering using Depth-First Search (DFS)
        topo: list[Tensor] = []
        visited: set[int] = set()

        def build_topo(node: Tensor) -> None:
            node_id = id(node)
            if node_id not in visited:
                visited.add(node_id)
                for parent in node._prev:
                    build_topo(parent)
                topo.append(node)

        build_topo(self)

        # Ensure all requires_grad nodes in the active graph have initialized grad buffers
        for node in topo:
            if node.requires_grad and node.grad is None:
                node.grad = np.zeros_like(node.data, dtype=np.float64)

        self.grad = seed_grad

        # Reverse topological pass
        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------------------
    # Arithmetic & Matrix Operations
    # -------------------------------------------------------------------------
    def __add__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data + other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="+",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad += self._unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="neg",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += -out.grad

        out._backward = _backward
        return out

    def __sub__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data - other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="-",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad += self._unbroadcast(-out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __rsub__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t - self

    def __mul__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data * other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="*",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += self._unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad:
                other_t.grad += self._unbroadcast(out.grad * self.data, other_t.shape)

        out._backward = _backward
        return out

    def __rmul__(self, other: ArrayLike) -> Tensor:
        return self * other

    def __truediv__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data / other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="/",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad / other_t.data
                self.grad += self._unbroadcast(grad_self, self.shape)
            if other_t.requires_grad:
                grad_other = -out.grad * self.data / (other_t.data ** 2)
                other_t.grad += self._unbroadcast(grad_other, other_t.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t / self

    def __pow__(self, exponent: Union[int, float, Tensor]) -> Tensor:
        exp_t = self._ensure_tensor(exponent)
        req_grad = self.requires_grad or exp_t.requires_grad
        out = Tensor(
            self.data ** exp_t.data,
            requires_grad=req_grad,
            _prev=(self, exp_t) if req_grad else (),
            _op=f"**{exponent}" if isinstance(exponent, (int, float)) else "**",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad * exp_t.data * (self.data ** (exp_t.data - 1.0))
                self.grad += self._unbroadcast(grad_self, self.shape)
            if exp_t.requires_grad:
                safe_base = np.where(self.data > 0, self.data, 1.0)
                grad_exp = out.grad * out.data * np.log(safe_base)
                exp_t.grad += self._unbroadcast(grad_exp, exp_t.shape)

        out._backward = _backward
        return out

    def __matmul__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data @ other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="@",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            a_1d = self.ndim == 1
            b_1d = other_t.ndim == 1

            a = self.data[np.newaxis, :] if a_1d else self.data
            b = other_t.data[:, np.newaxis] if b_1d else other_t.data
            g = out.grad
            if a_1d and b_1d:
                g = np.asarray(g).reshape(1, 1)
            elif a_1d:
                g = np.expand_dims(g, axis=-2)
            elif b_1d:
                g = np.expand_dims(g, axis=-1)

            if self.requires_grad:
                grad_a = g @ np.swapaxes(b, -1, -2)
                if a_1d:
                    grad_a = grad_a.squeeze(axis=-2)
                self.grad += self._unbroadcast(grad_a, self.shape)

            if other_t.requires_grad:
                grad_b = np.swapaxes(a, -1, -2) @ g
                if b_1d:
                    grad_b = grad_b.squeeze(axis=-1)
                other_t.grad += self._unbroadcast(grad_b, other_t.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reductions & Shape Manipulation
    # -------------------------------------------------------------------------
    def sum(
        self,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        out = Tensor(
            self.data.sum(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="sum",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if not keepdims and axis is not None:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                norm_axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                grad = np.expand_dims(grad, axis=norm_axes)
            self.grad += np.broadcast_to(grad, self.shape)

        out._backward = _backward
        return out

    def mean(
        self,
        axis: Optional[Union[int, tuple[int, ...]]] = None,
        keepdims: bool = False,
    ) -> Tensor:
        out = Tensor(
            self.data.mean(axis=axis, keepdims=keepdims),
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="mean",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            grad = out.grad
            if axis is None:
                count = self.data.size
            else:
                axes = (axis,) if isinstance(axis, int) else tuple(axis)
                norm_axes = tuple(a if a >= 0 else a + self.ndim for a in axes)
                count = int(np.prod([self.shape[a] for a in norm_axes]))
                if not keepdims:
                    grad = np.expand_dims(grad, axis=norm_axes)
            self.grad += np.broadcast_to(grad / count, self.shape)

        out._backward = _backward
        return out

    def reshape(self, *shape: Union[int, Sequence[int]]) -> Tensor:
        if len(shape) == 1 and isinstance(shape[0], (tuple, list)):
            target_shape = tuple(shape[0])
        else:
            target_shape = tuple(int(s) for s in shape)

        out = Tensor(
            self.data.reshape(target_shape),
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="reshape",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad.reshape(self.shape)

        out._backward = _backward
        return out

    def transpose(self, *axes: Union[int, Sequence[int]]) -> Tensor:
        if len(axes) == 0:
            perm = None
        elif len(axes) == 1 and isinstance(axes[0], (tuple, list)):
            perm = tuple(axes[0])
        elif len(axes) == 1 and axes[0] is None:
            perm = None
        else:
            perm = tuple(int(a) for a in axes)

        out = Tensor(
            self.data.transpose(perm),
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="transpose",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            if perm is None:
                self.grad += out.grad.transpose()
            else:
                norm_perm = [a if a >= 0 else a + self.ndim for a in perm]
                inv_perm = tuple(np.argsort(norm_perm))
                self.grad += out.grad.transpose(inv_perm)

        out._backward = _backward
        return out
