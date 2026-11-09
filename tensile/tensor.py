# tensile/tensor.py
"""
Core Tensor and Reverse-Mode Autograd Engine for Tensile.
Pure Python / NumPy implementation with exact broadcast adjoint reduction
and topological graph serialization for interactive visualization.
"""

from __future__ import annotations
from typing import Any, Callable, Iterable, Optional, Sequence, Union
import numpy as np

ArrayLike = Union[int, float, list, np.ndarray, "Tensor"]


def unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
    """
    Mathematically unbroadcasts incoming adjoint `grad` so its shape matches `target_shape`.
    1. Sums out prepended higher-rank dimensions when `grad.ndim > len(target_shape)`.
    2. Sums with `keepdims=True` over axes where `target_shape` has singleton dimension 1.
    """
    if grad.shape == target_shape:
        return grad

    # 1. Contract leading dimensions added by broadcasting
    ndim_diff = grad.ndim - len(target_shape)
    if ndim_diff > 0:
        grad = grad.sum(axis=tuple(range(ndim_diff)))

    # 2. Sum across singleton dimensions (where target_shape[i] == 1 and grad.shape[i] > 1)
    singleton_axes = tuple(
        i for i, (g_dim, t_dim) in enumerate(zip(grad.shape, target_shape))
        if t_dim == 1 and g_dim > 1
    )
    if singleton_axes:
        grad = grad.sum(axis=singleton_axes, keepdims=True)

    return grad.reshape(target_shape)


class Tensor:
    """
    Multi-dimensional array wrapper tracking computational history and reverse-mode
    adjoints over a dynamic Directed Acyclic Graph (DAG).
    """

    _id_counter: int = 0

    def __init__(
        self,
        data: ArrayLike,
        requires_grad: bool = False,
        _prev: Iterable["Tensor"] = (),
        _op: str = "",
        name: str = "",
    ) -> None:
        Tensor._id_counter += 1
        self._id: int = Tensor._id_counter
        self.name: str = name

        if isinstance(data, Tensor):
            raw = data.data.copy()
        elif isinstance(data, np.ndarray):
            raw = data
        else:
            raw = np.asarray(data)

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
        # Preserve deterministic parent ordering for visualization while supporting set semantics
        self._parents: tuple[Tensor, ...] = tuple(_prev)
        self._prev: set[Tensor] = set(self._parents)
        self.creator: str = _op
        self._op: str = _op
        self._backward: Callable[[], None] = lambda: None

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

    def __hash__(self) -> int:
        return id(self)

    def __eq__(self, other: object) -> bool:
        return self is other

    def __repr__(self) -> str:
        op_str = f", creator='{self._op}'" if self._op else ""
        return (
            f"Tensor({np.array2string(self.data, separator=', ', prefix='Tensor(')}, "
            f"requires_grad={self.requires_grad}{op_str})"
        )

    @staticmethod
    def _unbroadcast(grad: np.ndarray, target_shape: tuple[int, ...]) -> np.ndarray:
        return unbroadcast(grad, target_shape)

    @staticmethod
    def _ensure_tensor(val: ArrayLike) -> Tensor:
        return val if isinstance(val, Tensor) else Tensor(val, requires_grad=False)

    # -------------------------------------------------------------------------
    # Autograd Engine & Topological Traversal
    # -------------------------------------------------------------------------
    def zero_grad(self) -> None:
        """Resets the gradient buffer of this tensor to zero."""
        if self.requires_grad:
            self.grad = np.zeros_like(self.data, dtype=np.float64)
        else:
            self.grad = None

    def _topological_sort(self) -> list[Tensor]:
        """Returns nodes in topological order (inputs -> root output) via DFS."""
        topo: list[Tensor] = []
        visited: set[int] = set()
        # Iterative post-order DFS: deep graphs (long chains, unrolled loops) must not hit Python's recursion limit.
        stack: list[tuple[Tensor, bool]] = [(self, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                topo.append(node)
                continue
            if id(node) in visited:
                continue
            visited.add(id(node))
            stack.append((node, True))
            for parent in reversed(node._parents):
                if id(parent) not in visited:
                    stack.append((parent, False))
        return topo

    def backward(self, grad: Optional[np.ndarray] = None) -> None:
        """
        Executes reverse-mode automatic differentiation over the DAG rooted at `self`.
        """
        if not self.requires_grad:
            raise RuntimeError("Cannot call backward() on a Tensor with requires_grad=False.")

        if grad is None:
            if self.data.size != 1:
                raise RuntimeError(
                    "Grad can be implicitly created only for scalar outputs (size == 1)."
                )
            seed_grad = np.ones_like(self.data, dtype=np.float64)
        else:
            seed_grad = np.asarray(grad, dtype=np.float64)
            if seed_grad.shape != self.data.shape:
                raise ValueError(
                    f"Seed grad shape {seed_grad.shape} does not match Tensor shape {self.data.shape}."
                )

        topo = self._topological_sort()

        for node in topo:
            if node.requires_grad and (node.grad is None or node._parents):
                # Intermediate (non-leaf) nodes start from zero on every backward pass so that a second
                # backward() does not re-propagate stale adjoints. Leaf grads still accumulate across calls.
                node.grad = np.zeros_like(node.data, dtype=np.float64)

        self.grad = seed_grad

        for node in reversed(topo):
            node._backward()

    # -------------------------------------------------------------------------
    # Core Elementwise Operations
    # -------------------------------------------------------------------------
    def __add__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data + other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="Add",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad += unbroadcast(out.grad, other_t.shape)

        out._backward = _backward
        return out

    def __radd__(self, other: ArrayLike) -> Tensor:
        return self + other

    def __neg__(self) -> Tensor:
        out = Tensor(
            -self.data,
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="Neg",
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
            _op="Sub",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad, self.shape)
            if other_t.requires_grad:
                other_t.grad += unbroadcast(-out.grad, other_t.shape)

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
            _op="Mul",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad * other_t.data, self.shape)
            if other_t.requires_grad:
                other_t.grad += unbroadcast(out.grad * self.data, other_t.shape)

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
            _op="Div",
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                self.grad += unbroadcast(out.grad / other_t.data, self.shape)
            if other_t.requires_grad:
                grad_other = -out.grad * self.data / (other_t.data ** 2)
                other_t.grad += unbroadcast(grad_other, other_t.shape)

        out._backward = _backward
        return out

    def __rtruediv__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        return other_t / self

    def __pow__(self, exponent: Union[int, float, Tensor]) -> Tensor:
        exp_t = self._ensure_tensor(exponent)
        req_grad = self.requires_grad or exp_t.requires_grad
        op_label = f"Pow({exponent})" if isinstance(exponent, (int, float)) else "Pow"
        out = Tensor(
            self.data ** exp_t.data,
            requires_grad=req_grad,
            _prev=(self, exp_t) if req_grad else (),
            _op=op_label,
        )

        def _backward() -> None:
            if out.grad is None:
                return
            if self.requires_grad:
                grad_self = out.grad * exp_t.data * (self.data ** (exp_t.data - 1.0))
                self.grad += unbroadcast(grad_self, self.shape)
            if exp_t.requires_grad:
                safe_base = np.where(self.data > 0.0, self.data, 1.0)
                grad_exp = out.grad * out.data * np.log(safe_base)
                exp_t.grad += unbroadcast(grad_exp, exp_t.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Batched Matrix Multiplication
    # -------------------------------------------------------------------------
    def __matmul__(self, other: ArrayLike) -> Tensor:
        other_t = self._ensure_tensor(other)
        req_grad = self.requires_grad or other_t.requires_grad
        out = Tensor(
            self.data @ other_t.data,
            requires_grad=req_grad,
            _prev=(self, other_t) if req_grad else (),
            _op="MatMul",
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
                self.grad += unbroadcast(grad_a, self.shape)

            if other_t.requires_grad:
                grad_b = np.swapaxes(a, -1, -2) @ g
                if b_1d:
                    grad_b = grad_b.squeeze(axis=-1)
                other_t.grad += unbroadcast(grad_b, other_t.shape)

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Non-Linear Activations (ReLU, Sigmoid, Tanh, GELU)
    # -------------------------------------------------------------------------
    def relu(self) -> Tensor:
        out = Tensor(
            np.maximum(0.0, self.data),
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="ReLU",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * (self.data > 0.0)

        out._backward = _backward
        return out

    def sigmoid(self) -> Tensor:
        # Numerically stable sigmoid computation
        x = self.data
        sig = np.where(
            x >= 0.0,
            1.0 / (1.0 + np.exp(-np.clip(x, -500.0, 500.0))),
            np.exp(np.clip(x, -500.0, 500.0)) / (1.0 + np.exp(np.clip(x, -500.0, 500.0))),
        )
        out = Tensor(
            sig,
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="Sigmoid",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * sig * (1.0 - sig)

        out._backward = _backward
        return out

    def tanh(self) -> Tensor:
        t = np.tanh(self.data)
        out = Tensor(
            t,
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="Tanh",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            self.grad += out.grad * (1.0 - t ** 2)

        out._backward = _backward
        return out

    def gelu(self) -> Tensor:
        """
        Gaussian Error Linear Unit (Hendrycks & Gimpel tanh approximation, matching
        PyTorch `nn.GELU(approximate='tanh')` with exact analytical derivative):
        GELU(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
        """
        x = self.data
        c = np.sqrt(2.0 / np.pi)
        alpha = 0.044715
        inner = c * (x + alpha * (x ** 3))
        tanh_inner = np.tanh(inner)
        out_data = 0.5 * x * (1.0 + tanh_inner)

        out = Tensor(
            out_data,
            requires_grad=self.requires_grad,
            _prev=(self,) if self.requires_grad else (),
            _op="GELU",
        )

        def _backward() -> None:
            if out.grad is None or not self.requires_grad:
                return
            sech2_inner = 1.0 - (tanh_inner ** 2)
            d_inner_dx = c * (1.0 + 3.0 * alpha * (x ** 2))
            d_gelu_dx = 0.5 * (1.0 + tanh_inner) + 0.5 * x * sech2_inner * d_inner_dx
            self.grad += out.grad * d_gelu_dx

        out._backward = _backward
        return out

    # -------------------------------------------------------------------------
    # Reductions & Shape Transformations
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
            _op="Sum",
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
            _op="Mean",
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
            _op="Reshape",
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
            _op="Transpose",
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

    # -------------------------------------------------------------------------
    # Computational Graph Export for Web Visualizer
    # -------------------------------------------------------------------------
    def to_graph_dict(self) -> dict[str, Any]:
        """
        Exports the topological DAG rooted at `self` into a JSON-serializable
        dictionary containing tensor data nodes, operator nodes, directed edges,
        value summaries (min, max, norm), gradient statistics, and reverse-mode
        backprop step ordering for the interactive playback scrubber.
        """
        topo = self._topological_sort()
        nodes: list[dict[str, Any]] = []
        edges: list[dict[str, Any]] = []
        backprop_steps: list[dict[str, Any]] = []

        for idx, t in enumerate(topo):
            t_id = f"tensor_{t._id}"
            flat = t.data.astype(np.float64).ravel()
            val_min = float(np.min(flat)) if flat.size > 0 else 0.0
            val_max = float(np.max(flat)) if flat.size > 0 else 0.0
            val_norm = float(np.linalg.norm(flat)) if flat.size > 0 else 0.0

            if t.grad is not None and t.grad.size > 0:
                g_flat = t.grad.astype(np.float64).ravel()
                grad_min = float(np.min(g_flat))
                grad_max = float(np.max(g_flat))
                grad_norm = float(np.linalg.norm(g_flat))
            else:
                grad_min = 0.0
                grad_max = 0.0
                grad_norm = 0.0

            role = "leaf_param" if (not t._op and t.requires_grad) else (
                "input_data" if not t._op else ("root_loss" if t is self else "intermediate")
            )
            label = t.name if t.name else (
                "Loss" if t is self else (f"Param {t._id}" if role == "leaf_param" else f"T{t._id}")
            )

            nodes.append({
                "id": t_id,
                "type": "tensor",
                "role": role,
                "label": label,
                "op": t._op,
                "shape": list(t.shape),
                "requires_grad": t.requires_grad,
                "topo_index": idx,
                "stats": {
                    "min": round(val_min, 5),
                    "max": round(val_max, 5),
                    "norm": round(val_norm, 5),
                },
                "grad_stats": {
                    "min": round(grad_min, 6),
                    "max": round(grad_max, 6),
                    "norm": round(grad_norm, 6),
                },
            })

            if t._op:
                op_id = f"op_{t._id}"
                nodes.append({
                    "id": op_id,
                    "type": "operator",
                    "role": "operator",
                    "label": t._op,
                    "topo_index": idx,
                    "grad_norm": round(grad_norm, 6),
                })
                edges.append({
                    "id": f"e_{op_id}_to_{t_id}",
                    "source": op_id,
                    "target": t_id,
                    "grad_norm": round(grad_norm, 6),
                })

                parent_edges: list[str] = []
                parent_tensor_ids: list[str] = []
                for p in t._parents:
                    p_id = f"tensor_{p._id}"
                    edge_id = f"e_{p_id}_to_{op_id}"
                    p_grad_norm = (
                        float(np.linalg.norm(p.grad))
                        if (p.grad is not None and p.grad.size > 0)
                        else 0.0
                    )
                    edges.append({
                        "id": edge_id,
                        "source": p_id,
                        "target": op_id,
                        "grad_norm": round(p_grad_norm, 6),
                    })
                    parent_edges.append(edge_id)
                    parent_tensor_ids.append(p_id)

                backprop_steps.append({
                    "tensor_id": t_id,
                    "op_id": op_id,
                    "op_label": t._op,
                    "incoming_edge": f"e_{op_id}_to_{t_id}",
                    "outgoing_edges": parent_edges,
                    "parent_tensors": parent_tensor_ids,
                    "grad_norm": round(grad_norm, 6),
                })

        # Reverse order for backpropagation playback
        backprop_steps.reverse()
        for step_idx, step in enumerate(backprop_steps):
            step["step_index"] = step_idx

        return {
            "nodes": nodes,
            "edges": edges,
            "backprop_steps": backprop_steps,
        }
