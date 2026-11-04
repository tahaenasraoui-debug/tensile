# tests/test_gradcheck.py
"""
Finite-difference numerical gradient verification suite for Tensile.
Enforces relative error < 1e-5 across all elementwise, broadcast, matmul,
activation, reduction, and loss operations.
"""

from __future__ import annotations
from typing import Callable
import numpy as np
import pytest
from tensile import Tensor, nn, unbroadcast


def numerical_gradient(
    func: Callable[[], Tensor],
    target_tensor: Tensor,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes central finite-difference numerical approximation:
        (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    num_grad = np.zeros_like(target_tensor.data, dtype=np.float64)
    it = np.nditer(target_tensor.data, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        orig = float(target_tensor.data[idx])

        target_tensor.data[idx] = orig + eps
        f_plus = float(func().data.item())

        target_tensor.data[idx] = orig - eps
        f_minus = float(func().data.item())

        target_tensor.data[idx] = orig
        num_grad[idx] = (f_plus - f_minus) / (2.0 * eps)
        it.iternext()

    return num_grad


def compute_relative_error(a: np.ndarray, b: np.ndarray) -> float:
    """Computes relative L2 error and max elementwise relative error."""
    diff_norm = float(np.linalg.norm(a - b))
    denom = float(np.linalg.norm(a) + np.linalg.norm(b))
    if denom < 1e-12:
        return diff_norm
    return diff_norm / denom


def assert_gradcheck(
    func: Callable[[], Tensor],
    inputs: list[Tensor],
    eps: float = 1e-6,
    max_rel_error: float = 1e-5,
) -> None:
    """Verifies analytical gradients match finite differences within `max_rel_error` (1e-5)."""
    for inp in inputs:
        inp.zero_grad()

    out = func()
    out.backward()

    for inp in inputs:
        assert inp.grad is not None, f"Analytical grad is None for tensor {inp.shape}"
        num_grad = numerical_gradient(func, inp, eps=eps)
        rel_err = compute_relative_error(inp.grad, num_grad)
        assert rel_err < max_rel_error, (
            f"Gradcheck failed for shape {inp.shape}: relative error {rel_err:.3e} >= {max_rel_error:.1e}\n"
            f"Analytical:\n{inp.grad}\nNumerical:\n{num_grad}"
        )


def test_unbroadcast_invariance() -> None:
    grad = np.ones((2, 3, 4, 5), dtype=np.float64)
    # Leading dimension contraction + singleton axis reduction
    reduced = unbroadcast(grad, (1, 4, 1))
    assert reduced.shape == (1, 4, 1)
    assert np.all(reduced == 2 * 3 * 5)


def test_elementwise_arithmetic_gradcheck() -> None:
    np.random.seed(101)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.uniform(0.6, 2.0, size=(3, 4)), requires_grad=True)

    def forward() -> Tensor:
        z = ((a + b) * (a - b)) / b + (b ** 2.3) - (-a)
        return z.mean()

    assert_gradcheck(forward, [a, b])


def test_broadcasting_binary_ops_gradcheck() -> None:
    np.random.seed(102)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    bias_1d = Tensor(np.random.randn(4), requires_grad=True)
    scale_2d = Tensor(np.random.uniform(0.8, 1.8, size=(3, 1)), requires_grad=True)

    def forward() -> Tensor:
        y = ((x + bias_1d) * scale_2d) - (bias_1d / scale_2d)
        return y.sum()

    assert_gradcheck(forward, [x, bias_1d, scale_2d])


def test_batched_matmul_gradcheck() -> None:
    np.random.seed(103)
    # 4D batched tensor @ 2D broadcasted weight matrix
    a = Tensor(np.random.randn(2, 3, 4, 5), requires_grad=True)
    b = Tensor(np.random.randn(5, 3), requires_grad=True)
    c = Tensor(np.random.randn(2, 1, 3, 2), requires_grad=True)

    def forward() -> Tensor:
        ab = a @ b
        abc = ab @ c
        return abc.mean()

    assert_gradcheck(forward, [a, b, c])


def test_activations_gradcheck() -> None:
    np.random.seed(104)
    x = Tensor(np.random.randn(4, 5), requires_grad=True)
    # Avoid x=0 non-differentiable point for ReLU
    x.data = np.where(np.abs(x.data) < 0.15, 0.25, x.data)

    def forward() -> Tensor:
        r = x.relu()
        s = x.sigmoid()
        t = x.tanh()
        g = x.gelu()
        return (r * 0.4 + s * 0.7 - t * 0.5 + g * 1.2).mean()

    assert_gradcheck(forward, [x])


def test_reductions_and_shapes_gradcheck() -> None:
    np.random.seed(105)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)

    def forward() -> Tensor:
        s1 = x.sum(axis=1, keepdims=True)
        m1 = x.mean(axis=(0, 2), keepdims=False)
        combined = (x + s1).reshape(6, 4).transpose(1, 0)
        return combined.mean() + m1.sum()

    assert_gradcheck(forward, [x])


def test_cross_entropy_and_mlp_gradcheck() -> None:
    np.random.seed(106)
    x = Tensor(np.random.randn(5, 6), requires_grad=True)
    y_idx = Tensor(np.array([0, 2, 1, 3, 2]), requires_grad=False)

    fc1 = nn.Linear(6, 8)
    fc2 = nn.Linear(8, 4)
    criterion = nn.CrossEntropyLoss()

    def forward() -> Tensor:
        h = fc1(x).gelu()
        logits = fc2(h)
        return criterion(logits, y_idx)

    assert_gradcheck(forward, [x, fc1.weight, fc1.bias, fc2.weight, fc2.bias])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
