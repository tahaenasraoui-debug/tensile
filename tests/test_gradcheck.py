"""
Finite-difference numerical gradient verification suite for Tensile.
Compatible with both `pytest tests/test_gradcheck.py` and direct execution
via `python tests/test_gradcheck.py`.
"""

from __future__ import annotations
from typing import Callable
import numpy as np

try:
    import pytest
except ImportError:
    pytest = None  # type: ignore

import tensile as tl
from tensile import Tensor, nn


def numerical_gradient(
    func: Callable[[], Tensor],
    target_tensor: Tensor,
    eps: float = 1e-6,
) -> np.ndarray:
    """
    Computes the central finite-difference gradient of scalar-valued `func`
    with respect to `target_tensor`:
        grad_numerical ≈ (f(x + eps) - f(x - eps)) / (2 * eps)
    """
    num_grad = np.zeros_like(target_tensor.data, dtype=np.float64)
    it = np.nditer(target_tensor.data, flags=["multi_index"], op_flags=["readwrite"])

    while not it.finished:
        idx = it.multi_index
        orig_val = float(target_tensor.data[idx])

        # f(x + eps)
        target_tensor.data[idx] = orig_val + eps
        loss_plus = float(func().data.item())

        # f(x - eps)
        target_tensor.data[idx] = orig_val - eps
        loss_minus = float(func().data.item())

        # Restore original value
        target_tensor.data[idx] = orig_val

        num_grad[idx] = (loss_plus - loss_minus) / (2.0 * eps)
        it.iternext()

    return num_grad


def assert_gradcheck(
    func: Callable[[], Tensor],
    inputs: list[Tensor],
    eps: float = 1e-6,
    atol: float = 1e-5,
    rtol: float = 1e-4,
) -> None:
    """Runs analytical backward pass and compares against numerical gradients."""
    for inp in inputs:
        inp.zero_grad()

    out = func()
    out.backward()

    for inp in inputs:
        assert inp.grad is not None, "Analytical gradient was not populated."
        num_grad = numerical_gradient(func, inp, eps=eps)
        np.testing.assert_allclose(
            inp.grad,
            num_grad,
            atol=atol,
            rtol=rtol,
            err_msg=f"Gradient mismatch for tensor of shape {inp.shape}",
        )


def test_add_and_mul_grad() -> None:
    np.random.seed(42)
    a = Tensor(np.random.randn(4, 5), requires_grad=True)
    b = Tensor(np.random.randn(4, 5), requires_grad=True)

    def forward() -> Tensor:
        c = a + b
        d = (a * b) + (c * a)
        return d.mean()

    assert_gradcheck(forward, [a, b])


def test_sub_div_pow_neg_grad() -> None:
    np.random.seed(42)
    a = Tensor(np.random.randn(3, 4), requires_grad=True)
    b = Tensor(np.random.uniform(0.5, 2.5, size=(3, 4)), requires_grad=True)

    def forward() -> Tensor:
        res = ((-a - b) / b) + (b ** 2.5)
        return res.sum()

    assert_gradcheck(forward, [a, b])


def test_matmul_grad() -> None:
    np.random.seed(42)
    x = Tensor(np.random.randn(6, 4), requires_grad=True)
    w = Tensor(np.random.randn(4, 3), requires_grad=True)

    def forward() -> Tensor:
        return (x @ w).mean()

    assert_gradcheck(forward, [x, w])


def test_broadcast_addition_and_mul_grad() -> None:
    np.random.seed(42)
    x = Tensor(np.random.randn(5, 7), requires_grad=True)
    bias = Tensor(np.random.randn(7), requires_grad=True)
    scale = Tensor(np.random.randn(5, 1), requires_grad=True)

    def forward() -> Tensor:
        out = (x + bias) * scale
        return out.sum()

    assert_gradcheck(forward, [x, bias, scale])


def test_reshape_transpose_reductions_grad() -> None:
    np.random.seed(42)
    x = Tensor(np.random.randn(2, 3, 4), requires_grad=True)
    w = Tensor(np.random.randn(6, 4), requires_grad=True)

    def forward() -> Tensor:
        reshaped = x.reshape(6, 4)
        transposed = (reshaped * w).transpose(1, 0)
        reduced = transposed.sum(axis=0, keepdims=False)
        return reduced.mean()

    assert_gradcheck(forward, [x, w])


def test_linear_layer_and_cross_entropy_backprop() -> None:
    np.random.seed(42)
    batch_size, in_dim, hidden_dim, num_classes = 4, 6, 5, 3
    x = Tensor(np.random.randn(batch_size, in_dim), requires_grad=True)
    y = Tensor(np.array([0, 2, 1, 0]), requires_grad=False)

    linear1 = nn.Linear(in_dim, hidden_dim, bias=True)
    relu = nn.ReLU()
    linear2 = nn.Linear(hidden_dim, num_classes, bias=True)

    x.data = np.where(np.abs(x.data) < 0.1, 0.2, x.data)

    def forward() -> Tensor:
        h = relu(linear1(x))
        logits = linear2(h)
        return nn.cross_entropy(logits, y)

    assert_gradcheck(
        forward,
        [x, linear1.weight, linear1.bias, linear2.weight, linear2.bias],
    )


if __name__ == "__main__":
    tests = [
        ("test_add_and_mul_grad", test_add_and_mul_grad),
        ("test_sub_div_pow_neg_grad", test_sub_div_pow_neg_grad),
        ("test_matmul_grad", test_matmul_grad),
        ("test_broadcast_addition_and_mul_grad", test_broadcast_addition_and_mul_grad),
        ("test_reshape_transpose_reductions_grad", test_reshape_transpose_reductions_grad),
        ("test_linear_layer_and_cross_entropy_backprop", test_linear_layer_and_cross_entropy_backprop),
    ]
    print("Running Tensile numerical gradient checks...")
    for name, fn in tests:
        fn()
        print(f"  [PASS] {name}")
    print("All 6 numerical gradient test suites passed successfully!")
