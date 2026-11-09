import numpy as np
from tensile import Tensor


def test_second_backward_accumulates_leaf_grad_exactly_once_per_call():
    x = Tensor(np.array([1.0, 2.0]), requires_grad=True)
    z = (x * x).sum()
    z.backward()
    first = x.grad.copy()
    z.backward()
    np.testing.assert_allclose(x.grad, 2 * first)  # leaf accumulates; no stale intermediate adjoints


def test_deep_chain_does_not_hit_recursion_limit():
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    for _ in range(5000):
        y = y + 1.0
    y.backward()
    np.testing.assert_allclose(x.grad, 1.0)


def test_shared_subexpression_gradient():
    x = Tensor(np.array(3.0), requires_grad=True)
    y = x * x + x * x
    y.backward()
    np.testing.assert_allclose(x.grad, 12.0)
