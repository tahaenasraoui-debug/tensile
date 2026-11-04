# tests/test_pytorch_parity.py
"""
PyTorch parity verification harness.
Asserts that Tensile matches PyTorch forward activations, loss values,
analytical gradients, and optimizer updates within `atol=1e-6`.
"""

from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import tensile as tl
from tensile import Tensor, nn


def test_activations_and_batched_matmul_pytorch_parity() -> None:
    np.random.seed(7)
    x_np = np.random.randn(2, 3, 4, 5).astype(np.float64)
    w_np = np.random.randn(5, 6).astype(np.float64)

    # Tensile pass
    x_tl = Tensor(x_np.copy(), requires_grad=True)
    w_tl = Tensor(w_np.copy(), requires_grad=True)
    out_tl = ((x_tl @ w_tl).gelu() + (x_tl @ w_tl).sigmoid() - (x_tl @ w_tl).tanh()).mean()
    out_tl.backward()

    # PyTorch pass
    x_pt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)
    mm_pt = x_pt @ w_pt
    out_pt = (
        F.gelu(mm_pt, approximate="tanh") + torch.sigmoid(mm_pt) - torch.tanh(mm_pt)
    ).mean()
    out_pt.backward()

    assert np.allclose(out_tl.data, out_pt.detach().numpy(), atol=1e-6)
    assert np.allclose(x_tl.grad, x_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(w_tl.grad, w_pt.grad.numpy(), atol=1e-6)


def test_mlp_forward_backward_and_adam_step_parity() -> None:
    np.random.seed(42)
    b, in_d, hid_d, out_d = 8, 4, 10, 3
    x_np = np.random.randn(b, in_d).astype(np.float64)
    y_np = np.random.randint(0, out_d, size=(b,), dtype=np.int64)

    w1_np = np.random.randn(in_d, hid_d).astype(np.float64) * 0.3
    b1_np = np.random.randn(hid_d).astype(np.float64) * 0.1
    w2_np = np.random.randn(hid_d, out_d).astype(np.float64) * 0.3
    b2_np = np.random.randn(out_d).astype(np.float64) * 0.1

    # --- Tensile Setup ---
    fc1_tl = nn.Linear(in_d, hid_d)
    fc2_tl = nn.Linear(hid_d, out_d)
    fc1_tl.weight.data = w1_np.copy()
    fc1_tl.bias.data = b1_np.copy()
    fc2_tl.weight.data = w2_np.copy()
    fc2_tl.bias.data = b2_np.copy()
    crit_tl = nn.CrossEntropyLoss()
    opt_tl = tl.optim.Adam([fc1_tl.weight, fc1_tl.bias, fc2_tl.weight, fc2_tl.bias], lr=1e-2)

    # --- PyTorch Setup ---
    w1_pt = torch.tensor(w1_np.copy(), dtype=torch.float64, requires_grad=True)
    b1_pt = torch.tensor(b1_np.copy(), dtype=torch.float64, requires_grad=True)
    w2_pt = torch.tensor(w2_np.copy(), dtype=torch.float64, requires_grad=True)
    b2_pt = torch.tensor(b2_np.copy(), dtype=torch.float64, requires_grad=True)
    opt_pt = torch.optim.Adam([w1_pt, b1_pt, w2_pt, b2_pt], lr=1e-2)

    for _ in range(3):
        # Tensile step
        opt_tl.zero_grad()
        h_tl = fc1_tl(Tensor(x_np)).relu()
        logits_tl = fc2_tl(h_tl)
        loss_tl = crit_tl(logits_tl, y_np)
        loss_tl.backward()

        # PyTorch step
        opt_pt.zero_grad()
        h_pt = F.relu(torch.tensor(x_np, dtype=torch.float64) @ w1_pt + b1_pt)
        logits_pt = h_pt @ w2_pt + b2_pt
        loss_pt = F.cross_entropy(logits_pt, torch.tensor(y_np, dtype=torch.long))
        loss_pt.backward()

        # Verify forward loss and backward gradients before optimizer step
        assert np.allclose(loss_tl.data, loss_pt.detach().numpy(), atol=1e-6)
        assert np.allclose(fc1_tl.weight.grad, w1_pt.grad.numpy(), atol=1e-6)
        assert np.allclose(fc1_tl.bias.grad, b1_pt.grad.numpy(), atol=1e-6)
        assert np.allclose(fc2_tl.weight.grad, w2_pt.grad.numpy(), atol=1e-6)
        assert np.allclose(fc2_tl.bias.grad, b2_pt.grad.numpy(), atol=1e-6)

        opt_tl.step()
        opt_pt.step()

        # Verify updated weights match after Adam step
        assert np.allclose(fc1_tl.weight.data, w1_pt.detach().numpy(), atol=1e-6)
        assert np.allclose(fc2_tl.weight.data, w2_pt.detach().numpy(), atol=1e-6)
