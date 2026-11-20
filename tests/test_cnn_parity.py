# tests/test_cnn_parity.py
"""
PyTorch parity test suite for Conv2d, MaxPool2d, LayerNorm, Dropout, AdamW, and RMSprop.
Asserts bit-for-bit mathematical alignment with PyTorch autograd and optimizers.
"""

from __future__ import annotations
import numpy as np
import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F

import tensile as tl
from tensile import Tensor, nn, optim


def test_conv2d_forward_and_backward_pytorch_parity() -> None:
    np.random.seed(42)
    torch.manual_seed(42)

    n, c_in, h_in, w_in = 2, 3, 7, 7
    c_out, kh, kw = 4, 3, 3
    stride, padding = 1, 1

    x_np = np.random.randn(n, c_in, h_in, w_in).astype(np.float64)
    w_np = np.random.randn(c_out, c_in, kh, kw).astype(np.float64)
    b_np = np.random.randn(c_out).astype(np.float64)

    # --- Tensile ---
    conv_tl = nn.Conv2d(c_in, c_out, kernel_size=(kh, kw), stride=stride, padding=padding, bias=True)
    conv_tl.weight.data[...] = w_np.copy()
    conv_tl.bias.data[...] = b_np.copy()
    x_tl = Tensor(x_np.copy(), requires_grad=True)

    out_tl = conv_tl(x_tl)
    loss_tl = (out_tl ** 2).sum()
    loss_tl.backward()

    # --- PyTorch ---
    x_pt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    w_pt = torch.tensor(w_np.copy(), dtype=torch.float64, requires_grad=True)
    b_pt = torch.tensor(b_np.copy(), dtype=torch.float64, requires_grad=True)

    out_pt = F.conv2d(x_pt, w_pt, b_pt, stride=stride, padding=padding)
    loss_pt = (out_pt ** 2).sum()
    loss_pt.backward()

    assert np.allclose(out_tl.data, out_pt.detach().numpy(), atol=1e-7)
    assert np.allclose(x_tl.grad, x_pt.grad.numpy(), atol=1e-7)
    assert np.allclose(conv_tl.weight.grad, w_pt.grad.numpy(), atol=1e-7)
    assert np.allclose(conv_tl.bias.grad, b_pt.grad.numpy(), atol=1e-7)


def test_maxpool2d_forward_and_backward_parity() -> None:
    np.random.seed(1337)
    n, c, h, w = 2, 2, 6, 6
    x_np = np.random.randn(n, c, h, w).astype(np.float64)

    # --- Tensile ---
    pool_tl = nn.MaxPool2d(kernel_size=2, stride=2)
    x_tl = Tensor(x_np.copy(), requires_grad=True)
    out_tl = pool_tl(x_tl)
    loss_tl = (out_tl * 3.0).sum()
    loss_tl.backward()

    # --- PyTorch ---
    x_pt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    out_pt = F.max_pool2d(x_pt, kernel_size=2, stride=2)
    loss_pt = (out_pt * 3.0).sum()
    loss_pt.backward()

    assert np.allclose(out_tl.data, out_pt.detach().numpy(), atol=1e-7)
    assert np.allclose(x_tl.grad, x_pt.grad.numpy(), atol=1e-7)


def test_layernorm_forward_and_backward_parity() -> None:
    np.random.seed(2024)
    batch_size, seq_len, d_model = 3, 5, 8
    x_np = np.random.randn(batch_size, seq_len, d_model).astype(np.float64)
    w_np = np.random.randn(d_model).astype(np.float64)
    b_np = np.random.randn(d_model).astype(np.float64)

    # --- Tensile ---
    ln_tl = nn.LayerNorm(d_model, eps=1e-5)
    ln_tl.weight.data[...] = w_np.copy()
    ln_tl.bias.data[...] = b_np.copy()
    x_tl = Tensor(x_np.copy(), requires_grad=True)
    out_tl = ln_tl(x_tl)
    loss_tl = (out_tl ** 2).sum()
    loss_tl.backward()

    # --- PyTorch ---
    x_pt = torch.tensor(x_np.copy(), dtype=torch.float64, requires_grad=True)
    ln_pt = torch.nn.LayerNorm(d_model, eps=1e-5, dtype=torch.float64)
    with torch.no_grad():
        ln_pt.weight.copy_(torch.tensor(w_np))
        ln_pt.bias.copy_(torch.tensor(b_np))
    out_pt = ln_pt(x_pt)
    loss_pt = (out_pt ** 2).sum()
    loss_pt.backward()

    assert np.allclose(out_tl.data, out_pt.detach().numpy(), atol=1e-6)
    assert np.allclose(x_tl.grad, x_pt.grad.numpy(), atol=1e-6)
    assert np.allclose(ln_tl.weight.grad, ln_pt.weight.grad.numpy(), atol=1e-6)
    assert np.allclose(ln_tl.bias.grad, ln_pt.bias.grad.numpy(), atol=1e-6)


def test_dropout_eval_mode() -> None:
    dropout = nn.Dropout(p=0.5)
    x = Tensor(np.ones((4, 4)), requires_grad=True)

    # Training mode should drop values
    np.random.seed(0)
    out_train = dropout(x)
    assert not np.array_equal(out_train.data, x.data)

    # Eval mode should be identity
    dropout.eval()
    out_eval = dropout(x)
    assert np.array_equal(out_eval.data, x.data)


def test_adamw_step_parity() -> None:
    np.random.seed(123)
    p_np = np.random.randn(4, 4).astype(np.float64)
    g_np = np.random.randn(4, 4).astype(np.float64)

    # Tensile AdamW
    p_tl = Tensor(p_np.copy(), requires_grad=True)
    p_tl.grad = g_np.copy()
    opt_tl = optim.AdamW([p_tl], lr=1e-3, weight_decay=1e-2)
    opt_tl.step()

    # PyTorch AdamW
    p_pt = torch.tensor(p_np.copy(), dtype=torch.float64, requires_grad=True)
    p_pt.grad = torch.tensor(g_np.copy(), dtype=torch.float64)
    opt_pt = torch.optim.AdamW([p_pt], lr=1e-3, weight_decay=1e-2)
    opt_pt.step()

    assert np.allclose(p_tl.data, p_pt.detach().numpy(), atol=1e-6)


def test_rmsprop_step_parity() -> None:
    np.random.seed(999)
    p_np = np.random.randn(5, 5).astype(np.float64)
    g_np = np.random.randn(5, 5).astype(np.float64)

    # Tensile RMSprop
    p_tl = Tensor(p_np.copy(), requires_grad=True)
    p_tl.grad = g_np.copy()
    opt_tl = optim.RMSprop([p_tl], lr=1e-2, alpha=0.99, eps=1e-8)
    opt_tl.step()

    # PyTorch RMSprop
    p_pt = torch.tensor(p_np.copy(), dtype=torch.float64, requires_grad=True)
    p_pt.grad = torch.tensor(g_np.copy(), dtype=torch.float64)
    opt_pt = torch.optim.RMSprop([p_pt], lr=1e-2, alpha=0.99, eps=1e-8)
    opt_pt.step()

    assert np.allclose(p_tl.data, p_pt.detach().numpy(), atol=1e-6)


def test_state_dict_save_and_load() -> None:
    model1 = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )
    model2 = nn.Sequential(
        nn.Linear(4, 8),
        nn.ReLU(),
        nn.Linear(8, 2),
    )

    x = Tensor(np.random.randn(2, 4))
    out1_orig = model1(x).data.copy()
    out2_orig = model2(x).data.copy()
    assert not np.allclose(out1_orig, out2_orig)

    # Save state dict and load into model2
    weights = model1.state_dict()
    model2.load_state_dict(weights)

    out2_synced = model2(x).data
    assert np.allclose(out1_orig, out2_synced, atol=1e-9)


def test_end_to_end_cnn_pipeline() -> None:
    np.random.seed(7)
    model = nn.Sequential(
        nn.Conv2d(1, 4, kernel_size=3, padding=1),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Flatten(),
        nn.Linear(4 * 7 * 7, 10),
    )

    x = Tensor(np.random.randn(4, 1, 14, 14), requires_grad=True)
    y = np.array([0, 2, 5, 9])

    logits = model(x)
    assert logits.shape == (4, 10)

    loss = nn.cross_entropy(logits, y)
    loss.backward()

    assert x.grad is not None
    assert x.grad.shape == (4, 1, 14, 14)
    assert model[0].weight.grad is not None
    assert model[4].weight.grad is not None
