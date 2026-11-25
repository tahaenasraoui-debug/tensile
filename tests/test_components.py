# tests/test_components.py
"""
Unit and regression tests for core Tensile modules, optimizers, losses,
data pipeline, functional ops, and iterative graph tracing (pure NumPy, no PyTorch required).
"""

from __future__ import annotations

import numpy as np
import pytest

from tensile import Tensor, nn, ops, optim
from tensile.data import DataLoader, Dataset, TensorDataset
from tensile.utils.vis import _trace

# =========================================================================
# Module Buffers & BatchNorm1d
# =========================================================================


def test_module_buffer_registration_and_state_dict() -> None:
    class BufferModule(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.weight = Tensor(np.array([1.0, 2.0]), requires_grad=True)
            self.register_buffer("running_stat", np.array([3.0, 4.0]))
            self.register_buffer("tensor_buf", Tensor(np.array([5.0]), requires_grad=False))
            self.register_buffer("optional_buf", None)

    mod = BufferModule()

    # Buffers must not appear in trainable parameters
    assert list(dict(mod.named_parameters()).keys()) == ["weight"]
    assert len(mod.parameters()) == 1

    # Non-None buffers appear in named_buffers() and buffers()
    buf_map = dict(mod.named_buffers())
    assert set(buf_map.keys()) == {"running_stat", "tensor_buf"}
    assert len(mod.buffers()) == 2

    # state_dict includes both trainable parameters and non-None buffers as copies
    sd = mod.state_dict()
    assert set(sd.keys()) == {"weight", "running_stat", "tensor_buf"}
    np.testing.assert_allclose(sd["running_stat"], [3.0, 4.0])
    np.testing.assert_allclose(sd["tensor_buf"], [5.0])

    # Mutating returned state_dict must not mutate module state
    sd["running_stat"][0] = 999.0
    np.testing.assert_allclose(mod.running_stat, [3.0, 4.0])

    # Reassigning a registered buffer attribute stays synchronized with state_dict
    mod.running_stat = np.array([7.0, 8.0])
    np.testing.assert_allclose(mod.state_dict()["running_stat"], [7.0, 8.0])

    # Loading into a fresh module restores parameters and buffers cleanly
    mod2 = BufferModule()
    mod2.load_state_dict(mod.state_dict())
    np.testing.assert_allclose(mod2.running_stat, [7.0, 8.0])
    np.testing.assert_allclose(mod2.tensor_buf.data, [5.0])

    # Validation checks
    with pytest.raises(ValueError, match="requires_grad=True"):
        mod.register_buffer("bad_buf", Tensor(np.ones(2), requires_grad=True))
    with pytest.raises(ValueError, match="non-empty string"):
        mod.register_buffer("a.b", np.ones(2))
    with pytest.raises(ValueError, match="Size mismatch for buffer"):
        mod2.load_state_dict(
            {"weight": np.zeros(2), "running_stat": np.zeros(3), "tensor_buf": np.zeros(1)}
        )
    with pytest.raises(KeyError, match="Missing key"):
        mod2.load_state_dict({"weight": np.zeros(2)}, strict=True)
    with pytest.raises(KeyError, match="Unexpected key"):
        mod2.load_state_dict(
            {
                "weight": np.zeros(2),
                "running_stat": np.zeros(2),
                "tensor_buf": np.zeros(1),
                "extra": np.zeros(1),
            },
            strict=True,
        )

    # Non-strict load updates matching keys without raising
    mod2.load_state_dict({"running_stat": np.array([-1.0, -2.0]), "extra": np.zeros(1)}, strict=False)
    np.testing.assert_allclose(mod2.running_stat, [-1.0, -2.0])


def test_batchnorm1d_save_load_eval_parity() -> None:
    rng = np.random.default_rng(123)

    model1 = nn.Sequential(
        nn.Linear(4, 3),
        nn.BatchNorm1d(3, momentum=0.25),
        nn.ReLU(),
    )
    model2 = nn.Sequential(
        nn.Linear(4, 3),
        nn.BatchNorm1d(3, momentum=0.25),
        nn.ReLU(),
    )

    # Train model1 for several steps so running_mean and running_var diverge from (0, 1)
    model1.train()
    for _ in range(5):
        batch = Tensor(rng.normal(loc=5.0, scale=2.5, size=(8, 4)), requires_grad=False)
        _ = model1(batch)

    bn1 = model1[1]
    assert isinstance(bn1, nn.BatchNorm1d)
    assert not np.allclose(bn1.running_mean, 0.0)
    assert not np.allclose(bn1.running_var, 1.0)

    sd = model1.state_dict()
    assert "1.running_mean" in sd
    assert "1.running_var" in sd
    # Buffers must not be in parameters()
    assert len(bn1.parameters()) == 2

    model2.load_state_dict(sd)
    model1.eval()
    model2.eval()

    x_eval = Tensor(rng.normal(loc=3.0, scale=1.5, size=(6, 4)), requires_grad=False)
    np.testing.assert_allclose(model1(x_eval).data, model2(x_eval).data, atol=1e-12)

    # Also verify affine=False BatchNorm1d save/load eval parity on 3D input
    bn_no_aff1 = nn.BatchNorm1d(3, momentum=0.3, affine=False)
    bn_no_aff2 = nn.BatchNorm1d(3, momentum=0.3, affine=False)
    for _ in range(4):
        _ = bn_no_aff1(Tensor(rng.normal(loc=-2.0, scale=3.0, size=(4, 3, 5))))
    bn_no_aff2.load_state_dict(bn_no_aff1.state_dict())
    bn_no_aff1.eval()
    bn_no_aff2.eval()
    x_3d = Tensor(rng.normal(size=(2, 3, 5)))
    np.testing.assert_allclose(bn_no_aff1(x_3d).data, bn_no_aff2(x_3d).data, atol=1e-12)


@pytest.mark.parametrize("shape", [(6, 1), (4, 1, 5)])
def test_batchnorm1d_num_features_one(shape: tuple[int, ...]) -> None:
    rng = np.random.default_rng(77)
    bn = nn.BatchNorm1d(1, eps=1e-5, momentum=0.2)
    bn.weight.data[...] = np.array([1.7])
    bn.bias.data[...] = np.array([-0.4])

    x_data = rng.normal(loc=4.0, scale=2.0, size=shape)
    x = Tensor(x_data, requires_grad=True)

    out = bn(x)
    assert out.shape == shape
    assert bn.running_mean.shape == (1,)
    assert bn.running_var.shape == (1,)

    # Verify running stats math matches expected channel mean & unbiased variance
    expected_mean = np.mean(x_data)
    n = x_data.size
    expected_unbiased_var = np.var(x_data) * (n / (n - 1))
    np.testing.assert_allclose(bn.running_mean, [0.2 * expected_mean], atol=1e-12)
    np.testing.assert_allclose(bn.running_var, [0.8 + 0.2 * expected_unbiased_var], atol=1e-12)

    loss = (out ** 2).sum()
    loss.backward()
    assert x.grad is not None and x.grad.shape == shape
    assert bn.weight.grad is not None and bn.weight.grad.shape == (1,)
    assert bn.bias.grad is not None and bn.bias.grad.shape == (1,)

    # Save/load eval parity with num_features=1
    bn_copy = nn.BatchNorm1d(1, eps=1e-5, momentum=0.2)
    bn_copy.load_state_dict(bn.state_dict())
    bn.eval()
    bn_copy.eval()

    x_eval = Tensor(rng.normal(loc=2.0, scale=1.0, size=shape), requires_grad=True)
    y1 = bn(x_eval)
    y2 = bn_copy(Tensor(x_eval.data.copy(), requires_grad=False))
    np.testing.assert_allclose(y1.data, y2.data, atol=1e-12)

    # Eval-mode backward with num_features=1
    y1.sum().backward()
    expected_dx = bn.weight.data[0] / np.sqrt(bn.running_var[0] + bn.eps)
    np.testing.assert_allclose(x_eval.grad, np.full(shape, expected_dx), atol=1e-12)


# =========================================================================
# SGD Optimizer
# =========================================================================


def test_sgd_plain_momentum_and_weight_decay() -> None:
    # 1. Plain SGD
    p = Tensor(np.array([1.0, -2.0]), requires_grad=True)
    frozen = Tensor(np.array([5.0, 6.0]), requires_grad=False)
    opt = optim.SGD([p, frozen], lr=0.1)

    p.grad = np.array([2.0, -4.0])
    opt.step()
    np.testing.assert_allclose(p.data, [0.8, -1.6], atol=1e-12)
    np.testing.assert_allclose(frozen.data, [5.0, 6.0], atol=1e-12)

    opt.zero_grad()
    np.testing.assert_allclose(p.grad, [0.0, 0.0])

    # 2. Momentum + weight decay over multiple steps
    p2 = Tensor(np.array([2.0, -1.0]), requires_grad=True)
    opt2 = optim.SGD([p2], lr=0.1, momentum=0.9, weight_decay=0.05)

    p_ref = np.array([2.0, -1.0])
    v_ref = np.zeros_like(p_ref)
    for g in [np.array([1.0, -0.5]), np.array([-0.2, 0.8]), np.array([0.4, 0.3])]:
        p2.grad = g.copy()
        opt2.step()

        g_wd = g + 0.05 * p_ref
        v_ref = 0.9 * v_ref + g_wd
        p_ref = p_ref - 0.1 * v_ref
        np.testing.assert_allclose(p2.data, p_ref, atol=1e-12)

    with pytest.raises(ValueError, match="Invalid learning rate"):
        optim.SGD([p], lr=-0.01)


# =========================================================================
# MSELoss, LeakyReLU, Softmax
# =========================================================================


def test_mse_loss_forward_and_backward() -> None:
    criterion = nn.MSELoss()
    assert repr(criterion) == "MSELoss()"

    pred = Tensor(np.array([[1.0, 3.0], [-2.0, 4.0]]), requires_grad=True)
    target_np = np.array([[1.5, 1.0], [0.0, 2.0]])

    loss = criterion(pred, target_np)
    expected_loss = np.mean((pred.data - target_np) ** 2)
    np.testing.assert_allclose(loss.data, expected_loss, atol=1e-12)

    loss.backward()
    expected_grad = 2.0 * (pred.data - target_np) / pred.data.size
    np.testing.assert_allclose(pred.grad, expected_grad, atol=1e-12)

    # Functional mse_loss with trainable Tensor target
    p_t = Tensor(np.array([2.0, -1.0]), requires_grad=True)
    t_t = Tensor(np.array([0.5, 1.0]), requires_grad=True)
    loss2 = nn.mse_loss(p_t, t_t)
    loss2.backward()
    np.testing.assert_allclose(p_t.grad, 2.0 * (p_t.data - t_t.data) / 2.0, atol=1e-12)
    np.testing.assert_allclose(t_t.grad, -2.0 * (p_t.data - t_t.data) / 2.0, atol=1e-12)


def test_leaky_relu_forward_and_backward() -> None:
    act = nn.LeakyReLU(negative_slope=0.2)
    assert repr(act) == "LeakyReLU(negative_slope=0.2)"

    x = Tensor(np.array([-3.0, -0.5, 0.0, 0.5, 2.0]), requires_grad=True)
    y = act(x)
    np.testing.assert_allclose(y.data, [-0.6, -0.1, 0.0, 0.5, 2.0], atol=1e-12)

    weights = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    (y * weights).sum().backward()
    np.testing.assert_allclose(x.grad, [0.2, 0.4, 0.6, 4.0, 5.0], atol=1e-12)

    # requires_grad=False passthrough
    x_nograd = Tensor(np.array([-1.0, 2.0]), requires_grad=False)
    y_nograd = nn.LeakyReLU()(x_nograd)
    assert not y_nograd.requires_grad
    np.testing.assert_allclose(y_nograd.data, [-0.01, 2.0], atol=1e-12)


def test_softmax_forward_stability_and_backward() -> None:
    # Numerical stability with large shifted logits
    sm = nn.Softmax(dim=-1)
    huge = Tensor(np.array([[1000.0, 1001.0, 999.0]]), requires_grad=True)
    probs = sm(huge)
    assert np.all(np.isfinite(probs.data))
    np.testing.assert_allclose(probs.data.sum(axis=-1), [1.0], atol=1e-12)

    expected = np.exp([0.0, 1.0, -1.0])
    expected /= expected.sum()
    np.testing.assert_allclose(probs.data[0], expected, atol=1e-12)

    # Non-trivial upstream gradient along dim=0 and dim=1
    rng = np.random.default_rng(99)
    for dim in (0, 1, -1):
        x_np = rng.normal(size=(3, 4))
        upstream = rng.normal(size=(3, 4))
        x = Tensor(x_np, requires_grad=True)
        out = nn.Softmax(dim=dim)(x)
        np.testing.assert_allclose(out.data.sum(axis=dim), np.ones(out.data.sum(axis=dim).shape), atol=1e-12)

        (out * upstream).sum().backward()
        dot = np.sum(upstream * out.data, axis=dim, keepdims=True)
        expected_dx = out.data * (upstream - dot)
        np.testing.assert_allclose(x.grad, expected_dx, atol=1e-12)

    # requires_grad=False
    x_ng = Tensor(np.array([1.0, 2.0, 3.0]), requires_grad=False)
    out_ng = nn.Softmax(dim=0)(x_ng)
    assert not out_ng.requires_grad
    np.testing.assert_allclose(out_ng.data.sum(), 1.0, atol=1e-12)


# =========================================================================
# DataLoader & Dataset
# =========================================================================


def test_tensor_dataset_and_dataloader() -> None:
    x_np = np.arange(20, dtype=np.float64).reshape(10, 2)
    y_t = Tensor(np.arange(10, dtype=np.int64), requires_grad=False)

    ds = TensorDataset(x_np, y_t)
    assert len(ds) == 10
    row_x, row_y = ds[3]
    np.testing.assert_array_equal(row_x, x_np[3])
    assert row_y == 3

    # Sequential batching with partial final batch
    loader = DataLoader(ds, batch_size=4, shuffle=False)
    assert len(loader) == 3
    batches = list(loader)
    assert [b[0].shape[0] for b in batches] == [4, 4, 2]
    assert all(not b[0].requires_grad and not b[1].requires_grad for b in batches)
    np.testing.assert_array_equal(np.vstack([b[0].data for b in batches]), x_np)
    np.testing.assert_array_equal(np.concatenate([b[1].data for b in batches]), y_t.data)

    # Shuffled batching visits every sample exactly once
    np.random.seed(1234)
    shuffled_loader = DataLoader(ds, batch_size=3, shuffle=True)
    assert len(shuffled_loader) == 4
    seen_y = np.concatenate([b[1].data for b in shuffled_loader])
    np.testing.assert_array_equal(np.sort(seen_y), np.arange(10))
    assert not np.array_equal(seen_y, np.arange(10))

    # Validation errors
    with pytest.raises(ValueError, match="at least one"):
        TensorDataset()
    with pytest.raises(ValueError, match="same size"):
        TensorDataset(np.zeros((5, 2)), np.zeros((4,)))
    with pytest.raises(ValueError, match="positive integer"):
        DataLoader(ds, batch_size=0)


def test_dataloader_custom_dataset_tuple_and_single() -> None:
    class PairDataset(Dataset):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> tuple[np.ndarray, int]:
            return np.array([float(index), float(index * 2)]), index

    class SingleDataset(Dataset):
        def __len__(self) -> int:
            return 5

        def __getitem__(self, index: int) -> np.ndarray:
            return np.array([float(index), float(index + 1)])

    pair_batches = list(DataLoader(PairDataset(), batch_size=2, shuffle=False))
    assert len(pair_batches) == 3
    assert pair_batches[0][0].shape == (2, 2)
    assert pair_batches[2][0].shape == (1, 2)
    np.testing.assert_array_equal(
        np.concatenate([b[1].data for b in pair_batches]), np.arange(5)
    )

    single_batches = list(DataLoader(SingleDataset(), batch_size=2, shuffle=False))
    assert len(single_batches) == 3
    assert isinstance(single_batches[0], Tensor)
    assert single_batches[0].shape == (2, 2)
    assert single_batches[2].shape == (1, 2)

    base = Dataset()
    with pytest.raises(NotImplementedError):
        len(base)
    with pytest.raises(NotImplementedError):
        _ = base[0]


# =========================================================================
# Iterative Graph Tracing (tensile.utils.vis._trace)
# =========================================================================


def test_vis_trace_deep_graph_and_diamond_dag() -> None:
    # Deep chain well beyond Python's default recursion limit (~1000)
    x = Tensor(np.array(1.0), requires_grad=True)
    y = x
    depth = 5000
    for _ in range(depth):
        y = y + 1.0

    nodes, edges = _trace(y)
    # 1 root leaf `x` + `depth` intermediate sums + `depth` constant `1.0` tensors
    assert len(nodes) == 2 * depth + 1
    # Each addition has 2 incoming edges (prev_y -> cur_y and const_1 -> cur_y)
    assert len(edges) == 2 * depth
    assert x in nodes and y in nodes

    # Shared subexpression (diamond DAG): both paths from `a` must be captured
    a = Tensor(np.array(2.0), requires_grad=True)
    b = a.relu()
    c = -a
    d = b + c
    d_nodes, d_edges = _trace(d)
    assert d_nodes == {a, b, c, d}
    assert d_edges == {(a, b), (a, c), (b, d), (c, d)}


# =========================================================================
# Functional Operators (tensile.ops)
# =========================================================================


def test_ops_all_functions_forward_and_backward() -> None:
    expected_exports = {
        "unbroadcast",
        "add",
        "sub",
        "mul",
        "div",
        "matmul",
        "relu",
        "sigmoid",
        "tanh",
        "gelu",
        "sum",
        "mean",
    }
    assert set(ops.__all__) == expected_exports

    # Binary ops + unbroadcast
    a_np = np.array([[1.2, -0.7], [0.5, 2.0]])
    b_np = np.array([[0.8, 1.5], [1.2, 0.9]])

    for fn, ref_fn in [
        (ops.add, lambda u, v: u + v),
        (ops.sub, lambda u, v: u - v),
        (ops.mul, lambda u, v: u * v),
        (ops.div, lambda u, v: u / v),
        (ops.matmul, lambda u, v: u @ v),
    ]:
        a1, b1 = Tensor(a_np.copy(), requires_grad=True), Tensor(b_np.copy(), requires_grad=True)
        a2, b2 = Tensor(a_np.copy(), requires_grad=True), Tensor(b_np.copy(), requires_grad=True)
        out1 = fn(a1, b1)
        out2 = ref_fn(a2, b2)
        np.testing.assert_allclose(out1.data, out2.data, atol=1e-12)
        out1.sum().backward()
        out2.sum().backward()
        np.testing.assert_allclose(a1.grad, a2.grad, atol=1e-12)
        np.testing.assert_allclose(b1.grad, b2.grad, atol=1e-12)

    # Unary activations
    for fn, method in [
        (ops.relu, "relu"),
        (ops.sigmoid, "sigmoid"),
        (ops.tanh, "tanh"),
        (ops.gelu, "gelu"),
    ]:
        x1 = Tensor(a_np.copy(), requires_grad=True)
        x2 = Tensor(a_np.copy(), requires_grad=True)
        out1 = fn(x1)
        out2 = getattr(x2, method)()
        np.testing.assert_allclose(out1.data, out2.data, atol=1e-12)
        out1.sum().backward()
        out2.sum().backward()
        np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-12)

    # Reductions with axis and keepdims
    x1 = Tensor(a_np.copy(), requires_grad=True)
    x2 = Tensor(a_np.copy(), requires_grad=True)
    s1 = ops.sum(x1, axis=1, keepdims=True)
    s2 = x2.sum(axis=1, keepdims=True)
    np.testing.assert_allclose(s1.data, s2.data, atol=1e-12)
    m1 = ops.mean(s1, axis=0, keepdims=False)
    m2 = s2.mean(axis=0, keepdims=False)
    np.testing.assert_allclose(m1.data, m2.data, atol=1e-12)
    m1.sum().backward()
    m2.sum().backward()
    np.testing.assert_allclose(x1.grad, x2.grad, atol=1e-12)

    # unbroadcast re-export
    g = np.ones((2, 3, 4))
    ub = ops.unbroadcast(g, (1, 4))
    assert ub.shape == (1, 4)
    np.testing.assert_allclose(ub, np.full((1, 4), 6.0))
