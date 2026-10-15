<h1 align="center">Tensile</h1>

<p align="center">
  A deep learning framework built from scratch in pure NumPy.
</p>

<p align="center">
  <img alt="Status" src="https://img.shields.io/badge/status-just%20starting-orange">
  <img alt="Python" src="https://img.shields.io/badge/python-3.10%2B-blue">
  <img alt="Backend" src="https://img.shields.io/badge/backend-NumPy-013243">
  <img alt="License" src="https://img.shields.io/badge/license-MIT-green">
</p>

---

## Status

This project has just started. There is no working code yet. The repo exists to document the goal while I build it.

## Goal

A small, readable framework where every piece is written from first principles, with no PyTorch or TensorFlow underneath. The whole codebase should be short enough to read in an afternoon.

## Target API

This is what I want the finished framework to feel like. None of it works yet.

```python
import tensile as tl
import tensile.nn as nn

model = nn.Sequential(
    nn.Linear(784, 128),
    nn.ReLU(),
    nn.Linear(128, 10),
)

optimizer = tl.optim.Adam(model.parameters(), lr=1e-3)

for x, y in loader:
    loss = nn.cross_entropy(model(x), y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

## Scope

| Component | Contents | Priority |
|-----------|----------|----------|
| Autograd | Tensor class, reverse-mode backprop, broadcasting | Core |
| Layers | Linear, ReLU, Softmax, Sequential | Core |
| Losses | Cross-entropy, MSE | Core |
| Optimizers | SGD, Adam | Core |
| Data | Minimal dataset and loader | Core |
| Testing | Numerical gradient checks, pytest | Core |
| Convolutions | Conv2d, pooling | Later |
| Regularization | BatchNorm, Dropout | Later |
| Sequence models | RNN, attention | Maybe |

## Project structure

Planned layout, subject to change:

```
tensile/
    tensor.py       # Tensor class and autograd engine
    nn/             # Modules, layers, losses
    optim/          # SGD, Adam
    data/           # Dataset and loader
tests/              # Gradient checks and unit tests
examples/           # MNIST training script
```

## License

MIT
