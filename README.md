
## Tensile

**Tensile** is a clean, readable, and educational reverse-mode automatic differentiation and deep learning framework built entirely from scratch using **pure NumPy** (Python 3.10+). It features zero external heavy ML frameworks (no PyTorch, TensorFlow, or JAX), offering complete algorithmic transparency into how tensors, autograd engines, neural network layers, optimizers, and dataloaders work under the hood.

---

## Repository Structure

```text
tensile/
├── tensile/
│   ├── __init__.py           # Top-level package exports
│   ├── tensor.py             # Core Tensor class & reverse-mode autograd engine
│   ├── nn/
│   │   ├── __init__.py       # Neural network module exports
│   │   ├── module.py         # Base Module & parameter collector
│   │   ├── layers.py         # Linear, ReLU, Softmax, Sequential layers
│   │   └── losses.py         # MSE loss & numerically stable cross-entropy
│   ├── optim/
│   │   ├── __init__.py       # Optimizer exports
│   │   └── optimizer.py      # SGD (with momentum) & Adam optimizers
│   ├── data/
│   │   ├── __init__.py       # Data pipeline exports
│   │   └── dataloader.py     # Dataset, TensorDataset, & DataLoader batching
│   └── utils/
│       ├── __init__.py       # Utility exports
│       └── vis.py            # Graphviz computation graph visualization
├── tests/
│   └── test_gradcheck.py     # Finite-difference numerical gradient verification suite
├── examples/
│   └── train_mnist_synthetic.py  # End-to-end multi-class classification demo
├── pyproject.toml            # Project build configuration & metadata
└── README.md

```

---

## Getting Started & Installation

### 1. Clone the Repository

```bash
git clone [https://github.com/tahaenasraoui-debug/tensile.git](https://github.com/tahaenasraoui-debug/tensile.git)
cd tensile

```

### 2. Set Up a Virtual Environment (Arch Linux / PEP 668 Compliant)

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

```

### 3. Install in Editable Mode

```bash
pip install -e ".[dev]"

```

---

## Running Tests & Examples

### Run Numerical Gradient Checks

Verify analytical backward pass gradients against central finite-difference approximations ($\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$):

```bash
pytest tests/test_gradcheck.py
# Or run directly via Python:
python tests/test_gradcheck.py

```

### Run Synthetic MNIST Classification Benchmark

Train a multi-layer perceptron (`Linear -> ReLU -> Linear -> CrossEntropy`) on a synthetic 10-class high-dimensional classification task:

```bash
python examples/train_mnist_synthetic.py

```

---

## Core Architecture Highlights

* **Reverse-Mode Autograd (`Tensor`):** Implements dynamic computational graph tracing via `_prev` pointers and Depth-First Search (DFS) topological sorting during `.backward()`.
* **Robust Unbroadcasting (`unbroadcast`):** Automatically handles reduction and summation across prepended and broadcasted dimensions during elementwise operations and matrix multiplications.
* **Numerically Stable Losses:** Computes log-softmax cross-entropy using the log-sum-exp stabilization trick to prevent numerical underflow/overflow.
* **Adaptive Optimizers:** Implements fully vectorized bias-corrected moment updates for Adam and momentum-buffered updates for SGD.

```

```
