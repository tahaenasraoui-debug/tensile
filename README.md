# Tensile

[![Live Demo](https://img.shields.io/badge/Live_Demo-GitHub_Pages-2ea44f?style=for-the-badge&logo=github)](https://tahaenasraoui-debug.github.io/tensile/)

**Tensile** is a clean, readable, and production-grade reverse-mode automatic differentiation engine and deep learning framework built entirely from scratch using **pure NumPy** (Python 3.11+). Paired with an interactive **Computational Graph (DAG) Inspector & 2D Neural Network Sandbox**, it offers complete algorithmic transparency into how tensors, reverse-mode autograd, broadcasting adjoint reduction, neural network layers, optimizers, and backpropagation work under the hood.

🌐 **Live Interactive Web Sandbox:** **[https://tahaenasraoui-debug.github.io/tensile/](https://tahaenasraoui-debug.github.io/tensile/)**

---

## Repository Structure

```text
tensile/
├── index.html                    # GitHub Pages live interactive sandbox entrypoint
├── app.js                        # Hybrid client (FastAPI + in-browser Autograd fallback)
├── tensile/
│   ├── __init__.py               # Top-level package exports
│   ├── tensor.py                 # Core Tensor class, reverse-mode DAG autograd & to_graph_dict()
│   ├── ops.py                    # Functional operator interface (add, matmul, relu, gelu, etc.)
│   ├── nn/
│   │   ├── __init__.py           # Neural network module exports
│   │   ├── module.py             # Base Module & recursive parameter collector
│   │   ├── layers.py             # Linear, ReLU, Sigmoid, Tanh, GELU, Softmax, Sequential
│   │   └── losses.py             # MSELoss & numerically stable log-sum-exp CrossEntropyLoss
│   ├── optim/
│   │   ├── __init__.py           # Optimizer exports
│   │   └── optimizer.py          # SGD (with momentum) & Adam optimizers
│   ├── data/
│   │   ├── __init__.py           # Data pipeline exports
│   │   └── dataloader.py         # Dataset, TensorDataset, & DataLoader batching
│   └── utils/
│       ├── __init__.py           # Utility exports
│       └── vis.py                # Graphviz computation graph visualization
├── visualizer/
│   ├── server.py                 # FastAPI backend (/step, /graph) & 2D dataset generator
│   └── web/
│       ├── index.html            # Local FastAPI dashboard UI
│       └── app.js                # Real-time canvas contour renderer & backprop playback scrubber
├── tests/
│   ├── test_gradcheck.py         # Finite-difference numerical gradient checker (rel error < 1e-5)
│   └── test_pytorch_parity.py    # Bit-for-bit forward/backward parity checks against PyTorch
├── examples/
│   └── train_mnist_synthetic.py  # End-to-end 10-class MLP classification benchmark
├── pyproject.toml                # Project build configuration & dependencies
└── README.md
```

---

## Download & Installation

### 1. Clone / Download the Repository
```bash
git clone https://github.com/tahaenasraoui-debug/tensile.git
cd tensile
```

*(Alternatively, download the ZIP archive directly from [https://github.com/tahaenasraoui-debug/tensile/archive/refs/heads/main.zip](https://github.com/tahaenasraoui-debug/tensile/archive/refs/heads/main.zip).)*

### 2. Set Up a Virtual Environment (Arch Linux / PEP 668 Compliant)
```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
```

### 3. Install Tensile & Dependencies
```bash
pip install -e ".[dev]"
pip install fastapi uvicorn torch pytest
```

---

## Running Tests & Verification Suites

### 1. Finite-Difference Numerical Gradient Checks (`< 1e-5` Relative Error)
Verifies analytical `.backward()` gradients against central finite-difference approximations $\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$ across elementwise arithmetic, broadcasting, batched N-D `matmul`, activations (`ReLU`, `Sigmoid`, `Tanh`, `GELU`), reductions, and `CrossEntropyLoss`:
```bash
pytest tests/test_gradcheck.py -v
# Or run directly via Python:
python tests/test_gradcheck.py
```

### 2. PyTorch Parity Verification Harness (`atol=1e-6`)
Verifies that Tensile's forward activations, log-sum-exp cross-entropy loss, analytical gradients, and multi-step `Adam` updates match `torch.Tensor` within `1e-6`:
```bash
pytest tests/test_pytorch_parity.py -v
```

### 3. Run All Tests Together
```bash
pytest -v
```

---

## Interactive Web UI (DAG Inspector & 2D Sandbox)

### Option A: Instant Live Web Demo (GitHub Pages)
Open **[https://tahaenasraoui-debug.github.io/tensile/](https://tahaenasraoui-debug.github.io/tensile/)** in any browser to train 2D neural networks on Moons, Circles, and XOR and scrub step-by-step through reverse-mode backpropagation without installing anything.

### Option B: Run Local FastAPI + Python NumPy Server
Launch the local FastAPI + Uvicorn backend:
```bash
uvicorn visualizer.server:app --reload
```
Then open:
- **Local Dashboard:** [http://127.0.0.1:8000](http://127.0.0.1:8000)
- **Interactive Swagger API Docs:** [http://127.0.0.1:8000/docs](http://127.0.0.1:8000/docs)
- **Live Graph JSON Endpoint:** [http://127.0.0.1:8000/graph](http://127.0.0.1:8000/graph)

---

## Running CLI Examples

### Synthetic MNIST Classification Benchmark
Train a multi-layer perceptron (`Linear(784, 128) -> ReLU -> Linear(128, 10)`) with `Adam` on a synthetic 10-class classification dataset:
```bash
python examples/train_mnist_synthetic.py
```

---

## Core Architecture Highlights

- **Reverse-Mode Autograd (`Tensor`):** Tracks dynamic computational graphs via `_prev` / `_parents` pointers and executes Depth-First Search (DFS) topological sorting during `.backward()`.
- **Broadcasting Invariance (`unbroadcast`):** Automatically contracts prepended higher-rank dimensions and sums across singleton broadcast axes (`keepdims=True`) so gradient shapes strictly match operand shapes across elementwise ops and N-D batched `matmul`.
- **Rich Non-Linearities:** Built-in analytical forward and backward passes for `ReLU`, numerically clipped `Sigmoid`, `Tanh`, `Softmax`, and `GELU` (tanh approximation).
- **Numerically Stable `CrossEntropyLoss`:** Computes categorical cross-entropy directly via the log-sum-exp trick:
  $$\mathcal{L} = \frac{1}{N} \sum_{i=1}^N \left( \log \sum_{j} e^{z_{i, j} - m_i} + m_i - z_{i, y_i} \right)$$
  with exact $\frac{1}{N}(\text{softmax}(Z) - Y)$ backward propagation.
- **DAG Serialization (`to_graph_dict`):** Exports node IDs, operator labels, tensor shapes, forward value summaries (`min`, `max`, L2 `norm`), gradient norms, and reverse topological playback steps directly to JSON.
