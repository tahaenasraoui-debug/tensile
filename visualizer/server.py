# visualizer/server.py
"""
FastAPI backend for the Tensile Interactive Computational Graph Inspector
and 2D Neural Network Decision Boundary Sandbox.

Launch via:
    uvicorn visualizer.server:app --reload
"""

from __future__ import annotations
from pathlib import Path
from typing import Any, Literal
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import tensile as tl
from tensile import Tensor, nn

WEB_DIR = Path(__file__).resolve().parent / "web"

app = FastAPI(title="Tensile Autograd & Neural Sandbox")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------------------------------------------------------------------------
# Synthetic 2D Datasets (Moons, Circles, XOR)
# -------------------------------------------------------------------------
def make_dataset(
    kind: Literal["moons", "circles", "xor"] = "moons",
    n_samples: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_half = n_samples // 2

    if kind == "moons":
        t1 = np.linspace(0, np.pi, n_half)
        t2 = np.linspace(0, np.pi, n_samples - n_half)
        x1 = np.stack([np.cos(t1), np.sin(t1)], axis=1)
        x2 = np.stack([1.0 - np.cos(t2), 0.5 - np.sin(t2)], axis=1)
        X = np.vstack([x1, x2]) + rng.normal(0.0, 0.14, size=(n_samples, 2))
        X = (X - X.mean(axis=0)) / (X.std(axis=0) + 1e-8)
        y = np.hstack([np.zeros(n_half, dtype=np.int64), np.ones(n_samples - n_half, dtype=np.int64)])

    elif kind == "circles":
        angles1 = rng.uniform(0, 2 * np.pi, n_half)
        angles2 = rng.uniform(0, 2 * np.pi, n_samples - n_half)
        r1 = rng.normal(0.45, 0.08, size=n_half)
        r2 = rng.normal(1.15, 0.10, size=n_samples - n_half)
        x1 = np.stack([r1 * np.cos(angles1), r1 * np.sin(angles1)], axis=1)
        x2 = np.stack([r2 * np.cos(angles2), r2 * np.sin(angles2)], axis=1)
        X = np.vstack([x1, x2])
        y = np.hstack([np.zeros(n_half, dtype=np.int64), np.ones(n_samples - n_half, dtype=np.int64)])

    else:  # "xor"
        X = rng.uniform(-1.4, 1.4, size=(n_samples, 2))
        y = ((X[:, 0] * X[:, 1]) < 0).astype(np.int64)

    return X.astype(np.float64), y


# -------------------------------------------------------------------------
# Sandbox Training Session State
# -------------------------------------------------------------------------
class StepRequest(BaseModel):
    dataset: Literal["moons", "circles", "xor"] = "moons"
    activation: Literal["relu", "tanh", "gelu", "sigmoid"] = "tanh"
    hidden_dim: int = Field(default=16, ge=4, le=64)
    optimizer: Literal["adam", "sgd"] = "adam"
    lr: float = Field(default=0.03, gt=0.0, le=1.0)
    steps: int = Field(default=1, ge=1, le=25)
    reset: bool = False


class SandboxSession:
    def __init__(self) -> None:
        self.dataset_name: Literal["moons", "circles", "xor"] = "moons"
        self.activation_name: Literal["relu", "tanh", "gelu", "sigmoid"] = "tanh"
        self.hidden_dim: int = 16
        self.optimizer_name: Literal["adam", "sgd"] = "adam"
        self.lr: float = 0.03
        self.step_count: int = 0
        self.last_loss: float = 0.0
        self.last_acc: float = 0.0
        self.last_graph: dict[str, Any] = {"nodes": [], "edges": [], "backprop_steps": []}
        self.configure()

    def _make_activation(self, name: str) -> nn.Module:
        if name == "relu":
            return nn.ReLU()
        if name == "gelu":
            return nn.GELU()
        if name == "sigmoid":
            return nn.Sigmoid()
        return nn.Tanh()

    def configure(
        self,
        dataset: Literal["moons", "circles", "xor"] = "moons",
        activation: Literal["relu", "tanh", "gelu", "sigmoid"] = "tanh",
        hidden_dim: int = 16,
        optimizer: Literal["adam", "sgd"] = "adam",
        lr: float = 0.03,
    ) -> None:
        np.random.seed(42)
        self.dataset_name = dataset
        self.activation_name = activation
        self.hidden_dim = hidden_dim
        self.optimizer_name = optimizer
        self.lr = lr
        self.step_count = 0

        self.X, self.y = make_dataset(dataset, n_samples=200, seed=42)
        self.fc1 = nn.Linear(2, hidden_dim, name_prefix="L1")
        self.act1 = self._make_activation(activation)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, name_prefix="L2")
        self.act2 = self._make_activation(activation)
        self.fc3 = nn.Linear(hidden_dim, 2, name_prefix="L3")
        self.criterion = nn.CrossEntropyLoss()

        params = [
            self.fc1.weight,
            self.fc1.bias,
            self.fc2.weight,
            self.fc2.bias,
            self.fc3.weight,
            self.fc3.bias,
        ]
        if optimizer == "adam":
            self.opt: tl.optim.Optimizer = tl.optim.Adam(params, lr=lr)
        else:
            self.opt = tl.optim.SGD(params, lr=lr, momentum=0.9)

        # Perform an initial forward+backward pass so the graph is immediately populated
        self._forward_backward_only()

    def _forward_backward_only(self) -> tuple[Tensor, Tensor]:
        self.opt.zero_grad()
        x_t = Tensor(self.X, requires_grad=False, name="Input X")
        h1 = self.act1(self.fc1(x_t))
        h1.name = "Hidden 1"
        h2 = self.act2(self.fc2(h1))
        h2.name = "Hidden 2"
        logits = self.fc3(h2)
        logits.name = "Logits"
        loss = self.criterion(logits, self.y)
        loss.backward()

        preds = np.argmax(logits.data, axis=-1)
        self.last_loss = float(loss.data.item())
        self.last_acc = float(np.mean(preds == self.y) * 100.0)
        self.last_graph = loss.to_graph_dict()
        return logits, loss

    def step(self, num_steps: int = 1) -> dict[str, Any]:
        for _ in range(num_steps):
            self._forward_backward_only()
            self.opt.step()
            self.step_count += 1

        return self.snapshot()

    def snapshot(self, grid_res: int = 42) -> dict[str, Any]:
        x_min, x_max = -2.0, 2.0
        y_min, y_max = -2.0, 2.0
        xx, yy = np.meshgrid(
            np.linspace(x_min, x_max, grid_res),
            np.linspace(y_min, y_max, grid_res),
        )
        grid_pts = np.c_[xx.ravel(), yy.ravel()]

        # Fast forward evaluation over the 2D contour grid
        g_t = Tensor(grid_pts, requires_grad=False)
        logits = self.fc3(self.act2(self.fc2(self.act1(self.fc1(g_t))))).data
        shifted = logits - np.max(logits, axis=-1, keepdims=True)
        exps = np.exp(shifted)
        probs_class1 = (exps[:, 1] / np.sum(exps, axis=-1)).reshape(grid_res, grid_res)

        return {
            "step": self.step_count,
            "loss": round(self.last_loss, 5),
            "accuracy": round(self.last_acc, 2),
            "dataset": self.dataset_name,
            "points": self.X.tolist(),
            "labels": self.y.tolist(),
            "boundary": {
                "x_min": x_min,
                "x_max": x_max,
                "y_min": y_min,
                "y_max": y_max,
                "resolution": grid_res,
                "grid_probs": np.round(probs_class1, 4).tolist(),
            },
            "graph": self.last_graph,
        }


SESSION = SandboxSession()


@app.post("/step")
def run_step(req: StepRequest) -> dict[str, Any]:
    needs_reinit = (
        req.reset
        or req.dataset != SESSION.dataset_name
        or req.activation != SESSION.activation_name
        or req.hidden_dim != SESSION.hidden_dim
        or req.optimizer != SESSION.optimizer_name
    )
    if needs_reinit:
        SESSION.configure(
            dataset=req.dataset,
            activation=req.activation,
            hidden_dim=req.hidden_dim,
            optimizer=req.optimizer,
            lr=req.lr,
        )
        if req.reset:
            return SESSION.snapshot()
    else:
        SESSION.opt.lr = req.lr

    return SESSION.step(num_steps=req.steps)


@app.get("/graph")
def get_graph() -> dict[str, Any]:
    return {
        "step": SESSION.step_count,
        "loss": round(SESSION.last_loss, 5),
        "accuracy": round(SESSION.last_acc, 2),
        "graph": SESSION.last_graph,
    }


@app.get("/")
def serve_index() -> FileResponse:
    return FileResponse(WEB_DIR / "index.html")


if WEB_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(WEB_DIR)), name="static")
