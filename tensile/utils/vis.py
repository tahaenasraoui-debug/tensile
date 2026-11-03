"""
Computational graph visualization utility using Graphviz.
"""

from __future__ import annotations
from typing import TYPE_CHECKING
import numpy as np

if TYPE_CHECKING:
    import graphviz
    from tensile.tensor import Tensor


def _trace(root: Tensor) -> tuple[set[Tensor], set[tuple[Tensor, Tensor]]]:
    """Traverses `_prev` references from `root` to collect all nodes and directed edges."""
    nodes: set[Tensor] = set()
    edges: set[tuple[Tensor, Tensor]] = set()

    def build(v: Tensor) -> None:
        if v not in nodes:
            nodes.add(v)
            for child in v._prev:
                edges.add((child, v))
                build(child)

    build(root)
    return nodes, edges


def draw_graph(
    root_tensor: Tensor,
    format: str = "svg",
    rankdir: str = "LR",
) -> graphviz.Digraph:
    """
    Builds and returns a Graphviz `Digraph` representing the forward/backward
    computation graph rooted at `root_tensor`.
    """
    try:
        from graphviz import Digraph
    except ImportError as exc:
        raise ImportError(
            "The `graphviz` package is required for draw_graph(). "
            "Install it via `pip install graphviz`."
        ) from exc

    dot = Digraph(format=format, graph_attr={"rankdir": rankdir})
    nodes, edges = _trace(root_tensor)

    for n in nodes:
        uid = str(id(n))
        grad_info = (
            f"grad_norm={np.linalg.norm(n.grad):.3e}"
            if (n.requires_grad and n.grad is not None)
            else f"req_grad={n.requires_grad}"
        )
        label = f"{{ shape={n.shape} | {grad_info} }}"
        fill_color = "#e8f4f8" if n.requires_grad else "#f4f4f4"

        dot.node(
            name=uid,
            label=label,
            shape="record",
            style="filled",
            fillcolor=fill_color,
        )

        if n._op:
            op_uid = f"{uid}_{n._op}"
            dot.node(
                name=op_uid,
                label=n._op,
                shape="oval",
                style="filled",
                fillcolor="#ffe5b4",
            )
            dot.edge(op_uid, uid)

    for n1, n2 in edges:
        dot.edge(str(id(n1)), f"{id(n2)}_{n2._op}")

    return dot
