// visualizer/web/app.js
(() => {
  const canvas = document.getElementById("boundaryCanvas");
  const ctx = canvas.getContext("2d");

  const datasetSelect = document.getElementById("datasetSelect");
  const actSelect = document.getElementById("actSelect");
  const hiddenInput = document.getElementById("hiddenInput");
  const optSelect = document.getElementById("optSelect");
  const lrInput = document.getElementById("lrInput");

  const stepBtn = document.getElementById("stepBtn");
  const playTrainBtn = document.getElementById("playTrainBtn");
  const resetBtn = document.getElementById("resetBtn");

  const metricStep = document.getElementById("metricStep");
  const metricLoss = document.getElementById("metricLoss");
  const metricAcc = document.getElementById("metricAcc");
  const metricBackpropOp = document.getElementById("metricBackpropOp");

  const backpropSlider = document.getElementById("backpropSlider");
  const prevBackpropBtn = document.getElementById("prevBackpropBtn");
  const nextBackpropBtn = document.getElementById("nextBackpropBtn");
  const playBackpropBtn = document.getElementById("playBackpropBtn");
  const stepCounterBadge = document.getElementById("stepCounterBadge");
  const scrubberStatus = document.getElementById("scrubberStatus");
  const nodeInspector = document.getElementById("nodeInspector");

  let trainTimer = null;
  let backpropTimer = null;
  let currentGraph = { nodes: [], edges: [], backprop_steps: [] };

  // Initialize Cytoscape DAG view
  const cy = cytoscape({
    container: document.getElementById("cy"),
    style: [
      {
        selector: 'node[type="tensor"]',
        style: {
          shape: "round-rectangle",
          "background-color": "#1f2937",
          "border-width": 2,
          "border-color": "#58a6ff",
          label: "data(displayLabel)",
          color: "#e6edf3",
          "text-valign": "center",
          "text-halign": "center",
          "text-wrap": "wrap",
          "font-size": "10px",
          "font-family": "monospace",
          width: 118,
          height: 50,
        },
      },
      {
        selector: 'node[role="leaf_param"]',
        style: {
          "border-color": "#3fb950",
          "background-color": "#12261e",
        },
      },
      {
        selector: 'node[role="root_loss"]',
        style: {
          "border-color": "#ff7b72",
          "background-color": "#341a1d",
        },
      },
      {
        selector: 'node[type="operator"]',
        style: {
          shape: "ellipse",
          "background-color": "#9e6a03",
          "border-width": 2,
          "border-color": "#d29922",
          label: "data(label)",
          color: "#ffffff",
          "font-weight": "bold",
          "font-size": "10px",
          "text-valign": "center",
          "text-halign": "center",
          width: 68,
          height: 34,
        },
      },
      {
        selector: "edge",
        style: {
          width: 2,
          "line-color": "#484f58",
          "target-arrow-color": "#484f58",
          "target-arrow-shape": "triangle",
          "curve-style": "bezier",
          label: "data(gradLabel)",
          "font-size": "9px",
          color: "#8b949e",
          "text-background-opacity": 0.85,
          "text-background-color": "#0d1117",
          "text-background-padding": "2px",
        },
      },
      {
        selector: ".active-backprop-node",
        style: {
          "border-color": "#ff7b72",
          "border-width": 4,
          "background-color": "#58191c",
        },
      },
      {
        selector: ".active-backprop-edge",
        style: {
          width: 4,
          "line-color": "#ff7b72",
          "target-arrow-color": "#ff7b72",
          color: "#ffb4a9",
        },
      },
    ],
  });

  cy.on("tap", "node", (evt) => {
    const d = evt.target.data();
    if (d.type === "tensor") {
      nodeInspector.innerHTML = `
        <strong>Tensor: ${d.label}</strong> (role: <code>${d.role}</code>, shape: <code>[${d.shape.join(", ")}]</code>)<br/>
        Forward Value — min: <code>${d.stats.min}</code> | max: <code>${d.stats.max}</code> | L2 norm: <code>${d.stats.norm}</code><br/>
        Backward Grad — min: <code>${d.grad_stats.min}</code> | max: <code>${d.grad_stats.max}</code> | <strong>‖grad‖₂: <code>${d.grad_stats.norm}</code></strong>
      `;
    } else {
      nodeInspector.innerHTML = `
        <strong>Operator: ${d.label}</strong> (topo index: <code>${d.topo_index}</code>)<br/>
        Output Adjoint Norm ‖∂L/∂out‖₂: <code>${d.grad_norm}</code>
      `;
    }
  });

  function renderDecisionBoundary(boundary, points, labels) {
    const w = canvas.width;
    const h = canvas.height;
    const res = boundary.resolution;
    const probs = boundary.grid_probs;

    const cellW = w / res;
    const cellH = h / res;

    // 1. Render probability contour heatmap
    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        // Flip Y so y_max is at top of canvas
        const p = probs[res - 1 - r][c];
        const red = Math.round(30 + 210 * p);
        const green = Math.round(90 - 40 * Math.abs(p - 0.5));
        const blue = Math.round(220 * (1 - p) + 35);
        ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, 0.45)`;
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));

        // Highlight decision threshold p ≈ 0.5
        if (Math.abs(p - 0.5) < 0.035) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
          ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

    // 2. Render 2D data points
    const { x_min, x_max, y_min, y_max } = boundary;
    for (let i = 0; i < points.length; i++) {
      const [px, py] = points[i];
      const cx = ((px - x_min) / (x_max - x_min)) * w;
      const cyCoord = h - ((py - y_min) / (y_max - y_min)) * h;

      ctx.beginPath();
      ctx.arc(cx, cyCoord, 4.5, 0, Math.PI * 2);
      ctx.fillStyle = labels[i] === 1 ? "#ff7b72" : "#58a6ff";
      ctx.fill();
      ctx.lineWidth = 1.2;
      ctx.strokeStyle = "#0d1117";
      ctx.stroke();
    }
  }

  function renderGraph(graph) {
    currentGraph = graph;
    const elements = [];

    for (const n of graph.nodes) {
      if (n.type === "tensor") {
        const displayLabel = `${n.label} [${n.shape.join("×")}]\n‖g‖=${n.grad_stats.norm.toExponential(2)}`;
        elements.push({ data: { ...n, displayLabel } });
      } else {
        elements.push({ data: { ...n } });
      }
    }

    for (const e of graph.edges) {
      elements.push({
        data: {
          ...e,
          gradLabel: `‖g‖=${e.grad_norm.toExponential(1)}`,
        },
      });
    }

    cy.elements().remove();
    cy.add(elements);

    cy.layout({
      name: "breadthfirst",
      directed: true,
      padding: 24,
      spacingFactor: 0.85,
    }).run();

    const maxStep = Math.max(0, graph.backprop_steps.length - 1);
    backpropSlider.max = String(maxStep);
    if (parseInt(backpropSlider.value, 10) > maxStep) {
      backpropSlider.value = "-1";
    }
    applyBackpropHighlight(parseInt(backpropSlider.value, 10));
  }

  function applyBackpropHighlight(stepIndex) {
    cy.elements().removeClass("active-backprop-node active-backprop-edge");
    const steps = currentGraph.backprop_steps || [];

    if (stepIndex < 0 || stepIndex >= steps.length) {
      stepCounterBadge.textContent = "Backprop Step: All";
      metricBackpropOp.textContent = "Full Graph";
      scrubberStatus.textContent =
        "Showing full forward/backward graph. Move the slider to scrub step-by-step through reverse-mode autodiff.";
      return;
    }

    const active = steps[stepIndex];
    stepCounterBadge.textContent = `Backprop Step: ${stepIndex + 1} / ${steps.length}`;
    metricBackpropOp.textContent = `${active.op_label} (‖g‖=${active.grad_norm})`;
    scrubberStatus.textContent = `Step ${stepIndex + 1}/${steps.length}: Propagating adjoint through ${active.op_label} -> [${active.parent_tensors.join(", ")}] (‖grad‖₂ = ${active.grad_norm})`;

    cy.getElementById(active.tensor_id).addClass("active-backprop-node");
    cy.getElementById(active.op_id).addClass("active-backprop-node");
    cy.getElementById(active.incoming_edge).addClass("active-backprop-edge");

    for (const edgeId of active.outgoing_edges) {
      cy.getElementById(edgeId).addClass("active-backprop-edge");
    }
    for (const ptId of active.parent_tensors) {
      cy.getElementById(ptId).addClass("active-backprop-node");
    }
  }

  async function postStep(stepsCount = 1, reset = false) {
    const payload = {
      dataset: datasetSelect.value,
      activation: actSelect.value,
      hidden_dim: parseInt(hiddenInput.value, 10) || 16,
      optimizer: optSelect.value,
      lr: parseFloat(lrInput.value) || 0.03,
      steps: stepsCount,
      reset: reset,
    };

    const res = await fetch("/step", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload),
    });
    const data = await res.json();

    metricStep.textContent = data.step;
    metricLoss.textContent = data.loss.toFixed(5);
    metricAcc.textContent = `${data.accuracy.toFixed(2)}%`;

    renderDecisionBoundary(data.boundary, data.points, data.labels);
    renderGraph(data.graph);
  }

  // Event Listeners
  stepBtn.addEventListener("click", () => postStep(1, false));
  resetBtn.addEventListener("click", () => {
    if (trainTimer) {
      clearInterval(trainTimer);
      trainTimer = null;
      playTrainBtn.textContent = "Auto Train";
    }
    postStep(1, true);
  });

  for (const el of [datasetSelect, actSelect, hiddenInput, optSelect]) {
    el.addEventListener("change", () => postStep(1, true));
  }

  playTrainBtn.addEventListener("click", () => {
    if (trainTimer) {
      clearInterval(trainTimer);
      trainTimer = null;
      playTrainBtn.textContent = "Auto Train";
    } else {
      playTrainBtn.textContent = "Pause Train";
      trainTimer = setInterval(() => postStep(2, false), 180);
    }
  });

  backpropSlider.addEventListener("input", (e) => {
    applyBackpropHighlight(parseInt(e.target.value, 10));
  });

  prevBackpropBtn.addEventListener("click", () => {
    const cur = parseInt(backpropSlider.value, 10);
    const nextVal = Math.max(-1, cur - 1);
    backpropSlider.value = String(nextVal);
    applyBackpropHighlight(nextVal);
  });

  nextBackpropBtn.addEventListener("click", () => {
    const steps = currentGraph.backprop_steps || [];
    const cur = parseInt(backpropSlider.value, 10);
    const nextVal = Math.min(steps.length - 1, cur + 1);
    backpropSlider.value = String(nextVal);
    applyBackpropHighlight(nextVal);
  });

  playBackpropBtn.addEventListener("click", () => {
    if (backpropTimer) {
      clearInterval(backpropTimer);
      backpropTimer = null;
      playBackpropBtn.textContent = "▶ Play Backprop";
      return;
    }
    const steps = currentGraph.backprop_steps || [];
    if (!steps.length) return;

    playBackpropBtn.textContent = "⏸ Pause Backprop";
    let idx = parseInt(backpropSlider.value, 10);
    if (idx < 0 || idx >= steps.length - 1) idx = -1;

    backpropTimer = setInterval(() => {
      idx += 1;
      if (idx >= steps.length) {
        clearInterval(backpropTimer);
        backpropTimer = null;
        playBackpropBtn.textContent = "▶ Play Backprop";
        backpropSlider.value = "-1";
        applyBackpropHighlight(-1);
        return;
      }
      backpropSlider.value = String(idx);
      applyBackpropHighlight(idx);
    }, 650);
  });

  // Initial load
  postStep(1, true);
})();
