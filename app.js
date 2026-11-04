// app.js — Hybrid Client: Uses FastAPI /step when running via Uvicorn,
// and seamlessly falls back to an in-browser JS port of Tensile on GitHub Pages!
(() => {
  const canvas = document.getElementById("boundaryCanvas");
  const ctx = canvas.getContext("2d");
  const engineBadge = document.getElementById("engineBadge");

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
  let useLocalEngine = false;

  // =========================================================================
  // In-Browser Tensile Autograd Engine (for GitHub Pages static deployment)
  // =========================================================================
  class PRNG {
    constructor(seed = 42) { this.state = seed >>> 0; }
    next() {
      let t = (this.state += 0x6d2b79f5);
      t = Math.imul(t ^ (t >>> 15), t | 1);
      t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    }
    randn() {
      const u1 = Math.max(1e-12, this.next());
      const u2 = this.next();
      return Math.sqrt(-2.0 * Math.log(u1)) * Math.cos(2.0 * Math.PI * u2);
    }
  }

  function statsOf2D(mat) {
    let min = Infinity, max = -Infinity, sumSq = 0;
    for (let i = 0; i < mat.length; i++) {
      const row = mat[i];
      for (let j = 0; j < row.length; j++) {
        const v = row[j];
        if (v < min) min = v;
        if (v > max) max = v;
        sumSq += v * v;
      }
    }
    if (min === Infinity) return { min: 0, max: 0, norm: 0 };
    return {
      min: Number(min.toFixed(5)),
      max: Number(max.toFixed(5)),
      norm: Number(Math.sqrt(sumSq).toFixed(5)),
    };
  }

  function zeros2D(r, c) {
    return Array.from({ length: r }, () => new Float64Array(c));
  }

  class BrowserSandbox {
    constructor() {
      this.configure("moons", "tanh", 16, "adam", 0.03);
    }

    makeDataset(kind, n = 160) {
      const rng = new PRNG(42);
      const half = n >> 1;
      const X = [];
      const y = [];
      if (kind === "moons") {
        for (let i = 0; i < half; i++) {
          const t = (Math.PI * i) / (half - 1);
          X.push([Math.cos(t) + rng.randn() * 0.14 - 0.5, Math.sin(t) + rng.randn() * 0.14 - 0.25]);
          y.push(0);
        }
        for (let i = 0; i < n - half; i++) {
          const t = (Math.PI * i) / (n - half - 1);
          X.push([1 - Math.cos(t) + rng.randn() * 0.14 - 0.5, 0.5 - Math.sin(t) + rng.randn() * 0.14 - 0.25]);
          y.push(1);
        }
      } else if (kind === "circles") {
        for (let i = 0; i < half; i++) {
          const a = rng.next() * Math.PI * 2;
          const r = 0.48 + rng.randn() * 0.08;
          X.push([r * Math.cos(a), r * Math.sin(a)]);
          y.push(0);
        }
        for (let i = 0; i < n - half; i++) {
          const a = rng.next() * Math.PI * 2;
          const r = 1.25 + rng.randn() * 0.1;
          X.push([r * Math.cos(a), r * Math.sin(a)]);
          y.push(1);
        }
      } else {
        for (let i = 0; i < n; i++) {
          const px = (rng.next() - 0.5) * 2.8;
          const py = (rng.next() - 0.5) * 2.8;
          X.push([px, py]);
          y.push(px * py < 0 ? 1 : 0);
        }
      }
      return { X, y };
    }

    configure(dataset, activation, hiddenDim, optimizer, lr) {
      this.dataset = dataset;
      this.activation = activation;
      this.H = hiddenDim;
      this.optimizer = optimizer;
      this.lr = lr;
      this.stepCount = 0;
      this.t = 0;

      const { X, y } = this.makeDataset(dataset, 160);
      this.X = X;
      this.y = y;

      const rng = new PRNG(42);
      const initW = (r, c) => {
        const scale = Math.sqrt(2.0 / r);
        return Array.from({ length: r }, () =>
          Float64Array.from({ length: c }, () => rng.randn() * scale)
        );
      };

      this.W1 = initW(2, this.H);
      this.b1 = [new Float64Array(this.H)];
      this.W2 = initW(this.H, this.H);
      this.b2 = [new Float64Array(this.H)];
      this.W3 = initW(this.H, 2);
      this.b3 = [new Float64Array(2)];

      this.params = [
        { name: "L1.W", data: this.W1, shape: [2, this.H] },
        { name: "L1.b", data: this.b1, shape: [this.H] },
        { name: "L2.W", data: this.W2, shape: [this.H, this.H] },
        { name: "L2.b", data: this.b2, shape: [this.H] },
        { name: "L3.W", data: this.W3, shape: [this.H, 2] },
        { name: "L3.b", data: this.b3, shape: [2] },
      ];

      for (const p of this.params) {
        p.grad = zeros2D(p.data.length, p.data[0].length);
        p.m = zeros2D(p.data.length, p.data[0].length);
        p.v = zeros2D(p.data.length, p.data[0].length);
      }

      this.forwardBackward();
    }

    actForward(Z) {
      const N = Z.length, C = Z[0].length;
      const A = zeros2D(N, C);
      const dA = zeros2D(N, C);
      const c = Math.sqrt(2.0 / Math.PI);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < C; j++) {
          const x = Z[i][j];
          if (this.activation === "relu") {
            A[i][j] = x > 0 ? x : 0;
            dA[i][j] = x > 0 ? 1 : 0;
          } else if (this.activation === "sigmoid") {
            const s = 1.0 / (1.0 + Math.exp(-x));
            A[i][j] = s;
            dA[i][j] = s * (1.0 - s);
          } else if (this.activation === "gelu") {
            const inner = c * (x + 0.044715 * x * x * x);
            const th = Math.tanh(inner);
            A[i][j] = 0.5 * x * (1.0 + th);
            dA[i][j] = 0.5 * (1.0 + th) + 0.5 * x * (1.0 - th * th) * c * (1.0 + 3 * 0.044715 * x * x);
          } else {
            const th = Math.tanh(x);
            A[i][j] = th;
            dA[i][j] = 1.0 - th * th;
          }
        }
      }
      return { A, dA };
    }

    linearForward(X, W, b) {
      const N = X.length, inD = W.length, outD = W[0].length;
      const mm = zeros2D(N, outD);
      const out = zeros2D(N, outD);
      for (let i = 0; i < N; i++) {
        for (let k = 0; k < inD; k++) {
          const xik = X[i][k];
          for (let j = 0; j < outD; j++) mm[i][j] += xik * W[k][j];
        }
        for (let j = 0; j < outD; j++) out[i][j] = mm[i][j] + b[0][j];
      }
      return { mm, out };
    }

    forwardBackward() {
      const N = this.X.length;
      const H = this.H;

      const { mm: mm1, out: z1 } = this.linearForward(this.X, this.W1, this.b1);
      const { A: h1, dA: dh1Mask } = this.actForward(z1);

      const { mm: mm2, out: z2 } = this.linearForward(h1, this.W2, this.b2);
      const { A: h2, dA: dh2Mask } = this.actForward(z2);

      const { mm: mm3, out: logits } = this.linearForward(h2, this.W3, this.b3);

      // Log-Sum-Exp CrossEntropyLoss & Softmax
      const dLogits = zeros2D(N, 2);
      let lossSum = 0;
      let correct = 0;
      for (let i = 0; i < N; i++) {
        const l0 = logits[i][0], l1 = logits[i][1];
        const m = Math.max(l0, l1);
        const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
        const sumE = e0 + e1;
        const logSumExp = m + Math.log(sumE);
        const yi = this.y[i];
        lossSum += logSumExp - logits[i][yi];
        if ((l1 > l0 ? 1 : 0) === yi) correct++;
        dLogits[i][0] = (e0 / sumE - (yi === 0 ? 1 : 0)) / N;
        dLogits[i][1] = (e1 / sumE - (yi === 1 ? 1 : 0)) / N;
      }
      this.lastLoss = lossSum / N;
      this.lastAcc = (correct / N) * 100.0;

      // Backward L3
      const dW3 = zeros2D(H, 2);
      const db3 = zeros2D(1, 2);
      const dh2 = zeros2D(N, H);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < 2; j++) {
          const g = dLogits[i][j];
          db3[0][j] += g;
          for (let k = 0; k < H; k++) {
            dW3[k][j] += h2[i][k] * g;
            dh2[i][k] += g * this.W3[k][j];
          }
        }
      }

      // Backward Act2 & L2
      const dz2 = zeros2D(N, H);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < H; j++) dz2[i][j] = dh2[i][j] * dh2Mask[i][j];

      const dW2 = zeros2D(H, H);
      const db2 = zeros2D(1, H);
      const dh1 = zeros2D(N, H);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < H; j++) {
          const g = dz2[i][j];
          db2[0][j] += g;
          for (let k = 0; k < H; k++) {
            dW2[k][j] += h1[i][k] * g;
            dh1[i][k] += g * this.W2[k][j];
          }
        }
      }

      // Backward Act1 & L1
      const dz1 = zeros2D(N, H);
      for (let i = 0; i < N; i++)
        for (let j = 0; j < H; j++) dz1[i][j] = dh1[i][j] * dh1Mask[i][j];

      const dW1 = zeros2D(2, H);
      const db1 = zeros2D(1, H);
      for (let i = 0; i < N; i++) {
        for (let j = 0; j < H; j++) {
          const g = dz1[i][j];
          db1[0][j] += g;
          dW1[0][j] += this.X[i][0] * g;
          dW1[1][j] += this.X[i][1] * g;
        }
      }

      const grads = [dW1, db1, dW2, db2, dW3, db3];
      for (let pIdx = 0; pIdx < this.params.length; pIdx++) {
        this.params[pIdx].grad = grads[pIdx];
      }

      this.lastGraph = this.buildGraphDict({
        mm1, z1, dz1, h1, dh1, mm2, z2, dz2, h2, dh2, mm3, logits, dLogits
      });
    }

    buildGraphDict(t) {
      const N = this.X.length, H = this.H;
      const actLabel = this.activation.toUpperCase();
      const nodes = [];
      const edges = [];
      const backpropSteps = [];

      const addTensor = (id, label, role, shape, dataMat, gradMat, op, topo) => {
        const st = statsOf2D(dataMat);
        const gst = statsOf2D(gradMat);
        nodes.push({
          id, type: "tensor", role, label, op, shape,
          requires_grad: role !== "input_data", topo_index: topo,
          stats: st, grad_stats: gst
        });
        return gst.norm;
      };

      const zGrad = zeros2D(1, 1);
      addTensor("tensor_x", "Input X", "input_data", [N, 2], this.X, zGrad, "", 0);
      addTensor("tensor_w1", "L1.W", "leaf_param", [2, H], this.W1, this.params[0].grad, "", 1);
      addTensor("tensor_b1", "L1.b", "leaf_param", [H], this.b1, this.params[1].grad, "", 2);
      addTensor("tensor_w2", "L2.W", "leaf_param", [H, H], this.W2, this.params[2].grad, "", 3);
      addTensor("tensor_b2", "L2.b", "leaf_param", [H], this.b2, this.params[3].grad, "", 4);
      addTensor("tensor_w3", "L3.W", "leaf_param", [H, 2], this.W3, this.params[4].grad, "", 5);
      addTensor("tensor_b3", "L3.b", "leaf_param", [2], this.b3, this.params[5].grad, "", 6);

      const stages = [
        { id: "mm1", label: "T_mm1", op: "MatMul", shape: [N, H], d: t.mm1, g: t.dz1, parents: ["tensor_x", "tensor_w1"] },
        { id: "z1", label: "T_z1", op: "Add", shape: [N, H], d: t.z1, g: t.dz1, parents: ["tensor_mm1", "tensor_b1"] },
        { id: "h1", label: "Hidden 1", op: actLabel, shape: [N, H], d: t.h1, g: t.dh1, parents: ["tensor_z1"] },
        { id: "mm2", label: "T_mm2", op: "MatMul", shape: [N, H], d: t.mm2, g: t.dz2, parents: ["tensor_h1", "tensor_w2"] },
        { id: "z2", label: "T_z2", op: "Add", shape: [N, H], d: t.z2, g: t.dz2, parents: ["tensor_mm2", "tensor_b2"] },
        { id: "h2", label: "Hidden 2", op: actLabel, shape: [N, H], d: t.h2, g: t.dh2, parents: ["tensor_z2"] },
        { id: "mm3", label: "T_mm3", op: "MatMul", shape: [N, 2], d: t.mm3, g: t.dLogits, parents: ["tensor_h2", "tensor_w3"] },
        { id: "logits", label: "Logits", op: "Add", shape: [N, 2], d: t.logits, g: t.dLogits, parents: ["tensor_mm3", "tensor_b3"] },
        { id: "loss", label: "Loss", op: "CrossEntropyLoss", shape: [1], d: [[this.lastLoss]], g: [[1.0]], parents: ["tensor_logits"], role: "root_loss" },
      ];

      stages.forEach((s, idx) => {
        const tId = `tensor_${s.id}`;
        const opId = `op_${s.id}`;
        const topo = 7 + idx;
        const gNorm = addTensor(tId, s.label, s.role || "intermediate", s.shape, s.d, s.g, s.op, topo);

        nodes.push({ id: opId, type: "operator", role: "operator", label: s.op, topo_index: topo, grad_norm: gNorm });
        const inEdge = `e_${opId}_to_${tId}`;
        edges.push({ id: inEdge, source: opId, target: tId, grad_norm: gNorm });

        const outEdges = [];
        for (const pId of s.parents) {
          const pNode = nodes.find((n) => n.id === pId);
          const pNorm = pNode ? pNode.grad_stats.norm : gNorm;
          const eId = `e_${pId}_to_${opId}`;
          edges.push({ id: eId, source: pId, target: opId, grad_norm: pNorm });
          outEdges.push(eId);
        }

        backpropSteps.push({
          tensor_id: tId, op_id: opId, op_label: s.op,
          incoming_edge: inEdge, outgoing_edges: outEdges,
          parent_tensors: s.parents, grad_norm: gNorm
        });
      });

      backpropSteps.reverse();
      backpropSteps.forEach((st, i) => (st.step_index = i));
      return { nodes, edges, backprop_steps: backpropSteps };
    }

    step(stepsCount = 1) {
      for (let s = 0; s < stepsCount; s++) {
        this.forwardBackward();
        this.t += 1;
        this.stepCount += 1;
        const b1Corr = 1.0 - Math.pow(0.9, this.t);
        const b2Corr = 1.0 - Math.pow(0.999, this.t);

        for (const p of this.params) {
          const R = p.data.length, C = p.data[0].length;
          for (let r = 0; r < R; r++) {
            for (let c = 0; c < C; c++) {
              const g = p.grad[r][c];
              if (this.optimizer === "adam") {
                p.m[r][c] = 0.9 * p.m[r][c] + 0.1 * g;
                p.v[r][c] = 0.999 * p.v[r][c] + 0.001 * g * g;
                const mHat = p.m[r][c] / b1Corr;
                const vHat = p.v[r][c] / b2Corr;
                p.data[r][c] -= (this.lr * mHat) / (Math.sqrt(vHat) + 1e-8);
              } else {
                p.m[r][c] = 0.9 * p.m[r][c] + g;
                p.data[r][c] -= this.lr * p.m[r][c];
              }
            }
          }
        }
      }
      return this.snapshot();
    }

    snapshot(res = 38) {
      const xMin = -2.0, xMax = 2.0, yMin = -2.0, yMax = 2.0;
      const pts = [];
      for (let r = 0; r < res; r++) {
        const py = yMin + ((yMax - yMin) * r) / (res - 1);
        for (let c = 0; c < res; c++) {
          const px = xMin + ((xMax - xMin) * c) / (res - 1);
          pts.push([px, py]);
        }
      }
      const { out: z1 } = this.linearForward(pts, this.W1, this.b1);
      const { A: h1 } = this.actForward(z1);
      const { out: z2 } = this.linearForward(h1, this.W2, this.b2);
      const { A: h2 } = this.actForward(z2);
      const { out: logits } = this.linearForward(h2, this.W3, this.b3);

      const gridProbs = [];
      let idx = 0;
      for (let r = 0; r < res; r++) {
        const row = [];
        for (let c = 0; c < res; c++) {
          const l0 = logits[idx][0], l1 = logits[idx][1];
          const m = Math.max(l0, l1);
          const e0 = Math.exp(l0 - m), e1 = Math.exp(l1 - m);
          row.push(e1 / (e0 + e1));
          idx++;
        }
        gridProbs.push(row);
      }

      return {
        step: this.stepCount,
        loss: this.lastLoss,
        accuracy: this.lastAcc,
        dataset: this.dataset,
        points: this.X,
        labels: this.y,
        boundary: { x_min: xMin, x_max: xMax, y_min: yMin, y_max: yMax, resolution: res, grid_probs: gridProbs },
        graph: this.lastGraph,
      };
    }
  }

  const localSandbox = new BrowserSandbox();

  // =========================================================================
  // Cytoscape & Canvas Rendering
  // =========================================================================
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
        style: { "border-color": "#3fb950", "background-color": "#12261e" },
      },
      {
        selector: 'node[role="root_loss"]',
        style: { "border-color": "#ff7b72", "background-color": "#341a1d" },
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
        style: { "border-color": "#ff7b72", "border-width": 4, "background-color": "#58191c" },
      },
      {
        selector: ".active-backprop-edge",
        style: { width: 4, "line-color": "#ff7b72", "target-arrow-color": "#ff7b72", color: "#ffb4a9" },
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

    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        const p = probs[res - 1 - r][c];
        const red = Math.round(30 + 210 * p);
        const green = Math.round(90 - 40 * Math.abs(p - 0.5));
        const blue = Math.round(220 * (1 - p) + 35);
        ctx.fillStyle = `rgba(${red}, ${green}, ${blue}, 0.45)`;
        ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));

        if (Math.abs(p - 0.5) < 0.035) {
          ctx.fillStyle = "rgba(255, 255, 255, 0.35)";
          ctx.fillRect(c * cellW, r * cellH, Math.ceil(cellW), Math.ceil(cellH));
        }
      }
    }

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
        data: { ...e, gradLabel: `‖g‖=${e.grad_norm.toExponential(1)}` },
      });
    }

    cy.elements().remove();
    cy.add(elements);
    cy.layout({ name: "breadthfirst", directed: true, padding: 24, spacingFactor: 0.85 }).run();

    const maxStep = Math.max(0, graph.backprop_steps.length - 1);
    backpropSlider.max = String(maxStep);
    if (parseInt(backpropSlider.value, 10) > maxStep) backpropSlider.value = "-1";
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

    for (const edgeId of active.outgoing_edges) cy.getElementById(edgeId).addClass("active-backprop-edge");
    for (const ptId of active.parent_tensors) cy.getElementById(ptId).addClass("active-backprop-node");
  }

  async function postStep(stepsCount = 1, reset = false) {
    const dataset = datasetSelect.value;
    const activation = actSelect.value;
    const hiddenDim = parseInt(hiddenInput.value, 10) || 16;
    const optimizer = optSelect.value;
    const lr = parseFloat(lrInput.value) || 0.03;

    let data = null;

    if (!useLocalEngine) {
      try {
        const res = await fetch("/step", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ dataset, activation, hidden_dim: hiddenDim, optimizer, lr, steps: stepsCount, reset }),
        });
        if (!res.ok) throw new Error("FastAPI endpoint unavailable");
        data = await res.json();
        engineBadge.textContent = "Engine: Python FastAPI";
      } catch (_err) {
        useLocalEngine = true;
      }
    }

    if (useLocalEngine) {
      engineBadge.textContent = "Engine: In-Browser Autograd (GitHub Pages)";
      if (
        reset ||
        dataset !== localSandbox.dataset ||
        activation !== localSandbox.activation ||
        hiddenDim !== localSandbox.H ||
        optimizer !== localSandbox.optimizer
      ) {
        localSandbox.configure(dataset, activation, hiddenDim, optimizer, lr);
        data = reset ? localSandbox.snapshot() : localSandbox.step(stepsCount);
      } else {
        localSandbox.lr = lr;
        data = localSandbox.step(stepsCount);
      }
    }

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

  backpropSlider.addEventListener("input", (e) => applyBackpropHighlight(parseInt(e.target.value, 10)));

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

  postStep(1, true);
})();
