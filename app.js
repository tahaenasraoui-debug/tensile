<<<<<<< HEAD
// app.js — 3Blue1Brown / Manim Edition Visual Autograd & NumberPlane Renderer
(() => {
  const bCanvas = document.getElementById("boundaryCanvas");
  const bCtx = bCanvas.getContext("2d");
  const dCanvas = document.getElementById("dagCanvas");
  const dCtx = dCanvas.getContext("2d");

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
  const inspectorLatex = document.getElementById("inspectorLatex");
  const inspectorStats = document.getElementById("inspectorStats");
  const headerFormula = document.getElementById("headerFormula");

  // Official 3Blue1Brown / Manim Color Palette
  const MANIM = {
    BG: "#000000",
    GRID_MAIN: "#1F3B4D",
    GRID_SUB: "#0E1A24",
    AXES: "#ECECEC",
    BLUE: "#58C4DD",
    BLUE_DARK: "#236B8E",
    TEAL: "#5CD0B3",
    GREEN: "#83C167",
    YELLOW: "#FFFF00",
    GOLD: "#F0AC5F",
    RED: "#FC6255",
    MAROON: "#C55F73",
    PURPLE: "#9A72AC",
    WHITE: "#ECECEC",
    MUTED: "#8E9BAE",
  };

  let trainTimer = null;
  let backpropTimer = null;
  let useLocalEngine = false;
  let currentData = null;
  let currentGraph = { nodes: [], edges: [], backprop_steps: [] };
  let activeBackpropStep = -1;
  let selectedNodeId = null;
  let hoveredNodeId = null;
  let animTime = 0;

  // Layout map for custom Manim DAG Canvas nodes
  let dagNodePositions = {};
  let dagEdges = [];

  // Render KaTeX helper
  function renderMath(el, latexStr) {
    if (!el) return;
    if (window.katex) {
      try {
        window.katex.render(latexStr, el, { throwOnError: false, displayMode: false });
        return;
      } catch (_e) {}
    }
    el.textContent = latexStr;
  }

  // =========================================================================
  // In-Browser Tensile Autograd Engine (for GitHub Pages & instant 60fps UI)
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
      min: Number(min.toFixed(4)),
      max: Number(max.toFixed(4)),
      norm: Number(Math.sqrt(sumSq).toFixed(4)),
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
          X.push([Math.cos(t) + rng.randn() * 0.13 - 0.5, Math.sin(t) + rng.randn() * 0.13 - 0.25]);
          y.push(0);
        }
        for (let i = 0; i < n - half; i++) {
          const t = (Math.PI * i) / (n - half - 1);
          X.push([1 - Math.cos(t) + rng.randn() * 0.13 - 0.5, 0.5 - Math.sin(t) + rng.randn() * 0.13 - 0.25]);
          y.push(1);
        }
      } else if (kind === "circles") {
        for (let i = 0; i < half; i++) {
          const a = rng.next() * Math.PI * 2;
          const r = 0.52 + rng.randn() * 0.08;
          X.push([r * Math.cos(a), r * Math.sin(a)]);
          y.push(0);
        }
        for (let i = 0; i < n - half; i++) {
          const a = rng.next() * Math.PI * 2;
          const r = 1.32 + rng.randn() * 0.09;
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
        { id: "mm1", label: "Z1_mm", op: "MatMul", shape: [N, H], d: t.mm1, g: t.dz1, parents: ["tensor_x", "tensor_w1"] },
        { id: "z1", label: "Z1", op: "Add", shape: [N, H], d: t.z1, g: t.dz1, parents: ["tensor_mm1", "tensor_b1"] },
        { id: "h1", label: "Hidden 1", op: actLabel, shape: [N, H], d: t.h1, g: t.dh1, parents: ["tensor_z1"] },
        { id: "mm2", label: "Z2_mm", op: "MatMul", shape: [N, H], d: t.mm2, g: t.dz2, parents: ["tensor_h1", "tensor_w2"] },
        { id: "z2", label: "Z2", op: "Add", shape: [N, H], d: t.z2, g: t.dz2, parents: ["tensor_mm2", "tensor_b2"] },
        { id: "h2", label: "Hidden 2", op: actLabel, shape: [N, H], d: t.h2, g: t.dh2, parents: ["tensor_z2"] },
        { id: "mm3", label: "Z3_mm", op: "MatMul", shape: [N, 2], d: t.mm3, g: t.dLogits, parents: ["tensor_h2", "tensor_w3"] },
        { id: "logits", label: "Logits", op: "Add", shape: [N, 2], d: t.logits, g: t.dLogits, parents: ["tensor_mm3", "tensor_b3"] },
        { id: "loss", label: "Loss L", op: "CrossEntropyLoss", shape: [1], d: [[this.lastLoss]], g: [[1.0]], parents: ["tensor_logits"], role: "root_loss" },
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

    snapshot(res = 64) {
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
  // STAGE 1: Manim NumberPlane & Decision Manifold Renderer
  // =========================================================================
  function renderManimNumberPlane(data) {
    if (!data) return;
    const { boundary, points, labels } = data;
    const w = bCanvas.width;
    const h = bCanvas.height;
    const res = boundary.resolution;
    const probs = boundary.grid_probs;
    const { x_min, x_max, y_min, y_max } = boundary;

    bCtx.clearRect(0, 0, w, h);
    bCtx.fillStyle = MANIM.BG;
    bCtx.fillRect(0, 0, w, h);

    // 1. Smooth Bilinear Probability Field (Manim Blue #58C4DD <-> Manim Red #FC6255)
    const cellW = w / res;
    const cellH = h / res;
    for (let r = 0; r < res; r++) {
      for (let c = 0; c < res; c++) {
        const p = probs[res - 1 - r][c];
        // Interpolate between deep Manim Blue and deep Manim Red/Coral
        const rCol = Math.round(28 * (1 - p) + 235 * p);
        const gCol = Math.round(145 * (1 - p) + 65 * p);
        const bCol = Math.round(195 * (1 - p) + 60 * p);
        const alpha = 0.22 + 0.25 * Math.abs(p - 0.5);
        bCtx.fillStyle = `rgba(${rCol}, ${gCol}, ${bCol}, ${alpha})`;
        bCtx.fillRect(c * cellW, r * cellH, Math.ceil(cellW) + 0.5, Math.ceil(cellH) + 0.5);
      }
    }

    // 2. Manim NumberPlane Sub-Grid & Major Grid Lines
    const toScreenX = (x) => ((x - x_min) / (x_max - x_min)) * w;
    const toScreenY = (y) => h - ((y - y_min) / (y_max - y_min)) * h;

    // Minor grid lines (every 0.25 units)
    bCtx.strokeStyle = "rgba(35, 107, 142, 0.22)";
    bCtx.lineWidth = 1;
    bCtx.beginPath();
    for (let v = -2.0; v <= 2.0; v += 0.25) {
      const sx = toScreenX(v);
      const sy = toScreenY(v);
      bCtx.moveTo(sx, 0); bCtx.lineTo(sx, h);
      bCtx.moveTo(0, sy); bCtx.lineTo(w, sy);
    }
    bCtx.stroke();

    // Major grid lines (every 0.5 units)
    bCtx.strokeStyle = "rgba(88, 196, 221, 0.38)";
    bCtx.lineWidth = 1.25;
    bCtx.beginPath();
    for (let v = -2.0; v <= 2.0; v += 0.5) {
      if (Math.abs(v) < 1e-6) continue;
      const sx = toScreenX(v);
      const sy = toScreenY(v);
      bCtx.moveTo(sx, 0); bCtx.lineTo(sx, h);
      bCtx.moveTo(0, sy); bCtx.lineTo(w, sy);
    }
    bCtx.stroke();

    // 3. Coordinate Axes with Manim Vector Arrowheads & LaTeX-style ticks
    const ox = toScreenX(0);
    const oy = toScreenY(0);
    bCtx.strokeStyle = "rgba(236, 236, 236, 0.85)";
    bCtx.lineWidth = 2;
    bCtx.beginPath();
    bCtx.moveTo(0, oy); bCtx.lineTo(w, oy);
    bCtx.moveTo(ox, 0); bCtx.lineTo(ox, h);
    bCtx.stroke();

    // Tick numbers in Georgia Italic
    bCtx.fillStyle = "rgba(236, 236, 236, 0.75)";
    bCtx.font = "italic 12px Georgia, serif";
    for (let v = -1.5; v <= 1.5; v += 0.5) {
      if (Math.abs(v) < 1e-6) continue;
      bCtx.fillText(v.toFixed(1), toScreenX(v) - 10, oy + 16);
      bCtx.fillText(v.toFixed(1), ox + 6, toScreenY(v) + 4);
    }
    bCtx.font = "italic bold 14px Georgia, serif";
    bCtx.fillText("x₁", w - 22, oy - 8);
    bCtx.fillText("x₂", ox + 10, 18);

    // 4. Glowing Manim Decision Boundary Contour (P(y=1|x) = 0.5)
    bCtx.save();
    bCtx.fillStyle = MANIM.YELLOW;
    bCtx.shadowColor = MANIM.YELLOW;
    bCtx.shadowBlur = 10;
    for (let r = 0; r < res - 1; r++) {
      for (let c = 0; c < res - 1; c++) {
        const p00 = probs[res - 1 - r][c] - 0.5;
        const p10 = probs[res - 1 - r][c + 1] - 0.5;
        const p01 = probs[res - 2 - r][c] - 0.5;
        if (p00 * p10 <= 0 || p00 * p01 <= 0) {
          bCtx.beginPath();
          bCtx.arc((c + 0.5) * cellW, (r + 0.5) * cellH, 2.2, 0, Math.PI * 2);
          bCtx.fill();
        }
      }
    }
    bCtx.restore();

    // 5. Data Points as Glowing Manim Mobjects
    for (let i = 0; i < points.length; i++) {
      const [px, py] = points[i];
      const cx = toScreenX(px);
      const cy = toScreenY(py);
      const isClass1 = labels[i] === 1;
      const color = isClass1 ? MANIM.RED : MANIM.BLUE;

      bCtx.save();
      bCtx.beginPath();
      bCtx.arc(cx, cy, 5.2, 0, Math.PI * 2);
      bCtx.fillStyle = color;
      bCtx.shadowColor = color;
      bCtx.shadowBlur = 8;
      bCtx.fill();
      bCtx.lineWidth = 1.5;
      bCtx.strokeStyle = "#000000";
      bCtx.stroke();
      bCtx.restore();
    }
  }

  // =========================================================================
  // STAGE 2: Custom Manim Horizontal Layered DAG & Bezier Adjoint Flow
  // =========================================================================
  function computeManimDagLayout(graph) {
    const w = dCanvas.width;
    const h = dCanvas.height;

    // Explicit clean left-to-right 3Blue1Brown neural architecture coordinates
    // Normalized (0..1, 0..1) so every node sits in its exact mathematical layer column!
    const layoutSpec = {
      // Layer 0 & Parameter leaves (top & bottom rows feeding into operators)
      "tensor_x":      { x: 0.06, y: 0.52, w: 92, h: 46, tex: "X" },

      "tensor_w1":     { x: 0.12, y: 0.17, w: 88, h: 42, tex: "W_1" },
      "op_mm1":        { x: 0.17, y: 0.36, r: 22, tex: "\times" },
      "tensor_mm1":    { x: 0.23, y: 0.36, w: 86, h: 42, tex: "X W_1" },
      "tensor_b1":     { x: 0.23, y: 0.78, w: 86, h: 42, tex: "b_1" },
      "op_z1":         { x: 0.29, y: 0.52, r: 20, tex: "+" },
      "tensor_z1":     { x: 0.35, y: 0.52, w: 86, h: 42, tex: "Z_1" },
      "op_h1":         { x: 0.41, y: 0.52, r: 23, tex: "\sigma" },
      "tensor_h1":     { x: 0.48, y: 0.52, w: 92, h: 46, tex: "H_1" },

      "tensor_w2":     { x: 0.48, y: 0.17, w: 88, h: 42, tex: "W_2" },
      "op_mm2":        { x: 0.54, y: 0.36, r: 22, tex: "\times" },
      "tensor_mm2":    { x: 0.60, y: 0.36, w: 86, h: 42, tex: "H_1 W_2" },
      "tensor_b2":     { x: 0.60, y: 0.78, w: 86, h: 42, tex: "b_2" },
      "op_z2":         { x: 0.65, y: 0.52, r: 20, tex: "+" },
      "tensor_z2":     { x: 0.70, y: 0.52, w: 86, h: 42, tex: "Z_2" },
      "op_h2":         { x: 0.75, y: 0.52, r: 23, tex: "\sigma" },
      "tensor_h2":     { x: 0.81, y: 0.52, w: 92, h: 46, tex: "H_2" },

      "tensor_w3":     { x: 0.78, y: 0.17, w: 88, h: 42, tex: "W_3" },
      "op_mm3":        { x: 0.84, y: 0.32, r: 20, tex: "\times" },
      "tensor_mm3":    { x: 0.88, y: 0.20, w: 84, h: 40, tex: "H_2 W_3" },
      "tensor_b3":     { x: 0.84, y: 0.82, w: 84, h: 40, tex: "b_3" },
      "op_logits":     { x: 0.89, y: 0.52, r: 20, tex: "+" },
      "tensor_logits": { x: 0.94, y: 0.52, w: 86, h: 44, tex: "\hat{Y}" },
      "op_loss":       { x: 0.94, y: 0.74, r: 22, tex: "\mathcal{L}" },
      "tensor_loss":   { x: 0.94, y: 0.91, w: 86, h: 42, tex: "\mathcal{L}(\theta)" },
    };

    dagNodePositions = {};
    const totalNodes = graph.nodes.length || 1;

    graph.nodes.forEach((n, idx) => {
      let spec = layoutSpec[n.id];
      // Dynamic fallback if connected to Python FastAPI server with auto-incremented IDs
      if (!spec) {
        const col = idx / totalNodes;
        const row = n.role === "leaf_param" ? 0.18 : (n.type === "operator" ? 0.45 : 0.68);
        spec = { x: 0.06 + col * 0.88, y: row, w: 88, h: 42, r: 21, tex: n.label };
      }
      dagNodePositions[n.id] = {
        ...n,
        cx: spec.x * w,
        cy: spec.y * h,
        w: spec.w || 88,
        h: spec.h || 42,
        r: spec.r || 21,
        tex: spec.tex || n.label,
      };
    });

    dagEdges = graph.edges.map((e) => ({
      ...e,
      src: dagNodePositions[e.source],
      dst: dagNodePositions[e.target],
    })).filter((e) => e.src && e.dst);
  }

  function drawRoundedRect(ctx, x, y, w, h, r) {
    ctx.beginPath();
    ctx.moveTo(x + r, y);
    ctx.lineTo(x + w - r, y);
    ctx.quadraticCurveTo(x + w, y, x + w, y + r);
    ctx.lineTo(x + w, y + h - r);
    ctx.quadraticCurveTo(x + w, y + h, x + w - r, y + h);
    ctx.lineTo(x + r, y + h);
    ctx.quadraticCurveTo(x, y + h, x, y + h - r);
    ctx.lineTo(x, y + r);
    ctx.quadraticCurveTo(x, y, x + r, y);
    ctx.closePath();
  }

  function renderManimDAG() {
    const w = dCanvas.width;
    const h = dCanvas.height;
    dCtx.clearRect(0, 0, w, h);
    dCtx.fillStyle = MANIM.BG;
    dCtx.fillRect(0, 0, w, h);

    // Subtle Manim background coordinate grid
    dCtx.strokeStyle = "rgba(35, 107, 142, 0.14)";
    dCtx.lineWidth = 1;
    dCtx.beginPath();
    for (let x = 0; x < w; x += 40) { dCtx.moveTo(x, 0); dCtx.lineTo(x, h); }
    for (let y = 0; y < h; y += 40) { dCtx.moveTo(0, y); dCtx.lineTo(w, y); }
    dCtx.stroke();

    // Determine active nodes & edges during Backprop Scrubber
    const steps = currentGraph.backprop_steps || [];
    const activeStep = (activeBackpropStep >= 0 && activeBackpropStep < steps.length)
      ? steps[activeBackpropStep]
      : null;

    const activeNodeIds = new Set();
    const activeEdgeIds = new Set();
    if (activeStep) {
      activeNodeIds.add(activeStep.tensor_id);
      activeNodeIds.add(activeStep.op_id);
      activeEdgeIds.add(activeStep.incoming_edge);
      (activeStep.outgoing_edges || []).forEach((id) => activeEdgeIds.add(id));
      (activeStep.parent_tensors || []).forEach((id) => activeNodeIds.add(id));
    }

    // 1. Draw Smooth Cubic Bezier Edges + Glowing Adjoint Pulses
    for (const e of dagEdges) {
      const { src, dst } = e;
      const isHighlighted = activeEdgeIds.has(e.id);
      const dx = dst.cx - src.cx;
      const dy = dst.cy - src.cy;
      const c1x = src.cx + dx * 0.45;
      const c1y = src.cy;
      const c2x = src.cx + dx * 0.55;
      const c2y = dst.cy;

      dCtx.save();
      dCtx.beginPath();
      dCtx.moveTo(src.cx, src.cy);
      dCtx.bezierCurveTo(c1x, c1y, c2x, c2y, dst.cx, dst.cy);

      if (isHighlighted) {
        dCtx.strokeStyle = MANIM.YELLOW;
        dCtx.lineWidth = 3.2;
        dCtx.shadowColor = MANIM.YELLOW;
        dCtx.shadowBlur = 12;
      } else {
        // Color-code by gradient magnitude
        const gNorm = Math.min(1.0, (e.grad_norm || 0) * 3.5);
        dCtx.strokeStyle = `rgba(88, 196, 221, ${0.28 + 0.45 * gNorm})`;
        dCtx.lineWidth = 1.6 + 1.2 * gNorm;
      }
      dCtx.stroke();

      // Animated Manim Adjoint Pulse travelling in REVERSE (from dst back to src!)
      if (isHighlighted || activeBackpropStep === -1) {
        const speed = isHighlighted ? 0.025 : 0.008;
        const t = 1.0 - ((animTime * speed + (src.topo_index || 0) * 0.17) % 1.0);
        const mt = 1 - t;
        const px = mt*mt*mt*src.cx + 3*mt*mt*t*c1x + 3*mt*t*t*c2x + t*t*t*dst.cx;
        const py = mt*mt*mt*src.cy + 3*mt*mt*t*c1y + 3*mt*t*t*c2y + t*t*t*dst.cy;

        dCtx.beginPath();
        dCtx.arc(px, py, isHighlighted ? 4.5 : 2.3, 0, Math.PI * 2);
        dCtx.fillStyle = isHighlighted ? MANIM.YELLOW : MANIM.TEAL;
        dCtx.shadowColor = isHighlighted ? MANIM.YELLOW : MANIM.TEAL;
        dCtx.shadowBlur = isHighlighted ? 14 : 6;
        dCtx.fill();
      }
      dCtx.restore();
    }

    // 2. Draw Nodes (Manim Vector Mobjects)
    Object.values(dagNodePositions).forEach((node) => {
      const isHighlighted = activeNodeIds.has(node.id);
      const isSelected = selectedNodeId === node.id || hoveredNodeId === node.id;

      dCtx.save();
      if (node.type === "operator") {
        // Circular Manim Operator Node
        dCtx.beginPath();
        dCtx.arc(node.cx, node.cy, node.r, 0, Math.PI * 2);
        dCtx.fillStyle = isHighlighted ? "#2a2405" : "#12101b";
        dCtx.fill();
        dCtx.lineWidth = isHighlighted || isSelected ? 2.8 : 1.8;
        const strokeCol = isHighlighted ? MANIM.YELLOW : MANIM.GOLD;
        dCtx.strokeStyle = strokeCol;
        if (isHighlighted || isSelected) {
          dCtx.shadowColor = strokeCol;
          dCtx.shadowBlur = 14;
        }
        dCtx.stroke();

        // Operator Mathematical Symbol
        dCtx.fillStyle = isHighlighted ? MANIM.YELLOW : MANIM.WHITE;
        dCtx.font = "italic bold 13px Georgia, serif";
        dCtx.textAlign = "center";
        dCtx.textBaseline = "middle";
        const sym = node.label === "MatMul" ? "A·B" : (node.label === "Add" ? "+" : (node.label === "CrossEntropyLoss" ? "L(θ)" : node.label));
        dCtx.fillText(sym, node.cx, node.cy);
      } else {
        // Tensor Mobject Bracketed Card
        const x = node.cx - node.w / 2;
        const y = node.cy - node.h / 2;
        let strokeCol = MANIM.BLUE;
        let fillCol = "#08121a";
        if (node.role === "leaf_param") {
          strokeCol = MANIM.GREEN;
          fillCol = "#09170e";
        } else if (node.role === "root_loss") {
          strokeCol = MANIM.RED;
          fillCol = "#1c0a0a";
        } else if (node.role === "input_data") {
          strokeCol = MANIM.PURPLE;
          fillCol = "#130d18";
        }
        if (isHighlighted) {
          strokeCol = MANIM.YELLOW;
          fillCol = "#221e06";
        }

        drawRoundedRect(dCtx, x, y, node.w, node.h, 7);
        dCtx.fillStyle = fillCol;
        dCtx.fill();
        dCtx.lineWidth = isHighlighted || isSelected ? 2.8 : 1.6;
        dCtx.strokeStyle = strokeCol;
        if (isHighlighted || isSelected) {
          dCtx.shadowColor = strokeCol;
          dCtx.shadowBlur = 15;
        }
        dCtx.stroke();

        // Tensor Name & Shape in Computer Modern / Georgia Serif
        dCtx.fillStyle = MANIM.WHITE;
        dCtx.font = "italic bold 12px Georgia, serif";
        dCtx.textAlign = "center";
        const shapeStr = node.shape ? `[${node.shape.join("×")}]` : "";
        dCtx.fillText(`${node.label} ${shapeStr}`, node.cx, node.cy - 6);

        // Gradient Norm Bar & Value
        const gNorm = node.grad_stats ? node.grad_stats.norm : 0;
        dCtx.fillStyle = isHighlighted ? MANIM.YELLOW : strokeCol;
        dCtx.font = "10px 'JetBrains Mono', monospace";
        dCtx.fillText(`‖∇‖=${gNorm.toExponential(2)}`, node.cx, node.cy + 10);
      }
      dCtx.restore();
    });
  }

  // Hit-testing on the Manim DAG Canvas
  function findNodeAt(mx, my) {
    const rect = dCanvas.getBoundingClientRect();
    const scaleX = dCanvas.width / rect.width;
    const scaleY = dCanvas.height / rect.height;
    const x = (mx - rect.left) * scaleX;
    const y = (my - rect.top) * scaleY;

    for (const node of Object.values(dagNodePositions)) {
      if (node.type === "operator") {
        const dist = Math.hypot(x - node.cx, y - node.cy);
        if (dist <= node.r + 4) return node;
      } else {
        if (
          x >= node.cx - node.w / 2 &&
          x <= node.cx + node.w / 2 &&
          y >= node.cy - node.h / 2 &&
          y <= node.cy + node.h / 2
        ) {
          return node;
        }
      }
    }
    return null;
  }

  function inspectNode(node) {
    if (!node) return;
    selectedNodeId = node.id;
    if (node.type === "tensor") {
      const shapeLatex = node.shape ? `\mathbb{R}^{${node.shape.join(" \times ")}}` : "\mathbb{R}";
      const gNorm = node.grad_stats ? node.grad_stats.norm : 0;
      const vNorm = node.stats ? node.stats.norm : 0;
      renderMath(
        inspectorLatex,
        `\mathbf{${node.label.replace("_", "\_")}} \in ${shapeLatex}, \quad \|\mathbf{T}\|_2 = ${vNorm}, \quad \left\|\frac{\partial \mathcal{L}}{\partial \mathbf{T}}\right\|_2 = ${gNorm}`
      );
      inspectorStats.innerHTML = `
        <span>Role: <strong style="color:${MANIM.BLUE}">${node.role}</strong></span>
        <span>Forward Range: <code>[${node.stats.min}, ${node.stats.max}]</code></span>
        <span>Adjoint Range: <code>[${node.grad_stats.min}, ${node.grad_stats.max}]</code></span>
      `;
    } else {
      const rules = {
        MatMul: "\frac{\partial \mathcal{L}}{\partial A} = \frac{\partial \mathcal{L}}{\partial C} B^\top, \quad \frac{\partial \mathcal{L}}{\partial B} = A^\top \frac{\partial \mathcal{L}}{\partial C}",
        Add: "\frac{\partial \mathcal{L}}{\partial X} = \text{unbroadcast}\left(\frac{\partial \mathcal{L}}{\partial Z}\right)",
        CrossEntropyLoss: "\mathcal{L} = \frac{1}{N}\sum_i \left(\log\sum_j e^{z_{i,j}} - z_{i,y_i}\right), \quad \frac{\partial \mathcal{L}}{\partial Z} = \frac{1}{N}(\text{softmax}(Z) - Y)",
        TANH: "\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial H} \odot (1 - \tanh^2(Z))",
        RELU: "\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial H} \odot \mathbb{I}(Z > 0)",
        GELU: "\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial H} \odot \text{GELU}'(Z)",
        SIGMOID: "\frac{\partial \mathcal{L}}{\partial Z} = \frac{\partial \mathcal{L}}{\partial H} \odot \sigma(Z)(1 - \sigma(Z))",
      };
      const ruleLatex = rules[node.label] || `\text{Operator } ${node.label}`;
      renderMath(inspectorLatex, ruleLatex);
      inspectorStats.innerHTML = `
        <span>Operator: <strong style="color:${MANIM.GOLD}">${node.label}</strong></span>
        <span>Topological Order: <code>#${node.topo_index}</code></span>
        <span>Upstream Adjoint Norm: <code>${node.grad_norm}</code></span>
      `;
    }
  }

  dCanvas.addEventListener("mousemove", (e) => {
    const hit = findNodeAt(e.clientX, e.clientY);
    hoveredNodeId = hit ? hit.id : null;
    dCanvas.style.cursor = hit ? "pointer" : "default";
  });

  dCanvas.addEventListener("click", (e) => {
    const hit = findNodeAt(e.clientX, e.clientY);
    if (hit) inspectNode(hit);
  });

  function applyBackpropHighlight(stepIndex) {
    activeBackpropStep = stepIndex;
    const steps = currentGraph.backprop_steps || [];

    if (stepIndex < 0 || stepIndex >= steps.length) {
      stepCounterBadge.textContent = "Chain Rule: Full DAG";
      metricBackpropOp.textContent = "Full Forward & Backward DAG";
      scrubberStatus.textContent =
        "Animating continuous adjoint flow across all edges. Drag the yellow scrubber to isolate individual Chain Rule steps.";
      return;
    }

    const active = steps[stepIndex];
    stepCounterBadge.textContent = `Backprop Step ${stepIndex + 1} / ${steps.length}`;
    metricBackpropOp.textContent = `${active.op_label} (‖∇‖₂ = ${active.grad_norm})`;
    scrubberStatus.textContent = `Step ${stepIndex + 1}/${steps.length}: Propagating adjoint δ through ${active.op_label} → [${active.parent_tensors.join(", ")}]`;

    const opNode = dagNodePositions[active.op_id];
    if (opNode) inspectNode(opNode);
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
        if (!res.ok) throw new Error("FastAPI unavailable");
        data = await res.json();
        engineBadge.textContent = "Scene: FastAPI NumPy";
      } catch (_e) {
        useLocalEngine = true;
      }
    }

    if (useLocalEngine) {
      engineBadge.textContent = "Scene: 60fps In-Browser Autograd";
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

    currentData = data;
    currentGraph = data.graph;
    metricStep.textContent = data.step;
    metricLoss.textContent = data.loss.toFixed(5);
    metricAcc.textContent = `${data.accuracy.toFixed(2)}%`;

    computeManimDagLayout(currentGraph);
    renderManimNumberPlane(currentData);

    const maxStep = Math.max(0, currentGraph.backprop_steps.length - 1);
    backpropSlider.max = String(maxStep);
    if (parseInt(backpropSlider.value, 10) > maxStep) backpropSlider.value = "-1";
    applyBackpropHighlight(parseInt(backpropSlider.value, 10));
  }

  // 60 FPS Manim Animation Loop for glowing adjoint pulses
  function animateLoop() {
    animTime += 1;
    renderManimDAG();
    requestAnimationFrame(animateLoop);
  }

  // Controls & Event Listeners
  stepBtn.addEventListener("click", () => postStep(1, false));
  resetBtn.addEventListener("click", () => {
    if (trainTimer) {
      clearInterval(trainTimer);
      trainTimer = null;
      playTrainBtn.textContent = "▶ Animate Training";
      playTrainBtn.classList.remove("active");
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
      playTrainBtn.textContent = "▶ Animate Training";
      playTrainBtn.classList.remove("active");
    } else {
      playTrainBtn.textContent = "⏸ Pause Animation";
      playTrainBtn.classList.add("active");
      trainTimer = setInterval(() => postStep(2, false), 90);
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
      playBackpropBtn.textContent = "▶ Animate Backprop";
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
        playBackpropBtn.textContent = "▶ Animate Backprop";
        backpropSlider.value = "-1";
        applyBackpropHighlight(-1);
        return;
      }
      backpropSlider.value = String(idx);
      applyBackpropHighlight(idx);
    }, 600);
  });

  window.addEventListener("load", () => {
    renderMath(
      headerFormula,
      "\theta_{t+1} = \theta_t - \eta \cdot \nabla_\theta \mathcal{L}(f_\theta(X), Y)"
    );
    renderMath(
      inspectorLatex,
      "\frac{\partial \mathcal{L}}{\partial W_l} = H_{l-1}^\top \delta_l, \qquad \delta_l = \left(\delta_{l+1} W_{l+1}^\top\right) \odot \sigma'(Z_l)"
    );
  });

  postStep(1, true);
  requestAnimationFrame(animateLoop);
=======
(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ui = {
    dataset: $('dataset'), datasetHint: $('dataset-hint'), hidden: $('hidden'), hiddenValue: $('hidden-value'),
    activation: $('activation'), optimizer: $('optimizer'), learningRate: $('learning-rate'), lrValue: $('lr-value'),
    batchSize: $('batch-size'), regenerate: $('regenerate'), train: $('train'), trainLabel: $('train-label'),
    step: $('step'), reset: $('reset'), status: $('status'), trainingMessage: $('training-message'),
    epoch: $('epoch'), loss: $('loss'), accuracy: $('accuracy'), confidence: $('confidence'),
    surfaceCaption: $('surface-caption'), architectureLabel: $('architecture-label'), parameters: $('parameters'),
    chartCurrent: $('chart-current'), insightText: $('insight-text'), surface: $('surface'), network: $('network'), chart: $('loss-chart')
  };

  const datasetDescriptions = {
    xor: ['A pattern that a straight line can\'t separate.', 'XOR · 320 training examples', 'A network needs a hidden layer to bend a decision boundary around this pattern.'],
    moons: ['Two curved clusters, nested like a pair of crescents.', 'TWO MOONS · 320 training examples', 'The boundary has to curve between two interlocking groups.'],
    circles: ['One class inside a ring, the other wrapped around it.', 'CONCENTRIC CIRCLES · 320 training examples', 'A radial pattern rewards a model that can combine features non-linearly.'],
    spiral: ['Two intertwined arms. A deceptively difficult classification task.', 'DOUBLE SPIRAL · 320 training examples', 'A spiral asks the network to learn a boundary that twists through the plane.']
  };
  const insightBase = 'The model does not know what this pattern is called. It only adjusts weights to reduce its mistakes.';
  let randomSeed = 75319;
  let data = [];
  let model = null;
  let epoch = 0;
  let history = [];
  let running = false;
  let timer = null;
  let latestMetrics = { loss: Math.log(2), accuracy: .5 };
  let surfaceResizeObserver;

  function random() {
    randomSeed |= 0;
    randomSeed = randomSeed + 0x6D2B79F5 | 0;
    let t = Math.imul(randomSeed ^ randomSeed >>> 15, 1 | randomSeed);
    t = t + Math.imul(t ^ t >>> 7, 61 | t) ^ t;
    return ((t ^ t >>> 14) >>> 0) / 4294967296;
  }
  function randn() {
    const u = Math.max(1e-9, random());
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * random());
  }
  function clamp(x, lo, hi) { return Math.max(lo, Math.min(hi, x)); }
  function sigmoid(x) { return 1 / (1 + Math.exp(-clamp(x, -30, 30))); }
  function activate(z, kind) {
    if (kind === 'relu') return Math.max(0, z);
    if (kind === 'sigmoid') return sigmoid(z);
    return Math.tanh(z);
  }
  function derivative(z, a, kind) {
    if (kind === 'relu') return z > 0 ? 1 : 0;
    if (kind === 'sigmoid') return a * (1 - a);
    return 1 - a * a;
  }
  function getLearningRate() { return 0.0005 * Math.pow(100, Number(ui.learningRate.value) / 100); }
  function currentConfig() {
    return { hidden: Number(ui.hidden.value), activation: ui.activation.value, optimizer: ui.optimizer.value, lr: getLearningRate(), batch: Number(ui.batchSize.value) };
  }

  function generateDataset(kind) {
    randomSeed = (Math.floor(Math.random() * 1e9) ^ 0x51f15e) >>> 0;
    const out = [];
    const n = 320;
    for (let i = 0; i < n; i++) {
      let x = 0, y = 0, label = 0;
      if (kind === 'xor') {
        x = (random() * 2 - 1) * .88;
        y = (random() * 2 - 1) * .88;
        label = x * y >= 0 ? 1 : 0;
        x += randn() * .035; y += randn() * .035;
      } else if (kind === 'moons') {
        const upper = random() > .5;
        const t = random() * Math.PI;
        if (upper) { x = Math.cos(t) * .58 - .15; y = Math.sin(t) * .54 + .23; label = 0; }
        else { x = 1 - Math.cos(t) * .58 + .15; y = -.23 - Math.sin(t) * .54; label = 1; }
        x += randn() * .075; y += randn() * .075;
      } else if (kind === 'circles') {
        const outer = random() > .5;
        const t = random() * Math.PI * 2;
        const r = outer ? .68 + randn() * .065 : .31 + randn() * .055;
        x = Math.cos(t) * r + randn() * .025;
        y = Math.sin(t) * r + randn() * .025;
        label = outer ? 1 : 0;
      } else {
        const arm = random() > .5 ? 1 : 0;
        const t = .16 + random() * .88;
        const angle = t * Math.PI * 2.3 + arm * Math.PI;
        const r = .08 + t * .72;
        x = Math.cos(angle) * r + randn() * .035;
        y = Math.sin(angle) * r + randn() * .035;
        label = arm;
      }
      out.push({ x: clamp(x, -1, 1), y: clamp(y, -1, 1), label });
    }
    // A few edge examples keep each quadrant represented and the boundary visible.
    return out;
  }

  function zerosLike(params) { return params.map(layer => layer.map(row => row.map(() => 0))); }
  function zerosBiasLike(params) { return params.map(layer => layer.map(() => 0)); }
  function createModel() {
    const cfg = currentConfig();
    const sizes = [2, cfg.hidden, 1];
    const weights = [], biases = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l], fanOut = sizes[l + 1];
      const scale = Math.sqrt(2 / (fanIn + fanOut));
      weights.push(Array.from({ length: fanOut }, () => Array.from({ length: fanIn }, () => randn() * scale)));
      biases.push(Array.from({ length: fanOut }, () => 0));
    }
    const mW = weights.map(layer => layer.map(row => row.map(() => 0)));
    const vW = weights.map(layer => layer.map(row => row.map(() => 0)));
    const mB = biases.map(layer => layer.map(() => 0));
    const vB = biases.map(layer => layer.map(() => 0));
    model = { sizes, weights, biases, mW, vW, mB, vB, step: 0, activation: cfg.activation };
    epoch = 0;
    history = [];
    latestMetrics = evaluate();
    history.push(latestMetrics.loss);
    updateArchitectureText();
    renderAll();
  }

  function forward(x, y, save = false) {
    const activations = [[x, y]], zs = [];
    let a = [x, y];
    for (let l = 0; l < model.weights.length; l++) {
      const z = model.weights[l].map((row, j) => row.reduce((sum, w, i) => sum + w * a[i], model.biases[l][j]));
      zs.push(z);
      a = z.map(v => l === model.weights.length - 1 ? sigmoid(v) : activate(v, model.activation));
      activations.push(a);
    }
    return save ? { output: a[0], activations, zs } : a[0];
  }

  function evaluate() {
    if (!model || !data.length) return { loss: Math.log(2), accuracy: .5 };
    let loss = 0, correct = 0;
    for (const p of data) {
      const pred = clamp(forward(p.x, p.y), 1e-7, 1 - 1e-7);
      loss += -(p.label * Math.log(pred) + (1 - p.label) * Math.log(1 - pred));
      if ((pred >= .5 ? 1 : 0) === p.label) correct++;
    }
    return { loss: loss / data.length, accuracy: correct / data.length };
  }

  function trainEpoch() {
    if (!model || !data.length) return;
    const cfg = currentConfig();
    const indices = Array.from({ length: data.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }
    const gradW = model.weights.map(layer => layer.map(row => row.map(() => 0)));
    const gradB = model.biases.map(layer => layer.map(() => 0));
    for (let start = 0; start < indices.length; start += cfg.batch) {
      for (const layer of gradW) for (const row of layer) row.fill(0);
      for (const layer of gradB) layer.fill(0);
      const end = Math.min(indices.length, start + cfg.batch);
      const batchCount = end - start;
      for (let k = start; k < end; k++) {
        const p = data[indices[k]];
        const f = forward(p.x, p.y, true);
        const deltas = Array(model.weights.length);
        const last = model.weights.length - 1;
        deltas[last] = [f.output - p.label]; // sigmoid + binary cross-entropy
        for (let l = last - 1; l >= 0; l--) {
          deltas[l] = model.weights[l].map((_, j) => {
            let propagated = 0;
            for (let next = 0; next < deltas[l + 1].length; next++) propagated += model.weights[l + 1][next][j] * deltas[l + 1][next];
            return propagated * derivative(f.zs[l][j], f.activations[l + 1][j], model.activation);
          });
        }
        for (let l = 0; l < model.weights.length; l++) {
          for (let j = 0; j < model.weights[l].length; j++) {
            gradB[l][j] += deltas[l][j];
            for (let i = 0; i < model.weights[l][j].length; i++) gradW[l][j][i] += deltas[l][j] * f.activations[l][i];
          }
        }
      }
      applyGradients(gradW, gradB, batchCount, cfg);
    }
    epoch++;
    latestMetrics = evaluate();
    history.push(latestMetrics.loss);
    if (history.length > 150) history.shift();
  }

  function applyGradients(gW, gB, count, cfg) {
    model.step++;
    const adam = cfg.optimizer === 'adam';
    const beta1 = .9, beta2 = .999, epsilon = 1e-8;
    const t = model.step;
    for (let l = 0; l < model.weights.length; l++) {
      for (let j = 0; j < model.weights[l].length; j++) {
        const gb = gB[l][j] / count;
        if (adam) {
          model.mB[l][j] = beta1 * model.mB[l][j] + (1 - beta1) * gb;
          model.vB[l][j] = beta2 * model.vB[l][j] + (1 - beta2) * gb * gb;
          const mh = model.mB[l][j] / (1 - Math.pow(beta1, t));
          const vh = model.vB[l][j] / (1 - Math.pow(beta2, t));
          model.biases[l][j] -= cfg.lr * mh / (Math.sqrt(vh) + epsilon);
        } else model.biases[l][j] -= cfg.lr * gb;
        for (let i = 0; i < model.weights[l][j].length; i++) {
          const g = gW[l][j][i] / count;
          if (adam) {
            model.mW[l][j][i] = beta1 * model.mW[l][j][i] + (1 - beta1) * g;
            model.vW[l][j][i] = beta2 * model.vW[l][j][i] + (1 - beta2) * g * g;
            const mh = model.mW[l][j][i] / (1 - Math.pow(beta1, t));
            const vh = model.vW[l][j][i] / (1 - Math.pow(beta2, t));
            model.weights[l][j][i] -= cfg.lr * mh / (Math.sqrt(vh) + epsilon);
          } else model.weights[l][j][i] -= cfg.lr * g;
        }
      }
    }
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) { canvas.width = width; canvas.height = height; }
    const ctx = canvas.getContext('2d');
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { ctx, width: rect.width, height: rect.height };
  }

  function drawSurface() {
    const { ctx, width: w, height: h } = fitCanvas(ui.surface);
    ctx.clearRect(0, 0, w, h);
    const left = 35, right = 12, top = 12, bottom = 29;
    const pw = w - left - right, ph = h - top - bottom;
    const sx = x => left + (x + 1) * .5 * pw;
    const sy = y => top + (1 - y) * .5 * ph;
    const steps = Math.max(24, Math.min(48, Math.floor(pw / 12)));
    for (let gx = 0; gx < steps; gx++) {
      for (let gy = 0; gy < steps; gy++) {
        const x = -1 + (gx + .5) * 2 / steps;
        const y = 1 - (gy + .5) * 2 / steps;
        const p = forward(x, y);
        const t = clamp((p - .5) * 1.35, -.5, .5);
        const alpha = .10 + Math.abs(t) * .26;
        ctx.fillStyle = p >= .5 ? `rgba(241,92,61,${alpha})` : `rgba(52,127,121,${alpha})`;
        const x0 = sx(-1 + gx * 2 / steps), x1 = sx(-1 + (gx + 1) * 2 / steps);
        const y0 = sy(1 - gy * 2 / steps), y1 = sy(1 - (gy + 1) * 2 / steps);
        ctx.fillRect(x0, y0, x1 - x0 + .6, y1 - y0 + .6);
      }
    }
    ctx.strokeStyle = 'rgba(97,103,93,.13)'; ctx.lineWidth = 1;
    for (let i = 0; i <= 4; i++) {
      const x = left + pw * i / 4, y = top + ph * i / 4;
      ctx.beginPath(); ctx.moveTo(x, top); ctx.lineTo(x, top + ph); ctx.stroke();
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(left + pw, y); ctx.stroke();
    }
    ctx.strokeStyle = 'rgba(83,91,81,.34)'; ctx.lineWidth = 1;
    ctx.strokeRect(left, top, pw, ph);
    ctx.fillStyle = '#a0a39a'; ctx.font = '9px DM Mono, monospace'; ctx.textAlign = 'center';
    [-1, -.5, 0, .5, 1].forEach(v => ctx.fillText(v.toFixed(v === 0 || Math.abs(v) === 1 ? 0 : 1), sx(v), h - 11));
    ctx.textAlign = 'right';
    [-1, -.5, 0, .5, 1].forEach(v => ctx.fillText(v.toFixed(v === 0 || Math.abs(v) === 1 ? 0 : 1), left - 9, sy(v) + 3));
    for (const p of data) {
      const x = sx(p.x), y = sy(p.y);
      ctx.beginPath(); ctx.arc(x, y, w < 520 ? 2.5 : 3.1, 0, Math.PI * 2);
      ctx.fillStyle = p.label ? '#f15c3d' : '#347f79'; ctx.fill();
      ctx.lineWidth = 1.1; ctx.strokeStyle = '#fbfaf6'; ctx.stroke();
    }
  }

  function drawNetwork() {
    const { ctx, width: w, height: h } = fitCanvas(ui.network);
    ctx.clearRect(0, 0, w, h);
    const padX = 23, top = 25, bottom = 24;
    const layerCount = model.sizes.length;
    const xs = model.sizes.map((_, i) => padX + (w - padX * 2) * (layerCount === 1 ? .5 : i / (layerCount - 1)));
    const positions = model.sizes.map((count, l) => {
      const displayCount = Math.min(count, 8);
      const spacing = Math.min(25, (h - top - bottom) / Math.max(1, displayCount - 1));
      const total = spacing * (displayCount - 1);
      return Array.from({ length: displayCount }, (_, i) => ({ x: xs[l], y: h / 2 - total / 2 + i * spacing, actual: i }));
    });
    for (let l = 0; l < model.weights.length; l++) {
      for (let j = 0; j < model.weights[l].length; j++) {
        for (let i = 0; i < model.weights[l][j].length; i++) {
          if (i >= positions[l].length || j >= positions[l + 1].length) continue;
          const a = positions[l][i], b = positions[l + 1][j], value = model.weights[l][j][i];
          ctx.beginPath(); ctx.moveTo(a.x, a.y); ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = value >= 0 ? 'rgba(241,92,61,.48)' : 'rgba(52,127,121,.48)';
          ctx.lineWidth = .55 + Math.min(2.0, Math.abs(value) * .55); ctx.stroke();
        }
      }
    }
    const colors = ['#e8e7df', '#f4ded6', '#dbe8e2'];
    positions.forEach((layer, l) => {
      layer.forEach((p, i) => {
        ctx.beginPath(); ctx.arc(p.x, p.y, l === 0 || l === layerCount - 1 ? 6 : 5.4, 0, Math.PI * 2);
        ctx.fillStyle = colors[l] || colors[1]; ctx.fill();
        ctx.strokeStyle = l === layerCount - 1 ? '#f15c3d' : l === 0 ? '#8b988b' : '#e18b72';
        ctx.lineWidth = 1.25; ctx.stroke();
      });
      if (model.sizes[l] > layer.length) {
        const last = layer[layer.length - 1];
        ctx.fillStyle = '#999e94'; ctx.font = '9px DM Mono, monospace'; ctx.textAlign = 'center';
        ctx.fillText(`+${model.sizes[l] - layer.length}`, last.x, Math.min(h - 13, last.y + 18));
      }
      ctx.fillStyle = '#969b91'; ctx.font = '8px DM Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText(l === 0 ? 'INPUT' : l === layerCount - 1 ? 'OUTPUT' : 'HIDDEN', xs[l], 12);
    });
  }

  function drawLoss() {
    const { ctx, width: w, height: h } = fitCanvas(ui.chart);
    ctx.clearRect(0, 0, w, h);
    const left = 4, right = 4, top = 8, bottom = 12, pw = w - left - right, ph = h - top - bottom;
    for (let i = 0; i < 3; i++) {
      const y = top + ph * i / 2;
      ctx.beginPath(); ctx.moveTo(left, y); ctx.lineTo(w - right, y); ctx.strokeStyle = '#eeece5'; ctx.lineWidth = 1; ctx.stroke();
    }
    if (history.length < 2) {
      ctx.fillStyle = '#a3a69d'; ctx.font = '9px DM Mono, monospace'; ctx.textAlign = 'center';
      ctx.fillText('LOSS TRACE WILL APPEAR HERE', w / 2, h / 2 + 3); return;
    }
    const max = Math.max(.75, Math.min(2.5, Math.max(...history) * 1.08));
    const min = Math.max(0, Math.min(...history) * .82 - .015);
    const points = history.map((v, i) => ({ x: left + (i / Math.max(1, history.length - 1)) * pw, y: top + (1 - (v - min) / Math.max(.001, max - min)) * ph }));
    ctx.beginPath(); ctx.moveTo(points[0].x, top + ph); points.forEach(p => ctx.lineTo(p.x, p.y)); ctx.lineTo(points[points.length - 1].x, top + ph); ctx.closePath();
    const gradient = ctx.createLinearGradient(0, top, 0, h); gradient.addColorStop(0, 'rgba(241,92,61,.20)'); gradient.addColorStop(1, 'rgba(241,92,61,0)'); ctx.fillStyle = gradient; ctx.fill();
    ctx.beginPath(); points.forEach((p, i) => i ? ctx.lineTo(p.x, p.y) : ctx.moveTo(p.x, p.y));
    ctx.strokeStyle = '#ed6849'; ctx.lineWidth = 1.8; ctx.lineJoin = 'round'; ctx.lineCap = 'round'; ctx.stroke();
    const p = points[points.length - 1]; ctx.beginPath(); ctx.arc(p.x, p.y, 3, 0, Math.PI * 2); ctx.fillStyle = '#ed6849'; ctx.fill(); ctx.strokeStyle = '#fff'; ctx.lineWidth = 1.2; ctx.stroke();
  }

  function updateArchitectureText() {
    ui.architectureLabel.textContent = model.sizes.join('–');
    const params = model.weights.reduce((n, layer) => n + layer.reduce((m, row) => m + row.length, 0), 0) + model.biases.reduce((n, layer) => n + layer.length, 0);
    ui.parameters.textContent = params.toLocaleString('en-US');
  }

  function renderMetrics() {
    ui.epoch.textContent = String(epoch).padStart(4, '0');
    ui.loss.textContent = latestMetrics.loss.toFixed(3);
    ui.accuracy.innerHTML = `${(latestMetrics.accuracy * 100).toFixed(1)}<span>%</span>`;
    ui.confidence.textContent = `${Math.round(latestMetrics.accuracy * 100)}%`;
    ui.chartCurrent.textContent = `LOSS ${latestMetrics.loss.toFixed(3)}`;
  }

  function renderAll() {
    drawSurface(); drawNetwork(); drawLoss(); renderMetrics();
    ui.surfaceCaption.textContent = datasetDescriptions[ui.dataset.value][1];
  }

  function setRunning(value) {
    running = value;
    if (timer) { clearTimeout(timer); timer = null; }
    ui.trainLabel.textContent = running ? 'Pause training' : 'Start training';
    ui.train.querySelector('.play-icon').textContent = running ? 'Ⅱ' : '▶';
    ui.status.classList.toggle('running', running);
    ui.status.innerHTML = running ? '<i></i> TRAINING' : `<i></i> ${epoch ? 'PAUSED' : 'INITIALIZED'}`;
    ui.trainingMessage.textContent = running ? 'Weights are updating. Watch the boundary shift.' : epoch ? `Training paused at epoch ${epoch.toLocaleString('en-US')}.` : 'The model is ready. Give it a few epochs.';
    if (running) scheduleTraining();
  }
  function scheduleTraining() {
    if (!running) return;
    timer = setTimeout(() => {
      if (!running) return;
      trainEpoch(); renderAll(); scheduleTraining();
    }, 45);
  }

  function resetModel(message = 'The model is ready. Give it a few epochs.') {
    setRunning(false);
    createModel();
    ui.trainingMessage.textContent = message;
    ui.status.innerHTML = '<i></i> INITIALIZED';
    ui.status.classList.remove('running');
  }

  ui.dataset.addEventListener('change', () => {
    ui.datasetHint.textContent = datasetDescriptions[ui.dataset.value][0];
    ui.insightText.textContent = datasetDescriptions[ui.dataset.value][2];
    data = generateDataset(ui.dataset.value);
    resetModel('New dataset generated. The model is ready.');
  });
  ui.regenerate.addEventListener('click', () => {
    data = generateDataset(ui.dataset.value);
    resetModel('Fresh data, fresh weights. Start training when ready.');
  });
  ui.hidden.addEventListener('input', () => { ui.hiddenValue.textContent = ui.hidden.value; });
  ui.hidden.addEventListener('change', () => resetModel('Architecture changed. New weights initialized.'));
  ui.activation.addEventListener('change', () => resetModel('Activation changed. New weights initialized.'));
  ui.optimizer.addEventListener('change', () => { if (model) { model.step = 0; model.mW = zerosLike(model.weights); model.vW = zerosLike(model.weights); model.mB = zerosBiasLike(model.biases); model.vB = zerosBiasLike(model.biases); } ui.trainingMessage.textContent = `${ui.optimizer.value.toUpperCase()} selected. Optimizer state cleared.`; });
  ui.learningRate.addEventListener('input', () => { ui.lrValue.textContent = getLearningRate().toFixed(4); });
  ui.batchSize.addEventListener('change', () => { ui.trainingMessage.textContent = `Batch size set to ${ui.batchSize.value}.`; });
  ui.train.addEventListener('click', () => setRunning(!running));
  ui.step.addEventListener('click', () => { setRunning(false); trainEpoch(); renderAll(); ui.trainingMessage.textContent = `One epoch complete. You are at epoch ${epoch.toLocaleString('en-US')}.`; ui.status.innerHTML = '<i></i> PAUSED'; });
  ui.reset.addEventListener('click', () => resetModel('Weights reset. The same data is ready for another run.'));

  data = generateDataset(ui.dataset.value);
  ui.lrValue.textContent = getLearningRate().toFixed(4);
  ui.hiddenValue.textContent = ui.hidden.value;
  ui.datasetHint.textContent = datasetDescriptions[ui.dataset.value][0];
  ui.insightText.textContent = insightBase;
  createModel();

  if ('ResizeObserver' in window) {
    surfaceResizeObserver = new ResizeObserver(() => renderAll());
    [ui.surface, ui.network, ui.chart].forEach(canvas => surfaceResizeObserver.observe(canvas));
  } else window.addEventListener('resize', renderAll);
  window.addEventListener('beforeunload', () => { if (timer) clearTimeout(timer); if (surfaceResizeObserver) surfaceResizeObserver.disconnect(); });
>>>>>>> e0dcee5 (Redesign Tensile website and branding)
})();
