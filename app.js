(() => {
  'use strict';

  const $ = (id) => document.getElementById(id);
  const ui = {
    dataset: $('dataset'),
    datasetHint: $('dataset-hint'),
    depth: $('depth'),
    depthValue: $('depth-value'),
    hidden: $('hidden'),
    hiddenValue: $('hidden-value'),
    activation: $('activation'),
    optimizer: $('optimizer'),
    learningRate: $('learning-rate'),
    lrValue: $('lr-value'),
    batchSize: $('batch-size'),
    noise: $('noise'),
    noiseValue: $('noise-value'),
    featuresCount: $('features-count'),
    featureToggles: $('feature-toggles'),
    regenerate: $('regenerate'),
    train: $('train'),
    trainLabel: $('train-label'),
    step: $('step'),
    reset: $('reset'),
    status: $('status'),
    trainingMessage: $('training-message'),
    epoch: $('epoch'),
    loss: $('loss'),
    accuracy: $('accuracy'),
    confidence: $('confidence'),
    surfaceCaption: $('surface-caption'),
    architectureLabel: $('architecture-label'),
    parameters: $('parameters'),
    chartCurrent: $('chart-current'),
    insightText: $('insight-text'),
    surface: $('surface'),
    canvasProbe: $('canvas-probe'),
    exportWeights: $('export-weights'),
    network: $('network'),
    chart: $('loss-chart'),
    heroArchMeta: $('hero-arch-meta')
  };

  const datasetDescriptions = {
    xor: [
      "A non-linear pattern that a straight line cannot separate.",
      "XOR · 320 examples (256 Train / 64 Test)",
      "A network needs a hidden layer or polynomial features (X₁X₂) to separate this pattern."
    ],
    moons: [
      "Two curved crescent clusters, nested together.",
      "TWO MOONS · 320 examples (256 Train / 64 Test)",
      "The decision surface has to bend smoothly between two interlocking arcs."
    ],
    circles: [
      "Concentric rings: one class in the center, the other enclosing it.",
      "CONCENTRIC CIRCLES · 320 examples (256 Train / 64 Test)",
      "Radial patterns reward non-linear activations or quadratic inputs (X₁² + X₂²)."
    ],
    spiral: [
      "Two intertwined arms twisting outward from the origin.",
      "DOUBLE SPIRAL · 320 examples (256 Train / 64 Test)",
      "A spiral requires deeper networks (2-3 layers) or harmonic features to resolve."
    ]
  };

  const activeFeatures = {
    x1: true,
    x2: true,
    x1_sq: false,
    x2_sq: false,
    x1_x2: false,
    sin_x1: false,
    sin_x2: false
  };

  let randomSeed = 0x51f15e;
  function random() {
    randomSeed = (randomSeed * 1664525 + 1013904223) >>> 0;
    return randomSeed / 4294967296;
  }
  function randn() {
    let u = 0, v = 0;
    while (u === 0) u = random();
    while (v === 0) v = random();
    return Math.sqrt(-2.0 * Math.log(u)) * Math.cos(2.0 * Math.PI * v);
  }
  function clamp(v, min, max) { return Math.max(min, Math.min(max, v)); }

  let allData = [];
  let trainData = [];
  let testData = [];
  let model = null;
  let epoch = 0;
  let isTraining = false;
  let trainHistory = [];
  let testHistory = [];
  let animationId = null;

  function sigmoid(x) { return 1 / (1 + Math.exp(-clamp(x, -25, 25))); }

  function activate(x, type) {
    if (type === 'relu') return Math.max(0, x);
    if (type === 'leaky_relu') return x > 0 ? x : 0.05 * x;
    if (type === 'sigmoid') return sigmoid(x);
    return Math.tanh(x); // default tanh
  }

  function derivative(z, a, type) {
    if (type === 'relu') return z > 0 ? 1 : 0;
    if (type === 'leaky_relu') return z > 0 ? 1 : 0.05;
    if (type === 'sigmoid') return a * (1 - a);
    return 1 - a * a; // tanh
  }

  function extractFeatures(x, y) {
    const vec = [];
    if (activeFeatures.x1) vec.push(x);
    if (activeFeatures.x2) vec.push(y);
    if (activeFeatures.x1_sq) vec.push(x * x);
    if (activeFeatures.x2_sq) vec.push(y * y);
    if (activeFeatures.x1_x2) vec.push(x * y);
    if (activeFeatures.sin_x1) vec.push(Math.sin(Math.PI * x));
    if (activeFeatures.sin_x2) vec.push(Math.sin(Math.PI * y));
    return vec.length ? vec : [x, y];
  }

  function getLearningRate() {
    return 0.0005 * Math.pow(100, Number(ui.learningRate.value) / 100);
  }

  function generateDataset(kind) {
    randomSeed = (Math.floor(Math.random() * 1e9) ^ 0x51f15e) >>> 0;
    const noiseLevel = ui.noise ? Number(ui.noise.value) / 100 : 0.15;
    const out = [];
    const n = 320;

    for (let i = 0; i < n; i++) {
      let x = 0, y = 0, label = 0;
      if (kind === 'xor') {
        x = (random() * 2 - 1) * 0.88;
        y = (random() * 2 - 1) * 0.88;
        label = x * y >= 0 ? 1 : 0;
        x += randn() * (0.02 + noiseLevel * 0.2);
        y += randn() * (0.02 + noiseLevel * 0.2);
      } else if (kind === 'moons') {
        const upper = random() > 0.5;
        const t = random() * Math.PI;
        if (upper) {
          x = Math.cos(t) * 0.58 - 0.15;
          y = Math.sin(t) * 0.54 + 0.23;
          label = 0;
        } else {
          x = 1 - Math.cos(t) * 0.58 + 0.15;
          y = -0.23 - Math.sin(t) * 0.54;
          label = 1;
        }
        x += randn() * (0.04 + noiseLevel * 0.3);
        y += randn() * (0.04 + noiseLevel * 0.3);
      } else if (kind === 'circles') {
        const outer = random() > 0.5;
        const t = random() * Math.PI * 2;
        const r = outer ? 0.68 + randn() * (0.04 + noiseLevel * 0.2) : 0.31 + randn() * (0.04 + noiseLevel * 0.2);
        x = Math.cos(t) * r + randn() * (0.01 + noiseLevel * 0.1);
        y = Math.sin(t) * r + randn() * (0.01 + noiseLevel * 0.1);
        label = outer ? 1 : 0;
      } else {
        const arm = random() > 0.5 ? 1 : 0;
        const t = 0.16 + random() * 0.88;
        const angle = t * Math.PI * 2.3 + arm * Math.PI;
        const r = 0.08 + t * 0.72;
        x = Math.cos(angle) * r + randn() * (0.02 + noiseLevel * 0.2);
        y = Math.sin(angle) * r + randn() * (0.02 + noiseLevel * 0.2);
        label = arm;
      }
      out.push({ x: clamp(x, -1, 1), y: clamp(y, -1, 1), label });
    }

    allData = out;
    // 80/20 train/test split
    const splitIndex = Math.floor(allData.length * 0.8);
    trainData = allData.slice(0, splitIndex);
    testData = allData.slice(splitIndex);
  }

  function createModel() {
    const inputDim = extractFeatures(0, 0).length;
    const depth = Number(ui.depth.value);
    const hiddenSize = Number(ui.hidden.value);
    const activation = ui.activation.value;

    const sizes = [inputDim];
    for (let d = 0; d < depth; d++) {
      sizes.push(hiddenSize);
    }
    sizes.push(1); // Binary classification output (sigmoid)

    const weights = [];
    const biases = [];
    for (let l = 0; l < sizes.length - 1; l++) {
      const fanIn = sizes[l], fanOut = sizes[l + 1];
      const scale = Math.sqrt(2 / (fanIn + fanOut));
      weights.push(Array.from({ length: fanOut }, () =>
        Array.from({ length: fanIn }, () => randn() * scale)
      ));
      biases.push(Array.from({ length: fanOut }, () => 0));
    }

    const mW = weights.map(layer => layer.map(row => row.map(() => 0)));
    const vW = weights.map(layer => layer.map(row => row.map(() => 0)));
    const mB = biases.map(layer => layer.map(() => 0));
    const vB = biases.map(layer => layer.map(() => 0));

    model = {
      sizes,
      weights,
      biases,
      mW,
      vW,
      mB,
      vB,
      step: 0,
      activation
    };

    epoch = 0;
    trainHistory = [];
    testHistory = [];

    const trainMetrics = evaluate(trainData);
    const testMetrics = evaluate(testData);
    trainHistory.push(trainMetrics.loss);
    testHistory.push(testMetrics.loss);

    updateArchitectureText();
    updateMetricsUI(trainMetrics, testMetrics);
    renderAll();
  }

  function forward(x, y, save = false) {
    const inputVec = extractFeatures(x, y);
    const activations = [inputVec];
    const zs = [];
    let a = inputVec;

    for (let l = 0; l < model.weights.length; l++) {
      const z = model.weights[l].map((row, j) =>
        row.reduce((sum, w, i) => sum + w * a[i], model.biases[l][j])
      );
      zs.push(z);
      a = z.map(v =>
        l === model.weights.length - 1 ? sigmoid(v) : activate(v, model.activation)
      );
      activations.push(a);
    }

    return save ? { output: a[0], activations, zs } : a[0];
  }

  function evaluate(dataset) {
    if (!model || !dataset.length) return { loss: Math.log(2), accuracy: 0.5 };
    let loss = 0, correct = 0;
    for (const p of dataset) {
      const pred = clamp(forward(p.x, p.y), 1e-7, 1 - 1e-7);
      loss += -(p.label * Math.log(pred) + (1 - p.label) * Math.log(1 - pred));
      if ((pred >= 0.5 ? 1 : 0) === p.label) correct++;
    }
    return { loss: loss / dataset.length, accuracy: correct / dataset.length };
  }

  function trainEpoch() {
    if (!model || !trainData.length) return;
    const lr = getLearningRate();
    const batchSize = Number(ui.batchSize.value);
    const optType = ui.optimizer.value;

    const indices = Array.from({ length: trainData.length }, (_, i) => i);
    for (let i = indices.length - 1; i > 0; i--) {
      const j = Math.floor(random() * (i + 1));
      [indices[i], indices[j]] = [indices[j], indices[i]];
    }

    const gradW = model.weights.map(layer => layer.map(row => row.map(() => 0)));
    const gradB = model.biases.map(layer => layer.map(() => 0));

    for (let start = 0; start < indices.length; start += batchSize) {
      for (const layer of gradW) for (const row of layer) row.fill(0);
      for (const layer of gradB) layer.fill(0);

      const end = Math.min(indices.length, start + batchSize);
      const batchCount = end - start;

      for (let k = start; k < end; k++) {
        const p = trainData[indices[k]];
        const f = forward(p.x, p.y, true);
        const deltas = Array(model.weights.length);
        const last = model.weights.length - 1;

        // Output delta: binary cross entropy + sigmoid
        deltas[last] = [f.output - p.label];

        // Backprop through hidden layers
        for (let l = last - 1; l >= 0; l--) {
          deltas[l] = model.weights[l].map((_, j) => {
            let propagated = 0;
            for (let next = 0; next < deltas[l + 1].length; next++) {
              propagated += model.weights[l + 1][next][j] * deltas[l + 1][next];
            }
            return propagated * derivative(f.zs[l][j], f.activations[l + 1][j], model.activation);
          });
        }

        // Accumulate gradients
        for (let l = 0; l < model.weights.length; l++) {
          for (let j = 0; j < model.weights[l].length; j++) {
            gradB[l][j] += deltas[l][j];
            for (let i = 0; i < model.weights[l][j].length; i++) {
              gradW[l][j][i] += deltas[l][j] * f.activations[l][i];
            }
          }
        }
      }

      applyGradients(gradW, gradB, batchCount, lr, optType);
    }

    epoch++;
    const trainMetrics = evaluate(trainData);
    const testMetrics = evaluate(testData);

    trainHistory.push(trainMetrics.loss);
    testHistory.push(testMetrics.loss);
    if (trainHistory.length > 200) {
      trainHistory.shift();
      testHistory.shift();
    }

    updateMetricsUI(trainMetrics, testMetrics);
  }

  function applyGradients(gW, gB, count, lr, optType) {
    model.step++;
    const isAdam = optType === 'adam';
    const beta1 = 0.9, beta2 = 0.999, eps = 1e-8;
    const t = model.step;

    for (let l = 0; l < model.weights.length; l++) {
      for (let j = 0; j < model.weights[l].length; j++) {
        const gb = gB[l][j] / count;
        if (isAdam) {
          model.mB[l][j] = beta1 * model.mB[l][j] + (1 - beta1) * gb;
          model.vB[l][j] = beta2 * model.vB[l][j] + (1 - beta2) * gb * gb;
          const mh = model.mB[l][j] / (1 - Math.pow(beta1, t));
          const vh = model.vB[l][j] / (1 - Math.pow(beta2, t));
          model.biases[l][j] -= lr * mh / (Math.sqrt(vh) + eps);
        } else {
          model.biases[l][j] -= lr * gb;
        }

        for (let i = 0; i < model.weights[l][j].length; i++) {
          const g = gW[l][j][i] / count;
          if (isAdam) {
            model.mW[l][j][i] = beta1 * model.mW[l][j][i] + (1 - beta1) * g;
            model.vW[l][j][i] = beta2 * model.vW[l][j][i] + (1 - beta2) * g * g;
            const mh = model.mW[l][j][i] / (1 - Math.pow(beta1, t));
            const vh = model.vW[l][j][i] / (1 - Math.pow(beta2, t));
            model.weights[l][j][i] -= lr * mh / (Math.sqrt(vh) + eps);
          } else {
            model.weights[l][j][i] -= lr * g;
          }
        }
      }
    }
  }

  function fitCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    const dpr = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(rect.width * dpr));
    const height = Math.max(1, Math.round(rect.height * dpr));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const ctx = canvas.getContext('2d');
    return { ctx, width, height, dpr };
  }

  function drawSurface() {
    const { ctx, width: w, height: h } = fitCanvas(ui.surface);
    const res = 56;
    const imgData = ctx.createImageData(res, res);
    const dataBuf = imgData.data;

    for (let py = 0; py < res; py++) {
      const y = 1 - (py / (res - 1)) * 2;
      for (let px = 0; px < res; px++) {
        const x = (px / (res - 1)) * 2 - 1;
        const prob = forward(x, y);
        const idx = (py * res + px) * 4;

        // Class A: teal (#347f79), Class B: orange (#f15c3d)
        let r = Math.round(52 + (241 - 52) * prob);
        let g = Math.round(127 + (92 - 127) * prob);
        let b = Math.round(121 + (61 - 121) * prob);

        // Highlight the 0.5 decision boundary
        if (Math.abs(prob - 0.5) < 0.03) {
          r = 255; g = 255; b = 255;
        }

        dataBuf[idx] = r;
        dataBuf[idx + 1] = g;
        dataBuf[idx + 2] = b;
        dataBuf[idx + 3] = 210;
      }
    }

    const offscreen = document.createElement('canvas');
    offscreen.width = res;
    offscreen.height = res;
    offscreen.getContext('2d').putImageData(imgData, 0, 0);

    ctx.imageSmoothingEnabled = true;
    ctx.drawImage(offscreen, 0, 0, w, h);

    // Draw data points: Train data (filled dot), Test data (hollow circle)
    function drawPoint(p, isTest) {
      const px = ((p.x + 1) / 2) * w;
      const py = ((1 - p.y) / 2) * h;
      ctx.beginPath();
      ctx.arc(px, py, isTest ? 4.5 : 3.8, 0, Math.PI * 2);
      ctx.fillStyle = p.label === 1 ? '#fbfaf6' : '#142a27';
      ctx.strokeStyle = p.label === 1 ? '#f15c3d' : '#347f79';
      ctx.lineWidth = isTest ? 2.0 : 1.2;

      if (!isTest) {
        ctx.fillStyle = p.label === 1 ? '#f15c3d' : '#347f79';
        ctx.fill();
        ctx.strokeStyle = '#ffffff';
        ctx.stroke();
      } else {
        ctx.fillStyle = 'rgba(255, 255, 255, 0.45)';
        ctx.fill();
        ctx.stroke();
      }
    }

    trainData.forEach(p => drawPoint(p, false));
    testData.forEach(p => drawPoint(p, true));
  }

  function drawNetwork() {
    const { ctx, width: w, height: h } = fitCanvas(ui.network);
    ctx.clearRect(0, 0, w, h);
    if (!model) return;

    const padX = 26;
    const top = 26, bottom = 22;
    const layerCount = model.sizes.length;
    const xs = model.sizes.map((_, i) =>
      padX + (w - padX * 2) * (layerCount === 1 ? 0.5 : i / (layerCount - 1))
    );

    const positions = model.sizes.map((count, l) => {
      const displayCount = Math.min(count, 8);
      const spacing = Math.min(24, (h - top - bottom) / Math.max(1, displayCount - 1));
      const total = spacing * (displayCount - 1);
      return Array.from({ length: displayCount }, (_, i) => ({
        x: xs[l],
        y: h / 2 - total / 2 + i * spacing
      }));
    });

    // Draw connection weights
    for (let l = 0; l < model.weights.length; l++) {
      for (let j = 0; j < model.weights[l].length; j++) {
        for (let i = 0; i < model.weights[l][j].length; i++) {
          if (i >= positions[l].length || j >= positions[l + 1].length) continue;
          const a = positions[l][i], b = positions[l + 1][j];
          const val = model.weights[l][j][i];

          ctx.beginPath();
          ctx.moveTo(a.x, a.y);
          ctx.lineTo(b.x, b.y);
          ctx.strokeStyle = val >= 0 ? 'rgba(241,92,61,0.48)' : 'rgba(52,127,121,0.48)';
          ctx.lineWidth = 0.5 + Math.min(2.2, Math.abs(val) * 0.5);
          ctx.stroke();
        }
      }
    }

    // Draw neuron nodes
    positions.forEach((layer, l) => {
      layer.forEach(p => {
        ctx.beginPath();
        ctx.arc(p.x, p.y, l === 0 || l === layerCount - 1 ? 5.8 : 5.0, 0, Math.PI * 2);
        ctx.fillStyle = l === layerCount - 1 ? '#fdf0ec' : l === 0 ? '#f0f3ee' : '#ffffff';
        ctx.fill();
        ctx.strokeStyle = l === layerCount - 1 ? '#f15c3d' : l === 0 ? '#347f79' : '#a2a79d';
        ctx.lineWidth = 1.3;
        ctx.stroke();
      });

      if (model.sizes[l] > layer.length) {
        const last = layer[layer.length - 1];
        ctx.fillStyle = '#999e94';
        ctx.font = '8px DM Mono, monospace';
        ctx.textAlign = 'center';
        ctx.fillText(`+${model.sizes[l] - layer.length}`, last.x, Math.min(h - 10, last.y + 16));
      }

      ctx.fillStyle = '#8f948a';
      ctx.font = '8px DM Mono, monospace';
      ctx.textAlign = 'center';
      const labelText = l === 0 ? 'INPUT' : l === layerCount - 1 ? 'OUTPUT' : `H${l}`;
      ctx.fillText(labelText, xs[l], 12);
    });
  }

  function drawLoss() {
    const { ctx, width: w, height: h } = fitCanvas(ui.chart);
    ctx.clearRect(0, 0, w, h);
    const left = 6, right = 6, top = 8, bottom = 12;
    const pw = w - left - right, ph = h - top - bottom;

    // Gridlines
    for (let i = 0; i < 3; i++) {
      const y = top + (ph * i) / 2;
      ctx.beginPath();
      ctx.moveTo(left, y);
      ctx.lineTo(w - right, y);
      ctx.strokeStyle = '#eae7df';
      ctx.lineWidth = 1;
      ctx.stroke();
    }

    if (trainHistory.length < 2) {
      ctx.fillStyle = '#a3a69d';
      ctx.font = '9px DM Mono, monospace';
      ctx.textAlign = 'center';
      ctx.fillText('LOSS TRACE WILL APPEAR HERE', w / 2, h / 2 + 3);
      return;
    }

    const allLosses = trainHistory.concat(testHistory);
    const max = Math.max(0.7, Math.min(2.5, Math.max(...allLosses) * 1.05));
    const min = Math.max(0, Math.min(...allLosses) * 0.85);

    function plotCurve(series, color, dashed = false) {
      ctx.beginPath();
      series.forEach((val, i) => {
        const x = left + (pw * i) / (series.length - 1);
        const y = top + ph * (1 - (clamp(val, min, max) - min) / Math.max(0.001, max - min));
        if (i === 0) ctx.moveTo(x, y);
        else ctx.lineTo(x, y);
      });
      ctx.strokeStyle = color;
      ctx.lineWidth = dashed ? 1.6 : 1.8;
      if (dashed) ctx.setLineDash([4, 3]);
      else ctx.setLineDash([]);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Plot test loss (teal dashed) and train loss (orange solid)
    plotCurve(testHistory, '#347f79', true);
    plotCurve(trainHistory, '#f15c3d', false);

    const latestTrain = trainHistory[trainHistory.length - 1];
    const latestTest = testHistory[testHistory.length - 1];
    ui.chartCurrent.textContent = `TRAIN: ${latestTrain.toFixed(3)} | TEST: ${latestTest.toFixed(3)}`;
  }

  function updateMetricsUI(trainMetrics, testMetrics) {
    ui.epoch.textContent = String(epoch).padStart(4, '0');
    ui.loss.innerHTML = `${trainMetrics.loss.toFixed(3)} <em>/ ${testMetrics.loss.toFixed(3)}</em>`;
    ui.accuracy.innerHTML = `${(trainMetrics.accuracy * 100).toFixed(1)}% <em>/ ${(testMetrics.accuracy * 100).toFixed(1)}%</em>`;
    ui.confidence.textContent = `${(testMetrics.accuracy * 100).toFixed(1)}%`;
  }

  function updateArchitectureText() {
    if (!model) return;
    const archString = model.sizes.join('–');
    const heroArchString = model.sizes.join(' → ');
    ui.architectureLabel.textContent = archString;
    if (ui.heroArchMeta) ui.heroArchMeta.innerHTML = heroArchString;

    let paramCount = 0;
    for (let l = 0; l < model.weights.length; l++) {
      paramCount += model.weights[l].length * model.weights[l][0].length;
      paramCount += model.biases[l].length;
    }
    ui.parameters.textContent = String(paramCount);

    const activeCount = Object.values(activeFeatures).filter(Boolean).length;
    ui.featuresCount.textContent = `${activeCount} active`;
  }

  function renderAll() {
    drawSurface();
    drawNetwork();
    drawLoss();
  }

  function animate() {
    if (!isTraining) return;
    trainEpoch();
    renderAll();
    animationId = requestAnimationFrame(animate);
  }

  function setTrainingState(running) {
    isTraining = running;
    if (running) {
      ui.status.classList.add('running');
      ui.status.innerHTML = '<i></i> TRAINING';
      ui.trainLabel.textContent = 'Pause';
      ui.train.querySelector('.play-icon').textContent = '❚❚';
      ui.trainingMessage.textContent = 'Optimizer is updating parameters in real time.';
      animate();
    } else {
      ui.status.classList.remove('running');
      ui.status.innerHTML = '<i></i> PAUSED';
      ui.trainLabel.textContent = 'Start training';
      ui.train.querySelector('.play-icon').textContent = '▶';
      ui.trainingMessage.textContent = 'Paused. Resume or step through epochs.';
      if (animationId) cancelAnimationFrame(animationId);
    }
  }

  // Event Listeners
  ui.dataset.addEventListener('change', () => {
    const kind = ui.dataset.value;
    const desc = datasetDescriptions[kind] || datasetDescriptions.xor;
    ui.datasetHint.textContent = desc[0];
    ui.surfaceCaption.textContent = desc[1];
    ui.insightText.textContent = desc[2];
    generateDataset(kind);
    createModel();
  });

  ui.featureToggles.addEventListener('click', (e) => {
    const btn = e.target.closest('.feature-pill');
    if (!btn) return;
    const featKey = btn.dataset.feature;
    if (!featKey) return;

    activeFeatures[featKey] = !activeFeatures[featKey];
    // Ensure at least one feature is active
    if (!Object.values(activeFeatures).some(Boolean)) {
      activeFeatures[featKey] = true;
    }

    document.querySelectorAll('.feature-pill').forEach(pill => {
      const k = pill.dataset.feature;
      if (activeFeatures[k]) pill.classList.add('active');
      else pill.classList.remove('active');
    });

    createModel();
  });

  ui.depth.addEventListener('change', () => {
    ui.depthValue.textContent = `${ui.depth.value} layer${ui.depth.value > 1 ? 's' : ''}`;
    createModel();
  });

  ui.hidden.addEventListener('input', () => {
    ui.hiddenValue.textContent = ui.hidden.value;
    createModel();
  });

  ui.activation.addEventListener('change', () => {
    createModel();
  });

  ui.learningRate.addEventListener('input', () => {
    ui.lrValue.textContent = getLearningRate().toFixed(4);
  });

  if (ui.noise) {
    ui.noise.addEventListener('input', () => {
      ui.noiseValue.textContent = `${ui.noise.value}%`;
      generateDataset(ui.dataset.value);
      createModel();
    });
  }

  ui.train.addEventListener('click', () => {
    setTrainingState(!isTraining);
  });

  ui.step.addEventListener('click', () => {
    if (isTraining) setTrainingState(false);
    trainEpoch();
    renderAll();
  });

  ui.reset.addEventListener('click', () => {
    setTrainingState(false);
    createModel();
  });

  ui.regenerate.addEventListener('click', () => {
    generateDataset(ui.dataset.value);
    createModel();
  });

  // Interactive Click to Add Point on Decision Surface
  ui.surface.addEventListener('click', (e) => {
    const rect = ui.surface.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const x = clamp(px * 2 - 1, -1, 1);
    const y = clamp(1 - py * 2, -1, 1);
    const label = e.shiftKey ? 1 : 0;

    const newPt = { x, y, label };
    allData.push(newPt);
    trainData.push(newPt);

    const trainMetrics = evaluate(trainData);
    const testMetrics = evaluate(testData);
    updateMetricsUI(trainMetrics, testMetrics);
    renderAll();
  });

  // Hover Probe on Surface Canvas
  ui.surface.addEventListener('mousemove', (e) => {
    if (!ui.canvasProbe) return;
    const rect = ui.surface.getBoundingClientRect();
    const px = (e.clientX - rect.left) / rect.width;
    const py = (e.clientY - rect.top) / rect.height;
    const x = clamp(px * 2 - 1, -1, 1);
    const y = clamp(1 - py * 2, -1, 1);
    const prob = forward(x, y);

    ui.canvasProbe.style.display = 'block';
    ui.canvasProbe.style.left = `${Math.min(rect.width - 160, e.clientX - rect.left + 14)}px`;
    ui.canvasProbe.style.top = `${e.clientY - rect.top}px`;
    ui.canvasProbe.innerHTML = `(x: ${x >= 0 ? '+' : ''}${x.toFixed(2)}, y: ${y >= 0 ? '+' : ''}${y.toFixed(2)}) · P(B): <b>${(prob * 100).toFixed(1)}%</b>`;
  });

  ui.surface.addEventListener('mouseleave', () => {
    if (ui.canvasProbe) ui.canvasProbe.style.display = 'none';
  });

  // Export Weights
  if (ui.exportWeights) {
    ui.exportWeights.addEventListener('click', () => {
      if (!model) return;
      const payload = {
        name: "Tensile Model Checkpoint",
        architecture: model.sizes,
        activation: model.activation,
        weights: model.weights,
        biases: model.biases,
        step: model.step,
        exported_at: new Date().toISOString()
      };
      const blob = new Blob([JSON.stringify(payload, null, 2)], { type: 'application/json' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `tensile_weights_${model.sizes.join('-')}.json`;
      a.click();
      URL.revokeObjectURL(url);
    });
  }

  window.addEventListener('resize', renderAll);

  // Initialize
  generateDataset('xor');
  createModel();
})();
