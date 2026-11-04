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
})();
