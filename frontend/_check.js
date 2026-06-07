
const API = 'http://localhost:8000';
const CLASS_COLORS = ['#00c800','#c8c800','#dc5000','#dc0000','#c800c8','#0064c8','#c80064','#00c8c8','#969696','#333'];

// Цвета для каждого из 8 режимов (для наложенных графиков)
const MODE_KEYS   = ['sequential','parallel_cpu','parallel_images','parallel_gpu','parallel_images_gpu','parallel_combined','parallel_combined_gpu','parallel_batch_embed'];
const MODE_COLORS_8 = ['#555555','#2255cc','#22aacc','#228800','#00cc44','#9922cc','#cc5500','#cc0066'];
const MODE_SHORT  = ['Seq','CPU Par','Img Par','GPU','Img+GPU','Combined','Batch GPU','Batch Emb'];

const charts = {};

// ── NDJSON stream ─────────────────────────────────────────────
async function streamNDJSON(url, body, onChunk) {
  const res = await fetch(url, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify(body)
  });
  if (!res.ok) throw new Error(`HTTP ${res.status}`);
  const reader = res.body.getReader();
  const dec = new TextDecoder();
  let buf = '';
  while (true) {
    const {done, value} = await reader.read();
    if (done) break;
    buf += dec.decode(value, {stream: true});
    const lines = buf.split('\n');
    buf = lines.pop();
    for (const l of lines) { if (l.trim()) onChunk(JSON.parse(l)); }
  }
}

// ── Chart helpers ─────────────────────────────────────────────
const FONT = {family: 'Manrope', size: 11};
const TICK = '#777';
const GRID = 'rgba(0,0,0,0.07)';

function makeOpts(xLabel, yLabel, horizontal) {
  const indexAxis = horizontal ? 'y' : 'x';
  return {
    responsive: true, maintainAspectRatio: false, indexAxis,
    plugins: { legend: { labels: { color: '#333', font: FONT, boxWidth: 12 } } },
    scales: {
      x: { ticks: { color: TICK, font: FONT }, grid: { color: GRID },
           title: { display: !!xLabel, text: xLabel, color: TICK, font: FONT } },
      y: { ticks: { color: TICK, font: FONT }, grid: { color: GRID },
           title: { display: !!yLabel, text: yLabel, color: TICK, font: FONT } }
    }
  };
}

function mkChart(id, type, opts) {
  if (charts[id]) charts[id].destroy();
  const el = document.getElementById(id);
  if (!el) return null;
  charts[id] = new Chart(el, { type, data: { labels: [], datasets: [] }, options: opts });
  return charts[id];
}

// ── Init ─────────────────────────────────────────────────────
async function init() {
  Chart.defaults.font.family = 'Manrope';

  // Tab 1 row 1
  mkChart('ca-amdahl',      'line', makeOpts('Воркеры', 'Speedup'));
  mkChart('ca-timing',      'bar',  makeOpts(null, 'Время, мс', true));
  mkChart('ca-speedup-hist','bar',  makeOpts(null, 'Speedup', true));
  // Tab 1 row 2
  mkChart('ca-dist',   'bar',     makeOpts('Класс', 'Количество'));
  mkChart('ca-pfrac',  'bar',     makeOpts('Прогон', 'Доля p'));
  mkChart('ca-scatter','scatter', makeOpts('Дефектов', 'Время, мс'));
  // Tab 1 row 3 (dir)
  mkChart('cd-timings','bar',  makeOpts('Фото №', 'Время, мс'));
  mkChart('cd-cumul',  'line', makeOpts('Фото №', 'Накопл. время, мс'));
  mkChart('cd-defects','bar',  makeOpts('Фото №', 'Дефектов'));
  // Tab 2 row 1
  mkChart('cc-speedup', 'bar', makeOpts(null, 'Speedup', true));
  mkChart('cc-time',    'bar', makeOpts(null, 'Wall time, мс', true));
  // Tab 2 row 2 (scaling)
  mkChart('cc-scaling', 'line', makeOpts('Число изображений', 'Время, мс'));
  // Tab 2 row 3
  mkChart('cc-avgtime',   'bar', makeOpts(null, 'Ср. время/фото, мс', true));
  mkChart('cc-defects',   'bar', makeOpts(null, 'Дефектов', true));
  mkChart('cc-efficiency','bar', makeOpts(null, 'Эффективность, %', true));
  // Tab 2 row 4 — per-image combined overlay
  mkChart('cc-per-img', 'line', makeOpts('Изображение №', 'Время, мс'));
  // Tab 2 row 5 — 8 individual mini charts (timing)
  for (let i = 0; i < 8; i++) mkChart('cc-m' + i, 'bar', makeOpts('Фото №', 'Время, мс'));
  // Tab 2 row 6 — cumulative + speedup per image overlay
  mkChart('cc-cumul-all',       'line', makeOpts('Изображение №', 'Накопл. время, мс'));
  mkChart('cc-speedup-per-img', 'line', makeOpts('Изображение №', 'Speedup'));
  // Tab 2 row 7 — summary charts
  mkChart('cc-minmaxavg',  'bar',    makeOpts(null, 'Время, мс', true));
  mkChart('cc-defects-pie','doughnut', {responsive:true, maintainAspectRatio:false,
    plugins:{legend:{position:'right', labels:{color:'#333',font:FONT,boxWidth:12}}}});
  mkChart('cc-norm-bar',   'bar',    makeOpts(null, 'Относительно Sequential', true));
  // Tab 2 row 8 — 8 cumulative mini charts
  for (let i = 0; i < 8; i++) mkChart('cc-c' + i, 'line', makeOpts('Фото №', 'Накопл., мс'));

  await Promise.all([loadImages(), loadSubfolders(), loadHistory(), loadBinarizeParams()]);
  bindEvents();
}

async function loadImages() {
  try {
    const d = await fetch(`${API}/images`).then(r => r.json());
    const opts = d.images.map(p => `<option value="${p}">${p}</option>`).join('');
    document.getElementById('a-image').innerHTML = opts;
  } catch { document.getElementById('a-image').innerHTML = '<option>Сервер недоступен</option>'; }
}

async function loadSubfolders() {
  try {
    const d = await fetch(`${API}/subfolders`).then(r => r.json());
    const opts = '<option value="">Все изображения</option>' +
      d.subfolders.map(s => `<option value="${s}">${s}</option>`).join('');
    ['c-folder','d-folder','oa-folder','ob-folder','cl-folder','od-folder'].forEach(id => {
      const el = document.getElementById(id);
      if (el) el.innerHTML = opts;
    });
  } catch {}
}

async function loadBinarizeParams() {
  try {
    const d = await fetch(`${API}/binarize_params`).then(r => r.json());
    document.getElementById('bp-thr').value    = d.threshold ?? 0.5;
    document.getElementById('bp-area').value   = d.min_area ?? 0;
    document.getElementById('bp-kernel').value = d.morphology_kernel ?? 3;
    document.getElementById('bp-iters').value  = d.morphology_iterations ?? 1;
    document.getElementById('bp-blur').value   = d.gaussian_blur_kernel ?? 0;
  } catch {}
}

async function loadHistory() {
  try {
    const d = await fetch(`${API}/history`).then(r => r.json());
    const runs = d.runs || [];
    const tbody = document.getElementById('h-body');
    if (!runs.length) {
      tbody.innerHTML = '<tr><td colspan="7" style="text-align:center;padding:16px;color:var(--muted)">История пуста</td></tr>';
    } else {
      tbody.innerHTML = runs.slice(-60).reverse().map(r => `<tr>
        <td>${(r.timestamp||'—').replace('T',' ')}</td>
        <td style="max-width:180px;overflow:hidden;text-overflow:ellipsis">${r.image||'—'}</td>
        <td>${r.mode||'—'}</td>
        <td>${r.workers||'—'}</td>
        <td>${r.n_defects_after_filter ?? r.n_defects ?? '—'}</td>
        <td>${r.time_ms != null ? r.time_ms.toFixed(1) : '—'}</td>
        <td>${r.speedup != null ? r.speedup.toFixed(2)+'×' : '—'}</td>
      </tr>`).join('');
    }
    updateAmdahlChart(runs);
    updateTimingHistChart(runs);
    updateSpeedupHistChart(runs);
    updatePFracChart(runs);
    updateScatterChart(runs);
  } catch {}
}

// ── Bind events ───────────────────────────────────────────────
function bindEvents() {
  // Main tabs
  document.querySelectorAll('.tab-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('tab-' + btn.dataset.tab).classList.add('active');
    })
  );

  // Inner tabs (Фото / Директория)
  document.querySelectorAll('.inner-tab-btn').forEach(btn =>
    btn.addEventListener('click', () => {
      document.querySelectorAll('.inner-tab-btn').forEach(b => b.classList.remove('active'));
      document.querySelectorAll('.inner-tab-pane').forEach(p => p.classList.remove('active'));
      btn.classList.add('active');
      document.getElementById('atab-' + btn.dataset.atab).classList.add('active');
    })
  );

  // Tab 1 single image
  document.querySelector('input[name=a-mode][value=parallel_cpu]')
    .addEventListener('change', () => document.getElementById('a-workers-row').classList.remove('hidden'));
  document.querySelectorAll('input[name=a-mode]:not([value=parallel_cpu])').forEach(r =>
    r.addEventListener('change', () => document.getElementById('a-workers-row').classList.add('hidden')));
  document.getElementById('a-thr').addEventListener('input', e =>
    document.getElementById('a-thr-val').textContent = parseFloat(e.target.value).toFixed(2));
  document.getElementById('a-workers').addEventListener('input', e =>
    document.getElementById('a-workers-val').innerHTML = e.target.value + ' <small>workers</small>');
  document.getElementById('a-btn').addEventListener('click', runAnalysis);

  // Tab 1 directory
  document.getElementById('d-btn').addEventListener('click', runDirAnalysis);

  // Tab 2
  document.getElementById('c-workers').addEventListener('input', e =>
    document.getElementById('c-workers-val').innerHTML = e.target.value + ' <small>workers</small>');
  document.getElementById('c-btn').addEventListener('click', runCompare);

  // Tab 3
  document.getElementById('oa-btn').addEventListener('click', runOptimize);
  document.getElementById('ob-btn').addEventListener('click', runBinarizeOptimize);
  document.getElementById('bp-btn').addEventListener('click', saveBinarizeParams);
  document.getElementById('bp-reload-btn').addEventListener('click', loadBinarizeParams);

  // Tab 4
  document.getElementById('cal-pct').addEventListener('input', e =>
    document.getElementById('cal-pct-val').textContent = e.target.value);
  document.getElementById('cal-btn').addEventListener('click', runCalibrate);

  // History
  document.getElementById('h-clear-btn').addEventListener('click', async () => {
    if (!confirm('Очистить всю историю прогонов?')) return;
    await fetch(`${API}/history`, {method: 'DELETE'});
    loadHistory();
  });

  // ── Глобальная синхронизация параметров ───────────────────────
  // Поля с data-global="s_min"/"s_max"/"threshold"/"k" синхронизируются между вкладками.
  const _syncGlobal = (sourceEl) => {
    const key = sourceEl.dataset.global;
    const val = sourceEl.value;
    document.querySelectorAll(`[data-global="${key}"]`).forEach(other => {
      if (other !== sourceEl) other.value = val;
    });
    if (key === 'threshold') {
      ['a-thr-val', 'cl-thr-val'].forEach(id => {
        const el = document.getElementById(id);
        if (el) el.textContent = parseFloat(val).toFixed(2);
      });
    }
  };
  document.querySelectorAll('[data-global]').forEach(input => {
    input.addEventListener('change', () => _syncGlobal(input));
    if (input.type === 'range') input.addEventListener('input', () => _syncGlobal(input));
  });

  // Tab 3 — Классификация
  const autoKEl = document.getElementById('cl-autok');
  if (autoKEl) {
    const updateAutoK = () => {
      const auto = autoKEl.checked;
      document.getElementById('cl-k').disabled = auto;
      document.getElementById('cl-kmin').disabled = !auto;
      document.getElementById('cl-kmax').disabled = !auto;
    };
    autoKEl.addEventListener('change', updateAutoK);
    updateAutoK();
  }
  const clThrEl = document.getElementById('cl-thr');
  if (clThrEl) clThrEl.addEventListener('input', e => {
    document.getElementById('cl-thr-val').textContent = parseFloat(e.target.value).toFixed(2);
  });
  const clBtnEl = document.getElementById('cl-btn');
  if (clBtnEl) clBtnEl.addEventListener('click', () => {
    if (_clArch === 'emb') runClassifyEmbeddings();
    else runClassify();
  });
  const clBenchEl = document.getElementById('cl-bench-btn');
  if (clBenchEl) clBenchEl.addEventListener('click', runClassifyBenchmark);
  const odBtnEl = document.getElementById('od-btn');
  if (odBtnEl) odBtnEl.addEventListener('click', runOptimizeClassify);
}

// ════════════════════════ TAB 1 — ОДНО ФОТО ══════════════════════════
async function runAnalysis() {
  const imgPath = document.getElementById('a-image').value;
  if (!imgPath) return;
  const btn = document.getElementById('a-btn');
  const status = document.getElementById('a-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Анализ...';
  try {
    const res = await fetch(`${API}/analyze`, {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({
        image_path: imgPath,
        mode:      document.querySelector('input[name=a-mode]:checked').value,
        workers:   parseInt(document.getElementById('a-workers').value),
        S_min:     parseFloat(document.getElementById('a-smin').value),
        S_max:     parseFloat(document.getElementById('a-smax').value),
        K:         parseInt(document.getElementById('a-k').value),
        threshold: parseFloat(document.getElementById('a-thr').value),
      })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    showAnalysisResults(imgPath, data);
    updateDistChart(data.defects, parseInt(document.getElementById('a-k').value));
    await loadHistory();
    status.textContent = 'Готово';
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

function showAnalysisResults(imgPath, d) {
  document.getElementById('a-img-orig').src = `${API}/original/${imgPath}`;
  document.getElementById('a-img-ann').src  = 'data:image/png;base64,' + d.annotated_image;
  const sp  = d.speedup  != null ? d.speedup.toFixed(2) + '×' : '—';
  const pf  = d.parallel_fraction != null ? (d.parallel_fraction * 100).toFixed(1) + '%' : '—';
  const dev = d.device || 'cpu';
  document.getElementById('a-chips').innerHTML = `
    <div class="chip"><div class="v">${d.n_defects_after_filter}</div><div class="l">Дефектов</div></div>
    <div class="chip"><div class="v">${d.time_ms.toFixed(1)}</div><div class="l">Время, мс</div></div>
    <div class="chip chip-acc"><div class="v">${sp}</div><div class="l">Speedup</div></div>
    <div class="chip"><div class="v">${pf}</div><div class="l">Парал. доля</div></div>
    <div class="chip"><div class="v" style="font-size:0.85rem">${dev}</div><div class="l">Устройство</div></div>
  `;
  const K = parseInt(document.getElementById('a-k').value);
  document.getElementById('a-defects').innerHTML = !d.defects.length
    ? '<tr><td colspan="5" style="text-align:center;padding:14px;color:var(--muted)">Дефектов не найдено</td></tr>'
    : d.defects.map(df => `<tr>
        <td>${df.id}</td><td>${df.area.toFixed(1)}</td><td>${df.diameter.toFixed(1)}</td>
        <td>${(df.major_axis||0).toFixed(1)}</td>
        <td><span class="dot" style="background:${CLASS_COLORS[df.class]||'#888'}"></span>${df.class}</td>
      </tr>`).join('');
  document.getElementById('a-results').classList.remove('hidden');
}

// ════════════════════════ TAB 1 — ДИРЕКТОРИЯ ══════════════════════════
async function runDirAnalysis() {
  const btn    = document.getElementById('d-btn');
  const status = document.getElementById('d-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Запуск...';

  document.getElementById('d-results').classList.remove('hidden');
  document.getElementById('d-rows').innerHTML = '';
  document.getElementById('d-summary-chips').classList.add('hidden');
  document.getElementById('d-bar').style.width = '0%';
  document.getElementById('d-progress-text').textContent = 'Инициализация...';
  document.getElementById('d-charts-card').classList.remove('hidden');

  const timings  = [];
  const defCounts = [];
  const labels   = [];
  let total = 0, done = 0;
  let cumulMs = 0;
  const cumulData = [];

  try {
    await streamNDJSON(`${API}/analyze_dir`, {
      subfolder:  document.getElementById('d-folder').value,
      mode:       document.querySelector('input[name=d-mode]:checked').value,
      workers:    parseInt(document.getElementById('d-workers').value),
      S_min:      parseFloat(document.getElementById('d-smin').value),
      S_max:      parseFloat(document.getElementById('d-smax').value),
      K:          parseInt(document.getElementById('d-k').value),
      threshold:  parseFloat(document.getElementById('d-thr').value),
      max_images: parseInt(document.getElementById('d-maximg').value),
    }, chunk => {
      if (chunk.type === 'start') {
        total = chunk.total;
        document.getElementById('d-progress-text').textContent = `0 / ${total} изображений`;
      } else if (chunk.type === 'image') {
        done++;
        const r = chunk.result;
        const pct = total ? Math.round(done / total * 100) : 0;
        document.getElementById('d-bar').style.width = pct + '%';
        document.getElementById('d-progress-text').textContent = `${done} / ${total} (${pct}%)`;

        // накапливаем данные для графиков
        labels.push(done);
        timings.push(+(r.time_ms || 0).toFixed(1));
        defCounts.push(r.n_defects_filtered || 0);
        cumulMs += (r.time_ms || 0);
        cumulData.push(+cumulMs.toFixed(1));

        // добавляем строку в таблицу
        addDirRow(r, done);
        updateDirCharts(labels, timings, cumulData, defCounts);
      } else if (chunk.type === 'done') {
        const s = chunk.summary;
        document.getElementById('d-bar').style.width = '100%';
        document.getElementById('d-progress-text').textContent = `Готово — ${s.total_images} фото`;
        const chips = document.getElementById('d-summary-chips');
        chips.innerHTML = `
          <div class="chip"><div class="v">${s.total_images}</div><div class="l">Фото</div></div>
          <div class="chip"><div class="v">${s.total_defects_filtered}</div><div class="l">Дефектов</div></div>
          <div class="chip"><div class="v">${s.avg_time_ms.toFixed(1)}</div><div class="l">Ср. время, мс</div></div>
          <div class="chip chip-acc"><div class="v">${s.wall_time_ms.toFixed(0)}</div><div class="l">Wall time, мс</div></div>
        `;
        chips.classList.remove('hidden');
      }
    });
    status.textContent = 'Готово';
    await loadHistory();
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

function addDirRow(r, idx) {
  const tbody = document.getElementById('d-rows');
  const sp    = r.speedup != null ? r.speedup.toFixed(2) + '×' : '—';
  const tag   = r.classified_as === 'defective'
    ? '<span class="tag-def tag-bad">Дефект</span>'
    : '<span class="tag-def tag-ok">Норма</span>';
  const name  = (r.image || '').split('/').pop();
  const tr = document.createElement('tr');
  tr.innerHTML = `<td>${idx}</td><td title="${r.image}">${name}</td><td>${r.n_defects_filtered}</td><td>${(r.time_ms||0).toFixed(1)}</td><td>${sp}</td><td>${tag}</td>`;
  tbody.appendChild(tr);
  // авто-скролл последней строки
  tbody.parentElement.scrollTop = tbody.parentElement.scrollHeight;
}

// ════════════════════════ CHARTS TAB 1 ══════════════════════════

function updateAmdahlChart(runs) {
  const N = [1,2,4,8];
  const real = N.map(w => {
    if (w === 1) return 1.0;
    const m = runs.filter(r => r.mode === 'parallel_cpu' && r.workers === w && r.speedup != null);
    return m.length ? m.reduce((s,r) => s+r.speedup, 0)/m.length : null;
  });
  const pVals = runs.filter(r => r.parallel_fraction != null).map(r => r.parallel_fraction);
  const p = pVals.length ? pVals.reduce((a,b)=>a+b,0)/pVals.length : 0.8;
  const theory = N.map(n => 1/((1-p)+p/n));
  const c = charts['ca-amdahl'];
  if (!c) return;
  c.data.labels = N;
  c.data.datasets = [
    { label:'Теория (p='+p.toFixed(2)+')', data:theory, borderColor:'#0f0f0f', borderDash:[5,4], pointRadius:3, tension:0.3, fill:false },
    { label:'Реальный', data:real, borderColor:'#339900', pointRadius:5, pointBackgroundColor:'#72ff00', tension:0.3, fill:false, spanGaps:true }
  ];
  c.update();
}

function updateTimingHistChart(runs) {
  const modes = [...new Set(runs.map(r => r.mode).filter(Boolean))];
  const avgs  = modes.map(m => {
    const ms = runs.filter(r => r.mode === m && r.time_ms != null).map(r => r.time_ms);
    return ms.length ? ms.reduce((a,b)=>a+b,0)/ms.length : 0;
  });
  const c = charts['ca-timing'];
  if (!c) return;
  c.data.labels = modes.map(m => MODE_SHORT[MODE_KEYS.indexOf(m)] || m);
  c.data.datasets = [{
    label:'Ср. время, мс', data: avgs,
    backgroundColor: modes.map(m => { const i=MODE_KEYS.indexOf(m); return (i>=0?MODE_COLORS_8[i]:'#888')+'99'; }),
    borderColor: modes.map(m => MODE_COLORS_8[MODE_KEYS.indexOf(m)] || '#888'),
    borderWidth:1.5, borderRadius:0
  }];
  c.update();
}

function updateSpeedupHistChart(runs) {
  const modes = [...new Set(runs.map(r => r.mode).filter(Boolean))];
  const avgs  = modes.map(m => {
    const sp = runs.filter(r => r.mode === m && r.speedup != null).map(r => r.speedup);
    return sp.length ? sp.reduce((a,b)=>a+b,0)/sp.length : (m==='sequential'?1.0:0);
  });
  const c = charts['ca-speedup-hist'];
  if (!c) return;
  c.data.labels = modes.map(m => MODE_SHORT[MODE_KEYS.indexOf(m)] || m);
  c.data.datasets = [{
    label:'Ср. speedup', data: avgs,
    backgroundColor: avgs.map((v,i) => (v>1?'#72ff0099':'#0f0f0f33')),
    borderColor:'#0f0f0f', borderWidth:1.5, borderRadius:0
  }];
  c.update();
}

function updateDistChart(defects, K) {
  const counts = Array(K).fill(0);
  defects.forEach(d => { if (d.class < K) counts[d.class]++; });
  const c = charts['ca-dist'];
  if (!c) return;
  c.data.labels = counts.map((_,i) => `C${i}`);
  c.data.datasets = [{ label:'Дефектов', data:counts,
    backgroundColor: counts.map((_,i)=>CLASS_COLORS[i]+'cc'),
    borderColor:'#0f0f0f', borderWidth:1.5, borderRadius:0 }];
  c.update();
}

function updatePFracChart(runs) {
  const pts = runs.filter(r => r.parallel_fraction != null).slice(-20);
  const c = charts['ca-pfrac'];
  if (!c) return;
  c.data.labels = pts.map((_,i)=>i+1);
  c.data.datasets = [{
    label:'p', data: pts.map(r=>+(r.parallel_fraction||0).toFixed(3)),
    backgroundColor:'#72ff0099', borderColor:'#0f0f0f', borderWidth:1.5, borderRadius:0
  }];
  c.update();
}

function updateScatterChart(runs) {
  const pts = runs.filter(r => r.time_ms != null && r.n_defects_after_filter != null).slice(-50);
  const c = charts['ca-scatter'];
  if (!c) return;
  const modeMap = {};
  pts.forEach(r => {
    const m = r.mode || 'unknown';
    if (!modeMap[m]) modeMap[m] = [];
    modeMap[m].push({x: r.n_defects_after_filter, y: r.time_ms});
  });
  c.data.datasets = Object.entries(modeMap).map(([m, pts]) => {
    const i = MODE_KEYS.indexOf(m);
    return { label: MODE_SHORT[i]||m, data: pts, pointRadius:4,
      backgroundColor:(i>=0?MODE_COLORS_8[i]:'#888')+'cc',
      borderColor:(i>=0?MODE_COLORS_8[i]:'#888') };
  });
  c.update();
}

function updateDirCharts(labels, timings, cumulData, defCounts) {
  const ct = charts['cd-timings'];
  if (ct) {
    ct.data.labels = labels;
    ct.data.datasets = [{ label:'Время, мс', data:timings,
      backgroundColor:'#0f0f0f33', borderColor:'#0f0f0f', borderWidth:1, borderRadius:0 }];
    ct.update('none');
  }
  const cc = charts['cd-cumul'];
  if (cc) {
    cc.data.labels = labels;
    cc.data.datasets = [{ label:'Кум. время', data:cumulData,
      borderColor:'#339900', pointRadius:0, tension:0.3, fill:true,
      backgroundColor:'rgba(114,255,0,0.15)' }];
    cc.update('none');
  }
  const cd = charts['cd-defects'];
  if (cd) {
    cd.data.labels = labels;
    cd.data.datasets = [{ label:'Дефектов', data:defCounts,
      backgroundColor:'#72ff0099', borderColor:'#0f0f0f', borderWidth:1, borderRadius:0 }];
    cd.update('none');
  }
}

// ════════════════════════ TAB 2 — СРАВНЕНИЕ ══════════════════════════
async function runCompare() {
  const btn = document.getElementById('c-btn');
  const status = document.getElementById('c-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Запуск...';
  document.getElementById('c-modes-body').innerHTML = '';
  document.getElementById('c-quality-card').classList.add('hidden');
  document.getElementById('c-charts-card').classList.add('hidden');
  document.getElementById('c-charts2-card').classList.add('hidden');
  document.getElementById('c-charts3-card').classList.add('hidden');
  document.getElementById('c-charts4-card').classList.add('hidden');
  document.getElementById('c-charts5-card').classList.add('hidden');
  document.getElementById('c-charts6-card').classList.add('hidden');
  document.getElementById('c-charts7-card').classList.add('hidden');
  document.getElementById('c-charts8-card').classList.add('hidden');
  document.getElementById('c-progress-text').textContent = 'Запуск...';

  const modeStats = [];
  const perImageData = {};   // mode → float[]
  let quality = null;
  let modesDone = 0;
  const n_modes_total = 8;

  try {
    await streamNDJSON(`${API}/analyze_all`, {
      subfolder:  document.getElementById('c-folder').value,
      S_min:      parseFloat(document.getElementById('c-smin').value),
      S_max:      parseFloat(document.getElementById('c-smax').value),
      K:          parseInt(document.getElementById('c-k').value),
      threshold:  parseFloat(document.getElementById('c-thr').value),
      workers:    parseInt(document.getElementById('c-workers').value),
      max_images: parseInt(document.getElementById('c-maximg').value),
    }, chunk => {
      if (chunk.type === 'start') {
        document.getElementById('c-progress-text').textContent =
          `Обрабатываем ${chunk.n_images} изображений в ${chunk.n_modes} режимах...`;
      } else if (chunk.type === 'detection_quality') {
        quality = chunk.data;
      } else if (chunk.type === 'mode') {
        modesDone++;
        document.getElementById('c-progress-text').textContent =
          `Режим ${modesDone} из ${n_modes_total}: ${chunk.data.label}`;
        modeStats.push(chunk.data);
        if (chunk.data.per_image_times) perImageData[chunk.data.mode] = chunk.data.per_image_times;
        addModeRow(chunk.data);
      } else if (chunk.type === 'done') {
        document.getElementById('c-progress-text').textContent =
          `Готово — ${chunk.n_images} изображений, ${chunk.modes.length} режимов`;
        showCompareQuality(quality);
        showCompareCharts(modeStats);
        showCompareChartsExtra(modeStats, chunk.n_images);
        showPerImageCharts(modeStats, perImageData);
      }
    });
    status.textContent = 'Готово';
    await loadHistory();
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

function addModeRow(m) {
  const sp = m.speedup_vs_sequential != null ? m.speedup_vs_sequential.toFixed(2) + '×' : '—';
  const tr = document.createElement('tr');
  tr.innerHTML = `
    <td><strong>${m.label}</strong></td>
    <td>${m.n_images}</td>
    <td>${m.total_defects_filtered}</td>
    <td>${m.avg_time_ms.toFixed(1)}</td>
    <td>${m.wall_time_ms.toFixed(0)}</td>
    <td style="color:${m.speedup_vs_sequential>1?'#228800':'#333'};font-weight:700">${sp}</td>
  `;
  document.getElementById('c-modes-body').appendChild(tr);
}

function showCompareQuality(q) {
  if (!q) return;
  document.getElementById('c-quality-chips').innerHTML = `
    <div class="chip"><div class="v">${(q.accuracy*100).toFixed(1)}%</div><div class="l">Accuracy</div></div>
    <div class="chip chip-acc"><div class="v">${(q.f1*100).toFixed(1)}%</div><div class="l">F1</div></div>
    <div class="chip"><div class="v">${(q.precision*100).toFixed(1)}%</div><div class="l">Precision</div></div>
    <div class="chip"><div class="v">${(q.recall*100).toFixed(1)}%</div><div class="l">Recall</div></div>
    <div class="chip"><div class="v">${(q.mean_pixel_iou*100).toFixed(1)}%</div><div class="l">Mean IoU</div></div>
  `;
  document.getElementById('c-quality-body').innerHTML = `<tr>
    <td>${q.tp}</td><td>${q.tn}</td><td>${q.fp}</td><td>${q.fn}</td>
    <td>${q.gt_defective}</td><td>${q.gt_normal}</td>
  </tr>`;
  document.getElementById('c-quality-card').classList.remove('hidden');
}

function showCompareCharts(stats) {
  const labels = stats.map(s => MODE_SHORT[MODE_KEYS.indexOf(s.mode)] || s.mode);
  const su = stats.map(s => +(s.speedup_vs_sequential || 1).toFixed(3));
  const tm = stats.map(s => s.wall_time_ms);
  const colors = stats.map(s => { const i=MODE_KEYS.indexOf(s.mode); return (i>=0?MODE_COLORS_8[i]:'#888'); });

  const cs = charts['cc-speedup'];
  if (cs) {
    cs.data.labels = labels;
    cs.data.datasets = [{ label:'Speedup', data:su,
      backgroundColor: su.map((v,i)=>colors[i]+(v>1?'bb':'44')),
      borderColor: colors, borderWidth:1.5, borderRadius:0 }];
    cs.update();
  }

  const ct = charts['cc-time'];
  if (ct) {
    ct.data.labels = labels;
    ct.data.datasets = [{ label:'Wall time, мс', data:tm,
      backgroundColor: colors.map(c=>c+'99'), borderColor:colors, borderWidth:1.5, borderRadius:0 }];
    ct.update();
  }

  document.getElementById('c-charts-card').classList.remove('hidden');
}

function showCompareChartsExtra(stats, nImages) {
  // ── График масштабирования: все 8 линий наложенные ────────────────
  // X: 1..nImages, Y: wall_time_ms * (x/n) — линейная экстраполяция
  const N = nImages || 1;
  const xLabels = Array.from({length: Math.min(N, 50)}, (_,i) => Math.ceil((i+1) * N/Math.min(N,50)));

  const cs = charts['cc-scaling'];
  if (cs) {
    cs.data.labels = xLabels;
    cs.data.datasets = stats.map(s => {
      const i = MODE_KEYS.indexOf(s.mode);
      const wallPerImg = s.wall_time_ms / (s.n_images || 1);
      return {
        label: MODE_SHORT[i] || s.mode,
        data:  xLabels.map(x => +(wallPerImg * x).toFixed(1)),
        borderColor: MODE_COLORS_8[i] || '#888',
        borderWidth: s.mode === 'sequential' ? 2 : 1.5,
        borderDash:  s.mode === 'sequential' ? [6,3] : [],
        pointRadius: 0,
        tension: 0.1,
        fill: false,
      };
    });
    cs.update();
    document.getElementById('c-charts2-card').classList.remove('hidden');
  }

  // ── Avg time per image ──────────────────────────────────────────────
  const labels  = stats.map(s => MODE_SHORT[MODE_KEYS.indexOf(s.mode)] || s.mode);
  const colors  = stats.map(s => MODE_COLORS_8[MODE_KEYS.indexOf(s.mode)] || '#888');
  const avgTms  = stats.map(s => +s.avg_time_ms.toFixed(2));
  const defects = stats.map(s => s.total_defects_filtered);

  // Эффективность = sequential_wall / (mode_wall * n_workers) — только для parallel
  const seqStat = stats.find(s => s.mode === 'sequential');
  const seqWall = seqStat ? seqStat.wall_time_ms : null;
  const workersCnt = parseInt(document.getElementById('c-workers').value) || 4;
  const efficiencies = stats.map(s => {
    if (!seqWall || s.mode === 'sequential') return 100;
    const sp = s.speedup_vs_sequential || 1;
    return +(sp / workersCnt * 100).toFixed(1);
  });

  const ca = charts['cc-avgtime'];
  if (ca) {
    ca.data.labels = labels;
    ca.data.datasets = [{ label:'Ср. время/фото, мс', data:avgTms,
      backgroundColor:colors.map(c=>c+'99'), borderColor:colors, borderWidth:1.5, borderRadius:0 }];
    ca.update();
  }

  const cd = charts['cc-defects'];
  if (cd) {
    cd.data.labels = labels;
    cd.data.datasets = [{ label:'Дефектов', data:defects,
      backgroundColor:colors.map(c=>c+'99'), borderColor:colors, borderWidth:1.5, borderRadius:0 }];
    cd.update();
  }

  const ce = charts['cc-efficiency'];
  if (ce) {
    ce.data.labels = labels;
    ce.data.datasets = [{
      label:`Эффективность (speedup / ${workersCnt} workers × 100%)`, data:efficiencies,
      backgroundColor: efficiencies.map(v => v>=75?'#72ff0099':v>=50?'#ffcc0099':'#ff444499'),
      borderColor:'#0f0f0f', borderWidth:1.5, borderRadius:0
    }];
    ce.update();
  }

  document.getElementById('c-charts3-card').classList.remove('hidden');
}

function showPerImageCharts(stats, perImageData) {
  const modesWithData = stats.filter(s => perImageData[s.mode] && perImageData[s.mode].length > 0);
  if (!modesWithData.length) return;

  const maxLen = Math.max(...modesWithData.map(s => perImageData[s.mode].length));
  const xLabels = Array.from({length: maxLen}, (_, i) => i + 1);

  // helper: build cumulative array
  function cumsum(arr) {
    let s = 0;
    return arr.map(v => +(s += v).toFixed(2));
  }

  // sequential per-image data for speedup computation
  const seqTimes = perImageData['sequential'] || [];

  // ── Ряд 4: per-image timing overlay ───────────────────────────────
  const comb = charts['cc-per-img'];
  if (comb) {
    comb.data.labels = xLabels;
    comb.data.datasets = modesWithData.map(s => {
      const idx = MODE_KEYS.indexOf(s.mode);
      const color = MODE_COLORS_8[idx] || '#888';
      return {
        label: MODE_SHORT[idx] || s.mode,
        data: perImageData[s.mode],
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: s.mode === 'sequential' ? 2 : 1.5,
        borderDash: s.mode === 'sequential' ? [6, 3] : [],
        pointRadius: maxLen > 30 ? 0 : 3,
        tension: 0.2, fill: false,
      };
    });
    comb.update();
    document.getElementById('c-charts4-card').classList.remove('hidden');
  }

  // ── Ряд 5: 8 мини timing bar charts ──────────────────────────────
  modesWithData.forEach((s, pos) => {
    if (pos >= 8) return;
    const idx = MODE_KEYS.indexOf(s.mode);
    const color = MODE_COLORS_8[idx] || '#888';
    const times = perImageData[s.mode];
    const imgLabels = times.map((_, i) => i + 1);
    const avg = times.reduce((a, b) => a + b, 0) / times.length;
    const mc = charts['cc-m' + pos];
    if (mc) {
      mc.data.labels = imgLabels;
      mc.data.datasets = [
        {
          label: 'Время, мс', data: times,
          backgroundColor: times.map(v => v > avg ? color + 'dd' : color + '66'),
          borderColor: color, borderWidth: 1, borderRadius: 0,
        },
        {
          label: 'Среднее', data: imgLabels.map(() => +avg.toFixed(2)),
          type: 'line', borderColor: '#0f0f0f', borderDash: [4, 3],
          borderWidth: 1, pointRadius: 0, fill: false,
        },
      ];
      mc.update();
      const lbl = document.getElementById('cc-m' + pos + '-lbl');
      if (lbl) lbl.textContent = (MODE_SHORT[idx] || s.mode) + ' · avg ' + avg.toFixed(1) + ' мс';
    }
  });
  document.getElementById('c-charts5-card').classList.remove('hidden');

  // ── Ряд 6a: накопленное время overlay ─────────────────────────────
  const cc6 = charts['cc-cumul-all'];
  if (cc6) {
    cc6.data.labels = xLabels;
    cc6.data.datasets = modesWithData.map(s => {
      const idx = MODE_KEYS.indexOf(s.mode);
      const color = MODE_COLORS_8[idx] || '#888';
      return {
        label: MODE_SHORT[idx] || s.mode,
        data: cumsum(perImageData[s.mode].slice(0, maxLen)),
        borderColor: color,
        borderWidth: s.mode === 'sequential' ? 2 : 1.5,
        borderDash: s.mode === 'sequential' ? [6, 3] : [],
        pointRadius: 0, tension: 0.2, fill: false,
      };
    });
    cc6.update();
  }

  // ── Ряд 6b: speedup per image overlay ─────────────────────────────
  const cc6sp = charts['cc-speedup-per-img'];
  if (cc6sp && seqTimes.length) {
    const parallelModes = modesWithData.filter(s => s.mode !== 'sequential');
    cc6sp.data.labels = xLabels;
    cc6sp.data.datasets = [
      // baseline 1×
      {
        label: '1× (Sequential)', data: xLabels.map(() => 1),
        borderColor: MODE_COLORS_8[0],
        borderDash: [6, 3], borderWidth: 1.5, pointRadius: 0, fill: false,
      },
      ...parallelModes.map(s => {
        const idx = MODE_KEYS.indexOf(s.mode);
        const color = MODE_COLORS_8[idx] || '#888';
        const mTimes = perImageData[s.mode];
        const spData = mTimes.map((t, i) =>
          seqTimes[i] && t > 0 ? +(seqTimes[i] / t).toFixed(3) : null
        );
        return {
          label: MODE_SHORT[idx] || s.mode, data: spData,
          borderColor: color, borderWidth: 1.5,
          pointRadius: maxLen > 30 ? 0 : 2,
          tension: 0.2, fill: false, spanGaps: true,
        };
      }),
    ];
    cc6sp.update();
  }
  document.getElementById('c-charts6-card').classList.remove('hidden');

  // ── Ряд 7a: min/avg/max grouped bar ───────────────────────────────
  const ccMMA = charts['cc-minmaxavg'];
  if (ccMMA) {
    const mLabels = modesWithData.map(s => MODE_SHORT[MODE_KEYS.indexOf(s.mode)] || s.mode);
    const mins = modesWithData.map(s => +Math.min(...perImageData[s.mode]).toFixed(2));
    const avgs = modesWithData.map(s => {
      const t = perImageData[s.mode];
      return +(t.reduce((a,b)=>a+b,0)/t.length).toFixed(2);
    });
    const maxs = modesWithData.map(s => +Math.max(...perImageData[s.mode]).toFixed(2));
    ccMMA.data.labels = mLabels;
    ccMMA.data.datasets = [
      { label: 'Min',  data: mins, backgroundColor: '#72ff0099', borderColor: '#339900', borderWidth: 1.5, borderRadius: 0 },
      { label: 'Avg',  data: avgs, backgroundColor: '#2255cc99', borderColor: '#2255cc', borderWidth: 1.5, borderRadius: 0 },
      { label: 'Max',  data: maxs, backgroundColor: '#cc000099', borderColor: '#cc0000', borderWidth: 1.5, borderRadius: 0 },
    ];
    ccMMA.update();
  }

  // ── Ряд 7b: doughnut дефектов ─────────────────────────────────────
  const ccPie = charts['cc-defects-pie'];
  if (ccPie) {
    // Use only unique defect counts (sequential as reference, others as-is)
    const seqDef = stats.find(s => s.mode === 'sequential');
    const refDef = seqDef ? seqDef.total_defects_filtered : 1;
    const mLabels = modesWithData.map(s => MODE_SHORT[MODE_KEYS.indexOf(s.mode)] || s.mode);
    const defs = modesWithData.map(s => s.total_defects_filtered || 0);
    const colors = modesWithData.map(s => MODE_COLORS_8[MODE_KEYS.indexOf(s.mode)] || '#888');
    ccPie.data.labels = mLabels;
    ccPie.data.datasets = [{
      data: defs,
      backgroundColor: colors.map(c => c + 'cc'),
      borderColor: colors,
      borderWidth: 1.5,
    }];
    ccPie.update();
  }

  // ── Ряд 7c: normalized metrics bar ────────────────────────────────
  const ccNorm = charts['cc-norm-bar'];
  if (ccNorm) {
    const seqStat = stats.find(s => s.mode === 'sequential');
    const seqWall = seqStat ? seqStat.wall_time_ms : 1;
    const seqAvg  = seqStat ? seqStat.avg_time_ms  : 1;
    const mLabels = modesWithData.map(s => MODE_SHORT[MODE_KEYS.indexOf(s.mode)] || s.mode);
    const normWall = modesWithData.map(s => seqWall > 0 ? +(s.wall_time_ms / seqWall).toFixed(3) : 1);
    const normAvg  = modesWithData.map(s => seqAvg  > 0 ? +(s.avg_time_ms  / seqAvg ).toFixed(3) : 1);
    const colors = modesWithData.map(s => MODE_COLORS_8[MODE_KEYS.indexOf(s.mode)] || '#888');
    ccNorm.data.labels = mLabels;
    ccNorm.data.datasets = [
      {
        label: 'Wall time / Seq wall',
        data: normWall,
        backgroundColor: normWall.map(v => v <= 1 ? '#72ff0099' : '#cc000066'),
        borderColor: colors, borderWidth: 1.5, borderRadius: 0,
      },
      {
        label: 'Avg time / Seq avg',
        data: normAvg,
        backgroundColor: normAvg.map(v => v <= 1 ? '#2255cc66' : '#cc550066'),
        borderColor: colors, borderWidth: 1.5, borderRadius: 0,
      },
    ];
    ccNorm.update();
  }
  document.getElementById('c-charts7-card').classList.remove('hidden');

  // ── Ряд 8: 8 мини cumulative time line charts ──────────────────────
  modesWithData.forEach((s, pos) => {
    if (pos >= 8) return;
    const idx = MODE_KEYS.indexOf(s.mode);
    const color = MODE_COLORS_8[idx] || '#888';
    const times = perImageData[s.mode];
    const cData = cumsum(times);
    const imgLabels = times.map((_, i) => i + 1);
    const cc = charts['cc-c' + pos];
    if (cc) {
      cc.data.labels = imgLabels;
      cc.data.datasets = [{
        label: 'Накопл., мс', data: cData,
        borderColor: color,
        backgroundColor: color + '22',
        borderWidth: 1.5, pointRadius: 0, tension: 0.2, fill: true,
      }];
      cc.update();
      const lbl = document.getElementById('cc-c' + pos + '-lbl');
      if (lbl) lbl.textContent = (MODE_SHORT[idx] || s.mode) + ' · итого ' + cData[cData.length-1] + ' мс';
    }
  });
  document.getElementById('c-charts8-card').classList.remove('hidden');
}

// ════════════════════════ TAB 3 ══════════════════════════════
function parseList(str, isFloat) {
  return str.split(',').map(s => isFloat ? parseFloat(s.trim()) : parseFloat(s.trim())).filter(v => !isNaN(v));
}

async function runOptimize() {
  const btn = document.getElementById('oa-btn');
  const status = document.getElementById('oa-status');
  const mode = document.getElementById('oa-mode').value;
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Поиск...';
  document.getElementById('oa-results').classList.add('hidden');

  const body = {
    subfolder:  document.getElementById('oa-folder').value,
    max_images: parseInt(document.getElementById('oa-maximg').value),
    K:          parseInt(document.getElementById('oa-k').value),
    thresholds: parseList(document.getElementById('oa-thr').value, true),
    s_mins:     parseList(document.getElementById('oa-smins').value, false),
    s_maxs:     parseList(document.getElementById('oa-smaxs').value, false),
    workers: 4,
  };

  try {
    if (mode === 'all') {
      const rows = [];
      await streamNDJSON(`${API}/optimize_all`, body, chunk => {
        if (chunk.type === 'mode') {
          status.innerHTML = `<span class="spinner"></span>${chunk.data.label}...`;
          if (chunk.data.top10) rows.push(...chunk.data.top10.map(r => ({...r, _mode: chunk.data.label})));
        } else if (chunk.type === 'done') {
          showOptimizeTop(rows.slice(0, 10));
        }
      });
    } else {
      const d = await fetch(`${API}/optimize`, {
        method: 'POST', headers: {'Content-Type':'application/json'},
        body: JSON.stringify({...body, mode})
      }).then(r => r.json());
      showOptimizeTop(d.top10 || []);
    }
    status.textContent = 'Готово';
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

function showOptimizeTop(rows) {
  document.getElementById('oa-top-body').innerHTML = rows.map((r, i) => `<tr>
    <td>${i+1}</td>
    <td>${r.threshold ?? '—'}</td>
    <td>${r.S_min ?? '—'}</td>
    <td>${r.S_max ?? '—'}</td>
    <td style="font-weight:700;color:#228800">${r.f1 != null ? r.f1.toFixed(4) : '—'}</td>
    <td>${r.precision != null ? r.precision.toFixed(4) : '—'}</td>
    <td>${r.recall != null ? r.recall.toFixed(4) : '—'}</td>
  </tr>`).join('');
  document.getElementById('oa-results').classList.remove('hidden');
  addClassifyOptimHistory(rows);
}

async function runBinarizeOptimize() {
  const btn = document.getElementById('ob-btn');
  const status = document.getElementById('ob-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Поиск...';
  document.getElementById('ob-bar').style.width = '0%';
  document.getElementById('ob-progress-text').textContent = '';
  document.getElementById('ob-results').classList.add('hidden');

  const parseOpt = id => {
    const v = document.getElementById(id).value.trim();
    return v ? parseList(v, true) : null;
  };

  const body = {
    subfolder:  document.getElementById('ob-folder').value,
    max_images: parseInt(document.getElementById('ob-maximg').value),
    workers:    parseInt(document.getElementById('ob-workers').value),
    use_gpu:    document.getElementById('ob-gpu').checked,
    thresholds:            parseOpt('ob-thr'),
    min_areas:             parseOpt('ob-areas'),
    morphology_kernels:    parseOpt('ob-kernels'),
    morphology_iterations: parseOpt('ob-iters'),
    gaussian_blur_kernels: parseOpt('ob-blur'),
  };

  try {
    await streamNDJSON(`${API}/binarize_optimize`, body, chunk => {
      if (chunk.type === 'start') {
        document.getElementById('ob-progress-text').textContent = `0 / ${chunk.n_combos} комбинаций`;
      } else if (chunk.type === 'progress') {
        const pct = Math.round(chunk.done / chunk.total * 100);
        document.getElementById('ob-bar').style.width = pct + '%';
        document.getElementById('ob-progress-text').textContent = `${chunk.done} / ${chunk.total} (${pct}%)`;
      } else if (chunk.type === 'done') {
        document.getElementById('ob-bar').style.width = '100%';
        showBinarizeResults(chunk);
        loadBinarizeParams();
      }
    });
    status.textContent = 'Готово — параметры применены';
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

function showBinarizeResults(data) {
  const b = data.best || {};
  document.getElementById('ob-chips').innerHTML = `
    <div class="chip"><div class="v">${b.threshold ?? '—'}</div><div class="l">Threshold</div></div>
    <div class="chip"><div class="v">${b.min_area ?? '—'}</div><div class="l">Min area</div></div>
    <div class="chip"><div class="v">${b.morphology_kernel ?? '—'}</div><div class="l">Morph kernel</div></div>
    <div class="chip"><div class="v">${b.morphology_iterations ?? '—'}</div><div class="l">Iterations</div></div>
    <div class="chip chip-acc"><div class="v">✓</div><div class="l">Применено</div></div>
  `;
  document.getElementById('ob-top-body').innerHTML = (data.top10 || []).map((r,i) => `<tr>
    <td>${i+1}</td><td>${r.threshold??'—'}</td><td>${r.min_area??'—'}</td>
    <td>${r.morphology_kernel??'—'}</td><td>${r.morphology_iterations??'—'}</td>
    <td>${r.gaussian_blur_kernel??'—'}</td>
    <td>${r.loss != null ? r.loss.toFixed(6) : '—'}</td>
  </tr>`).join('');
  document.getElementById('ob-results').classList.remove('hidden');
}

async function saveBinarizeParams() {
  const btn = document.getElementById('bp-btn');
  const status = document.getElementById('bp-status');
  btn.disabled = true;
  try {
    await fetch(`${API}/binarize_params`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        threshold:             parseFloat(document.getElementById('bp-thr').value),
        min_area:              parseFloat(document.getElementById('bp-area').value),
        morphology_kernel:     parseInt(document.getElementById('bp-kernel').value),
        morphology_iterations: parseInt(document.getElementById('bp-iters').value),
        gaussian_blur_kernel:  parseInt(document.getElementById('bp-blur').value),
      })
    });
    status.textContent = 'Сохранено ✓';
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

// ════════════════════════ TAB 4 ══════════════════════════════
async function runCalibrate() {
  const btn = document.getElementById('cal-btn');
  const status = document.getElementById('cal-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Калибровка...';
  document.getElementById('cal-results').classList.add('hidden');

  try {
    const d = await fetch(`${API}/calibrate`, {
      method: 'POST', headers: {'Content-Type':'application/json'},
      body: JSON.stringify({
        source:     document.querySelector('input[name=cal-src]:checked').value,
        percentile: parseFloat(document.getElementById('cal-pct').value),
      })
    }).then(r => { if (!r.ok) throw new Error(`HTTP ${r.status}`); return r.json(); });

    document.getElementById('cal-chips').innerHTML = `
      <div class="chip chip-acc"><div class="v">${d.threshold}</div><div class="l">Порог аномальности</div></div>
      <div class="chip"><div class="v">${d.total_images}</div><div class="l">Изображений</div></div>
      <div class="chip"><div class="v">${d.percentile}%</div><div class="l">Процентиль</div></div>
    `;
    const statCard = (title, s) => !s || !s.n ? '' : `
      <div class="card">
        <div class="card-title">${title}</div>
        <div class="chips">
          <div class="chip"><div class="v">${s.n}</div><div class="l">Изображений</div></div>
          <div class="chip"><div class="v">${s.mean}</div><div class="l">Среднее</div></div>
          <div class="chip"><div class="v">${s.std}</div><div class="l">Std</div></div>
          <div class="chip"><div class="v">${s.p95}</div><div class="l">P95</div></div>
        </div>
      </div>`;
    document.getElementById('cal-stat-cards').innerHTML =
      statCard('test/good', d.test_good) + statCard('train/good', d.train_good);
    const warnEl = document.getElementById('cal-warning');
    if (d.warning) { warnEl.textContent = '⚠ ' + d.warning; warnEl.classList.remove('hidden'); }
    else { warnEl.classList.add('hidden'); }
    document.getElementById('cal-results').classList.remove('hidden');
    status.textContent = 'Готово';
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally { btn.disabled = false; }
}

// ════════════════════════ TAB 3 — КЛАССИФИКАЦИЯ ══════════════════════════

const CL_CLUSTER_COLORS = [
  '#e63946','#2a9d8f','#e9c46a','#6a0572','#264653',
  '#f4a261','#8338ec','#06d6a0','#ff6b6b','#457b9d',
];
const GT_TYPE_NAMES = ['color','cut','hole','metal_contamination','thread'];
const GT_TYPE_RU    = ['Цвет','Порез','Дырка','Металл','Нитка'];

// Состояние классификации
let clState = {
  allDefects: [],       // [{cluster, area, mean_score, gt_label, image_path}]
  clusterCounts: {},    // cluster_id → count
  clusterFeatures: {},  // cluster_id → {avg features}
  n_clusters: 0,
  mapping: {},          // cluster_id → gt_type_idx (строки от сервера)
  silhouettes: [],
  imageTimes: [],
};

// Auto-K toggle — настраивается из bindEvents()

async function runClassify() {
  const btn    = document.getElementById('cl-btn');
  const status = document.getElementById('cl-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Запуск...';

  // Сброс состояния
  clState = { allDefects:[], clusterCounts:{}, clusterFeatures:{}, n_clusters:0, mapping:{}, silhouettes:[], imageTimes:[] };
  document.getElementById('cl-rows').innerHTML = '';
  document.getElementById('cl-bar').style.width = '0%';
  document.getElementById('cl-progress-text').textContent = 'Инференс PatchCore...';
  ['cl-metrics-card','cl-charts1-card','cl-charts2-card','cl-cm-card'].forEach(id => {
    document.getElementById(id)?.classList.add('hidden');
  });

  const autoK = document.getElementById('cl-autok').checked;
  const mode  = document.getElementById('cl-mode').value;
  document.getElementById('cl-active-mode').textContent = mode;

  const body = {
    subfolder:       document.getElementById('cl-folder').value,
    method:          document.getElementById('cl-method').value,
    n_clusters:      autoK ? null : parseInt(document.getElementById('cl-k').value),
    k_min:           parseInt(document.getElementById('cl-kmin').value),
    k_max:           parseInt(document.getElementById('cl-kmax').value),
    threshold:       parseFloat(document.getElementById('cl-thr').value),
    S_min:           parseFloat(document.getElementById('cl-smin').value),
    S_max:           parseFloat(document.getElementById('cl-smax').value),
    max_images:      parseInt(document.getElementById('cl-maximg').value),
    workers:         parseInt(document.getElementById('cl-workers').value),
    mode:            mode,
    score_only:      document.getElementById('cl-score-only').checked,
    run_benchmark:   false,
  };

  let n_total = 0, n_done = 0;

  try {
    await streamNDJSON(`${API}/classify_batch`, body, chunk => {
      if (chunk.type === 'start') {
        n_total = chunk.total;
        document.getElementById('cl-progress-text').textContent =
          `Классификация: 0 / ${n_total} (режим: ${chunk.mode}, метод: ${chunk.method})`;
      } else if (chunk.type === 'infer_done') {
        document.getElementById('cl-progress-text').textContent =
          `Инференс завершён (${chunk.infer_ms} мс). Классифицируем...`;
      } else if (chunk.type === 'image') {
        n_done++;
        const r = chunk.result;
        const pct = n_total > 0 ? Math.round(n_done / n_total * 100) : 0;
        document.getElementById('cl-bar').style.width = pct + '%';
        document.getElementById('cl-progress-text').textContent = `${n_done} / ${n_total} изображений`;

        // Обновляем состояние
        clState.n_clusters = Math.max(clState.n_clusters, r.n_clusters || 0);
        const gtIdx = r.gt_label;
        r.cluster_labels?.forEach((cl, i) => {
          const feat = r.features?.[i] || {};
          clState.allDefects.push({
            cluster: cl, gt_label: gtIdx, image_path: r.image_path,
            area: feat.area || 0,
            mean_score: feat.mean_anomaly_score || 0,
            aspect_ratio: feat.aspect_ratio || 1,
            solidity: feat.solidity || 0,
          });
          clState.clusterCounts[cl] = (clState.clusterCounts[cl] || 0) + 1;
          if (!clState.clusterFeatures[cl]) clState.clusterFeatures[cl] = {};
          Object.entries(feat).forEach(([k, v]) => {
            if (!clState.clusterFeatures[cl][k]) clState.clusterFeatures[cl][k] = [];
            clState.clusterFeatures[cl][k].push(v);
          });
          if (r.mapping) Object.assign(clState.mapping, r.mapping);
        });
        if (r.silhouette != null) clState.silhouettes.push(r.silhouette);
        if (r.time_ms != null)    clState.imageTimes.push(r.time_ms);

        // Строка таблицы
        const gtName = gtIdx != null ? GT_TYPE_NAMES[gtIdx] : '—';
        const accStr = r.accuracy_single != null ? (r.accuracy_single * 100).toFixed(1) + '%' : '—';
        const silStr = r.silhouette != null ? r.silhouette.toFixed(3) : '—';
        document.getElementById('cl-rows').insertAdjacentHTML('beforeend', `<tr>
          <td>${n_done}</td>
          <td style="font-size:0.7rem;max-width:160px;overflow:hidden;text-overflow:ellipsis">${r.image_path || '—'}</td>
          <td><span class="dot" style="background:${gtIdx!=null?CL_CLUSTER_COLORS[gtIdx%CL_CLUSTER_COLORS.length]:'#ccc'}"></span>${gtName}</td>
          <td>${r.n_valid ?? r.n_defects ?? 0}</td>
          <td>${r.n_clusters || 0}</td>
          <td style="font-weight:700;color:${r.accuracy_single>0.5?'#228800':'#cc3300'}">${accStr}</td>
          <td>${silStr}</td>
          <td>${r.time_ms?.toFixed(1) ?? '—'}</td>
        </tr>`);

      } else if (chunk.type === 'done') {
        document.getElementById('cl-bar').style.width = '100%';
        document.getElementById('cl-progress-text').textContent =
          `Готово: ${chunk.n_images} изображений, метод: ${chunk.method}`;
        clState.n_clusters = chunk.n_clusters || clState.n_clusters || 5;
        if (chunk.overall_mapping) {
          Object.assign(clState.mapping, chunk.overall_mapping);
        }
        updateClassifyMetrics(chunk);
        updateClassifyCharts();
        status.textContent = 'Готово';
      } else if (chunk.type === 'error') {
        throw new Error(chunk.detail);
      }
    });
  } catch (e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
    document.getElementById('cl-active-mode').textContent = document.getElementById('cl-mode').value;
  }
}

function updateClassifyMetrics(done) {
  const acc  = done.mean_accuracy;
  const sil  = done.mean_silhouette;
  const oa   = done.overall_accuracy;
  const k    = clState.n_clusters;
  const nd   = clState.allDefects.length;
  const inMs = done.infer_ms;
  const clMs = done.classify_ms;

  document.getElementById('cl-chips').innerHTML = [
    chip(oa != null ? (oa*100).toFixed(1)+'%' : '—', 'Overall Accuracy', !!oa && oa >= 0.4),
    chip(acc != null ? (acc*100).toFixed(1)+'%' : '—', 'Mean Accuracy'),
    chip(sil != null ? sil.toFixed(3) : '—', 'Silhouette Score'),
    chip(k, 'Кластеров (K)'),
    chip(nd, 'Дефектов'),
    chip(inMs?.toFixed(0)+'мс', 'Инференс'),
    chip(clMs?.toFixed(0)+'мс', 'Классификация'),
    chip(done.classify_per_image_ms?.toFixed(1)+'мс', 'Кл./фото'),
  ].join('');

  // Mapping info
  const mappingInfo = Object.entries(clState.mapping)
    .filter(([k]) => !isNaN(parseInt(k)))
    .map(([cl, gtIdx]) => `C${cl} → ${GT_TYPE_NAMES[parseInt(gtIdx)] || '?'}`)
    .join(', ');
  document.getElementById('cl-mapping-info').textContent =
    mappingInfo ? 'Hungarian mapping: ' + mappingInfo : '';

  document.getElementById('cl-metrics-card').classList.remove('hidden');
}

function chip(val, lbl, highlight) {
  const cls = highlight ? 'chip chip-acc' : 'chip';
  return `<div class="${cls}"><div class="v">${val}</div><div class="l">${lbl}</div></div>`;
}

function updateClassifyCharts() {
  // ── Scatter: area vs mean_score ─────────────────────────────
  const scatterData = {};
  clState.allDefects.forEach(d => {
    const cl = d.cluster;
    if (!scatterData[cl]) scatterData[cl] = [];
    scatterData[cl].push({ x: d.area, y: d.mean_score });
  });
  const scatterDS = Object.entries(scatterData).map(([cl, pts]) => ({
    label: `C${cl}`,
    data: pts.slice(0, 200),
    backgroundColor: CL_CLUSTER_COLORS[parseInt(cl) % CL_CLUSTER_COLORS.length] + 'aa',
    pointRadius: 4,
  }));

  const scatterOpts = {
    responsive: true, maintainAspectRatio: false,
    plugins: { legend: { labels: { color:'#333', font: FONT, boxWidth: 10 } } },
    scales: {
      x: { ticks:{color:TICK,font:FONT}, grid:{color:GRID}, title:{display:true,text:'Площадь (px²)',color:TICK,font:FONT} },
      y: { ticks:{color:TICK,font:FONT}, grid:{color:GRID}, title:{display:true,text:'Anomaly Score',color:TICK,font:FONT} },
    }
  };
  if (charts['cl-scatter']) charts['cl-scatter'].destroy();
  const scEl = document.getElementById('cl-scatter');
  if (scEl) charts['cl-scatter'] = new Chart(scEl, { type:'scatter', data:{datasets:scatterDS}, options:scatterOpts });

  // ── Bar: дефекты по кластерам ────────────────────────────────
  const clusters = Object.keys(clState.clusterCounts).sort((a,b)=>+a-+b);
  const bColors  = clusters.map(cl => CL_CLUSTER_COLORS[parseInt(cl) % CL_CLUSTER_COLORS.length]);
  if (charts['cl-bardist']) charts['cl-bardist'].destroy();
  const barEl = document.getElementById('cl-bardist');
  if (barEl) charts['cl-bardist'] = new Chart(barEl, {
    type: 'bar',
    data: {
      labels: clusters.map(cl => `C${cl}`),
      datasets: [{ label:'Дефектов', data: clusters.map(cl => clState.clusterCounts[cl]),
        backgroundColor: bColors, borderColor:'#0f0f0f', borderWidth:1 }]
    },
    options: { responsive:true, maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID}},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID},beginAtZero:true}
      }
    }
  });

  // ── Silhouette history ────────────────────────────────────────
  if (charts['cl-sil-hist']) charts['cl-sil-hist'].destroy();
  const silEl = document.getElementById('cl-sil-hist');
  if (silEl && clState.silhouettes.length > 0) {
    charts['cl-sil-hist'] = new Chart(silEl, {
      type:'line',
      data:{
        labels: clState.silhouettes.map((_,i) => i+1),
        datasets:[{label:'Silhouette', data:clState.silhouettes,
          borderColor:'#2a9d8f', pointRadius:2, tension:0.3,
          fill:true, backgroundColor:'rgba(42,157,143,0.1)'}]
      },
      options:{ responsive:true, maintainAspectRatio:false,
        plugins:{legend:{display:false}},
        scales:{
          x:{ticks:{color:TICK,font:FONT},grid:{color:GRID},title:{display:true,text:'Изображение',color:TICK,font:FONT}},
          y:{ticks:{color:TICK,font:FONT},grid:{color:GRID},min:-1,max:1}
        }
      }
    });
  }
  document.getElementById('cl-charts1-card').classList.remove('hidden');

  // ── Radar: средние признаки по кластерам ─────────────────────
  const radarFeatures = ['max_anomaly_score','mean_anomaly_score','area','aspect_ratio','solidity','extent','eccentricity'];
  const radarLabels   = ['Max Score','Mean Score','Area','Aspect Ratio','Solidity','Extent','Eccentricity'];

  const radarDS = clusters.map(cl => {
    const feats = clState.clusterFeatures[cl] || {};
    const vals  = radarFeatures.map(f => {
      const arr = feats[f];
      return arr && arr.length ? arr.reduce((s,v)=>s+v,0)/arr.length : 0;
    });
    // Нормализуем 0..1 (min-max по кластерам ниже)
    return { label:`C${cl}`, data: vals, borderColor: CL_CLUSTER_COLORS[parseInt(cl)%CL_CLUSTER_COLORS.length], fill:true,
      backgroundColor: CL_CLUSTER_COLORS[parseInt(cl)%CL_CLUSTER_COLORS.length] + '22', pointRadius:3 };
  });

  // Нормализация по каждому признаку отдельно
  const maxVals = radarFeatures.map((_, fi) => Math.max(...radarDS.map(ds => ds.data[fi]), 1e-6));
  radarDS.forEach(ds => { ds.data = ds.data.map((v,fi) => v / maxVals[fi]); });

  if (charts['cl-radar']) charts['cl-radar'].destroy();
  const radEl = document.getElementById('cl-radar');
  if (radEl) charts['cl-radar'] = new Chart(radEl, {
    type:'radar',
    data:{ labels:radarLabels, datasets:radarDS },
    options:{ responsive:true, maintainAspectRatio:false,
      plugins:{ legend:{ labels:{color:'#333',font:FONT,boxWidth:10} } },
      scales:{ r:{ ticks:{color:TICK,font:FONT,backdropColor:'transparent'},
        grid:{color:GRID}, pointLabels:{color:TICK,font:FONT}, min:0, max:1 } }
    }
  });

  // ── Легенда кластеров ─────────────────────────────────────────
  document.getElementById('cl-legend-body').innerHTML = clusters.map(cl => {
    const gtIdx = clState.mapping[cl] ?? clState.mapping[parseInt(cl)];
    const gtName = gtIdx != null ? GT_TYPE_NAMES[gtIdx] : '—';
    const feats  = clState.clusterFeatures[cl] || {};
    const avgScore = feats.mean_anomaly_score?.reduce((s,v)=>s+v,0) / (feats.mean_anomaly_score?.length||1) || 0;
    const avgArea  = feats.area?.reduce((s,v)=>s+v,0) / (feats.area?.length||1) || 0;
    return `<tr>
      <td style="padding:5px 8px">
        <span class="dot" style="background:${CL_CLUSTER_COLORS[parseInt(cl)%CL_CLUSTER_COLORS.length]}"></span>
        C${cl}
      </td>
      <td style="padding:5px 8px;font-weight:600">${gtName}</td>
      <td style="padding:5px 8px;text-align:right">${clState.clusterCounts[cl] || 0}</td>
      <td style="padding:5px 8px;text-align:right">${avgScore.toFixed(3)}</td>
      <td style="padding:5px 8px;text-align:right">${avgArea.toFixed(0)}</td>
    </tr>`;
  }).join('');

  document.getElementById('cl-charts2-card').classList.remove('hidden');

  // ── Confusion matrix table ─────────────────────────────────────
  buildConfusionMatrix();
}

function buildConfusionMatrix() {
  const K = clState.n_clusters;
  if (K < 1 || clState.allDefects.length === 0) return;

  // Группируем по gt_label × cluster
  const cm = {};
  GT_TYPE_NAMES.forEach((_, gtIdx) => { cm[gtIdx] = {}; });
  clState.allDefects.forEach(d => {
    if (d.gt_label == null) return;
    const gt = parseInt(d.gt_label);
    const cl = parseInt(d.cluster);
    if (!cm[gt]) cm[gt] = {};
    cm[gt][cl] = (cm[gt][cl] || 0) + 1;
  });

  const clusterCols = Array.from({length: K}, (_, i) => i);
  const header = `<table style="border-collapse:collapse;font-size:0.75rem;width:100%">
    <thead><tr>
      <th style="padding:5px 8px;background:var(--text);color:#fff;text-align:left">GT \\ Кластер</th>
      ${clusterCols.map(cl => `<th style="padding:5px 8px;background:${CL_CLUSTER_COLORS[cl%CL_CLUSTER_COLORS.length]}33;text-align:center">C${cl}</th>`).join('')}
    </tr></thead>`;
  const rows = GT_TYPE_NAMES.map((name, gtIdx) => {
    const rowSum = Math.max(1, clusterCols.reduce((s, cl) => s + (cm[gtIdx]?.[cl] || 0), 0));
    const cells = clusterCols.map(cl => {
      const v = cm[gtIdx]?.[cl] || 0;
      const pct = Math.round(v / rowSum * 100);
      const bg  = `rgba(114,255,0,${(pct/100)*0.7})`;
      return `<td style="padding:5px 8px;text-align:center;background:${bg};font-weight:${pct>40?'700':'400'}">${v}${pct>5?'<br><small style="font-size:0.6rem;color:#555">'+pct+'%</small>':''}</td>`;
    });
    return `<tr>
      <td style="padding:5px 8px;font-weight:700;white-space:nowrap">
        <span class="dot" style="background:${CL_CLUSTER_COLORS[gtIdx%CL_CLUSTER_COLORS.length]}"></span>${name}
      </td>${cells.join('')}
    </tr>`;
  }).join('');
  document.getElementById('cl-cm-table-wrap').innerHTML = header + '<tbody>' + rows + '</tbody></table>';

  // Accuracy per GT type bar chart
  const perTypeAcc = GT_TYPE_NAMES.map((_, gtIdx) => {
    const rowTotal = clusterCols.reduce((s, cl) => s + (cm[gtIdx]?.[cl] || 0), 0);
    if (rowTotal === 0) return 0;
    const bestCl = clusterCols.reduce((best, cl) => (cm[gtIdx]?.[cl]||0) > (cm[gtIdx]?.[best]||0) ? cl : best, 0);
    return Math.round((cm[gtIdx]?.[bestCl] || 0) / rowTotal * 100);
  });

  if (charts['cl-acc-chart']) charts['cl-acc-chart'].destroy();
  const accEl = document.getElementById('cl-acc-chart');
  if (accEl) charts['cl-acc-chart'] = new Chart(accEl, {
    type: 'bar',
    data: {
      labels: GT_TYPE_NAMES,
      datasets: [{ label:'Accuracy, %', data: perTypeAcc,
        backgroundColor: GT_TYPE_NAMES.map((_, i) => CL_CLUSTER_COLORS[i%CL_CLUSTER_COLORS.length]+'cc'),
        borderColor:'#0f0f0f', borderWidth:1 }]
    },
    options: { responsive:true, maintainAspectRatio:false, indexAxis:'y',
      plugins:{ legend:{display:false} },
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID},min:0,max:100},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID}}
      }
    }
  });

  document.getElementById('cl-cm-card').classList.remove('hidden');
}

// ── Benchmark: все 8 режимов ─────────────────────────────────────────────
async function runClassifyBenchmark() {
  const btn    = document.getElementById('cl-bench-btn');
  const status = document.getElementById('cl-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Benchmark...';

  const body = {
    subfolder:  document.getElementById('cl-folder').value,
    method:     document.getElementById('cl-method').value,
    n_clusters: document.getElementById('cl-autok').checked ? null : parseInt(document.getElementById('cl-k').value),
    k_min:      parseInt(document.getElementById('cl-kmin').value),
    k_max:      parseInt(document.getElementById('cl-kmax').value),
    threshold:  parseFloat(document.getElementById('cl-thr').value),
    S_min:      parseFloat(document.getElementById('cl-smin').value),
    S_max:      parseFloat(document.getElementById('cl-smax').value),
    max_images: parseInt(document.getElementById('cl-maximg').value),
    workers:    parseInt(document.getElementById('cl-workers').value),
    score_only: document.getElementById('cl-score-only').checked,
    mode:       'sequential',
  };

  try {
    const res = await fetch(`${API}/classify_benchmark`, {
      method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(body)
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const d = await res.json();
    updateBenchmarkCharts(d);
    status.textContent = `Benchmark готов — лучший режим: ${d.best_mode}`;
  } catch (e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function updateBenchmarkCharts(d) {
  const list  = d.benchmark_list || Object.values(d.benchmark || {});
  const labels = list.map(r => MODE_SHORT[MODE_KEYS.indexOf(r.mode)] || r.mode);
  const times  = list.map(r => r.time_ms);
  const speeds = list.map(r => r.speedup_vs_sequential || 1);
  const colors = list.map(r => {
    const i = MODE_KEYS.indexOf(r.mode);
    return i >= 0 ? MODE_COLORS_8[i] : '#888';
  });

  if (charts['cl-bench-chart']) charts['cl-bench-chart'].destroy();
  const bEl = document.getElementById('cl-bench-chart');
  if (bEl) charts['cl-bench-chart'] = new Chart(bEl, {
    type:'bar',
    data:{ labels, datasets:[{
      label:'Время, мс', data:times,
      backgroundColor:colors.map(c=>c+'cc'), borderColor:'#0f0f0f', borderWidth:1
    }]},
    options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID},beginAtZero:true},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID}}
      }
    }
  });

  if (charts['cl-speedup-chart']) charts['cl-speedup-chart'].destroy();
  const spEl = document.getElementById('cl-speedup-chart');
  if (spEl) charts['cl-speedup-chart'] = new Chart(spEl, {
    type:'bar',
    data:{ labels, datasets:[{
      label:'Speedup', data:speeds,
      backgroundColor:colors.map(c=>c+'99'), borderColor:'#0f0f0f', borderWidth:1
    }]},
    options:{responsive:true,maintainAspectRatio:false,indexAxis:'y',
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID},beginAtZero:true},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID}}
      }
    }
  });

  document.getElementById('cl-bench-body').innerHTML = list.map(r => `<tr>
    <td><span class="dot" style="background:${MODE_COLORS_8[MODE_KEYS.indexOf(r.mode)]||'#888'}"></span>${r.label}</td>
    <td>${r.time_ms?.toFixed(1)}</td>
    <td style="font-weight:700;color:${r.speedup_vs_sequential>1?'#228800':'#0f0f0f'}">${(r.speedup_vs_sequential||1).toFixed(2)}×</td>
    <td>${r.n_defects || 0}</td>
    <td>${r.per_image_ms?.toFixed(1) ?? '—'}</td>
  </tr>`).join('');

  document.getElementById('cl-bench-card').classList.remove('hidden');
}

// ════════ EMBEDDING CLASSIFICATION ══════════════════════════════════════

let _clArch = 'geo';  // 'geo' | 'emb'

function onMethodChange() {
  const m = document.getElementById('cl-method').value;
  const supervised = m === 'prototype' || m === 'linear';
  document.getElementById('cl-supervised-hint').classList.toggle('hidden', !supervised);
  // supervised методы требуют эмбеддинги — переключаем архитектуру
  if (supervised && _clArch !== 'emb') {
    switchClassifyArch('emb');
  }
}

function switchClassifyArch(arch) {
  _clArch = arch;
  document.getElementById('cl-arch-geo').classList.toggle('active', arch === 'geo');
  document.getElementById('cl-arch-emb').classList.toggle('active', arch === 'emb');
  const embParams = document.getElementById('cl-emb-params');
  const hint      = document.getElementById('cl-arch-hint-emb');
  if (arch === 'emb') {
    embParams.classList.remove('hidden');
    hint.classList.remove('hidden');
  } else {
    embParams.classList.add('hidden');
    hint.classList.add('hidden');
  }
}

async function runClassifyEmbeddings() {
  const btn    = document.getElementById('cl-btn');
  const status = document.getElementById('cl-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>GPU инференс + PCA + UMAP...';

  clState = { allDefects:[], clusterCounts:{}, clusterFeatures:{}, n_clusters:0, mapping:{}, silhouettes:[], imageTimes:[] };
  document.getElementById('cl-rows').innerHTML = '';
  document.getElementById('cl-bar').style.width = '0%';
  document.getElementById('cl-progress-text').textContent = 'Извлечение патч-эмбеддингов...';
  ['cl-metrics-card','cl-charts1-card','cl-charts2-card','cl-cm-card','cl-umap-card'].forEach(id => {
    document.getElementById(id)?.classList.add('hidden');
  });

  const autoK = document.getElementById('cl-autok').checked;
  const body = {
    subfolder:      document.getElementById('cl-folder').value,
    method:         document.getElementById('cl-method').value,
    n_clusters:     autoK ? null : parseInt(document.getElementById('cl-k').value),
    k_min:          parseInt(document.getElementById('cl-kmin').value),
    k_max:          parseInt(document.getElementById('cl-kmax').value),
    pca_components: parseInt(document.getElementById('cl-pca-n').value),
    umap_method:    document.getElementById('cl-umap-method').value,
    compute_umap:   document.getElementById('cl-compute-umap').checked,
    n_prototypes:   3,
    test_size:      0.5,
    threshold:      parseFloat(document.getElementById('cl-thr').value),
    S_min:          parseFloat(document.getElementById('cl-smin').value),
    S_max:          parseFloat(document.getElementById('cl-smax').value),
    max_images:     parseInt(document.getElementById('cl-maximg').value),
    workers:        parseInt(document.getElementById('cl-workers').value),
    mode:           document.getElementById('cl-mode').value,
    include_good:   false,
  };

  let n_done = 0;
  document.getElementById('cl-active-mode').textContent = body.mode + ' + Embeddings';

  try {
    await streamNDJSON(`${API}/classify_embeddings`, body, chunk => {
      if (chunk.type === 'start') {
        document.getElementById('cl-progress-text').textContent =
          `GPU: ${chunk.total} изображений | PCA(${chunk.pca_components}) | UMAP: ${chunk.umap_available}`;
      } else if (chunk.type === 'infer_done') {
        document.getElementById('cl-progress-text').textContent =
          `Инференс: ${chunk.infer_ms}мс. PCA + кластеризация + UMAP...`;
      } else if (chunk.type === 'image') {
        n_done++;
        const r = chunk.result;
        const pct = body.max_images > 0 ? Math.round(n_done / body.max_images * 100) : 50;
        document.getElementById('cl-bar').style.width = Math.min(95, pct) + '%';

        // Обновляем clState для scatter chart
        const gtIdx = r.gt_label;
        (r.cluster_labels || []).forEach(cl => {
          clState.allDefects.push({cluster: cl, gt_label: gtIdx, image_path: r.image_path, area: 100, mean_score: 0.5});
          clState.clusterCounts[cl] = (clState.clusterCounts[cl] || 0) + 1;
        });

        const gtName = gtIdx != null ? GT_TYPE_NAMES[gtIdx] : '—';
        const accStr = r.accuracy != null ? (r.accuracy*100).toFixed(1)+'%' : '—';
        document.getElementById('cl-rows').insertAdjacentHTML('beforeend', `<tr>
          <td>${n_done}</td>
          <td style="font-size:0.7rem;max-width:160px;overflow:hidden;text-overflow:ellipsis">${r.image_path||'—'}</td>
          <td><span class="dot" style="background:${gtIdx!=null?CL_CLUSTER_COLORS[gtIdx%CL_CLUSTER_COLORS.length]:'#ccc'}"></span>${gtName}</td>
          <td>${r.n_defects||0}</td>
          <td>${(r.cluster_labels||[]).length}</td>
          <td style="font-weight:700;color:${r.accuracy>0.5?'#228800':'#cc3300'}">${accStr}</td>
          <td>—</td><td>—</td>
        </tr>`);

      } else if (chunk.type === 'done') {
        document.getElementById('cl-bar').style.width = '100%';
        document.getElementById('cl-progress-text').textContent =
          `Готово: ${chunk.n_images} изображений | Accuracy: ${chunk.overall_accuracy!=null?(chunk.overall_accuracy*100).toFixed(1)+'%':'—'} | PCA: ${chunk.pca_components} компонент`;
        clState.n_clusters = chunk.n_clusters || 5;

        // Metrics chips
        const isSupervised = chunk.method === 'prototype' || chunk.method === 'linear';
        const mainAcc = chunk.test_accuracy ?? chunk.overall_accuracy;
        const accLabel = chunk.method === 'linear' ? 'Test Accuracy (50%)' :
                         chunk.method === 'prototype' ? 'Prototype Accuracy' :
                         'Overall Accuracy (Hungarian)';
        document.getElementById('cl-chips').innerHTML = [
          chip(mainAcc != null ? (mainAcc*100).toFixed(1)+'%' : '—', accLabel, mainAcc >= 0.4),
          chunk.train_accuracy != null ? chip((chunk.train_accuracy*100).toFixed(1)+'%', 'Train Accuracy', false) : '',
          chunk.silhouette?.toFixed(3) ? chip(chunk.silhouette.toFixed(3), 'Silhouette') : '',
          chip(chunk.n_clusters, 'Классов'),
          chip(chunk.n_defects, 'Дефектов'),
          chip(chunk.pca_components, 'PCA компонент'),
          chip(chunk.embedding_dim, 'Embedding dim'),
          chip(chunk.infer_ms+'мс', 'Инференс GPU'),
          chunk.classify_ms ? chip(chunk.classify_ms+'мс', 'PCA+Classify+UMAP') : '',
        ].join('');

        // Per-class accuracy table (для supervised)
        if (chunk.per_class_accuracy && Object.keys(chunk.per_class_accuracy).length > 0) {
          const pcRows = Object.entries(chunk.per_class_accuracy)
            .sort(([a],[b])=>+a-+b)
            .map(([cls, acc]) => {
              const name = GT_TYPE_NAMES[parseInt(cls)] || '?';
              const color = CL_CLUSTER_COLORS[parseInt(cls)%CL_CLUSTER_COLORS.length];
              const pct = (acc*100).toFixed(1);
              return `<tr><td><span class="dot" style="background:${color}"></span>${name}</td>
                <td style="font-weight:700;color:${acc>=0.5?'#228800':'#cc3300'}">${pct}%</td>
                <td style="width:120px"><div style="height:8px;background:${color};width:${pct}%;border-radius:2px"></div></td></tr>`;
            }).join('');
          document.getElementById('cl-mapping-info').innerHTML = `
            <div style="margin-top:8px"><strong>Per-class accuracy (${chunk.method}):</strong></div>
            <table style="font-size:0.75rem;border-collapse:collapse;margin-top:4px">${pcRows}</table>`;
        } else {
          const mp = chunk.mapping || {};
          const mpStr = Object.entries(mp).filter(([k])=>!isNaN(parseInt(k)))
            .map(([cl,gt]) => 'C'+cl+'→'+(GT_TYPE_NAMES[parseInt(gt)]||'?')).join(', ');
          document.getElementById('cl-mapping-info').textContent =
            (mpStr ? 'Hungarian mapping: ' + mpStr : '') +
            (chunk.pca_variance ? ' | PCA variance: '+chunk.pca_variance.slice(0,3).map(v=>(v*100).toFixed(1)+'%').join(', ') : '');
        }

        // Mapping
        const mp = chunk.mapping || {};
        const mpStr = Object.entries(mp).filter(([k])=>!isNaN(parseInt(k)))
          .map(([cl,gt]) => `C${cl}→${GT_TYPE_NAMES[parseInt(gt)]||'?'}`).join(', ');
        document.getElementById('cl-mapping-info').textContent =
          (mpStr ? 'Hungarian mapping: ' + mpStr : '') +
          (chunk.pca_variance ? ' | PCA variance: ' + chunk.pca_variance.slice(0,3).map(v=>(v*100).toFixed(1)+'%').join(', ') : '');
        document.getElementById('cl-metrics-card').classList.remove('hidden');

        // Bar chart (кластеры)
        const clusters = Object.keys(clState.clusterCounts).sort((a,b)=>+a-+b);
        if (charts['cl-bardist']) charts['cl-bardist'].destroy();
        const barEl = document.getElementById('cl-bardist');
        if (barEl) charts['cl-bardist'] = new Chart(barEl, {
          type:'bar',
          data:{labels:clusters.map(c=>'C'+c),datasets:[{label:'Дефектов',
            data:clusters.map(c=>clState.clusterCounts[c]),
            backgroundColor:clusters.map(c=>CL_CLUSTER_COLORS[parseInt(c)%CL_CLUSTER_COLORS.length]),
            borderColor:'#0f0f0f',borderWidth:1}]},
          options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},
            scales:{x:{ticks:{color:TICK,font:FONT}},y:{ticks:{color:TICK,font:FONT},beginAtZero:true}}}
        });
        document.getElementById('cl-charts1-card').classList.remove('hidden');

        // UMAP
        if (chunk.umap_points && chunk.umap_points.length > 0) {
          renderUMAPCharts(chunk.umap_points, chunk.n_clusters, chunk.umap_method);
        }

        status.textContent = 'Готово (Embedding архитектура)';
      } else if (chunk.type === 'error') {
        throw new Error(chunk.detail);
      }
    });
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderUMAPCharts(points, n_clusters, umapMethod) {
  const GT_COLORS = ['#e63946','#2a9d8f','#e9c46a','#6a0572','#264653'];
  const GT_MARKERS = ['circle','triangle','rect','rectRot','star'];

  // ── Chart 1: раскраска по кластеру ────────────────────────────────
  const byCluster = {};
  points.forEach(p => {
    const k = p.cluster;
    if (!byCluster[k]) byCluster[k] = [];
    byCluster[k].push({x: p.x, y: p.y,
      _gt: p.gt_label, _gt_name: p.gt_name, _path: p.image_path});
  });
  const dsCluster = Object.entries(byCluster).map(([k, pts]) => ({
    label: `C${k}`,
    data: pts,
    backgroundColor: CL_CLUSTER_COLORS[parseInt(k) % CL_CLUSTER_COLORS.length] + 'cc',
    pointRadius: 5,
  }));

  if (charts['cl-umap-chart']) charts['cl-umap-chart'].destroy();
  const u1 = document.getElementById('cl-umap-chart');
  if (u1) charts['cl-umap-chart'] = new Chart(u1, {
    type:'scatter',
    data:{datasets: dsCluster},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#333',font:FONT,boxWidth:10}},
        tooltip:{callbacks:{label:p => `C${p.dataset.label}: GT=${p.raw._gt_name||'?'} ${p.raw._path}`}}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID}},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID}}
      }
    }
  });

  // ── Chart 2: раскраска по GT ──────────────────────────────────────
  const byGT = {};
  points.forEach(p => {
    const k = p.gt_label != null ? p.gt_label : -1;
    if (!byGT[k]) byGT[k] = [];
    byGT[k].push({x: p.x, y: p.y, _cluster: p.cluster, _path: p.image_path});
  });
  const dsGT = Object.entries(byGT).map(([k, pts]) => {
    const ki = parseInt(k);
    return {
      label: ki >= 0 ? GT_TYPE_NAMES[ki] : '?',
      data: pts,
      backgroundColor: (ki >= 0 ? GT_COLORS[ki % GT_COLORS.length] : '#999') + 'cc',
      pointRadius: 5,
    };
  });

  if (charts['cl-umap-gt-chart']) charts['cl-umap-gt-chart'].destroy();
  const u2 = document.getElementById('cl-umap-gt-chart');
  if (u2) charts['cl-umap-gt-chart'] = new Chart(u2, {
    type:'scatter',
    data:{datasets: dsGT},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#333',font:FONT,boxWidth:10}},
        tooltip:{callbacks:{label:p => `GT=${p.dataset.label} C${p.raw._cluster} ${p.raw._path}`}}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID}},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID}}
      }
    }
  });

  document.getElementById('cl-umap-chips').innerHTML =
    chip(points.length, 'Дефектов на UMAP') +
    chip(n_clusters, 'Кластеров') +
    chip(umapMethod?.toUpperCase() || 'UMAP', 'Метод 2D');
  document.getElementById('cl-umap-info').textContent =
    'Если точки GT одного типа образуют компактный кластер — эмбеддинги разделяют типы дефектов.';
  document.getElementById('cl-umap-card').classList.remove('hidden');
}

// Перегружаем runClassify чтобы выбирать архитектуру
const _origRunClassify = runClassify;
window._runClassifyWrapped = async function() {
  if (_clArch === 'emb') {
    await runClassifyEmbeddings();
  } else {
    await _origRunClassify();
  }
};

// ════════ ПОЛНЫЙ БЕНЧМАРК (все методы × режимы) ══════════════════════════

const FB_METHOD_COLORS = {
  kmeans:       '#2255cc',
  gmm:          '#cc5500',
  hierarchical: '#228800',
  prototype:    '#e63946',
  linear:       '#6a0572',
};
const FB_METHOD_LABELS = {
  kmeans: 'KMeans', gmm: 'GMM', hierarchical: 'Hierarchical',
  prototype: 'Prototype ★', linear: 'Linear ★★',
};

// matrix: {method: {mode: time_ms}}
let _fbMatrix = {};
let _fbMethods = [];

async function runFullBenchmark() {
  const btn    = document.getElementById('fb-btn');
  const status = document.getElementById('fb-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Запуск...';
  document.getElementById('fb-bar').style.width = '0%';
  document.getElementById('fb-progress-text').textContent = '';
  document.getElementById('fb-table-wrap').classList.add('hidden');
  document.getElementById('fb-charts-wrap').classList.add('hidden');

  const methods = [];
  if (document.getElementById('fb-m-kmeans').checked)  methods.push('kmeans');
  if (document.getElementById('fb-m-gmm').checked)     methods.push('gmm');
  if (document.getElementById('fb-m-hier').checked)    methods.push('hierarchical');
  if (document.getElementById('fb-m-proto').checked)   methods.push('prototype');
  if (document.getElementById('fb-m-linear').checked)  methods.push('linear');
  if (!methods.length) { status.textContent = 'Выберите хотя бы один метод'; btn.disabled=false; return; }

  _fbMatrix = {}; _fbMethods = methods;
  methods.forEach(m => _fbMatrix[m] = {});

  const body = {
    subfolder:      document.getElementById('cl-folder').value,
    max_images:     parseInt(document.getElementById('fb-maximg').value),
    threshold:      parseFloat(document.getElementById('cl-thr').value),
    S_min:          parseFloat(document.getElementById('cl-smin').value),
    S_max:          parseFloat(document.getElementById('cl-smax').value),
    n_clusters:     document.getElementById('cl-autok').checked
                      ? 5 : parseInt(document.getElementById('cl-k').value),
    pca_components: parseInt(document.getElementById('cl-pca-n').value) || 20,
    n_prototypes:   3,
    workers:        parseInt(document.getElementById('cl-workers').value),
    methods,
  };

  let n_total = 0;

  try {
    await streamNDJSON(`${API}/benchmark_comprehensive`, body, chunk => {
      if (chunk.type === 'start') {
        n_total = chunk.n_combos;
        document.getElementById('fb-progress-text').textContent =
          `GPU инференс: ${chunk.n_images} фото | ${chunk.methods.length} методов × 8 режимов = ${n_total} комбинаций`;
      } else if (chunk.type === 'infer_done') {
        document.getElementById('fb-progress-text').textContent =
          `Инференс ${chunk.infer_ms}мс (${chunk.n_images} фото). Классифицирую...`;
      } else if (chunk.type === 'result') {
        const pct = Math.round(chunk.done / chunk.total * 100);
        document.getElementById('fb-bar').style.width = pct + '%';
        document.getElementById('fb-progress-text').textContent =
          `${chunk.done}/${chunk.total}: [${chunk.method}][${chunk.mode}] = ${chunk.time_ms}мс (×${chunk.speedup})`;
        _fbMatrix[chunk.method][chunk.mode] = chunk.time_ms;
      } else if (chunk.type === 'done') {
        document.getElementById('fb-bar').style.width = '100%';
        _fbMatrix = chunk.matrix || _fbMatrix;
        _fbMethods = chunk.methods || _fbMethods;
        renderFullBenchmark(chunk);
        status.textContent = `Готово: ${_fbMethods.length} методов × ${(chunk.modes||[]).length} режимов`;
      } else if (chunk.type === 'error') {
        throw new Error(chunk.detail);
      }
    });
  } catch(e) {
    status.textContent = 'Ошибка: ' + e.message;
  } finally {
    btn.disabled = false;
  }
}

function renderFullBenchmark(done) {
  const modes   = done.modes || MODE_KEYS;
  const methods = done.methods || _fbMethods;
  const labels  = done.mode_labels || {};

  // ── Таблица ───────────────────────────────────────────────────────────
  let thead = '<thead><tr><th style="background:var(--text);color:#fff;padding:5px 8px">Метод \\ Режим</th>';
  modes.forEach(m => {
    thead += `<th style="background:var(--text);color:#fff;padding:5px 8px;font-size:0.65rem;white-space:nowrap">
      ${MODE_SHORT[MODE_KEYS.indexOf(m)] || m}</th>`;
  });
  thead += '</tr></thead>';

  let tbody = '<tbody>';
  methods.forEach(method => {
    const row = _fbMatrix[method] || {};
    const seqTime = row['sequential'] || 1;
    tbody += `<tr><td style="padding:5px 8px;font-weight:700;white-space:nowrap;border-right:1.5px solid var(--border)">
      <span style="display:inline-block;width:10px;height:10px;background:${FB_METHOD_COLORS[method]||'#888'};border-radius:50%;margin-right:4px"></span>
      ${FB_METHOD_LABELS[method]||method}</td>`;
    modes.forEach(mode => {
      const t = row[mode];
      const speedup = mode === 'sequential' ? 1.0 : (seqTime / Math.max(t||seqTime, 0.01));
      const best = t && t === Math.min(...Object.values(row).filter(v=>v>0));
      const color = best ? 'background:rgba(114,255,0,0.25);font-weight:700' : '';
      tbody += `<td style="padding:5px 8px;text-align:right;font-size:0.75rem;${color}">
        ${t ? t.toFixed(0)+'мс' : '—'}<br>
        <span style="font-size:0.62rem;color:${speedup>1?'#228800':speedup<0.8?'#cc0000':'#777'}">
          ×${speedup.toFixed(2)}</span></td>`;
    });
    tbody += '</tr>';
  });
  tbody += '</tbody>';

  document.getElementById('fb-matrix-table').innerHTML = thead + tbody;
  document.getElementById('fb-table-wrap').classList.remove('hidden');

  // ── Chart 1: время по режимам (группировано по методам) ──────────────
  const timeDatasets = methods.map(m => ({
    label: FB_METHOD_LABELS[m] || m,
    data:  modes.map(mode => (_fbMatrix[m]?.[mode] || 0)),
    backgroundColor: (FB_METHOD_COLORS[m] || '#888') + 'cc',
    borderColor:     FB_METHOD_COLORS[m] || '#888',
    borderWidth: 1,
  }));

  if (charts['fb-time-chart']) charts['fb-time-chart'].destroy();
  const tc = document.getElementById('fb-time-chart');
  if (tc) charts['fb-time-chart'] = new Chart(tc, {
    type: 'bar',
    data: { labels: modes.map(m => MODE_SHORT[MODE_KEYS.indexOf(m)] || m), datasets: timeDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color:'#333', font:FONT, boxWidth:10 } } },
      scales: {
        x: { ticks:{color:TICK,font:FONT}, grid:{color:GRID} },
        y: { ticks:{color:TICK,font:FONT}, grid:{color:GRID}, beginAtZero:true,
             title:{display:true,text:'Время, мс',color:TICK,font:FONT} }
      }
    }
  });

  // ── Chart 2: speedup vs sequential ────────────────────────────────────
  const speedupDatasets = methods.map(m => {
    const seqT = _fbMatrix[m]?.['sequential'] || 1;
    return {
      label: FB_METHOD_LABELS[m] || m,
      data:  modes.map(mode => {
        const t = _fbMatrix[m]?.[mode];
        return t ? parseFloat((seqT / Math.max(t, 0.01)).toFixed(2)) : 1.0;
      }),
      backgroundColor: (FB_METHOD_COLORS[m] || '#888') + 'aa',
      borderColor:     FB_METHOD_COLORS[m] || '#888',
      borderWidth: 1,
    };
  });

  if (charts['fb-speedup-chart']) charts['fb-speedup-chart'].destroy();
  const sc2 = document.getElementById('fb-speedup-chart');
  if (sc2) charts['fb-speedup-chart'] = new Chart(sc2, {
    type: 'bar',
    data: { labels: modes.map(m => MODE_SHORT[MODE_KEYS.indexOf(m)] || m), datasets: speedupDatasets },
    options: {
      responsive: true, maintainAspectRatio: false,
      plugins: { legend: { labels: { color:'#333', font:FONT, boxWidth:10 } } },
      scales: {
        x: { ticks:{color:TICK,font:FONT}, grid:{color:GRID} },
        y: { ticks:{color:TICK,font:FONT}, grid:{color:GRID}, beginAtZero:true,
             title:{display:true,text:'Speedup (×)',color:TICK,font:FONT} }
      }
    }
  });

  document.getElementById('fb-charts-wrap').classList.remove('hidden');
}

document.getElementById('fb-btn')?.addEventListener('click', runFullBenchmark);

// ════════ OPTIMIZE TAB: улучшения ════════════════════════════════════════

function addClassifyOptimHistory(results) {
  if (!results || !results.length) return;
  // История F1 score по итерациям
  const sorted = [...results].sort((a,b) => (b.f1||0) - (a.f1||0));
  const top3   = sorted.slice(0, 3);

  // Топ-3 chips
  document.getElementById('oa-top3').innerHTML = top3.map((r, i) => `
    <div class="chip" style="min-width:120px;border-color:${i===0?'var(--acc)':'var(--border)'}">
      <div class="v" style="font-size:0.9rem">F1: ${r.f1?.toFixed(4)||'—'}</div>
      <div class="l">thr=${r.threshold} | S[${r.S_min},${r.S_max}]</div>
    </div>
  `).join('');

  // Scatter: iteration vs F1
  const vals = results.map((r, i) => ({ x: i+1, y: r.f1 || 0 }));
  if (charts['oa-history']) charts['oa-history'].destroy();
  const el = document.getElementById('oa-history-chart');
  if (el) charts['oa-history'] = new Chart(el, {
    type:'scatter',
    data:{ datasets:[{
      label:'F1 score', data:vals,
      backgroundColor:'rgba(114,255,0,0.5)', pointRadius:2,
      borderColor:'rgba(0,0,0,0.3)', borderWidth:0.5,
    }]},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{display:false}},
      scales:{
        x:{ticks:{color:TICK,font:FONT},grid:{color:GRID},title:{display:true,text:'Итерация',color:TICK,font:FONT}},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID},min:0,max:1,title:{display:true,text:'F1',color:TICK,font:FONT}}
      }
    }
  });
}

// ═══ Optimize Tab: секция D — автоподбор K и метода ══════════════════════

async function runOptimizeClassify() {
  const btn    = document.getElementById('od-btn');
  const status = document.getElementById('od-status');
  btn.disabled = true;
  status.innerHTML = '<span class="spinner"></span>Перебор...';
  document.getElementById('od-results').classList.add('hidden');

  const subfolder  = document.getElementById('od-folder').value;
  const threshold  = parseFloat(document.getElementById('od-thr').value);
  const method     = document.getElementById('od-method').value;
  const mode       = document.getElementById('od-mode').value;
  const workers    = parseInt(document.getElementById('od-workers').value);
  const maxImages  = parseInt(document.getElementById('od-maximg').value);
  const cmpAll     = document.getElementById('od-cmp-methods').checked;
  const kStr       = document.getElementById('od-ks').value;
  const ks         = kStr.split(',').map(v => parseInt(v.trim())).filter(v => !isNaN(v) && v > 0);

  const methods = cmpAll ? ['kmeans','gmm','hierarchical'] : [method];
  const results = [];

  // Кэш: уже вычисленные комбинации
  const cache = {};

  for (const m of methods) {
    for (const k of ks) {
      const cacheKey = `${subfolder}|${threshold}|${m}|${k}`;
      if (cache[cacheKey]) {
        results.push(cache[cacheKey]);
        continue;
      }
      try {
        const body = {
          subfolder, method: m, n_clusters: k,
          threshold, max_images: maxImages,
          workers, mode, score_only: false,
          run_benchmark: false,
        };
        let acc = null, sil = null, t = 0;
        await streamNDJSON(`${API}/classify_batch`, body, chunk => {
          if (chunk.type === 'done') {
            acc = chunk.mean_accuracy;
            sil = chunk.mean_silhouette;
            t   = chunk.classify_ms || 0;
          }
        });
        const r = { k, method: m, acc, sil, time_ms: t };
        cache[cacheKey] = r;
        results.push(r);
        status.textContent = `${results.length}/${methods.length*ks.length}: K=${k}, ${m}, acc=${acc?.toFixed(3)||'—'}`;
      } catch (e) {
        results.push({ k, method: m, acc: null, sil: null, time_ms: 0 });
      }
    }
  }

  // Выбираем лучший
  const best = results.reduce((b, r) => (r.acc||0) > (b.acc||0) ? r : b, results[0] || {});
  document.getElementById('od-chips').innerHTML = [
    chip(`K=${best.k}`, 'Лучший K', true),
    chip(best.method, 'Лучший метод', true),
    chip(best.acc != null ? (best.acc*100).toFixed(1)+'%' : '—', 'Accuracy'),
    chip(best.sil?.toFixed(3) || '—', 'Silhouette'),
  ].join('');

  // K chart (линии по методам)
  const kValues = [...new Set(results.map(r => r.k))].sort((a,b)=>a-b);
  const dsMap = {};
  results.forEach(r => {
    if (!dsMap[r.method]) dsMap[r.method] = [];
    dsMap[r.method].push({ x: r.k, y: r.acc || 0 });
  });
  const methodColors = { kmeans:'#e63946', gmm:'#2a9d8f', hierarchical:'#6a0572' };
  if (charts['od-k-chart']) charts['od-k-chart'].destroy();
  const kEl = document.getElementById('od-k-chart');
  if (kEl) charts['od-k-chart'] = new Chart(kEl, {
    type:'line',
    data:{ datasets: Object.entries(dsMap).map(([m, pts]) => ({
      label: m, data: pts.sort((a,b)=>a.x-b.x),
      borderColor: methodColors[m]||'#888', pointRadius:4, tension:0.2,
      fill:false,
    }))},
    options:{responsive:true,maintainAspectRatio:false,
      plugins:{legend:{labels:{color:'#333',font:FONT,boxWidth:10}}},
      scales:{
        x:{type:'linear',ticks:{color:TICK,font:FONT,stepSize:1},grid:{color:GRID},title:{display:true,text:'K',color:TICK,font:FONT}},
        y:{ticks:{color:TICK,font:FONT},grid:{color:GRID},min:0,title:{display:true,text:'Accuracy',color:TICK,font:FONT}}
      }
    }
  });

  // Таблица
  document.getElementById('od-k-body').innerHTML = [...results]
    .sort((a,b)=>(b.acc||0)-(a.acc||0))
    .map((r, i) => `<tr style="${i===0?'font-weight:700;background:rgba(114,255,0,0.1)':''}">
      <td>${r.k}</td>
      <td>${r.method}</td>
      <td>${r.acc != null ? (r.acc*100).toFixed(1)+'%' : '—'}</td>
      <td>${r.sil?.toFixed(3) ?? '—'}</td>
      <td>${r.time_ms?.toFixed(0) ?? '—'}</td>
    </tr>`).join('');

  document.getElementById('od-results').classList.remove('hidden');
  status.textContent = `Готово. Лучший: K=${best.k}, ${best.method}, acc=${best.acc!=null?(best.acc*100).toFixed(1)+'%':'—'}`;
  btn.disabled = false;
}

init();
