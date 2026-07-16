/**
 * ZeroDefect Dashboard — app.js
 * Handles live status polling, log loading, image upload inference,
 * and dynamic UI updates.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const POLL_INTERVAL_MS  = 3000;   // Status refresh interval
const LOG_POLL_MS       = 6000;   // Log refresh interval
const API_BASE          = '';     // Empty = same origin

// ── State ────────────────────────────────────────────────────────────────────
let logPage = 1;
const LOG_PER_PAGE = 15;
let lastDecision = null;
let uploadedFile = null;

// ── Helpers ──────────────────────────────────────────────────────────────────

function $(id) { return document.getElementById(id); }

function formatTs(ts) {
  if (!ts) return '—';
  try {
    const d = new Date(ts);
    return d.toLocaleString('en-IN', { hour12: false });
  } catch { return ts; }
}

function confBar(pct) {
  const filled = Math.round(pct * 10);
  return '█'.repeat(filled) + '░'.repeat(10 - filled);
}

function animateNumber(el, target) {
  const current = parseInt(el.textContent) || 0;
  if (current === target) return;
  const step = target > current ? 1 : -1;
  const diff = Math.abs(target - current);
  const delay = diff > 10 ? 20 : 40;
  let val = current;
  const timer = setInterval(() => {
    val += step;
    el.textContent = val;
    if (val === target) clearInterval(timer);
  }, delay);
}

// ── Status polling ────────────────────────────────────────────────────────────

async function pollStatus() {
  try {
    const res = await fetch(`${API_BASE}/api/status`);
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    updateSystemStatus(data);
  } catch (err) {
    $('status-dot').className = 'status-dot error';
    $('status-text').textContent = 'API offline';
  }
}

function updateSystemStatus(data) {
  // Connection indicator
  $('status-dot').className = 'status-dot ok';
  $('status-text').textContent = data.models_ready ? 'Models loaded' : 'No model — train first';

  // Uptime
  $('stat-uptime').textContent = data.uptime_human || '0s';

  // Stats
  const s = data.stats || {};
  animateNumber($('stat-total'),  s.total   || 0);
  animateNumber($('stat-ok'),     s.n_ok    || 0);
  animateNumber($('stat-defect'), s.n_defect || 0);

  // Defect breakdown
  const byType = s.by_defect_type || {};
  const total_defects = s.n_defect || 1;
  const listEl = $('breakdown-list');
  const keys = Object.keys(byType);
  if (keys.length === 0) {
    listEl.innerHTML = '<div class="breakdown-empty">No defects logged yet.</div>';
  } else {
    listEl.innerHTML = keys.map(type => {
      const count = byType[type];
      const pct = Math.round(count / total_defects * 100);
      return `
        <div class="breakdown-item">
          <span class="breakdown-label">${type}</span>
          <div class="breakdown-bar-bg">
            <div class="breakdown-bar-fill" style="width:${pct}%"></div>
          </div>
          <span class="breakdown-count">${count}</span>
        </div>`;
    }).join('');
  }

  // Model info
  $('mi-status').textContent      = data.models_ready ? '✓ Loaded' : '✗ Not trained';
  $('mi-status').style.color      = data.models_ready ? 'var(--ok-primary)' : 'var(--defect-primary)';
  $('mi-classifier').textContent  = data.classifier_ready ? '✓ Ready' : '✗ Run finetune_fewshot.py';
  $('mi-threshold').textContent   = data.threshold != null ? data.threshold.toFixed(4) : '—';
}

// ── Log polling ───────────────────────────────────────────────────────────────

async function loadLog(page = 1) {
  try {
    const res = await fetch(`${API_BASE}/api/log?page=${page}&per_page=${LOG_PER_PAGE}`);
    if (!res.ok) return;
    const data = await res.json();
    renderLog(data);
    logPage = page;
  } catch (err) {
    console.warn('Log load error:', err);
  }
}

function renderLog(data) {
  const tbody = $('log-tbody');
  const { rows, total, page, pages } = data;

  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No inspections logged yet. Run live_demo.py to start.</td></tr>';
    $('log-pagination').innerHTML = '';
    return;
  }

  const offset = (page - 1) * LOG_PER_PAGE;
  tbody.innerHTML = rows.map((r, i) => {
    const isDefect = (r.decision || '').toUpperCase() === 'DEFECT';
    const badge = isDefect
      ? `<span class="badge-defect">✗ DEFECT</span>`
      : `<span class="badge-ok">✓ OK</span>`;
    const confNum = parseFloat(r.confidence || 0);
    const confPct = isNaN(confNum) ? '—' : `${(confNum * 100).toFixed(1)}%`;
    const scoreNum = parseFloat(r.score || 0);
    const score = isNaN(scoreNum) ? '—' : scoreNum.toFixed(4);
    const id = r.id || (total - offset - i);

    return `<tr>
      <td class="mono" style="color:var(--text-secondary)">${id}</td>
      <td style="color:var(--text-secondary);font-size:0.78rem">${formatTs(r.timestamp)}</td>
      <td>${badge}</td>
      <td style="text-transform:capitalize">${r.defect_type || '—'}</td>
      <td class="mono">${confPct}</td>
      <td class="mono" style="color:var(--text-secondary)">${score}</td>
    </tr>`;
  }).join('');

  // Pagination
  const paginEl = $('log-pagination');
  if (pages <= 1) { paginEl.innerHTML = ''; return; }

  let btns = '';
  for (let p = 1; p <= Math.min(pages, 10); p++) {
    btns += `<button class="page-btn ${p === page ? 'active' : ''}" onclick="loadLog(${p})">${p}</button>`;
  }
  paginEl.innerHTML = btns;
}

// ── Upload / Inference ────────────────────────────────────────────────────────

const uploadZone  = $('upload-zone');
const uploadInput = $('upload-input');

uploadZone.addEventListener('click', () => uploadInput.click());

uploadInput.addEventListener('change', (e) => {
  const file = e.target.files[0];
  if (!file) return;
  handleFileSelected(file);
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('dragover');
});

uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) handleFileSelected(file);
});

function handleFileSelected(file) {
  uploadedFile = file;
  const reader = new FileReader();
  reader.onload = (e) => {
    $('preview-img').src = e.target.result;
    $('upload-preview').style.display = 'block';
    $('upload-result').innerHTML = '';
    $('upload-result').className = 'upload-result';
    $('btn-run-inference').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

$('btn-run-inference').addEventListener('click', async () => {
  if (!uploadedFile) return;

  const btn = $('btn-run-inference');
  btn.textContent = 'Running…';
  btn.disabled = true;

  const formData = new FormData();
  formData.append('file', uploadedFile);

  try {
    const res = await fetch(`${API_BASE}/api/infer`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    displayUploadResult(result);

    // Also update the main decision card to reflect latest inference
    updateDecisionCard(result);
  } catch (err) {
    $('upload-result').className = 'upload-result defect';
    $('upload-result').innerHTML = `<strong>Error:</strong> ${err.message}`;
  } finally {
    btn.textContent = 'Run Inference';
    btn.disabled = false;
  }
});

function displayUploadResult(result) {
  const el = $('upload-result');
  const isDefect = result.decision === 'DEFECT';
  el.className = `upload-result ${isDefect ? 'defect' : 'ok'}`;
  const typeStr = isDefect ? ` — <strong>${result.defect_type || 'unknown'}</strong>` : '';
  el.innerHTML = `
    <strong>${isDefect ? '✗ DEFECT' : '✓ OK'}${typeStr}</strong><br/>
    Confidence: ${(result.confidence * 100).toFixed(1)}% &nbsp;|&nbsp;
    Score: <span class="mono">${result.score.toFixed(4)}</span> &nbsp;/&nbsp;
    Threshold: <span class="mono">${result.threshold.toFixed(4)}</span>
  `;
}

function updateDecisionCard(result) {
  const card = document.querySelector('.decision-card');
  const isDefect = result.decision === 'DEFECT';

  // State classes
  card.classList.remove('state-ok', 'state-defect');
  card.classList.add(isDefect ? 'state-defect' : 'state-ok');

  // Icons
  $('icon-idle').style.display   = 'none';
  $('icon-ok').style.display     = isDefect ? 'none' : 'block';
  $('icon-defect').style.display = isDefect ? 'block' : 'none';

  // Labels
  $('decision-label').textContent = isDefect ? '✗ DEFECT' : '✓ OK';
  $('decision-type').textContent  = isDefect ? (result.defect_type || 'unknown').toUpperCase() : 'No anomaly detected';

  // Scores
  $('score-value').textContent    = result.score.toFixed(4);
  $('threshold-value').textContent = result.threshold.toFixed(4);

  // Confidence bar
  const confPct = Math.round(result.confidence * 100);
  $('conf-bar').style.width = `${confPct}%`;
  $('conf-pct').textContent = `${confPct}%`;
}

// ── Refresh log button ────────────────────────────────────────────────────────
$('btn-refresh-log').addEventListener('click', () => loadLog(logPage));

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  await pollStatus();
  await loadLog(1);

  setInterval(pollStatus, POLL_INTERVAL_MS);
  setInterval(() => loadLog(logPage), LOG_POLL_MS);
})();
