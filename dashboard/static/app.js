/**
 * ZeroDefect Dashboard — app.js
 * Handles live status polling, log loading, image upload inference,
 * and dynamic UI updates.
 *
 * v2.0 — Uses local YOLOv5n ONNX backend instead of Roboflow cloud API.
 */

// ── Config ──────────────────────────────────────────────────────────────────
const POLL_INTERVAL_MS = 3000;   // Status refresh interval
const LOG_POLL_MS = 6000;   // Log refresh interval
const API_BASE = '';     // Empty = same origin
const SCAN_INTERVAL_MS = 1000;   // 1 frame per second for live scan

// ── State ────────────────────────────────────────────────────────────────────
let logPage = 1;
const LOG_PER_PAGE = 15;
let lastDecision = null;
let uploadedFile = null;
let currentEngine = 'hybrid'; // default

// ── Analytics Charts ──────────────────────────────────────────────────────────
let timelineChart = null;
let breakdownChart = null;

function initCharts() {
  const ctxTimeline = document.getElementById('timeline-chart');
  if (ctxTimeline) {
    timelineChart = new Chart(ctxTimeline, {
      type: 'line',
      data: {
        labels: [],
        datasets: [{
          label: 'Confidence (%)',
          data: [],
          borderColor: '#00d9f5',
          backgroundColor: 'rgba(0, 217, 245, 0.1)',
          tension: 0.3,
          fill: true
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false }
        },
        scales: {
          y: { beginAtZero: true, max: 100, grid: { color: 'rgba(255,255,255,0.05)' } },
          x: { grid: { display: false } }
        }
      }
    });
  }

  const ctxBreakdown = document.getElementById('breakdown-chart');
  if (ctxBreakdown) {
    breakdownChart = new Chart(ctxBreakdown, {
      type: 'doughnut',
      data: {
        labels: [],
        datasets: [{
          data: [],
          backgroundColor: [
            '#ff4444', '#ff8800', '#ffcc00', '#00f5a0', '#00d9f5', '#9933cc'
          ],
          borderWidth: 0
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { position: 'right', labels: { color: '#a0aabf', boxWidth: 12 } }
        },
        cutout: '70%'
      }
    });
  }
}


function setEngine(engine) {
  currentEngine = engine;
  document.getElementById('engine-hybrid').classList.remove('active');
  document.getElementById('engine-yolo').classList.remove('active');
  document.getElementById('engine-moondream').classList.remove('active');
  document.getElementById(`engine-${engine}`).classList.add('active');
}

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
  $('status-text').textContent = data.models_ready ? 'YOLOv5n loaded' : 'No model — place best.onnx in models/';

  // Uptime
  $('stat-uptime').textContent = data.uptime_human || '0s';

  // Stats
  const s = data.stats || {};
  animateNumber($('stat-total'), s.total || 0);
  animateNumber($('stat-ok'), s.n_ok || 0);
  animateNumber($('stat-defect'), s.n_defect || 0);

  // Defect breakdown
  const byType = s.by_defect_type || {};
  const total_defects = s.n_defect || 1;
  const listEl = $('breakdown-list'); // Keep for fallback, but hidden
  const keys = Object.keys(byType);
  
  if (breakdownChart) {
    breakdownChart.data.labels = keys;
    breakdownChart.data.datasets[0].data = keys.map(k => byType[k]);
    breakdownChart.update();
  }

  // Model info
  const engine = data.engine || 'unknown';
  $('mi-status').textContent = data.models_ready ? `✓ ${engine.toUpperCase()}` : '✗ Not loaded';
  $('mi-status').style.color = data.models_ready ? 'var(--ok-primary)' : 'var(--defect-primary)';

  if ($('mi-model')) {
    $('mi-model').textContent = data.yolo_ready ? 'YOLOv5 Nano (best_balanced.onnx)' : (data.patchcore_ready ? 'PatchCore (ResNet18)' : '—');
  }
  if ($('mi-classes')) {
    $('mi-classes').textContent = data.yolo_ready ? '7 defect types' : (data.patchcore_ready ? 'Anomaly binary' : '—');
  }
  $('mi-threshold').textContent = data.threshold != null ? data.threshold.toFixed(4) : '—';
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
  const cardsGrid = $('cards-grid');
  const { rows, total, page, pages } = data;

  if (!rows || rows.length === 0) {
    tbody.innerHTML = '<tr><td colspan="6" class="empty-row">No inspections logged yet. Start a Live Scan or upload a file to begin.</td></tr>';
    if (cardsGrid) cardsGrid.innerHTML = '<div class="empty-row" style="grid-column: 1 / -1;">No inferences to display yet.</div>';
    $('log-pagination').innerHTML = '';
    return;
  }

  const offset = (page - 1) * LOG_PER_PAGE;
  
  // Render text log table
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
      <td class="mono" style="color:var(--accent);font-size:0.72rem;text-transform:uppercase">${r.source || 'yolo'}</td>
      <td class="mono">${confPct}</td>
      <td class="mono" style="color:var(--text-secondary)">${score}</td>
    </tr>`;
  }).join('');

  // Render Visual Inference Cards
  if (cardsGrid) {
    // Only show cards for the first 10 items to prevent huge DOM sizes
    const cardRows = rows.slice(0, 10);
    cardsGrid.innerHTML = cardRows.map((r) => {
      const isDefect = (r.decision || '').toUpperCase() === 'DEFECT';
      const cardClass = isDefect ? 'defect-card' : 'ok-card';
      const badgeText = isDefect ? 'DEFECT' : 'OK';
      const typeText = isDefect ? (r.defect_type || 'Unknown').toUpperCase() : 'NO DEFECTS';
      
      const confNum = parseFloat(r.confidence || 0);
      const confPct = isNaN(confNum) ? '—' : `${(confNum * 100).toFixed(1)}%`;
      const engineName = (r.source || 'yolo').toUpperCase();
      
      // If there's an image path, format it correctly for the frontend mount point
      // Backend returns e.g. "logs/thumbnails/1_defect_scratch.jpg"
      // Endpoint is mounted at "/logs/thumbnails/"
      let imgSrc = '';
      if (r.image_path) {
        // Just take the filename from the path
        const filename = r.image_path.split('/').pop();
        imgSrc = `/logs/thumbnails/${filename}`;
      } else {
        // Placeholder if no image saved
        imgSrc = 'data:image/svg+xml;utf8,<svg xmlns="http://www.w3.org/2000/svg" width="200" height="200" fill="%231a243b"><rect width="100%" height="100%"/><text x="50%" y="50%" fill="%235a6b8a" font-family="sans-serif" font-size="14" text-anchor="middle" dominant-baseline="middle">No Thumbnail</text></svg>';
      }

      return `
        <div class="inference-card ${cardClass}">
          <div class="card-thumb">
            <span class="card-badge">${badgeText}</span>
            <img src="${imgSrc}" alt="Inference thumbnail" loading="lazy" onerror="this.src='data:image/svg+xml;utf8,<svg xmlns=\\'http://www.w3.org/2000/svg\\' width=\\'200\\' height=\\'200\\' fill=\\'%231a243b\\'><rect width=\\'100%\\' height=\\'100%\\'/><text x=\\'50%\\' y=\\'50%\\' fill=\\'%235a6b8a\\' font-family=\\'sans-serif\\' font-size=\\'14\\' text-anchor=\\'middle\\' dominant-baseline=\\'middle\\'>Missing Image</text></svg>'"/>
          </div>
          <div class="card-details">
            <div class="card-type" title="${typeText}">${typeText}</div>
            <div style="font-size: 0.65rem; color: var(--accent); margin-bottom: 4px; font-family: 'JetBrains Mono', monospace; letter-spacing: 0.3px;">ENGINE: ${engineName}</div>
            <div class="card-metrics">
              <span>Conf: <span class="mono">${confPct}</span></span>
              <span>Time: <span class="mono">${formatTs(r.timestamp).split(' ')[1] || '—'}</span></span>
            </div>
          </div>
        </div>
      `;
    }).join('');
  }

  // Pagination
  const paginEl = $('log-pagination');
  if (pages <= 1) { paginEl.innerHTML = ''; } else {
    let btns = '';
    for (let p = 1; p <= Math.min(pages, 10); p++) {
      btns += `<button class="page-btn ${p === page ? 'active' : ''}" onclick="loadLog(${p})">${p}</button>`;
    }
    paginEl.innerHTML = btns;
  }

  // Update Timeline Chart (first page only for live view)
  if (page === 1 && timelineChart && rows.length > 0) {
    // Take up to 15 latest logs, reverse to chronological order for line chart
    const recentRows = [...rows].slice(0, 15).reverse();
    const labels = recentRows.map(r => formatTs(r.timestamp).split(' ')[1] || ''); // Just time
    const confData = recentRows.map(r => (parseFloat(r.confidence || 0) * 100).toFixed(1));
    
    timelineChart.data.labels = labels;
    timelineChart.data.datasets[0].data = confData;
    
    // Change color based on if there are recent defects
    const hasDefect = recentRows.some(r => (r.decision || '').toUpperCase() === 'DEFECT');
    timelineChart.data.datasets[0].borderColor = hasDefect ? '#ff4444' : '#00d9f5';
    timelineChart.data.datasets[0].backgroundColor = hasDefect ? 'rgba(255, 68, 68, 0.1)' : 'rgba(0, 217, 245, 0.1)';
    
    timelineChart.update();
  }
}

// ── Upload & Webcam Inference ──────────────────────────────────────────────────

const uploadZone = $('upload-zone');
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
    $('upload-result').style.display = 'none';
    $('upload-result').className = 'upload-result';
    $('btn-run-inference').style.display = 'block';
  };
  reader.readAsDataURL(file);
}

// Unified Inference Engine — sends to local YOLO backend
async function runInference(fileOrBlob, isWebcam = false) {
  const formData = new FormData();
  formData.append('file', fileOrBlob, isWebcam ? 'webcam_scan.jpg' : fileOrBlob.name);

  const videoWrap = document.querySelector('.video-wrap');
  if (isWebcam && videoWrap) {
    videoWrap.classList.add('scanning-active');
  }

  try {
    const res = await fetch(`${API_BASE}/api/infer?engine=${currentEngine}`, { method: 'POST', body: formData });
    if (!res.ok) {
      const err = await res.json();
      throw new Error(err.detail || `HTTP ${res.status}`);
    }
    const result = await res.json();
    displayUploadResult(result);
    updateDecisionCard(result);

    // Dynamic session updates
    await loadLog(1);
    await pollStatus();

    return result;
  } catch (err) {
    $('upload-result').className = 'upload-result defect';
    $('upload-result').style.display = 'block';
    $('upload-result').innerHTML = `<strong>Error:</strong> ${err.message}`;
    console.error("Inference Error:", err);
  } finally {
    if (isWebcam && videoWrap) {
      videoWrap.classList.remove('scanning-active');
    }
  }
}

$('btn-run-inference').addEventListener('click', async () => {
  if (!uploadedFile) return;

  const btn = $('btn-run-inference');
  btn.textContent = 'Running…';
  btn.disabled = true;

  try {
    await runInference(uploadedFile, false);
  } finally {
    btn.textContent = 'Run Inference';
    btn.disabled = false;
  }
});

function displayUploadResult(result) {
  const el = $('upload-result');
  const isDefect = result.decision === 'DEFECT';
  el.className = `upload-result ${isDefect ? 'defect' : 'ok'}`;
  el.style.display = 'block';

  const typeStr = isDefect ? ` — <strong>${result.defect_type || 'unknown'}</strong>` : '';
  const engineStr = result.engine ? ` &nbsp;|&nbsp; Engine: <span class="mono">${result.engine.toUpperCase()}</span>` : '';
  const reasonStr = result.reasoning ? `<br/><span style="color:var(--text-secondary);font-size:0.82rem">💬 ${result.reasoning}</span>` : '';

  // Show detection count for YOLO engine
  const detectCountStr = (result.num_detections != null && result.num_detections > 0)
    ? ` &nbsp;|&nbsp; Detections: <span class="mono">${result.num_detections}</span>`
    : '';

  el.innerHTML = `
    <strong>${isDefect ? '✗ DEFECT' : '✓ OK'}${typeStr}</strong><br/>
    Confidence: ${(result.confidence * 100).toFixed(1)}% &nbsp;|&nbsp;
    Score: <span class="mono">${result.score.toFixed(4)}</span> &nbsp;/&nbsp;
    Threshold: <span class="mono">${result.threshold.toFixed(4)}</span>${engineStr}${detectCountStr}${reasonStr}
  `;
}

function updateDecisionCard(result) {
  const card = document.querySelector('.decision-card');
  const isDefect = result.decision === 'DEFECT';

  // State classes
  card.classList.remove('state-ok', 'state-defect');
  card.classList.add(isDefect ? 'state-defect' : 'state-ok');

  // Icons
  $('icon-idle').style.display = 'none';
  $('icon-ok').style.display = isDefect ? 'none' : 'block';
  $('icon-defect').style.display = isDefect ? 'block' : 'none';

  // Labels
  $('decision-label').textContent = isDefect ? '✗ DEFECT' : '✓ OK';
  $('decision-type').textContent = isDefect ? (result.defect_type || 'unknown').toUpperCase() : 'No defects detected';

  // Scores
  $('score-value').textContent = result.score.toFixed(4);
  $('threshold-value').textContent = result.threshold.toFixed(4);

  // Confidence bar
  const confPct = Math.round(result.confidence * 100);
  $('conf-bar').style.width = `${confPct}%`;
  $('conf-pct').textContent = `${confPct}%`;
}

function resetDecisionCard() {
  const card = document.querySelector('.decision-card');
  if (card) {
    card.classList.remove('state-ok', 'state-defect');
    $('icon-ok').style.display = 'none';
    $('icon-defect').style.display = 'none';
    $('icon-idle').style.display = 'block';
    
    $('decision-label').textContent = 'Waiting for inspection…';
    $('decision-type').textContent = '';
    
    $('score-value').textContent = '—';
    $('threshold-value').textContent = '—';
    
    $('conf-bar').style.width = '0%';
    $('conf-pct').textContent = '—';
    lastDecision = null;
  }
}

// ── Webcam Controller ──────────────────────────────────────────────────────────

let webcamStream = null;
let scanLoopActive = false;
let autoScanTimeout = null;

async function startWebcam() {
  if (webcamStream) return;

  $('upload-result').style.display = 'none';
  $('upload-result').innerHTML = '';

  try {
    // First try with facingMode: user
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user" }
    });
  } catch (err) {
    console.warn("getUserMedia with facingMode:user failed, trying generic video constraint...", err);
    try {
      // Generic fallback (matches any camera or virtual camera device)
      webcamStream = await navigator.mediaDevices.getUserMedia({ video: true });
    } catch (fallbackErr) {
      console.error("Webcam access error:", fallbackErr);
      alert("Could not access camera. Please check permissions and device availability.");
      $('toggle-upload').click();
      return;
    }
  }

  const video = $('webcam-video');
  video.srcObject = webcamStream;
  video.play();

  const videoWrap = document.querySelector('.video-wrap');
  if (videoWrap) videoWrap.classList.add('scanning-active');

  document.querySelector('.scan-status-text').textContent = 'CAMERA READY';
  $('webcam-container').style.display = 'block';
  $('file-upload-container').style.display = 'none';
}

function stopWebcam() {
  scanLoopActive = false;
  if (autoScanTimeout) {
    clearTimeout(autoScanTimeout);
    autoScanTimeout = null;
  }
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  const video = $('webcam-video');
  video.srcObject = null;

  const videoWrap = document.querySelector('.video-wrap');
  if (videoWrap) {
    videoWrap.classList.remove('scanning-active', 'defect-frozen');
  }

  $('webcam-container').style.display = 'none';
  $('file-upload-container').style.display = 'block';
}

async function runAutoScanLoop() {
  if (!scanLoopActive || !webcamStream) return;

  const startTs = Date.now();

  // Capture and scan frame via local YOLO backend
  const result = await captureAndScanLocal();

  // Handle Defect Flagging and Freezing
  if (result && result.decision === 'DEFECT') {
    const video = $('webcam-video');
    const videoWrap = document.querySelector('.video-wrap');

    // Freeze frame visually for inspection
    video.pause();
    if (videoWrap) videoWrap.classList.add('defect-frozen');
    document.querySelector('.scan-status-text').textContent = 'DEFECT FLAGGED';

    // Temporarily halt scanning loop
    scanLoopActive = false;

    setTimeout(() => {
      // Auto-resume after 2 seconds
      if (webcamStream) {
        video.play().catch(err => console.warn('Failed to resume video:', err));
        if (videoWrap) videoWrap.classList.remove('defect-frozen');
        document.querySelector('.scan-status-text').textContent = 'CAMERA READY';
        scanLoopActive = true;
        runAutoScanLoop();
      }
    }, 2000);

    return;
  }

  // Schedule next scan at 1-second interval
  const elapsed = Date.now() - startTs;
  const delay = Math.max(0, SCAN_INTERVAL_MS - elapsed);

  if (scanLoopActive) {
    autoScanTimeout = setTimeout(runAutoScanLoop, delay);
  }
}

/**
 * Capture a frame from the webcam and send it to the local YOLO backend.
 * The video stream stays live — this function is fully non-blocking.
 * Returns a normalized result object on detection, or null if nothing found.
 */
async function captureAndScanLocal() {
  const video = $('webcam-video');
  if (!webcamStream || video.paused || video.ended) return null;
  if (video.videoWidth === 0) return null; // stream not ready yet

  // Draw current frame to a hidden canvas
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  try {
    // Convert canvas to blob and send to local YOLO backend
    const blob = await new Promise(resolve => canvas.toBlob(resolve, 'image/jpeg', 0.85));
    if (!blob) return null;

    const formData = new FormData();
    formData.append('file', blob, 'webcam_scan.jpg');

    const resp = await fetch(`${API_BASE}/api/infer?engine=${currentEngine}`, {
      method: 'POST',
      body: formData,
    });

    if (!resp.ok) {
      console.warn('[YOLO] HTTP error:', resp.status);
      return null;
    }

    const result = await resp.json();
    console.log('[YOLO] response:', result);

    // Show result banner in UI
    displayUploadResult(result);
    updateDecisionCard(result);

    // Refresh log table immediately
    await loadLog(1);

    return result;

  } catch (err) {
    console.error('[YOLO] Fetch error:', err);
    return null;
  }
}

// Event Listeners for webcam & toggle UI
$('toggle-upload').addEventListener('click', () => {
  $('toggle-upload').classList.add('active');
  $('toggle-webcam').classList.remove('active');
  stopWebcam();
});

$('toggle-webcam').addEventListener('click', () => {
  $('toggle-webcam').classList.add('active');
  $('toggle-upload').classList.remove('active');
  startWebcam();
});
$('btn-webcam-stop').addEventListener('click', () => {
  $('toggle-upload').click();
});

// ── Scan Frame button — manual single-shot YOLO scan ──────────────────────
$('btn-scan-frame').addEventListener('click', async () => {
  const btn = $('btn-scan-frame');
  if (!webcamStream) return;

  // Loading state
  btn.disabled = true;
  btn.textContent = '⏳ Scanning…';
  document.querySelector('.scan-status-text').textContent = 'SCANNING…';

  try {
    const result = await captureAndScanLocal();

    if (result && result.decision === 'DEFECT') {
      // Flash red briefly to highlight the detection
      const videoWrap = document.querySelector('.video-wrap');
      if (videoWrap) videoWrap.classList.add('defect-frozen');
      document.querySelector('.scan-status-text').textContent = 'DEFECT FLAGGED';
      setTimeout(() => {
        if (videoWrap) videoWrap.classList.remove('defect-frozen');
        document.querySelector('.scan-status-text').textContent = 'CAMERA READY';
      }, 2000);
    } else {
      document.querySelector('.scan-status-text').textContent = 'CAMERA READY';
    }
  } finally {
    btn.disabled = false;
    btn.textContent = '📸 Scan Frame';
  }
});

// ── Refresh log button ────────────────────────────────────────────────────────
$('btn-refresh-log').addEventListener('click', () => loadLog(logPage));

// ── Clear History function ───────────────────────────────────────────────────
async function clearHistory() {
  if (!confirm("Are you sure you want to clear all inspection history, database records, and thumbnails? This cannot be undone.")) {
    return;
  }
  
  const btn = $('btn-clear-history');
  if (btn) {
    btn.disabled = true;
    btn.textContent = "⏳ Clearing...";
  }

  try {
    const res = await fetch(`${API_BASE}/api/clear`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    if (res.ok) {
      // Reload stats and first page of logs immediately
      await pollStatus();
      await loadLog(1);
      
      // Reset decision card UI
      resetDecisionCard();
      
      // Reset timeline chart
      if (timelineChart) {
        timelineChart.data.labels = [];
        timelineChart.data.datasets[0].data = [];
        timelineChart.data.datasets[0].borderColor = '#00d9f5';
        timelineChart.data.datasets[0].backgroundColor = 'rgba(0, 217, 245, 0.1)';
        timelineChart.update();
      }
    } else {
      const errData = await res.json();
      alert(`Error clearing history: ${errData.detail || 'Unknown error'}`);
    }
  } catch (err) {
    console.error('Error clearing history:', err);
    alert('Failed to connect to backend to clear history.');
  } finally {
    if (btn) {
      btn.disabled = false;
      btn.textContent = "🗑 Clear History";
    }
  }
}

window.clearHistory = clearHistory;

// ── Init ──────────────────────────────────────────────────────────────────────
(async function init() {
  initCharts();
  await pollStatus();
  await loadLog(1);

  setInterval(pollStatus, POLL_INTERVAL_MS);
  setInterval(() => loadLog(logPage), LOG_POLL_MS);
})();
