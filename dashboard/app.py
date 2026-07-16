"""
dashboard/app.py — ZeroDefect Local Dashboard API
===================================================

FastAPI server providing:
  GET  /                  → Serve the dashboard HTML page
  GET  /api/status        → Current system status + recent stats
  GET  /api/log           → Paginated inspection log
  GET  /api/stats         → Aggregate stats
  POST /api/infer         → Upload image, get anomaly decision
  GET  /api/export        → Download inspections.csv

Run with:
  uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import sys
import os
import io
import time
import json
import pickle
import traceback
from pathlib import Path
from datetime import datetime
from typing import Optional

import numpy as np
from PIL import Image

# Add src/ to path for imports
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from fastapi import FastAPI, File, UploadFile, HTTPException, Query
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

# ── App setup ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="ZeroDefect Dashboard API",
    description="Local anomaly detection dashboard for automotive QC",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

DASHBOARD_DIR = Path(__file__).parent
STATIC_DIR = DASHBOARD_DIR / "static"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"

# ── Lazy model loading ─────────────────────────────────────────────────────────
_nn_model = None
_extractor = None
_classifier_data = None
_model_info = None
_start_time = time.time()


def _load_models():
    """Load models lazily on first inference request."""
    global _nn_model, _extractor, _classifier_data, _model_info

    if _nn_model is not None:
        return True

    try:
        from utils import get_extractor, load_model_info, resolve_device

        memory_bank_path = MODELS_DIR / "memory_bank.pkl"
        if not memory_bank_path.exists():
            return False

        with open(memory_bank_path, "rb") as f:
            _nn_model = pickle.load(f)

        device = resolve_device()
        _extractor = get_extractor(device)
        _model_info = load_model_info()

        classifier_path = MODELS_DIR / "classifier.pkl"
        if classifier_path.exists():
            with open(classifier_path, "rb") as f:
                _classifier_data = pickle.load(f)

        return True

    except Exception as e:
        print(f"[WARN] Could not load models: {e}")
        return False


def _score_pil_image(img: Image.Image) -> dict:
    """Run inference on a PIL image."""
    from utils import get_transform, DEFAULT_IMAGE_SIZE

    image_size = _model_info.get("image_size", DEFAULT_IMAGE_SIZE) if _model_info else DEFAULT_IMAGE_SIZE
    threshold = _model_info.get("threshold", 1.0) if _model_info else 1.0

    transform = get_transform(image_size)
    tensor = transform(img.convert("RGB")).unsqueeze(0)

    patches = _extractor.extract_flat_patches(tensor)      # [N, 384]
    dists, _ = _nn_model.kneighbors(patches)               # [N, 1]
    score = float(np.percentile(dists.flatten(), 99))

    if score > threshold:
        defect_type = "unknown"
        confidence = min(1.0, (score - threshold) / threshold)

        if _classifier_data:
            embedding = _extractor.extract_image_embedding(tensor).reshape(1, -1)
            knn = _classifier_data["knn"]
            le = _classifier_data["label_encoder"]
            pred = knn.predict(embedding)[0]
            defect_type = le.inverse_transform([pred])[0]
            proba = knn.predict_proba(embedding)[0]
            confidence = float(np.max(proba))

        return {
            "decision": "DEFECT",
            "defect_type": defect_type,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "threshold": round(threshold, 4),
        }
    else:
        confidence = max(0.0, 1.0 - (score / threshold))
        return {
            "decision": "OK",
            "defect_type": None,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "threshold": round(threshold, 4),
        }


def _get_log_stats() -> dict:
    """Read stats from inspections.csv."""
    csv_path = LOGS_DIR / "inspections.csv"
    if not csv_path.exists():
        return {
            "total": 0, "n_ok": 0, "n_defect": 0,
            "pct_ok": 0, "pct_defect": 0, "by_defect_type": {}
        }

    import csv
    rows = []
    try:
        with open(csv_path, newline="") as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception:
        return {"total": 0, "n_ok": 0, "n_defect": 0, "by_defect_type": {}}

    total = len(rows)
    n_ok = sum(1 for r in rows if r.get("decision", "").upper() == "OK")
    n_defect = total - n_ok
    by_type: dict = {}
    for r in rows:
        if r.get("decision", "").upper() == "DEFECT":
            dt = r.get("defect_type") or "unknown"
            by_type[dt] = by_type.get(dt, 0) + 1

    return {
        "total": total,
        "n_ok": n_ok,
        "n_defect": n_defect,
        "pct_ok": round(n_ok / total * 100, 1) if total else 0,
        "pct_defect": round(n_defect / total * 100, 1) if total else 0,
        "by_defect_type": by_type,
    }


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse)
async def serve_dashboard():
    """Serve the main dashboard HTML page."""
    index_path = STATIC_DIR / "index.html"
    if not index_path.exists():
        return HTMLResponse("<h1>Dashboard not found</h1><p>Expected: dashboard/static/index.html</p>", status_code=404)
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"))


@app.get("/api/status")
async def get_status():
    """Current system status."""
    models_ready = _load_models()
    stats = _get_log_stats()
    uptime = int(time.time() - _start_time)

    return {
        "status": "ok",
        "models_ready": models_ready,
        "classifier_ready": _classifier_data is not None,
        "uptime_s": uptime,
        "uptime_human": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s",
        "threshold": _model_info.get("threshold") if _model_info else None,
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
    }


@app.get("/api/stats")
async def get_stats():
    """Aggregate inspection statistics."""
    return _get_log_stats()


@app.get("/api/log")
async def get_log(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=200),
    decision: Optional[str] = None,
):
    """Paginated inspection log."""
    import csv
    csv_path = LOGS_DIR / "inspections.csv"
    if not csv_path.exists():
        return {"rows": [], "total": 0, "page": page, "per_page": per_page}

    try:
        with open(csv_path, newline="") as f:
            rows = list(csv.DictReader(f))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    rows.reverse()  # newest first

    if decision:
        rows = [r for r in rows if r.get("decision", "").upper() == decision.upper()]

    total = len(rows)
    start = (page - 1) * per_page
    end = start + per_page
    page_rows = rows[start:end]

    return {
        "rows": page_rows,
        "total": total,
        "page": page,
        "per_page": per_page,
        "pages": (total + per_page - 1) // per_page,
    }


@app.post("/api/infer")
async def run_inference(file: UploadFile = File(...)):
    """Upload an image and get an anomaly decision."""
    if not _load_models():
        raise HTTPException(
            status_code=503,
            detail="Models not loaded. Run python src/train_model.py first.",
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))
        result = _score_pil_image(img)
        result["filename"] = file.filename
        result["timestamp"] = datetime.now().isoformat()
        
        # Log the result to inspections log (DB and CSV)
        from logger import InspectionLogger
        logger = InspectionLogger(LOGS_DIR)
        
        # Convert PIL image to BGR numpy array for opencv thumbnail saving
        cv_img = np.array(img.convert("RGB"))
        cv_img = cv_img[:, :, ::-1].copy()
        
        logger.log(
            decision=result["decision"],
            score=result["score"],
            confidence=result["confidence"],
            defect_type=result["defect_type"],
            image=cv_img,
            source="upload"
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}\n{traceback.format_exc()}")


@app.get("/api/export")
async def export_csv():
    """Download the full inspection log as CSV."""
    csv_path = LOGS_DIR / "inspections.csv"
    if not csv_path.exists():
        raise HTTPException(status_code=404, detail="No inspection log found yet.")
    return FileResponse(
        path=str(csv_path),
        filename=f"zerodefect_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        media_type="text/csv",
    )


@app.get("/api/model-info")
async def get_model_info():
    """Return metadata about the loaded model."""
    _load_models()
    if not _model_info:
        return {"status": "no_model", "message": "Run python src/train_model.py first."}
    return {"status": "loaded", **_model_info}


# Serve static files from the static directory at the root path
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(DASHBOARD_DIR))
