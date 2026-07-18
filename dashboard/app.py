"""
dashboard/app.py — ZeroDefect Local Dashboard API
===================================================

FastAPI server providing:
  GET  /                  → Serve the dashboard HTML page
  GET  /api/status        → Current system status + recent stats
  GET  /api/log           → Paginated inspection log
  GET  /api/stats         → Aggregate stats
  POST /api/infer         → Upload image, get defect detection result
  GET  /api/export        → Download inspections.csv

Inference Engines:
  - yolo      (default) — Local YOLOv5n ONNX detector (7 defect classes)
  - vlm       — Ollama Gemma 3 vision model
  - patchcore — Legacy PatchCore anomaly detection

Run with:
  uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload

Then open: http://localhost:8000
"""

import sys
import os
import io
import time
import json
import base64
import pickle
import traceback
import httpx
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
    description="Aircraft surface defect detection dashboard — YOLOv5n + VLM",
    version="2.0.0",
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
_yolo_detector = None
_hybrid_detector = None
_nn_model = None
_extractor = None
_classifier_data = None
_model_info = None
_start_time = time.time()


def _load_yolo():
    """Load YOLOv5n ONNX detector lazily on first use."""
    global _yolo_detector

    if _yolo_detector is not None:
        return True

    try:
        from yolo_detector import get_yolo_detector

        yolo_path = MODELS_DIR / "best_balanced.onnx"
        if not yolo_path.exists():
            print(f"[WARN] YOLO model not found at {yolo_path}")
            return False

        _yolo_detector = get_yolo_detector(yolo_path)
        return True

    except Exception as e:
        print(f"[WARN] Could not load YOLO model: {e}")
        return False

def _load_hybrid():
    """Load Hybrid (YOLO + Moondream) detector lazily."""
    global _hybrid_detector
    if _hybrid_detector is not None:
        return True
    try:
        from fallback_detector import FallbackDetector
        if not _load_yolo():
            return False
            
        moondream_path = MODELS_DIR / "moondream-2b-int8.mf"
        if not moondream_path.exists():
            print(f"[WARN] Moondream model not found at {moondream_path}")
            return False
            
        _hybrid_detector = FallbackDetector(_yolo_detector, str(moondream_path), conf_threshold=0.05)
        return True
    except Exception as e:
        print(f"[WARN] Could not load Hybrid model: {e}")
        return False


def _load_models():
    """Load legacy PatchCore models lazily on first inference request."""
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
        print(f"[WARN] Could not load PatchCore models: {e}")
        return False


# ── YOLO Inference ─────────────────────────────────────────────────────────────

def _score_with_yolo(img: Image.Image) -> dict:
    """Run YOLOv5n ONNX detection on a PIL image."""
    CONF_THRESHOLD = 0.25
    IOU_THRESHOLD = 0.45

    # Convert PIL to BGR numpy (OpenCV format)
    img_rgb = np.array(img.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()

    detections = _yolo_detector.detect(img_bgr, conf=CONF_THRESHOLD, iou=IOU_THRESHOLD)

    if len(detections) > 0:
        # Use the highest-confidence detection as the primary result
        best = max(detections, key=lambda d: d["confidence"])
        return {
            "decision": "DEFECT",
            "defect_type": best["class_name"],
            "confidence": best["confidence"],
            "score": best["confidence"],
            "threshold": CONF_THRESHOLD,
            "engine": "yolo",
            "num_detections": len(detections),
            "all_detections": detections,
        }
    else:
        return {
            "decision": "OK",
            "defect_type": None,
            "confidence": 1.0,
            "score": 0.0,
            "threshold": CONF_THRESHOLD,
            "engine": "yolo",
            "num_detections": 0,
            "all_detections": [],
        }

def _parse_vlm_defect_type(desc: str) -> str:
    desc_lower = desc.lower()
    # Check for specific defects in order of priority (specific to generic)
    if "hole" in desc_lower:
        return "hole"
    elif "scratch" in desc_lower:
        return "scratch"
    elif "dent" in desc_lower:
        return "dent"
    elif "crack" in desc_lower:
        return "crack"
    elif "peel" in desc_lower or "paint" in desc_lower:
        return "peeling paint"
    elif "corrosion" in desc_lower:
        return "corrosion"
    elif "patch" in desc_lower:
        return "white patch"
    return "unknown"


def _score_with_hybrid(img: Image.Image) -> dict:
    """Run Hybrid detection on a PIL image."""
    img_rgb = np.array(img.convert("RGB"))
    img_bgr = img_rgb[:, :, ::-1].copy()
    
    res = _hybrid_detector.detect(img, img_bgr)
    
    if res["engine"] == "yolo":
        best = max(res["detections"], key=lambda x: x["confidence"])
        return {
            "decision": "DEFECT",
            "defect_type": best["class_name"],
            "confidence": best["confidence"],
            "score": best["confidence"],
            "threshold": _hybrid_detector.conf_threshold,
            "engine": "yolo (hybrid)",
            "num_detections": len(res["detections"]),
            "all_detections": res["detections"],
        }
    else:
        is_ok = not res.get("is_defect", False)
        defect_type = None
        if not is_ok:
            defect_type = _parse_vlm_defect_type(res.get("description", ""))
            if defect_type == "unknown":
                defect_type = _parse_vlm_defect_type(res.get("ans_type", ""))
        return {
            "decision": "OK" if is_ok else "DEFECT",
            "defect_type": defect_type,
            "confidence": 1.0 if is_ok else 0.75,
            "score": 0.0 if is_ok else 0.75,
            "threshold": _hybrid_detector.conf_threshold,
            "engine": "moondream_fallback",
            "num_detections": 0,
            "all_detections": [],
            "reasoning": res.get("description", ""),
        }

def _score_with_moondream(img: Image.Image) -> dict:
    """Run standalone Moondream detection on a PIL image."""
    encoded = _hybrid_detector.moondream.encode_image(img)
    
    prompt_is_defect = "Does this aircraft skin have defects? Answer YES or NO."
    ans_is_defect = _hybrid_detector.moondream.query(encoded, prompt_is_defect)["answer"].strip().lower()
    is_defect = "yes" in ans_is_defect or ("no" not in ans_is_defect and "normal" not in ans_is_defect)
    
    if is_defect:
        prompt_type = (
            "Which of the following defects is most prominent in this image: "
            "scratch, dent, crack, hole, peeling paint, white patch, or corrosion? "
            "Answer with only the category name."
        )
        ans_type = _hybrid_detector.moondream.query(encoded, prompt_type)["answer"].strip().lower()
        prompt_desc = "Describe the defect in one short sentence."
        ans_desc = _hybrid_detector.moondream.query(encoded, prompt_desc)["answer"].strip()
        
        defect_type = _parse_vlm_defect_type(ans_desc)
        if defect_type == "unknown":
            defect_type = _parse_vlm_defect_type(ans_type)
        
        return {
            "decision": "DEFECT",
            "defect_type": defect_type,
            "confidence": 0.75,
            "score": 0.75,
            "threshold": 0.0,
            "engine": "moondream",
            "num_detections": 0,
            "all_detections": [],
            "reasoning": ans_desc,
        }
    else:
        return {
            "decision": "OK",
            "defect_type": None,
            "confidence": 1.0,
            "score": 0.0,
            "threshold": 0.0,
            "engine": "moondream",
            "num_detections": 0,
            "all_detections": [],
            "reasoning": "No defects detected.",
        }

# ── PatchCore Inference (Legacy) ───────────────────────────────────────────────

def _score_pil_image(img: Image.Image) -> dict:
    """Run PatchCore inference on a PIL image (legacy engine)."""
    from utils import get_transform, DEFAULT_IMAGE_SIZE

    image_size = _model_info.get("image_size", DEFAULT_IMAGE_SIZE) if _model_info else DEFAULT_IMAGE_SIZE
    threshold = _model_info.get("threshold", 1.0) if _model_info else 1.0

    # Aerospace Domain Specifics: Isolate defects specifically on structural aircraft components
    import cv2
    img_np = np.array(img.convert("RGB"))
    hsv = cv2.cvtColor(img_np, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv)
    # Mask of the aircraft panel surface (low saturation gray/silver/white, reasonable brightness)
    panel_mask = (s < 50) & (v > 40) & (v < 252)
    # Isolate component: fill background clutter with canonical normal metal gray (180, 183, 188)
    isolated_np = img_np.copy()
    isolated_np[~panel_mask] = [180, 183, 188]
    img_processed = Image.fromarray(isolated_np)

    transform = get_transform(image_size)
    tensor = transform(img_processed.convert("RGB")).unsqueeze(0)

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


# ── VLM Inference via Ollama ─────────────────────────────────────────────────
OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "gemma3:4b"

VLM_PROMPT = """You are an aerospace quality control inspector. Analyze this image and determine if it shows a defect on a metal aircraft surface.

Rules:
1. If the image shows a human face, hand, body part, person, or any non-metal object, respond with decision "OK" and defect_type null. These are NOT defects.
2. If the image shows a clean, undamaged metal/aluminum aircraft surface, respond with decision "OK".
3. If the image shows a metal surface WITH a visible defect (scratch, crack, dent, corrosion, delamination, paint damage), respond with decision "DEFECT" and specify the defect_type.
4. Confidence should be between 0.0 and 1.0.

Respond ONLY with this exact JSON format, no other text:
{"decision": "OK" or "DEFECT", "defect_type": null or "scratch" or "crack" or "dent" or "corrosion" or "delamination" or "paint_damage", "confidence": 0.0 to 1.0, "reasoning": "brief explanation"}"""


def _score_with_vlm(img: Image.Image) -> dict:
    """Run inference using Ollama vision model (Gemma 3 4B)."""
    # Convert PIL image to base64
    buf = io.BytesIO()
    img.convert("RGB").save(buf, format="JPEG", quality=85)
    img_b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    payload = {
        "model": OLLAMA_MODEL,
        "prompt": VLM_PROMPT,
        "images": [img_b64],
        "stream": False,
        "options": {
            "temperature": 0.1,
            "num_predict": 256,
        },
    }

    try:
        resp = httpx.post(OLLAMA_URL, json=payload, timeout=60.0)
        resp.raise_for_status()
        raw_response = resp.json().get("response", "")
        print(f"[VLM] Raw response: {raw_response}")

        # Extract JSON from the response (handle markdown code fences)
        json_str = raw_response.strip()
        if "```" in json_str:
            parts = json_str.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("{"):
                    json_str = part
                    break

        vlm_result = json.loads(json_str)

        decision = vlm_result.get("decision", "OK").upper()
        defect_type = vlm_result.get("defect_type", None)
        confidence = float(vlm_result.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        if decision == "OK":
            defect_type = None

        score = confidence if decision == "DEFECT" else (1.0 - confidence)

        return {
            "decision": decision,
            "defect_type": defect_type,
            "confidence": round(confidence, 4),
            "score": round(score, 4),
            "threshold": 0.5,
            "engine": "vlm",
            "reasoning": vlm_result.get("reasoning", ""),
        }

    except httpx.ConnectError:
        print("[VLM] Ollama not running. Falling back.")
        return None
    except Exception as e:
        print(f"[VLM] Error: {e}\n{traceback.format_exc()}")
        return None


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
    
    headers = {
        "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
        "Pragma": "no-cache",
        "Expires": "0"
    }
    return HTMLResponse(content=index_path.read_text(encoding="utf-8"), headers=headers)


@app.get("/api/status")
async def get_status():
    """Current system status."""
    yolo_ready = _load_yolo()
    patchcore_ready = _load_models()
    stats = _get_log_stats()
    uptime = int(time.time() - _start_time)

    return {
        "status": "ok",
        "models_ready": yolo_ready,
        "yolo_ready": yolo_ready,
        "patchcore_ready": patchcore_ready,
        "classifier_ready": _classifier_data is not None,
        "uptime_s": uptime,
        "uptime_human": f"{uptime // 3600}h {(uptime % 3600) // 60}m {uptime % 60}s",
        "threshold": 0.25 if yolo_ready else (_model_info.get("threshold") if _model_info else None),
        "stats": stats,
        "timestamp": datetime.now().isoformat(),
        "engine": "yolo" if yolo_ready else "patchcore",
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
async def run_inference(
    file: UploadFile = File(...),
    engine: str = Query("yolo", pattern="^(yolo|vlm|patchcore|hybrid|moondream)$"),
):
    """Upload an image and get a defect detection result.
    
    Query params:
        engine: 'yolo' (YOLOv5n ONNX), 'vlm' (Gemma 3 vision), 'patchcore' (legacy), 'hybrid', or 'moondream'
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        result = None

        if engine == "hybrid":
            if _load_hybrid():
                result = _score_with_hybrid(img)
                if result and "all_detections" in result:
                    result["all_detections"] = result["all_detections"][:10]
            else:
                print("[INFER] Hybrid unavailable, falling back to YOLO")
                engine = "yolo"

        if engine == "moondream":
            if _load_hybrid():
                result = _score_with_moondream(img)
            else:
                print("[INFER] Moondream unavailable, falling back to YOLO")
                engine = "yolo"

        if engine == "yolo":
            if _load_yolo():
                result = _score_with_yolo(img)
                # Strip all_detections from the response to keep it lightweight
                if result and "all_detections" in result:
                    result["all_detections"] = result["all_detections"][:10]  # cap at 10
            else:
                print("[INFER] YOLO unavailable, falling back to VLM")
                engine = "vlm"

        if engine == "vlm":
            result = _score_with_vlm(img)
            if result is None:
                print("[INFER] VLM unavailable, falling back to PatchCore")
                engine = "patchcore"

        if engine == "patchcore":
            if not _load_models():
                raise HTTPException(
                    status_code=503,
                    detail="No models loaded. Place best.onnx in models/ or run train_model.py.",
                )
            result = _score_pil_image(img)
            result["engine"] = "patchcore"

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
            score=result.get("score", 0.0),
            confidence=result["confidence"],
            defect_type=result.get("defect_type"),
            image=cv_img,
            source=result.get("engine", engine)
        )
        
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference failed: {e}\n{traceback.format_exc()}")


@app.post("/api/log-result")
async def log_result(
    file: UploadFile = File(...),
    decision: str    = Query("DEFECT"),
    defect_type: str = Query(""),
    confidence: float = Query(0.5),
    score: float      = Query(0.5),
    engine: str       = Query("yolo"),
):
    """
    Accept a pre-scored result from the frontend and save it
    to the inspection audit log (CSV + DB + thumbnail) without running inference.
    """
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents))

        from logger import InspectionLogger
        logger = InspectionLogger(LOGS_DIR)

        cv_img = np.array(img.convert("RGB"))
        cv_img = cv_img[:, :, ::-1].copy()

        logger.log(
            decision=decision.upper(),
            score=round(score, 4),
            confidence=round(confidence, 4),
            defect_type=defect_type or None,
            image=cv_img,
            source=engine,
        )

        return {
            "status": "logged",
            "decision": decision.upper(),
            "defect_type": defect_type or None,
            "confidence": round(confidence, 4),
            "engine": engine,
        }
    except Exception as e:
        # Non-fatal — frontend handles logging failures gracefully
        raise HTTPException(status_code=500, detail=f"Log failed: {e}")


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


@app.post("/api/clear")
async def clear_logs():
    """Clear all inspection log history and thumbnails."""
    from logger import InspectionLogger
    try:
        logger = InspectionLogger(LOGS_DIR)
        if logger.clear():
            return {"status": "success", "message": "All inspection history cleared."}
        else:
            raise HTTPException(status_code=500, detail="Failed to clear logs on disk.")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/model-info")
async def get_model_info():
    """Return metadata about the loaded model."""
    yolo_ready = _load_yolo()

    if yolo_ready and _yolo_detector:
        info = _yolo_detector.get_model_info()
        info["status"] = "loaded"
        info["engine"] = "yolo"
        return info

    # Fallback to PatchCore info
    _load_models()
    if _model_info:
        return {"status": "loaded", "engine": "patchcore", **_model_info}

    return {"status": "no_model", "message": "Place best.onnx in models/ or run train_model.py."}


# Mount thumbnails first so it doesn't get intercepted by the root catch-all
app.mount("/logs/thumbnails", StaticFiles(directory=LOGS_DIR / "thumbnails"), name="thumbnails")

# Serve static files from the static directory at the root path
app.mount("/", StaticFiles(directory=STATIC_DIR), name="static")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, app_dir=str(DASHBOARD_DIR))
