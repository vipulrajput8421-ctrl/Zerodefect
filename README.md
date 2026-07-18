# ZeroDefect

> **YOLOv5n Object Detection for Aircraft Surface Defect Inspection**
> Tata Technologies InnoVent 2026-27 | Track 3.2.3.4: Intelligent Inspection & Defect Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![ONNX](https://img.shields.io/badge/ONNX-Runtime-orange)](https://onnxruntime.ai)
[![YOLOv5](https://img.shields.io/badge/YOLOv5-Nano-green)](https://github.com/ultralytics/yolov5)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

ZeroDefect detects surface defects on aircraft skin panels using a **YOLOv5 Nano** object detection model trained on 22k+ images:

1. **Pre-trained model** — detects 7 defect types out-of-the-box (no training required)
2. **Real-time bounding boxes** — localizes defects with class labels and confidence scores
3. **Edge-deployable** — ONNX (7.2 MB) and RKNN INT8 (2.6 MB) formats included
4. **Local inference** — no cloud API, no internet required

### Detected Defect Classes
| # | Class | Description |
|---|-------|-------------|
| 0 | `crack` | Structural cracks, fatigue lines |
| 1 | `dent` | Impact dents, surface deformations |
| 2 | `corrosion` | Rust, chemical wear, oxidation |
| 3 | `scratch` | Superficial scrapes, paint scratches |
| 4 | `paint-peel` | Flaking paint, coating degradation |
| 5 | `missing-head` | Missing rivet heads or fasteners |
| 6 | `defect` | Generic surface anomalies |

---

## Quick Start

```bash
# 1. Clone and enter project
cd ZeroDefect

# 2. Create virtual environment
python -m venv zerodefect_env

# Windows:
zerodefect_env\Scripts\activate

# macOS / Linux:
source zerodefect_env/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the model (if not already present)
pip install huggingface_hub
hf sync hf://buckets/prath0029/Zerodefect-1.0-bucket ./local
copy local\best.onnx models\best.onnx
copy local\best.pt models\best.pt
copy local\best.rknn models\best.rknn

# 5. Run inference on an image
python local/infer_onnx.py --model models/best.onnx --image test.jpg --conf 0.25

# 6. Start the live webcam demo
python src/live_demo.py

# 7. Start the web dashboard
python -m uvicorn dashboard.app:app --host 0.0.0.0 --port 8000 --reload
# Open: http://localhost:8000

# 8. View audit log
python src/view_log.py

# 9. Analyze model for edge deployment
python aim/quantize_model.py
```

---

## Project Structure

```
ZeroDefect/
├── models/
│   ├── best.onnx                ← YOLOv5n ONNX model (7.2 MB)
│   ├── best.pt                  ← PyTorch weights (3.8 MB)
│   ├── best.rknn                ← RKNN INT8 for RV1106 NPU (2.6 MB)
│   ├── data.yaml                ← Dataset config (7 classes)
│   └── (legacy: memory_bank.*, classifier.pkl, model_info.json)
│
├── src/
│   ├── yolo_detector.py         ← YOLOv5n ONNX detector class (NEW)
│   ├── utils.py                 ← Shared constants + transforms
│   ├── live_demo.py             ← Live webcam: bounding box detection
│   ├── logger.py                ← SQLite + CSV inspection logger
│   ├── view_log.py              ← Print audit log summary
│   ├── environment_check.py     ← Verify all deps are installed
│   ├── capture_good.py          ← Webcam: collect good-part images
│   ├── capture_defects.py       ← Webcam: collect defect examples
│   ├── train_model.py           ← (Legacy) PatchCore training
│   ├── evaluate_model.py        ← (Legacy) PatchCore evaluation
│   ├── finetune_fewshot.py      ← (Legacy) KNN classifier fine-tuning
│   └── download_aerospace_data.py ← Dataset downloader
│
├── dashboard/
│   ├── app.py                   ← FastAPI server (YOLO + VLM + PatchCore engines)
│   └── static/
│       ├── index.html           ← Dashboard UI
│       ├── app.js               ← Frontend logic (local YOLO inference)
│       └── style.css            ← Dashboard styling
│
├── local/                       ← HF bucket sync target
│   ├── infer_onnx.py            ← Standalone ONNX inference script
│   └── README.md                ← Model documentation
│
├── logs/                        ← Audit trail
│   ├── inspections.csv          ← Every decision logged here
│   ├── inspections.db           ← SQLite backing store
│   └── thumbnails/              ← Flagged-frame thumbnails
│
├── aim/                         ← Edge deployment
│   ├── quantize_model.py        ← Model analysis + export
│   ├── exported/                ← Quantized model outputs
│   └── firmware/
│       └── ZeroDefect_AIM.ino   ← Arduino sketch for AIM
│
├── requirements.txt
├── .gitignore
├── run_demo.bat                 ← Windows: one-click live demo
├── run_capture.bat              ← Windows: one-click capture menu
├── run_dashboard.bat            ← Windows: one-click dashboard
├── demo_day_checklist.md        ← Step-by-step demo day guide
└── presentation_outline.md      ← Stage 2 slide structure
```

---

## How It Works

### YOLOv5n Object Detection (Primary — v2.0)

The model is a **YOLOv5 Nano** trained on 22k+ images from merged aircraft surface defect datasets. It directly detects and localizes defects with bounding boxes — no separate training or anomaly scoring needed.

**Inference pipeline:**
1. Image is letterbox-resized to 640×640
2. ONNX Runtime runs the YOLOv5n forward pass (~7 MB model)
3. Non-Maximum Suppression filters overlapping detections
4. Bounding boxes are rescaled to original image coordinates
5. Each detection has: class name, confidence score, and bbox coordinates

**Validation metrics (on merged test set):**
- Precision: 42.8%
- Recall: 35.8%
- mAP@0.5: 31.9%
- mAP@0.5:0.95: 23.5%

### PatchCore Anomaly Detection (Legacy — v1.0)

The legacy pipeline learns "normal" from good-part images only, then flags anything abnormal using patch-level feature distances. Still available via `engine=patchcore` in the dashboard.

---

## Dashboard Engines

The web dashboard supports three inference engines:

| Engine | Command | Description |
|--------|---------|-------------|
| `yolo` (default) | `?engine=yolo` | Local YOLOv5n ONNX — fast, offline, bounding boxes |
| `vlm` | `?engine=vlm` | Ollama Gemma 3 4B vision model — detailed reasoning |
| `patchcore` | `?engine=patchcore` | Legacy anomaly detection — requires trained memory bank |

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 / AMD Ryzen 5 | Any modern laptop |
| RAM | 4 GB | 8 GB+ |
| GPU | None required | CUDA GPU (speeds inference) |
| Webcam | 720p USB | 1080p or better |
| Python | 3.10 | 3.11+ |

---

## Edge Deployment

### Luckfox Pico Max (RV1106 NPU)

The `best.rknn` model is pre-compiled and INT8-quantized for the RV1106 NPU:

```bash
# Push model to the board
adb push models/best.rknn /userdata/

# Cross-compile the inference runner
./build-linux.sh -t rv1106 -a armv7l -d yolov5

# Execute on board
./rknn_yolov5_demo /userdata/best.rknn test.jpg
```

### AIM (Aerospace Inspection Module)

See [`aim/README.md`](aim/README.md) for the full hardware guide. Run `python aim/quantize_model.py` for model size analysis.

---

## Audit Trail & Traceability

Every inspection is logged with timestamp, image path, decision, defect type, and confidence. This design directly addresses the **EU AI Act Article 12** traceability requirements for high-risk industrial AI systems, effective August 2026.

```bash
python src/view_log.py        # print summary
python src/view_log.py --export audit_export.csv   # export
```

---

## License

MIT License. See [LICENSE](LICENSE).

---

*ZeroDefect v2.0 — InnoVent 2026-27 Submission*
