# ZeroDefect

> **Few-Shot Anomaly Detection for Automotive Component QC**
> Tata Technologies InnoVent 2026-27 | Track 3.2.3.4: Intelligent Inspection & Defect Detection

[![Python](https://img.shields.io/badge/Python-3.10+-blue)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.3-orange)](https://pytorch.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

---

## What This Is

ZeroDefect detects manufacturing defects in automotive components (M8 hex bolts) using an **anomaly-first, few-shot** approach:

1. **Train on good parts only** — the model learns what "normal" looks like
2. **Flag anything abnormal** — no labeled defect data required at training time
3. **Fine-tune with 5–20 examples per defect type** — classify crack vs scratch vs dent
4. **Deploy standalone on ESP32-CAM** — no laptop required at production time

This is the same architectural approach used by commercial systems (Overview.ai, Averroes.ai) as of 2026.

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

# 4. Check your environment
python src/environment_check.py

# 5. Collect good-part images (aim for 120)
python src/capture_good.py --target-count 120

# 6. Train the anomaly model
python src/train_model.py

# 7. Validate (put some test images in data/test/ first)
python src/evaluate_model.py

# 8. Collect defect examples (5–20 each)
python src/capture_defects.py --defect-type crack
python src/capture_defects.py --defect-type scratch
python src/capture_defects.py --defect-type dent

# 9. Fine-tune few-shot classifier
python src/finetune_fewshot.py

# 10. Run live demo
python src/live_demo.py

# 11. View audit log
python src/view_log.py

# 12. Presentation stats
python data/summary_stats.py
```

---

## Project Structure

```
ZeroDefect/
├── data/
│   ├── good/                    ← 100–150 good-part images (Step 2)
│   ├── defects/
│   │   ├── crack/               ← 5–20 crack examples (Step 6)
│   │   ├── scratch/             ← 5–20 scratch examples (Step 6)
│   │   └── dent/                ← 5–20 dent examples (Step 6)
│   ├── test/                    ← Mixed good+bad for evaluation (Step 5)
│   ├── README.md                ← Dataset spec & photography guide
│   └── summary_stats.py         ← Print all dataset + model stats
│
├── src/
│   ├── utils.py                 ← Shared backbone + transforms (import from here)
│   ├── environment_check.py     ← Verify all deps are installed (Step 3)
│   ├── capture_good.py          ← Webcam: collect good-part images (Step 2)
│   ├── capture_defects.py       ← Webcam: collect defect examples (Step 6)
│   ├── train_model.py           ← PatchCore training on good parts (Step 4)
│   ├── evaluate_model.py        ← Score test images, compute metrics (Step 5)
│   ├── finetune_fewshot.py      ← KNN classifier on defect embeddings (Step 7)
│   ├── live_demo.py             ← Live webcam: OK/DEFECT overlay (Step 8+9)
│   ├── logger.py                ← SQLite + CSV inspection logger (Step 9)
│   └── view_log.py              ← Print audit log summary (Step 9)
│
├── models/                      ← Saved model artifacts
│   ├── memory_bank.npy          ← Coreset patch features (after training)
│   ├── memory_bank.pkl          ← Fitted NearestNeighbors model
│   ├── model_info.json          ← Threshold + metadata
│   └── classifier.pkl           ← Few-shot KNN classifier (after fine-tuning)
│
├── logs/                        ← Audit trail
│   ├── inspections.csv          ← Every decision logged here
│   ├── inspections.db           ← SQLite backing store
│   └── thumbnails/              ← Flagged-frame thumbnails
│
├── esp32/                       ← Phase 2: Edge deployment
│   ├── README.md                ← Hardware guide + wiring diagram
│   ├── quantize_model.py        ← Export to ONNX / TFLite INT8
│   ├── exported/                ← Quantized model outputs
│   └── firmware/
│       └── ZeroDefect_ESP32.ino ← Arduino sketch for ESP32-CAM
│
├── web/
│   └── index.html               ← Self-contained page served by ESP32
│
├── requirements.txt
├── .gitignore
├── run_demo.bat                 ← Windows: one-click live demo
├── run_capture.bat              ← Windows: one-click capture menu
├── demo_day_checklist.md        ← Step-by-step demo day guide
└── presentation_outline.md     ← Stage 2 slide structure
```

---

## How It Works (Plain Language)

**Why no labeled defect data for training?**

Traditional inspection AI learns "here's a crack, here's a scratch" — which means you need hundreds of labeled defect photos before training can start. That takes weeks and thousands of dollars.

ZeroDefect does the opposite: we only show it good parts. The model (PatchCore, using a pretrained ResNet18 backbone) learns to represent every patch of every good-part image as a point in a high-dimensional feature space. We save a compact "memory bank" of these points.

When a new image arrives, we extract its features and ask: *how far are these features from the nearest point in the memory bank?* Good parts are close to something we've seen before — low distance. Defective parts have patches that look nothing like any good part — high distance.

That distance IS the anomaly score. We flag anything above a threshold (learned automatically from the training data distribution).

The few-shot step then adds a lightweight KNN classifier that uses the same feature space to distinguish *which kind* of defect it is — using just 5–20 examples per type.

---

## Hardware Requirements

| Component | Minimum | Recommended |
|---|---|---|
| CPU | Intel i5 / AMD Ryzen 5 | Any modern laptop |
| RAM | 4 GB | 8 GB+ |
| GPU | None required | CUDA GPU (speeds training) |
| Webcam | 720p USB | 1080p or better |
| Python | 3.10 | 3.11+ |

---

## Phase 2 — ESP32-CAM Edge Deployment

See [`esp32/README.md`](esp32/README.md) for the full hardware guide.

The trained model is exported to ONNX → TFLite INT8 and flashed to an ESP32-CAM board (~$5). The board runs its own web server — any browser on the same Wi-Fi network sees live inspection results with no laptop.

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

*ZeroDefect v1.0 — InnoVent 2026-27 Submission*
