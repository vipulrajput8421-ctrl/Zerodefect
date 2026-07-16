"""
live_demo.py — Real-time visual quality control feed with on-screen HUD.

Throttles capture to 3-5 FPS, extracts features, checks anomaly threshold,
runs defect classification, and records decisions to the SQLite/CSV audit logs.
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path
import cv2
import numpy as np
import torch

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS_DIR, LOGS_DIR, DEFAULT_IMAGE_SIZE,
    load_model_info, get_extractor, resolve_device
)
from logger import InspectionLogger


def draw_hud(frame, decision: str, score: float, threshold: float, 
             confidence: float, defect_type: str = None, fps: float = 0.0):
    """Draw a semi-opaque HUD card with status overlays on the video stream."""
    h, w, _ = frame.shape
    
    # HUD Box coordinates (top-left)
    hud_w = 400
    hud_h = 160
    
    # Overlay transparent box (alpha blending)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Choose theme color based on inspection result
    if decision == "OK":
        status_color = (0, 245, 160)     # Bright Green (BGR: 160, 245, 0)
        status_text = "✓ OK"
        details_text = "NORMAL VARIATION"
    else:
        status_color = (68, 68, 255)     # Bright Red (BGR: 255, 68, 68)
        status_text = f"✗ DEFECT: {defect_type.upper() if defect_type else 'ANOMALY'}"
        details_text = "OUT OF SPECIFICATION"

    # Status title
    cv2.putText(frame, status_text, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
    cv2.putText(frame, details_text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 146, 164), 1)

    # Anomaly Scores
    cv2.putText(frame, f"Score    : {score:.4f}", (25, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 237, 245), 1)
    cv2.putText(frame, f"Threshold: {threshold:.4f}", (25, 115), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 146, 164), 1)

    # ASCII Confidence Bar
    conf_pct = int(confidence * 100)
    bar_chars = int(confidence * 10)
    bar_chars = max(0, min(10, bar_chars))
    ascii_bar = "█" * bar_chars + "░" * (10 - bar_chars)
    
    cv2.putText(frame, f"Conf     : {ascii_bar} {conf_pct}%", (25, 145), cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # Status border highlights
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), status_color, 1)

    # FPS status bottom-left
    cv2.rectangle(frame, (10, h - 35), (100, h - 10), (10, 10, 10), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)


def main():
    parser = argparse.ArgumentParser(description="Live Webcam Inspection Client")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR), help="Path to models folder")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Size model expects")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps or auto")
    parser.add_argument("--fps-limit", type=float, default=3.0, help="Maximum evaluation rate (FPS)")
    parser.add_argument("--log-dir", type=str, default=str(LOGS_DIR), help="Folder to save DB/CSVs")
    parser.add_argument("--no-log", action="store_true", help="Disable audit logging")
    parser.add_argument("--threshold", type=float, default=None, help="Override baseline threshold")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    memory_bank_path = model_dir / "memory_bank.pkl"
    model_info_path = model_dir / "model_info.json"

    # Pre-checks
    if not memory_bank_path.exists():
        print(f"[ERROR] Model not trained. File not found: {memory_bank_path}")
        print("Please run baseline training first: python src/train_model.py")
        sys.exit(1)

    # Load model info
    model_info = load_model_info(model_info_path)
    threshold = model_info.get("threshold", 1.0)
    image_size = model_info.get("image_size", args.image_size)

    if args.threshold is not None:
        threshold = args.threshold

    # Load NearestNeighbors model
    with open(memory_bank_path, "rb") as f:
        nn_model = pickle.load(f)

    # Check for classifier
    classifier = None
    classifier_path = model_dir / "classifier.pkl"
    if classifier_path.exists():
        with open(classifier_path, "rb") as f:
            classifier = pickle.load(f)
        print("[*] Loaded few-shot classifier payload successfully.")
    else:
        print("[WARN] No classifier file found. Defective parts will be classified as 'anomaly'.")

    device = resolve_device(args.device)
    extractor = get_extractor(device)

    # Initialize logger
    logger = None if args.no_log else InspectionLogger(Path(args.log_dir))
    if logger:
        print(f"[*] Audit logs writing to: {logger.db_path}")

    # Set up OpenCV Capture
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with device ID: {args.camera_id}")
        sys.exit(1)

    # Window configuration
    window_name = "ZeroDefect — Live Inspection"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[*] Starting live inspection thread. Limit: {args.fps_limit} FPS.")
    print("[*] PRESS [Q] on the camera window to quit.\n")

    frame_delay = 1.0 / args.fps_limit
    last_eval_time = 0.0
    fps = 0.0

    while True:
        start_time = time.time()
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read camera frame.")
            break

        current_time = time.time()
        
        # Throttled inspection evaluation
        if (current_time - last_eval_time) >= frame_delay:
            # Preprocess current frame for PyTorch model
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = cv2.resize(rgb_frame, (image_size, image_size))
            
            # Convert to normalized tensor [1, 3, H, W]
            tensor = torch.from_numpy(pil_img).permute(2, 0, 1).float().div(255.0)
            # Normalize with ImageNet stats
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            tensor = ((tensor - mean) / std).unsqueeze(0).to(device)

            # Score anomaly
            patches = extractor.extract_flat_patches(tensor)
            dists, _ = nn_model.kneighbors(patches)
            score = float(np.percentile(dists.flatten(), 99))

            decision = "DEFECT" if score > threshold else "OK"
            defect_type = None

            if decision == "DEFECT":
                if classifier:
                    embedding = extractor.extract_image_embedding(tensor).reshape(1, -1)
                    knn = classifier["knn"]
                    le = classifier["label_encoder"]
                    pred = knn.predict(embedding)[0]
                    defect_type = le.inverse_transform([pred])[0]
                    
                    # Estimate confidence by KNN class probabilities
                    proba = knn.predict_proba(embedding)[0]
                    confidence = float(np.max(proba))
                else:
                    defect_type = "anomaly"
                    # Scale confidence distance from threshold
                    confidence = min(1.0, (score - threshold) / threshold)
            else:
                # OK confidence scales with proximity to threshold
                confidence = max(0.0, 1.0 - (score / threshold))

            # Logging thread
            if logger:
                logger.log(
                    decision=decision,
                    score=score,
                    confidence=confidence,
                    defect_type=defect_type,
                    image=frame
                )

            # FPS counter
            fps = 1.0 / (current_time - last_eval_time) if last_eval_time > 0 else args.fps_limit
            last_eval_time = current_time

        # Update frame overlay and window
        display_frame = frame.copy()
        draw_hud(display_frame, decision, score, threshold, confidence, defect_type, fps)
        
        cv2.imshow(window_name, display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    if logger:
        stats = logger.get_summary()
        print("\n" + "="*50)
        print(" INSPECTION SESSION SUMMARY")
        print("="*50)
        print(f" Total Inspections : {stats['total']}")
        print(f" OK Parts          : {stats['n_ok']} ({stats['pct_ok']:.1f}%)")
        print(f" Defective Parts   : {stats['n_defect']} ({stats['pct_defect']:.1f}%)")
        if stats["by_defect_type"]:
            print(" Breakdown by Defect:")
            for k, v in stats["by_defect_type"].items():
                print(f"   - {k:<12} : {v}")
        print("="*50)


if __name__ == "__main__":
    main()
