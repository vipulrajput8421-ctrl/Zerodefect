"""
live_demo.py — Real-time YOLOv5n defect detection with on-screen HUD.

Uses the pre-trained ONNX model to detect aircraft surface defects in real-time
from a webcam feed. Draws bounding boxes and a HUD overlay with detection stats.

Throttles capture to 3-5 FPS and records decisions to the SQLite/CSV audit logs.
"""

import os
import sys
import time
import argparse
from pathlib import Path
import cv2
import numpy as np

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    MODELS_DIR, LOGS_DIR, YOLO_MODEL_PATH,
    YOLO_CONF_THRESHOLD, YOLO_IOU_THRESHOLD,
    resolve_device
)
from yolo_detector import YOLODetector, get_yolo_detector
from logger import InspectionLogger


def draw_hud(frame, decision: str, detections: list, confidence: float,
             defect_type: str = None, fps: float = 0.0):
    """Draw a semi-opaque HUD card with detection status overlays."""
    h, w, _ = frame.shape
    
    # HUD Box coordinates (top-left)
    hud_w = 420
    hud_h = 160
    
    # Overlay transparent box (alpha blending)
    overlay = frame.copy()
    cv2.rectangle(overlay, (10, 10), (10 + hud_w, 10 + hud_h), (20, 20, 20), -1)
    cv2.addWeighted(overlay, 0.75, frame, 0.25, 0, frame)
    
    # Choose theme color based on inspection result
    if decision == "OK":
        status_color = (0, 245, 160)     # Bright Green (BGR)
        status_text = "✓ OK — No Defects"
        details_text = "SURFACE CLEAR"
    else:
        status_color = (68, 68, 255)     # Bright Red (BGR)
        status_text = f"✗ DEFECT: {defect_type.upper() if defect_type else 'DETECTED'}"
        details_text = f"{len(detections)} DETECTION{'S' if len(detections) != 1 else ''} FOUND"

    # Status title
    cv2.putText(frame, status_text, (25, 45), cv2.FONT_HERSHEY_DUPLEX, 0.8, status_color, 2)
    cv2.putText(frame, details_text, (25, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (136, 146, 164), 1)

    # Detection summary
    if detections:
        # Show top 3 detections
        for i, det in enumerate(detections[:3]):
            y_pos = 95 + i * 20
            label = f"{det['class_name']}: {det['confidence']:.1%}"
            cv2.putText(frame, label, (25, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 237, 245), 1)
    else:
        cv2.putText(frame, f"Conf: {confidence:.1%}", (25, 95),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (232, 237, 245), 1)
        cv2.putText(frame, f"Threshold: {YOLO_CONF_THRESHOLD}", (25, 115),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (136, 146, 164), 1)

    # ASCII Confidence Bar
    conf_pct = int(confidence * 100)
    bar_chars = max(0, min(10, int(confidence * 10)))
    ascii_bar = "█" * bar_chars + "░" * (10 - bar_chars)
    cv2.putText(frame, f"Conf     : {ascii_bar} {conf_pct}%", (25, 145),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1)

    # Status border highlights
    cv2.rectangle(frame, (10, 10), (10 + hud_w, 10 + hud_h), status_color, 1)

    # FPS status bottom-left
    cv2.rectangle(frame, (10, h - 35), (100, h - 10), (10, 10, 10), -1)
    cv2.putText(frame, f"FPS: {fps:.1f}", (20, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)

    # Engine badge bottom-right
    badge_text = "YOLO ONNX"
    cv2.rectangle(frame, (w - 120, h - 35), (w - 10, h - 10), (10, 10, 10), -1)
    cv2.putText(frame, badge_text, (w - 110, h - 17), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 245, 160), 1)


def main():
    parser = argparse.ArgumentParser(description="Live Webcam Defect Detection (YOLOv5n ONNX)")
    parser.add_argument("--camera-id", type=int, default=0, help="Camera device index")
    parser.add_argument("--model", type=str, default=str(YOLO_MODEL_PATH), help="Path to ONNX model")
    parser.add_argument("--conf", type=float, default=YOLO_CONF_THRESHOLD, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=YOLO_IOU_THRESHOLD, help="IoU threshold for NMS")
    parser.add_argument("--fps-limit", type=float, default=5.0, help="Maximum evaluation rate (FPS)")
    parser.add_argument("--log-dir", type=str, default=str(LOGS_DIR), help="Folder to save DB/CSVs")
    parser.add_argument("--no-log", action="store_true", help="Disable audit logging")
    args = parser.parse_args()

    # Load YOLO detector
    model_path = Path(args.model)
    if not model_path.exists():
        print(f"[ERROR] ONNX model not found: {model_path}")
        print("Download it with: hf sync hf://buckets/prath0029/Zerodefect-1.0-bucket ./local")
        print("Then copy: copy local\\best.onnx models\\best.onnx")
        sys.exit(1)

    detector = get_yolo_detector(model_path)
    print(f"[*] Model loaded: {model_path.name} ({len(detector.classes)} classes)")

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
    window_name = "ZeroDefect — Live Inspection (YOLOv5n)"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    print(f"\n[*] Starting live detection. Conf: {args.conf}, Limit: {args.fps_limit} FPS.")
    print("[*] PRESS [Q] on the camera window to quit.\n")

    frame_delay = 1.0 / args.fps_limit
    last_eval_time = 0.0
    fps = 0.0

    # Initialize state for display between inference frames
    decision = "OK"
    confidence = 1.0
    defect_type = None
    detections = []

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read camera frame.")
            break

        current_time = time.time()
        
        # Throttled inference
        if (current_time - last_eval_time) >= frame_delay:
            # Run YOLO detection on current frame
            detections = detector.detect(frame, conf=args.conf, iou=args.iou)

            if detections:
                best = max(detections, key=lambda d: d["confidence"])
                decision = "DEFECT"
                defect_type = best["class_name"]
                confidence = best["confidence"]
            else:
                decision = "OK"
                defect_type = None
                confidence = 1.0

            # Logging
            if logger:
                logger.log(
                    decision=decision,
                    score=confidence if decision == "DEFECT" else 0.0,
                    confidence=confidence,
                    defect_type=defect_type,
                    image=frame,
                    source="yolo"
                )

            # FPS counter
            fps = 1.0 / (current_time - last_eval_time) if last_eval_time > 0 else args.fps_limit
            last_eval_time = current_time

        # Draw bounding boxes from latest detections
        display_frame = frame.copy()
        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = detector.colors[cls_id % len(detector.colors)]

            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, 2)
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(display_frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(display_frame, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Draw HUD overlay
        draw_hud(display_frame, decision, detections, confidence, defect_type, fps)
        
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
