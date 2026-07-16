"""
capture_defects.py — OpenCV webcam tool to capture 5-20 defect examples per type.

Usage:
  python src/capture_defects.py --defect-type crack

Controls:
  SPACE - Save current frame to data/defects/<type>/
  Q     - Exit tool
"""

import os
import sys
import argparse
import re
from pathlib import Path
import cv2

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_DEFECTS, get_defect_types


def main():
    parser = argparse.ArgumentParser(description="Webcam Defect Image Capturing Utility")
    parser.add_argument("--defect-type", type=str, required=True, help="Type of defect (e.g. crack, scratch, dent)")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default 0)")
    parser.add_argument("--output-dir", type=str, default=str(DATA_DEFECTS), help="Base folder for defects")
    parser.add_argument("--max-count", type=int, default=20, help="Target max count for defect images")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=960, help="Webcam capture height")
    args = parser.parse_args()

    # Validate defect type name format
    defect_type = args.defect_type.lower().strip()
    if not re.match(r"^[a-zA-Z0-9_-]+$", defect_type):
        print("[ERROR] Defect type name must contain only alphanumeric characters, underscores, or hyphens.")
        sys.exit(1)

    defect_dir = Path(args.output_dir) / defect_type
    defect_dir.mkdir(parents=True, exist_ok=True)

    # Resolve next filename index
    existing_files = list(defect_dir.glob(f"{defect_type}_*.jpg"))
    if existing_files:
        indices = []
        for f in existing_files:
            try:
                indices.append(int(f.stem.split("_")[1]))
            except ValueError:
                pass
        next_idx = max(indices) + 1 if indices else 1
    else:
        next_idx = 1

    # Open webcam
    cap = cv2.VideoCapture(args.camera_id)
    if not cap.isOpened():
        print(f"[ERROR] Could not open camera with device ID: {args.camera_id}")
        sys.exit(1)

    # Set frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read dimensions back
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Camera opened for defect class: {defect_type.upper()}")
    print(f"[*] Saving to: {defect_dir}")
    print("[*] Controls: PRESS [SPACE] to capture | [Q] to quit")

    count = len(list(defect_dir.glob(f"{defect_type}_*.jpg")))

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        preview = frame.copy()
        
        # Red styling for defect mode
        status_color = (0, 0, 255) if count < 5 else (0, 255, 0)

        # Draw overlays
        cv2.rectangle(preview, (10, 10), (450, 95), (0, 0, 0), -1)
        cv2.putText(preview, f"ZeroDefect - DEFECT Capture", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        cv2.putText(preview, f"Defect Type: {defect_type.upper()}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(preview, f"Captured: {count} / {args.max_count} (Goal: 5-20)", (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 2)

        cv2.imshow("ZeroDefect - Defect Capture", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if count >= args.max_count:
                print(f"[WARN] Target max count ({args.max_count}) reached. Press Q to exit or continue manually.")
            
            filename = f"{defect_type}_{next_idx:04d}.jpg"
            file_path = defect_dir / filename
            cv2.imwrite(str(file_path), frame)
            
            print(f"[{next_idx:03d}] Saved: {file_path.relative_to(defect_dir.parent.parent.parent)}")
            count += 1
            next_idx += 1

            if count < 5:
                print(f"Captured {count} images. Aim for at least 5 to train the few-shot classifier.")
            elif count == 5:
                print("Minimum threshold of 5 images hit. You can now use these for calibration.")

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    # Print final summary of all defect folders
    print("\n" + "="*50)
    print(" ALL DEFECT CATEGORIES STATUS")
    print("="*50)
    base_dir = Path(args.output_dir)
    if base_dir.exists():
        for d in base_dir.iterdir():
            if d.is_dir() and not d.name.startswith("."):
                n_imgs = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
                marker = " [OK]" if 5 <= n_imgs <= 20 else " [Need 5-20]"
                print(f" - {d.name:<15} : {n_imgs} images {marker}")
    print("="*50)


if __name__ == "__main__":
    main()

ZeroDefect


