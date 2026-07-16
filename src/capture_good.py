"""
capture_good.py — OpenCV webcam tool to capture 100-150 good component photos.

Usage:
  python src/capture_good.py --target-count 120

Controls:
  SPACE - Save current frame to data/good/
  Q     - Exit tool
"""

import os
import sys
import argparse
from pathlib import Path
import cv2

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import DATA_GOOD


def main():
    parser = argparse.ArgumentParser(description="Webcam Good-Part Image Capturing Utility")
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam device ID (default 0)")
    parser.add_argument("--output-dir", type=str, default=str(DATA_GOOD), help="Folder to save images")
    parser.add_argument("--target-count", type=int, default=120, help="Target count of good photos")
    parser.add_argument("--width", type=int, default=1280, help="Webcam capture width")
    parser.add_argument("--height", type=int, default=960, help="Webcam capture height")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Resolve next filename index
    existing_files = list(output_dir.glob("good_*.jpg"))
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
        print("Please check your camera connections or try a different ID (e.g., --camera-id 1).")
        sys.exit(1)

    # Set frame properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)

    # Read dimensions back
    cap_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cap_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    print(f"[*] Camera opened. Resolution: {cap_w}x{cap_h}")
    print(f"[*] Saving to: {output_dir}")
    print("[*] Controls: PRESS [SPACE] to capture | [Q] to quit")

    count = len(list(output_dir.glob("good_*.jpg")))
    blink_timer = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from camera.")
            break

        preview = frame.copy()
        
        # Determine status colors based on targets
        if count >= args.target_count:
            status_color = (0, 255, 0)      # Green
        elif count >= 100:
            status_color = (0, 255, 255)    # Yellow/Amber
        else:
            status_color = (255, 255, 255)  # White

        # Draw overlays
        cv2.rectangle(preview, (10, 10), (450, 95), (0, 0, 0), -1)
        cv2.putText(preview, f"ZeroDefect - Good Part Capture", (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 217, 245), 2)
        cv2.putText(preview, f"Captured: {count} / {args.target_count}", (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.65, status_color, 2)
        cv2.putText(preview, "Press SPACE to save | Q to quit", (20, 85), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1)

        # Blink tip message every 20 frames
        blink_timer = (blink_timer + 1) % 40
        if count > 0 and count % 20 == 0 and blink_timer < 20:
            cv2.rectangle(preview, (10, cap_h - 45), (cap_w - 10, cap_h - 10), (0, 0, 120), -1)
            cv2.putText(preview, "TIP: Vary lighting angle, rotation, and distance slightly!", (30, cap_h - 22), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

        cv2.imshow("ZeroDefect - Good Part Capture", preview)
        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            filename = f"good_{next_idx:04d}.jpg"
            file_path = output_dir / filename
            cv2.imwrite(str(file_path), frame)
            
            print(f"[{next_idx:03d}] Saved: {file_path.relative_to(output_dir.parent.parent)}")
            count += 1
            next_idx += 1
            
            # Print feedback milestones
            if count == 100:
                print("\n[MILESTONE] Great! You have captured 100 images — minimum range reached.")
                print(f"Target count is {args.target_count}. Keep going!\n")
            elif count == args.target_count:
                print(f"\n[MILESTONE] Target reached! You have captured {args.target_count} images.")
                print("Press Q to exit capture and proceed to training.\n")
                
            if count % 20 == 0:
                print(">>> Tip: Vary the component's angle, rotation, and lighting slightly to improve model robustness.")

        elif key == ord('q') or key == ord('Q'):
            break

    cap.release()
    cv2.destroyAllWindows()

    print("\n" + "="*50)
    print(" CAPTURE SESSION SUMMARY")
    print("="*50)
    print(f" Output Folder       : {output_dir}")
    print(f" Total Images now    : {count}")
    print(f" Minimum Required    : 100")
    print(f" Target Goal         : {args.target_count}")
    print("="*50)


if __name__ == "__main__":
    main()
