import os
from pathlib import Path
from roboflow import Roboflow

api_key = "9mViG3jN95FODMYcUdsy"
workspace = "university-of-technology-sydney-21uto"
project = "aircraft-defect-detection"
version = 3

print("[*] Downloading UTS dataset version 3...")
rf = Roboflow(api_key=api_key)
project_obj = rf.workspace(workspace).project(project)
dataset_obj = project_obj.version(version).download("yolov8")

download_path = Path(dataset_obj.location)
print(f"[OK] Downloaded to: {download_path}")

clean_images = 0
total_images = 0

for split in ["train", "valid", "test"]:
    split_dir = download_path / split
    if split_dir.exists():
        img_dir = split_dir / "images"
        lbl_dir = split_dir / "labels"
        
        if img_dir.exists():
            for img_path in img_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    total_images += 1
                    lbl_path = lbl_dir / f"{img_path.stem}.txt"
                    if not lbl_path.exists() or lbl_path.stat().st_size == 0:
                        clean_images += 1
                    else:
                        with open(lbl_path) as f:
                            content = f.read().strip()
                        if not content:
                            clean_images += 1

print(f"\nUTS Version 3 Dataset Summary:")
print(f" - Total images checked   : {total_images}")
print(f" - Clean/Normal images     : {clean_images}")
print(f" - Defective images        : {total_images - clean_images}")
