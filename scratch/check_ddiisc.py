import os
from pathlib import Path
from roboflow import Roboflow

api_key = "9mViG3jN95FODMYcUdsy"
workspace = "sutd-4mhea"
project = "aircraft-ai-dataset"

rf = Roboflow(api_key=api_key)
project_obj = rf.workspace(workspace).project(project)

print("Project Type:", project_obj.type)
versions = project_obj.versions()
for v in versions:
    print(f" - Version {v.version} ({v.name}) splits: {v.splits}")

# Let's download the first version
version = int(versions[0].version)
print(f"[*] Downloading Version {version}...")
dataset_obj = project_obj.version(version).download("yolov8")
download_path = Path(dataset_obj.location)

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

print(f"\nSUTD Dataset Summary:")
print(f" - Total images checked   : {total_images}")
print(f" - Clean/Normal images     : {clean_images}")
print(f" - Defective images        : {total_images - clean_images}")
