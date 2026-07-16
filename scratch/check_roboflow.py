import os
from pathlib import Path
from roboflow import Roboflow

api_key = "9mViG3jN95FODMYcUdsy"
workspace = "dibya-dillip"
project = "aircraft-skin-defects-classification-new-dataset"
version = 2

print("[*] Downloading dataset in folder format...")
rf = Roboflow(api_key=api_key)
project_obj = rf.workspace(workspace).project(project)
dataset_obj = project_obj.version(version).download("folder")

download_path = Path(dataset_obj.location)
print(f"[OK] Downloaded to: {download_path}")

print("\nListing root contents:")
for item in download_path.iterdir():
    print(f" - {item.name} ({'dir' if item.is_dir() else 'file'})")

for split in ["train", "valid", "test"]:
    split_dir = download_path / split
    if split_dir.exists():
        print(f"\nListing contents of {split}:")
        for item in split_dir.iterdir():
            if item.is_dir():
                num_files = len(list(item.glob("*")))
                print(f"  - Folder '{item.name}': {num_files} files")
