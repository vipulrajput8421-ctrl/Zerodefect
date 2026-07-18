import os
import sys
import shutil
import random
import argparse
import numpy as np
from pathlib import Path
from PIL import Image, ImageDraw, ImageFilter

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import ROOT, DATA_GOOD, DATA_DEFECTS, DATA_TEST

# Class mappings to standardize names
CLASS_MAPPING = {
    "paint-off": "paint-peel",
    "paint-peel-off": "paint-peel",
    "missing-head": "missing-rivet",
}

def clean_directory(directory: Path):
    """Clean directory of files, keeping structure."""
    if directory.exists():
        for item in directory.iterdir():
            if item.is_file():
                item.unlink()
            elif item.is_dir():
                shutil.rmtree(item)
    directory.mkdir(parents=True, exist_ok=True)

def draw_rivet(draw, cx, cy, r):
    """Draw a metallic 3D rivet with highlights and shadow arcs."""
    # Base circle
    draw.ellipse([cx - r, cy - r, cx + r, cy + r], fill=(165, 168, 172), outline=(130, 134, 138), width=1)
    # Highlight arc (top-left)
    draw.arc([cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1], start=180, end=270, fill=(235, 240, 245), width=2)
    # Shadow arc (bottom-right)
    draw.arc([cx - r + 1, cy - r + 1, cx + r - 1, cy + r - 1], start=0, end=90, fill=(90, 93, 98), width=2)

def generate_normal_skin_image(dx=0, dy=0, brightness=1.0):
    """Generate a clean aircraft skin panel image with rows of rivets."""
    # Base metal gray background
    bg_color = tuple(int(c * brightness) for c in (180, 183, 188))
    img = Image.new("RGB", (640, 480), color=bg_color)
    draw = ImageDraw.Draw(img)
    
    # Grid of rivets
    rivet_radius = 8
    cols = [120, 320, 520]
    rows = [80, 240, 400]
    
    for cx in cols:
        for cy in rows:
            draw_rivet(draw, cx + dx, cy + dy, rivet_radius)
            
    # Add subtle pixel noise to simulate metal texture
    img_arr = np.array(img).astype(np.float32)
    noise = np.random.normal(0, 2.0, img_arr.shape)
    img_arr = np.clip(img_arr + noise, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(img_arr)
    # Apply a light blur
    img = img.filter(ImageFilter.GaussianBlur(0.3))
    return img

def main():
    parser = argparse.ArgumentParser(description="Aerospace Dataset Downloader & Organizer")
    parser.add_argument("--api-key", type=str, default=os.getenv("ROBOFLOW_API_KEY"),
                        help="Roboflow API Key (falls back to ROBOFLOW_API_KEY env var)")
    parser.add_argument("--dataset", type=str, 
                        default="prath0029/Zerodefect-1.0-bucket",
                        help="Roboflow dataset identifier: workspace/project/version")
    parser.add_argument("--max-train-good", type=int, default=120,
                        help="Maximum normal/good images to generate for training baseline")
    parser.add_argument("--few-shot-count", type=int, default=15,
                        help="Images per defect class for few-shot classifier training")
    parser.add_argument("--test-count-per-class", type=int, default=10,
                        help="Images per class (good/defects) for testing")
    args = parser.parse_args()

    api_key = args.api_key
    if not api_key:
        print("[ERROR] Roboflow API key is required.")
        print("Please provide it via --api-key or set the ROBOFLOW_API_KEY environment variable.")
        sys.exit(1)

    # Parse dataset workspace, project, and version
    slug_parts = [p for p in args.dataset.split("/") if p]
    if len(slug_parts) < 2:
        print(f"[ERROR] Invalid dataset format: '{args.dataset}'. Expected 'workspace/project' or 'workspace/project/version'")
        sys.exit(1)
    
    workspace = slug_parts[0]
    project = slug_parts[1]
    version = int(slug_parts[2]) if len(slug_parts) > 2 else 1

    # Clean existing directories to avoid mixing datasets
    print("[*] Preparing workspace directories...")
    clean_directory(DATA_GOOD)
    clean_directory(DATA_DEFECTS)
    clean_directory(DATA_TEST)

    # Initialize Roboflow and download
    print(f"[*] Authenticating with Roboflow and downloading {workspace}/{project} (v{version}) in folder format...")
    try:
        from roboflow import Roboflow
    except ImportError:
        print("[ERROR] roboflow package not installed. Run: pip install roboflow")
        sys.exit(1)

    rf = Roboflow(api_key=api_key)
    project_obj = rf.workspace(workspace).project(project)
    dataset_obj = project_obj.version(version).download("folder")
    
    download_path = Path(dataset_obj.location)
    print(f"[OK] Dataset downloaded successfully to: {download_path}")

    # Gather all defect images from download folder
    defect_pools = {}
    
    splits = ["train", "valid", "test"]
    for split in splits:
        split_dir = download_path / split
        if not split_dir.exists():
            continue
        
        for class_dir in split_dir.iterdir():
            if not class_dir.is_dir():
                continue
            
            raw_name = class_dir.name
            std_name = CLASS_MAPPING.get(raw_name.lower(), raw_name.lower())
            
            if std_name not in defect_pools:
                defect_pools[std_name] = []
                
            for img_path in class_dir.iterdir():
                if img_path.suffix.lower() in (".jpg", ".jpeg", ".png"):
                    defect_pools[std_name].append(img_path)

    # Distribute Defect Images
    test_defect_imgs = []
    
    for std_name, paths in defect_pools.items():
        # Create output defect dir
        std_defect_dir = DATA_DEFECTS / std_name
        std_defect_dir.mkdir(parents=True, exist_ok=True)
        
        # Shuffle/take first N for training
        random.seed(42)
        random.shuffle(paths)
        
        num_train_defect = min(args.few_shot_count, len(paths) // 2)
        num_test_defect = min(args.test_count_per_class, len(paths) - num_train_defect)
        
        train_paths = paths[:num_train_defect]
        test_paths = paths[num_train_defect:num_train_defect + num_test_defect]
        
        print(f"[*] Copying {len(train_paths)} few-shot '{std_name}' images to {DATA_DEFECTS}/{std_name}...")
        for idx, img_path in enumerate(train_paths):
            shutil.copy(img_path, std_defect_dir / f"{std_name}_{idx+1:04d}{img_path.suffix}")
            
        test_defect_imgs.extend(test_paths)

    # Generate Synthetic Good (Normal Skin) Images
    print(f"\n[*] Generating {args.max_train_good} synthetic good/normal aircraft skin panels...")
    for idx in range(args.max_train_good):
        dx = random.randint(-15, 15)
        dy = random.randint(-15, 15)
        brightness = random.uniform(0.92, 1.08)
        img = generate_normal_skin_image(dx, dy, brightness)
        img.save(DATA_GOOD / f"good_{idx+1:04d}.jpg")

    # Generate Test Set & Labels CSV
    test_rows = []
    print(f"[*] Copying/generating test images to {DATA_TEST}...")
    
    # 1. Copy test defect images
    for idx, img_path in enumerate(test_defect_imgs):
        filename = f"test_defect_{idx+1:04d}{img_path.suffix}"
        shutil.copy(img_path, DATA_TEST / filename)
        test_rows.append(f"{filename},defect")
        
    # 2. Generate test good images
    num_test_good = args.test_count_per_class * len(defect_pools)  # Keep balanced test set
    for idx in range(num_test_good):
        dx = random.randint(-20, 20)
        dy = random.randint(-20, 20)
        brightness = random.uniform(0.90, 1.10)
        img = generate_normal_skin_image(dx, dy, brightness)
        filename = f"test_good_{idx+1:04d}.jpg"
        img.save(DATA_TEST / filename)
        test_rows.append(f"{filename},good")

    # Write labels CSV
    labels_csv_path = ROOT / "data" / "test_labels.csv"
    with open(labels_csv_path, "w") as f:
        for row in test_rows:
            f.write(row + "\n")

    print(f"\n[OK] Aerospace dataset successfully set up!")
    print(f"  - Good training baseline (synthetic): {len(list(DATA_GOOD.glob('*.jpg')))}")
    print(f"  - Defect few-shot folders (Roboflow):")
    for d in DATA_DEFECTS.iterdir():
        if d.is_dir():
            count = len(list(d.glob("*.jpg"))) + len(list(d.glob("*.png")))
            print(f"    * {d.name:<12} : {count}")
    print(f"  - Test validation images          : {len(list(DATA_TEST.glob('*.jpg')))}")
    print(f"  - Labels CSV saved to             : {labels_csv_path}")

if __name__ == "__main__":
    main()
