"""
environment_check.py — Verify the environment setup.

Checks python version, libraries, hardware support (CUDA/MPS),
and project directory structure. Exits with 0 if clean, 1 if missing critical components.
"""

import os
import sys
from pathlib import Path


def main():
    print("="*60)
    print(" ZERODEFECT — ENVIRONMENT CHECKER")
    print("="*60)

    # 1. Check Python Version
    py_major = sys.version_info.major
    py_minor = sys.version_info.minor
    py_ver_str = f"{py_major}.{py_minor}.{sys.version_info.micro}"
    if py_major == 3 and py_minor >= 10:
        print(f"[OK] Python Version      : {py_ver_str} (OK, >= 3.10 required)")
        py_ok = True
    else:
        print(f"[FAIL] Python Version    : {py_ver_str} (FAIL, require Python >= 3.10)")
        py_ok = False

    # 2. Check Package Imports
    packages = [
        # (module_name, import_name, required)
        ("torch", "torch", True),
        ("torchvision", "torchvision", True),
        ("opencv-python", "cv2", True),
        ("numpy", "numpy", True),
        ("pillow", "PIL", True),
        ("scikit-learn", "sklearn", True),
        ("scipy", "scipy", True),
        ("fastapi", "fastapi", False),
        ("uvicorn", "uvicorn", False),
        ("tqdm", "tqdm", False),
        ("pandas", "pandas", False),
        ("onnx", "onnx", False),
        ("onnxruntime", "onnxruntime", True),
    ]

    print("\nPackage dependencies:")
    print("-" * 50)
    
    missing_critical = False
    pkg_checked = 0
    pkg_ok = 0

    for name, imp_name, required in packages:
        pkg_checked += 1
        req_label = "Required" if required else "Optional"
        try:
            mod = __import__(imp_name)
            ver = getattr(mod, "__version__", "loaded")
            print(f"  [OK] {name:<18} : {ver:<15} ({req_label})")
            pkg_ok += 1
        except ImportError:
            if required:
                print(f"  [FAIL] {name:<16} : MISSING         ({req_label}) - Run: pip install {name}")
                missing_critical = True
            else:
                print(f"  [WARN] {name:<16} : MISSING         ({req_label}) - Optional")

    print("-" * 50)

    # 3. Check Hardware Acceleration
    print("\nHardware Support:")
    print("-" * 50)
    try:
        import torch
        # CUDA
        cuda_avail = torch.cuda.is_available()
        if cuda_avail:
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  CUDA GPU support   : Available (GPU: {gpu_name})")
        else:
            print(f"  CUDA GPU support   : Not Available (Using CPU)")

        # Apple Silicon MPS
        mps_avail = False
        if hasattr(torch.backends, "mps"):
            mps_avail = torch.backends.mps.is_available()
        if mps_avail:
            print(f"  MPS Apple support  : Available")
        
        # Thread count
        threads = torch.get_num_threads()
        print(f"  CPU Threads        : {threads}")
        
        device_rec = "cuda" if cuda_avail else ("mps" if mps_avail else "cpu")
        print(f"  Recommended Device : {device_rec}")
    except Exception as e:
        print(f"  GPU support check failed: {e}")
        device_rec = "cpu"
    print("-" * 50)

    # 4. Check Project Folders
    print("\nDirectory Scaffolding:")
    print("-" * 50)
    root = Path(__file__).resolve().parent.parent
    folders = [
        "data/good",
        "data/defects",
        "models",
        "logs",
        "logs/thumbnails"
    ]
    all_folders_exist = True
    for f in folders:
        fpath = root / f
        if fpath.exists():
            print(f"  [OK] {f:<18} : Exists")
        else:
            print(f"  [MISSING] {f:<13} : MISSING")
            all_folders_exist = False
    print("-" * 50)

    # Final summary verdict
    print("\n" + "="*60)
    print(" VERDICT SUMMARY")
    print("="*60)
    print(f" Packages OK : {pkg_ok} / {pkg_checked}")
    
    if not py_ok:
        print("[STATUS] FAILED. Upgrade your Python version to >= 3.10.")
        sys.exit(1)
    elif missing_critical:
        print("[STATUS] FAILED. Install the missing required packages first:")
        print("  pip install -r requirements.txt")
        sys.exit(1)
    elif not all_folders_exist:
        print("[STATUS] WARNING. Some directories are missing. Creating them now...")
        for f in folders:
            (root / f).mkdir(parents=True, exist_ok=True)
        print("[STATUS] Directories created successfully. Ready!")
        sys.exit(0)
    else:
        print(f"[STATUS] SUCCESS! Everything is correctly configured on device '{device_rec}'.")
        print("You are ready to run: run_dashboard.bat or run_demo.bat")
        sys.exit(0)


if __name__ == "__main__":
    main()
