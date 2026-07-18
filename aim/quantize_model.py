"""
quantize_model.py — Analyze and export model assets for edge microcontrollers.

Handles:
  1. YOLO ONNX model analysis and size reporting
  2. Legacy PyTorch feature extractor export to ONNX
  3. Subsampling coreset patch features to fit AIM memory budget
  4. RKNN model info for Luckfox Pico Max (RV1106)

HARDWARE_VALIDATION_REQUIRED:
Flashing converted binaries and loading them on actual microcontrollers needs testing.
"""

import os
import sys
import argparse
import json
from pathlib import Path
import numpy as np

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import (
    MODELS_DIR, DEFAULT_IMAGE_SIZE,
    YOLO_MODEL_PATH, YOLO_PT_PATH, YOLO_RKNN_PATH, YOLO_CLASSES
)


def analyze_yolo_model(model_path: Path):
    """Analyze the YOLOv5n ONNX model for edge deployment."""
    print("\n" + "="*55)
    print(" YOLO ONNX MODEL ANALYSIS FOR EDGE DEPLOYMENT")
    print("="*55)

    size_mb = model_path.stat().st_size / (1024 * 1024)
    print(f" Model File       : {model_path.name}")
    print(f" File Size        : {size_mb:.2f} MB")
    print(f" Format           : ONNX (opset 12)")
    print(f" Architecture     : YOLOv5 Nano")
    print(f" Input Size       : 640×640×3")
    print(f" Num Classes      : {len(YOLO_CLASSES)}")
    print(f" Classes          : {', '.join(YOLO_CLASSES)}")
    print("-" * 55)

    # Try to get detailed info from onnxruntime
    try:
        import onnxruntime as ort
        session = ort.InferenceSession(str(model_path))
        inputs = session.get_inputs()
        outputs = session.get_outputs()
        print(f" Input Name       : {inputs[0].name}")
        print(f" Input Shape      : {inputs[0].shape}")
        print(f" Input Type       : {inputs[0].type}")
        print(f" Output Name      : {outputs[0].name}")
        print(f" Output Shape     : {outputs[0].shape}")
    except Exception as e:
        print(f" [WARN] Could not read ONNX metadata: {e}")

    print("-" * 55)

    # Check companion files
    pt_path = model_path.parent / "best.pt"
    rknn_path = model_path.parent / "best.rknn"

    if pt_path.exists():
        pt_size = pt_path.stat().st_size / (1024 * 1024)
        print(f" PyTorch Weights  : {pt_path.name} ({pt_size:.2f} MB)")
    else:
        print(f" PyTorch Weights  : Not found")

    if rknn_path.exists():
        rknn_size = rknn_path.stat().st_size / (1024 * 1024)
        print(f" RKNN Model       : {rknn_path.name} ({rknn_size:.2f} MB) — INT8 quantized")
    else:
        print(f" RKNN Model       : Not found")

    print("-" * 55)
    print(" TARGET HARDWARE COMPATIBILITY:")
    print("  - AIM Device (ESP32-CAM compatible)")
    print(f"    ONNX ({size_mb:.1f} MB) exceeds AIM PSRAM (4 MB).")
    print("    Requires further quantization or use RKNN path instead.")
    print("  - Luckfox Pico Max (RV1106 NPU)")
    if rknn_path.exists():
        print(f"    RKNN ({rknn_size:.1f} MB) fits in flash. Use rknn_model_zoo for inference.")
        print("    Push: adb push best.rknn /userdata/")
        print("    Run:  ./rknn_yolov5_demo /userdata/best.rknn test.jpg")
    else:
        print("    Convert ONNX to RKNN using rknn-toolkit2.")
    print("="*55)


def analyze_legacy_memory_bank(npy_path: Path, max_coreset: int, out_dir: Path):
    """Analyze legacy PatchCore memory bank for AIM deployment."""
    if not npy_path.exists():
        print(f"[WARN] No memory_bank.npy found at {npy_path}. Legacy model not trained.")
        return

    coreset = np.load(npy_path)
    n_patches, feat_dim = coreset.shape
    print("\n" + "="*50)
    print(" LEGACY MEMORY BANK ANALYSIS FOR AIM")
    print("="*50)
    print(f" Current Coreset Patches : {n_patches} (Dim: {feat_dim})")

    fp32_size = (n_patches * feat_dim * 4) / 1024
    fp16_size = (n_patches * feat_dim * 2) / 1024
    int8_size = (n_patches * feat_dim * 1) / 1024

    print(f" Size in RAM (FP32)      : {fp32_size:.2f} KB")
    print(f" Size in RAM (FP16)      : {fp16_size:.2f} KB")
    print(f" Size in RAM (INT8)      : {int8_size:.2f} KB")
    print("-" * 50)
    print(" AIM Device Memory Constraints:")
    print("  - Internal SRAM        : ~520 KB")
    print("  - External PSRAM       : 4 MB or 8 MB")
    print("-" * 50)

    if int8_size > 400:
        print(f"[!] Warning: Current memory bank is too large for internal RAM.")
        print(f"    Must use PSRAM allocations or subsample the coreset.")
    else:
        print("[✓] Memory bank fits safely in standard AIM SRAM (INT8).")

    if n_patches > max_coreset:
        print(f"\n[*] Subsampling memory bank to max {max_coreset} patches...")
        indices = np.random.choice(n_patches, max_coreset, replace=False)
        reduced_coreset = coreset[indices]
        reduced_path = out_dir / "memory_bank_micro.npy"
        np.save(reduced_path, reduced_coreset)
        print(f"[SUCCESS] Saved reduced micro-bank to: {reduced_path}")
        print(f"          New Size (INT8): {(max_coreset * feat_dim)/1024:.2f} KB")
        print("          [HARDWARE_VALIDATION_REQUIRED] Verify micro-bank classification rate on board.")
    else:
        print(f"\n[*] Current bank size ({n_patches}) is under limit ({max_coreset}). No subsampling needed.")
    print("="*50)


def main():
    parser = argparse.ArgumentParser(description="ZeroDefect Model Analysis & Edge Export Utility")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR), help="Folder with models")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR.parent / "aim" / "exported"), help="Output path")
    parser.add_argument("--analyze-yolo", action="store_true", default=True, help="Analyze YOLO ONNX model")
    parser.add_argument("--analyze-legacy", action="store_true", help="Analyze legacy PatchCore memory bank")
    parser.add_argument("--export-onnx", action="store_true", help="Export legacy extractor to ONNX format")
    parser.add_argument("--report", action="store_true", help="Run all analyses")
    parser.add_argument("--max-coreset", type=int, default=1000, help="Subsampled patch memory bank limit")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize dimension")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # YOLO Model Analysis (default)
    yolo_path = model_dir / "best.onnx"
    if args.analyze_yolo or args.report:
        if yolo_path.exists():
            analyze_yolo_model(yolo_path)
        else:
            print(f"[WARN] YOLO ONNX model not found at {yolo_path}")
            print("       Download: hf sync hf://buckets/prath0029/Zerodefect-1.0-bucket ./local")

    # Legacy PatchCore Analysis
    if args.analyze_legacy or args.report:
        npy_path = model_dir / "memory_bank.npy"
        analyze_legacy_memory_bank(npy_path, args.max_coreset, out_dir)

    # Legacy ONNX export
    if args.export_onnx:
        print("[*] Exporting Legacy Feature Extractor backbone to ONNX...")
        try:
            import torch
            from utils import get_extractor
            extractor = get_extractor("cpu")
            extractor.eval()

            dummy_input = torch.randn(1, 3, args.image_size, args.image_size)
            onnx_output = out_dir / "feature_extractor.onnx"

            torch.onnx.export(
                extractor,
                dummy_input,
                str(onnx_output),
                export_params=True,
                opset_version=11,
                do_constant_folding=True,
                input_names=["input"],
                output_names=["output"]
            )
            print(f"[SUCCESS] Legacy backbone exported to: {onnx_output}")
        except Exception as e:
            print(f"[ERROR] Legacy ONNX export failed: {e}")


if __name__ == "__main__":
    main()
