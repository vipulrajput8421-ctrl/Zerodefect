"""
quantize_model.py — Quantize and export the model features for edge microcontrollers.

Handles:
  1. PyTorch model feature extractor export to ONNX
  2. Subsampling coreset patch features to fit ESP32 memory budget
  3. TF/TFLite INT8 quantization guidelines

HARDWARE_VALIDATION_REQUIRED:
Flashing converted binaries and loading them on actual microcontrollers needs testing.
"""

import os
import sys
import argparse
import json
import pickle
from pathlib import Path
import numpy as np
import torch

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from utils import MODELS_DIR, DEFAULT_IMAGE_SIZE, get_extractor


def main():
    parser = argparse.ArgumentParser(description="ZeroDefect Quantization and Microcontroller Export Utility")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR), help="Folder with models")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR.parent / "esp32" / "exported"), help="Output path")
    parser.add_argument("--export-onnx", action="store_true", help="Export extractor to ONNX format")
    parser.add_argument("--export-tflite", action="store_true", help="Attempt conversion to TFLite INT8")
    parser.add_argument("--report", action="store_true", help="Analyze memory budget sizes for ESP32")
    parser.add_argument("--max-coreset", type=int, default=1000, help="Subsampled patch memory bank limit")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize dimension")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    npy_path = model_dir / "memory_bank.npy"
    pkl_path = model_dir / "memory_bank.pkl"

    if args.export_onnx:
        print("[*] Exporting Feature Extractor backbone to ONNX...")
        extractor = get_extractor("cpu")
        extractor.eval()
        
        # Fake input frame: [B, C, H, W]
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
        print(f"[SUCCESS] Backbone exported to: {onnx_output}")
        print("    Input shape  : [1, 3, 256, 256]")
        print("    Output shape : [1, 384, 32, 32]")

    if args.export_tflite:
        print("[*] Converting ONNX to TensorFlow Lite...")
        try:
            import tensorflow as tf
            print(f"    TensorFlow Version: {tf.__version__}")
            print("    [NOTE] Full TF quantization requires onnx-tf utility or tf-keras:")
            print("           1. pip install onnx-tf tensorflow")
            print("           2. Convert to TF SavedModel, then load with TFLiteConverter")
            print("    [HARDWARE_VALIDATION_REQUIRED] Quantization output verification on ESP32.")
        except ImportError:
            print("    [ERROR] TensorFlow is required for direct converter export.")
            print("            Run: pip install tensorflow")

    if args.report or (not args.export_onnx and not args.export_tflite):
        if not npy_path.exists():
            print(f"[WARN] No memory_bank.npy found at {npy_path}. Run training first.")
            sys.exit(1)
        
        coreset = np.load(npy_path)
        n_patches, feat_dim = coreset.shape
        print("\n" + "="*50)
        print(" MEMORY BANK ANALYSIS FOR ESP32")
        print("="*50)
        print(f" Current Coreset Patches : {n_patches} (Dim: {feat_dim})")
        
        # Calculate sizes in memory
        fp32_size = (n_patches * feat_dim * 4) / 1024
        fp16_size = (n_patches * feat_dim * 2) / 1024
        int8_size = (n_patches * feat_dim * 1) / 1024
        
        print(f" Size in RAM (FP32)      : {fp32_size:.2f} KB")
        print(f" Size in RAM (FP16)      : {fp16_size:.2f} KB")
        print(f" Size in RAM (INT8)      : {int8_size:.2f} KB")
        print("-" * 50)
        print(" ESP32-CAM (AI-Thinker) Memory Constraints:")
        print("  - Internal SRAM        : ~520 KB")
        print("  - External PSRAM       : 4 MB or 8 MB")
        print("-" * 50)

        if int8_size > 400:
            print(f"[!] Warning: Current memory bank is too large for internal RAM.")
            print(f"    Must use PSRAM allocations or subsample the coreset.")
        else:
            print("[✓] Memory bank fits safely in standard ESP32 SRAM (INT8).")

        # Create a compressed microcontroller subsampled bank if coreset is too big
        if n_patches > args.max_coreset:
            print(f"\n[*] Subsampling memory bank to max {args.max_coreset} patches...")
            indices = np.random.choice(n_patches, args.max_coreset, replace=False)
            reduced_coreset = coreset[indices]
            reduced_path = out_dir / "memory_bank_micro.npy"
            np.save(reduced_path, reduced_coreset)
            print(f"[SUCCESS] Saved reduced micro-bank to: {reduced_path}")
            print(f"          New Size (INT8): {(args.max_coreset * feat_dim)/1024:.2f} KB")
            print("          [HARDWARE_VALIDATION_REQUIRED] Verify micro-bank classification rate on board.")
        else:
            print(f"\n[*] Current bank size ({n_patches}) is under limit ({args.max_coreset}). No subsampling needed.")
        print("="*50)


if __name__ == "__main__":
    main()
