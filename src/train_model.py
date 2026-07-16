"""
train_model.py — Train the PatchCore baseline anomaly model on good parts.

This script extracts patch-level features from good images (data/good/),
builds a coreset feature memory bank, and fits a nearest-neighbors index.
"""

import os
import sys
import time
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_GOOD, MODELS_DIR, DEFAULT_IMAGE_SIZE,
    get_extractor, load_images_from_folder,
    save_model_info, resolve_device
)


def greedy_coreset(features: np.ndarray, ratio: float = 0.1, seed: int = 42) -> np.ndarray:
    """
    Optimized greedy farthest-point coreset sampling using PyTorch.
    Reduces feature memory footprint while preserving the bounding hull.
    """
    import torch
    
    np.random.seed(seed)
    n_samples, feat_dim = features.shape
    n_coreset = max(1, int(n_samples * ratio))

    # Determine execution device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    features_t = torch.from_numpy(features).to(device)

    # Initialize coreset with a random sample
    coreset_indices = [np.random.randint(0, n_samples)]
    
    # Store minimum squared distance from each point to selected coreset points
    first_feat = features_t[coreset_indices[0]]
    min_dists = torch.sum((features_t - first_feat) ** 2, dim=1)

    for _ in tqdm(range(1, n_coreset), desc="Coreset Sampling", leave=False):
        # Pick the point furthest from the current coreset
        new_idx = torch.argmax(min_dists).item()
        coreset_indices.append(new_idx)
        
        # Update minimum distances
        new_feat = features_t[new_idx]
        new_dists = torch.sum((features_t - new_feat) ** 2, dim=1)
        min_dists = torch.minimum(min_dists, new_dists)

    return features_t[coreset_indices].cpu().numpy()


def main():
    parser = argparse.ArgumentParser(description="Train Baseline PatchCore Anomaly Model")
    parser.add_argument("--data-dir", type=str, default=str(DATA_GOOD), help="Path to good images")
    parser.add_argument("--output-dir", type=str, default=str(MODELS_DIR), help="Path to save models")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize dimension")
    parser.add_argument("--coreset-ratio", type=float, default=0.1, help="Coreset sampling fraction")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps or auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[*] Initializing training. Device: {device}")
    
    # Load dataset
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"[ERROR] Data directory does not exist: {data_dir}")
        sys.exit(1)

    try:
        paths, tensors = load_images_from_folder(data_dir, args.image_size)
    except FileNotFoundError as e:
        print(f"[ERROR] {e}")
        sys.exit(1)

    print(f"[*] Loaded {len(paths)} good images from {data_dir}")

    start_time = time.time()
    extractor = get_extractor(device)

    # Extract all patches
    all_patches = []
    for tensor in tqdm(tensors, desc="Extracting features"):
        patches = extractor.extract_flat_patches(tensor)
        all_patches.append(patches)

    # Concatenate all patches into a [N, 384] array
    features = np.concatenate(all_patches, axis=0)
    print(f"[*] Total raw patches extracted: {features.shape[0]} (Feature dim: {features.shape[1]})")

    # Coreset downsampling
    print(f"[*] Reducing patch memory size (ratio: {args.coreset_ratio})")
    coreset = greedy_coreset(features, args.coreset_ratio)
    print(f"[*] Coreset shape: {coreset.shape}")

    # Fit NearestNeighbors model
    print("[*] Fitting NearestNeighbors model on coreset")
    nn_model = NearestNeighbors(n_neighbors=1, algorithm="auto", metric="minkowski", p=2)
    nn_model.fit(coreset)

    # Save models
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    np.save(output_dir / "memory_bank.npy", coreset)
    with open(output_dir / "memory_bank.pkl", "wb") as f:
        pickle.dump(nn_model, f)

    # Calculate threshold (95th percentile on train images)
    print("[*] Calibrating anomaly score threshold on training images")
    train_scores = []
    for tensor in tensors:
        patches = extractor.extract_flat_patches(tensor)
        dists, _ = nn_model.kneighbors(patches)
        # Anomaly score is 99th percentile of patch distances
        score = float(np.percentile(dists.flatten(), 99))
        train_scores.append(score)

    threshold = float(np.percentile(train_scores, 95))
    training_time = time.time() - start_time

    # Save metadata
    info = {
        "threshold": threshold,
        "image_size": args.image_size,
        "n_images": len(paths),
        "coreset_ratio": args.coreset_ratio,
        "n_patches_raw": features.shape[0],
        "n_patches_coreset": coreset.shape[0],
        "training_time_s": training_time,
        "classifier_ready": False,
        "defect_types": []
    }
    save_model_info(info, output_dir / "model_info.json")

    # Memory bank size
    pkl_size_mb = (output_dir / "memory_bank.pkl").stat().st_size / (1024 * 1024)

    # Format output box
    print("\n" + "="*50)
    print(" ZERODEFECT MODEL TRAINING COMPLETE")
    print("="*50)
    print(f" Training Images Used   : {len(paths)}")
    print(f" Raw Patches Extracted  : {features.shape[0]}")
    print(f" Coreset Patches Saved  : {coreset.shape[0]}")
    print(f" Calibrated Threshold   : {threshold:.4f}")
    print(f" Saved Model Size (PKL) : {pkl_size_mb:.2f} MB")
    print(f" Total Training Time    : {training_time:.2f} seconds")
    print("="*50)
    print("\n[EXPLANATION] Why this works without labeled defect data:")
    print("The model extracts deep patch-level features from good images only.")
    print("It builds a compact 'memory bank' representing normal variations.")
    print("A new part is scored by finding its maximum patch distance to the memory bank.")
    print("If the distance (anomaly score) exceeds the threshold, it is flagged.")
    print("\n[NEXT STEP] Run model evaluation:")
    print("  python src/evaluate_model.py")


if __name__ == "__main__":
    main()
