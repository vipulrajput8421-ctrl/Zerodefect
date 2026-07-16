"""
evaluate_model.py — Validate baseline PatchCore anomaly scores against test parts.

Computes anomaly scores for images in a test directory and prints a sorted summary.
Can evaluate performance metrics if a ground-truth labels CSV is provided.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_TEST, MODELS_DIR, DEFAULT_IMAGE_SIZE,
    load_model_info, get_extractor, load_image, resolve_device
)


def score_image(image_path: Path, nn_model, extractor, image_size: int) -> float:
    """Return the anomaly score (99th percentile patch distance) for one image."""
    tensor = load_image(image_path, image_size)
    patches = extractor.extract_flat_patches(tensor)  # [N, 384]
    dists, _ = nn_model.kneighbors(patches)           # [N, 1]
    return float(np.percentile(dists.flatten(), 99))


def main():
    parser = argparse.ArgumentParser(description="Evaluate Anomaly Model")
    parser.add_argument("--test-dir", type=str, default=str(DATA_TEST), help="Folder with test images")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR), help="Folder with model files")
    parser.add_argument("--labels", type=str, default=None, help="Path to ground truth labels CSV (filename,label)")
    parser.add_argument("--threshold", type=float, default=None, help="Override anomaly threshold")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize dimension")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps or auto")
    args = parser.parse_args()

    model_dir = Path(args.model_dir)
    model_info_path = model_dir / "model_info.json"
    memory_bank_path = model_dir / "memory_bank.pkl"

    if not memory_bank_path.exists():
        print(f"[ERROR] Model file not found: {memory_bank_path}")
        print("Please train the model first: python src/train_model.py")
        sys.exit(1)

    # Load model files
    with open(memory_bank_path, "rb") as f:
        nn_model = pickle.load(f)

    # Load metadata
    try:
        model_info = load_model_info(model_info_path)
        threshold = model_info.get("threshold", 1.0)
    except FileNotFoundError:
        threshold = 1.0
        print("[WARN] model_info.json not found. Using default threshold 1.0")

    if args.threshold is not None:
        threshold = args.threshold

    device = resolve_device(args.device)
    extractor = get_extractor(device)

    # Load test images
    test_dir = Path(args.test_dir)
    if not test_dir.exists():
        print(f"[WARN] Test directory not found: {test_dir}. Creating empty one.")
        test_dir.mkdir(parents=True, exist_ok=True)

    extensions = (".jpg", ".jpeg", ".png")
    test_paths = sorted([p for p in test_dir.iterdir() if p.suffix.lower() in extensions])

    if not test_paths:
        print(f"[WARN] No test images found in {test_dir}.")
        print("Place test images in data/test/ to evaluate your model.")
        sys.exit(0)

    print(f"[*] Evaluating {len(test_paths)} test images against threshold: {threshold:.4f}")

    results = []
    for path in test_paths:
        score = score_image(path, nn_model, extractor, args.image_size)
        decision = "DEFECT" if score > threshold else "OK"
        results.append((path.name, score, decision))

    # Print results table
    print("\n" + "="*70)
    print(f" {'FILENAME':<35} | {'ANOMALY SCORE':<15} | {'DECISION':<10}")
    print("-"*70)
    for name, score, decision in sorted(results, key=lambda x: x[1], reverse=True):
        marker = " [*]" if decision == "DEFECT" else ""
        print(f" {name:<35} | {score:15.4f} | {decision:<10}{marker}")
    print("="*70)

    # Check for labels file to run classification metrics
    if args.labels:
        labels_path = Path(args.labels)
        if not labels_path.exists():
            print(f"[ERROR] Ground truth labels file not found: {labels_path}")
            sys.exit(1)

        import csv
        # Load CSV mapping filename -> label (good or defect)
        ground_truth = {}
        with open(labels_path, newline="") as f:
            reader = csv.reader(f)
            for row in reader:
                if len(row) >= 2:
                    ground_truth[row[0].strip()] = row[1].strip().lower()

        y_true = []
        y_pred = []
        scores = []

        for name, score, decision in results:
            if name in ground_truth:
                true_label = 1 if ground_truth[name] == "defect" else 0
                pred_label = 1 if decision == "DEFECT" else 0
                y_true.append(true_label)
                y_pred.append(pred_label)
                scores.append(score)

        if len(y_true) > 0:
            y_true = np.array(y_true)
            y_pred = np.array(y_pred)
            scores = np.array(scores)

            tp = np.sum((y_true == 1) & (y_pred == 1))
            tn = np.sum((y_true == 0) & (y_pred == 0))
            fp = np.sum((y_true == 0) & (y_pred == 1))
            fn = np.sum((y_true == 1) & (y_pred == 0))

            accuracy = (tp + tn) / len(y_true)
            precision = tp / (tp + fp) if (tp + fp) > 0 else 0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0
            f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

            print("\n" + "="*50)
            print(" PERFORMANCE METRICS AGAINST LABELS")
            print("="*50)
            print(f" Accuracy  : {accuracy * 100:.2f}%")
            print(f" Precision : {precision * 100:.2f}%")
            print(f" Recall    : {recall * 100:.2f}%")
            print(f" F1-Score  : {f1 * 100:.2f}%")
            print(f" True Positives: {tp} | True Negatives: {tn}")
            print(f" False Positives: {fp} | False Negatives: {fn}")
            print("="*50)

            # Optional score histogram plot if matplotlib is installed
            try:
                import matplotlib.pyplot as plt
                plt.figure(figsize=(10, 5))
                plt.hist(scores[y_true == 0], bins=15, alpha=0.5, label="Good", color="green")
                plt.hist(scores[y_true == 1], bins=15, alpha=0.5, label="Defect", color="red")
                plt.axvline(threshold, color="blue", linestyle="dashed", linewidth=2, label="Threshold")
                plt.xlabel("Anomaly Score")
                plt.ylabel("Count")
                plt.title("Distribution of Anomaly Scores")
                plt.legend()
                plt.grid(True, alpha=0.3)
                plot_path = model_dir / "eval_scores.png"
                plt.savefig(plot_path)
                print(f"[*] Anomaly distribution plot saved to: {plot_path}")
            except Exception as e:
                print(f"[*] Skipping plot generation (matplotlib failed: {e})")
        else:
            print("[WARN] No matching labels found in CSV.")

    # Print general recommendations
    defect_scores = [s for _, s, d in results if d == "DEFECT"]
    ok_scores = [s for _, s, d in results if d == "OK"]
    
    if len(defect_scores) > 0 and len(ok_scores) > 0:
        separation = np.mean(defect_scores) - np.mean(ok_scores)
        if separation > 0.15:
            print("\n[VERDICT] Model separation looks good. Ready to collect few-shot defects and calibrate.")
        else:
            print("\n[VERDICT] Separation is narrow. Consider capturing more training images with consistent background/lighting.")
    else:
        print("\n[VERDICT] Ready. Collect few-shot defect images using 'python src/capture_defects.py'.")


if __name__ == "__main__":
    main()
