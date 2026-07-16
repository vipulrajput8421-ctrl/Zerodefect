"""
finetune_fewshot.py — Calibrate and train a lightweight few-shot defect classifier.

Loads few-shot defect images, extracts spatial embeddings, and trains a KNN classifier
to classify WHICH defect type is present on anomalous parts.
"""

import os
import sys
import pickle
import argparse
from pathlib import Path

import numpy as np
import torch
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder

# Import utilities using relative import path
sys.path.insert(0, str(Path(__file__).resolve().parent))
from utils import (
    DATA_DEFECTS, MODELS_DIR, DEFAULT_IMAGE_SIZE,
    get_extractor, load_model_info, save_model_info,
    load_image, resolve_device, get_defect_types
)


def main():
    parser = argparse.ArgumentParser(description="Few-Shot Defect Classifier Fine-tuning")
    parser.add_argument("--defects-dir", type=str, default=str(DATA_DEFECTS), help="Folder with defect folders")
    parser.add_argument("--model-dir", type=str, default=str(MODELS_DIR), help="Folder with model files")
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Resize dimension")
    parser.add_argument("--n-neighbors", type=int, default=3, help="KNN neighbors")
    parser.add_argument("--device", type=str, default="auto", help="cuda, cpu, mps or auto")
    args = parser.parse_args()

    device = resolve_device(args.device)
    print(f"[*] Starting few-shot calibration. Device: {device}")

    defects_dir = Path(args.defects-dir if hasattr(args, "defects-dir") else args.defects_dir)
    model_dir = Path(args.model-dir if hasattr(args, "model-dir") else args.model_dir)

    model_info_path = model_dir / "model_info.json"
    if not model_info_path.exists():
        print(f"[ERROR] Baseline model info not found at {model_info_path}")
        print("Please train the model first: python src/train_model.py")
        sys.exit(1)

    defect_types = get_defect_types()
    if not defect_types:
        print("[ERROR] No non-empty defect folders found in data/defects/.")
        print("Please run src/capture_defects.py to capture defect examples.")
        sys.exit(1)

    print(f"[*] Found {len(defect_types)} defect types: {defect_types}")

    extractor = get_extractor(device)

    embeddings = []
    labels = []

    # Load all defect images
    extensions = (".jpg", ".jpeg", ".png")
    for defect_type in defect_types:
        folder = defects_dir / defect_type
        img_paths = sorted([p for p in folder.iterdir() if p.suffix.lower() in extensions])
        
        if len(img_paths) < 2:
            print(f"[WARN] Defect type '{defect_type}' has fewer than 2 images ({len(img_paths)}).")
            print("Please capture at least 2 (ideally 5-20) images per defect.")
        
        for path in img_paths:
            tensor = load_image(path, args.image_size)
            emb = extractor.extract_image_embedding(tensor)
            embeddings.append(emb)
            labels.append(defect_type)

    if len(embeddings) < 4:
        print("[ERROR] Insufficient defect images to train classifier. Need at least 4 total.")
        sys.exit(1)

    X = np.array(embeddings)
    y = np.array(labels)

    print(f"[*] Extracted embeddings: {X.shape} for few-shot training")

    # Encode label strings to integers
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)

    # Leave-One-Out Cross Validation
    n_samples = len(X)
    correct = 0
    
    # Train neighbors parameter clamped to sample size
    k_neighbors = min(args.n_neighbors, n_samples - 1)
    k_neighbors = max(1, k_neighbors)

    for i in range(n_samples):
        # Splitting indices
        X_train = np.delete(X, i, axis=0)
        y_train = np.delete(y_encoded, i, axis=0)
        X_test = X[i].reshape(1, -1)
        y_test = y_encoded[i]

        knn_loo = KNeighborsClassifier(n_neighbors=min(k_neighbors, len(X_train)))
        knn_loo.fit(X_train, y_train)
        pred = knn_loo.predict(X_test)[0]
        if pred == y_test:
            correct += 1

    loo_acc = correct / n_samples
    print(f"[*] Leave-One-Out (LOO) Classification Accuracy: {loo_acc * 100:.2f}%")

    # Fit final classifier
    knn = KNeighborsClassifier(n_neighbors=k_neighbors)
    knn.fit(X, y_encoded)

    # Load model info and update it
    model_info = load_model_info(model_info_path)
    threshold = model_info.get("threshold", 1.0)

    # Save classifier payload
    classifier_payload = {
        "knn": knn,
        "label_encoder": le,
        "defect_types": list(le.classes_),
        "threshold": threshold
    }

    classifier_path = model_dir / "classifier.pkl"
    with open(classifier_path, "wb") as f:
        pickle.dump(classifier_payload, f)

    # Update metadata
    model_info["classifier_ready"] = True
    model_info["defect_types"] = list(le.classes_)
    save_model_info(model_info, model_info_path)

    # Final summary box
    print("\n" + "="*50)
    print(" FEW-SHOT CALIBRATION COMPLETE")
    print("="*50)
    print(f" Defect Classes Configured : {len(le.classes_)}")
    print(f" Total Few-Shot Images     : {len(X)}")
    print(f" KNN Neighbors Used        : {k_neighbors}")
    print(f" Cross-Validation Accuracy : {loo_acc * 100:.2f}%")
    print(f" Saved Classifier Path     : {classifier_path}")
    print("="*50)
    print("\n[NEXT STEP] Run the live webcam demo:")
    print("  python src/live_demo.py")


if __name__ == "__main__":
    main()
