"""
utils.py — Shared preprocessing and backbone utilities for ZeroDefect.

All scripts import from here to ensure consistent image transforms,
feature extraction, and path resolution across training, evaluation,
and live inference.
"""

import os
import json
import numpy as np
from pathlib import Path
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torchvision.transforms as transforms

# ─────────────────────────────────────────────
# Constants / Project paths
# ─────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent  # d:\ZeroDefect
DATA_GOOD = ROOT / "data" / "good"
DATA_DEFECTS = ROOT / "data" / "defects"
DATA_TEST = ROOT / "data" / "test"
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / "logs"
THUMBS_DIR = LOGS_DIR / "thumbnails"

DEFAULT_IMAGE_SIZE = 256   # Resize all images to this before feature extraction
FEATURE_DIM = 384          # 128 (layer2) + 256 (layer3) after upsampling

# ─────────────────────────────────────────────
# Image transform (must be same at train + inference)
# ─────────────────────────────────────────────

def get_transform(image_size: int = DEFAULT_IMAGE_SIZE) -> transforms.Compose:
    """Returns the canonical image transform for ZeroDefect.
    Always use this — never roll your own transforms — to avoid train/test skew.
    """
    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],   # ImageNet stats
            std=[0.229, 0.224, 0.225],
        ),
    ])


def load_image(path: str | Path, image_size: int = DEFAULT_IMAGE_SIZE) -> torch.Tensor:
    """Load an image from disk and return a [1, 3, H, W] tensor ready for inference."""
    img = Image.open(path).convert("RGB")
    transform = get_transform(image_size)
    return transform(img).unsqueeze(0)  # add batch dim


def load_images_from_folder(
    folder: str | Path,
    image_size: int = DEFAULT_IMAGE_SIZE,
    extensions: tuple = (".jpg", ".jpeg", ".png"),
) -> tuple[list[Path], list[torch.Tensor]]:
    """Load all images from a folder. Returns (paths, tensors)."""
    folder = Path(folder)
    paths = sorted([
        p for p in folder.iterdir()
        if p.suffix.lower() in extensions
    ])
    if not paths:
        raise FileNotFoundError(
            f"No images found in {folder}. "
            f"Expected files with extensions: {extensions}"
        )
    tensors = [load_image(p, image_size) for p in paths]
    return paths, tensors


# ─────────────────────────────────────────────
# PatchCore Feature Extractor (ResNet18 backbone)
# ─────────────────────────────────────────────

class PatchFeatureExtractor(nn.Module):
    """
    Extracts multi-scale patch features from a ResNet18 backbone.

    Uses layer2 (128 ch, stride-8) and layer3 (256 ch, stride-16).
    layer3 is upsampled to match layer2 spatial size, then concatenated.
    Final feature dim = 128 + 256 = 384 per spatial location.

    No fine-tuning — pretrained ImageNet weights only.
    This is the core of PatchCore: we borrow the rich features learned
    from ImageNet and use them as a proxy for "does this look normal?"
    """

    def __init__(self, device: str = "cpu"):
        super().__init__()
        self.device = device

        # Load pretrained ResNet18, freeze all weights
        backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad = False

        # Build two sub-networks up to layer2 and layer3
        self.early = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2,
        ).to(device)
        self.late = backbone.layer3.to(device)

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, 3, H, W] normalized tensor
        Returns:
            features: [B, FEATURE_DIM, H/8, W/8] — one feature vector per patch
        """
        x = x.to(self.device)
        feat2 = self.early(x)                                      # [B, 128, H/8, W/8]
        feat3 = self.late(feat2)                                   # [B, 256, H/16, W/16]
        feat3_up = F.interpolate(
            feat3, size=feat2.shape[2:], mode="bilinear", align_corners=False
        )                                                          # [B, 256, H/8, W/8]
        return torch.cat([feat2, feat3_up], dim=1)                 # [B, 384, H/8, W/8]

    def extract_flat_patches(self, x: torch.Tensor) -> np.ndarray:
        """
        Convenience: extract all patch features for one image as a flat 2D array.
        Args:
            x: [1, 3, H, W] — single image tensor
        Returns:
            patches: [N_patches, 384] numpy array  (N = H/8 * W/8)
        """
        feat = self(x)                   # [1, 384, h, w]
        B, C, h, w = feat.shape
        # Rearrange to [N_patches, C]
        patches = feat.permute(0, 2, 3, 1).reshape(-1, C).cpu().numpy()
        return patches  # [h*w, 384]

    def extract_image_embedding(self, x: torch.Tensor) -> np.ndarray:
        """
        Returns a single [384] embedding per image (spatial mean of patches).
        Used for the few-shot defect classifier.
        """
        patches = self.extract_flat_patches(x)  # [N, 384]
        return patches.mean(axis=0)             # [384]


_extractor_cache: dict = {}

def get_extractor(device: str | None = None) -> PatchFeatureExtractor:
    """Cached extractor — only loads backbone once per device per process."""
    global _extractor_cache
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device not in _extractor_cache:
        _extractor_cache[device] = PatchFeatureExtractor(device=device)
    return _extractor_cache[device]


# ─────────────────────────────────────────────
# Model info helpers
# ─────────────────────────────────────────────

def save_model_info(info: dict, path: Path = None) -> None:
    if path is None:
        path = MODELS_DIR / "model_info.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(info, f, indent=2)


def load_model_info(path: Path = None) -> dict:
    if path is None:
        path = MODELS_DIR / "model_info.json"
    if not path.exists():
        raise FileNotFoundError(
            f"model_info.json not found at {path}. "
            "Did you run src/train_model.py yet?"
        )
    with open(path) as f:
        return json.load(f)


def get_defect_types() -> list[str]:
    """Return sorted list of defect type folders under data/defects/."""
    if not DATA_DEFECTS.exists():
        return []
    return sorted([
        d.name for d in DATA_DEFECTS.iterdir()
        if d.is_dir() and not d.name.startswith(".")
        and any(d.iterdir())  # non-empty
    ])


def resolve_device(device_arg: str | None = None) -> str:
    """Resolve device string: auto-detect CUDA/MPS/CPU."""
    if device_arg and device_arg != "auto":
        return device_arg
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"
