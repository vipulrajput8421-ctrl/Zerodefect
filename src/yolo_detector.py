"""
yolo_detector.py — YOLOv5 Nano ONNX Detector for ZeroDefect.

Provides the YOLODetector class for running inference with the pre-trained
YOLOv5n model exported to ONNX format. Detects 7 aircraft surface defect
categories with bounding boxes, class labels, and confidence scores.

Usage:
    from yolo_detector import YOLODetector
    detector = YOLODetector("models/best.onnx")
    detections = detector.detect(image_bgr, conf=0.25, iou=0.45)
"""

import numpy as np
import cv2
import onnxruntime as ort
from pathlib import Path

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────

YOLO_CLASSES = [
    "crack", "dent", "corrosion", "scratch",
    "paint-peel", "missing-head", "defect"
]

YOLO_COLORS = [
    (0, 0, 255),      # crack       → red
    (0, 255, 0),      # dent        → green
    (255, 0, 0),      # corrosion   → blue
    (0, 255, 255),    # scratch     → yellow
    (255, 0, 255),    # paint-peel  → magenta
    (255, 255, 0),    # missing-head→ cyan
    (128, 128, 128),  # defect      → gray
]

YOLO_INPUT_SIZE = 640


# ─────────────────────────────────────────────
# Preprocessing Utilities
# ─────────────────────────────────────────────

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114),
              auto=False, scaleFill=False, scaleup=True, stride=32):
    """Resize and pad image while meeting stride-multiple constraints."""
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]

    if auto:
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = new_shape[1], new_shape[0]
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]

    dw /= 2
    dh /= 2

    if shape[::-1] != new_unpad:
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right,
                            cv2.BORDER_CONSTANT, value=color)
    return im, ratio, (dw, dh)


def xywh2xyxy(x):
    """Convert nx4 boxes from [x, y, w, h] to [x1, y1, x2, y2]."""
    y = np.copy(x)
    y[..., 0] = x[..., 0] - x[..., 2] / 2  # top left x
    y[..., 1] = x[..., 1] - x[..., 3] / 2  # top left y
    y[..., 2] = x[..., 0] + x[..., 2] / 2  # bottom right x
    y[..., 3] = x[..., 1] + x[..., 3] / 2  # bottom right y
    return y


def non_max_suppression(prediction, conf_thres=0.25, iou_thres=0.45):
    """Run Non-Maximum Suppression (NMS) on YOLOv5 inference results."""
    nc = prediction.shape[2] - 5  # number of classes
    xc = prediction[..., 4] > conf_thres  # candidates

    max_wh = 7680
    max_det = 300

    output = [np.zeros((0, 6))] * prediction.shape[0]
    for xi, x in enumerate(prediction):
        x = x[xc[xi]]

        if not x.shape[0]:
            continue

        box = xywh2xyxy(x[:, :4])
        conf = x[:, 4:5]
        class_prob = x[:, 5:]
        j = np.argmax(class_prob, axis=1, keepdims=True)
        c_prob = np.take_along_axis(class_prob, j, axis=1)

        x = np.concatenate((box, c_prob * conf, j.astype(np.float32)), axis=1)
        x = x[x[:, 4] > conf_thres]

        if not x.shape[0]:
            continue

        x = x[x[:, 4].argsort()[::-1]]

        c = x[:, 5:6] * max_wh
        boxes, scores = x[:, :4] + c, x[:, 4]

        indices = cv2.dnn.NMSBoxes(
            bboxes=boxes.tolist(),
            scores=scores.tolist(),
            score_threshold=conf_thres,
            nms_threshold=iou_thres
        )

        if len(indices) > 0:
            if isinstance(indices, np.ndarray):
                indices = indices.flatten()
            x = x[indices[:max_det]]
            output[xi] = x

    return output


# ─────────────────────────────────────────────
# YOLODetector Class
# ─────────────────────────────────────────────

class YOLODetector:
    """
    YOLOv5 Nano ONNX detector for aircraft surface defect detection.

    Loads a pre-trained ONNX model and provides a simple detect() API
    that accepts BGR numpy images and returns structured detection results.

    Classes detected:
        crack, dent, corrosion, scratch, paint-peel, missing-head, defect
    """

    def __init__(self, model_path: str | Path, input_size: int = YOLO_INPUT_SIZE):
        self.model_path = Path(model_path)
        self.input_size = input_size
        self.classes = YOLO_CLASSES
        self.colors = YOLO_COLORS

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {self.model_path}. "
                "Download it with: hf sync hf://buckets/prath0029/Zerodefect-1.0-bucket ./local"
            )

        self.session = ort.InferenceSession(str(self.model_path))
        self.input_name = self.session.get_inputs()[0].name
        print(f"[YOLODetector] Loaded {self.model_path.name} "
              f"({self.model_path.stat().st_size / 1024 / 1024:.1f} MB, "
              f"{len(self.classes)} classes)")

    def _preprocess(self, img_bgr: np.ndarray):
        """Letterbox resize, normalize, and format for ONNX input."""
        img, ratio, (dw, dh) = letterbox(img_bgr, new_shape=(self.input_size, self.input_size), auto=False)
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img).astype(np.float32) / 255.0
        if len(img.shape) == 3:
            img = np.expand_dims(img, 0)
        return img, ratio, dw, dh

    def detect(self, image: np.ndarray, conf: float = 0.25, iou: float = 0.45) -> list[dict]:
        """
        Run YOLOv5 detection on a BGR numpy image.

        Args:
            image: BGR numpy array (H, W, 3) from OpenCV
            conf: Confidence threshold (0.0-1.0)
            iou: IoU threshold for NMS (0.0-1.0)

        Returns:
            List of detection dicts, each containing:
                - class_name: str (e.g. "crack", "dent")
                - class_id: int (0-6)
                - confidence: float (0.0-1.0)
                - bbox: [x1, y1, x2, y2] in original image coordinates
        """
        if image is None or image.size == 0:
            return []

        img_input, ratio, dw, dh = self._preprocess(image)

        # Run ONNX inference
        outputs = self.session.run(None, {self.input_name: img_input})
        raw_detections = non_max_suppression(outputs[0], conf_thres=conf, iou_thres=iou)[0]

        results = []
        if len(raw_detections) > 0:
            # Scale boxes back to original image size
            raw_detections[:, [0, 2]] -= dw
            raw_detections[:, [1, 3]] -= dh
            raw_detections[:, :4] /= ratio[0]

            # Clamp to image bounds
            h, w = image.shape[:2]
            raw_detections[:, [0, 2]] = np.clip(raw_detections[:, [0, 2]], 0, w)
            raw_detections[:, [1, 3]] = np.clip(raw_detections[:, [1, 3]], 0, h)

            for det in raw_detections:
                x1, y1, x2, y2, confidence, cls_id = det
                cls_id = int(cls_id)
                results.append({
                    "class_name": self.classes[cls_id] if cls_id < len(self.classes) else "unknown",
                    "class_id": cls_id,
                    "confidence": round(float(confidence), 4),
                    "bbox": [int(x1), int(y1), int(x2), int(y2)],
                })

        return results

    def detect_and_draw(self, image: np.ndarray, conf: float = 0.25,
                        iou: float = 0.45) -> tuple[np.ndarray, list[dict]]:
        """
        Run detection and draw bounding boxes on a copy of the image.

        Returns:
            (annotated_image, detections)
        """
        detections = self.detect(image, conf=conf, iou=iou)
        annotated = image.copy()

        for det in detections:
            x1, y1, x2, y2 = det["bbox"]
            cls_id = det["class_id"]
            label = f"{det['class_name']} {det['confidence']:.2f}"
            color = self.colors[cls_id % len(self.colors)]

            cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)

            # Label background
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(annotated, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
            cv2.putText(annotated, label, (x1 + 2, y1 - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return annotated, detections

    def get_model_info(self) -> dict:
        """Return metadata about the loaded YOLO model."""
        return {
            "model_type": "YOLOv5 Nano",
            "model_path": str(self.model_path),
            "model_size_mb": round(self.model_path.stat().st_size / 1024 / 1024, 2),
            "input_size": self.input_size,
            "num_classes": len(self.classes),
            "classes": self.classes,
            "format": "ONNX",
            "metrics": {
                "precision": 0.428,
                "recall": 0.358,
                "mAP50": 0.319,
                "mAP50_95": 0.235,
            },
        }


# ─────────────────────────────────────────────
# Cached Singleton
# ─────────────────────────────────────────────

_detector_cache: dict = {}


def get_yolo_detector(model_path: str | Path | None = None) -> YOLODetector:
    """Get a cached YOLODetector instance. Only loads the model once per process."""
    global _detector_cache

    if model_path is None:
        # Default path: models/best_balanced.onnx relative to project root
        model_path = Path(__file__).resolve().parent.parent / "models" / "best_balanced.onnx"

    model_path = Path(model_path)
    key = str(model_path)

    if key not in _detector_cache:
        _detector_cache[key] = YOLODetector(model_path)

    return _detector_cache[key]
