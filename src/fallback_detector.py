# src/fallback_detector.py

import moondream as md
from PIL import Image

class FallbackDetector:
    def __init__(self, yolo_detector, moondream_path="models/moondream-2b-int8.mf", conf_threshold=0.25):
        """
        yolo_detector: your existing YOLOv5n detector instance
        conf_threshold: below this, YOLO result is considered unreliable
        """
        self.yolo = yolo_detector
        self.conf_threshold = conf_threshold
        # Initialize moondream with local .mf file
        print(f"[FallbackDetector] Loading Moondream from {moondream_path}...")
        self.moondream = md.vl(model=moondream_path)
        print("[FallbackDetector] Moondream loaded successfully!")

    def detect(self, img_pil, img_bgr):
        # Step 1: try YOLO first (fast, cheap)
        yolo_results = self.yolo.detect(img_bgr, conf=0.1, iou=0.45) # We use a low confidence so we can check max

        # If we have no detections at all, or the best detection is below our threshold
        # Then we consider it unconfident and fall back to Moondream
        if yolo_results and max(r["confidence"] for r in yolo_results) >= self.conf_threshold:
            # We have a confident YOLO detection!
            return {
                "engine": "yolo",
                "detections": yolo_results
            }

        # Step 2: YOLO failed or low confidence — fallback to Moondream
        print("[FallbackDetector] Low YOLO confidence. Falling back to Moondream...")
        encoded = self.moondream.encode_image(img_pil)
        
        prompt_is_defect = "Does this aircraft skin have defects? Answer YES or NO."
        ans_is_defect = self.moondream.query(encoded, prompt_is_defect)["answer"].strip().lower()
        is_defect = "yes" in ans_is_defect or ("no" not in ans_is_defect and "normal" not in ans_is_defect)
        
        if is_defect:
            prompt_type = (
                "Which of the following defects is most prominent in this image: "
                "scratch, dent, crack, hole, peeling paint, white patch, or corrosion? "
                "Answer with only the category name."
            )
            ans_type = self.moondream.query(encoded, prompt_type)["answer"].strip().lower()
            
            prompt_desc = "Describe the defect in one short sentence."
            ans_desc = self.moondream.query(encoded, prompt_desc)["answer"].strip()
            
            return {
                "engine": "moondream_fallback",
                "yolo_detections": yolo_results,
                "is_defect": True,
                "ans_type": ans_type,
                "description": ans_desc
            }
        else:
            return {
                "engine": "moondream_fallback",
                "yolo_detections": yolo_results,
                "is_defect": False,
                "description": "No defects detected."
            }
