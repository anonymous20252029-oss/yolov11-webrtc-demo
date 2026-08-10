import time
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from typing import Tuple, Dict

class AcceleratedYOLOv11Engine:
    def __init__(self, model_path: str = "models/best.onnx", conf_threshold: float = 0.40):
        self.conf_threshold = conf_threshold
        self.model = YOLO(model_path)
        dummy = np.zeros((640, 640, 3), dtype=np.uint8)
        _ = self.model(dummy, verbose=False)

    def process_frame(self, frame_bgr: np.ndarray) -> Tuple[np.ndarray, Dict[str, float]]:
        t0 = time.perf_counter()
        h, w, _ = frame_bgr.shape
        resized = cv2.resize(frame_bgr, (640, 640))
        t1 = time.perf_counter()
        
        results = self.model.predict(
            source=resized,
            conf=self.conf_threshold,
            verbose=False,
            device=0 if torch.cuda.is_available() else "cpu"
        )[0]
        t2 = time.perf_counter()
        
        annotated = results.plot()
        annotated = cv2.resize(annotated, (w, h))
        t3 = time.perf_counter()

        metrics = {
            "preprocess_ms": (t1 - t0) * 1000.0,
            "inference_ms": (t2 - t1) * 1000.0,
            "postprocess_ms": (t3 - t2) * 1000.0,
            "total_ms": (t3 - t0) * 1000.0,
            "fps": 1000.0 / ((t3 - t0) * 1000.0) if (t3 - t0) > 0 else 0.0
        }
        return annotated, metrics
