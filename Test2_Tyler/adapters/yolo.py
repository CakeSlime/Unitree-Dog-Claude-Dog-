"""CPU YOLOv10n ONNX inference, sharing the group's Detection interface.

Weights: THU-MIG YOLOv10, ONNX Community export (AGPL-3.0).
Uses ONNX Runtime instead of installing a second PyTorch/Ultralytics stack.
"""
from pathlib import Path
import hashlib
import time

import cv2
import numpy as np

from go2_stream.perception.detector import Detection

MODEL_URL = "https://huggingface.co/onnx-community/yolov10n/resolve/main/onnx/model.onnx"
MODEL_SHA256 = "a77dd863933f184a19e84361c64b788228a7c7dacc2c78939239a96ad3efca3b"
MODEL_PATH = Path(__file__).resolve().parents[1] / ".cache/models/yolov10n.onnx"


def verify_model(path):
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    if digest != MODEL_SHA256:
        raise ValueError(f"YOLO checksum mismatch: {digest}")
    return digest


def decode(rows, width, height, confidence=0.3, scale=None):
    """YOLOv10 end-to-end output: xyxy in 640-space, score, COCO class index."""
    rows = np.asarray(rows).reshape(-1, 6)
    scale = scale or min(640 / width, 640 / height)
    detections = []
    for x1, y1, x2, y2, score, cls in rows:
        if not np.isfinite([x1, y1, x2, y2, score, cls]).all() or score < confidence or int(cls) != 56:
            continue
        box = (max(0., float(x1) / scale), max(0., float(y1) / scale),
               min(width - 1., float(x2) / scale), min(height - 1., float(y2) / scale))
        # Reject tiny/edge-clipped fragments; they are not an inspection target.
        if box[2] - box[0] >= 20 and box[3] - box[1] >= 30 and box[0] > 2 and box[2] < width - 3:
            detections.append(Detection(56, "chair", float(score), box))
    return detections


class ChairDetector:
    def __init__(self, confidence=0.3, model_path=MODEL_PATH):
        import onnxruntime as ort
        self.sha256 = verify_model(model_path)
        options = ort.SessionOptions()
        options.intra_op_num_threads = options.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(model_path), sess_options=options,
                                           providers=["CPUExecutionProvider"])
        self.input = self.session.get_inputs()[0].name
        self.confidence = confidence
        self.last_latency_s = 0.

    def detect_rgb(self, rgb):
        # Published HF processor: longest edge 640, RGB [0,1], bottom/right zero pad.
        h, w = rgb.shape[:2]
        scale = min(640 / w, 640 / h)
        resized = cv2.resize(rgb, (round(w * scale), round(h * scale)))
        padded = np.zeros((640, 640, 3), np.uint8)
        padded[:resized.shape[0], :resized.shape[1]] = resized
        x = padded.transpose(2, 0, 1)[None].astype(np.float32) / 255.
        start = time.perf_counter()
        out = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0]
        self.last_latency_s = time.perf_counter() - start
        if out.shape[-1] != 6:
            raise ValueError(f"Unexpected YOLOv10 output shape: {out.shape}")
        return decode(out, rgb.shape[1], rgb.shape[0], self.confidence)
