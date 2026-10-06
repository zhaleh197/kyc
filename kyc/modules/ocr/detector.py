"""ONNX YOLO field detector, CPU only.

The training half of this lives in `training/kaggle/`: a YOLOv8 model is
trained on a Kaggle GPU, exported with `format="onnx"`, and the resulting file
is dropped into `models/`. Nothing at inference time depends on ultralytics,
torch or a Roboflow API key - just onnxruntime and numpy.

Both detection and segmentation exports are accepted. Segmentation adds 32
mask coefficients per box and a second output tensor; we read the boxes and
ignore the mask branch, because the card is already rectified so axis-aligned
boxes are enough to crop a field.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable


@dataclass(frozen=True)
class Detection:
    class_index: int
    class_name: str
    confidence: float
    box: tuple[int, int, int, int]  # x1, y1, x2, y2 in source-image pixels

    @property
    def area(self) -> int:
        x1, y1, x2, y2 = self.box
        return max(0, x2 - x1) * max(0, y2 - y1)


def letterbox(img: np.ndarray, size: int) -> tuple[np.ndarray, float, int, int]:
    """Resize preserving aspect ratio and pad to a square.

    Returns the padded image plus the scale and offsets needed to map
    predictions back to source coordinates.
    """
    h, w = img.shape[:2]
    scale = min(size / h, size / w)
    new_w, new_h = int(round(w * scale)), int(round(h * scale))
    resized = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((size, size, 3), 114, dtype=np.uint8)
    dx, dy = (size - new_w) // 2, (size - new_h) // 2
    canvas[dy:dy + new_h, dx:dx + new_w] = resized
    return canvas, scale, dx, dy


class FieldDetector:
    """Thread-safe lazy wrapper around one ONNX field-detection model."""

    _sessions: dict[str, FieldDetector] = {}
    _lock = threading.Lock()

    def __init__(self, model_path: Path, class_names: tuple[str, ...], providers: list[str], intra_threads: int = 0):
        if not model_path.exists():
            raise ModelNotAvailable(
                f"Field detector not found at {model_path}. Train it with "
                f"training/kaggle/train_field_detector.py and copy the exported .onnx into models/.",
                details={"model": model_path.name},
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - install-time problem
            raise ModelNotAvailable("onnxruntime is not installed; pip install -r requirements/base.txt") from exc

        options = ort.SessionOptions()
        if intra_threads:
            options.intra_op_num_threads = intra_threads
        self.session = ort.InferenceSession(str(model_path), options, providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        shape = self.session.get_inputs()[0].shape
        # Static exports carry the size in the shape; dynamic ones give a string.
        self.imgsz = shape[2] if isinstance(shape[2], int) else 640
        self.class_names = class_names
        self.model_path = model_path

    @classmethod
    def load(
        cls,
        model_path: Path,
        class_names: tuple[str, ...],
        providers: list[str],
        intra_threads: int = 0,
    ) -> FieldDetector:
        """Cached by path: ONNX sessions are expensive to build and safe to share."""
        key = str(model_path)
        with cls._lock:
            cached = cls._sessions.get(key)
            if cached is None:
                cached = cls(model_path, class_names, providers, intra_threads)
                cls._sessions[key] = cached
            return cached

    def detect(self, img: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[Detection]:
        blob, scale, dx, dy = self._preprocess(img)
        outputs = self.session.run(None, {self.input_name: blob})
        return self._postprocess(outputs[0], img.shape[:2], scale, dx, dy, conf_threshold, iou_threshold)

    def _preprocess(self, img: np.ndarray) -> tuple[np.ndarray, float, int, int]:
        padded, scale, dx, dy = letterbox(img, self.imgsz)
        rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
        blob = rgb.astype(np.float32) / 255.0
        blob = np.transpose(blob, (2, 0, 1))[None]  # NCHW
        return np.ascontiguousarray(blob), scale, dx, dy

    def _postprocess(
        self,
        raw: np.ndarray,
        src_shape: tuple[int, int],
        scale: float,
        dx: int,
        dy: int,
        conf_threshold: float,
        iou_threshold: float,
    ) -> list[Detection]:
        # YOLOv8 head: (1, 4 + num_classes [+ 32 mask coeffs], num_anchors)
        pred = np.squeeze(raw, axis=0)
        if pred.shape[0] < pred.shape[1]:
            pred = pred.T  # -> (num_anchors, channels)

        num_classes = len(self.class_names)
        if pred.shape[1] < 4 + num_classes:
            raise ModelNotAvailable(
                f"Model {self.model_path.name} emits {pred.shape[1]} channels, too few for "
                f"{num_classes} classes. The profile's class_names is out of sync with the export.",
            )
        scores = pred[:, 4:4 + num_classes]
        class_indices = scores.argmax(axis=1)
        confidences = scores[np.arange(scores.shape[0]), class_indices]

        keep = confidences >= conf_threshold
        if not keep.any():
            return []
        boxes_xywh = pred[keep, :4]
        confidences = confidences[keep]
        class_indices = class_indices[keep]

        # cx,cy,w,h (letterboxed space) -> x1,y1,x2,y2 (source space)
        cx, cy, bw, bh = boxes_xywh.T
        x1 = (cx - bw / 2 - dx) / scale
        y1 = (cy - bh / 2 - dy) / scale
        x2 = (cx + bw / 2 - dx) / scale
        y2 = (cy + bh / 2 - dy) / scale

        h, w = src_shape
        x1 = np.clip(x1, 0, w - 1)
        y1 = np.clip(y1, 0, h - 1)
        x2 = np.clip(x2, 0, w - 1)
        y2 = np.clip(y2, 0, h - 1)

        nms_boxes = np.stack([x1, y1, x2 - x1, y2 - y1], axis=1).tolist()
        indices = cv2.dnn.NMSBoxes(nms_boxes, confidences.tolist(), conf_threshold, iou_threshold)
        if len(indices) == 0:
            return []

        detections = []
        for i in np.array(indices).ravel():
            index = int(class_indices[i])
            detections.append(
                Detection(
                    class_index=index,
                    class_name=self.class_names[index] if index < len(self.class_names) else str(index),
                    confidence=float(confidences[i]),
                    box=(int(x1[i]), int(y1[i]), int(x2[i]), int(y2[i])),
                )
            )
        return detections


def best_per_class(detections: list[Detection]) -> dict[str, Detection]:
    """An ID card has exactly one of each field; keep the highest-confidence box.

    Guards against a second, lower-confidence box on a reflection or on the
    back-side text bleeding through a thin card.
    """
    best: dict[str, Detection] = {}
    for det in detections:
        current = best.get(det.class_name)
        if current is None or det.confidence > current.confidence:
            best[det.class_name] = det
    return best
