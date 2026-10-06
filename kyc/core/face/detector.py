"""SCRFD face detector (InsightFace `det_10g`), run through onnxruntime on CPU.

Shared by face capture, face match and anti-spoof so an image is detected once
and every module agrees on where the face is. No `insightface` package: the
pre/post-processing below reproduces what `insightface.model_zoo.scrfd` does,
which keeps the serving image free of its Cython build dependencies.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

# det_10g's graph has a fixed 640x640 input (its output shapes are static:
# 12800 = 80*80 anchors*2 at stride 8), so the input size is not a tunable.
INPUT_SIZE = 640
STRIDES = (8, 16, 32)
ANCHORS_PER_CELL = 2
MEAN = 127.5
STD = 128.0


@dataclass(frozen=True)
class Face:
    box: tuple[float, float, float, float]  # x1, y1, x2, y2 in source-image pixels
    score: float
    kps: np.ndarray  # (5, 2): left eye, right eye, nose, left mouth, right mouth (image left/right)

    @property
    def width(self) -> float:
        return self.box[2] - self.box[0]

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]

    @property
    def area(self) -> float:
        return max(0.0, self.width) * max(0.0, self.height)

    def int_box(self) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = self.box
        return int(round(x1)), int(round(y1)), int(round(x2)), int(round(y2))


def _anchor_centers(stride: int) -> np.ndarray:
    size = INPUT_SIZE // stride
    ys, xs = np.mgrid[:size, :size]
    centers = (np.stack([xs, ys], axis=-1).astype(np.float32) * stride).reshape(-1, 2)
    return np.repeat(centers, ANCHORS_PER_CELL, axis=0)


_CENTERS = {s: _anchor_centers(s) for s in STRIDES}


def _nms(boxes: np.ndarray, scores: np.ndarray, iou_threshold: float) -> list[int]:
    order = scores.argsort()[::-1]
    keep: list[int] = []
    areas = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    while order.size:
        i = order[0]
        keep.append(int(i))
        xx1 = np.maximum(boxes[i, 0], boxes[order[1:], 0])
        yy1 = np.maximum(boxes[i, 1], boxes[order[1:], 1])
        xx2 = np.minimum(boxes[i, 2], boxes[order[1:], 2])
        yy2 = np.minimum(boxes[i, 3], boxes[order[1:], 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        iou = inter / (areas[i] + areas[order[1:]] - inter + 1e-9)
        order = order[1:][iou <= iou_threshold]
    return keep


class FaceDetector:
    def __init__(self, session, *, nms_iou: float = 0.4):
        self._session = session
        self._input = session.get_inputs()[0].name
        self._nms_iou = nms_iou

    @classmethod
    def load(cls, path: Path, providers: list[str], intra_threads: int = 0) -> FaceDetector:
        if not path.exists():
            raise ModelNotAvailable(f"Face detector model not found: {path.name}", details={"path": str(path)})
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        return cls(ort.InferenceSession(str(path), opts, providers=providers))

    def detect(self, img: np.ndarray, conf_threshold: float = 0.5) -> list[Face]:
        """BGR image -> faces, largest first."""
        blob, scale = self._preprocess(img)
        outputs = self._session.run(None, {self._input: blob})
        n = len(STRIDES)

        boxes, scores, kpss = [], [], []
        for i, stride in enumerate(STRIDES):
            score = outputs[i].reshape(-1)
            keep = score >= conf_threshold
            if not keep.any():
                continue
            centers = _CENTERS[stride][keep]
            dist = outputs[i + n][keep] * stride
            kps = outputs[i + 2 * n][keep].reshape(-1, 5, 2) * stride
            boxes.append(np.concatenate([centers - dist[:, :2], centers + dist[:, 2:]], axis=1))
            kpss.append(kps + centers[:, None, :])
            scores.append(score[keep])

        if not boxes:
            return []
        boxes_a = np.concatenate(boxes) / scale
        scores_a = np.concatenate(scores)
        kps_a = np.concatenate(kpss) / scale
        h, w = img.shape[:2]
        faces = []
        for i in _nms(boxes_a, scores_a, self._nms_iou):
            x1, y1, x2, y2 = boxes_a[i]
            box = (float(max(0, x1)), float(max(0, y1)), float(min(w, x2)), float(min(h, y2)))
            faces.append(Face(box=box, score=float(scores_a[i]), kps=kps_a[i].astype(np.float32)))
        return sorted(faces, key=lambda f: f.area, reverse=True)

    @staticmethod
    def _preprocess(img: np.ndarray) -> tuple[np.ndarray, float]:
        """Resize keeping aspect ratio into the top-left of a 640x640 canvas,
        as insightface does, so box decoding is a single division."""
        h, w = img.shape[:2]
        scale = INPUT_SIZE / max(h, w)
        resized = cv2.resize(img, (int(round(w * scale)), int(round(h * scale))))
        canvas = np.zeros((INPUT_SIZE, INPUT_SIZE, 3), np.uint8)
        canvas[: resized.shape[0], : resized.shape[1]] = resized
        # The model was trained on RGB; OpenCV decodes BGR.
        blob = cv2.dnn.blobFromImage(canvas, 1.0 / STD, (INPUT_SIZE, INPUT_SIZE), (MEAN, MEAN, MEAN), swapRB=True)
        return blob, scale
