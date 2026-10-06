"""106-point 2D face landmarks (InsightFace `2d106det`), onnxruntime on CPU.

The graph starts with its own Sub/Mul normalisation nodes, so it is fed raw
0-255 RGB pixels - feeding it mean/std-normalised input (as the detector
wants) silently produces landmarks that look plausible and are wrong.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

from .detector import Face

INPUT_SIZE = 192
# The crop is the face box enlarged so the whole contour fits, as insightface does.
BOX_ENLARGE = 1.5


class Landmarker:
    def __init__(self, session):
        self._session = session
        self._input = session.get_inputs()[0].name

    @classmethod
    def load(cls, path: Path, providers: list[str], intra_threads: int = 0) -> Landmarker:
        if not path.exists():
            raise ModelNotAvailable(f"Landmark model not found: {path.name}", details={"path": str(path)})
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        return cls(ort.InferenceSession(str(path), opts, providers=providers))

    def landmarks(self, img: np.ndarray, face: Face) -> np.ndarray:
        """BGR image + detected face -> (106, 2) points in image pixels."""
        x1, y1, x2, y2 = face.box
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        scale = INPUT_SIZE / (max(x2 - x1, y2 - y1) * BOX_ENLARGE)
        # Similarity transform: scale about the box centre, centre -> crop centre.
        m = np.array(
            [[scale, 0, INPUT_SIZE / 2 - cx * scale], [0, scale, INPUT_SIZE / 2 - cy * scale]],
            np.float32,
        )
        crop = cv2.warpAffine(img, m, (INPUT_SIZE, INPUT_SIZE), borderValue=0)
        blob = cv2.dnn.blobFromImage(crop, 1.0, (INPUT_SIZE, INPUT_SIZE), (0, 0, 0), swapRB=True)
        pred = self._session.run(None, {self._input: blob})[0].reshape(-1, 2)
        pts = (pred + 1) * (INPUT_SIZE / 2)
        inv = cv2.invertAffineTransform(m)
        return (pts @ inv[:, :2].T + inv[:, 2]).astype(np.float32)
