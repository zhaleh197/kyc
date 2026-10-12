"""MiniFASNet passive anti-spoof models (Minivision Silent-Face, Apache-2.0),
exported to ONNX by scripts/export_antispoof_onnx.py.

Each model looks at the face box enlarged by its own context scale (2.7x,
4.0x), resized to 80x80, BGR, raw 0-255 floats - upstream's ToTensor does not
divide by 255. Output: 3 logits; class 1 = real, 0 and 2 = two spoof types.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

SIZE = 80


def context_crop(img: np.ndarray, box: tuple[float, float, float, float], scale: float) -> tuple[np.ndarray, bool]:
    """Upstream CropImage._get_new_box: the box enlarged `scale` times about
    its centre, shifted back inside the frame. If it cannot fit, the scale is
    reduced (clamped) - returned as the second value, since the model then sees
    less context than it was trained with."""
    src_h, src_w = img.shape[:2]
    x1, y1, x2, y2 = box
    bw, bh = max(x2 - x1, 1.0), max(y2 - y1, 1.0)
    s = min((src_h - 1) / bh, (src_w - 1) / bw, scale)
    nw, nh = bw * s, bh * s
    cx, cy = x1 + bw / 2, y1 + bh / 2
    left, top, right, bottom = cx - nw / 2, cy - nh / 2, cx + nw / 2, cy + nh / 2
    if left < 0:
        right, left = right - left, 0
    if top < 0:
        bottom, top = bottom - top, 0
    if right > src_w - 1:
        left, right = left - (right - src_w + 1), src_w - 1
    if bottom > src_h - 1:
        top, bottom = top - (bottom - src_h + 1), src_h - 1
    crop = img[int(top) : int(bottom) + 1, int(left) : int(right) + 1]
    return cv2.resize(crop, (SIZE, SIZE)), s < scale


class AntiSpoofModel:
    def __init__(self, session, scale: float, name: str):
        self._session = session
        self._input = session.get_inputs()[0].name
        self.scale = scale
        self.name = name

    @classmethod
    def load(cls, path: Path, scale: float, providers: list[str], intra_threads: int = 0) -> AntiSpoofModel:
        if not path.exists():
            raise ModelNotAvailable(f"Anti-spoof model not found: {path.name}", details={"path": str(path)})
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        return cls(ort.InferenceSession(str(path), opts, providers=providers), scale, path.stem)

    def live_probabilities(self, crops: list[np.ndarray]) -> np.ndarray:
        """P(real) for a batch of 80x80 BGR crops."""
        batch = np.stack([c.transpose(2, 0, 1) for c in crops]).astype(np.float32)
        logits = self._session.run(None, {self._input: batch})[0]
        z = np.exp(logits - logits.max(axis=1, keepdims=True))
        return (z / z.sum(axis=1, keepdims=True))[:, 1]
