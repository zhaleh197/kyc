"""Face-occlusion classifier: the prototype's MobileNetV3
(`phase1_facedetect/obstruction_detector50.pt`) exported to ONNX.

Output is one raw logit; higher = clearer face (the prototype called a face
"clear" at logit >= 0.25). See models/manifest.yaml for why serving uses a far
lower threshold: at 0.25 it flagged 5 of 22 uncovered test faces as covered -
four women in headscarves and a bearded man.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

from .detector import Face

SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class OcclusionClassifier:
    def __init__(self, session):
        self._session = session
        self._input = session.get_inputs()[0].name

    @classmethod
    def load(cls, path: Path, providers: list[str], intra_threads: int = 0) -> OcclusionClassifier:
        if not path.exists():
            raise ModelNotAvailable(f"Occlusion model not found: {path.name}", details={"path": str(path)})
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        return cls(ort.InferenceSession(str(path), opts, providers=providers))

    def clear_logit(self, img: np.ndarray, face: Face) -> float:
        """Raw logit on the face box crop (as the prototype was trained)."""
        x1, y1, x2, y2 = face.int_box()
        crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
        if crop.size == 0:
            return float("-inf")
        rgb = cv2.cvtColor(cv2.resize(crop, (SIZE, SIZE)), cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
        blob = ((rgb - MEAN) / STD).transpose(2, 0, 1)[None].astype(np.float32)
        return float(self._session.run(None, {self._input: blob})[0].reshape(-1)[0])
