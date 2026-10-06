"""ArcFace embeddings (InsightFace `w600k_r50`), onnxruntime on CPU.

Used by face capture to check that the still photo shows the same person as
the camera stream, and later by face match. Alignment follows insightface's
`norm_crop`: a similarity transform from the 5 SCRFD keypoints onto the
canonical 112x112 ArcFace template.
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

from .detector import Face

SIZE = 112
# insightface.utils.face_align.arcface_dst
ARCFACE_TEMPLATE = np.array(
    [[38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366], [41.5493, 92.3655], [70.7299, 92.2041]],
    np.float32,
)


def align(img: np.ndarray, kps: np.ndarray) -> np.ndarray:
    """BGR image + 5 keypoints -> 112x112 aligned face (BGR)."""
    m, _ = cv2.estimateAffinePartial2D(kps.astype(np.float32), ARCFACE_TEMPLATE, method=cv2.LMEDS)
    if m is None:
        raise ValueError("Could not estimate face alignment from keypoints")
    return cv2.warpAffine(img, m, (SIZE, SIZE), borderValue=0)


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


class FaceEmbedder:
    def __init__(self, session):
        self._session = session
        self._input = session.get_inputs()[0].name

    @classmethod
    def load(cls, path: Path, providers: list[str], intra_threads: int = 0) -> FaceEmbedder:
        if not path.exists():
            raise ModelNotAvailable(f"Face embedding model not found: {path.name}", details={"path": str(path)})
        import onnxruntime as ort

        opts = ort.SessionOptions()
        if intra_threads:
            opts.intra_op_num_threads = intra_threads
        return cls(ort.InferenceSession(str(path), opts, providers=providers))

    def embed(self, img: np.ndarray, face: Face) -> np.ndarray:
        """L2-normalised 512-d embedding of `face` in a BGR image."""
        aligned = align(img, face.kps)
        blob = cv2.dnn.blobFromImage(aligned, 1.0 / 127.5, (SIZE, SIZE), (127.5, 127.5, 127.5), swapRB=True)
        vec = self._session.run(None, {self._input: blob})[0][0]
        return vec / (np.linalg.norm(vec) + 1e-9)
