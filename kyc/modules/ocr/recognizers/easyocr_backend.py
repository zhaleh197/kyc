"""EasyOCR backend - the no-training default.

EasyOCR loads a detector plus a Persian recogniser and is heavy to construct,
so the Reader is built once per process behind a lock. It is also not
thread-safe for concurrent `readtext` calls, hence the second lock around
inference. That serialisation is acceptable because this backend is the
bootstrap path, not the production one.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np

from kyc.core.errors import BackendNotInstalled

from .base import Recognition

_reader = None
_build_lock = threading.Lock()
_infer_lock = threading.Lock()


def _get_reader():
    global _reader
    if _reader is None:
        with _build_lock:
            if _reader is None:
                try:
                    import easyocr
                except ImportError as exc:
                    raise BackendNotInstalled(
                        "easyocr is not installed. pip install -r requirements/ocr.txt, "
                        "or switch KYC_OCR_TEXT_BACKEND to a trained crnn_onnx model."
                    ) from exc
                # gpu=False is not a fallback here - it is the deployment target.
                _reader = easyocr.Reader(["fa"], gpu=False, verbose=False)
    return _reader


class EasyOcrRecognizer:
    name = "easyocr"

    def recognize(self, image: np.ndarray, *, allowlist: str | None = None) -> Recognition:
        if image is None or image.size == 0:
            return Recognition.empty()
        reader = _get_reader()
        rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        kwargs = {"detail": 1, "paragraph": False}
        if allowlist:
            kwargs["allowlist"] = allowlist
        with _infer_lock:
            results = reader.readtext(rgb, **kwargs)
        if not results:
            return Recognition.empty()

        # A field crop may still split into several boxes (e.g. a date around
        # its separators). Join left-to-right and weight confidence by length
        # so a stray one-character box cannot drag the score down.
        results.sort(key=lambda r: r[0][0][0])
        texts = [str(r[1]).strip() for r in results if str(r[1]).strip()]
        if not texts:
            return Recognition.empty()
        weights = [len(t) for t in texts]
        confs = [float(r[2]) for r in results if str(r[1]).strip()]
        weighted = sum(c * w for c, w in zip(confs, weights, strict=True)) / max(1, sum(weights))
        return Recognition(text=" ".join(texts), confidence=float(min(1.0, max(0.0, weighted))))
