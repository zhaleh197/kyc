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

        # A field crop may still split into several boxes (e.g. two words of
        # a name, or a date around its separators). Free Persian/Arabic text
        # is right-to-left - the box that reads FIRST sits at the largest x -
        # so sorting ascending (correct for Latin script) joined a two-word
        # name backwards ("کانی پان" -> "پان کانی"). Digits and dates go the
        # other way: numerals render left-to-right even inside RTL text, so
        # day/month/year boxes must stay in ascending x order. `allowlist` is
        # only ever passed for the digit/date fields (see profiles/iran.py -
        # no FieldSpec of kind TEXT sets charset_hint), so its presence is
        # what tells the two cases apart here.
        results.sort(key=lambda r: r[0][0][0], reverse=allowlist is None)
        texts = [str(r[1]).strip() for r in results if str(r[1]).strip()]
        if not texts:
            return Recognition.empty()
        weights = [len(t) for t in texts]
        confs = [float(r[2]) for r in results if str(r[1]).strip()]
        weighted = sum(c * w for c, w in zip(confs, weights, strict=True)) / max(1, sum(weights))
        return Recognition(text=" ".join(texts), confidence=float(min(1.0, max(0.0, weighted))))
