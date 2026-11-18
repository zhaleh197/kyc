"""PaddleOCR text-line recogniser (PP-OCRv5 Arabic-script, mobile) on ONNX.

The model is PaddleOCR's, converted to ONNX by the RapidOCR project
(Apache-2.0); see models/manifest.yaml for its source and checksum. Nothing
from paddle or rapidocr is imported: the preprocessing and CTC decoding below
follow rapidocr's TextRecognizer, so this runs on onnxruntime alone.

On the real-card set (.data/id_cards_real, 14 photos) it read 36 of 42 names
on field crops, against 24 for easyocr, with no training. Two things to know:

  * Its output is in visual order (left to right as drawn), as PaddleOCR's is
    for Arabic script: it is turned back to reading order here.
  * It was trained on Arabic, so Persian-only letters come out weakly - «گ» is
    often read as «ک», «ژ» as «ز». The name list (ocr/lexicon) repairs first
    and father's names; last names keep the error.
"""

from __future__ import annotations

import math
import threading
from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

from .base import Recognition

INPUT_HEIGHT = 48
MIN_WIDTH = 320  # rapidocr's rec_img_shape [3, 48, 320]
BLANK_INDEX = 0


def to_reading_order(text: str) -> str:
    """Visual (left-to-right) order back to logical order for right-to-left
    script: reverse the line, then put runs of digits / Latin back the right
    way round. Enough for a single card field."""
    if not text:
        return text
    out: list[str] = []
    run: list[str] = []
    for ch in reversed(text):
        if ch.isascii() and ch.isalnum() or "٠" <= ch <= "٩" or "۰" <= ch <= "۹":
            run.append(ch)
            continue
        if run:
            out.extend(reversed(run))
            run = []
        out.append(ch)
    out.extend(reversed(run))
    return "".join(out)


class PpOcrRecognizer:
    """One ONNX session per model file, shared across requests."""

    _instances: dict[str, PpOcrRecognizer] = {}
    _lock = threading.Lock()

    def __init__(self, model_path: Path, providers: list[str], name: str = "ppocr_arabic"):
        if not model_path.exists():
            raise ModelNotAvailable(
                f"PaddleOCR recogniser not found at {model_path}. Download it as described in "
                f"models/manifest.yaml (ppocr_arabic_v5_rec.onnx)."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailable("onnxruntime is not installed") from exc

        self.name = name
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        meta = self.session.get_modelmeta().custom_metadata_map
        if "character" not in meta:
            raise ModelNotAvailable(f"{model_path.name} carries no character list in its metadata")
        # Model classes: blank, the characters in the file, then a space.
        self.charset = ["<blank>", *meta["character"].splitlines(), " "]
        self._infer_lock = threading.Lock()

    @classmethod
    def load(cls, model_path: Path, providers: list[str], name: str = "ppocr_arabic") -> PpOcrRecognizer:
        key = str(model_path)
        with cls._lock:
            if key not in cls._instances:
                cls._instances[key] = cls(model_path, providers, name)
            return cls._instances[key]

    def recognize(self, image: np.ndarray, *, allowlist: str | None = None) -> Recognition:
        if image is None or image.size == 0:
            return Recognition.empty()
        probs = self._probs(image)
        best = probs.argmax(axis=1)
        keep = np.ones(len(best), bool)
        keep[1:] = best[1:] != best[:-1]
        keep &= best != BLANK_INDEX
        if allowlist:
            keep &= np.array([self.charset[i] in allowlist for i in best])
        chars = [self.charset[i] for i in best[keep] if i < len(self.charset)]
        if not chars:
            return Recognition.empty()
        confidence = float(probs[np.arange(len(best)), best][keep].mean())
        return Recognition(text=to_reading_order("".join(chars)).strip(), confidence=confidence)

    def log_probs(self, image: np.ndarray) -> np.ndarray:
        """(T, C) log-probabilities over `self.charset`, in READING order (the
        frames are flipped), for scoring a list of names (ocr/lexicon)."""
        return np.log(self._probs(image)[::-1] + 1e-12)

    def _probs(self, image: np.ndarray) -> np.ndarray:
        blob = self._preprocess(image)
        with self._infer_lock:
            out = self.session.run(None, {self.input_name: blob})[0]
        return np.asarray(out)[0]  # (T, C) softmax

    @staticmethod
    def _preprocess(image: np.ndarray) -> np.ndarray:
        """BGR crop -> (1, 3, 48, W), as rapidocr's resize_norm_img."""
        if image.ndim == 2:
            image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        h, w = image.shape[:2]
        ratio = w / max(h, 1)
        width = max(MIN_WIDTH, int(INPUT_HEIGHT * ratio))
        resized_w = min(width, int(math.ceil(INPUT_HEIGHT * ratio)))
        resized = cv2.resize(image, (max(1, resized_w), INPUT_HEIGHT)).astype(np.float32)
        resized = (resized.transpose(2, 0, 1) / 255.0 - 0.5) / 0.5
        blob = np.zeros((3, INPUT_HEIGHT, width), np.float32)
        blob[:, :, : resized.shape[2]] = resized
        return blob[None]
