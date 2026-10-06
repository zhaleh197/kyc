"""CRNN + CTC recogniser served from ONNX on CPU.

This is the production path. The model is trained in
`training/kaggle/train_crnn.py` on a Kaggle GPU and exported to ONNX; the
charset that was used for training is stored next to the weights as a UTF-8
text file, one symbol per line, and index 0 is reserved for the CTC blank.

Two instances are typically registered: one over the full Persian charset for
names, and one over `0123456789/` for the national id and dates. Constraining
the alphabet is the single cheapest accuracy win available - a digit model
cannot hallucinate a letter into a national id.
"""

from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np

from kyc.core.errors import ModelNotAvailable

from .base import Recognition

BLANK_INDEX = 0
# Fallback only. The real height is read from each model's own ONNX input
# shape (see __init__) - models trained from scratch here use 32, but a model
# fine-tuned from EasyOCR's pretrained weights (training/kaggle/train_crnn.py
# --finetune-from) inherits their convention of 64.
DEFAULT_INPUT_HEIGHT = 32


def load_charset(path: Path) -> list[str]:
    """Index 0 is the CTC blank; the file provides indices 1..N."""
    if not path.exists():
        raise ModelNotAvailable(f"Charset file not found at {path}")
    symbols = path.read_text(encoding="utf-8").splitlines()
    return ["<blank>"] + [s for s in symbols if s != ""]


def ctc_greedy_decode(logits: np.ndarray, charset: list[str]) -> tuple[str, float]:
    """Best-path CTC decode of a (T, C) logit matrix.

    Confidence is the mean probability of the chosen symbol across the
    timesteps that actually emitted a character - averaging over blanks would
    inflate the score on short text in a wide crop.
    """
    probs = _softmax(logits)
    best = probs.argmax(axis=1)
    peak = probs[np.arange(probs.shape[0]), best]

    chars: list[str] = []
    kept: list[float] = []
    previous = -1
    for t, index in enumerate(best):
        index = int(index)
        if index != previous and index != BLANK_INDEX and index < len(charset):
            chars.append(charset[index])
            kept.append(float(peak[t]))
        previous = index

    text = "".join(chars)
    confidence = float(np.mean(kept)) if kept else 0.0
    return text, confidence


def _softmax(x: np.ndarray) -> np.ndarray:
    shifted = x - x.max(axis=-1, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=-1, keepdims=True)


class CrnnOnnxRecognizer:
    """One ONNX CRNN. Sessions are cached per model path."""

    _instances: dict[str, CrnnOnnxRecognizer] = {}
    _lock = threading.Lock()

    def __init__(self, model_path: Path, charset_path: Path, providers: list[str], name: str = "crnn_onnx"):
        if not model_path.exists():
            raise ModelNotAvailable(
                f"CRNN model not found at {model_path}. Train it with "
                f"training/kaggle/train_crnn.py and copy the exported .onnx into models/."
            )
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelNotAvailable("onnxruntime is not installed") from exc

        self.name = name
        self.session = ort.InferenceSession(str(model_path), providers=providers)
        self.input_name = self.session.get_inputs()[0].name
        self.charset = load_charset(charset_path)
        input_shape = self.session.get_inputs()[0].shape
        # Fixed-width exports report an int here; dynamic ones report a string.
        width = input_shape[3]
        self.fixed_width: int | None = width if isinstance(width, int) else None
        # Read off the graph rather than assuming: a model fine-tuned from
        # EasyOCR's pretrained weights was trained at height 64, not the 32
        # this project's from-scratch models use, and there is no other
        # signal here for which convention a given .onnx file follows.
        height = input_shape[2]
        self.input_height: int = height if isinstance(height, int) else DEFAULT_INPUT_HEIGHT

    @classmethod
    def load(
        cls,
        model_path: Path,
        charset_path: Path,
        providers: list[str],
        name: str = "crnn_onnx",
    ) -> CrnnOnnxRecognizer:
        key = f"{model_path}|{charset_path}"
        with cls._lock:
            instance = cls._instances.get(key)
            if instance is None:
                instance = cls(model_path, charset_path, providers, name)
                cls._instances[key] = instance
            return instance

    def recognize(self, image: np.ndarray, *, allowlist: str | None = None) -> Recognition:
        if image is None or image.size == 0:
            return Recognition.empty()
        blob = self._preprocess(image)
        logits = self.session.run(None, {self.input_name: blob})[0]
        logits = self._to_time_major(logits)
        if allowlist:
            logits = self._mask_to_allowlist(logits, allowlist)
        text, confidence = ctc_greedy_decode(logits, self.charset)
        return Recognition(text=text, confidence=confidence)

    def _preprocess(self, image: np.ndarray) -> np.ndarray:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        h, w = gray.shape[:2]
        target_h = self.input_height
        scale = target_h / max(h, 1)
        new_w = max(8, int(round(w * scale)))
        if self.fixed_width:
            new_w = min(new_w, self.fixed_width)
        resized = cv2.resize(gray, (new_w, target_h), interpolation=cv2.INTER_CUBIC)
        if self.fixed_width and new_w < self.fixed_width:
            # Pad on the right with white; the model was trained on white
            # background crops, so black padding would look like ink.
            pad = np.full((target_h, self.fixed_width - new_w), 255, np.uint8)
            resized = np.hstack([resized, pad])
        normalized = (resized.astype(np.float32) / 127.5) - 1.0
        return normalized[None, None]  # NCHW with C=1

    @staticmethod
    def _to_time_major(logits: np.ndarray) -> np.ndarray:
        """Accept (T, N, C) or (N, T, C) and return (T, C) for batch size 1."""
        arr = np.asarray(logits)
        if arr.ndim == 3:
            if arr.shape[1] == 1:      # (T, N=1, C)
                return arr[:, 0, :]
            if arr.shape[0] == 1:      # (N=1, T, C)
                return arr[0]
        if arr.ndim == 2:
            return arr
        raise ModelNotAvailable(f"Unexpected CRNN output shape {arr.shape}")

    def _mask_to_allowlist(self, logits: np.ndarray, allowlist: str) -> np.ndarray:
        """Force the decoder to pick only permitted symbols.

        Cheaper and stricter than filtering the decoded string afterwards: a
        banned symbol can never win a timestep, so the runner-up character is
        emitted instead of the position being dropped.
        """
        permitted = set(allowlist)
        mask = np.full(len(self.charset), -np.inf, dtype=np.float32)
        mask[BLANK_INDEX] = 0.0
        for index, symbol in enumerate(self.charset):
            if index != BLANK_INDEX and symbol in permitted:
                mask[index] = 0.0
        width = min(logits.shape[1], mask.shape[0])
        out = logits.copy()
        out[:, :width] += mask[:width]
        return out
