"""Backend registry.

Backends are resolved by id at request time and cached per process. A backend
that cannot be constructed (missing package, missing weights) raises rather
than silently degrading - a KYC pipeline that quietly stops reading a field is
worse than one that fails loudly.
"""

from __future__ import annotations

import threading

from kyc.core.config import Settings
from kyc.core.errors import BackendNotInstalled

from .base import Recognition, TextRecognizer

_cache: dict[str, TextRecognizer] = {}
_lock = threading.Lock()

# Trained CRNN artifacts expected in models/. `digits` is a separate model with
# a 10-symbol alphabet; see the module docstring in crnn_onnx.py.
CRNN_ARTIFACTS = {
    "crnn_onnx": ("crnn_fa_text.onnx", "crnn_fa_text.charset.txt"),
    "crnn_onnx_digits": ("crnn_fa_digits.onnx", "crnn_fa_digits.charset.txt"),
}


def get_recognizer(backend_id: str, settings: Settings) -> TextRecognizer:
    with _lock:
        existing = _cache.get(backend_id)
        if existing is not None:
            return existing
        recognizer = _build(backend_id, settings)
        _cache[backend_id] = recognizer
        return recognizer


def _build(backend_id: str, settings: Settings) -> TextRecognizer:
    if backend_id == "easyocr":
        from .easyocr_backend import EasyOcrRecognizer

        return EasyOcrRecognizer()

    if backend_id in CRNN_ARTIFACTS:
        from .crnn_onnx import CrnnOnnxRecognizer

        model_file, charset_file = CRNN_ARTIFACTS[backend_id]
        return CrnnOnnxRecognizer.load(
            settings.model_path(model_file),
            settings.model_path(charset_file),
            settings.onnx_providers,
            name=backend_id,
        )

    known = ", ".join(["easyocr", *CRNN_ARTIFACTS])
    raise BackendNotInstalled(f"Unknown recognition backend {backend_id!r}. Known backends: {known}")


def reset_cache() -> None:
    """Used by tests; also handy after swapping a model file at runtime."""
    with _lock:
        _cache.clear()


__all__ = ["Recognition", "TextRecognizer", "get_recognizer", "reset_cache", "CRNN_ARTIFACTS"]
