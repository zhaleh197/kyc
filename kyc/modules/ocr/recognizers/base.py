"""Recognition backend interface.

Two backends ship today and they answer different needs:

  easyocr    - pretrained, works out of the box, ~1-2s per card on CPU.
               Good enough to get the pipeline running and to bootstrap a
               labelled dataset.
  crnn_onnx  - a small CRNN trained on Kaggle for exactly these fields.
               10-20x faster on CPU and far more accurate on Persian digits,
               but only exists once you have trained it.

Field-level backend selection matters: digits and names have almost nothing in
common, and a digit-only recogniser with a 10-symbol alphabet beats a general
model on the national id every time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class Recognition:
    text: str
    confidence: float  # 0..1

    @classmethod
    def empty(cls) -> Recognition:
        return cls(text="", confidence=0.0)


@runtime_checkable
class TextRecognizer(Protocol):
    name: str

    def recognize(self, image: np.ndarray, *, allowlist: str | None = None) -> Recognition:
        """Read a single tightly-cropped text line. `image` is BGR uint8."""
        ...
