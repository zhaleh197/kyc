"""Shared fixtures.

The fakes here stand in for the two trained artifacts (field detector and text
recogniser) so the whole pipeline - rectification, cropping, normalisation,
validation, decision - is covered without shipping model weights or real
identity documents into the repository.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kyc.core.config import OcrSettings, Settings
from kyc.modules.ocr.detector import Detection
from kyc.modules.ocr.profiles import IR_NATIONAL_CARD_FRONT
from kyc.modules.ocr.recognizers.base import Recognition

CARD_W, CARD_H = IR_NATIONAL_CARD_FRONT.rectified_size

# Field boxes on the rectified card, in the same coordinate space the real
# detector would emit. Roughly matches the layout of the Iranian card.
FIELD_BOXES: dict[str, tuple[int, int, int, int]] = {
    "img": (40, 120, 240, 400),
    "idnumber": (300, 120, 700, 175),
    "name": (300, 200, 700, 250),
    "lastname": (300, 275, 700, 325),
    "fathername": (300, 350, 700, 400),
    "birthday": (300, 425, 700, 475),
    "exp": (300, 500, 700, 550),
    "idcard": (0, 0, CARD_W - 1, CARD_H - 1),
}


def make_card(background: int = 235) -> np.ndarray:
    """A synthetic, sharp, well-exposed card with a dark block per field."""
    card = np.full((CARD_H, CARD_W, 3), background, np.uint8)
    for name, (x1, y1, x2, y2) in FIELD_BOXES.items():
        if name == "idcard":
            continue
        cv2.rectangle(card, (x1, y1), (x2, y2), (40, 40, 40), -1)
        # High-frequency detail so the sharpness gate sees a real photo, not a
        # flat fill.
        for x in range(x1 + 6, x2 - 6, 14):
            cv2.line(card, (x, y1 + 6), (x, y2 - 6), (220, 220, 220), 3)
    return card


def photograph(card: np.ndarray, frame: tuple[int, int] = (1400, 1050), tilt: int = 25) -> np.ndarray:
    """Place the card on a dark background with a slight perspective tilt,
    the way a phone photo would look."""
    scene = np.full((frame[1], frame[0], 3), 60, np.uint8)
    h, w = card.shape[:2]
    ox, oy = (frame[0] - w) // 2, (frame[1] - h) // 2
    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array(
        [[ox + tilt, oy], [ox + w - 1, oy + tilt], [ox + w - 1 - tilt, oy + h - 1], [ox, oy + h - 1 - tilt]],
        np.float32,
    )
    warped = cv2.warpPerspective(card, cv2.getPerspectiveTransform(src, dst), frame, borderValue=(60, 60, 60))
    mask = cv2.warpPerspective(
        np.full((h, w), 255, np.uint8), cv2.getPerspectiveTransform(src, dst), frame, borderValue=0
    )
    scene[mask > 0] = warped[mask > 0]
    return scene


class FakeDetector:
    """Returns the known layout, so tests exercise everything downstream of it."""

    def __init__(self, boxes: dict[str, tuple[int, int, int, int]] | None = None, confidence: float = 0.95):
        self.boxes = FIELD_BOXES if boxes is None else boxes
        self.confidence = confidence
        self.class_names = IR_NATIONAL_CARD_FRONT.class_names

    def detect(self, image, conf_threshold, iou_threshold):  # noqa: ARG002 - signature parity
        return [
            Detection(
                class_index=self.class_names.index(name) if name in self.class_names else 0,
                class_name=name,
                confidence=self.confidence,
                box=box,
            )
            for name, box in self.boxes.items()
        ]


class SequenceRecognizer:
    """Returns scripted values in the order the pipeline reads fields."""

    name = "sequence"

    def __init__(self, values: dict[str, str], confidence: float = 0.9):
        # Keyed by canonical field key; the pipeline reads profile.value_fields
        # in declaration order.
        self.order = [f.key for f in IR_NATIONAL_CARD_FRONT.value_fields]
        self.values = values
        self.confidence = confidence
        self.index = 0
        self.allowlists: list[str | None] = []

    def recognize(self, image, *, allowlist=None):  # noqa: ARG002
        self.allowlists.append(allowlist)
        # Wraps around so a single instance can serve several requests.
        key = self.order[self.index % len(self.order)]
        self.index += 1
        if key is None or key not in self.values:
            return Recognition.empty()
        return Recognition(text=self.values[key], confidence=self.confidence)


@pytest.fixture
def relaxed_settings() -> Settings:
    """Quality thresholds tuned for the synthetic fixture, so pipeline tests
    fail for pipeline reasons rather than for image-statistics reasons."""
    return Settings(
        ocr=OcrSettings(
            min_sharpness=1.0,
            min_brightness=10.0,
            max_brightness=250.0,
            max_glare_ratio=1.0,
            min_field_confidence=0.4,
        )
    )


@pytest.fixture
def good_values() -> dict[str, str]:
    return {
        "national_id": "۰۰۱۲۳۴۵۶۷۹",
        "first_name": "زهرا",
        "last_name": "محمدي",
        "father_name": "علي",
        "birth_date": "۱۳۷۰/۰۵/۱۲",
        "expiry_date": "۱۴۱۰/۰۵/۱۲",
    }
