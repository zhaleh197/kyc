"""Card orientation (kyc/modules/ocr/orient.py): skew estimation, layout
rules, and choosing the quarter turn - with a fake detector."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kyc.modules.ocr.detector import Detection
from kyc.modules.ocr.orient import estimate_skew, layout_score, orient, rotate_bound
from kyc.modules.ocr.profiles import IR_NATIONAL_CARD_FRONT as PROFILE

from .conftest import FIELD_BOXES, make_card


def lined_card() -> np.ndarray:
    """A card with strong horizontal text-like lines and a border."""
    card = np.full((400, 640, 3), 235, np.uint8)
    cv2.rectangle(card, (5, 5), (634, 394), (30, 30, 30), 4)
    for y in range(60, 360, 30):
        cv2.line(card, (200, y), (600, y), (40, 40, 40), 6)
    return card


@pytest.mark.parametrize("tilt", [-25, -10, 0, 8, 20])
def test_skew_is_recovered(tilt):
    tilted = rotate_bound(lined_card(), tilt)
    estimate = estimate_skew(cv2.cvtColor(tilted, cv2.COLOR_BGR2GRAY))
    # Rotating back by the estimate must level the card: estimate ~ -tilt.
    assert estimate == pytest.approx(-tilt, abs=2.0)


class LayoutDetector:
    """Reports the conftest layout only when the image is upright; on a
    turned image it reports the same boxes turned along with it - as a real
    detector trained with rotation augmentation would."""

    def __init__(self, upright_shape):
        self.h, self.w = upright_shape[:2]

    def _boxes_for(self, image):
        h, w = image.shape[:2]
        out = []
        for name, (x1, y1, x2, y2) in FIELD_BOXES.items():
            if (h, w) == (self.h, self.w) and image[0, 0, 0] != 7:  # upright marker
                box = (x1, y1, x2, y2)
            elif (h, w) == (self.h, self.w):  # 180: same shape, marked
                box = (self.w - x2, self.h - y2, self.w - x1, self.h - y1)
            else:  # sideways: x/y swapped
                box = (y1, x1, y2, x2)
            out.append(Detection(class_index=PROFILE.class_names.index(name), class_name=name, confidence=0.9, box=box))
        return out

    def __call__(self, image):
        return self._boxes_for(image)


def test_layout_rules_hold_on_the_upright_layout():
    located = {
        name: Detection(class_index=0, class_name=name, confidence=0.9, box=box) for name, box in FIELD_BOXES.items()
    }
    assert layout_score(located, PROFILE) == 1.0


def test_an_upright_card_is_left_alone():
    card = make_card()
    o = orient(card, LayoutDetector(card.shape), PROFILE)
    assert o.rotation == 0 and o.layout_score == 1.0 and o.image.shape == card.shape


def test_a_sideways_card_is_turned_upright():
    card = make_card()
    sideways = cv2.rotate(card, cv2.ROTATE_90_CLOCKWISE)
    o = orient(sideways, LayoutDetector(card.shape), PROFILE, min_skew_deg=90)  # isolate the quarter-turn step
    assert o.image.shape[:2] == card.shape[:2]
    assert o.layout_score == 1.0
