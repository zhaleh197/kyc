from __future__ import annotations

import cv2
import numpy as np

from kyc.modules.ocr import rectify
from kyc.modules.ocr.profiles import IR_NATIONAL_CARD_FRONT as PROFILE

from .conftest import make_card, photograph


def test_card_outline_is_found_in_a_tilted_photo():
    quad = rectify.find_card_quad(photograph(make_card()), PROFILE.aspect_ratio)
    assert quad is not None
    assert quad.shape == (4, 2)
    aspect = rectify._quad_aspect(quad)
    assert abs(aspect - PROFILE.aspect_ratio) < PROFILE.aspect_ratio * 0.45


def test_corners_are_ordered_clockwise_from_top_left():
    scrambled = np.array([[100, 200], [10, 20], [100, 20], [10, 200]], np.float32)
    tl, tr, br, bl = rectify.order_corners(scrambled)
    assert tuple(tl) == (10, 20)
    assert tuple(tr) == (100, 20)
    assert tuple(br) == (100, 200)
    assert tuple(bl) == (10, 200)


def test_rectified_card_matches_the_profile_size():
    card, _ = rectify.rectify(photograph(make_card()), PROFILE.rectified_size, PROFILE.aspect_ratio)
    assert (card.shape[1], card.shape[0]) == PROFILE.rectified_size


def test_borderless_scan_still_works_but_warns():
    """A tightly cropped scan has no visible outline. That is a warning, not a
    failure - plenty of users upload exactly this."""
    card, reasons = rectify.rectify(make_card(), PROFILE.rectified_size, PROFILE.aspect_ratio)
    assert (card.shape[1], card.shape[0]) == PROFILE.rectified_size
    assert "card_border_not_found" in {r.code for r in reasons}


def test_portrait_orientation_is_rotated_back():
    rotated = cv2.rotate(photograph(make_card()), cv2.ROTATE_90_CLOCKWISE)
    card, _ = rectify.rectify(rotated, PROFILE.rectified_size, PROFILE.aspect_ratio)
    assert card.shape[1] > card.shape[0]


def test_quality_gate_flags_a_blurred_card():
    blurred = cv2.GaussianBlur(make_card(), (31, 31), 0)
    reasons, metrics = rectify.assess_quality(
        blurred, min_sharpness=500.0, min_brightness=10.0, max_brightness=250.0, max_glare_ratio=1.0
    )
    assert "card_blurry" in {r.code for r in reasons}
    assert metrics["sharpness"] < 500.0


def test_quality_gate_flags_dark_and_bright_cards():
    dark = (make_card() * 0.1).astype(np.uint8)
    reasons, _ = rectify.assess_quality(
        dark, min_sharpness=0.0, min_brightness=45.0, max_brightness=215.0, max_glare_ratio=1.0
    )
    assert "card_too_dark" in {r.code for r in reasons}

    bright = np.full_like(make_card(), 240)
    reasons, _ = rectify.assess_quality(
        bright, min_sharpness=0.0, min_brightness=45.0, max_brightness=215.0, max_glare_ratio=1.0
    )
    assert "card_too_bright" in {r.code for r in reasons}


def test_glare_is_reported_as_a_warning():
    card = make_card()
    cv2.circle(card, (500, 300), 160, (255, 255, 255), -1)
    reasons, metrics = rectify.assess_quality(
        card, min_sharpness=0.0, min_brightness=10.0, max_brightness=250.0, max_glare_ratio=0.02
    )
    assert "card_glare" in {r.code for r in reasons}
    assert metrics["glare_ratio"] > 0.02
