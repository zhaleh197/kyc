"""The transform returned by rectify must actually map source pixels onto the card.

This matters beyond tidiness: `scripts/rectify_dataset.py` moves training
labels through this exact matrix. If it disagreed with the warp by even a few
pixels, every annotation in the training set would be quietly offset and the
detector would learn to aim slightly wrong.
"""

from __future__ import annotations

import cv2
import numpy as np

from kyc.modules.ocr import rectify
from kyc.modules.ocr.profiles import IR_NATIONAL_CARD_FRONT as PROFILE

from .conftest import FIELD_BOXES, make_card, photograph


def test_transform_agrees_with_the_warped_image():
    """Project the card's own corners and check they land on the card corners."""
    scene = photograph(make_card())
    card, _, matrix = rectify.rectify_with_transform(scene, PROFILE.rectified_size, PROFILE.aspect_ratio)

    quad = rectify.find_card_quad(scene, PROFILE.aspect_ratio)
    assert quad is not None, "fixture should present a findable card outline"

    projected = cv2.perspectiveTransform(quad.reshape(-1, 1, 2).astype(np.float32), matrix).reshape(4, 2)
    width, height = PROFILE.rectified_size
    expected = np.array([[0, 0], [width - 1, 0], [width - 1, height - 1], [0, height - 1]], np.float32)
    assert np.allclose(projected, expected, atol=1.0)
    assert (card.shape[1], card.shape[0]) == PROFILE.rectified_size


def test_transform_puts_a_known_field_where_it_belongs():
    """A field drawn at a known place on the card must, after going through the
    scene and back, land near that same place on the rectified card."""
    card_image = make_card()
    card_h, card_w = card_image.shape[:2]
    scene = photograph(card_image)

    _, _, matrix = rectify.rectify_with_transform(scene, PROFILE.rectified_size, PROFILE.aspect_ratio)

    # Where the portrait sits on the original card, in scene coordinates we do
    # not know - so instead check the round trip is self-consistent by mapping
    # the card's own idnumber box fractionally.
    x1, y1, x2, y2 = FIELD_BOXES["idnumber"]
    fractional = (x1 / card_w, y1 / card_h, x2 / card_w, y2 / card_h)

    out_w, out_h = PROFILE.rectified_size
    expected = (fractional[0] * out_w, fractional[1] * out_h, fractional[2] * out_w, fractional[3] * out_h)

    # Find the same box in the scene by warping the card corners forward.
    quad = rectify.find_card_quad(scene, PROFILE.aspect_ratio)
    src = np.array([[0, 0], [card_w - 1, 0], [card_w - 1, card_h - 1], [0, card_h - 1]], np.float32)
    to_scene = cv2.getPerspectiveTransform(src, quad.astype(np.float32))
    in_scene = rectify.transform_box(to_scene, FIELD_BOXES["idnumber"], (scene.shape[1], scene.shape[0]))
    assert in_scene is not None

    on_card = rectify.transform_box(matrix, in_scene, PROFILE.rectified_size)
    assert on_card is not None
    # Generous tolerance: the classical corner finder is a few pixels off, and
    # an AABB of a warped box is slightly larger than the box itself.
    assert np.allclose(on_card, expected, atol=25.0), f"{on_card} vs {expected}"


def test_transform_box_rejects_a_box_outside_the_frame():
    matrix = rectify.scale_matrix((100, 100), (100, 100))
    assert rectify.transform_box(matrix, (-50.0, -50.0, -10.0, -10.0), (100, 100)) is None


def test_scale_matrix_maps_corners():
    matrix = rectify.scale_matrix((200, 100), (400, 300))
    box = rectify.transform_box(matrix, (0.0, 0.0, 100.0, 50.0), (400, 300))
    assert box == (0.0, 0.0, 200.0, 150.0)


def test_rotate_matrix_matches_cv2_rotation():
    image = np.zeros((60, 100, 3), np.uint8)
    image[5:15, 80:95] = 255  # a marker in the top-right

    rotated = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    matrix = rectify.rotate_cw_matrix((100, 60))
    mapped = rectify.transform_box(matrix, (80.0, 5.0, 95.0, 15.0), (rotated.shape[1], rotated.shape[0]))
    assert mapped is not None

    x1, y1, x2, y2 = (int(round(v)) for v in mapped)
    patch = rotated[y1:y2, x1:x2]
    assert patch.size > 0
    assert patch.mean() > 200, "the mapped box should land on the white marker"


def test_rectify_still_returns_two_values():
    """The two-value form is what the pipeline uses; it must keep working."""
    card, reasons = rectify.rectify(photograph(make_card()), PROFILE.rectified_size, PROFILE.aspect_ratio)
    assert (card.shape[1], card.shape[0]) == PROFILE.rectified_size
    assert isinstance(reasons, list)
