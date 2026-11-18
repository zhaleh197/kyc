"""PaddleOCR ONNX recogniser: the parts that need no model file."""

from __future__ import annotations

import numpy as np

from kyc.modules.ocr.recognizers.ppocr_onnx import INPUT_HEIGHT, MIN_WIDTH, PpOcrRecognizer, to_reading_order


def visual(text: str) -> str:
    """How the model emits a right-to-left line: as drawn, left to right."""
    return text[::-1]


def test_visual_order_is_turned_back_to_reading_order():
    assert to_reading_order(visual("منبری")) == "منبری"
    assert to_reading_order(visual("گویلی کیلانه")) == "گویلی کیلانه"


def test_digit_and_latin_runs_keep_their_direction():
    # Digits and Latin are drawn left to right even inside a right-to-left line.
    # «شماره 1403» in a right-to-left line is drawn as: 1403, space, شماره.
    assert to_reading_order("1403 " + visual("شماره")) == "شماره 1403"
    assert to_reading_order("AB " + visual("سلام")) == "سلام AB"


def test_preprocess_keeps_height_and_pads_to_the_minimum_width():
    blob = PpOcrRecognizer._preprocess(np.full((20, 40, 3), 255, np.uint8))
    assert blob.shape == (1, 3, INPUT_HEIGHT, MIN_WIDTH)
    assert np.isclose(blob[0, :, :, 0].mean(), 1.0)  # white -> +1
    assert np.allclose(blob[0, :, :, -1], 0.0)  # padding


def test_wide_crops_are_not_squeezed():
    blob = PpOcrRecognizer._preprocess(np.zeros((30, 600, 3), np.uint8))
    assert blob.shape[3] == 960  # 48 * 600/30
