"""EasyOCR backend: join order for a field crop that splits into several boxes.

Uses a fake reader (readtext() returns scripted detections) so this tests the
join-order logic in isolation, without needing the real easyocr model.
"""

from __future__ import annotations

import numpy as np

from kyc.modules.ocr.recognizers import easyocr_backend


class FakeReader:
    """Returns pre-scripted easyocr-format detections regardless of input."""

    def __init__(self, detections):
        self.detections = detections

    def readtext(self, image, **kwargs):  # noqa: ARG002 - signature parity with easyocr.Reader
        return self.detections


def box_at(x: int) -> list[list[int]]:
    """A tiny box whose top-left x-coordinate is `x` - all the recognizer
    reads from a detection's bbox to decide join order."""
    return [[x, 0], [x + 10, 0], [x + 10, 10], [x, 10]]


def test_free_text_joins_right_to_left(monkeypatch):
    """Two word-boxes of a name must join in Persian reading order: the
    rightmost box (highest x) reads first. Sorting ascending by x - correct
    for Latin script - joined a real two-word name backwards:
    "کانی پان" came out as "پان کانی"."""
    # "پان" sits to the left of "کانی" in the image, as it would for RTL text
    # where the word read first is drawn on the right.
    detections = [
        (box_at(0), "پان", 0.9),
        (box_at(50), "کانی", 0.9),
    ]
    monkeypatch.setattr(easyocr_backend, "_get_reader", lambda: FakeReader(detections))

    recognizer = easyocr_backend.EasyOcrRecognizer()
    result = recognizer.recognize(np.zeros((10, 60, 3), np.uint8), allowlist=None)

    assert result.text == "کانی پان"


def test_digits_join_left_to_right(monkeypatch):
    """Date/national-id fragments must stay in ascending-x order: numerals
    render left-to-right even inside RTL text, unlike free-text words.
    `allowlist` is what distinguishes the two cases - profiles/iran.py never
    sets charset_hint on a FieldKind.TEXT field."""
    detections = [
        (box_at(0), "1370", 0.9),
        (box_at(50), "05", 0.9),
        (box_at(80), "12", 0.9),
    ]
    monkeypatch.setattr(easyocr_backend, "_get_reader", lambda: FakeReader(detections))

    recognizer = easyocr_backend.EasyOcrRecognizer()
    result = recognizer.recognize(np.zeros((10, 100, 3), np.uint8), allowlist="0123456789/")

    assert result.text == "1370 05 12"


def test_empty_detections_return_empty_recognition(monkeypatch):
    monkeypatch.setattr(easyocr_backend, "_get_reader", lambda: FakeReader([]))

    recognizer = easyocr_backend.EasyOcrRecognizer()
    result = recognizer.recognize(np.zeros((10, 10, 3), np.uint8))

    assert result.text == ""
    assert result.confidence == 0.0


def test_none_image_returns_empty_recognition_without_calling_the_reader(monkeypatch):
    def fail_if_called():
        raise AssertionError("recognize() must short-circuit on an empty image")

    monkeypatch.setattr(easyocr_backend, "_get_reader", fail_if_called)

    recognizer = easyocr_backend.EasyOcrRecognizer()

    assert recognizer.recognize(None).text == ""
    assert recognizer.recognize(np.zeros((0, 0, 3), np.uint8)).text == ""
