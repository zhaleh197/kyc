"""The Iraqi (Kurdistan) card profile and what it added to the pipeline:
per-profile recognisers and latin code fields."""

from __future__ import annotations

from dataclasses import replace

import pytest

from kyc.modules.ocr import normalize, profiles
from kyc.modules.ocr.pipeline import DocumentOcrPipeline
from kyc.modules.ocr.profiles import IR_NATIONAL_CARD_FRONT, FieldKind, get_profile

from .conftest import FakeDetector, SequenceRecognizer, make_card, photograph


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("L82931897", "L82931897"), (" l 8293-1897 ", "L8293-1897"), ("AB+", "AB+"), ("o-", "O-"), ("+AB", "AB+"), ("-O", "O-"),
     ("۴۲۱۰۹۴", "421094"), ("‏B‎", "B")],
)
def test_normalize_code(raw, expected):
    assert normalize.normalize_code(raw) == expected


def test_the_iraqi_card_is_registered_with_a_consistent_detector():
    profile = get_profile("iq_national_card_front")
    assert profile.country == "IQ"
    detector_classes = {f.detector_class for f in profile.fields}
    assert detector_classes == set(profile.class_names)
    for a, _, b in profile.layout:
        assert {a, b} <= detector_classes
    kinds = {f.key: f.kind for f in profile.fields}
    assert kinds["national_id"] is FieldKind.DIGITS
    assert kinds["document_number"] is FieldKind.CODE
    assert profile.text_backend == "ppocr_fa_ckb"


def test_a_profile_can_choose_its_own_recognisers(monkeypatch, relaxed_settings, good_values):
    custom = replace(IR_NATIONAL_CARD_FRONT, id="test_own_backends", text_backend="text_x", digit_backend="digit_x")
    monkeypatch.setitem(profiles._REGISTRY, custom.id, custom)
    asked = []
    recognizer = SequenceRecognizer(good_values, confidence=0.9)

    def factory(backend_id):
        asked.append(backend_id)
        return recognizer

    pipeline = DocumentOcrPipeline(relaxed_settings, detector_factory=lambda p: FakeDetector(),
                                   recognizer_factory=factory)
    result = pipeline.run(photograph(make_card()), custom.id)

    fields = result.data["fields"]
    assert fields["national_id"]["backend"] == "digit_x"
    assert fields["first_name"]["backend"] == "text_x"
    # The name-list scorer follows the profile's text recogniser as well.
    assert set(asked) == {"text_x", "digit_x"}


def test_a_rectifying_profile_still_maps_boxes_back_onto_the_photo(monkeypatch, relaxed_settings, good_values):
    """Forensics reads the submitted pixels, so the boxes must come back in
    photo coordinates even when the card was flattened first."""
    flat = replace(IR_NATIONAL_CARD_FRONT, id="test_rectified", rectify=True)
    monkeypatch.setitem(profiles._REGISTRY, flat.id, flat)
    pipeline = DocumentOcrPipeline(relaxed_settings, detector_factory=lambda p: FakeDetector(),
                                   recognizer_factory=lambda b: SequenceRecognizer(good_values, confidence=0.9))
    photo = photograph(make_card())
    result = pipeline.run(photo, flat.id)

    assert "card_border_not_found" not in [r.code for r in result.reasons]
    boxes = result.data["source_boxes"]
    assert boxes, "no boxes mapped back"
    h, w = photo.shape[:2]
    for x1, y1, x2, y2 in boxes.values():
        assert 0 <= x1 < x2 <= w and 0 <= y1 < y2 <= h
