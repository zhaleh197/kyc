"""End-to-end pipeline behaviour, using the fake detector and recogniser."""

from __future__ import annotations

import cv2
import numpy as np
import pytest

from kyc.core.config import OcrSettings, Settings
from kyc.core.errors import ImageQualityError
from kyc.core.schemas import Decision
from kyc.modules.ocr.pipeline import DocumentOcrPipeline

from .conftest import CARD_H, CARD_W, FakeDetector, SequenceRecognizer, make_card, photograph


def build(settings: Settings, values: dict, confidence: float = 0.9) -> tuple[DocumentOcrPipeline, SequenceRecognizer]:
    recognizer = SequenceRecognizer(values, confidence=confidence)
    pipeline = DocumentOcrPipeline(
        settings,
        detector_factory=lambda profile: FakeDetector(),
        recognizer_factory=lambda backend_id: recognizer,
    )
    return pipeline, recognizer


def test_clean_card_passes_and_normalises_every_field(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values)
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.PASS, [r.code for r in result.reasons]
    values = result.data["values"]
    assert values["national_id"] == "0012345679"     # persian digits -> ascii
    assert values["last_name"] == "محمدی"            # arabic yeh -> persian yeh
    assert values["birth_date"] == "1370/05/12"
    assert result.data["missing_fields"] == []
    assert result.elapsed_ms > 0


def test_derived_values_are_computed(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values)
    derived = pipeline.run(photograph(make_card())).data["derived"]

    assert derived["birth_date_gregorian"] == "1991-08-03"
    assert derived["national_id_valid"] is True
    assert derived["expired"] is False


def test_digit_fields_are_read_with_a_numeric_allowlist(relaxed_settings, good_values):
    """A digit-constrained alphabet is the cheapest accuracy win available,
    so the pipeline must actually pass the hint through."""
    pipeline, recognizer = build(relaxed_settings, good_values)
    pipeline.run(photograph(make_card()))

    assert recognizer.allowlists[0] == "0123456789"      # national_id
    assert recognizer.allowlists[1] is None              # first_name
    assert recognizer.allowlists[4] == "0123456789/"     # birth_date


def test_bad_national_id_checksum_fails(relaxed_settings, good_values):
    values = {**good_values, "national_id": "۰۰۱۲۳۴۵۶۷۸"}
    pipeline, _ = build(relaxed_settings, values)
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.FAIL
    assert "national_id_checksum" in {r.code for r in result.reasons}
    assert result.score <= 0.5


def test_expired_card_fails(relaxed_settings, good_values):
    values = {**good_values, "expiry_date": "۱۳۹۹/۰۵/۱۲"}
    pipeline, _ = build(relaxed_settings, values)
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.FAIL
    assert "card_expired" in {r.code for r in result.reasons}


def test_low_recognition_confidence_routes_to_review(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values, confidence=0.2)
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.REVIEW
    assert "field_low_confidence" in {r.code for r in result.reasons}


def test_undetected_required_field_fails_and_is_reported(relaxed_settings, good_values):
    from .conftest import FIELD_BOXES

    boxes = {k: v for k, v in FIELD_BOXES.items() if k != "idnumber"}
    recognizer = SequenceRecognizer(good_values)
    pipeline = DocumentOcrPipeline(
        relaxed_settings,
        detector_factory=lambda profile: FakeDetector(boxes),
        recognizer_factory=lambda backend_id: recognizer,
    )
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.FAIL
    codes = {r.code for r in result.reasons}
    assert "field_not_located" in codes
    assert "national_id" in result.data["missing_fields"]


def test_blurry_card_is_rejected_before_ocr(good_values):
    """A blurry card must not reach the recogniser: confident nonsense on an
    identity document is worse than an explicit failure."""
    settings = Settings(ocr=OcrSettings(min_sharpness=1e9, min_brightness=1.0, max_brightness=254.0))
    pipeline, recognizer = build(settings, good_values)
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.FAIL
    assert "card_blurry" in {r.code for r in result.reasons}
    assert recognizer.index == 0, "recogniser was called despite the quality gate"
    assert result.score == 0.0


def test_tiny_image_is_rejected(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values)
    with pytest.raises(ImageQualityError):
        pipeline.run(np.full((100, 160, 3), 200, np.uint8))


def test_portrait_and_card_crops_are_returned_on_request(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values)
    result = pipeline.run(photograph(make_card()), return_portrait=True, return_card=True)

    assert result.data["portrait_base64"]
    assert result.data["card_base64"]


def test_rotated_photo_is_straightened(relaxed_settings, good_values):
    """A card shot in portrait orientation must still read; the rectifier
    rotates it back to landscape."""
    scene = photograph(make_card())
    rotated = cv2.rotate(scene, cv2.ROTATE_90_COUNTERCLOCKWISE)
    pipeline, _ = build(relaxed_settings, good_values)
    result = pipeline.run(rotated)

    assert result.data["values"]["national_id"] == "0012345679"


def test_rectify_flag_controls_the_geometry_the_detector_sees(good_values):
    """Train and serve must agree on whether the card is flattened first.

    A detector trained on whole frames and served rectified crops (or the
    reverse) sees a distribution it never saw, so this is config, not a
    heuristic - and it needs to actually take effect.
    """
    from .conftest import FIELD_BOXES

    scene = photograph(make_card())
    seen: dict[str, tuple[int, int]] = {}

    class ShapeRecordingDetector(FakeDetector):
        def detect(self, image, conf_threshold, iou_threshold):
            seen["shape"] = image.shape[:2]
            return super().detect(image, conf_threshold, iou_threshold)

    for rectify_on in (True, False):
        settings = Settings(
            ocr=OcrSettings(
                rectify=rectify_on,
                min_sharpness=1.0,
                min_brightness=10.0,
                max_brightness=250.0,
                max_glare_ratio=1.0,
            )
        )
        recognizer = SequenceRecognizer(good_values)
        pipeline = DocumentOcrPipeline(
            settings,
            detector_factory=lambda profile: ShapeRecordingDetector(),
            # Bound as a default so each iteration keeps its own recogniser
            # rather than closing over the loop variable.
            recognizer_factory=lambda backend_id, r=recognizer: r,
        )
        pipeline.run(scene)

        if rectify_on:
            assert seen["shape"] == (CARD_H, CARD_W), "rectified run should hand over the flattened card"
        else:
            assert seen["shape"] == scene.shape[:2], "unrectified run should hand over the original frame"

    assert FIELD_BOXES  # layout fixture is what the fake detector reports
