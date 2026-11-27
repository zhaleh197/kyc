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


def test_a_card_image_without_background_is_not_sent_to_a_person(monkeypatch, relaxed_settings, good_values):
    """A scan or a tight crop has no card edge to find. The field detector's
    own card box fills the frame, so the card was found: information only."""
    flat = replace(IR_NATIONAL_CARD_FRONT, id="test_flat_scan", rectify=True)
    monkeypatch.setitem(profiles._REGISTRY, flat.id, flat)
    recognizer = SequenceRecognizer(good_values, confidence=0.9)
    pipeline = DocumentOcrPipeline(relaxed_settings, detector_factory=lambda p: FakeDetector(),
                                   recognizer_factory=lambda b: recognizer)
    result = pipeline.run(make_card(), flat.id)

    border = [r for r in result.reasons if r.code == "card_border_not_found"]
    assert border, "the scan has no outline, so the reason should still be reported"
    assert all(r.severity.value == "info" for r in border)
    assert result.decision.value == "pass", [(r.code, r.severity.value) for r in result.reasons]


def test_the_iraqi_card_keeps_arabic_letters_in_names():
    keep = get_profile("iq_national_card_front").keep_letters
    assert normalize.normalize_name("فاطمة", keep) == "فاطمة"
    assert normalize.normalize_name("فاطمة") == "فاطمه"  # Iranian cards still fold
    assert normalize.normalize_name("أحمد علي", keep) == "أحمد علی"


def test_text_lines_splits_two_lines_that_touch():
    import numpy as np

    from kyc.modules.ocr.pipeline import text_lines

    crop = np.full((100, 200, 3), 230, np.uint8)
    crop[10:48, 20:180] = 30   # upper line
    crop[48:90, 40:160] = 30   # lower line, no blank row between
    crop[48, 20:180] = 230     # only a thinner row marks the join
    lines = text_lines(crop)
    assert len(lines) == 2
    assert lines[0].shape[0] < 70 and lines[1].shape[0] < 70


def test_only_short_lists_are_matched_line_by_line():
    profile = get_profile("iq_national_card_front")
    per_line = {f.key for f in profile.fields if f.lexicon_per_line}
    assert per_line == {"sex"}
    assert not any(f.lexicon_per_line for f in IR_NATIONAL_CARD_FRONT.fields)


def test_a_label_colon_is_cut_off_the_name_crop():
    import numpy as np

    from kyc.modules.ocr.pipeline import trim_label_colon

    crop = np.full((70, 140, 3), 235, np.uint8)
    crop[20:50, 10:100] = 30          # the name
    crop[22:26, 126:130] = 60         # colon, upper dot
    crop[38:42, 126:130] = 60         # colon, lower dot
    crop[66:70, 126:130] = 60         # top of the next row's colon
    out = trim_label_colon(crop)
    assert 100 <= out.shape[1] < 126

    plain = crop.copy()
    plain[:, 110:] = 235
    assert trim_label_colon(plain).shape == plain.shape
    # Dots inside the text (a letter's dots) are left alone.
    inside = plain.copy()
    inside[22:26, 60:64] = 235
    inside[5:9, 60:64] = 60
    inside[12:16, 60:64] = 60
    assert trim_label_colon(inside).shape == inside.shape


def test_only_the_iraqi_card_trims_label_colons():
    assert get_profile("iq_national_card_front").trim_label_colon
    assert not IR_NATIONAL_CARD_FRONT.trim_label_colon


def test_the_tail_of_the_row_above_is_cut_off_a_name_crop():
    import numpy as np

    from kyc.modules.ocr.pipeline import drop_neighbour_rows

    crop = np.full((70, 140, 3), 235, np.uint8)
    crop[0:8, 10:30] = 30       # the tail of a letter in the row above
    crop[20:50, 10:120] = 30    # the name
    crop[64:70, 50:70] = 30     # the top of the row below
    out = drop_neighbour_rows(crop)
    assert 30 <= out.shape[0] < 70
    gray = out[..., 0]
    assert (gray[:3] > 200).all() and (gray[-3:] > 200).all()

    single = np.full((70, 140, 3), 235, np.uint8)
    single[20:50, 10:120] = 30
    assert drop_neighbour_rows(single).shape == single.shape


ROWS = ("name", "father", "grandfather", "surname", "mother", "mother_grandfather", "sex")


def _row_box(label, row, conf=0.9, pitch=48, top=250):
    from kyc.modules.ocr.detector import Detection

    y = top + row * pitch
    return Detection(0, label, conf, (680, y - 22, 785, y + 22))


def test_boxes_on_their_own_rows_stay_as_they_are():
    from kyc.modules.ocr.rows import place_rows

    located = {k: _row_box(k, i) for i, k in enumerate(ROWS)}
    placed = place_rows(located, ROWS)
    assert placed.moved == [] and placed.empty == []
    assert placed.located == located


def test_a_card_without_surname_gets_its_mother_name_back():
    """The detector never saw a card with an empty surname row: it labels the
    mother's row "surname" and finds no mother. The row it sits on wins."""
    from kyc.modules.ocr.rows import place_rows

    located = {k: _row_box(k, i) for i, k in enumerate(ROWS[:3])}
    located["surname"] = _row_box("surname", 4, conf=0.5)  # on the mother's row
    located["mother_grandfather"] = _row_box("mother_grandfather", 5)
    located["sex"] = _row_box("sex", 6)
    located["portrait"] = _row_box("portrait", 0)  # not a row field: kept
    placed = place_rows(located, ROWS)

    assert placed.moved == [("surname", "mother")]
    assert placed.located["mother"].box == located["surname"].box
    assert "surname" not in placed.located and "portrait" in placed.located
    assert placed.empty == ["surname"]
    x1, y1, x2, y2 = placed.boxes["surname"]
    assert y1 < 250 + 3 * 48 < y2


def test_a_stray_box_between_rows_is_dropped():
    from kyc.modules.ocr.rows import place_rows

    located = {k: _row_box(k, i) for i, k in enumerate(ROWS) if k != "surname"}
    located["surname"] = _row_box("surname", 3.5, conf=0.2)  # half a row off
    placed = place_rows(located, ROWS)
    assert "surname" not in placed.located and placed.empty == ["surname"]
    assert placed.located["mother"] == located["mother"]


def test_too_few_rows_change_nothing():
    from kyc.modules.ocr.rows import place_rows

    located = {"name": _row_box("surname", 0), "father": _row_box("father", 1)}
    assert place_rows(located, ROWS).located == located


def test_an_empty_optional_row_is_information_not_a_failure():
    from kyc.core.schemas import Severity

    profile = get_profile("iq_national_card_front")
    spec = {f.key: f for f in profile.fields}
    surname = DocumentOcrPipeline._blank_row(spec["last_name"])
    assert surname.severity is Severity.INFO and surname.message_fa == "این کارت «لقب» ندارد"
    assert DocumentOcrPipeline._blank_row(spec["mother_name"]).severity is Severity.ERROR


def test_the_iraqi_card_places_rows_and_rereads_names():
    profile = get_profile("iq_national_card_front")
    assert profile.row_order == ROWS and set(ROWS) <= set(profile.class_names)
    assert profile.reread_blurred
    assert not IR_NATIONAL_CARD_FRONT.row_order and not IR_NATIONAL_CARD_FRONT.reread_blurred
