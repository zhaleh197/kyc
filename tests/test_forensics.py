"""Document forensics (spec 06a): features, scoring, advisory-only decisions,
boxes on the submitted photo, and the case step."""

from __future__ import annotations

import io

import cv2
import numpy as np
import pytest

from kyc.core.config import ForensicsSettings, OcrSettings, Settings
from kyc.core.schemas import Decision, Severity
from kyc.modules.forensics.calibration import REGIONS, fit, region_score
from kyc.modules.forensics.pipeline import DocumentForensicsPipeline, editing_software
from kyc.modules.forensics.signals import PhotoMaps
from kyc.modules.ocr.orient import quarter_turn_matrix, rotate_bound_matrix
from kyc.modules.orchestrator.policy import UNAVAILABLE
from kyc.modules.orchestrator.service import CaseService

from .test_orchestrator import Fakes, full_case, result

BOX = (120, 80, 220, 200)


def jpeg_bytes(img: np.ndarray, quality: int = 90) -> bytes:
    return cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1].tobytes()


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    return cv2.imdecode(np.frombuffer(jpeg_bytes(img, quality), np.uint8), cv2.IMREAD_COLOR)


def card(seed: int = 0) -> np.ndarray:
    """A 'photographed card': textured background with sensor noise, saved
    twice as a phone and a messaging app would."""
    rng = np.random.default_rng(seed)
    base = cv2.GaussianBlur(rng.integers(90, 200, (320, 400, 3)).astype(np.uint8), (7, 7), 0)
    noisy = np.clip(base + rng.normal(0, 6, base.shape), 0, 255).astype(np.uint8)
    return jpeg(jpeg(noisy, 85), 85)


def pasted(img: np.ndarray, box=BOX) -> np.ndarray:
    """A noise-free, once-compressed patch pasted over `box`, then saved."""
    x1, y1, x2, y2 = box
    out = img.copy()
    out[y1:y2, x1:x2] = cv2.GaussianBlur(img[y1:y2, x1:x2], (9, 9), 0)
    return jpeg(out, 95)


def test_a_pasted_patch_moves_the_features_away_from_genuine():
    genuine = [PhotoMaps(card(s)).features(BOX)[0] for s in range(6)]
    edited = [PhotoMaps(pasted(card(s))).features(BOX)[0] for s in range(6)]
    # a smoothed patch has less high-frequency noise than the card around it
    assert max(e["noise"] for e in edited) < min(g["noise"] for g in genuine)


def test_digit_fields_are_measured_per_character():
    cells = PhotoMaps(card()).features((40, 40, 340, 80), cells=10)
    assert len(cells) == 10 and set(cells[0]) == {"noise", "ela", "grid", "clone"}
    assert PhotoMaps(card()).features((0, 0, 10, 4)) is None  # too small to judge


def test_a_character_copied_within_the_field_is_found_by_its_noise():
    field = (40, 40, 340, 80)  # 10 cells of 30 px
    genuine = card(3)
    edited = genuine.copy()
    edited[40:80, 220:250] = genuine[40:80, 70:100]  # copy cell 1 over cell 6
    edited = jpeg(edited, 92)
    before = max(c["clone"] for c in PhotoMaps(genuine).features(field, cells=10))
    after = max(c["clone"] for c in PhotoMaps(edited).features(field, cells=10))
    assert before < 0.5 < after


def calibration_from(genuine_photos, fpr=0.05) -> dict:
    rows = [{"portrait": PhotoMaps(p).features(BOX)} for p in genuine_photos]
    calib = fit(rows, fpr)
    calib["ela_quality"] = 90
    return calib


@pytest.fixture(scope="module")
def calib():
    return calibration_from([card(s) for s in range(30)])


def test_score_is_relative_to_the_genuine_reference(calib):
    assert set(calib["regions"]) == {"portrait"}  # only regions with enough rows
    s, feat = region_score(calib, "portrait", PhotoMaps(pasted(card(99))).features(BOX))
    assert s >= 1.0 and feat in {"noise", "ela", "grid"}


def pipeline(calib, **cfg) -> DocumentForensicsPipeline:
    return DocumentForensicsPipeline(Settings(forensics=ForensicsSettings(**cfg)), calibration=calib)


def test_an_edited_portrait_goes_to_review_never_to_fail(calib):
    out = pipeline(calib).run(jpeg_bytes(pasted(card(77)), 95), {"portrait": list(BOX)})
    assert out.decision is Decision.REVIEW
    assert [r.code for r in out.reasons] == ["portrait_region_inconsistent"]
    assert all(r.severity is not Severity.ERROR for r in out.reasons)  # F5: advisory only
    assert out.data["signals"][0]["name"] == "portrait"


def test_a_genuine_card_passes(calib):
    out = pipeline(calib).run(jpeg_bytes(card(55), 95), {"portrait": list(BOX)})
    assert out.decision is Decision.PASS, out.data["signals"]


def test_each_region_can_be_switched_off(calib):
    out = pipeline(calib, regions=[]).run(jpeg_bytes(pasted(card(77)), 95), {"portrait": list(BOX)})
    assert out.decision is Decision.PASS  # F4
    assert "forensics_no_regions" in {r.code for r in out.reasons}


def test_without_calibration_nothing_is_judged():
    out = pipeline({}).run(jpeg_bytes(card()), {"portrait": list(BOX)})
    assert out.decision is Decision.PASS
    assert [r.code for r in out.reasons] == ["forensics_not_calibrated"]


def test_editor_metadata_is_an_info_note_only(calib):
    from PIL import Image

    buf = io.BytesIO()
    exif = Image.Exif()
    exif[0x0131] = "Adobe Photoshop 25.0"
    Image.fromarray(card()[:, :, ::-1]).save(buf, "JPEG", quality=95, exif=exif)
    assert editing_software(buf.getvalue()) == "Adobe Photoshop 25.0"
    out = pipeline(calib).run(buf.getvalue(), {})
    info = [r for r in out.reasons if r.code == "editing_software_metadata"]
    assert info and info[0].severity is Severity.INFO
    assert editing_software(jpeg_bytes(card())) is None


def test_regions_follow_the_ocr_field_keys():
    from kyc.modules.ocr.profiles import get_profile

    keys = {f.key for f in get_profile(None).fields}
    assert set(REGIONS) <= keys


# ------------------------------------------------------- boxes on the source photo


@pytest.mark.parametrize("rotation", [90, 180, 270])
def test_quarter_turn_matrix_matches_cv2_rotate(rotation):
    img = np.zeros((30, 50), np.uint8)
    img[7, 11] = 255
    turned = cv2.rotate(img, {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180,
                              270: cv2.ROTATE_90_COUNTERCLOCKWISE}[rotation])
    x, y, _ = quarter_turn_matrix(50, 30, rotation) @ np.array([11, 7, 1.0])
    assert turned[int(round(y)), int(round(x))] == 255


def test_rotate_bound_matrix_keeps_the_centre_in_the_middle():
    m, (nw, nh) = rotate_bound_matrix(200, 100, 20)
    x, y = m @ np.array([100, 50, 1.0])
    assert abs(x - nw / 2) < 1 and abs(y - nh / 2) < 1


def test_ocr_reports_boxes_on_the_photo_as_submitted(good_values):
    from .test_pipeline import build

    settings = Settings(ocr=OcrSettings(min_sharpness=1.0, min_brightness=10.0, max_brightness=250.0,
                                        max_glare_ratio=1.0, max_image_side=600, auto_orient=False))
    rng = np.random.default_rng(0)
    photo = rng.integers(60, 200, (900, 1200, 3)).astype(np.uint8)  # downscaled by 2 before detection
    ocr, _ = build(settings, good_values)
    out = ocr.run(photo)
    assert out.data["source_size"] == [1200, 900]
    frame = out.data["fields"]["national_id"]["box"]
    source = out.data["source_boxes"]["national_id"]
    limits = [1200, 900, 1200, 900]  # clipped to the photo (the fake detector's boxes ignore its size)
    assert source == pytest.approx([min(2 * v, lim) for v, lim in zip(frame, limits, strict=True)], abs=2)
    assert "portrait" in out.data["source_boxes"]


# ---------------------------------------------------------------- case step


class FakeForensics:
    def __init__(self, outcome):
        self.outcome = outcome
        self.seen = None

    def run(self, data, boxes):
        self.seen = (data, boxes)
        if isinstance(self.outcome, Exception):
            raise self.outcome
        return self.outcome


def service(forensics):
    fakes = Fakes()
    return CaseService(Settings(), ocr=fakes, capture=fakes, antispoof=fakes, face_match=fakes, liveness=fakes,
                       forensics=forensics)


def test_a_forensic_warning_sends_the_case_to_a_human():
    from kyc.core.schemas import Reason

    flagged = result("forensics", Decision.REVIEW, Reason.warn("portrait_region_inconsistent", "x", "x"))
    fx = FakeForensics(flagged)
    svc = service(fx)
    verdict = svc.finalize(full_case(svc))
    assert verdict["verdict"] == "manual_review"
    assert {"step": "forensics", "code": "portrait_region_inconsistent"}.items() <= verdict["blocking_reasons"][0].items()
    assert fx.seen[0] == b"jpeg"  # the photo exactly as uploaded


def test_forensics_switched_off_does_not_block_approval():
    svc = service(None)
    assert svc.finalize(full_case(svc))["verdict"] == "approved"


def test_a_forensics_crash_keeps_the_upload_but_not_an_approval():
    svc = service(FakeForensics(RuntimeError("boom")))
    case = svc.create()
    svc.submit_document(case.case_id, b"jpeg")  # does not raise
    assert case.forensics == UNAVAILABLE
