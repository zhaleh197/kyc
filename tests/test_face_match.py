"""Face match (spec 03): ID-card portrait vs a server-held selfie.

Detector and embedder are fakes, so no weights or real faces are needed.
"""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.core.config import FaceMatchSettings, Settings
from kyc.core.face.detector import Face
from kyc.core.schemas import Decision
from kyc.modules.face_capture.session import Capture
from kyc.modules.face_match.pipeline import CaptureNotFound, FaceMatchPipeline, _rotate
from kyc.modules.face_match.router import get_pipeline


def face(box=(60, 40, 180, 200), score=0.9) -> Face:
    x1, y1, x2, y2 = box
    cx, cy, s = (x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 120
    kps = np.array(
        [[cx - 25 * s, cy - 20 * s], [cx + 25 * s, cy - 20 * s], [cx, cy], [cx - 20 * s, cy + 25 * s], [cx + 20 * s, cy + 25 * s]],
        np.float32,
    )
    return Face(box=tuple(float(v) for v in box), score=score, kps=kps)


def portrait() -> np.ndarray:
    return np.random.default_rng(1).integers(0, 255, (240, 180, 3)).astype(np.uint8)


def capture(capture_id="cap-1") -> Capture:
    selfie = np.random.default_rng(2).integers(0, 255, (480, 640, 3)).astype(np.uint8)
    f = face((240, 130, 400, 350))
    return Capture(
        capture_id=capture_id, session_id="s", jpeg=cv2.imencode(".jpg", selfie)[1].tobytes(),
        face_box=f.box, kps=f.kps, metrics={}, score=0.9,
    )


class FakeDetector:
    """Finds `faces` only from the `found_on_call`-th call on (to test the
    rotation fallback); records how many attempts were made."""

    def __init__(self, faces, found_on_call=1):
        self.faces = faces
        self.found_on_call = found_on_call
        self.calls = 0

    def detect(self, image, conf_threshold=0.5):  # noqa: ARG002
        self.calls += 1
        return list(self.faces) if self.calls >= self.found_on_call else []


class FakeEmbedder:
    """Reference vector first, then the selfie vector, alternating."""

    def __init__(self, similarity: float):
        angle = np.arccos(np.clip(similarity, -1, 1))
        self.vectors = [np.array([1.0, 0.0]), np.array([np.cos(angle), np.sin(angle)])]
        self.calls = 0

    def embed(self, image, f):  # noqa: ARG002
        vec = self.vectors[self.calls % 2]
        self.calls += 1
        return vec


def build(similarity=0.7, faces=None, found_on_call=1, captures=None, **cfg):
    store = {"cap-1": capture()} if captures is None else captures
    det = FakeDetector([face()] if faces is None else faces, found_on_call)
    emb = FakeEmbedder(similarity)
    pipeline = FaceMatchPipeline(
        Settings(face_match=FaceMatchSettings(**cfg)),
        capture_lookup=store.get,
        detector_factory=lambda: det,
        embedder_factory=lambda: emb,
    )
    return pipeline, det


def codes(result) -> set[str]:
    return {r.code for r in result.reasons}


@pytest.mark.parametrize(
    ("similarity", "decision", "code"),
    [
        (0.70, Decision.PASS, None),
        (0.4501, Decision.PASS, None),  # just above the threshold
        (0.35, Decision.REVIEW, "faces_match_uncertain"),
        (0.10, Decision.FAIL, "faces_do_not_match"),
    ],
)
def test_decision_bands(similarity, decision, code):
    pipeline, _ = build(similarity)
    result = pipeline.verify(portrait(), "cap-1")
    assert result.decision is decision
    assert result.data["similarity"] == pytest.approx(similarity, abs=1e-3)
    if code:
        assert code in codes(result)
    if decision is Decision.FAIL:
        assert result.score <= 0.5


def test_unknown_capture_is_refused():
    pipeline, _ = build(captures={})
    with pytest.raises(CaptureNotFound):
        pipeline.verify(portrait(), "nope")


def test_no_face_on_the_card_fails():
    pipeline, det = build(faces=[])
    result = pipeline.verify(portrait(), "cap-1")
    assert result.decision is Decision.FAIL
    assert codes(result) == {"reference_no_face"}
    assert det.calls == 12  # upright + every fallback angle


def test_rotated_card_portrait_is_found_by_the_fallback():
    pipeline, det = build(found_on_call=2)  # upright fails, first quarter turn succeeds
    result = pipeline.verify(portrait(), "cap-1")
    assert result.decision is Decision.PASS
    assert result.data["reference_face"]["rotation"] == 90


def test_fallback_can_be_disabled():
    pipeline, det = build(found_on_call=2, rotation_fallback=False)
    assert pipeline.verify(portrait(), "cap-1").decision is Decision.FAIL
    assert det.calls == 1


def test_small_or_crowded_card_portrait_goes_to_review():
    pipeline, _ = build(faces=[face((60, 40, 90, 70)), face((100, 100, 120, 120))])
    result = pipeline.verify(portrait(), "cap-1")
    assert result.decision is Decision.REVIEW
    assert {"reference_low_quality", "reference_multiple_faces"} <= codes(result)


def test_rotate_keeps_the_whole_image():
    img = np.zeros((100, 200, 3), np.uint8)
    assert _rotate(img, 90).shape[:2] == (200, 100)


def test_api_verify_takes_a_capture_id_not_a_selfie():
    pipeline, _ = build(0.7)
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    try:
        client = TestClient(app)
        jpg = cv2.imencode(".jpg", portrait())[1].tobytes()
        body = client.post(
            "/v1/face-match/verify", files={"reference": ("p.jpg", jpg, "image/jpeg")}, data={"capture_id": "cap-1"}
        ).json()
        assert body["module"] == "face_match" and body["decision"] == "pass"
        r = client.post(
            "/v1/face-match/verify", files={"reference": ("p.jpg", jpg, "image/jpeg")}, data={"capture_id": "gone"}
        )
        assert r.status_code == 404 and r.json()["error"] == "capture_not_found"
        params = client.get("/openapi.json").json()["paths"]["/v1/face-match/verify"]["post"]
        schema_ref = params["requestBody"]["content"]["multipart/form-data"]["schema"]["$ref"].split("/")[-1]
        fields = client.get("/openapi.json").json()["components"]["schemas"][schema_ref]["properties"]
        assert set(fields) == {"reference", "capture_id"}  # no selfie image field
    finally:
        app.dependency_overrides.clear()
