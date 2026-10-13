"""Passive anti-spoof (spec 04) with fake models - no weights needed."""

from __future__ import annotations

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.core.config import AntiSpoofSettings, Settings
from kyc.core.errors import CaptureNotFound
from kyc.core.face.antispoof import SIZE, context_crop
from kyc.core.schemas import Decision
from kyc.modules.antispoof.pipeline import AntiSpoofPipeline, jitter_boxes
from kyc.modules.antispoof.router import get_pipeline
from kyc.modules.face_capture.session import Capture


def capture(box=(270.0, 190.0, 370.0, 290.0), sharpness=50.0) -> Capture:
    img = np.random.default_rng(3).integers(0, 255, (480, 640, 3)).astype(np.uint8)
    return Capture(
        capture_id="cap-1", session_id="s", jpeg=cv2.imencode(".jpg", img)[1].tobytes(),
        face_box=box, kps=np.zeros((5, 2), np.float32), metrics={"sharpness": sharpness}, score=0.9,
    )


class FakeModel:
    def __init__(self, live: float, scale: float, name: str):
        self.live, self.scale, self.name = live, scale, name
        self.batch_sizes: list[int] = []

    def live_probabilities(self, crops):
        assert all(c.shape == (SIZE, SIZE, 3) for c in crops)
        self.batch_sizes.append(len(crops))
        return np.full(len(crops), self.live)


def build(live=(0.95, 0.95), captures=None, **cfg):
    models = [FakeModel(live[0], 2.7, "v2"), FakeModel(live[1], 4.0, "v1se")]
    store = {"cap-1": capture()} if captures is None else captures
    p = AntiSpoofPipeline(
        Settings(antispoof=AntiSpoofSettings(**cfg)), capture_lookup=store.get, models_factory=lambda: models
    )
    return p, models


# ------------------------------------------------------------------ crop geometry


def test_context_crop_fits_when_there_is_room():
    img = np.zeros((480, 640, 3), np.uint8)
    crop, clamped = context_crop(img, (300, 220, 340, 260), 2.7)  # 40px box -> 108px context
    assert crop.shape == (SIZE, SIZE, 3) and not clamped


def test_context_crop_clamps_a_close_face_like_upstream():
    """A 220px-tall face in a 480px frame cannot get 4x context: upstream
    shrinks the scale to fit - reported, not silent."""
    img = np.zeros((480, 640, 3), np.uint8)
    _, clamped = context_crop(img, (240, 130, 400, 350), 4.0)
    assert clamped


def test_context_crop_shifts_back_inside_the_frame():
    img = np.zeros((480, 640, 3), np.uint8)
    img[:, :100] = 255  # bright left strip
    crop, _ = context_crop(img, (0, 200, 40, 240), 2.7)  # box at the left edge
    assert crop[:, 0].mean() > 200  # the crop starts at x=0, not off-frame


def test_jitter_grid():
    boxes = jitter_boxes((100, 100, 200, 200), 0.1)
    assert len(boxes) == 9 and (100, 100, 200, 200) in [tuple(round(v) for v in b) for b in boxes]
    assert jitter_boxes((100, 100, 200, 200), 0) == [(100, 100, 200, 200)]


# ------------------------------------------------------------------ decisions


@pytest.mark.parametrize(
    ("live", "decision", "code"),
    [((0.98, 0.95), Decision.PASS, None), ((0.3, 0.2), Decision.REVIEW, "spoof_uncertain"),
     ((0.1, 0.05), Decision.FAIL, "spoof_detected")],
)
def test_decision_bands(live, decision, code):
    p, models = build(live)
    result = p.verify("cap-1")
    assert result.decision is decision
    assert result.data["live_score"] == pytest.approx(np.mean(live), abs=1e-3)
    if code:
        assert code in {r.code for r in result.reasons}
    assert all(m.batch_sizes == [9] for m in models)  # every model saw the whole jitter grid


def test_disagreeing_models_are_reported():
    p, _ = build((0.95, 0.2))
    assert "models_disagree" in {r.code for r in p.verify("cap-1").reasons}


def test_close_face_reports_clamped_context_as_info_only():
    p, _ = build(captures={"cap-1": capture(box=(240.0, 130.0, 400.0, 350.0))})
    result = p.verify("cap-1")
    assert result.decision is Decision.PASS
    assert "context_clamped" in {r.code for r in result.reasons}


def test_unknown_capture():
    p, _ = build(captures={})
    with pytest.raises(CaptureNotFound):
        p.verify("nope")


def test_api_takes_only_a_capture_id():
    p, _ = build((0.1, 0.05))
    app.dependency_overrides[get_pipeline] = lambda: p
    try:
        client = TestClient(app)
        body = client.post("/v1/antispoof/verify", json={"capture_id": "cap-1"}).json()
        assert body["module"] == "antispoof" and body["decision"] == "fail"
        assert client.post("/v1/antispoof/verify", json={"capture_id": "gone"}).status_code == 404
        schema = client.get("/openapi.json").json()["components"]["schemas"]["VerifyRequest"]["properties"]
        assert set(schema) == {"capture_id"}
    finally:
        app.dependency_overrides.clear()


@pytest.mark.parametrize("live", [(0.2, 0.1), (0.99, 0.99)])
def test_a_blurry_selfie_gets_retake_never_a_spoof_verdict(live):
    """Team decision: low quality -> "retake", whatever the models say."""
    p, models = build(live, captures={"cap-1": capture(sharpness=1.0)})
    result = p.verify("cap-1")
    assert result.decision is Decision.FAIL
    assert [r.code for r in result.reasons] == ["retake_low_quality"]
    assert result.data["retake"] is True and result.data["live_score"] is None
    assert all(m.batch_sizes == [] for m in models)  # the models were not consulted
