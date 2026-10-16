"""Active liveness (spec 05): task rules and the session, with fakes - the
per-frame measurement is scripted, so no landmark model is needed."""

from __future__ import annotations

import random

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.core.config import AntiSpoofSettings, LivenessSettings, Settings
from kyc.core.errors import CaptureNotFound
from kyc.core.face.detector import Face
from kyc.core.schemas import Decision
from kyc.modules.face_capture.session import Capture
from kyc.modules.liveness.pipeline import LivenessPipeline, SessionClosed
from kyc.modules.liveness.router import get_pipeline
from kyc.modules.liveness.tasks import Baseline, FrameMeasure, TaskState, advance

CFG = LivenessSettings()
BASE = Baseline(eye_openness=0.30, mouth_width=1.0, face_height=200.0)
FRONT = FrameMeasure(yaw=0.0, eye_openness=0.30, mouth_width=1.0, face_height=200.0)


def m(**kw) -> FrameMeasure:
    return FrameMeasure(**{**FRONT.__dict__, **kw})


# ------------------------------------------------------------------ task rules


def run(task: str, frames) -> bool:
    t = TaskState(task)
    return any(advance(t, f, BASE, CFG) for f in frames)


def test_blink_needs_closed_then_open():
    assert run("BLINK", [FRONT, m(eye_openness=0.10), FRONT])
    assert not run("BLINK", [m(eye_openness=0.10), m(eye_openness=0.10)])  # closed eyes in a photo
    assert not run("BLINK", [FRONT, FRONT])


def test_turn_must_start_frontal_and_go_the_right_way():
    left = m(yaw=25.0) if CFG.turn_left_is_positive_yaw else m(yaw=-25.0)
    right = m(yaw=-left.yaw)
    assert run("TURN_LEFT", [FRONT, left])
    assert not run("TURN_LEFT", [left, left])  # already turned: no movement seen
    assert not run("TURN_LEFT", [FRONT, right])
    assert run("TURN_RIGHT", [FRONT, right])


def test_smile_counts_only_facing_the_camera():
    assert run("SMILE", [FRONT, m(mouth_width=1.2)])
    assert not run("SMILE", [m(yaw=30.0, mouth_width=1.2)])
    assert not run("SMILE", [m(mouth_width=1.05)])


def test_move_closer_is_relative_to_the_baseline():
    assert run("MOVE_CLOSER", [m(face_height=250.0)])
    assert not run("MOVE_CLOSER", [m(face_height=220.0)])


# ------------------------------------------------------------------ session

STRUCTURED = np.full((480, 640, 3), 128, np.uint8)
for k in range(240, 400, 8):
    STRUCTURED[130:350, k : k + 2] = 40  # real edges, so frames count as sharp
FACE = Face(box=(240.0, 130.0, 400.0, 350.0), score=0.9,
            kps=np.array([[290, 210], [350, 210], [320, 240], [296, 268], [344, 268]], np.float32))


class Script:
    """Feeds the pipeline: one scripted FrameMeasure per frame (None = no face)."""

    def __init__(self, measures):
        self.measures = list(measures)
        self.i = 0

    def next(self):
        out = self.measures[min(self.i, len(self.measures) - 1)]
        self.i += 1
        return out


class FakeDetector:
    def __init__(self, script):
        self.script = script

    def detect(self, image, conf_threshold=0.5):  # noqa: ARG002
        return [] if self.script.measures[min(self.script.i, len(self.script.measures) - 1)] is None else [FACE]


class FakeLandmarker:
    """Measurements are scripted (pipeline._measure), so the points are unused."""

    def landmarks(self, image, face):  # noqa: ARG002
        return np.zeros((106, 2), np.float32)


class FakeEmbedder:
    def __init__(self, same=True):
        self.same = same
        self.calls = 0

    def embed(self, image, face):  # noqa: ARG002
        self.calls += 1
        if self.calls == 1 or self.same:  # first call embeds the selfie
            return np.array([1.0, 0.0])
        return np.array([0.0, 1.0])


class FakeSpoofModel:
    scale, name = 4.0, "fake"

    def __init__(self, live):
        self.live = live

    def live_probabilities(self, crops):
        return np.full(len(crops), self.live)


class InOrder:
    """Stands in for the session RNG: tasks in pool order, so a script knows
    which task comes first."""

    def sample(self, population, k):
        return list(population)[:k]


def build(script, *, tasks=("BLINK", "SMILE"), same_face=True, live=0.9, **cfg):
    capture = Capture(capture_id="cap-1", session_id="s", jpeg=cv2.imencode(".jpg", STRUCTURED)[1].tobytes(),
                      face_box=FACE.box, kps=FACE.kps, metrics={}, score=0.9)
    cfg.setdefault("tasks_per_session", len(tasks))
    settings = Settings(liveness=LivenessSettings(task_pool=list(tasks), **cfg),
                        antispoof=AntiSpoofSettings())
    p = LivenessPipeline(
        settings, capture_lookup={"cap-1": capture}.get,
        detector_factory=lambda: FakeDetector(script), landmarker_factory=lambda: FakeLandmarker(),
        embedder_factory=lambda: FakeEmbedder(same_face), antispoof_factory=lambda: [FakeSpoofModel(live)],
        rng=InOrder(),
    )
    p._measure = lambda face, landmarks: script.next()  # scripted measurements
    return p


def drive(p, n):
    sid = p.start("cap-1").session_id
    feedback = None
    for _ in range(n):
        try:
            feedback = p.submit_frame(sid, STRUCTURED)
        except SessionClosed:
            break
    return sid, feedback


def test_a_live_person_doing_the_tasks_passes():
    script = Script([FRONT, FRONT,  # baseline
                     m(eye_openness=0.1), FRONT,  # blink
                     m(mouth_width=1.2)])  # smile
    p = build(script, tasks=("BLINK", "SMILE"))
    sid, fb = drive(p, 5)
    assert fb["state"] == "VERIFIED"
    result = p.result(sid)
    assert result.decision is Decision.PASS
    assert result.data["identity_checks"] > 0 and result.data["antispoof_checks"] > 0


def test_tasks_come_in_random_order_from_the_pool():
    script = Script([FRONT])
    pool = ("BLINK", "SMILE", "MOVE_CLOSER", "TURN_LEFT")
    p = build(script, tasks=pool, tasks_per_session=2)
    p._rng = random.Random(1)
    seen = {tuple(t.name for t in p.start("cap-1").tasks) for _ in range(10)}
    assert len(seen) > 1


def test_a_task_not_done_in_time_fails():
    p = build(Script([FRONT]), task_time_limit_s=-1.0)  # Windows clock ticks ~15 ms
    sid, fb = drive(p, 4)
    assert fb["state"] == "FAILED"
    assert p.result(sid).decision is Decision.FAIL


def test_losing_the_face_fails_after_the_grace_period():
    p = build(Script([FRONT, FRONT, None]), max_face_lost_s=-1.0)
    sid, _ = drive(p, 4)
    assert {r.code for r in p.result(sid).reasons} == {"face_lost"}


def test_someone_else_taking_over_fails():
    p = build(Script([FRONT]), same_face=False)
    sid, fb = drive(p, 3)
    assert fb["state"] == "FAILED"
    assert {r.code for r in p.result(sid).reasons} == {"face_changed"}


def test_video_anti_spoof_is_reported_not_judged_by_default():
    """Thresholds are calibrated on stills; a live tester's video frames
    scored low on a phone. Default "info": reported, decision unchanged."""
    script = Script([FRONT, FRONT, m(eye_openness=0.1), FRONT, m(mouth_width=1.2)])
    p = build(script, live=0.05)
    sid, _ = drive(p, 5)
    result = p.result(sid)
    assert result.decision is Decision.PASS
    assert "antispoof_video_score" in {r.code for r in result.reasons}
    assert result.data["live_scores"]


def test_a_screen_doing_the_tasks_fails_on_anti_spoof_when_blocking():
    script = Script([FRONT, FRONT, m(eye_openness=0.1), FRONT, m(mouth_width=1.2)])
    p = build(script, live=0.05, antispoof_on_frames="block")
    sid, fb = drive(p, 5)
    assert fb["state"] == "VERIFIED"  # the moves were made...
    result = p.result(sid)
    assert result.decision is Decision.FAIL  # ...but not by a live face
    assert "spoof_during_session" in {r.code for r in result.reasons}


def test_unknown_capture_cannot_start_a_session():
    p = build(Script([FRONT]))
    with pytest.raises(CaptureNotFound):
        p.start("nope")


def test_result_before_the_end_is_refused():
    p = build(Script([FRONT]))
    sid = p.start("cap-1").session_id
    with pytest.raises(SessionClosed):
        p.result(sid)


def test_api_flow():
    script = Script([FRONT, FRONT, m(eye_openness=0.1), FRONT, m(mouth_width=1.2)])
    p = build(script)
    app.dependency_overrides[get_pipeline] = lambda: p
    try:
        client = TestClient(app)
        start = client.post("/v1/liveness/sessions", json={"capture_id": "cap-1"}).json()
        assert [t["task"] for t in start["tasks"]] and start["first_instruction_fa"]
        jpg = cv2.imencode(".jpg", STRUCTURED)[1].tobytes()
        for _ in range(5):
            fb = client.post(f"/v1/liveness/sessions/{start['session_id']}/frames",
                             files={"file": ("f.jpg", jpg, "image/jpeg")}).json()
        assert fb["state"] == "VERIFIED"
        body = client.get(f"/v1/liveness/sessions/{start['session_id']}/result").json()
        assert body["module"] == "liveness" and body["decision"] == "pass"
        assert client.post("/v1/liveness/sessions", json={"capture_id": "gone"}).status_code == 404
    finally:
        app.dependency_overrides.clear()
