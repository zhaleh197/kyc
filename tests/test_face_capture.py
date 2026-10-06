"""Face capture (spec 02): geometry, per-frame checks, sessions and the API.

Detector and landmarker are fakes: a scripted face on a synthetic textured
frame, so no model weights and no real person's photo are needed.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from kyc.api.app import app
from kyc.core.config import FaceQualitySettings, Settings
from kyc.core.face import detector as det_mod
from kyc.core.face.detector import Face, FaceDetector
from kyc.core.face.geometry import EYE_LEFT_IMG, EYE_RIGHT_IMG, eye_openness, head_pose
from kyc.core.schemas import Decision
from kyc.modules.face_capture.pipeline import FaceCapturePipeline, SessionClosed
from kyc.modules.face_capture.quality import assess, context_scale
from kyc.modules.face_capture.router import get_pipeline

FRAME_W, FRAME_H = 640, 480


# ------------------------------------------------------------------ fixtures


def frontal_kps(cx: float = 320, cy: float = 240, s: float = 1.0) -> np.ndarray:
    """Symmetric, level 5-point layout: eyes, nose, mouth corners."""
    return np.array(
        [[cx - 30 * s, cy - 30 * s], [cx + 30 * s, cy - 30 * s], [cx, cy],
         [cx - 24 * s, cy + 28 * s], [cx + 24 * s, cy + 28 * s]],
        np.float32,
    )


def make_face(box=(240, 130, 400, 350), kps=None, score=0.9) -> Face:
    x1, y1, x2, y2 = box
    if kps is None:
        kps = frontal_kps((x1 + x2) / 2, (y1 + y2) / 2, (x2 - x1) / 160)
    return Face(box=tuple(float(v) for v in box), score=score, kps=kps)


def landmarks_106(eye_open: float = 0.3) -> np.ndarray:
    """Only the eye points matter to the checks; place them with a given EAR."""
    pts = np.zeros((106, 2), np.float32)
    for eye, cx in ((EYE_LEFT_IMG, 290.0), (EYE_RIGHT_IMG, 350.0)):
        c1, c2 = eye["corners"]
        pts[c1], pts[c2] = (cx - 20, 210), (cx + 20, 210)
        for i, (up, down) in enumerate(eye["pairs"]):
            x = cx - 10 + 10 * i
            pts[up], pts[down] = (x, 210 - 20 * eye_open), (x, 210 + 20 * eye_open)
    return pts


def textured_frame(brightness: int = 128, blur: float = 0.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    frame = rng.integers(brightness - 40, brightness + 40, (FRAME_H, FRAME_W, 3)).astype(np.uint8)
    if blur:
        frame = cv2.GaussianBlur(frame, (0, 0), blur)
    return frame


class FakeDetector:
    def __init__(self, faces):
        self.faces = faces

    def detect(self, image, conf_threshold=0.5):  # noqa: ARG002
        return list(self.faces)


class FakeLandmarker:
    def __init__(self, eye_open=0.3):
        self.eye_open = eye_open

    def landmarks(self, image, face):  # noqa: ARG002
        return landmarks_106(self.eye_open)


def build(faces=None, eye_open=0.3, **cfg) -> FaceCapturePipeline:
    settings = Settings(face_quality=FaceQualitySettings(**cfg))
    det = FakeDetector([make_face()] if faces is None else faces)
    lm = FakeLandmarker(eye_open)
    return FaceCapturePipeline(settings, detector_factory=lambda: det, landmarker_factory=lambda: lm)


def codes(reasons) -> set[str]:
    return {r.code for r in reasons}


# ------------------------------------------------------------------ geometry


def test_frontal_pose_is_near_zero():
    yaw, pitch, roll = head_pose(frontal_kps())
    assert abs(yaw) < 1 and abs(roll) < 1 and abs(pitch) < 10


def test_yaw_flips_sign_when_mirrored():
    kps = frontal_kps()
    kps[2, 0] += 12  # nose towards image right
    mirrored = kps.copy()
    mirrored[:, 0] = 640 - mirrored[:, 0]
    mirrored = mirrored[[1, 0, 2, 4, 3]]  # left/right labels swap in a mirror
    yaw, _, _ = head_pose(kps)
    yaw_m, _, _ = head_pose(mirrored)
    assert yaw > 10
    assert yaw_m == pytest.approx(-yaw, abs=0.5)


def test_roll_follows_the_eye_line():
    kps = frontal_kps()
    kps[1, 1] += 30  # right eye lower -> clockwise tilt
    assert head_pose(kps)[2] > 20


def test_chin_up_gives_positive_pitch():
    kps = frontal_kps()
    kps[2, 1] -= 15  # nose closer to the eyes, as when the head tips back
    assert head_pose(kps)[1] > 15


def test_eye_openness_tracks_lid_distance():
    open_l, open_r = eye_openness(landmarks_106(0.3))
    closed_l, closed_r = eye_openness(landmarks_106(0.05))
    assert open_l > 0.25 and open_r > 0.25
    assert closed_l < 0.1 and closed_r < 0.1


def test_context_scale_is_limited_by_the_nearest_edge():
    face = make_face(box=(270, 190, 370, 290))  # 100px box centred in 640x480
    assert context_scale(face, FRAME_W, FRAME_H) == pytest.approx(4.8)


# ------------------------------------------------------------------ per-frame checks


def check(faces, eye_open=0.3, frame=None, **cfg):
    lm = landmarks_106(eye_open) if faces else None
    return assess(textured_frame() if frame is None else frame, faces, lm, FaceQualitySettings(**cfg))


def test_good_frame_passes():
    a = check([make_face()])
    assert a.ok, codes(a.reasons)


def test_no_face():
    assert codes(check([]).reasons) == {"no_face"}


def test_second_large_face_blocks_but_a_small_background_face_does_not():
    main = make_face()
    big = make_face(box=(20, 20, 140, 200))
    tiny = make_face(box=(20, 20, 50, 50))
    assert "multiple_faces" in codes(check([main, big]).reasons)
    assert "multiple_faces" not in codes(check([main, tiny]).reasons)


def test_framing_checks():
    assert "face_too_small" in codes(check([make_face(box=(300, 220, 340, 260))]).reasons)
    assert "face_too_large" in codes(check([make_face(box=(150, 20, 490, 470))]).reasons)
    assert "face_cut_off" in codes(check([make_face(box=(0, 100, 150, 330))]).reasons)


def test_blurry_dark_and_bright_frames():
    assert "face_blurry" in codes(check([make_face()], frame=textured_frame(blur=6)).reasons)
    assert "face_too_dark" in codes(check([make_face()], frame=textured_frame(brightness=45)).reasons)
    assert "face_too_bright" in codes(check([make_face()], frame=textured_frame(brightness=215)).reasons)


def test_closed_eyes_block():
    assert "eyes_closed" in codes(check([make_face()], eye_open=0.05).reasons)


def test_turned_and_tilted_head_block():
    turned = make_face()
    turned.kps[2, 0] += 25
    tilted = make_face()
    tilted.kps[1, 1] += 30
    assert "head_turned" in codes(check([turned]).reasons)
    assert "head_tilted" in codes(check([tilted]).reasons)


def test_uneven_lighting_is_informational_only():
    frame = textured_frame()
    frame[:, :320] //= 4
    a = check([make_face()], frame=frame, min_brightness=10)
    assert "uneven_lighting" in codes(a.reasons)
    assert a.ok


def test_context_check_is_off_by_default_and_enforceable():
    face = make_face()  # 220px tall in a 480px frame -> about 2.2x context
    assert "face_too_close_for_context" not in codes(check([face]).reasons)
    assert "face_too_close_for_context" in codes(check([face], min_context_scale=2.7).reasons)


# ------------------------------------------------------------------ sessions


def test_selfie_is_captured_after_consecutive_good_frames():
    p = build()
    sid = p.start().session_id
    assert p.submit_frame(sid, textured_frame())["state"] == "SEARCHING"
    fb = p.submit_frame(sid, textured_frame())
    assert fb["state"] == "CAPTURED" and fb["capture_id"]

    result = p.result(sid, include_selfie=True)
    assert result.decision is Decision.PASS
    assert result.data["capture_id"] == fb["capture_id"]
    assert result.data["selfie_base64"]
    capture = p.get_capture(fb["capture_id"])
    assert cv2.imdecode(np.frombuffer(capture.jpeg, np.uint8), cv2.IMREAD_COLOR).shape[:2] == (FRAME_H, FRAME_W)


def test_a_bad_frame_resets_the_streak():
    p = build()
    sid = p.start().session_id
    p.submit_frame(sid, textured_frame())
    p.submit_frame(sid, textured_frame(blur=6))
    assert p.submit_frame(sid, textured_frame())["state"] == "SEARCHING"
    assert p.submit_frame(sid, textured_frame())["state"] == "CAPTURED"


def test_hints_carry_persian_instructions():
    p = build(faces=[])
    fb = p.submit_frame(p.start().session_id, textured_frame())
    assert fb["hints"][0]["code"] == "no_face"
    assert fb["hints"][0]["message_fa"]


def test_session_is_closed_after_capture():
    p = build(frames_required_ok=1)
    sid = p.start().session_id
    p.submit_frame(sid, textured_frame())
    with pytest.raises(SessionClosed):
        p.submit_frame(sid, textured_frame())


def test_result_before_capture_is_refused():
    p = build()
    with pytest.raises(SessionClosed):
        p.result(p.start().session_id)


def test_expired_session_fails_with_capture_timeout():
    p = build(faces=[])
    session = p.start()
    p.submit_frame(session.session_id, textured_frame())
    session.expires_at = time.time() - 1
    result = p.result(session.session_id)
    assert result.decision is Decision.FAIL
    assert codes(result.reasons) == {"capture_timeout"}
    assert result.data["capture_id"] is None


def test_frame_budget_is_enforced():
    p = build(faces=[], max_frames_per_session=2)
    sid = p.start().session_id
    p.submit_frame(sid, textured_frame())
    p.submit_frame(sid, textured_frame())
    with pytest.raises(SessionClosed):
        p.submit_frame(sid, textured_frame())


# ------------------------------------------------------------------ detector decoding


def test_scrfd_decoding_maps_an_anchor_back_to_image_pixels():
    """One confident anchor at stride 8, cell (10, 5): box and keypoints must
    land where insightface's decoding would put them, scaled to the source."""

    class FakeSession:
        def get_inputs(self):
            return [type("I", (), {"name": "input.1"})()]

        def run(self, _, feeds):
            outs = []
            for stride in det_mod.STRIDES:
                n = (det_mod.INPUT_SIZE // stride) ** 2 * det_mod.ANCHORS_PER_CELL
                outs.append(np.zeros((n, 1), np.float32))
            boxes = [np.zeros((len(o), 4), np.float32) for o in outs]
            kps = [np.zeros((len(o), 10), np.float32) for o in outs]
            idx = (5 * 80 + 10) * 2  # row 5, col 10, first anchor
            outs[0][idx] = 0.9
            boxes[0][idx] = [2, 3, 4, 5]  # distances in stride units
            kps[0][idx] = [1, 1] * 5
            return outs + boxes + kps

    detector = FaceDetector(FakeSession())
    image = np.zeros((320, 320, 3), np.uint8)  # scale 2 to 640
    faces = detector.detect(image, 0.5)
    assert len(faces) == 1
    cx, cy = 10 * 8, 5 * 8
    expected = np.array([cx - 16, cy - 24, cx + 32, cy + 40]) / 2
    assert np.allclose(faces[0].box, expected)
    assert np.allclose(faces[0].kps[0], [(cx + 8) / 2, (cy + 8) / 2])


# ------------------------------------------------------------------ API


@pytest.fixture
def client():
    pipeline = build()
    app.dependency_overrides[get_pipeline] = lambda: pipeline
    yield TestClient(app)
    app.dependency_overrides.clear()


def jpeg(img) -> bytes:
    return cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])[1].tobytes()


def test_api_capture_flow(client):
    sid = client.post("/v1/face-capture/sessions").json()["session_id"]
    for _ in range(2):
        fb = client.post(
            f"/v1/face-capture/sessions/{sid}/frames", files={"file": ("f.jpg", jpeg(textured_frame()), "image/jpeg")}
        ).json()
    assert fb["state"] == "CAPTURED"
    body = client.get(f"/v1/face-capture/sessions/{sid}/result").json()
    assert body["module"] == "face_quality"
    assert body["data"]["capture_id"] == fb["capture_id"]
    assert body["data"]["selfie_base64"] is None


def test_api_unknown_session_is_404(client):
    r = client.post("/v1/face-capture/sessions/nope/frames", files={"file": ("f.jpg", jpeg(textured_frame()), "image/jpeg")})
    assert r.status_code == 404
    assert r.json()["error"] == "session_not_found"


def test_api_check_never_issues_a_capture_id(client):
    body = client.post(
        "/v1/face-quality/check", files={"file": ("f.jpg", jpeg(textured_frame()), "image/jpeg")}
    ).json()
    assert body["decision"] == "pass"
    assert "capture_id" not in body["data"]


def test_no_route_accepts_a_selfie_file_outside_a_session():
    paths = TestClient(app).get("/openapi.json").json()["paths"]
    face_posts = [p for p, ops in paths.items() if "face" in p and "post" in ops]
    assert set(face_posts) == {
        "/v1/face-capture/sessions",
        "/v1/face-capture/sessions/{session_id}/frames",
        "/v1/face-capture/sessions/{session_id}/frames/base64",
        "/v1/face-quality/check",
    }
