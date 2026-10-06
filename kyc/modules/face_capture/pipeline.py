"""Face capture: a guided loop that ends with a server-held selfie.

    frame -> detect faces -> 106 landmarks -> per-frame checks -> hints
          -> N consecutive passing frames -> keep the best -> capture_id

Detector and landmarker are injectable, as in the OCR pipeline, so the suite
runs without model weights or real faces.
"""

from __future__ import annotations

import base64
import time
from collections.abc import Callable

import cv2
import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import ImageQualityError, InvalidInput
from kyc.core.face.detector import Face, FaceDetector
from kyc.core.face.landmarks import Landmarker
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

from . import quality
from .session import Capture, CaptureSession, CaptureState, CaptureStore

MODULE_NAME = "face_quality"
MODULE_VERSION = "0.1.0"

DetectorFactory = Callable[[], "FaceDetector"]
LandmarkerFactory = Callable[[], "Landmarker"]


class SessionNotFound(InvalidInput):
    code = "session_not_found"
    http_status = 404


class SessionClosed(InvalidInput):
    code = "session_closed"
    http_status = 409


class FaceCapturePipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        detector_factory: DetectorFactory | None = None,
        landmarker_factory: LandmarkerFactory | None = None,
        store: CaptureStore | None = None,
    ):
        self.settings = settings or get_settings()
        cfg = self.settings.face_quality
        self._detector_factory = detector_factory or self._default_detector
        self._landmarker_factory = landmarker_factory or self._default_landmarker
        self._detector: FaceDetector | None = None
        self._landmarker: Landmarker | None = None
        self.store = store or CaptureStore(cfg.session_ttl_s)

    def _default_detector(self) -> FaceDetector:
        s = self.settings
        return FaceDetector.load(
            s.model_path(s.face_quality.detector_model), s.onnx_providers, s.onnx_intra_threads
        )

    def _default_landmarker(self) -> Landmarker:
        s = self.settings
        return Landmarker.load(
            s.model_path(s.face_quality.landmark_model), s.onnx_providers, s.onnx_intra_threads
        )

    # ---------------------------------------------------------------- analysis

    def assess(self, image: np.ndarray) -> quality.FrameAssessment:
        cfg = self.settings.face_quality
        h, w = image.shape[:2]
        if min(h, w) < cfg.min_image_side:
            raise ImageQualityError(
                f"Frame is too small ({w}x{h}); the short side must be at least {cfg.min_image_side}px",
                details={"width": w, "height": h},
            )
        if self._detector is None:
            self._detector = self._detector_factory()
        faces: list[Face] = self._detector.detect(image, cfg.detector_conf)
        landmarks = None
        if faces:
            if self._landmarker is None:
                self._landmarker = self._landmarker_factory()
            landmarks = self._landmarker.landmarks(image, faces[0])
        return quality.assess(image, faces, landmarks, cfg)

    def check(self, image: np.ndarray) -> ModuleResult:
        """Stateless single-image check (back-office use). Never yields a capture_id."""
        with Timer() as timer:
            a = self.assess(image)
            decision = _decision(a.reasons)
            score = quality.quality_score(a.metrics, self.settings.face_quality) if a.face else 0.0
            data = {"metrics": a.metrics, "face_box": _box(a.face), "landmarks_5": _kps(a.face)}
        return _envelope(decision, score, a.reasons, data, timer)

    # ---------------------------------------------------------------- sessions

    def start(self) -> CaptureSession:
        return self.store.create()

    def submit_frame(self, session_id: str, image: np.ndarray) -> dict:
        cfg = self.settings.face_quality
        session = self._session(session_id)
        with session.lock:
            if session.state is CaptureState.SEARCHING and session.expired():
                session.state = CaptureState.EXPIRED
            if session.state is not CaptureState.SEARCHING:
                raise SessionClosed(f"Capture session is {session.state.value.lower()}")
            session.frames_seen += 1
            if session.frames_seen > cfg.max_frames_per_session:
                session.state = CaptureState.EXPIRED
                raise SessionClosed("Too many frames in this capture session")

            a = self.assess(image)
            session.last_hints = a.reasons
            if a.ok:
                session.ok_streak += 1
                score = quality.quality_score(a.metrics, cfg)
                if session.candidate is None or score > session.candidate[0]:
                    session.candidate = (score, _jpeg(image, cfg.selfie_jpeg_quality), a)
                if session.ok_streak >= cfg.frames_required_ok:
                    self._capture(session)
            else:
                session.ok_streak = 0
                session.candidate = None
            return self._feedback(session, a.reasons)

    def result(self, session_id: str, *, include_selfie: bool = False) -> ModuleResult:
        session = self._session(session_id)
        with session.lock, Timer() as timer:
            if session.state is CaptureState.SEARCHING and session.expired():
                session.state = CaptureState.EXPIRED
            if session.state is CaptureState.SEARCHING:
                raise SessionClosed("Capture session has not finished yet")
            if session.state is CaptureState.EXPIRED:
                reasons = [
                    Reason.error(
                        "capture_timeout",
                        "No usable selfie was captured before the session ended",
                        "در زمان مقرر عکس قابل‌قبولی گرفته نشد؛ دوباره تلاش کنید",
                    )
                ]
                data = {"capture_id": None, "frames_seen": session.frames_seen,
                        "last_hints": [r.code for r in session.last_hints]}
                return _envelope(Decision.FAIL, 0.0, reasons, data, timer)

            capture = self.store.get_capture(session.capture_id)
            if capture is None:
                raise SessionClosed("The captured selfie has expired")
            data = {
                "capture_id": capture.capture_id,
                "face_box": list(capture.face_box),
                "landmarks_5": capture.kps.tolist(),
                "metrics": capture.metrics,
                "frames_seen": session.frames_seen,
                "selfie_base64": _b64(capture.jpeg) if include_selfie else None,
            }
            return _envelope(_decision(capture.warnings), capture.score, capture.warnings, data, timer)

    def get_capture(self, capture_id: str) -> Capture | None:
        """Public lookup for face match / anti-spoof / the orchestrator."""
        return self.store.get_capture(capture_id)

    # ---------------------------------------------------------------- helpers

    def _session(self, session_id: str) -> CaptureSession:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFound("Unknown or expired capture session")
        return session

    def _capture(self, session: CaptureSession) -> None:
        score, jpeg, a = session.candidate
        capture = Capture(
            capture_id=self.store.new_capture_id(),
            session_id=session.session_id,
            jpeg=jpeg,
            face_box=a.face.box,
            kps=a.face.kps,
            metrics=a.metrics,
            score=score,
            # Everything non-blocking stays on the record for the audit trail.
            warnings=[r for r in a.reasons if r.severity is not Severity.ERROR],
        )
        self.store.save_capture(capture)
        session.capture_id = capture.capture_id
        session.state = CaptureState.CAPTURED
        session.candidate = None

    @staticmethod
    def _feedback(session: CaptureSession, reasons: list[Reason]) -> dict:
        return {
            "state": session.state.value,
            "hints": [{"code": r.code, "severity": r.severity.value, "message_fa": r.message_fa,
                       "message_en": r.message_en} for r in reasons],
            "capture_id": session.capture_id,
            "frames_seen": session.frames_seen,
            "expires_in_s": max(0.0, round(session.expires_at - time.time(), 1)),
        }


def _decision(reasons: list[Reason]) -> Decision:
    if any(r.severity is Severity.ERROR for r in reasons):
        return Decision.FAIL
    if any(r.severity is Severity.WARN for r in reasons):
        return Decision.REVIEW
    return Decision.PASS


def _envelope(decision: Decision, score: float, reasons: list[Reason], data: dict, timer: Timer) -> ModuleResult:
    if decision is Decision.FAIL:
        score = min(score, 0.5)
    return ModuleResult(
        module=MODULE_NAME,
        version=MODULE_VERSION,
        decision=decision,
        score=max(0.0, min(1.0, score)),
        reasons=reasons,
        data=data,
        elapsed_ms=round(timer.ms, 2),
    )


def _jpeg(image: np.ndarray, quality_: int) -> bytes:
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality_])
    if not ok:
        raise InvalidInput("JPEG encoding failed")
    return buf.tobytes()


def _b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def _box(face: Face | None) -> list[float] | None:
    return [round(v, 1) for v in face.box] if face else None


def _kps(face: Face | None) -> list[list[float]] | None:
    return face.kps.round(1).tolist() if face is not None else None
