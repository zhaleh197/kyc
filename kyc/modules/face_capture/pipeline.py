"""Face capture: a guided camera session that ends with a server-held selfie.

    stream frames -> detect -> landmarks -> occlusion -> checks -> hints
        -> N consecutive passing frames      (state READY_FOR_STILL)
    still photo   -> the same checks + same person as the stream
        -> stored server-side -> capture_id  (state CAPTURED)

Video frames give live guidance; the still, taken from the same camera once
the stream qualifies, gives face match a higher-resolution selfie than a
video frame. With `require_still=false` the best stream frame is kept instead.

Every model is injectable, as in the OCR pipeline, so the suite runs without
weights or real faces.
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
from kyc.core.face.embedder import FaceEmbedder, cosine
from kyc.core.face.landmarks import Landmarker
from kyc.core.face.occlusion import OcclusionClassifier
from kyc.core.imaging import limit_size
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

from . import quality
from .session import Capture, CaptureSession, CaptureState, CaptureStore

MODULE_NAME = "face_quality"
MODULE_VERSION = "0.2.0"


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
        detector_factory: Callable[[], FaceDetector] | None = None,
        landmarker_factory: Callable[[], Landmarker] | None = None,
        occlusion_factory: Callable[[], OcclusionClassifier] | None = None,
        embedder_factory: Callable[[], FaceEmbedder] | None = None,
        store: CaptureStore | None = None,
    ):
        self.settings = settings or get_settings()
        cfg = self.settings.face_quality
        self._factories = {
            "detector": detector_factory or (lambda: self._load(FaceDetector, cfg.detector_model)),
            "landmarker": landmarker_factory or (lambda: self._load(Landmarker, cfg.landmark_model)),
            "occlusion": occlusion_factory or (lambda: self._load(OcclusionClassifier, cfg.occlusion_model)),
            "embedder": embedder_factory or (lambda: self._load(FaceEmbedder, cfg.embedding_model)),
        }
        self._models: dict[str, object] = {}
        self.store = store or CaptureStore(cfg.session_ttl_s)

    def _load(self, cls, filename: str):
        s = self.settings
        return cls.load(s.model_path(filename), s.onnx_providers, s.onnx_intra_threads)

    def _model(self, name: str):
        if name not in self._models:
            self._models[name] = self._factories[name]()
        return self._models[name]

    # ---------------------------------------------------------------- analysis

    def assess(self, image: np.ndarray) -> quality.FrameAssessment:
        cfg = self.settings.face_quality
        h, w = image.shape[:2]
        if min(h, w) < cfg.min_image_side:
            raise ImageQualityError(
                f"Frame is too small ({w}x{h}); the short side must be at least {cfg.min_image_side}px",
                details={"width": w, "height": h},
            )
        faces: list[Face] = self._model("detector").detect(image, cfg.detector_conf)
        landmarks = clear_logit = None
        if faces:
            landmarks = self._model("landmarker").landmarks(image, faces[0])
            clear_logit = self._model("occlusion").clear_logit(image, faces[0])
        return quality.assess(image, faces, landmarks, cfg, clear_logit)

    def check(self, image: np.ndarray) -> ModuleResult:
        """Stateless single-image check (back-office use). Never yields a capture_id."""
        with Timer() as timer:
            a = self.assess(image)
            score = quality.quality_score(a.metrics, self.settings.face_quality) if a.face else 0.0
            data = {"metrics": a.metrics, "face_box": _box(a.face), "landmarks_5": _kps(a.face)}
        return _envelope(_decision(a.reasons), score, a.reasons, data, timer)

    # ---------------------------------------------------------------- sessions

    def start(self) -> CaptureSession:
        return self.store.create()

    def submit_frame(self, session_id: str, image: np.ndarray) -> dict:
        """One video frame. Gives hints; qualifies the stream for the still."""
        cfg = self.settings.face_quality
        session = self._session(session_id)
        with session.lock:
            self._accept_input(session, (CaptureState.SEARCHING, CaptureState.READY_FOR_STILL))
            a = self.assess(image)
            session.last_hints = a.reasons
            if session.state is CaptureState.READY_FOR_STILL:
                # Keep guiding while the client takes the still; qualification stands.
                return self._feedback(session, a)

            if a.ok:
                session.ok_streak += 1
                score = quality.quality_score(a.metrics, cfg)
                if session.candidate is None or score > session.candidate[0]:
                    session.candidate = (score, _jpeg(image, cfg.selfie_jpeg_quality), a, image)
                if session.ok_streak >= cfg.frames_required_ok:
                    if cfg.require_still:
                        _, _, best, best_image = session.candidate
                        session.stream_embedding = self._model("embedder").embed(best_image, best.face)
                        session.state = CaptureState.READY_FOR_STILL
                    else:
                        score, jpeg, best, _ = session.candidate
                        self._capture(session, jpeg, best, score)
            else:
                session.ok_streak = 0
                session.candidate = None
            return self._feedback(session, a)

    def submit_still(self, session_id: str, image: np.ndarray) -> dict:
        """The selfie itself: a still photo from the same camera, after the stream qualified."""
        cfg = self.settings.face_quality
        session = self._session(session_id)
        with session.lock:
            self._accept_input(session, (CaptureState.READY_FOR_STILL,))
            image = limit_size(image, cfg.max_still_side)
            a = self.assess(image)
            if a.face is not None and a.ok:
                similarity = cosine(session.stream_embedding, self._model("embedder").embed(image, a.face))
                a.metrics["still_similarity"] = round(similarity, 4)
                if similarity < cfg.min_still_similarity:
                    a.reasons.append(
                        Reason.error(
                            "still_face_mismatch",
                            "The photo does not show the same person as the video",
                            "عکس با تصویر ویدیو مطابقت ندارد؛ دوباره تلاش کنید",
                        )
                    )
            session.last_hints = a.reasons
            if a.ok:
                score = quality.quality_score(a.metrics, cfg)
                self._capture(session, _jpeg(image, cfg.selfie_jpeg_quality), a, score)
            else:
                # Back to the stream: the user must re-qualify before another still.
                session.stills_rejected += 1
                session.state = CaptureState.SEARCHING
                session.ok_streak = 0
                session.candidate = None
                session.stream_embedding = None
            return self._feedback(session, a)

    def result(self, session_id: str, *, include_selfie: bool = False) -> ModuleResult:
        session = self._session(session_id)
        with session.lock, Timer() as timer:
            self._expire_if_due(session)
            if session.state in (CaptureState.SEARCHING, CaptureState.READY_FOR_STILL):
                raise SessionClosed("Capture session has not finished yet")
            if session.state is CaptureState.EXPIRED:
                reasons = [
                    Reason.error(
                        "capture_timeout",
                        "No usable selfie was captured before the session ended",
                        "در زمان مقرر عکس قابل‌قبولی گرفته نشد؛ دوباره تلاش کنید",
                    )
                ]
                data = {
                    "capture_id": None,
                    "frames_seen": session.frames_seen,
                    "stills_rejected": session.stills_rejected,
                    "last_hints": [r.code for r in session.last_hints],
                }
                return _envelope(Decision.FAIL, 0.0, reasons, data, timer)

            capture = self.store.get_capture(session.capture_id)
            if capture is None:
                raise SessionClosed("The captured selfie has expired")
            data = {
                "capture_id": capture.capture_id,
                "source": "still" if self.settings.face_quality.require_still else "stream",
                "face_box": list(capture.face_box),
                "landmarks_5": capture.kps.tolist(),
                "metrics": capture.metrics,
                "frames_seen": session.frames_seen,
                "stills_rejected": session.stills_rejected,
                "selfie_base64": base64.b64encode(capture.jpeg).decode("ascii") if include_selfie else None,
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

    @staticmethod
    def _expire_if_due(session: CaptureSession) -> None:
        open_states = (CaptureState.SEARCHING, CaptureState.READY_FOR_STILL)
        if session.state in open_states and session.expired():
            session.state = CaptureState.EXPIRED

    def _accept_input(self, session: CaptureSession, allowed: tuple[CaptureState, ...]) -> None:
        self._expire_if_due(session)
        if session.state not in allowed:
            raise SessionClosed(f"Capture session is {session.state.value.lower()}")
        session.frames_seen += 1
        if session.frames_seen > self.settings.face_quality.max_frames_per_session:
            session.state = CaptureState.EXPIRED
            raise SessionClosed("Too many frames in this capture session")

    def _capture(self, session: CaptureSession, jpeg: bytes, a: quality.FrameAssessment, score: float) -> None:
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
        session.stream_embedding = None

    def _stable_hint(self, session: CaptureSession, a: quality.FrameAssessment) -> Reason | None:
        """The one hint to show: a blocking problem seen in at least
        `hint_min_count` of the last `hint_window` frames (highest priority
        first). Until another problem qualifies, the current one stays."""
        cfg = self.settings.face_quality
        errors = [r for r in a.reasons if r.severity is Severity.ERROR]
        session.recent_codes = (session.recent_codes + [[r.code for r in errors]])[-cfg.hint_window :]
        rank = {code: i for i, code in enumerate(quality.HINT_PRIORITY)}
        counts: dict[str, int] = {}
        for codes in session.recent_codes:
            for code in codes:
                counts[code] = counts.get(code, 0) + 1
        steady = sorted((c for c, n in counts.items() if n >= cfg.hint_min_count), key=lambda c: rank.get(c, len(rank)))
        if not steady:
            # Nothing persistent: keep showing the previous hint unless the
            # recent frames have all been clean.
            if not any(session.recent_codes):
                session.shown_hint = None
        elif session.shown_hint not in steady:
            session.shown_hint = steady[0]
        if session.shown_hint is None:
            return None
        by_code = {r.code: r for r in errors}
        if session.shown_hint in by_code:
            return by_code[session.shown_hint]
        # Shown problem absent from this frame but still steady: reuse its text.
        for frame in reversed(session.last_reasons_by_code):
            if session.shown_hint in frame:
                return frame[session.shown_hint]
        return None

    def _feedback(self, session: CaptureSession, a: quality.FrameAssessment) -> dict:
        session.last_reasons_by_code = (
            session.last_reasons_by_code + [{r.code: r for r in a.reasons if r.severity is Severity.ERROR}]
        )[-self.settings.face_quality.hint_window :]
        primary = self._stable_hint(session, a)
        feedback = {
            "state": session.state.value,
            # The single, debounced instruction to show; `hints` is this frame only.
            "primary_hint": None
            if primary is None
            else {"code": primary.code, "message_fa": primary.message_fa, "message_en": primary.message_en},
            "hints": [
                {"code": r.code, "severity": r.severity.value, "message_fa": r.message_fa, "message_en": r.message_en}
                for r in a.reasons
            ],
            "capture_id": session.capture_id,
            "frames_seen": session.frames_seen,
            "expires_in_s": max(0.0, round(session.expires_at - time.time(), 1)),
        }
        if self.settings.debug:
            # Raw measurements help calibrate thresholds; not exposed in
            # production, where they would also help someone game the checks.
            feedback["metrics"] = a.metrics
        return feedback


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


def _box(face: Face | None) -> list[float] | None:
    return [round(v, 1) for v in face.box] if face else None


def _kps(face: Face | None) -> list[list[float]] | None:
    return face.kps.round(1).tolist() if face is not None else None
