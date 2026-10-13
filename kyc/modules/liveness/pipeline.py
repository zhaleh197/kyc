"""Active liveness on the camera stream, bound to a captured selfie.

    POST session(capture_id) -> random tasks
    each frame -> one face? -> landmarks -> measure (yaw, eyes, mouth, size)
               -> baseline (first frontal frames) -> current task advances
               -> on frontal frames: same person as the selfie? live (anti-spoof)?
    result -> pass only if every task was done in time, the face never
              changed, and the frames look live

Models are shared with the other modules (kyc.core.face.registry) and every
one is injectable, so the suite runs without weights.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable

import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import CaptureNotFound, InvalidInput
from kyc.core.face.antispoof import AntiSpoofModel, score_face
from kyc.core.face.detector import Face, FaceDetector
from kyc.core.face.embedder import FaceEmbedder, cosine
from kyc.core.face.geometry import MOUTH, eye_openness, head_pose
from kyc.core.face.landmarks import Landmarker
from kyc.core.face.registry import load_shared
from kyc.core.face.sharpness import face_crop_gray, face_sharpness
from kyc.core.imaging import decode_image
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

from .session import TERMINAL, LivenessSession, LivenessState, LivenessStore
from .tasks import INSTRUCTIONS_EN, INSTRUCTIONS_FA, Baseline, FrameMeasure, TaskState, advance, is_frontal

MODULE_NAME = "liveness"
MODULE_VERSION = "0.1.0"


class SessionNotFound(InvalidInput):
    code = "session_not_found"
    http_status = 404


class SessionClosed(InvalidInput):
    code = "session_closed"
    http_status = 409


def _fail(code: str, en: str, fa: str) -> Reason:
    return Reason.error(code, en, fa)


class LivenessPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        capture_lookup: Callable[[str], object | None],
        detector_factory: Callable[[], FaceDetector] | None = None,
        landmarker_factory: Callable[[], Landmarker] | None = None,
        embedder_factory: Callable[[], FaceEmbedder] | None = None,
        antispoof_factory: Callable[[], list[AntiSpoofModel]] | None = None,
        rng: random.Random | None = None,
    ):
        self.settings = settings or get_settings()
        cfg = self.settings.liveness
        self._capture_lookup = capture_lookup
        self._factories = {
            "detector": detector_factory or (lambda: self._load(FaceDetector, cfg.detector_model)),
            "landmarker": landmarker_factory or (lambda: self._load(Landmarker, cfg.landmark_model)),
            "embedder": embedder_factory or (lambda: self._load(FaceEmbedder, cfg.embedding_model)),
            "antispoof": antispoof_factory or self._load_antispoof,
        }
        self._models: dict[str, object] = {}
        self._rng = rng or random.SystemRandom()  # tasks must not be predictable
        self.store = LivenessStore(cfg.session_ttl_s)

    def _load(self, cls, filename: str):
        s = self.settings
        return load_shared(cls, s.model_path(filename), providers=s.onnx_providers, intra_threads=s.onnx_intra_threads)

    def _load_antispoof(self) -> list[AntiSpoofModel]:
        s = self.settings
        out = []
        for spec in s.antispoof.models:
            filename, _, scale = spec.rpartition(":")
            out.append(load_shared(AntiSpoofModel, s.model_path(filename), float(scale),
                                   providers=s.onnx_providers, intra_threads=s.onnx_intra_threads))
        return out

    def _model(self, name: str):
        if name not in self._models:
            self._models[name] = self._factories[name]()
        return self._models[name]

    # ------------------------------------------------------------------ start

    def start(self, capture_id: str) -> LivenessSession:
        cfg = self.settings.liveness
        capture = self._capture_lookup(capture_id)
        if capture is None:
            raise CaptureNotFound("Unknown or expired capture_id; take a new selfie")
        selfie = decode_image(capture.jpeg)
        face = Face(box=tuple(capture.face_box), score=1.0, kps=np.asarray(capture.kps, np.float32))
        embedding = self._model("embedder").embed(selfie, face)
        names = self._rng.sample(cfg.task_pool, min(cfg.tasks_per_session, len(cfg.task_pool)))
        return self.store.create(capture_id, embedding, [TaskState(n) for n in names])

    # ------------------------------------------------------------------ frames

    def submit_frame(self, session_id: str, image: np.ndarray) -> dict:
        cfg = self.settings.liveness
        session = self._session(session_id)
        with session.lock:
            now = time.time()
            if session.state not in TERMINAL and now >= session.expires_at:
                session.state = LivenessState.EXPIRED
                session.failure = _fail("session_expired", "The liveness session expired", "زمان بررسی تمام شد")
            if session.state in TERMINAL:
                raise SessionClosed(f"Liveness session is {session.state.value.lower()}")
            session.frames_seen += 1
            if session.frames_seen > cfg.max_frames_per_session:
                self._end(session, _fail("too_many_frames", "Too many frames", "تعداد فریم‌ها بیش از حد مجاز است"))
                return self._feedback(session, None)

            faces = self._model("detector").detect(image, cfg.detector_conf)
            if len(faces) != 1:
                hint = None if not faces else ("multiple_faces", "فقط خودتان در کادر باشید")
                if now - session.last_face_at > cfg.max_face_lost_s:
                    code, fa = ("multiple_faces", "بیش از یک چهره در کادر بود") if faces else (
                        "face_lost", "چهره از کادر خارج شد")
                    self._end(session, _fail(code, "Lost the single face in frame", fa))
                return self._feedback(session, hint or ("no_face", "صورتتان را در کادر نگه دارید"))
            session.last_face_at = now
            face = faces[0]
            landmarks = self._model("landmarker").landmarks(image, face)
            m = self._measure(face, landmarks)

            if is_frontal(m, cfg):
                if not self._check_identity(session, image, face):
                    return self._feedback(session, None)
                self._check_live(session, image, face)

            if session.state is LivenessState.BASELINE:
                self._baseline_step(session, m, now)
            elif session.state is LivenessState.RUNNING:
                self._task_step(session, m, now)
            return self._feedback(session, None, m)

    def _measure(self, face: Face, landmarks: np.ndarray) -> FrameMeasure:
        yaw, _, _ = head_pose(face.kps)
        left, right = eye_openness(landmarks)
        eye_distance = float(np.linalg.norm(face.kps[0] - face.kps[1])) or 1.0
        c1, c2 = MOUTH["corners"]
        mouth = float(np.linalg.norm(landmarks[c1] - landmarks[c2])) / eye_distance
        return FrameMeasure(yaw=yaw, eye_openness=(left + right) / 2, mouth_width=mouth, face_height=face.height)

    def _check_identity(self, session: LivenessSession, image: np.ndarray, face: Face) -> bool:
        cfg = self.settings.liveness
        if session.identity and session.frames_seen % cfg.identity_every_n_frames:
            return True
        similarity = cosine(session.selfie_embedding, self._model("embedder").embed(image, face))
        session.identity.append(round(similarity, 4))
        if similarity < cfg.identity_min_similarity:
            self._end(session, _fail(
                "face_changed", f"The face no longer matches the selfie ({similarity:.2f})",
                "چهرهٔ داخل تصویر با سلفی یکی نیست"))
            return False
        return True

    def _check_live(self, session: LivenessSession, image: np.ndarray, face: Face) -> None:
        cfg, spoof = self.settings.liveness, self.settings.antispoof
        if session.live_scores and session.frames_seen % cfg.antispoof_every_n_frames:
            return
        if face_sharpness(face_crop_gray(image, face)) < spoof.min_sharpness:
            return  # anti-spoof cannot judge a blurry frame (spec 04)
        live, _, _ = score_face(self._model("antispoof"), image, face.box, spoof.box_jitter)
        session.live_scores.append(round(live, 4))

    def _baseline_step(self, session: LivenessSession, m: FrameMeasure, now: float) -> None:
        cfg = self.settings.liveness
        eyes_open = m.eye_openness >= self.settings.face_quality.min_eye_openness
        if is_frontal(m, cfg) and eyes_open:
            session.baseline_samples.append(m)
        if len(session.baseline_samples) >= cfg.baseline_frames:
            samples = session.baseline_samples
            session.baseline = Baseline(
                eye_openness=float(np.median([s.eye_openness for s in samples])),
                mouth_width=float(np.median([s.mouth_width for s in samples])),
                face_height=float(np.median([s.face_height for s in samples])),
            )
            session.state = LivenessState.RUNNING
            session.task_started_at = now
        elif now - session.started_at > cfg.task_time_limit_s:
            self._end(session, _fail("baseline_timeout", "Never looked straight at the camera",
                                     "مستقیم به دوربین نگاه نکردید؛ دوباره تلاش کنید"))

    def _task_step(self, session: LivenessSession, m: FrameMeasure, now: float) -> None:
        cfg = self.settings.liveness
        task = session.current_task
        if advance(task, m, session.baseline, cfg):
            session.task_durations.append(round(now - session.task_started_at, 2))
            session.task_index += 1
            session.task_started_at = now
            if session.current_task is None:
                session.state = LivenessState.VERIFIED
        elif now - session.task_started_at > cfg.task_time_limit_s:
            self._end(session, _fail(
                "task_timeout", f"Task not done in time: {task.name}",
                f"این کار در زمان مقرر انجام نشد: «{INSTRUCTIONS_FA[task.name]}»"))

    @staticmethod
    def _end(session: LivenessSession, reason: Reason) -> None:
        session.state = LivenessState.FAILED
        session.failure = reason

    # ------------------------------------------------------------------ result

    def result(self, session_id: str) -> ModuleResult:
        spoof = self.settings.antispoof
        session = self._session(session_id)
        with session.lock, Timer() as timer:
            if session.state not in TERMINAL:
                raise SessionClosed("Liveness session has not finished yet")
            reasons: list[Reason] = []
            live = float(np.median(session.live_scores)) if session.live_scores else None
            if session.state is not LivenessState.VERIFIED:
                reasons.append(session.failure or _fail("not_verified", "Not verified", "تأیید نشد"))
            elif live is None:
                reasons.append(Reason.warn("antispoof_not_checked", "No frame was sharp enough to check for spoofing",
                                           "تصویر برای بررسی جعل به اندازهٔ کافی واضح نبود"))
            elif live < spoof.reject_below:
                reasons.append(_fail("spoof_during_session", f"The video looks like a screen or print ({live:.2f})",
                                     "تصویر ویدیو به‌نظر صفحه‌نمایش یا عکس چاپی است"))
            elif live < spoof.accept_above:
                reasons.append(Reason.warn("spoof_uncertain", f"Liveness of the video is inconclusive ({live:.2f})",
                                           "زنده بودن تصویر قطعی نیست و بررسی دستی لازم است"))
            if any(r.severity is Severity.ERROR for r in reasons):
                decision = Decision.FAIL
            elif reasons:
                decision = Decision.REVIEW
            else:
                decision = Decision.PASS
            done = sum(t.done for t in session.tasks)
            data = {
                "capture_id": session.capture_id,
                "state": session.state.value,
                "tasks": [{"task": t.name, "done": t.done} for t in session.tasks],
                "task_seconds": session.task_durations,
                "frames_seen": session.frames_seen,
                "identity_min_similarity": min(session.identity) if session.identity else None,
                "identity_checks": len(session.identity),
                "live_score_median": round(live, 4) if live is not None else None,
                "live_score_min": min(session.live_scores) if session.live_scores else None,
                "antispoof_checks": len(session.live_scores),
            }
            score = done / len(session.tasks)
            if live is not None:
                score = min(score, live)
            if decision is Decision.FAIL:
                score = min(score, 0.5)
        return ModuleResult(module=MODULE_NAME, version=MODULE_VERSION, decision=decision,
                            score=max(0.0, min(1.0, score)), reasons=reasons, data=data,
                            elapsed_ms=round(timer.ms, 2))

    # ------------------------------------------------------------------ helpers

    def _session(self, session_id: str) -> LivenessSession:
        session = self.store.get(session_id)
        if session is None:
            raise SessionNotFound("Unknown or expired liveness session")
        return session

    def _feedback(self, session: LivenessSession, hint: tuple[str, str] | None, m: FrameMeasure | None = None) -> dict:
        task = session.current_task
        if session.state is LivenessState.BASELINE:
            step = "BASELINE"
        elif session.state is LivenessState.RUNNING and task is not None:
            step = task.name
        else:
            step = None
        remaining = None
        if session.state is LivenessState.RUNNING:
            remaining = max(0.0, self.settings.liveness.task_time_limit_s - (time.time() - session.task_started_at))
        feedback = {
            "state": session.state.value,
            "step": step,
            "instruction_fa": INSTRUCTIONS_FA.get(step) if step else None,
            "instruction_en": INSTRUCTIONS_EN.get(step) if step else None,
            "task_index": session.task_index,
            "total_tasks": len(session.tasks),
            "time_remaining_s": round(remaining, 1) if remaining is not None else None,
            "hint": {"code": hint[0], "message_fa": hint[1]} if hint else None,
            "failure": session.failure.model_dump() if session.failure else None,
        }
        if self.settings.debug and m is not None:
            feedback["measure"] = {"yaw": round(m.yaw, 1), "eyes": round(m.eye_openness, 3),
                                   "mouth": round(m.mouth_width, 3), "face_height": round(m.face_height, 1)}
        return feedback
