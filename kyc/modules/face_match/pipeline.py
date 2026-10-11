"""Face match: is the person in the captured selfie the person on the ID card?

    card portrait -> margin -> detect (rotation fallback) -> align -> embed --+
                                                                              +-> cosine -> pass/review/fail
    capture_id -> server-held selfie + its keypoints -> align -> embed ------+

The probe is only ever a selfie the face-capture module approved and kept
(specs/02-face-quality.md §2) - there is no way to submit a selfie image here.
Detector, embedder and the capture lookup are injectable, so the suite runs
without weights or real faces.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import cv2
import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import KycError
from kyc.core.face.detector import Face, FaceDetector
from kyc.core.face.embedder import FaceEmbedder, cosine
from kyc.core.imaging import decode_image
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

MODULE_NAME = "face_match"
MODULE_VERSION = "0.1.0"

# Tried in order when no face is found upright: quarter turns first (a card
# photographed sideways), then common tilts.
FALLBACK_ANGLES = (90, -90, 180, 30, -30, 60, -60, 120, -120, 150, -150)


class CaptureNotFound(KycError):
    """The capture_id is unknown or the selfie has expired - take a new selfie."""

    code = "capture_not_found"
    http_status = 404


@dataclass
class ReferenceFace:
    image: np.ndarray  # the (possibly padded and rotated) image the face was found in
    face: Face
    faces_found: int
    rotation: int


class FaceMatchPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        capture_lookup: Callable[[str], object | None],
        detector_factory: Callable[[], FaceDetector] | None = None,
        embedder_factory: Callable[[], FaceEmbedder] | None = None,
    ):
        self.settings = settings or get_settings()
        cfg = self.settings.face_match
        self._capture_lookup = capture_lookup
        self._factories = {
            "detector": detector_factory or (lambda: self._load(FaceDetector, cfg.detector_model)),
            "embedder": embedder_factory or (lambda: self._load(FaceEmbedder, cfg.embedding_model)),
        }
        self._models: dict[str, object] = {}

    def _load(self, cls, filename: str):
        s = self.settings
        return cls.load(s.model_path(filename), s.onnx_providers, s.onnx_intra_threads)

    def _model(self, name: str):
        if name not in self._models:
            self._models[name] = self._factories[name]()
        return self._models[name]

    # ------------------------------------------------------------------ public

    def verify(self, reference: np.ndarray, capture_id: str) -> ModuleResult:
        cfg = self.settings.face_match
        capture = self._capture_lookup(capture_id)
        if capture is None:
            raise CaptureNotFound("Unknown or expired capture_id; take a new selfie")

        reasons: list[Reason] = []
        with Timer() as timer:
            ref = self._find_reference(reference)
            data: dict = {
                "capture_id": capture_id,
                "model": cfg.embedding_model,
                "thresholds": {"accept_above": cfg.accept_above, "reject_below": cfg.reject_below},
                "similarity": None,
                "reference_face": None,
            }
            if ref is None:
                reasons.append(
                    Reason.error(
                        "reference_no_face",
                        "No face found in the ID card portrait",
                        "چهره‌ای در عکس کارت ملی پیدا نشد؛ عکس واضح‌تری از کارت بگیرید",
                    )
                )
                return _envelope(Decision.FAIL, 0.0, reasons, data, timer)

            data["reference_face"] = {
                "box": [round(v, 1) for v in ref.face.box],
                "det_score": round(ref.face.score, 4),
                "height_px": round(ref.face.height, 1),
                "faces_found": ref.faces_found,
                "rotation": ref.rotation,
            }
            if ref.faces_found > 1:
                reasons.append(
                    Reason.warn(
                        "reference_multiple_faces",
                        "More than one face in the card portrait; the largest was used",
                        "بیش از یک چهره در عکس کارت دیده شد",
                    )
                )
            if ref.face.height < cfg.min_reference_face_px:
                reasons.append(
                    Reason.warn(
                        "reference_low_quality",
                        f"Card portrait face is only {ref.face.height:.0f}px tall",
                        "عکس روی کارت خیلی کوچک یا کم‌کیفیت است",
                    )
                )

            embedder = self._model("embedder")
            ref_vec = embedder.embed(ref.image, ref.face)
            selfie = decode_image(capture.jpeg)
            selfie_face = Face(box=tuple(capture.face_box), score=1.0, kps=np.asarray(capture.kps, np.float32))
            similarity = cosine(ref_vec, embedder.embed(selfie, selfie_face))
            data["similarity"] = round(similarity, 4)

            if similarity >= cfg.accept_above:
                decision = Decision.PASS
            elif similarity < cfg.reject_below:
                decision = Decision.FAIL
                reasons.append(
                    Reason.error(
                        "faces_do_not_match",
                        f"Selfie does not match the ID card (similarity {similarity:.2f})",
                        "چهرهٔ سلفی با عکس کارت ملی مطابقت ندارد",
                    )
                )
            else:
                decision = Decision.REVIEW
                reasons.append(
                    Reason.warn(
                        "faces_match_uncertain",
                        f"Selfie and ID card similarity is inconclusive ({similarity:.2f})",
                        "تطابق چهره با کارت ملی قطعی نیست و بررسی دستی لازم است",
                    )
                )
            if decision is Decision.PASS and any(r.severity is Severity.WARN for r in reasons):
                decision = Decision.REVIEW

        return _envelope(decision, similarity, reasons, data, timer)

    # ---------------------------------------------------------------- reference

    def _find_reference(self, portrait: np.ndarray) -> ReferenceFace | None:
        cfg = self.settings.face_match
        detector = self._model("detector")
        h, w = portrait.shape[:2]
        my, mx = int(cfg.reference_margin * h), int(cfg.reference_margin * w)
        padded = cv2.copyMakeBorder(portrait, my, my, mx, mx, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        angles = (0, *FALLBACK_ANGLES) if cfg.rotation_fallback else (0,)
        for angle in angles:
            image = padded if angle == 0 else _rotate(padded, angle)
            faces = detector.detect(image, cfg.detector_conf)
            if faces:
                return ReferenceFace(image=image, face=faces[0], faces_found=len(faces), rotation=angle)
        return None


def _rotate(img: np.ndarray, angle: float) -> np.ndarray:
    """Rotate about the centre, growing the canvas so nothing is cut off."""
    h, w = img.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    cos, sin = abs(m[0, 0]), abs(m[0, 1])
    nw, nh = int(h * sin + w * cos), int(h * cos + w * sin)
    m[0, 2] += nw / 2 - w / 2
    m[1, 2] += nh / 2 - h / 2
    return cv2.warpAffine(img, m, (nw, nh), borderValue=(0, 0, 0))


def _envelope(decision: Decision, score: float, reasons: list[Reason], data: dict, timer: Timer) -> ModuleResult:
    score = max(0.0, min(1.0, score))
    if decision is Decision.FAIL:
        score = min(score, 0.5)
    return ModuleResult(
        module=MODULE_NAME,
        version=MODULE_VERSION,
        decision=decision,
        score=score,
        reasons=reasons,
        data=data,
        elapsed_ms=round(timer.ms, 2),
    )
