"""Per-frame selfie checks.

Everything is measured on the *face*, never on the whole frame: the OCR
module's whole-frame sharpness was dragged down by background and had to be
recalibrated after the fact (specs/01-document-ocr.md §9). Each finding is a
`Reason` whose Persian message is the instruction shown to the user.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from kyc.core.config import FaceQualitySettings
from kyc.core.face.detector import Face
from kyc.core.face.geometry import eye_openness, head_pose, iris_offsets, mouth_openness
from kyc.core.schemas import Reason, Severity

CROP_SIZE = 256


@dataclass
class FrameAssessment:
    reasons: list[Reason] = field(default_factory=list)
    metrics: dict[str, float] = field(default_factory=dict)
    face: Face | None = None

    @property
    def ok(self) -> bool:
        """No blocking finding. Warnings (e.g. uneven light) do not block capture."""
        return not any(r.severity.value == "error" for r in self.reasons)


def _err(code: str, en: str, fa: str) -> Reason:
    return Reason.error(code, en, fa)


def context_scale(face: Face, frame_w: int, frame_h: int) -> float:
    """Largest factor the face box can be enlarged by, about its centre, and
    still fit inside the frame (what MiniFASNet's 2.7x / 4.0x crops need)."""
    cx, cy = (face.box[0] + face.box[2]) / 2, (face.box[1] + face.box[3]) / 2
    w, h = max(face.width, 1.0), max(face.height, 1.0)
    return float(min(2 * cx / w, 2 * (frame_w - cx) / w, 2 * cy / h, 2 * (frame_h - cy) / h))


def face_crop_gray(img: np.ndarray, face: Face) -> np.ndarray:
    x1, y1, x2, y2 = face.int_box()
    crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def background_edge_density(img: np.ndarray, face: Face) -> tuple[float, float]:
    """(edge density, fraction of frame used) on the background beside and
    above the head, at most down to chin level so the torso is excluded.

    Model-free approximation of the prototype's rembg/u2netp segmentation:
    the head box is widened generously for hair and headscarves and cut out.
    """
    scale = 640 / max(img.shape[:2])
    gray = cv2.cvtColor(cv2.resize(img, None, fx=scale, fy=scale), cv2.COLOR_BGR2GRAY)
    h, w = gray.shape
    x1, y1, x2, y2 = (v * scale for v in face.box)
    fw, fh = x2 - x1, y2 - y1
    chin = int(min(h, y2))
    mask = np.zeros((h, w), bool)
    mask[:chin, :] = True
    mask[int(max(0, y1 - 0.6 * fh)) : chin, int(max(0, x1 - 0.45 * fw)) : int(min(w, x2 + 0.45 * fw))] = False
    fraction = float(mask.mean())
    if not mask.any():
        return 0.0, 0.0
    edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 40, 120) > 0
    return float(edges[mask].mean()), fraction


def assess(
    img: np.ndarray,
    faces: list[Face],
    landmarks: np.ndarray | None,
    cfg: FaceQualitySettings,
    clear_logit: float | None = None,
) -> FrameAssessment:
    """`faces` largest first; `landmarks` are the 106 points of faces[0];
    `clear_logit` is the occlusion classifier's output for faces[0], if run."""
    out = FrameAssessment()
    h, w = img.shape[:2]

    if not faces:
        out.reasons.append(_err("no_face", "No face found", "چهره پیدا نشد؛ رو به دوربین قرار بگیرید"))
        return out

    face = out.face = faces[0]
    m = out.metrics
    m["face_count"] = float(len(faces))
    m["det_score"] = round(face.score, 4)

    others = [f for f in faces[1:] if f.height >= cfg.multiple_face_min_ratio * face.height]
    if others:
        out.reasons.append(
            _err("multiple_faces", "More than one face in frame", "فقط خودتان در کادر باشید")
        )

    # --- framing
    ratio = face.height / h
    m["face_height_ratio"] = round(ratio, 4)
    if ratio < cfg.min_face_height_ratio:
        out.reasons.append(_err("face_too_small", "Face is too small in frame", "نزدیک‌تر بیایید"))
    elif ratio > cfg.max_face_height_ratio:
        out.reasons.append(_err("face_too_large", "Face is too close", "کمی عقب‌تر بروید"))

    cx = (face.box[0] + face.box[2]) / 2 / w - 0.5
    cy = (face.box[1] + face.box[3]) / 2 / h - 0.5
    m["center_offset"] = round(float(np.hypot(cx, cy)), 4)
    x1, y1, x2, y2 = face.box
    cut_off = x1 <= 1 or y1 <= 1 or x2 >= w - 1 or y2 >= h - 1
    if cut_off:
        out.reasons.append(_err("face_cut_off", "Face is cut off by the frame edge", "تمام صورت در کادر باشد"))
    elif m["center_offset"] > cfg.max_center_offset:
        out.reasons.append(_err("face_off_center", "Face is off-centre", "صورت را وسط کادر بیاورید"))

    m["context_scale"] = round(context_scale(face, w, h), 3)
    if cfg.min_context_scale and m["context_scale"] < cfg.min_context_scale and not cut_off:
        out.reasons.append(
            _err("face_too_close_for_context", "Not enough background around the face", "کمی عقب‌تر بروید")
        )

    # --- image quality, on the face only
    gray = face_crop_gray(img, face)
    m["sharpness"] = round(float(cv2.Laplacian(gray, cv2.CV_64F).var()), 2)
    mean = float(gray.mean())
    m["brightness"] = round(mean, 2)
    left, right = gray[:, : CROP_SIZE // 2].mean(), gray[:, CROP_SIZE // 2 :].mean()
    m["illumination_asymmetry"] = round(float(abs(left - right) / max(mean, 1.0)), 4)

    if mean < cfg.min_brightness:
        out.reasons.append(_err("face_too_dark", "Face is too dark", "نور کافی نیست؛ به جای روشن‌تری بروید"))
    elif mean > cfg.max_brightness:
        out.reasons.append(_err("face_too_bright", "Face is overexposed", "نور زیاد است؛ از نور مستقیم دور شوید"))
    elif m["sharpness"] < cfg.min_sharpness:
        # Judged only at acceptable exposure: darkening a sharp photo to 25%
        # drops its Laplacian variance below the floor too, and the user was
        # then told "blurry" when the fix was more light.
        out.reasons.append(_err("face_blurry", "Face is blurry", "تصویر تار است؛ دوربین را ثابت نگه دارید"))
    if m["illumination_asymmetry"] > cfg.max_illumination_asymmetry:
        # Info, not warn: 6 of 15 ordinary indoor test photos measured above
        # 0.5 on both the whole face box and its inner region - one-sided
        # room light is normal. Promote only if it is shown to hurt matching.
        out.reasons.append(
            Reason.info("uneven_lighting", "One side of the face is much darker", "نور یکنواخت نیست؛ رو به نور بایستید")
        )

    # --- pose
    yaw, pitch, roll = head_pose(face.kps)
    m["yaw"], m["pitch"], m["roll"] = round(yaw, 2), round(pitch, 2), round(roll, 2)
    if abs(yaw) > cfg.max_yaw_deg or abs(pitch) > cfg.max_pitch_deg:
        out.reasons.append(_err("head_turned", "Head is turned away", "مستقیم به دوربین نگاه کنید"))
    if abs(roll) > cfg.max_roll_deg:
        out.reasons.append(_err("head_tilted", "Head is tilted", "سرتان را صاف نگه دارید"))

    # --- eyes
    if landmarks is not None:
        left_eye, right_eye = eye_openness(landmarks)
        m["eye_openness_left"], m["eye_openness_right"] = round(left_eye, 4), round(right_eye, 4)
        if min(left_eye, right_eye) < cfg.min_eye_openness:
            out.reasons.append(_err("eyes_closed", "Eyes are closed", "چشم‌هایتان را باز نگه دارید"))

        m["mouth_openness"] = round(mouth_openness(landmarks), 4)
        if m["mouth_openness"] > cfg.max_mouth_openness:
            out.reasons.append(_err("mouth_open", "Mouth is open", "دهانتان را ببندید"))

        offsets = iris_offsets(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), landmarks)
        if offsets:
            m["gaze_offset"] = round(float(np.mean(offsets)), 4)
            # Only meaningful with the head roughly frontal; a turned head
            # already has its own hint.
            if abs(m["gaze_offset"]) > cfg.max_gaze_offset and abs(yaw) <= cfg.max_yaw_deg:
                out.reasons.append(_err("gaze_off_center", "Not looking at the camera", "مستقیم به دوربین نگاه کنید"))

    # --- occlusion
    if clear_logit is not None:
        m["clear_logit"] = round(clear_logit, 3)
        if clear_logit < cfg.min_clear_logit:
            out.reasons.append(
                _err(
                    "face_occluded",
                    "Something is covering the face",
                    "ماسک، عینک آفتابی یا دست را از جلوی صورت بردارید",
                )
            )

    # --- background
    if cfg.background_check != "off":
        density, fraction = background_edge_density(img, face)
        m["background_fraction"] = round(fraction, 3)
        if fraction >= cfg.min_background_fraction:
            m["background_edge_density"] = round(density, 4)
            if density > cfg.max_background_edge_density:
                out.reasons.append(
                    Reason(
                        code="background_not_uniform",
                        severity=Severity(cfg.background_check),
                        message_en="Background is not plain",
                        message_fa="پس‌زمینه یکدست نیست؛ جلوی دیوار ساده بایستید",
                    )
                )

    return out


def quality_score(metrics: dict[str, float], cfg: FaceQualitySettings) -> float:
    """0-1, monotonic in every metric: the mean headroom of each check.

    Used to pick the best of the passing frames and as the module score.
    """

    def clip(x: float) -> float:
        return float(min(1.0, max(0.0, x)))

    parts = [
        clip(metrics.get("sharpness", 0.0) / (4 * cfg.min_sharpness)),
        clip(1 - abs(metrics.get("yaw", 0.0)) / cfg.max_yaw_deg),
        clip(1 - abs(metrics.get("pitch", 0.0)) / cfg.max_pitch_deg),
        clip(1 - abs(metrics.get("roll", 0.0)) / cfg.max_roll_deg),
        clip(metrics.get("det_score", 0.0)),
    ]
    if "eye_openness_left" in metrics:
        eyes = min(metrics["eye_openness_left"], metrics["eye_openness_right"])
        parts.append(clip(eyes / (2 * cfg.min_eye_openness)))
    return round(sum(parts) / len(parts), 4)
