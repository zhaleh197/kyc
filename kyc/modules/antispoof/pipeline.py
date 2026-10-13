"""Passive anti-spoof: is the captured selfie a live face or a print / screen?

    capture_id -> server-held selfie + its face box
               -> each model: box jitter grid -> context crops -> P(real)
               -> live score = mean over models and jitters -> pass/review/fail

Works on the selfie face capture approved - there is no way to submit an
image here. Models and the capture lookup are injectable for the tests.
"""

from __future__ import annotations

from collections.abc import Callable

from kyc.core.config import Settings, get_settings
from kyc.core.errors import CaptureNotFound
from kyc.core.face.antispoof import AntiSpoofModel, jitter_boxes, score_face  # noqa: F401 - jitter_boxes re-exported
from kyc.core.face.registry import load_shared
from kyc.core.imaging import decode_image
from kyc.core.schemas import Decision, ModuleResult, Reason, Timer

MODULE_NAME = "antispoof"
MODULE_VERSION = "0.1.0"


class AntiSpoofPipeline:
    def __init__(
        self,
        settings: Settings | None = None,
        *,
        capture_lookup: Callable[[str], object | None],
        models_factory: Callable[[], list[AntiSpoofModel]] | None = None,
    ):
        self.settings = settings or get_settings()
        self._capture_lookup = capture_lookup
        self._models_factory = models_factory or self._load_models
        self._models: list[AntiSpoofModel] | None = None

    def _load_models(self) -> list[AntiSpoofModel]:
        s = self.settings
        models = []
        for spec in s.antispoof.models:
            filename, _, scale = spec.rpartition(":")
            path = s.model_path(filename)
            models.append(
                load_shared(
                    AntiSpoofModel, path, float(scale), providers=s.onnx_providers, intra_threads=s.onnx_intra_threads
                )
            )
        return models

    def verify(self, capture_id: str) -> ModuleResult:
        cfg = self.settings.antispoof
        capture = self._capture_lookup(capture_id)
        if capture is None:
            raise CaptureNotFound("Unknown or expired capture_id; take a new selfie")
        if self._models is None:
            self._models = self._models_factory()

        sharpness = capture.metrics.get("sharpness")
        if sharpness is not None and sharpness < cfg.min_sharpness:
            # No verdict on an image the models cannot judge: a blurred live
            # face looks like a recapture to them. Retake, not "spoof".
            with Timer() as timer:
                reasons = [
                    Reason.error(
                        "retake_low_quality",
                        f"The selfie is too blurry to check for spoofing (sharpness {sharpness:.1f}); retake it",
                        "عکس تار یا کم‌کیفیت است؛ لطفاً دوباره عکس بگیرید",
                    )
                ]
                data = {"capture_id": capture_id, "retake": True, "live_score": None, "sharpness": sharpness,
                        "min_sharpness": cfg.min_sharpness}
            return ModuleResult(module=MODULE_NAME, version=MODULE_VERSION, decision=Decision.FAIL, score=0.0,
                                reasons=reasons, data=data, elapsed_ms=round(timer.ms, 2))

        with Timer() as timer:
            image = decode_image(capture.jpeg)
            live, per_model, clamped_any = score_face(self._models, image, capture.face_box, cfg.box_jitter)

            reasons: list[Reason] = []
            if live >= cfg.accept_above:
                decision = Decision.PASS
            elif live < cfg.reject_below:
                decision = Decision.FAIL
                reasons.append(
                    Reason.error(
                        "spoof_detected",
                        f"The selfie looks like a photo or screen, not a live face (live score {live:.2f})",
                        "تصویر به‌نظر عکسِ چاپی یا صفحه‌نمایش است، نه چهرهٔ زنده",
                    )
                )
            else:
                decision = Decision.REVIEW
                reasons.append(
                    Reason.warn(
                        "spoof_uncertain",
                        f"Liveness of the selfie is inconclusive (live score {live:.2f})",
                        "زنده بودن تصویر قطعی نیست و بررسی دستی لازم است",
                    )
                )
            scores = [m["live_score"] for m in per_model]
            if len(scores) > 1 and max(scores) - min(scores) > cfg.max_model_disagreement:
                reasons.append(
                    Reason.info(
                        "models_disagree",
                        f"Anti-spoof models disagree ({min(scores):.2f} vs {max(scores):.2f})",
                        "مدل‌های تشخیص جعل نظر متفاوتی دارند",
                    )
                )
            if clamped_any:
                # Info, not warn: on a normal selfie there is rarely room for
                # 2.7x/4.0x context (spec 02 §12), so this is nearly always set.
                reasons.append(
                    Reason.info(
                        "context_clamped",
                        "The face is too close to the frame edge for the models' full context",
                        "فضای کافی دور صورت برای بررسی کامل نیست",
                    )
                )
            data = {"capture_id": capture_id, "retake": False, "live_score": round(live, 4), "models": per_model,
                    "thresholds": {"accept_above": cfg.accept_above, "reject_below": cfg.reject_below},
                    "jitter_boxes": len(jitter_boxes(capture.face_box, cfg.box_jitter))}

        score = min(live, 0.5) if decision is Decision.FAIL else live
        return ModuleResult(
            module=MODULE_NAME,
            version=MODULE_VERSION,
            decision=decision,
            score=max(0.0, min(1.0, score)),
            reasons=reasons,
            data=data,
            elapsed_ms=round(timer.ms, 2),
        )
