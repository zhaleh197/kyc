"""Passive anti-spoof: is the captured selfie a live face or a print / screen?

    capture_id -> server-held selfie + its face box
               -> each model: box jitter grid -> context crops -> P(real)
               -> live score = mean over models and jitters -> pass/review/fail

Works on the selfie face capture approved - there is no way to submit an
image here. Models and the capture lookup are injectable for the tests.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import CaptureNotFound
from kyc.core.face.antispoof import AntiSpoofModel, context_crop
from kyc.core.imaging import decode_image
from kyc.core.schemas import Decision, ModuleResult, Reason, Timer

MODULE_NAME = "antispoof"
MODULE_VERSION = "0.1.0"


def jitter_boxes(box: tuple[float, float, float, float], amount: float) -> list[tuple[float, float, float, float]]:
    """The box itself plus a 3x3 grid of size (-amount, 0, +amount) x
    horizontal shift (-amount, 0, +amount) variants."""
    if amount <= 0:
        return [box]
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    out = []
    for ds in (-amount, 0.0, amount):
        for dx in (-amount, 0.0, amount):
            nw, nh = w * (1 + ds), h * (1 + ds)
            ncx = cx + dx * w
            out.append((ncx - nw / 2, cy - nh / 2, ncx + nw / 2, cy + nh / 2))
    return out


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
            models.append(AntiSpoofModel.load(path, float(scale), s.onnx_providers, s.onnx_intra_threads))
        return models

    def verify(self, capture_id: str) -> ModuleResult:
        cfg = self.settings.antispoof
        capture = self._capture_lookup(capture_id)
        if capture is None:
            raise CaptureNotFound("Unknown or expired capture_id; take a new selfie")
        if self._models is None:
            self._models = self._models_factory()

        with Timer() as timer:
            image = decode_image(capture.jpeg)
            boxes = jitter_boxes(tuple(capture.face_box), cfg.box_jitter)
            per_model, clamped_any = [], False
            for model in self._models:
                crops, clamped = zip(*(context_crop(image, b, model.scale) for b in boxes), strict=True)
                probs = model.live_probabilities(list(crops))
                clamped_any |= clamped[0]  # the unjittered box
                per_model.append(
                    {
                        "name": model.name,
                        "context_scale": model.scale,
                        "live_score": round(float(probs.mean()), 4),
                        "live_score_min": round(float(probs.min()), 4),
                        "context_clamped": bool(clamped[0]),
                    }
                )
            live = float(np.mean([m["live_score"] for m in per_model]))

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
            data = {"capture_id": capture_id, "live_score": round(live, 4), "models": per_model,
                    "thresholds": {"accept_above": cfg.accept_above, "reject_below": cfg.reject_below},
                    "jitter_boxes": len(boxes)}

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
