"""Document OCR pipeline.

    photo -> downscale -> rectify card -> quality gate -> detect fields
          -> crop -> recognise -> normalise -> validate -> ModuleResult

The pipeline is profile-driven: it never mentions Iran. Supporting the Iraqi
card later is a matter of registering a profile and shipping its detector.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.imaging import (
    crop_with_padding,
    decode_image,
    encode_jpeg_base64,
    ensure_min_size,
    upscale_to_height,
)
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

from . import normalize, rectify
from .detector import Detection, FieldDetector, best_per_class
from .profiles import DocumentProfile, FieldKind, FieldSpec, get_profile
from .recognizers import TextRecognizer, get_recognizer
from .schemas import DocumentOcrData, FieldResult

MODULE_NAME = "ocr"
MODULE_VERSION = "0.1.0"

# Quality problems that make any downstream reading untrustworthy rather than
# merely uncertain. Reading on would produce confident nonsense, which is the
# worst possible failure mode for identity data.
FATAL_QUALITY_CODES = {"card_blurry", "card_too_dark", "card_too_bright"}


DetectorFactory = Callable[[DocumentProfile], "FieldDetector"]
RecognizerFactory = Callable[[str], "TextRecognizer"]


class DocumentOcrPipeline:
    """The detector and recogniser are injectable.

    In production both factories default to the ONNX/registry implementations.
    Being able to substitute them is what makes the pipeline testable without
    shipping model weights, and it is also how a caller pins a specific model
    version per tenant.
    """

    def __init__(
        self,
        settings: Settings | None = None,
        *,
        detector_factory: DetectorFactory | None = None,
        recognizer_factory: RecognizerFactory | None = None,
    ):
        self.settings = settings or get_settings()
        self._detector_factory = detector_factory or self._default_detector
        self._recognizer_factory = recognizer_factory or self._default_recognizer

    def _default_detector(self, profile: DocumentProfile) -> FieldDetector:
        return FieldDetector.load(
            self.settings.model_path(profile.detector_model),
            profile.class_names,
            self.settings.onnx_providers,
            self.settings.onnx_intra_threads,
        )

    def _default_recognizer(self, backend_id: str) -> TextRecognizer:
        return get_recognizer(backend_id, self.settings)

    # ------------------------------------------------------------------ public

    def run_bytes(
        self,
        data: bytes,
        profile_id: str | None = None,
        *,
        return_portrait: bool = False,
        return_card: bool = False,
    ) -> ModuleResult:
        return self.run(
            decode_image(data),
            profile_id,
            return_portrait=return_portrait,
            return_card=return_card,
        )

    def run(
        self,
        image: np.ndarray,
        profile_id: str | None = None,
        *,
        return_portrait: bool = False,
        return_card: bool = False,
    ) -> ModuleResult:
        profile = get_profile(profile_id)
        cfg = self.settings.ocr
        reasons: list[Reason] = []

        with Timer() as timer:
            ensure_min_size(image, cfg.min_image_side)
            image = rectify.prepare(image, cfg.max_image_side)

            if cfg.rectify:
                card, rectify_reasons = rectify.rectify(image, profile.rectified_size, profile.aspect_ratio)
                reasons += rectify_reasons
            else:
                # The detector was trained on whole frames, so it must see one.
                # Field boxes are then reported in this frame's coordinates.
                card = image

            quality_reasons, metrics = rectify.assess_quality(
                card,
                min_sharpness=cfg.min_sharpness,
                min_brightness=cfg.min_brightness,
                max_brightness=cfg.max_brightness,
                max_glare_ratio=cfg.max_glare_ratio,
            )
            reasons += quality_reasons

            if any(r.code in FATAL_QUALITY_CODES for r in quality_reasons):
                empty = DocumentOcrData(
                    profile=profile.id,
                    country=profile.country,
                    quality=metrics,
                    missing_fields=[f.key for f in profile.value_fields],
                    card_base64=encode_jpeg_base64(card) if return_card else None,
                )
                return self._envelope(Decision.FAIL, 0.0, reasons, empty, timer)

            detections = self._detect(card, profile)
            located = best_per_class(detections)

            fields, field_reasons = self._read_fields(card, profile, located)
            reasons += field_reasons

            values = {key: field.value for key, field in fields.items()}
            reasons += profile.validate(values)

            data = DocumentOcrData(
                profile=profile.id,
                country=profile.country,
                fields=fields,
                values=values,
                derived=profile.enrich(values),
                missing_fields=[f.key for f in profile.value_fields if not values.get(f.key)],
                quality=metrics,
                portrait_base64=self._portrait(card, profile, located) if return_portrait else None,
                card_base64=encode_jpeg_base64(card) if return_card else None,
            )
            decision, score = self._decide(reasons, fields)

        return self._envelope(decision, score, reasons, data, timer)

    # ------------------------------------------------------------------ stages

    def _detect(self, card: np.ndarray, profile: DocumentProfile) -> list[Detection]:
        cfg = self.settings.ocr
        detector = self._detector_factory(profile)
        return detector.detect(card, cfg.detector_conf, cfg.detector_iou)

    def _read_fields(
        self,
        card: np.ndarray,
        profile: DocumentProfile,
        located: dict[str, Detection],
    ) -> tuple[dict[str, FieldResult], list[Reason]]:
        cfg = self.settings.ocr
        reasons: list[Reason] = []
        results: dict[str, FieldResult] = {}

        for spec in profile.value_fields:
            result = FieldResult(
                key=spec.key,
                label_fa=spec.label_fa,
                label_en=spec.label_en,
                kind=spec.kind.value,
            )
            detection = located.get(spec.detector_class)
            if detection is None:
                if spec.required:
                    reasons.append(
                        Reason.error(
                            "field_not_located",
                            f"Field {spec.key} was not found on the card",
                            f"محل «{spec.label_fa}» روی کارت پیدا نشد",
                            spec.key,
                        )
                    )
                results[spec.key] = result
                continue

            result.box = list(detection.box)
            result.detection_confidence = round(detection.confidence, 4)

            crop = crop_with_padding(card, detection.box, cfg.field_crop_padding)
            crop = upscale_to_height(crop, cfg.upscale_small_crops_to)

            backend_id = cfg.digit_backend if spec.kind in (FieldKind.DIGITS, FieldKind.DATE) else cfg.text_backend
            recognizer = self._recognizer_factory(backend_id)
            recognition = recognizer.recognize(crop, allowlist=spec.charset_hint)

            result.backend = backend_id
            result.raw = recognition.text
            result.confidence = round(recognition.confidence, 4)
            result.value = self._normalize(spec, recognition.text)
            result.low_confidence = recognition.confidence < cfg.min_field_confidence

            reasons += self._field_findings(spec, result)
            results[spec.key] = result

        return results, reasons

    @staticmethod
    def _field_findings(spec: FieldSpec, result: FieldResult) -> list[Reason]:
        found: list[Reason] = []
        if result.low_confidence and result.value:
            found.append(
                Reason.warn(
                    "field_low_confidence",
                    f"Low recognition confidence on {spec.key} ({result.confidence})",
                    f"اطمینان پایین در خواندن «{spec.label_fa}»",
                    spec.key,
                )
            )
        if spec.exact_len and result.value and len(result.value) != spec.exact_len:
            found.append(
                Reason.warn(
                    "field_length_mismatch",
                    f"{spec.key} has {len(result.value)} characters, expected {spec.exact_len}",
                    f"طول «{spec.label_fa}» برابر {len(result.value)} است، انتظار {spec.exact_len} بود",
                    spec.key,
                )
            )
        return found

    @staticmethod
    def _normalize(spec: FieldSpec, raw: str) -> str | None:
        if spec.kind is FieldKind.DIGITS:
            value = normalize.keep_digits(raw)
        elif spec.kind is FieldKind.DATE:
            value = normalize.normalize_date(raw)
        else:
            value = normalize.normalize_name(raw)
        return value or None

    @staticmethod
    def _portrait(card: np.ndarray, profile: DocumentProfile, located: dict[str, Detection]) -> str | None:
        for spec in profile.fields:
            if spec.kind is FieldKind.IMAGE and spec.detector_class in located:
                # Tight padding: this crop feeds face-match, and card border
                # pixels only add noise to the embedding.
                return encode_jpeg_base64(crop_with_padding(card, located[spec.detector_class].box, 0.02))
        return None

    # ---------------------------------------------------------------- decision

    def _decide(self, reasons: list[Reason], fields: dict[str, FieldResult]) -> tuple[Decision, float]:
        """Errors fail outright; warnings route to manual review.

        The score averages recognition confidence over *expected* fields, not
        over the ones that happened to read, so a card where three of six
        fields came back empty scores worse than one where all six read.
        """
        expected = max(1, len(fields))
        score = sum(f.confidence for f in fields.values() if f.value) / expected

        if any(r.severity is Severity.ERROR for r in reasons):
            # Cap a failing result well below any passing one so callers can
            # rank by score without re-reading the decision.
            return Decision.FAIL, round(min(score, 0.5), 4)
        if any(r.severity is Severity.WARN for r in reasons) and self.settings.ocr.review_on_low_confidence:
            return Decision.REVIEW, round(score, 4)
        return Decision.PASS, round(score, 4)

    @staticmethod
    def _envelope(
        decision: Decision,
        score: float,
        reasons: list[Reason],
        data: DocumentOcrData,
        timer: Timer,
    ) -> ModuleResult:
        return ModuleResult(
            module=MODULE_NAME,
            version=MODULE_VERSION,
            decision=decision,
            score=max(0.0, min(1.0, score)),
            reasons=reasons,
            data=data.model_dump(),
            elapsed_ms=round(timer.ms, 2),
        )
