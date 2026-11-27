"""Document OCR pipeline.

    photo -> downscale -> rectify card -> quality gate -> detect fields
          -> crop -> recognise -> normalise -> validate -> ModuleResult

The pipeline is profile-driven: it never mentions Iran. Supporting the Iraqi
card later is a matter of registering a profile and shipping its detector.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace

import cv2
import numpy as np

from kyc.core.config import Settings, get_settings
from kyc.core.errors import KycError
from kyc.core.imaging import (
    crop_with_padding,
    decode_image,
    encode_jpeg_base64,
    ensure_min_size,
    upscale_to_height,
)
from kyc.core.schemas import Decision, ModuleResult, Reason, Severity, Timer

from . import lexicon, normalize, rectify
from .rows import place_rows
from .detector import Detection, FieldDetector, best_per_class
from .orient import orient
from .profiles import DocumentProfile, FieldKind, FieldSpec, get_profile
from .recognizers import TextRecognizer, get_recognizer
from .schemas import DocumentOcrData, FieldResult

MODULE_NAME = "ocr"
MODULE_VERSION = "0.1.0"

def _ink_bands(crop: np.ndarray) -> tuple[np.ndarray, list[list[int]]]:
    """Rows with ink (dark on a light card) grouped into bands, top to bottom;
    a gap under 6% of the height joins two bands (dots above / below a line)."""
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    _, ink = cv2.threshold(gray, 0, 1, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    rows = ink.sum(1) > max(2, 0.02 * ink.shape[1])
    h = len(rows)
    bands, start = [], None
    for y, on in enumerate(np.r_[rows, False]):
        if on and start is None:
            start = y
        elif not on and start is not None:
            if bands and start - bands[-1][1] < 0.06 * h:
                bands[-1][1] = y
            else:
                bands.append([start, y])
            start = None
    return ink, bands


def drop_neighbour_rows(crop: np.ndarray) -> np.ndarray:
    """The crop without the edges of the rows above and below.

    Rows on the Iraqi card sit close, so a field box padded for tall letters
    takes in the tail of the line above (a final ي) or the top of the line
    below, and the recogniser reads it as one more letter. Ink bands that
    touch the top or bottom edge and are smaller than the main line are cut
    away; the main line keeps a little margin.
    """
    ink, bands = _ink_bands(crop)
    if len(bands) < 2:
        return crop
    h = ink.shape[0]
    weight = [int(ink[a:b].sum()) for a, b in bands]
    main = int(np.argmax(weight))
    top, bottom = 0, h
    for i, (a, b) in enumerate(bands):
        if i == main or weight[i] >= weight[main]:
            continue
        if a == 0 and i < main:
            top = max(top, (b + bands[main][0]) // 2)
        elif b == h and i > main:
            bottom = min(bottom, (bands[main][1] + a + 1) // 2)
    return crop[top:bottom] if bottom - top >= 0.4 * h else crop


def text_lines(crop: np.ndarray, min_share: float = 0.2) -> list[np.ndarray]:
    """The crop cut into its text lines (dark ink on a light card), top to
    bottom; a band shorter than `min_share` of the crop is a stray dot or the
    edge of a neighbouring row."""
    ink, bands = _ink_bands(crop)
    h = ink.shape[0]
    bands = [(a, b) for a, b in bands if b - a >= min_share * h]
    if len(bands) < 2:
        # Lines set so close that no blank row parts them: cut at the row
        # with the least ink in the middle of the crop.
        counts = ink.sum(1)
        lo, hi = int(0.3 * h), int(0.7 * h)
        if hi - lo < 2:
            return []
        cut = lo + int(np.argmin(counts[lo:hi]))
        bands = [(0, cut), (cut, h)]
    pad = int(0.05 * h)
    return [crop[max(0, a - pad):min(h, b + pad)] for a, b in bands]



def trim_label_colon(crop: np.ndarray) -> np.ndarray:
    """The crop without a label's colon (:) at its right edge.

    Iraqi cards print the label, a colon, then the value; the detector's box
    for the value often takes the colon in, and a recogniser reads its two
    dots as a letter (ب, إ, ی). A colon is two small, dot-sized blobs one
    above the other, with only background between them and the text.
    """
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    h, w = gray.shape
    _, ink = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    n, _, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    dot_max, dot_min = 0.25 * h, 0.04 * h
    dots = [s for s in stats[1:]
            if s[0] >= 0.6 * w and dot_min <= s[2] <= dot_max and dot_min <= s[3] <= dot_max
            and 0.5 <= s[2] / s[3] <= 2.0 and s[4] >= 0.5 * s[2] * s[3]]
    for i, a in enumerate(dots):
        for b in dots[i + 1:]:
            top, low = (a, b) if a[1] < b[1] else (b, a)
            overlap = min(top[0] + top[2], low[0] + low[2]) - max(top[0], low[0])
            if overlap < 0.5 * min(top[2], low[2]) or low[1] - (top[1] + top[3]) < 0.5 * top[3]:
                continue
            left = min(top[0], low[0])
            # Anything bigger than a dot inked at or right of the colon means it
            # is part of the text. (Dots are not: the next row's colon is often cut in.)
            others = [s for s in stats[1:] if s[4] >= 0.3 * min(top[4], low[4])
                      and s[0] + s[2] > left and not any((s == d).all() for d in dots)]
            if others:
                continue
            cut = max(1, left - int(0.04 * h))
            return crop[:, :cut]
    return crop

# The detector's card box covering this much of the frame means the frame is the card.
CARD_FILLS_FRAME = 0.8
# Quality problems that make any downstream reading untrustworthy rather than
# merely uncertain. Reading on would produce confident nonsense, which is the
# worst possible failure mode for identity data.
FATAL_QUALITY_CODES = {"card_blurry", "card_too_dark"}
# Lighting findings that matter only if they cost a field (see run).
LIGHTING_CODES = {"card_too_bright", "card_glare"}


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
        self._matchers: dict[str, lexicon.NameMatcher] = {}

    def _default_detector(self, profile: DocumentProfile) -> FieldDetector:
        return FieldDetector.load(
            self.settings.model_path(profile.detector_model),
            profile.class_names,
            self.settings.onnx_providers,
            self.settings.onnx_intra_threads,
        )

    def _default_recognizer(self, backend_id: str) -> TextRecognizer:
        return get_recognizer(backend_id, self.settings)

    def _matcher(self, name: str) -> lexicon.NameMatcher:
        cfg = self.settings.ocr
        if name not in self._matchers:
            self._matchers[name] = lexicon.NameMatcher(
                lexicon.load(name), min_score=cfg.lexicon_min_score, min_margin=cfg.lexicon_min_margin
            )
        return self._matchers[name]

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
            source_size = (image.shape[1], image.shape[0])
            if min(source_size) < cfg.small_image_side:
                w, h = source_size
                reasons.append(Reason.warn(
                    "image_small",
                    f"Small image ({w}x{h}); read anyway, a person should check it "
                    f"(at least {cfg.small_image_side}px on the short side is expected)",
                    f"تصویر کوچک است ({w}×{h} پیکسل)؛ خوانده شد ولی باید بررسی شود "
                    f"(ضلع کوتاه دست‌کم {cfg.small_image_side} پیکسل انتظار می‌رود)",
                ))
            image = rectify.prepare(image, cfg.max_image_side)
            # source photo -> the frame the boxes end up in
            to_frame = rectify.scale_matrix(source_size, (image.shape[1], image.shape[0]))

            # Serve the geometry the profile's detector was trained on.
            if cfg.rectify if profile.rectify is None else profile.rectify:
                card, rectify_reasons, to_card = rectify.rectify_with_transform(
                    image, profile.rectified_size, profile.aspect_ratio
                )
                to_frame = to_card @ to_frame
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

            card, detections, to_frame, orientation = self._locate(card, profile, to_frame)
            located = best_per_class(detections)
            if any(r.code == "card_border_not_found" for r in reasons):
                card, located, to_frame, orientation, reasons = self._card_from_detector(
                    card, profile, located, to_frame, orientation, reasons
                )

            empty_rows: dict[str, tuple[int, int, int, int]] = {}
            if profile.row_order:
                placement = place_rows(located, profile.row_order)
                located, empty_rows = placement.located, placement.boxes
                by_class = {f.detector_class: f for f in profile.fields}
                for was, row in placement.moved:
                    a, b = by_class.get(was), by_class.get(row)
                    if a and b:
                        reasons.append(Reason.info(
                            "field_row_corrected",
                            f"The box labelled {a.key} is on the {b.key} row; read as {b.key}",
                            f"کادری که «{a.label_fa}» تشخیص داده شده بود روی ردیف «{b.label_fa}» است و همان خوانده شد",
                            b.key,
                        ))

            fields, field_reasons = self._read_fields(card, profile, located, empty_rows)
            reasons += field_reasons
            source_boxes = {}
            # Forensics (06) reads the submitted pixels: a downscale, a deskew
            # or a rectification resamples them and erases the traces it
            # looks for, so the boxes are mapped back onto the photo.
            back = np.linalg.inv(to_frame)
            for spec in profile.fields:
                if spec.detector_class in located:
                    mapped = _map_box(list(located[spec.detector_class].box), back, source_size)
                    if mapped:
                        source_boxes[spec.key] = mapped

            values = {key: field.value for key, field in fields.items()}
            reasons += profile.validate(values)
            if all(values.get(f.key) for f in profile.value_fields) and not any(
                r.severity is not Severity.INFO for r in reasons if r.code not in LIGHTING_CODES
            ):
                # Every field read cleanly, so the light was good enough: a
                # bright card or a shine away from the text is no reason for
                # a person to look. Kept as info, with the metrics.
                reasons = [Reason.info(r.code, r.message_en, r.message_fa, r.field) if r.code in LIGHTING_CODES
                           else r for r in reasons]

            data = DocumentOcrData(
                profile=profile.id,
                country=profile.country,
                source_size=list(source_size),
                fields=fields,
                source_boxes=source_boxes,
                values=values,
                derived=profile.enrich(values),
                missing_fields=[f.key for f in profile.value_fields if not values.get(f.key)],
                quality=metrics,
                portrait_base64=self._portrait(card, profile, located) if return_portrait else None,
                card_base64=encode_jpeg_base64(card) if return_card else None,
                orientation=orientation,
            )
            decision, score = self._decide(reasons, fields)

        return self._envelope(decision, score, reasons, data, timer)

    # ------------------------------------------------------------------ stages

    def _locate(self, card: np.ndarray, profile: DocumentProfile, to_frame: np.ndarray):
        """(card, detections, source->card matrix, orientation info)."""
        cfg = self.settings.ocr
        if not cfg.auto_orient:
            return card, self._detect(card, profile), to_frame, None
        # Straighten a sideways / upside-down / tilted card before any field
        # is cropped: a field cut from a sideways card holds vertical text the
        # recognisers cannot read. Boxes, the portrait and `card_base64` all
        # refer to the upright image.
        detector = self._detector_factory(profile)  # one instance for every candidate turn
        o = orient(card, lambda im: detector.detect(im, cfg.detector_conf, cfg.detector_iou), profile)
        info = {"rotation": o.rotation, "deskew_deg": o.deskew_deg, "layout_score": round(o.layout_score, 2)}
        return o.image, o.detections, o.matrix @ to_frame, info

    def _card_from_detector(self, card, profile, located, to_frame, orientation, reasons):
        """The outline finder saw no card edge - usually because the image *is*
        the card (a scan or a tight crop), with no background around it. The
        field detector also finds the card itself: if its box fills the frame,
        the frame is the card; if not, crop to that box and detect again. Either
        way the card was found, so the warning becomes information."""
        container = next((f.detector_class for f in profile.fields if f.kind is FieldKind.CONTAINER), None)
        box = located.get(container) if container else None
        if box is None:
            return card, located, to_frame, orientation, reasons
        h, w = card.shape[:2]
        x1, y1, x2, y2 = box.box
        if (x2 - x1) * (y2 - y1) < CARD_FILLS_FRAME * w * h:
            pad_x, pad_y = 0.02 * (x2 - x1), 0.02 * (y2 - y1)
            x1, y1 = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
            x2, y2 = min(w, int(x2 + pad_x)), min(h, int(y2 + pad_y))
            size = profile.rectified_size
            crop = cv2.resize(card[y1:y2, x1:x2], size, interpolation=cv2.INTER_CUBIC)
            to_crop = rectify.scale_matrix((x2 - x1, y2 - y1), size) @ np.array(
                [[1.0, 0.0, -x1], [0.0, 1.0, -y1], [0.0, 0.0, 1.0]]
            )
            card, detections, to_frame, orientation = self._locate(crop, profile, to_crop @ to_frame)
            located = best_per_class(detections)
        reasons = [
            Reason.info(r.code, "No card outline in the photo; the field detector found the card",
                        "لبهٔ کارت در عکس دیده نشد؛ مدل تشخیص فیلد خود کارت را پیدا کرد", r.field)
            if r.code == "card_border_not_found" else r
            for r in reasons
        ]
        return card, located, to_frame, orientation, reasons

    def _detect(self, card: np.ndarray, profile: DocumentProfile) -> list[Detection]:
        cfg = self.settings.ocr
        detector = self._detector_factory(profile)
        return detector.detect(card, cfg.detector_conf, cfg.detector_iou)

    def _read_fields(
        self,
        card: np.ndarray,
        profile: DocumentProfile,
        located: dict[str, Detection],
        empty_rows: dict[str, tuple[int, int, int, int]] | None = None,
    ) -> tuple[dict[str, FieldResult], list[Reason]]:
        cfg = self.settings.ocr
        empty_rows = empty_rows or {}
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
            inferred = detection is None and spec.detector_class in empty_rows
            if inferred:
                # No box on this row: read where it would be. Nothing sure
                # there means the row is printed empty on this card.
                detection = Detection(-1, spec.detector_class, 0.0, empty_rows[spec.detector_class])
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

            pad_y = cfg.text_crop_padding_y if spec.kind is FieldKind.TEXT else None
            crop = crop_with_padding(card, detection.box, cfg.field_crop_padding, pad_y)
            crop = upscale_to_height(crop, cfg.upscale_small_crops_to)
            if profile.trim_label_colon and spec.kind is FieldKind.TEXT:
                crop = trim_label_colon(crop)
            if profile.drop_neighbour_rows and spec.kind is FieldKind.TEXT and not spec.lexicon_per_line:
                crop = drop_neighbour_rows(crop)

            if spec.kind in (FieldKind.DIGITS, FieldKind.DATE, FieldKind.CODE):
                backend_id = profile.digit_backend or cfg.digit_backend
            else:
                backend_id = profile.text_backend or cfg.text_backend
            recognizer = self._recognizer_factory(backend_id)
            recognition = recognizer.recognize(crop, allowlist=spec.charset_hint)
            if profile.reread_blurred and spec.kind is FieldKind.TEXT:
                again = recognizer.recognize(cv2.GaussianBlur(crop, (3, 3), 0), allowlist=spec.charset_hint)
                if again.confidence > recognition.confidence:
                    recognition = again

            if inferred:
                value = self._normalize(spec, recognition.text, profile.keep_letters)
                if not value or recognition.confidence < cfg.min_field_confidence:
                    reasons.append(self._blank_row(spec))
                    results[spec.key] = result
                    continue
                result.detection_confidence = 0.0
                reasons.append(Reason.warn(
                    "field_box_inferred",
                    f"{spec.key} was read from where its row should be; the detector found no box",
                    f"«{spec.label_fa}» از جای ردیفش خوانده شد؛ کادری برایش پیدا نشده بود",
                    spec.key,
                ))

            result.backend = backend_id
            result.raw = recognition.text
            result.confidence = round(recognition.confidence, 4)
            result.value = self._normalize(spec, recognition.text, profile.keep_letters)
            result.low_confidence = recognition.confidence < cfg.min_field_confidence
            if spec.lexicon and cfg.name_lexicon:
                reasons += self._match_lexicon(spec, result, crop, profile)

            reasons += self._field_findings(spec, result)
            results[spec.key] = result

        return results, reasons

    def _match_lexicon(
        self, spec: FieldSpec, result: FieldResult, crop: np.ndarray, profile: DocumentProfile
    ) -> list[Reason]:
        """Snap a closed-vocabulary field to its list, or flag it for review."""
        log_probs = charset = None
        try:
            # A profile with its own text recogniser scores names with it too.
            scorer = self._recognizer_factory(profile.text_backend or self.settings.ocr.lexicon_scorer_backend)
            if hasattr(scorer, "log_probs"):
                log_probs, charset = scorer.log_probs(crop), scorer.charset
        except KycError:
            pass  # no scorer model: exact and nearest matching still apply
        matcher = self._matcher(spec.lexicon)
        match = matcher.resolve(result.value or "", log_probs, charset)
        if match.value is None and spec.lexicon_per_line:
            # A box that slipped onto the next row holds two lines and reads
            # as neither: try each line of the crop on its own.
            match = self._match_lines(spec, crop, profile, matcher, scorer if log_probs is not None else None) or match
        result.lexicon_match = match.method
        if match.value is not None:
            result.value = match.value
            return []
        if not result.value:
            return []
        result.low_confidence = True
        return [
            Reason.warn(
                "name_not_in_list",
                f"{spec.key} read as {result.value!r}, which is not a known name",
                f"«{spec.label_fa}» به‌صورت «{result.value}» خوانده شد که نام شناخته‌شده‌ای نیست",
                spec.key,
            )
        ]

    def _match_lines(self, spec, crop, profile, matcher, scorer):
        lines = text_lines(crop)
        if len(lines) < 2:
            return None
        recognizer = self._recognizer_factory(profile.text_backend or self.settings.ocr.text_backend)
        for line in lines:
            text = self._normalize(spec, recognizer.recognize(line).text, profile.keep_letters) or ""
            log_probs = scorer.log_probs(line) if scorer is not None else None
            found = matcher.resolve(text, log_probs, scorer.charset if scorer is not None else None)
            if found.value is not None:
                return replace(found, method=f"{found.method}_line")
        return None

    @staticmethod
    def _blank_row(spec: FieldSpec) -> Reason:
        if spec.required:
            return Reason.error(
                "field_blank",
                f"The {spec.key} row is empty on this card",
                f"ردیف «{spec.label_fa}» روی این کارت خالی است",
                spec.key,
            )
        # Some cards are issued without it (no surname): worth saying, not a
        # reason to doubt the card.
        return Reason.info(
            "field_blank",
            f"This card has no {spec.label_en.lower()}",
            f"این کارت «{spec.label_fa}» ندارد",
            spec.key,
        )

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
    def _normalize(spec: FieldSpec, raw: str, keep_letters: str = "") -> str | None:
        if spec.kind is FieldKind.DIGITS:
            value = normalize.keep_digits(raw)
        elif spec.kind is FieldKind.DATE:
            value = normalize.normalize_date(raw)
        elif spec.kind is FieldKind.CODE:
            value = normalize.normalize_code(raw)
        else:
            value = normalize.normalize_name(raw, keep_letters)
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


def _map_box(box: list[int], matrix: np.ndarray, size: tuple[int, int]) -> list[int] | None:
    """Bounds of `box`'s corners under `matrix`, clipped to `size`."""
    x1, y1, x2, y2 = box
    mapped = rectify.transform_points(matrix, [(x1, y1), (x2, y1), (x2, y2), (x1, y2)], size)
    return [int(round(v)) for v in mapped] if mapped else None
