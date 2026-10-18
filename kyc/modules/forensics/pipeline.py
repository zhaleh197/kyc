"""Document forensics (06a): signs that a card photo was edited.

    card photo (as submitted) + field boxes on it (from OCR's source_box)
        -> per region: noise / ELA / JPEG-grid features (signals.py)
        -> deviation from genuine cards, per feature (calibration.py)
        -> score >= 1 on an enabled region -> warn -> review
        -> EXIF editor tag -> info only

Advisory by construction: nothing here emits an error (spec 06 F5), so the
worst it can do to a genuine customer is send them to a human. The limits
are measured on synthetic edits of the labelled card set, not on real
forgeries - see models/manifest.yaml.
"""

from __future__ import annotations

import io

from kyc.core.config import Settings, get_settings
from kyc.core.imaging import decode_image
from kyc.core.schemas import Decision, ModuleResult, Reason, Timer

from . import calibration as calib_mod
from .signals import PhotoMaps

MODULE_NAME = "forensics"
MODULE_VERSION = "0.1.0"

_FIELD_FA = {"national_id": "کد ملی", "birth_date": "تاریخ تولد", "expiry_date": "تاریخ اعتبار"}

# EXIF Software values that mean a photo editor wrote the file. Phones'
# own camera apps write their firmware/model here, which is fine.
EDITORS = ("photoshop", "gimp", "snapseed", "picsart", "canva", "lightroom", "pixlr", "paint.net",
           "affinity", "polarr", "facetune", "remini", "photoroom", "illustrator", "corel")


def editing_software(data: bytes) -> str | None:
    try:
        from PIL import Image

        with Image.open(io.BytesIO(data)) as im:
            software = str(im.getexif().get(0x0131, "") or "")
    except Exception:  # noqa: BLE001 - unreadable metadata is no evidence either way
        return None
    return software if any(e in software.lower() for e in EDITORS) else None


class DocumentForensicsPipeline:
    def __init__(self, settings: Settings | None = None, *, calibration: dict | None = None):
        self.settings = settings or get_settings()
        self._calibration = calibration

    @property
    def calibration(self) -> dict | None:
        if self._calibration is None:
            path = self.settings.model_path(self.settings.forensics.calibration_file)
            self._calibration = calib_mod.load(path) or {}
        return self._calibration or None

    def run(self, data: bytes, boxes: dict[str, list[int] | None]) -> ModuleResult:
        """`boxes`: region -> [x1, y1, x2, y2] on the photo exactly as submitted."""
        cfg = self.settings.forensics
        reasons: list[Reason] = []
        signals: list[dict] = []
        with Timer() as timer:
            if cfg.metadata_check and (software := editing_software(data)):
                reasons.append(Reason.info(
                    "editing_software_metadata", f"The photo's metadata names an editor: {software}",
                    f"در اطلاعات فایل نام نرم‌افزار ویرایش عکس آمده است: {software}",
                ))
            calib = self.calibration
            if calib is None:
                reasons.append(Reason.info("forensics_not_calibrated", "No calibration file; pixel checks skipped",
                                           "فایل کالیبراسیون نیست؛ بررسی تصویر انجام نشد"))
            else:
                image = decode_image(data)
                maps = PhotoMaps(image, calib.get("ela_quality", 90))
                for region, ref in calib["regions"].items():
                    box = boxes.get(region)
                    if region not in cfg.regions or not box:
                        continue
                    cells = maps.features(box, ref["cells"])
                    if cells is None:
                        continue
                    score, feature = calib_mod.region_score(calib, region, cells)
                    score /= cfg.limit_scale
                    signals.append({
                        "name": region, "score": round(score, 3), "feature": feature, "region": list(box),
                        "deviations": {k: round(v, 2) for k, v in calib_mod.deviations(calib, region, cells).items()},
                    })
                    if score >= 1.0:
                        reasons.append(self._finding(region, feature, score))
            if calib is not None and not signals:
                reasons.append(Reason.info("forensics_no_regions", "No card region could be checked",
                                           "هیچ ناحیه‌ای از کارت قابل بررسی نبود"))

        worst = max((s["score"] for s in signals), default=0.0)
        decision = Decision.REVIEW if any(r.severity.value == "warn" for r in reasons) else Decision.PASS
        return ModuleResult(
            module=MODULE_NAME, version=MODULE_VERSION, decision=decision,
            # 1.0 = nothing unusual; 0.5 = exactly at a warning line.
            score=round(max(0.0, min(1.0, 1.0 - worst / 2)), 4),
            reasons=reasons,
            data={"signals": signals, "calibration_version": (calib or {}).get("version")},
            elapsed_ms=round(timer.ms, 2),
        )

    @staticmethod
    def _finding(region: str, feature: str, score: float) -> Reason:
        if region == "portrait":
            return Reason.warn(
                "portrait_region_inconsistent",
                f"The portrait area differs from the card around it ({feature}, {score:.2f}x the limit)",
                "ناحیهٔ عکس با بقیهٔ کارت هم‌خوانی ندارد (احتمال جایگزینی عکس)",
                "portrait",
            )
        return Reason.warn(
            "field_region_inconsistent",
            f"A character of {region} differs from its neighbours ({feature}, {score:.2f}x the limit)",
            f"نویسه‌های «{_FIELD_FA.get(region, region)}» با هم هم‌خوانی ندارند (احتمال ویرایش)",
            region,
        )
