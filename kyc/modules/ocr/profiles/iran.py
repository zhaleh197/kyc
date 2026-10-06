"""Iranian national ID card (کارت ملی هوشمند) — front side.

Detector class names match the dataset the field-segmentation model was
trained on, so the trained artifact drops in without renaming anything.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import date

from kyc.core.schemas import Reason

from ..validators import (
    age_on,
    is_expired,
    is_valid_national_id,
    jalali_to_iso,
    parse_jalali,
)
from .base import DocumentProfile, FieldKind, FieldSpec

# Order below is the training class order of the field detector.
IR_CLASS_NAMES = (
    "birthday",
    "exp",
    "fathername",
    "idcard",
    "idnumber",
    "img",
    "lastname",
    "name",
)

MAX_PLAUSIBLE_AGE = 120
MIN_ADULT_AGE = 18


def _validate_iran_front(values: dict) -> list[Reason]:
    """`values` maps canonical key -> normalised string (or None when missing).

    Returns findings, never raises. A missing required field and a field that
    failed its checksum are both reported so an operator sees the full picture
    in one pass instead of one error at a time.
    """
    reasons: list[Reason] = []
    today = date.today()

    national_id = values.get("national_id")
    if not national_id:
        reasons.append(
            Reason.error("national_id_missing", "National id was not read", "کد ملی خوانده نشد", "national_id")
        )
    elif len(national_id) != 10:
        reasons.append(
            Reason.error(
                "national_id_length",
                f"National id has {len(national_id)} digits, expected 10",
                f"کد ملی {len(national_id)} رقم است، باید ۱۰ رقم باشد",
                "national_id",
            )
        )
    elif not is_valid_national_id(national_id):
        # Either a misread digit or a fabricated number; both block auto-approval.
        reasons.append(
            Reason.error(
                "national_id_checksum",
                "National id failed its check-digit validation",
                "رقم کنترل کد ملی نامعتبر است",
                "national_id",
            )
        )

    birth = values.get("birth_date")
    if not birth:
        reasons.append(
            Reason.error("birth_date_missing", "Birth date was not read", "تاریخ تولد خوانده نشد", "birth_date")
        )
    elif parse_jalali(birth) is None:
        reasons.append(
            Reason.error(
                "birth_date_invalid",
                f"{birth} is not a valid Jalali date",
                f"تاریخ تولد «{birth}» معتبر نیست",
                "birth_date",
            )
        )
    else:
        age = age_on(birth, today)
        if age is None or age < 0 or age > MAX_PLAUSIBLE_AGE:
            reasons.append(
                Reason.error(
                    "birth_date_implausible",
                    f"Birth date yields an implausible age ({age})",
                    f"تاریخ تولد سن غیرمنطقی می‌دهد ({age})",
                    "birth_date",
                )
            )
        elif age < MIN_ADULT_AGE:
            # Not an OCR problem - a policy one. Surfaced as a warning so the
            # orchestrator can apply its own age rule.
            reasons.append(
                Reason.warn(
                    "under_age",
                    f"Document holder is {age} years old",
                    f"دارندهٔ کارت {age} سال دارد",
                    "birth_date",
                )
            )

    expiry = values.get("expiry_date")
    if not expiry:
        reasons.append(
            Reason.warn("expiry_missing", "Expiry date was not read", "تاریخ اعتبار خوانده نشد", "expiry_date")
        )
    elif parse_jalali(expiry) is None:
        reasons.append(
            Reason.warn(
                "expiry_invalid",
                f"{expiry} is not a valid Jalali date",
                f"تاریخ اعتبار «{expiry}» معتبر نیست",
                "expiry_date",
            )
        )
    elif is_expired(expiry, today):
        reasons.append(
            Reason.error("card_expired", "The card has expired", "اعتبار کارت به پایان رسیده است", "expiry_date")
        )

    birth_parsed = parse_jalali(birth) if birth else None
    expiry_parsed = parse_jalali(expiry) if expiry else None
    if birth_parsed and expiry_parsed and birth_parsed >= expiry_parsed:
        reasons.append(
            Reason.error(
                "date_order",
                "Birth date is not before the expiry date",
                "تاریخ تولد قبل از تاریخ اعتبار نیست",
            )
        )

    for key, label_en, label_fa in (
        ("first_name", "First name", "نام"),
        ("last_name", "Last name", "نام خانوادگی"),
        ("father_name", "Father name", "نام پدر"),
    ):
        value = values.get(key)
        if not value:
            reasons.append(Reason.error(f"{key}_missing", f"{label_en} was not read", f"{label_fa} خوانده نشد", key))
        elif len(value) < 2:
            reasons.append(
                Reason.warn(
                    f"{key}_suspicious",
                    f"{label_en} is implausibly short: {value}",
                    f"{label_fa} خیلی کوتاه است: «{value}»",
                    key,
                )
            )

    return reasons


IR_NATIONAL_CARD_FRONT = DocumentProfile(
    id="ir_national_card_front",
    country="IR",
    name_en="Iranian national ID card (front)",
    name_fa="کارت ملی هوشمند ایران (رو)",
    detector_model="ir_national_card_front_fields.onnx",
    class_names=IR_CLASS_NAMES,
    validator=_validate_iran_front,
    fields=(
        FieldSpec("card", "idcard", FieldKind.CONTAINER, "کارت ملی", "ID card"),
        FieldSpec("portrait", "img", FieldKind.IMAGE, "عکس", "Portrait photo"),
        FieldSpec(
            "national_id", "idnumber", FieldKind.DIGITS, "کد ملی", "National id",
            exact_len=10, charset_hint="0123456789",
        ),
        FieldSpec("first_name", "name", FieldKind.TEXT, "نام", "First name"),
        FieldSpec("last_name", "lastname", FieldKind.TEXT, "نام خانوادگی", "Last name"),
        FieldSpec("father_name", "fathername", FieldKind.TEXT, "نام پدر", "Father name"),
        FieldSpec(
            "birth_date", "birthday", FieldKind.DATE, "تاریخ تولد", "Birth date",
            charset_hint="0123456789/",
        ),
        FieldSpec(
            "expiry_date", "exp", FieldKind.DATE, "تاریخ اعتبار", "Expiry date",
            required=False, charset_hint="0123456789/",
        ),
    ),
)


def enrich(values: dict) -> dict:
    """Derived values a consumer would otherwise have to compute itself."""
    extra: dict = {}
    birth = values.get("birth_date")
    if birth:
        iso = jalali_to_iso(birth)
        if iso:
            extra["birth_date_gregorian"] = iso
            extra["age"] = age_on(birth)
    expiry = values.get("expiry_date")
    if expiry:
        iso = jalali_to_iso(expiry)
        if iso:
            extra["expiry_date_gregorian"] = iso
            extra["expired"] = is_expired(expiry)
    nid = values.get("national_id")
    if nid and len(nid) == 10:
        extra["national_id_valid"] = is_valid_national_id(nid)
    return extra


# Bound after definition because the profile literal is declared above it.
IR_NATIONAL_CARD_FRONT = replace(IR_NATIONAL_CARD_FRONT, enricher=enrich)
