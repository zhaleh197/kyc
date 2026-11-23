"""Iraqi national card (البطاقة الوطنية / کارتی نیشتمانی) - front side,
as issued in the Kurdistan region: Sorani and Arabic labels, names written in
Kurdish or Arabic script, Latin digits.

The front carries no dates (they are on the back), so there is no age or
expiry check here.
"""

from __future__ import annotations

from kyc.core.schemas import Reason

from .base import DocumentProfile, FieldKind, FieldSpec

# Training class order of the field detector (training/krg_card).
IQ_CLASS_NAMES = (
    "card",
    "portrait",
    "national_id",
    "name",
    "father",
    "grandfather",
    "surname",
    "mother",
    "mother_grandfather",
    "sex",
    "blood_group",
    "document_number",
)

NATIONAL_ID_LEN = 12


def _validate_iraq_front(values: dict) -> list[Reason]:
    reasons: list[Reason] = []
    nid = values.get("national_id")
    if not nid:
        reasons.append(
            Reason.error("national_id_missing", "National id was not read", "شماره ملی خوانده نشد", "national_id")
        )
    elif len(nid) != NATIONAL_ID_LEN:
        reasons.append(
            Reason.error(
                "national_id_length",
                f"National id has {len(nid)} digits, expected {NATIONAL_ID_LEN}",
                f"شماره ملی {len(nid)} رقم است، باید {NATIONAL_ID_LEN} رقم باشد",
                "national_id",
            )
        )
    for key, en, fa in (("first_name", "First name", "نام"), ("father_name", "Father name", "نام پدر")):
        value = values.get(key)
        if value and len(value.replace(" ", "")) < 2:
            reasons.append(
                Reason.warn(f"{key}_suspicious", f"{en} is implausibly short: {value}",
                            f"{fa} خیلی کوتاه است: «{value}»", key)
            )
    return reasons


IQ_NATIONAL_CARD_FRONT = DocumentProfile(
    id="iq_national_card_front",
    country="IQ",
    name_en="Iraqi national card, Kurdistan (front)",
    name_fa="کارت ملی عراق - کردستان (رو)",
    detector_model="iq_national_card_front_fields.onnx",
    class_names=IQ_CLASS_NAMES,
    layout=(("portrait", "left_of", "name"), ("national_id", "above", "name"), ("name", "above", "sex")),
    # Names are Kurdish (Sorani) or Arabic: the recogniser fine-tuned for both.
    text_backend="ppocr_fa_ckb",
    # Latin digits and letters (id, card number, blood group): the base
    # Arabic model reads them; the Persian-digit CRNN does not know them, and
    # the fa/ckb fine-tune forgot them (its training lines had none).
    digit_backend="ppocr_arabic",
    # The detector was trained on flat card images (training/krg_card).
    rectify=True,
    # Arabic names keep ة أ إ ؤ. (ھ is still folded to ه: the recogniser
    # writes Sorani h as ھ, names are usually typed with ه.)
    keep_letters="ةأإؤ",
    # "ناو:" - the name box often takes the label's colon in.
    trim_label_colon=True,
    # The name rows sit close: a box takes in the tail of the row above.
    drop_neighbour_rows=True,
    validator=_validate_iraq_front,
    fields=(
        FieldSpec("card", "card", FieldKind.CONTAINER, "کارت", "ID card"),
        FieldSpec("portrait", "portrait", FieldKind.IMAGE, "عکس", "Portrait photo"),
        FieldSpec("national_id", "national_id", FieldKind.DIGITS, "شماره ملی", "National id",
                  exact_len=NATIONAL_ID_LEN, charset_hint="0123456789"),
        FieldSpec("first_name", "name", FieldKind.TEXT, "نام", "First name"),
        FieldSpec("father_name", "father", FieldKind.TEXT, "نام پدر", "Father name"),
        FieldSpec("grandfather_name", "grandfather", FieldKind.TEXT, "نام پدربزرگ", "Grandfather name"),
        FieldSpec("last_name", "surname", FieldKind.TEXT, "لقب", "Surname", required=False),
        FieldSpec("mother_name", "mother", FieldKind.TEXT, "نام مادر", "Mother name"),
        FieldSpec("mother_father_name", "mother_grandfather", FieldKind.TEXT, "نام پدر مادر",
                  "Mother's father name", required=False),
        FieldSpec("sex", "sex", FieldKind.TEXT, "جنسیت", "Sex", required=False,
                  lexicon="sex_iq", lexicon_per_line=True),
        FieldSpec("blood_group", "blood_group", FieldKind.CODE, "گروه خون", "Blood group",
                  required=False, charset_hint="ABO+-"),
        FieldSpec("document_number", "document_number", FieldKind.CODE, "شماره کارت", "Card number",
                  required=False, charset_hint="ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"),
    ),
)
