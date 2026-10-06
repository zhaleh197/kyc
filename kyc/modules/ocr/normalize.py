"""Text normalisation for Persian/Arabic script OCR output.

OCR engines return whatever glyph they matched: Arabic yeh instead of Persian
yeh, Eastern-Arabic digits, stray tatweel, invisible bidi marks. Comparing or
validating that text without folding it first produces false mismatches that
look like OCR errors but are really encoding differences.

Everything here is pure-python and dependency-free so it can be unit tested
without any model.
"""

from __future__ import annotations

import re
import unicodedata

# --- digits ------------------------------------------------------------
# Extended Arabic-Indic (Persian) U+06F0..U+06F9 and Arabic-Indic U+0660..U+0669
PERSIAN_DIGITS = "۰۱۲۳۴۵۶۷۸۹"
ARABIC_DIGITS = "٠١٢٣٤٥٦٧٨٩"
ASCII_DIGITS = "0123456789"

_DIGIT_MAP = str.maketrans(
    PERSIAN_DIGITS + ARABIC_DIGITS,
    ASCII_DIGITS + ASCII_DIGITS,
)

# --- letters -----------------------------------------------------------
# Fold Arabic-only forms onto their Persian equivalents. Alef-madda (آ) is
# deliberately NOT folded to bare alef: it distinguishes real names
# (آرش vs ارش), so folding it would corrupt the value.
_LETTER_MAP = {
    "ي": "ی",  # ARABIC YEH -> FARSI YEH
    "ى": "ی",  # ALEF MAKSURA -> FARSI YEH
    "ے": "ی",  # YEH BARREE -> FARSI YEH
    "ك": "ک",  # ARABIC KAF -> KEHEH
    "ڪ": "ک",  # SWASH KAF -> KEHEH
    "ة": "ه",  # TEH MARBUTA -> HEH
    "ۀ": "ه",  # HEH WITH YEH ABOVE -> HEH
    "أ": "ا",  # ALEF WITH HAMZA ABOVE -> ALEF
    "إ": "ا",  # ALEF WITH HAMZA BELOW -> ALEF
    "ٱ": "ا",  # ALEF WASLA -> ALEF
    "ھ": "ه",
    "ؤ": "و",  # WAW WITH HAMZA -> WAW
}
_LETTER_TABLE = {ord(k): v for k, v in _LETTER_MAP.items()}

# Harakat / tanwin / superscript alef / tatweel — decorative, never semantic
# in printed ID cards, and they wreck string comparison.
_MARKS = re.compile("[ً-ٰٟـۖ-ۭ]")

# Bidi and zero-width controls that PDF/OCR pipelines love to inject.
# ZWNJ (U+200C) is preserved: in Persian it is a real word-forming character.
_INVISIBLES = re.compile("[​‍‎‏‪-‮⁦-⁩﻿]")

_WS = re.compile(r"\s+")


def to_ascii_digits(text: str) -> str:
    """۱۲۳ / ١٢٣ -> 123"""
    return text.translate(_DIGIT_MAP)


def fold_letters(text: str) -> str:
    """Map Arabic letter variants onto their Persian counterparts."""
    return text.translate(_LETTER_TABLE)


def strip_marks(text: str) -> str:
    return _MARKS.sub("", text)


def strip_invisibles(text: str) -> str:
    return _INVISIBLES.sub("", text)


def normalize_text(text: str) -> str:
    """Full pipeline for a free-text field (names, place of issue).

    Order matters: NFC first so decomposed sequences become single code points,
    then fold, then strip decorations, then tidy whitespace around ZWNJ.
    """
    if not text:
        return ""
    out = unicodedata.normalize("NFC", text)
    out = strip_invisibles(out)
    out = fold_letters(out)
    out = strip_marks(out)
    out = to_ascii_digits(out)
    # A space next to a ZWNJ is always an OCR artefact - one or the other.
    out = re.sub(r"\s*‌\s*", "‌", out)
    out = _WS.sub(" ", out).strip()
    return out


def keep_digits(text: str) -> str:
    """Everything that is not a digit is dropped. For national id / serial."""
    return re.sub(r"\D", "", to_ascii_digits(strip_invisibles(text or "")))


# Letters that may legitimately appear in a Persian personal name.
_NAME_ALLOWED = re.compile("[^ء-غف-يٮ-ۓ‌ ]")


def normalize_name(text: str) -> str:
    """Names must not contain digits or latin characters; a recogniser that
    emits them has misread the crop, so we drop them rather than pass junk on."""
    out = normalize_text(text)
    out = _NAME_ALLOWED.sub(" ", out)
    return _WS.sub(" ", out).strip()


_DATE_SPLIT = re.compile(r"\D+")


def normalize_date(text: str) -> str | None:
    """Pull a YYYY/MM/DD out of whatever the recogniser produced.

    Iranian cards print dates as ۱۳۷۰/۰۵/۱۲, but OCR routinely returns
    '1370 05 12', '1370-5-12' or '1370.05.12'. Two-digit years are rejected
    rather than guessed: silently expanding '70' to '1370' would be a
    fabricated value in an identity document.
    """
    digits_only = to_ascii_digits(strip_invisibles(text or "")).strip()
    parts = [p for p in _DATE_SPLIT.split(digits_only) if p]
    if len(parts) == 3:
        y, m, d = parts
    elif len(parts) == 1 and len(parts[0]) == 8:
        y, m, d = parts[0][:4], parts[0][4:6], parts[0][6:]
    else:
        return None
    if len(y) != 4 or not (1 <= len(m) <= 2) or not (1 <= len(d) <= 2):
        return None
    return f"{int(y):04d}/{int(m):02d}/{int(d):02d}"
