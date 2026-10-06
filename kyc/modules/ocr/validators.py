"""Iranian identity-document validation rules.

These are the checks that turn "some text an OCR model produced" into
"a value we are willing to act on". A national id that fails its checksum is
either a misread or a forgery, and in both cases the answer is the same:
do not auto-approve.
"""

from __future__ import annotations

from datetime import date

# Jalali calendar bounds we accept on an ID card. Anything outside is a misread.
MIN_JALALI_YEAR = 1250
MAX_JALALI_YEAR = 1500


# ---------------------------------------------------------------- national id


def is_valid_national_id(code: str) -> bool:
    """Iranian national code (کد ملی) check-digit validation.

    Ten digits; the last one is a checksum over the first nine weighted 10..2:

        r = (sum(d[i] * (10 - i) for i in 0..8)) % 11
        check == r          if r < 2
        check == 11 - r     otherwise

    Leading zeros are significant (Tehran-issued codes start with 0), so the
    input must already be a zero-padded 10-character string.
    """
    if not code or len(code) != 10 or not code.isdigit():
        return False
    # Repdigit codes (0000000000, 1111111111, ...) satisfy the checksum by
    # accident but are not issued.
    if code == code[0] * 10:
        return False
    total = sum(int(code[i]) * (10 - i) for i in range(9))
    remainder = total % 11
    check = int(code[9])
    return check == remainder if remainder < 2 else check == 11 - remainder


# ------------------------------------------------------------------ calendar


def jalali_to_gregorian(jy: int, jm: int, jd: int) -> tuple[int, int, int]:
    """Convert a Jalali (Solar Hijri) date to Gregorian (y, m, d)."""
    jy += 1595
    days = -355668 + (365 * jy) + ((jy // 33) * 8) + (((jy % 33) + 3) // 4) + jd
    if jm < 7:
        days += (jm - 1) * 31
    else:
        days += ((jm - 7) * 30) + 186

    gy = 400 * (days // 146097)
    days %= 146097
    if days > 36524:
        days -= 1
        gy += 100 * (days // 36524)
        days %= 36524
        if days >= 365:
            days += 1
    gy += 4 * (days // 1461)
    days %= 1461
    if days > 365:
        gy += (days - 1) // 365
        days = (days - 1) % 365
    gd = days + 1

    feb = 29 if (gy % 4 == 0 and gy % 100 != 0) or (gy % 400 == 0) else 28
    month_lengths = [0, 31, feb, 31, 30, 31, 30, 31, 31, 30, 31, 30, 31]
    gm = 0
    while gm < 13 and gd > month_lengths[gm]:
        gd -= month_lengths[gm]
        gm += 1
    return gy, gm, gd


def jalali_is_leap(jy: int) -> bool:
    """Derived from the conversion itself rather than a separate cycle rule,
    so leap-year handling can never disagree with date arithmetic."""
    start = date(*jalali_to_gregorian(jy, 1, 1))
    next_start = date(*jalali_to_gregorian(jy + 1, 1, 1))
    return (next_start - start).days == 366


def jalali_days_in_month(jy: int, jm: int) -> int:
    if not 1 <= jm <= 12:
        return 0
    if jm <= 6:
        return 31
    if jm <= 11:
        return 30
    return 30 if jalali_is_leap(jy) else 29


def parse_jalali(value: str) -> tuple[int, int, int] | None:
    """'1370/05/12' -> (1370, 5, 12); None if the date cannot exist."""
    if not value:
        return None
    parts = value.split("/")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        return None
    jy, jm, jd = (int(p) for p in parts)
    if not MIN_JALALI_YEAR <= jy <= MAX_JALALI_YEAR:
        return None
    if not 1 <= jm <= 12:
        return None
    if not 1 <= jd <= jalali_days_in_month(jy, jm):
        return None
    return jy, jm, jd


def jalali_to_iso(value: str) -> str | None:
    """'1370/05/12' -> '1991-08-03'. Lets downstream systems work in one calendar."""
    parsed = parse_jalali(value)
    if parsed is None:
        return None
    return date(*jalali_to_gregorian(*parsed)).isoformat()


def age_on(birth_jalali: str, reference: date | None = None) -> int | None:
    """Completed years of age at `reference` (default: today)."""
    parsed = parse_jalali(birth_jalali)
    if parsed is None:
        return None
    born = date(*jalali_to_gregorian(*parsed))
    today = reference or date.today()
    return today.year - born.year - ((today.month, today.day) < (born.month, born.day))


def is_expired(expiry_jalali: str, reference: date | None = None) -> bool | None:
    """True if the card's validity date is in the past. None if unparseable."""
    parsed = parse_jalali(expiry_jalali)
    if parsed is None:
        return None
    expiry = date(*jalali_to_gregorian(*parsed))
    return expiry < (reference or date.today())
