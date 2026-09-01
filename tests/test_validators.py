from datetime import date

import pytest

from kyc.modules.ocr import validators as v


@pytest.mark.parametrize("code", ["0012345679", "0499370899"])
def test_valid_national_ids(code):
    assert v.is_valid_national_id(code)


@pytest.mark.parametrize(
    "code",
    [
        "0012345678",   # wrong check digit
        "1111111111",   # repdigit, passes the checksum by accident but is not issued
        "123",          # too short
        "abcdefghij",   # not numeric
        "",
    ],
)
def test_invalid_national_ids(code):
    assert not v.is_valid_national_id(code)


@pytest.mark.parametrize(
    "jalali,gregorian",
    [
        ((1300, 1, 1), (1921, 3, 21)),
        ((1358, 11, 22), (1980, 2, 11)),
        ((1370, 1, 1), (1991, 3, 21)),
        ((1398, 12, 29), (2020, 3, 19)),
        ((1399, 12, 30), (2021, 3, 20)),   # 1399 is a leap year
        ((1403, 1, 1), (2024, 3, 20)),
    ],
)
def test_jalali_to_gregorian(jalali, gregorian):
    assert v.jalali_to_gregorian(*jalali) == gregorian


@pytest.mark.parametrize("year,leap", [(1399, True), (1400, False), (1403, True), (1404, False)])
def test_jalali_leap_years(year, leap):
    assert v.jalali_is_leap(year) is leap


def test_esfand_length_follows_leap_year():
    assert v.jalali_days_in_month(1399, 12) == 30
    assert v.jalali_days_in_month(1400, 12) == 29
    assert v.jalali_days_in_month(1400, 1) == 31
    assert v.jalali_days_in_month(1400, 7) == 30


@pytest.mark.parametrize("value", ["1400/12/30", "1370/13/01", "1370/00/10", "1370/05/32", "1100/01/01", "junk"])
def test_parse_jalali_rejects_impossible_dates(value):
    assert v.parse_jalali(value) is None


def test_jalali_to_iso():
    assert v.jalali_to_iso("1370/05/12") == "1991-08-03"


def test_age_on_reference_date():
    assert v.age_on("1370/05/12", date(2024, 8, 2)) == 32   # day before the birthday
    assert v.age_on("1370/05/12", date(2024, 8, 3)) == 33


def test_is_expired():
    assert v.is_expired("1399/05/12", date(2026, 1, 1)) is True
    assert v.is_expired("1410/05/12", date(2026, 1, 1)) is False
    assert v.is_expired("nonsense") is None
