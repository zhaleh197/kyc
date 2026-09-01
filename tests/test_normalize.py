import pytest

from kyc.modules.ocr import normalize


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("۱۳۷۰/۰۵/۱۲", "1370/05/12"),   # extended arabic-indic
        ("١٢٣٤٥", "12345"),             # arabic-indic
        ("mixed ۱2٣", "mixed 123"),
    ],
)
def test_to_ascii_digits(raw, expected):
    assert normalize.to_ascii_digits(raw) == expected


def test_fold_letters_maps_arabic_variants_to_persian():
    assert normalize.fold_letters("كريمي") == "کریمی"


def test_alef_madda_is_preserved():
    # Folding it would turn آرش into ارش, a different name.
    assert "آ" in normalize.normalize_name("آرش")


def test_normalize_name_drops_diacritics_tatweel_and_digits():
    assert normalize.normalize_name("محمــدي  كريمي 123") == "محمدی کریمی"


def test_normalize_name_collapses_space_around_zwnj():
    assert normalize.normalize_text("نام‌ خانوادگی") == "نام‌خانوادگی"


def test_keep_digits_strips_separators_and_keeps_leading_zeros():
    assert normalize.keep_digits("۰۰۱-۲۳۴۵۶۷-۹") == "0012345679"


@pytest.mark.parametrize(
    "raw,expected",
    [
        ("۱۳۷۰/۰۵/۱۲", "1370/05/12"),
        ("1370-5-2", "1370/05/02"),
        ("1370.05.12", "1370/05/12"),
        ("13700512", "1370/05/12"),
    ],
)
def test_normalize_date_accepts_common_separators(raw, expected):
    assert normalize.normalize_date(raw) == expected


@pytest.mark.parametrize("raw", ["70/5/12", "", "1370/05", "garbage"])
def test_normalize_date_refuses_to_guess(raw):
    # Expanding a two-digit year would fabricate data in an identity document.
    assert normalize.normalize_date(raw) is None
