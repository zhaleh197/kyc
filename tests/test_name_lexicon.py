"""Closed-vocabulary name matching (ocr/lexicon)."""

from __future__ import annotations

import itertools

import numpy as np

from kyc.core.schemas import Decision
from kyc.modules.ocr import lexicon
from kyc.modules.ocr.lexicon import NameMatcher, ctc_log_likelihood, match_key

from .conftest import make_card, photograph
from .test_pipeline import build


def brute_force(log_probs: np.ndarray, label: list[int], blank: int = 0) -> float:
    """Sum over every frame path that collapses to `label`."""
    steps, classes = log_probs.shape
    total = -np.inf
    for path in itertools.product(range(classes), repeat=steps):
        collapsed = [c for i, c in enumerate(path) if c != blank and (i == 0 or c != path[i - 1])]
        if collapsed == label:
            total = np.logaddexp(total, sum(log_probs[t, c] for t, c in enumerate(path)))
    return float(total)


def test_ctc_likelihood_matches_brute_force():
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(5, 4))
    log_probs = logits - np.log(np.exp(logits).sum(axis=1, keepdims=True))
    labels = [[1], [1, 2], [2, 2], [3, 1, 2]]
    got = ctc_log_likelihood(log_probs, labels)
    for label, value in zip(labels, got, strict=True):
        assert np.isclose(value, brute_force(log_probs, label)), label


def test_compound_names_match_with_or_without_space():
    assert match_key("امیر حسین") == match_key("امیرحسین") == match_key("امیر‌حسین")


def test_the_shipped_list_loads_and_holds_common_names():
    names = lexicon.load("given_names_fa")
    assert len(names) > 400
    keys = {match_key(n) for n in names}
    for name in ("ژاله", "زهرا", "علی", "محمدرضا", "امیر حسین"):
        assert match_key(name) in keys, name


def matcher() -> NameMatcher:
    return NameMatcher(("ژاله", "زهره", "رضا", "علی"), min_score=-6.0, min_margin=5.0)


def test_a_listed_name_is_kept():
    assert matcher().resolve("زهره").method == "exact"


def test_a_near_miss_snaps_to_the_listed_name():
    match = matcher().resolve("ژالن")  # last letter misread
    assert (match.value, match.method) == ("ژاله", "nearest")


def test_a_far_read_is_not_guessed():
    match = matcher().resolve("نب")
    assert match.value is None and match.method == "none"


def test_ctc_picks_the_name_the_frames_support():
    charset = ["<blank>", "ژ", "ا", "ل", "ه", "ز", "ر", "ض", "ع", "ی"]
    index = {c: i for i, c in enumerate(charset)}
    # Frames that clearly spell ژاله, separated by blanks.
    frames = []
    for symbol in "ژاله":
        frames += [index[symbol], 0]
    log_probs = np.full((len(frames), len(charset)), np.log(0.01))
    for t, c in enumerate(frames):
        log_probs[t, c] = np.log(0.91)
    match = matcher().resolve("نب", log_probs, charset)
    assert (match.value, match.method) == ("ژاله", "ctc")


def test_unknown_first_name_goes_to_review(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, {**good_values, "first_name": "نب"})
    result = pipeline.run(photograph(make_card()))

    assert result.decision is Decision.REVIEW
    field = result.data["fields"]["first_name"]
    assert field["lexicon_match"] == "none" and field["low_confidence"]
    assert "name_not_in_list" in [r.code for r in result.reasons]


def test_last_name_is_not_matched_against_given_names(relaxed_settings, good_values):
    pipeline, _ = build(relaxed_settings, good_values)
    result = pipeline.run(photograph(make_card()))
    assert result.data["fields"]["last_name"]["lexicon_match"] is None
    assert result.data["fields"]["first_name"]["lexicon_match"] == "exact"
