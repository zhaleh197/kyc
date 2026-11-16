"""Name lists for fields with a closed vocabulary (first name, father's name).

A free recogniser reads «ژاله» as «نب» and reports it with confidence. A
given name, though, is one of a few thousand: the crop is scored against
every name on the list and the read is accepted only when it lands on one.

    1. the recogniser's text is already a listed name     -> accept it
    2. CTC: P(name | crop) under the CRNN, every listed name scored at once;
       accept the best one if it is confident and well clear of the runner-up
    3. the text is a small edit away from one listed name -> accept that name
    4. otherwise keep the raw read and send it to review

End to end on the held-out cards (eval_text.tsv, 39 names located): names
read right stayed at 33, names accepted with a wrong value fell from 6 to 4,
and no listed first or father's name was accepted wrong - the two left are
garbled names in that dataset, the other two are last names (no list).
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np

from ..normalize import normalize_name

HERE = Path(__file__).parent


def match_key(text: str) -> str:
    """Compound names are printed with a space, a ZWNJ or neither
    («امیر حسین», «امیرحسین»): compare the letters only."""
    return normalize_name(text or "").replace(" ", "").replace("‌", "")


@lru_cache(maxsize=8)
def load(name: str) -> tuple[str, ...]:
    """Names from `<name>.txt` next to this file; `#` starts a comment line."""
    path = HERE / f"{name}.txt"
    lines = path.read_text(encoding="utf-8").splitlines()
    names = (normalize_name(line) for line in lines if line.strip() and not line.lstrip().startswith("#"))
    return tuple(dict.fromkeys(n for n in names if n))


@dataclass(frozen=True)
class LexiconMatch:
    value: str | None   # the listed name, or None when nothing is certain
    method: str         # exact | ctc | nearest | none
    score: float | None = None    # CTC log-probability of the chosen name
    margin: float | None = None   # its lead over the runner-up, in log-probability


def ctc_log_likelihood(log_probs: np.ndarray, labels: list[list[int]], blank: int = 0) -> np.ndarray:
    """log P(label | frames) for many labels at once (CTC forward algorithm).

    `log_probs` is a (T, C) log-softmax matrix; each label is a list of class
    indices without blanks. Returns one log-likelihood per label.
    """
    neg = -1e30
    steps = log_probs.shape[0]
    count = len(labels)
    states = 2 * max(len(label) for label in labels) + 1
    lengths = np.array([2 * len(label) + 1 for label in labels])
    ext = np.full((count, states), blank, np.int64)  # blank, l1, blank, l2, ...
    for i, label in enumerate(labels):
        ext[i, 1 : 2 * len(label) : 2] = label
    # A state may be reached from two back unless it is a blank or repeats the
    # previous label (a repeat needs a blank between).
    skip = np.zeros((count, states), bool)
    skip[:, 2:] = (ext[:, 2:] != blank) & (ext[:, 2:] != ext[:, :-2])
    inside = np.arange(states)[None, :] < lengths[:, None]

    alpha = np.full((count, states), neg)
    alpha[:, 0] = log_probs[0, blank]
    alpha[:, 1] = log_probs[0, ext[:, 1]]
    alpha[~inside] = neg
    for t in range(1, steps):
        one = np.concatenate([np.full((count, 1), neg), alpha[:, :-1]], axis=1)
        two = np.where(skip, np.concatenate([np.full((count, 2), neg), alpha[:, :-2]], axis=1), neg)
        alpha = np.logaddexp(np.logaddexp(alpha, one), two) + log_probs[t, ext]
        alpha[~inside] = neg
    rows = np.arange(count)
    return np.logaddexp(alpha[rows, lengths - 1], alpha[rows, lengths - 2])


def edit_distance(a: str, b: str) -> int:
    row = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        diag, row[0] = row[0], i
        for j, cb in enumerate(b, 1):
            diag, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, diag + (ca != cb))
    return row[-1]


class NameMatcher:
    """Resolves one recognised name against a list. `min_score` and
    `min_margin` are CTC log-probabilities (see config.OcrSettings)."""

    def __init__(self, names: tuple[str, ...], *, min_score: float, min_margin: float):
        self.names = names
        self.by_key = {match_key(n): n for n in names}
        self.min_score = min_score
        self.min_margin = min_margin
        self._encoded: dict[int, tuple[list[str], list[list[int]]]] = {}

    def resolve(self, text: str, log_probs: np.ndarray | None = None, charset: list[str] | None = None) -> LexiconMatch:
        key = match_key(text)
        if key in self.by_key:
            return LexiconMatch(self.by_key[key], "exact")
        if log_probs is not None and charset is not None:
            found = self._by_ctc(log_probs, charset)
            if found is not None:
                return found
        if key:
            nearest = min(self.names, key=lambda n: edit_distance(key, match_key(n)))
            if edit_distance(key, match_key(nearest)) <= max(1, len(match_key(nearest)) // 3):
                return LexiconMatch(nearest, "nearest")
        return LexiconMatch(None, "none")

    def _by_ctc(self, log_probs: np.ndarray, charset: list[str]) -> LexiconMatch | None:
        names, labels = self._encode(charset)
        if len(labels) < 2:
            return None
        scores = ctc_log_likelihood(log_probs, labels)
        best, second = np.argsort(-scores)[:2]
        score, margin = float(scores[best]), float(scores[best] - scores[second])
        if score > self.min_score and margin >= self.min_margin:
            return LexiconMatch(names[best], "ctc", round(score, 3), round(margin, 3))
        return None

    def _encode(self, charset: list[str]) -> tuple[list[str], list[list[int]]]:
        cached = self._encoded.get(id(charset))
        if cached is None:
            index = {symbol: i for i, symbol in enumerate(charset)}
            pairs = [(n, [index[c] for c in match_key(n) if c in index]) for n in self.names]
            # A name with a letter the model cannot emit cannot be scored fairly.
            pairs = [(n, label) for n, label in pairs if label and len(label) == len(match_key(n))]
            cached = ([n for n, _ in pairs], [label for _, label in pairs])
            self._encoded[id(charset)] = cached
        return cached
