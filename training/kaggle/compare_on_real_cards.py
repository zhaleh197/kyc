"""Full OCR pipeline on the real cards, once per name recogniser.

    python -m training.kaggle.compare_on_real_cards                  # ppocr_arabic vs ppocr_fa_ckb
    python -m training.kaggle.compare_on_real_cards ppocr_arabic

A new model replaces the current one only if it reads more names here
without accepting more wrong ones. The photos and labels are personal data:
they stay on this machine (.data/ is git-ignored) and this prints counts only.

Labels: a TSV with a header `file first_name last_name father_name
national_id birth_date expiry_date`, one row per photo in --cards.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from kyc.core.config import OcrSettings, Settings  # noqa: E402
from kyc.modules.ocr.lexicon import match_key  # noqa: E402
from kyc.modules.ocr.pipeline import DocumentOcrPipeline  # noqa: E402

NAMES = ("first_name", "last_name", "father_name")
DIGITS = ("national_id", "birth_date", "expiry_date")


def score(backend: str, cards: Path, labels: dict[str, dict[str, str]]) -> None:
    ocr = OcrSettings(text_backend=backend, lexicon_scorer_backend=backend)
    pipe = DocumentOcrPipeline(Settings(ocr=ocr))
    right = dict.fromkeys(NAMES + DIGITS, 0)
    wrong: list[str] = []
    review = 0
    seconds = 0.0
    for fname, truth in labels.items():
        t0 = time.perf_counter()
        res = pipe.run_bytes((cards / fname).read_bytes())
        seconds += time.perf_counter() - t0
        for key in NAMES + DIGITS:
            field = res.data["fields"].get(key) or {}
            value = field.get("value") or ""
            same = match_key(value) == match_key(truth[key]) if key in NAMES else value == truth[key]
            if same:
                right[key] += 1
            elif not value or field.get("lexicon_match") == "none":
                review += 1
            else:
                wrong.append(key)
    n = len(labels)
    print(
        f"{backend:14} names {sum(right[k] for k in NAMES)}/{3 * n} "
        f"(first {right['first_name']}, last {right['last_name']}, father {right['father_name']})  "
        f"digits {sum(right[k] for k in DIGITS)}/{3 * n}  wrong-but-accepted {len(wrong)} "
        f"{sorted(set(wrong))}  review/empty {review}  {seconds / n:.1f}s/card",
        flush=True,
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("backends", nargs="*", default=["ppocr_arabic", "ppocr_fa_ckb"])
    p.add_argument("--cards", type=Path, default=REPO_ROOT / ".data/id_cards_real")
    p.add_argument("--labels", type=Path, default=REPO_ROOT / ".data/tmp/real_labels.tsv")
    args = p.parse_args()
    rows = [line.split("\t") for line in args.labels.read_text(encoding="utf-8").splitlines() if line.strip()]
    labels = {r[0]: dict(zip(rows[0][1:], r[1:], strict=True)) for r in rows[1:]}
    for backend in args.backends:
        score(backend, args.cards, labels)


if __name__ == "__main__":
    main()
