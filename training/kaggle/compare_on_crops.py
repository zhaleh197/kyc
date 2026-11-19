"""Score name recognisers on labelled field crops.

    python -m training.kaggle.compare_on_crops .data/tmp/ckb_eval/labels.tsv
    python -m training.kaggle.compare_on_crops LIST.tsv --backends ppocr_arabic ppocr_fa_ckb

LIST.tsv: `path<TAB>label` per line, paths relative to the file. Prints
counts only (the crops may come from real cards and stay on this machine).

Two scores per backend:

    letters   the name as the service compares it (lexicon.match_key):
              Arabic ي ك ة ھ folded to Persian, spaces / ZWNJ ignored
    base      also folding the Persian / Kurdish letters an Arabic-trained
              model confuses (گ->ک, ژ->ز, ە->ه, ڕ->ر, ێ->ی, ۆ->و, ڵ->ل, ڤ->ف):
              what is left is a misread, not a missing letter
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))

from kyc.core.config import Settings  # noqa: E402
from kyc.modules.ocr.lexicon import match_key  # noqa: E402
from kyc.modules.ocr.recognizers import get_recognizer  # noqa: E402

_BASE_FOLD = str.maketrans("گژپچەڕێۆڵڤ", "کزبجهریولف")
KURDISH_ONLY = set("ەڕێۆڵڤ")


def base_key(text: str) -> str:
    return match_key(text).translate(_BASE_FOLD)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("tsv", type=Path)
    p.add_argument("--backends", nargs="+", default=["ppocr_arabic", "ppocr_fa_ckb"])
    args = p.parse_args()
    rows = [line.split("\t") for line in args.tsv.read_text(encoding="utf-8").splitlines() if "\t" in line]
    crops = [(cv2.imread(str(args.tsv.parent / rel)), label) for rel, label in rows]
    settings = Settings()
    for backend in args.backends:
        try:
            rec = get_recognizer(backend, settings)
        except Exception as exc:  # noqa: BLE001 - a missing model is reported, not fatal
            print(f"{backend:14} not available: {exc}")
            continue
        total = {"all": [0, 0, 0], "with Kurdish letter": [0, 0, 0]}
        for img, label in crops:
            got = rec.recognize(img).text
            groups = ["all"] + (["with Kurdish letter"] if set(label) & KURDISH_ONLY else [])
            for g in groups:
                t = total[g]
                t[0] += match_key(got) == match_key(label)
                t[1] += base_key(got) == base_key(label)
                t[2] += 1
        line = "  ".join(f"{g}: letters {a}/{n} ({a / n:.0%}), base {b}/{n} ({b / n:.0%})"
                         for g, (a, b, n) in total.items() if n)
        print(f"{backend:14} {line}", flush=True)


if __name__ == "__main__":
    main()
