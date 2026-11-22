"""Run the whole OCR pipeline on Kurdistan card images, as a phone would
send them, and report what it found.

    python -m training.krg_card.evaluate .data/krg_card_yolo/val/images [--labels .data/tmp/ckb_labels.tsv]

Each card is placed on a dark background with a slight perspective tilt
first, so the card has to be found and flattened like in a real photo.
Prints counts only - no names or numbers.

    located   fields the detector found, per field
    shape     national id = 12 digits, card number = letter + 8 digits,
              blood group = A/B/AB/O with an optional sign
    names     with --labels (file, name, father, grandfather, surname, mother,
              mother_father): name fields read exactly, after normalisation
"""

from __future__ import annotations

import argparse
import re
from collections import Counter
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np

from kyc.core.config import get_settings
from kyc.core.errors import KycError
from kyc.modules.ocr import normalize
from kyc.modules.ocr.pipeline import DocumentOcrPipeline
from kyc.modules.ocr.profiles import get_profile, register

PROFILE = "iq_national_card_front"
SHAPES = {
    "national_id": re.compile(r"\d{12}"),
    "document_number": re.compile(r"[A-Z]\d{8}"),
    "blood_group": re.compile(r"(AB|A|B|O)[+-]?"),
}
LABEL_COLUMNS = ("first_name", "father_name", "grandfather_name", "last_name", "mother_name", "mother_father_name")


def photograph(card: np.ndarray, tilt: float = 0.03) -> np.ndarray:
    h, w = card.shape[:2]
    pad = int(0.15 * w)
    scene = np.full((h + 2 * pad, w + 2 * pad, 3), 50, np.uint8)
    t = tilt * w
    src = np.float32([[0, 0], [w, 0], [w, h], [0, h]])
    dst = np.float32([[pad + t, pad], [pad + w, pad + t], [pad + w - t, pad + h], [pad, pad + h - t]])
    m = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(card, m, scene.shape[1::-1], borderValue=(50, 50, 50))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), m, scene.shape[1::-1])
    scene[mask > 0] = warped[mask > 0]
    return scene


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("images", type=Path)
    ap.add_argument("--labels", type=Path, default=None)
    ap.add_argument("--detector", default=None, help="another detector file in models/, to compare")
    ap.add_argument("--rectify", choices=("on", "off"), default=None, help="override the profile's rectify")
    ap.add_argument("--flat", action="store_true", help="send the card image as it is (no background, no tilt)")
    args = ap.parse_args()
    profile = get_profile(PROFILE)
    if args.detector or args.rectify:
        register(replace(profile, detector_model=args.detector or profile.detector_model,
                         rectify=profile.rectify if args.rectify is None else args.rectify == "on"))
    labels = {}
    if args.labels:
        for line in args.labels.read_text(encoding="utf-8").splitlines()[1:]:
            cols = line.split("\t")
            labels[Path(cols[0]).stem] = dict(zip(LABEL_COLUMNS, cols[1:], strict=False))
    pipeline = DocumentOcrPipeline(get_settings())
    fields = [f.key for f in get_profile(PROFILE).value_fields]
    located, shaped, decisions, errors, reason_codes = Counter(), Counter(), Counter(), Counter(), Counter()
    names, n = Counter(), 0
    for p in sorted(args.images.glob("*.jpg")):
        card = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        try:
            result = pipeline.run(card if args.flat else photograph(card), PROFILE)
        except KycError as exc:
            errors[exc.code] += 1
            continue
        n += 1
        decisions[result.decision.value] += 1
        for r in result.reasons:
            if r.severity.value != "info":
                reason_codes[r.code] += 1
        values = result.data["values"]
        for key in fields:
            located[key] += values.get(key) is not None
            if key in SHAPES and values.get(key):
                shaped[key] += bool(SHAPES[key].fullmatch(values[key]))
        for key, truth in labels.get(p.stem, {}).items():
            names[key, "n"] += 1
            truth = normalize.normalize_name(truth, get_profile(PROFILE).keep_letters)
            names[key, "ok"] += truth == (values.get(key) or "")
    print(f"cards run: {n}; refused: {dict(errors) or 0}; decisions: {dict(decisions)}")
    print(f"reasons (warn/error): {dict(reason_codes.most_common())}")
    for key in fields:
        extra = f"   right shape {shaped[key]}/{located[key]}" if key in SHAPES else ""
        print(f"  {key:20s} read {located[key]:3d}/{n}{extra}")
    if names:
        ok = sum(v for (k, t), v in names.items() if t == "ok")
        total = sum(v for (k, t), v in names.items() if t == "n")
        print(f"names exact: {ok}/{total}")
        for key in LABEL_COLUMNS:
            if names[key, "n"]:
                print(f"  {key:20s} {names[key, 'ok']}/{names[key, 'n']}")


if __name__ == "__main__":
    main()
