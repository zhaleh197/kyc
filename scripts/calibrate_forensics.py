"""Calibrate and measure forensics (06) on real card photos.

    python -m scripts.calibrate_forensics PHOTOS_DIR [--edited .data/ir_card_norm] [--write]

PHOTOS_DIR holds one folder per kind; file names start with the card id
(`c03-samsung-07.jpg` -> card c03):

    genuine/     the real card, photographed straight from a phone camera
    forwarded/   genuine photos sent through a messenger as "photo" (recompressed)
    screen/      the card shown on a monitor / phone and photographed
    print/       a printed copy of the card, photographed

Everything is located by the OCR detector, which also gives the boxes on the
photo as submitted. Cards are split into fit / held-out groups by card id, so
a card's own photos never vouch for themselves. The genuine reference (median,
spread, per-feature limits) is fitted on fit-group genuine + forwarded; then on
held-out cards:

    FPR    genuine and forwarded photos that warn (target <= 2%, spec 06 F1)
    TPR    screen / print photos flagged by the "card" (recapture) region
    TPR    synthetic edits made from held-out genuine photos (portrait paste,
           digit swap, digit paste - see eval_forensics.py)

With --edited, also the share of the hand-edited labelled set that warns per
region - real human edits, but Roboflow resized and re-saved those files,
which erases much of the trace (spec 06 section 12).
--write stores the reference in models/forensics_calibration.json.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from kyc.core.config import get_settings
from kyc.core.errors import KycError
from kyc.modules.forensics.calibration import CARD, REGIONS, deviations, fit, region_score
from kyc.modules.forensics.recapture import card_features
from kyc.modules.forensics.signals import PhotoMaps
from kyc.modules.ocr.pipeline import DocumentOcrPipeline

from .eval_forensics import load_boxes, paste_digit, paste_portrait, swap_digits

KINDS = ("genuine", "forwarded", "screen", "print")
ATTACKS = ("screen", "print")
EDITS = ("portrait_paste", "digit_swap", "digit_paste")
EDIT_REGION = {"portrait_paste": "portrait", "digit_swap": "national_id", "digit_paste": "national_id"}


def measure(img: np.ndarray, boxes: dict) -> dict:
    """{region: cells} for every region we can see."""
    maps = PhotoMaps(img)
    out = {}
    for region, cells in REGIONS.items():
        if boxes.get(region):
            f = maps.features(boxes[region], cells)
            if f is not None:
                out[region] = f
    if boxes.get(CARD):
        f = card_features(img, boxes[CARD])
        if f is not None:
            out[CARD] = [f]
    return out


def photos(root: Path) -> list[tuple[str, str, Path]]:
    found = []
    for kind in KINDS:
        for p in sorted((root / kind).rglob("*")) if (root / kind).exists() else []:
            if p.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp", ".heic"}:
                found.append((kind, p.name.split("-")[0].lower(), p))
    return found


def rate(values: list[bool]) -> float | None:
    return round(float(np.mean(values)), 4) if values else None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("photos", type=Path)
    ap.add_argument("--edited", type=Path, default=None, help="hand-edited labelled set (yolo layout)")
    ap.add_argument("--fpr", type=float, default=0.02)
    ap.add_argument("--holdout", type=float, default=0.3, help="share of card ids held out")
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    rng = random.Random(1405)

    ocr = DocumentOcrPipeline(get_settings())
    rows = []  # (kind, card, feats, path of a genuine photo, boxes) - photos reloaded on demand
    for kind, card, path in photos(args.photos):
        data = path.read_bytes()
        img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
        if img is None:
            print(f"skip (unreadable): {path}")
            continue
        try:
            boxes = ocr.run_bytes(data).data.get("source_boxes", {})
        except KycError as exc:
            print(f"skip ({exc.code}): {path}")
            continue
        rows.append((kind, card, measure(img, boxes), path if kind == "genuine" else None, boxes))
        del img
        print(f"  {kind:9s} {path.name}: {sorted(rows[-1][2])}", flush=True)

    cards = sorted({c for k, c, *_ in rows if k == "genuine"})
    if not cards:
        raise SystemExit("no genuine photos found")
    held = set(rng.sample(cards, max(1, round(len(cards) * args.holdout)))) if len(cards) >= 4 else set()
    if not held:
        print("WARNING: fewer than 4 cards - fitting and reporting on the same photos (optimistic)")
    fit_rows = [f for k, c, f, *_ in rows if k in ("genuine", "forwarded") and (c not in held or not held)]
    test = [row for row in rows if row[1] in held or not held]

    calib = fit(fit_rows, args.fpr, (*REGIONS, CARD))
    calib["ela_quality"] = 90
    calib["measured_on"] = {"photos": str(args.photos), "cards": cards, "held_out_cards": sorted(held),
                            "fit_photos": len(fit_rows)}

    def warns(feats: dict, regions=None) -> bool:
        return any(region_score(calib, r, feats[r])[0] >= 1 for r in (regions or calib["regions"]) if r in feats
                   and r in calib["regions"])

    report: dict = {"fitted_regions": sorted(calib["regions"])}
    for kind in KINDS:
        sub = [f for k, _, f, *_ in test if k == kind]
        if kind in ATTACKS:
            report[kind] = {"tpr_card": rate([warns(f, [CARD]) for f in sub]), "tpr_any": rate([warns(f) for f in sub]),
                            "n": len(sub)}
        else:
            report[kind] = {"fpr": rate([warns(f) for f in sub]), "n": len(sub)}
            by_region = defaultdict(list)
            for f in sub:
                for r in calib["regions"]:
                    if r in f:
                        by_region[r].append(region_score(calib, r, f[r])[0] >= 1)
            report[kind]["fpr_by_region"] = {r: rate(v) for r, v in by_region.items()}

    # Synthetic edits on held-out genuine photos: donor = another held-out card.
    genuine = [(c, p, b) for k, c, _, p, b in test if k == "genuine" and "portrait" in b and "national_id" in b]
    edits = defaultdict(list)
    for n, (_, path, b) in enumerate(genuine):
        _, donor_path, db = genuine[(n + 1) % len(genuine)]
        img, donor = cv2.imread(str(path)), cv2.imread(str(donor_path))
        made = {"portrait_paste": paste_portrait(img, b, donor, db, rng), "digit_swap": swap_digits(img, b, rng),
                "digit_paste": paste_digit(img, b, donor, db, rng)}
        for kind, edited in made.items():
            f = measure(edited, b)
            region = EDIT_REGION[kind]
            if region in f and region in calib["regions"]:
                edits[kind].append(region_score(calib, region, f[region])[0] >= 1)
    report["synthetic_edits"] = {k: {"tpr": rate(v), "n": len(v)} for k, v in edits.items()}

    if args.edited:
        flagged = defaultdict(list)
        for split in ("train", "val", "test"):
            for p in sorted((args.edited / split / "images").glob("*")):
                label = args.edited / split / "labels" / (p.stem + ".txt")
                img = cv2.imread(str(p))
                if img is None or not label.exists():
                    continue
                f = measure(img, load_boxes(label, img.shape[1], img.shape[0]))
                for r in calib["regions"]:
                    if r in f:
                        flagged[r].append(region_score(calib, r, f[r])[0] >= 1)
        report["hand_edited_set_warn_rate"] = {r: {"rate": rate(v), "n": len(v)} for r, v in flagged.items()}

    # Which features carry the attacks (AUC of the deviation, card region).
    if CARD in calib["regions"]:
        g = [deviations(calib, CARD, f[CARD]) for k, _, f, *_ in test if k in ("genuine", "forwarded") and CARD in f]
        for kind in ATTACKS:
            a = [deviations(calib, CARD, f[CARD]) for k, _, f, *_ in test if k == kind and CARD in f]
            if g and a:
                report[kind]["auc_by_feature"] = {
                    feat: round(float(np.mean(np.array([d[feat] for d in a])[:, None]
                                              > np.array([d[feat] for d in g])[None, :])), 3)
                    for feat in calib["regions"][CARD]["median"]
                }
    calib["held_out_report"] = report
    print(json.dumps(report, indent=2, ensure_ascii=False))
    if args.write:
        out = get_settings().model_path(get_settings().forensics.calibration_file)
        out.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()


