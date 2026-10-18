"""Calibrate and measure document forensics (06a) on genuine vs synthetic
tampered cards.

    python -m scripts.eval_forensics [--data .data/ir_card_norm] [--write]

WARNING (2026-10-19): .data/ir_card_norm is NOT genuine - every card in it
was edited by hand (spec 06 section 12). Do not --write a calibration from
it; the genuine reference must be fresh real captures.

Genuine set: the labelled card photos, each also forwarded the way messaging
apps do it (downscaled to 1280 px, re-saved at JPEG 70-90) - F1 counts those
as genuine. Tampered set, made from the same photos with the label boxes:

    portrait_paste   another card's portrait resized into the img box
    digit_swap       two digit-wide slices of idnumber swapped (copy-move)
    digit_paste      one digit-wide slice of idnumber copied from another card

each saved again at JPEG 85-95, as an editor would. The train split fits the
genuine reference (median / spread per region and feature) and the warn
threshold at the configured false-positive rate; val+test report FPR and TPR.
With --write the reference goes to models/forensics_calibration.json, which
the pipeline reads.
"""

from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

from kyc.modules.forensics.calibration import REGIONS, deviations, fit, region_score
from kyc.modules.forensics.signals import FEATURES, PhotoMaps

CLASS_OF = {"portrait": 5, "national_id": 4, "birth_date": 0, "expiry_date": 1}  # label class ids


def load_boxes(label: Path, w: int, h: int) -> dict[str, tuple[int, int, int, int]]:
    boxes = {}
    for line in label.read_text().splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        cls, pts = int(parts[0]), np.array(parts[1:], float)
        if len(pts) == 4:  # cx cy bw bh
            cx, cy, bw, bh = pts
            xs, ys = np.array([cx - bw / 2, cx + bw / 2]), np.array([cy - bh / 2, cy + bh / 2])
        else:
            xs, ys = pts[0::2], pts[1::2]
        for name, c in CLASS_OF.items():
            if c == cls:
                boxes[name] = (int(xs.min() * w), int(ys.min() * h), int(xs.max() * w), int(ys.max() * h))
    return boxes


def jpeg(img: np.ndarray, quality: int) -> np.ndarray:
    return cv2.imdecode(cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, quality])[1], cv2.IMREAD_COLOR)


def forward(img: np.ndarray, boxes: dict, rng: random.Random) -> tuple[np.ndarray, dict]:
    """Like a messaging app: downscale to 1280 long side, re-save at 70-90."""
    h, w = img.shape[:2]
    s = min(1.0, 1280 / max(h, w))
    if s < 1.0:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    boxes = {k: tuple(int(v * s) for v in b) for k, b in boxes.items()}
    return jpeg(img, rng.randint(70, 90)), boxes


def paste_portrait(img, boxes, donor, donor_boxes, rng):
    x1, y1, x2, y2 = boxes["portrait"]
    a1, b1, a2, b2 = donor_boxes["portrait"]
    patch = cv2.resize(donor[b1:b2, a1:a2], (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[y1:y2, x1:x2] = patch
    return jpeg(out, rng.randint(85, 95))


def digit_slices(box, n=10):
    x1, y1, x2, y2 = box
    step = (x2 - x1) / n
    return [(int(x1 + i * step), y1, int(x1 + (i + 1) * step), y2) for i in range(n)]


def swap_digits(img, boxes, rng):
    cells = digit_slices(boxes["national_id"])
    i, j = rng.sample(range(len(cells)), 2)
    (a1, b1, a2, b2), (c1, d1, c2, d2) = cells[i], cells[j]
    wdt = min(a2 - a1, c2 - c1)
    out = img.copy()
    out[b1:b2, a1:a1 + wdt], out[d1:d2, c1:c1 + wdt] = img[d1:d2, c1:c1 + wdt], img[b1:b2, a1:a1 + wdt]
    return jpeg(out, rng.randint(85, 95))


def paste_digit(img, boxes, donor, donor_boxes, rng):
    x1, y1, x2, y2 = rng.choice(digit_slices(boxes["national_id"]))
    a1, b1, a2, b2 = rng.choice(digit_slices(donor_boxes["national_id"]))
    patch = cv2.resize(donor[b1:b2, a1:a2], (x2 - x1, y2 - y1), interpolation=cv2.INTER_CUBIC)
    out = img.copy()
    out[y1:y2, x1:x2] = patch
    return jpeg(out, rng.randint(85, 95))


def features(img, boxes) -> dict[str, dict[str, float]]:
    maps = PhotoMaps(img)
    out = {}
    for region, cells in REGIONS.items():
        if region in boxes:
            f = maps.features(boxes[region], cells)
            if f is not None:
                out[region] = f
    return out


def measure(data: Path, limit: int, rng: random.Random) -> list:
    items = []
    for split in ("train", "val", "test"):
        for p in sorted((data / split / "images").glob("*")):
            label = data / split / "labels" / (p.stem + ".txt")
            if label.exists():
                items.append((split, p, label))
    if limit:
        items = rng.sample(items, limit)

    loaded = []
    for split, p, label in items:
        img = cv2.imread(str(p))
        boxes = load_boxes(label, img.shape[1], img.shape[0])
        if "portrait" in boxes and "national_id" in boxes:
            loaded.append((split, img, boxes))
    print(f"{len(loaded)} labelled cards")

    samples = []  # (split, kind, {region: cells})
    for n, (split, img, boxes) in enumerate(loaded):
        _, donor, donor_boxes = loaded[(n + 1 + rng.randrange(len(loaded) - 1)) % len(loaded)]
        fwd, fwd_boxes = forward(img, boxes, rng)
        samples.append((split, "genuine", features(img, boxes)))
        samples.append((split, "genuine_forwarded", features(fwd, fwd_boxes)))
        samples.append((split, "portrait_paste", features(paste_portrait(img, boxes, donor, donor_boxes, rng), boxes)))
        samples.append((split, "digit_swap", features(swap_digits(img, boxes, rng), boxes)))
        samples.append((split, "digit_paste", features(paste_digit(img, boxes, donor, donor_boxes, rng), boxes)))
        if n % 50 == 0:
            print(f"  {n}/{len(loaded)}", flush=True)
    return samples


# Which region each tamper type should fire in.
TARGET = {"portrait_paste": "portrait", "digit_swap": "national_id", "digit_paste": "national_id"}


def held_out_report(calib: dict, held: list) -> dict:
    report = {}
    for region in REGIONS:
        gen = [region_score(calib, region, f[region])[0] for k, f in held if k.startswith("genuine") and region in f]
        report[region] = {"fpr": round(float(np.mean([g >= 1 for g in gen])), 4), "n_genuine": len(gen)}
    for kind, region in TARGET.items():
        hits = [region_score(calib, region, f[region]) for k, f in held if k == kind and region in f]
        fired = defaultdict(int)
        for score, feat in hits:
            if score >= 1:
                fired[feat] += 1
        report[kind] = {"region": region, "tpr": round(float(np.mean([h[0] >= 1 for h in hits])), 4),
                        "n": len(hits), "fired_by": dict(fired)}
    # AUC of each feature's deviation: which signal carries the result.
    auc = defaultdict(dict)
    for kind, region in TARGET.items():
        g = [deviations(calib, region, f[region]) for k, f in held if k.startswith("genuine") and region in f]
        t = [deviations(calib, region, f[region]) for k, f in held if k == kind and region in f]
        for feat in FEATURES:
            gv, tv = np.array([d[feat] for d in g]), np.array([d[feat] for d in t])
            auc[kind][feat] = round(float(np.mean(tv[:, None] > gv[None, :])), 3)
    report["auc_by_feature"] = dict(auc)
    return report


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, default=Path(".data/ir_card_norm"))
    ap.add_argument("--fpr", type=float, default=0.02, help="target false-positive rate per region (F1)")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--cache", type=Path, default=None, help="reuse / store the measured features (json)")
    args = ap.parse_args()
    rng = random.Random(1405)

    if args.cache and args.cache.exists():
        samples = json.loads(args.cache.read_text())
        rename = {"idnumber": "national_id", "birthday": "birth_date", "exp": "expiry_date"}  # older caches
        samples = [(s, k, {rename.get(r, r): v for r, v in f.items()}) for s, k, f in samples]
    else:
        samples = measure(args.data, args.limit, rng)
        if args.cache:
            args.cache.write_text(json.dumps(samples))

    genuine_train = [f for s, k, f in samples if s == "train" and k.startswith("genuine")]
    calib = fit(genuine_train, args.fpr)
    calib["ela_quality"] = 90  # PhotoMaps default; the pipeline must use the same
    calib["measured_on"] = {
        "dataset": str(args.data), "genuine_train_photos": len(genuine_train),
        "fit": "train split (originals + forwarded copies); median/MAD per region and feature; "
               "warn_above per feature = genuine deviation quantile at 1 - fpr/3",
    }
    report = held_out_report(calib, [(k, f) for s, k, f in samples if s != "train"])
    calib["held_out_report"] = report
    print(json.dumps(report, indent=2))
    if args.write:
        out = Path("models/forensics_calibration.json")
        out.write_text(json.dumps(calib, indent=2), encoding="utf-8")
        print(f"wrote {out}")


if __name__ == "__main__":
    main()
