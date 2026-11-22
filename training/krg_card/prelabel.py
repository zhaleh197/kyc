"""Pre-draw the field boxes on Kurdistan-region (Iraqi) national card fronts.

    python -m training.krg_card.prelabel CARDS_DIR BOXES_JSON --out .data/krg_card_labels

CARDS_DIR holds card images that are already cropped to the card (flat, as
the pipeline sees a card after rectification). BOXES_JSON holds the text-line
polygons of every image ({file name: [[[x, y], ...], ...]}, from a text
detector such as easyocr's CRAFT). The boxes written here are a first draft:
check and correct them with training/krg_card/label_tool.py before training.

The layout of the card's front (right to left, Sorani / Arabic labels on the
right, values left of a colon at about 77% of the width):

    portrait (left)          national id (12 digits, upper middle)
                             7 value rows: name, father, grandfather,
                             surname, mother, mother's father, sex
    document number          blood group (bottom middle)
    (bottom left)

Output: images/ (copies) and labels/ (YOLO txt, classes in CLASS_NAMES order).
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

import cv2
import numpy as np

from kyc.modules.ocr.profiles.iraq import IQ_CLASS_NAMES as CLASS_NAMES

ROWS = ("name", "father", "grandfather", "surname", "mother", "mother_grandfather", "sex")
COLON_X = 0.766  # values end at the colon
PORTRAIT = (0.05, 0.25, 0.315, 0.845)


def ink_rows(im: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Value rows found by dark ink left of the colon (light cards)."""
    h, w = im.shape[:2]
    x0, x1, y0, y1 = int(0.45 * w), int(0.727 * w), int(0.40 * h), int(0.92 * h)
    ink = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY) < 125
    prof = np.r_[ink.sum(1) > 3, False]
    bands, start = [], None
    for y, v in enumerate(prof):
        if v and start is None:
            start = y
        if not v and start is not None:
            bands.append([start, y])
            start = None
    merged: list[list[int]] = []
    for b in bands:  # the dots above / below a line are bands of their own
        if merged and b[0] - merged[-1][1] < 0.012 * h:
            merged[-1][1] = b[1]
        else:
            merged.append(b)
    merged = [b for b in merged if b[1] - b[0] > 0.02 * h]
    if not merged:
        return []
    med = float(np.median([b - a for a, b in merged]))
    rows = []
    for a, b in merged:  # two lines close together come out as one band
        k = max(1, round((b - a) / med))
        cuts = [a]
        for j in range(1, k):
            c = a + (b - a) * j // k
            lo, hi = c - int(0.3 * med), c + int(0.3 * med)
            cuts.append(lo + int(np.argmin(ink[lo:hi].sum(1))))
        cuts.append(b)
        for j in range(k):
            ra, rb = cuts[j], cuts[j + 1]
            cols = np.where(ink[ra:rb].sum(0) > 0)[0]
            if len(cols):
                rows.append((x0 + int(cols[0]), y0 + ra, x0 + int(cols[-1]) + 1, y0 + rb))
    return rows


def detector_rows(lines: list[np.ndarray], w: int, h: int) -> list[tuple[int, int, int, int]]:
    """Value rows from text-line boxes: a box that starts left of the colon
    is a value (a box that runs on into the label is cut at the colon); boxes
    on one line are joined."""
    found = []
    for b in lines:
        (bx0, by0), (bx1, by1) = b.min(0), b.max(0)
        cy = (by0 + by1) / 2
        if 0.40 * h < cy < 0.91 * h and 0.45 * w < bx0 < (COLON_X - 0.02) * w and bx1 > 0.55 * w:
            found.append([int(bx0), int(by0), int(min(bx1, COLON_X * w)), int(by1)])
    rows: list[list[int]] = []
    for r in sorted(found, key=lambda r: (r[1] + r[3]) / 2):
        if rows:
            last = rows[-1]
            overlap = min(last[3], r[3]) - max(last[1], r[1])
            if overlap > 0.6 * min(last[3] - last[1], r[3] - r[1]):
                rows[-1] = [min(last[0], r[0]), min(last[1], r[1]), max(last[2], r[2]), max(last[3], r[3])]
                continue
        rows.append(r)
    return [tuple(r) for r in rows]


def trim_colon(im: np.ndarray, box: tuple[int, int, int, int]) -> tuple[int, int, int, int]:
    """End a value box before the colon that follows it: a recogniser reads
    the colon's two dots as ن or ب. The colon is the thin ink run at the
    box's right end, with a gap before the value."""
    x0, y0, x1, y1 = box
    h, w = im.shape[:2]
    gray = cv2.cvtColor(im[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
    if gray.size == 0:
        return box
    cols = (gray < 125).sum(0) > 0
    runs, start = [], None
    for x, v in enumerate(np.r_[cols, False]):
        if v and start is None:
            start = x
        if not v and start is not None:
            runs.append((start, x))
            start = None
    if len(runs) < 2:
        return box
    (a, b), (_, prev_end) = runs[-1], runs[-2]
    if b - a < 0.012 * w and a - prev_end > 0.004 * w:
        return x0, y0, x0 + prev_end + int(0.004 * w), y1
    return box


def pick(lines, w, h, cx_range, cy_range, key):
    found = []
    for b in lines:
        (x0, y0), (x1, y1) = b.min(0), b.max(0)
        cx, cy = (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        if cx_range[0] < cx < cx_range[1] and cy_range[0] < cy < cy_range[1]:
            found.append((int(x0), int(y0), int(x1), int(y1)))
    return max(found, key=key) if found else None


def prelabel(im: np.ndarray, lines: list[np.ndarray]) -> tuple[dict, list[str]]:
    h, w = im.shape[:2]
    boxes: dict[str, tuple] = {"card": (0, 0, w, h)}
    boxes["portrait"] = tuple(int(v * s) for v, s in zip(PORTRAIT, (w, h, w, h), strict=True))
    problems = []
    # The digits sit just under the "national card" caption: the lower line.
    nid = pick(lines, w, h, (0.35, 0.75), (0.27, 0.42), key=lambda b: b[1])
    if nid:
        boxes["national_id"] = nid
    doc = pick(lines, w, h, (0.0, 0.35), (0.80, 1.0), key=lambda b: b[2] - b[0])
    if doc:
        boxes["document_number"] = doc
    blood = pick(lines, w, h, (0.45, 0.65), (0.85, 1.0), key=lambda b: -(b[2] - b[0]))
    if blood and blood[2] - blood[0] < 0.12 * w:
        boxes["blood_group"] = blood
    rows = detector_rows(lines, w, h)
    if len(rows) != len(ROWS):
        rows = ink_rows(im)
    if len(rows) != len(ROWS):
        problems.append(f"{len(rows)} value rows")
    for name, r in zip(ROWS, rows, strict=False):
        boxes[name] = trim_colon(im, r)
    problems += [f"no {c}" for c in CLASS_NAMES if c not in boxes]
    return boxes, problems


def to_yolo(boxes: dict, w: int, h: int) -> str:
    out = []
    for name, (x0, y0, x1, y1) in boxes.items():
        out.append(f"{CLASS_NAMES.index(name)} {(x0 + x1) / 2 / w:.6f} {(y0 + y1) / 2 / h:.6f} "
                   f"{(x1 - x0) / w:.6f} {(y1 - y0) / h:.6f}")
    return "\n".join(out) + "\n"


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cards", type=Path)
    ap.add_argument("boxes", type=Path)
    ap.add_argument("--out", type=Path, default=Path(".data/krg_card_labels"))
    ap.add_argument("--force", action="store_true", help="overwrite labels that already exist")
    args = ap.parse_args()
    lines_by_file = json.loads(args.boxes.read_text(encoding="utf-8"))
    (args.out / "images").mkdir(parents=True, exist_ok=True)
    (args.out / "labels").mkdir(parents=True, exist_ok=True)
    flagged = 0
    for p in sorted(args.cards.glob("*.jpg")):
        label = args.out / "labels" / (p.stem + ".txt")
        if label.exists() and not args.force:
            continue
        im = cv2.imdecode(np.fromfile(str(p), np.uint8), cv2.IMREAD_COLOR)
        lines = [np.array(b, float) for b in lines_by_file.get(p.name, [])]
        boxes, problems = prelabel(im, lines)
        shutil.copy2(p, args.out / "images" / p.name)
        label.write_text(to_yolo(boxes, im.shape[1], im.shape[0]), encoding="utf-8")
        if problems:
            flagged += 1
            print(f"{p.name}: {', '.join(problems)}")
    print(f"done; {flagged} images need a closer look")


if __name__ == "__main__":
    main()
