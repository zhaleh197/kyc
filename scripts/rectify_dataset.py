"""Rectify a labelled card dataset so training matches serving.

    python -m scripts.rectify_dataset --src .data/ir_card_yolo --out .data/ir_card_rectified

The pipeline flattens the card *before* detection, so the detector only ever
sees rectified cards at inference time. Training it on raw phone photos means
the training and serving distributions disagree, and the detector spends its
capacity learning perspective it will never encounter.

This runs the exact rectifier from `kyc/modules/ocr/rectify.py` over every
image and moves each annotation through the same 3x3 transform, so labels stay
on their fields. One implementation, one geometry - the dataset cannot drift
from the runtime.

Images where the card outline cannot be found are still emitted (resized, not
warped) and reported. Use --require-border to drop them instead.
"""

from __future__ import annotations

import argparse
import sys
from collections import Counter
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kyc.modules.ocr import rectify  # noqa: E402
from kyc.modules.ocr.profiles import get_profile  # noqa: E402
from scripts.prepare_dataset import find_pairs, parse_label, write_data_yaml  # noqa: E402


def label_to_points(parts: str, width: int, height: int) -> list[tuple[float, float]] | None:
    """Return the label's outline as pixel points, for boxes and polygons alike.

    Roboflow exports a segmentation project as polygons - `class x1 y1 x2 y2
    ... xn yn` - not as `class cx cy w h`. Reading only the first four numbers
    would silently treat the first two polygon *vertices* as a box, so every
    annotation would end up in the wrong place with no error anywhere.

    Four values means a YOLO box; any larger even count means a polygon.
    """
    try:
        values = [float(v) for v in parts.split()]
    except ValueError:
        return None

    if len(values) == 4:
        cx, cy, bw, bh = values
        x1, y1 = (cx - bw / 2) * width, (cy - bh / 2) * height
        x2, y2 = (cx + bw / 2) * width, (cy + bh / 2) * height
        return [(x1, y1), (x2, y1), (x2, y2), (x1, y2)]

    if len(values) >= 6 and len(values) % 2 == 0:
        return [(values[i] * width, values[i + 1] * height) for i in range(0, len(values), 2)]

    return None


def xyxy_to_yolo(box: tuple[float, float, float, float], width: int, height: int) -> str:
    x1, y1, x2, y2 = box
    cx = (x1 + x2) / 2 / width
    cy = (y1 + y2) / 2 / height
    bw = (x2 - x1) / width
    bh = (y2 - y1) / height
    return f"{cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="Dataset with train/val[/test] splits")
    parser.add_argument("--out", required=True)
    parser.add_argument("--profile", default="ir_national_card_front")
    parser.add_argument(
        "--require-border",
        action="store_true",
        help="Drop images where the card outline was not found instead of resizing them",
    )
    args = parser.parse_args()

    profile = get_profile(args.profile)
    class_names = profile.class_names
    target_size = profile.rectified_size
    src, out = Path(args.src), Path(args.out)
    if not src.is_dir():
        raise SystemExit(f"Source dataset not found: {src}")

    splits = [d.name for d in sorted(src.iterdir()) if d.is_dir() and (d / "images").is_dir()]
    if not splits:
        raise SystemExit(f"No split folders with an images/ subfolder under {src}")

    counts: Counter[str] = Counter()
    written: dict[str, int] = {}
    no_border = 0
    dropped_images = 0
    dropped_boxes = 0
    malformed = 0

    for split in splits:
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)
        kept = 0

        for image_path, label_path in find_pairs(src / split):
            image = cv2.imread(str(image_path))
            if image is None:
                print(f"[warn] unreadable, skipped: {image_path}")
                continue
            height, width = image.shape[:2]

            card, reasons, matrix = rectify.rectify_with_transform(image, target_size, profile.aspect_ratio)
            border_missing = any(r.code == "card_border_not_found" for r in reasons)
            if border_missing:
                no_border += 1
                if args.require_border:
                    dropped_images += 1
                    continue

            lines = []
            if label_path is not None:
                for class_id, rest in parse_label(label_path):
                    if not 0 <= class_id < len(class_names):
                        continue
                    points = label_to_points(rest, width, height)
                    if points is None:
                        malformed += 1
                        continue
                    # Every vertex goes through the warp, then we take the
                    # bounding box - the detector is a box model.
                    moved = rectify.transform_points(matrix, points, target_size)
                    if moved is None:
                        dropped_boxes += 1
                        continue
                    counts[class_names[class_id]] += 1
                    lines.append(f"{class_id} {xyxy_to_yolo(moved, *target_size)}")

            cv2.imwrite(str(out / split / "images" / f"{image_path.stem}.jpg"), card)
            body = "\n".join(lines)
            (out / split / "labels" / f"{image_path.stem}.txt").write_text(
                body + "\n" if body else "", encoding="utf-8"
            )
            kept += 1

        written[split] = kept

    write_data_yaml(out, class_names, list(written))

    print(f"[done] {out}   cards rendered at {target_size[0]}x{target_size[1]}")
    for split, count in written.items():
        print(f"  {split:<6} {count:5d} images")
    print("\n  annotations per class:")
    for name in class_names:
        flag = "   <-- NONE" if counts[name] == 0 else ""
        print(f"    {name:<14} {counts[name]:6d}{flag}")

    if no_border:
        share = no_border / max(1, no_border + sum(written.values()))
        print(f"\n[info] card outline not found in {no_border} image(s)")
        if args.require_border:
            print(f"       {dropped_images} dropped (--require-border)")
        else:
            print("       those were resized instead of warped; they stay in the set")
        if share > 0.25:
            print("[warn] that is a large share. The classical rectifier struggles on")
            print("       busy backgrounds and borderless scans - worth inspecting a few")
            print("       before training, since a mis-rectified card teaches bad boxes.")
    if malformed:
        print(f"\n[warn] {malformed} label line(s) had a shape we could not parse and were dropped")
    if dropped_boxes:
        print(f"\n[info] {dropped_boxes} annotation(s) fell outside the card after rectification and were dropped")

    missing = [n for n in class_names if counts[n] == 0]
    if missing:
        print(f"\n[error] no annotations at all for: {missing}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
