"""Bootstrap a labelled crop set for training the CRNN recognisers.

    python -m scripts.bootstrap_crnn_dataset --src .data/ir_card_norm --out .data/crnn_bootstrap

Runs the trained field detector over every image in a YOLO-format dataset (or
a flat folder of photos), crops each text/digit/date field exactly the way
the serving pipeline does, and writes an initial guess from easyocr next to
each crop.

The guess is not meant to be correct - it exists so correcting is faster than
transcribing from scratch. Open `labels_text.tsv` / `labels_digits.tsv` (any
spreadsheet app or text editor handles tab-separated files), fix the second
column, delete the row for a crop that is not legible, and the files are then
ready for training/kaggle/train_crnn.py as-is:

    python training/kaggle/train_crnn.py --labels .data/crnn_bootstrap/labels_text.tsv \
        --name crnn_fa_text
    python training/kaggle/train_crnn.py --labels .data/crnn_bootstrap/labels_digits.tsv \
        --name crnn_fa_digits --charset "0123456789/"

Crops are made with the exact padding and upscale settings the serving
pipeline uses (kyc/core/config.py's `field_crop_padding` /
`upscale_small_crops_to`). A CRNN trained on crops shaped differently from
what it is served in production would be learning a distribution it will
never see again - the same lesson that shaped `rectify=false` for the field
detector on this dataset.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kyc.core.config import get_settings  # noqa: E402
from kyc.core.errors import KycError  # noqa: E402
from kyc.core.imaging import crop_with_padding, upscale_to_height  # noqa: E402
from kyc.modules.ocr.detector import FieldDetector, best_per_class  # noqa: E402
from kyc.modules.ocr.profiles import FieldKind, get_profile  # noqa: E402
from kyc.modules.ocr.recognizers import get_recognizer  # noqa: E402

SPLIT_DIRS = ("train", "val", "valid", "test")
IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp"}


def find_images(src: Path) -> list[tuple[str, Path]]:
    """(split, path) pairs from a YOLO dataset's split folders, or a flat folder.

    The split name is folded into the crop filename so images with the same
    stem in `train/` and `test/` (which Roboflow's hashed filenames make
    unlikely but not impossible) cannot collide.
    """
    found: list[tuple[str, Path]] = []
    for split in SPLIT_DIRS:
        images_dir = src / split / "images"
        if images_dir.is_dir():
            found += [(split, p) for p in sorted(images_dir.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    if found:
        return found
    if (src / "images").is_dir():
        return [("all", p) for p in sorted((src / "images").iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]
    return [("all", p) for p in sorted(src.iterdir()) if p.suffix.lower() in IMAGE_SUFFIXES]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", required=True, help="YOLO dataset root (train/val/test) or a flat folder of photos")
    parser.add_argument("--out", required=True, help="Where crops and label TSVs are written")
    parser.add_argument("--profile", default="ir_national_card_front")
    parser.add_argument("--conf", type=float, default=None, help="Detector confidence threshold; defaults to config")
    parser.add_argument("--iou", type=float, default=None, help="Detector NMS IoU threshold; defaults to config")
    parser.add_argument("--text-backend", default=None, help="Override the text recogniser used for the guess")
    parser.add_argument("--digit-backend", default=None, help="Override the digit recogniser used for the guess")
    parser.add_argument("--limit", type=int, default=None, help="Only process the first N images, for a quick check")
    args = parser.parse_args()

    settings = get_settings()
    try:
        profile = get_profile(args.profile)
    except KycError as exc:
        raise SystemExit(str(exc)) from exc

    cfg = settings.ocr
    conf = args.conf if args.conf is not None else cfg.detector_conf
    iou = args.iou if args.iou is not None else cfg.detector_iou
    text_backend = args.text_backend or cfg.text_backend
    digit_backend = args.digit_backend or cfg.digit_backend

    try:
        detector = FieldDetector.load(
            settings.model_path(profile.detector_model), profile.class_names, settings.onnx_providers
        )
    except KycError as exc:
        raise SystemExit(str(exc)) from exc

    src = Path(args.src)
    images = find_images(src)
    if args.limit:
        images = images[: args.limit]
    if not images:
        raise SystemExit(f"No images found under {src}")
    print(f"[bootstrap] {len(images)} image(s) from {src}")
    print(f"[bootstrap] detector conf={conf} iou={iou} | text={text_backend} digits={digit_backend}")
    print("[bootstrap] this calls the OCR backend once per field per image - expect roughly 1-2s per call on CPU")

    out = Path(args.out)
    (out / "crops" / "text").mkdir(parents=True, exist_ok=True)
    (out / "crops" / "digits").mkdir(parents=True, exist_ok=True)

    rows: dict[str, list[tuple[str, str]]] = {"text": [], "digits": []}
    found_counts = dict.fromkeys((f.key for f in profile.value_fields), 0)
    missing_counts = dict.fromkeys((f.key for f in profile.value_fields), 0)

    for index, (split, image_path) in enumerate(images, start=1):
        img = cv2.imread(str(image_path))
        if img is None:
            print(f"[warn] could not read {image_path}; skipped")
            continue

        detections = detector.detect(img, conf, iou)
        located = best_per_class(detections)
        stem = f"{split}_{image_path.stem}"

        for spec in profile.value_fields:
            detection = located.get(spec.detector_class)
            if detection is None:
                missing_counts[spec.key] += 1
                continue

            crop = crop_with_padding(img, detection.box, cfg.field_crop_padding)
            crop = upscale_to_height(crop, cfg.upscale_small_crops_to)

            is_digit_like = spec.kind in (FieldKind.DIGITS, FieldKind.DATE)
            bucket = "digits" if is_digit_like else "text"
            backend_id = digit_backend if is_digit_like else text_backend
            recognizer = get_recognizer(backend_id, settings)
            guess = recognizer.recognize(crop, allowlist=spec.charset_hint)

            crop_name = f"{stem}__{spec.key}.png"
            cv2.imwrite(str(out / "crops" / bucket / crop_name), crop)
            # \t and \n cannot survive the TSV format; a recognition guess
            # should never contain either, but strip defensively rather than
            # emit a row that shifts every column after it.
            clean_text = guess.text.replace("\t", " ").replace("\n", " ").strip()
            rows[bucket].append((f"crops/{bucket}/{crop_name}", clean_text))
            found_counts[spec.key] += 1

        if index % 25 == 0 or index == len(images):
            print(f"[bootstrap] {index}/{len(images)} images processed")

    for bucket, path in (("text", out / "labels_text.tsv"), ("digits", out / "labels_digits.tsv")):
        with path.open("w", encoding="utf-8", newline="") as handle:
            for relative_path, text in rows[bucket]:
                handle.write(f"{relative_path}\t{text}\n")
        print(f"[done] {path}  ({len(rows[bucket])} rows)")

    print("\nfields located:")
    for spec in profile.value_fields:
        total = found_counts[spec.key] + missing_counts[spec.key]
        print(f"  {spec.key:<14} {found_counts[spec.key]:4d}/{total:<4d} found")

    print(
        "\nNext:\n"
        f"  1. Open {out / 'labels_text.tsv'} and {out / 'labels_digits.tsv'}\n"
        "     Correct the guessed text in the second column; delete the row for any crop\n"
        "     that is not legible. The crops are next to the TSVs under crops/text and\n"
        "     crops/digits, named <split>_<image>__<field>.png.\n"
        "  2. Train:\n"
        f"     python training/kaggle/train_crnn.py --labels {out / 'labels_text.tsv'} --name crnn_fa_text\n"
        f"     python training/kaggle/train_crnn.py --labels {out / 'labels_digits.tsv'} "
        '--name crnn_fa_digits --charset "0123456789/"'
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
