"""Generate a synthetic card dataset for smoke-testing the training chain.

    python -m scripts.make_synthetic_dataset --out .data/synthetic --count 80

This is not training data. It exists so the whole chain - train, export to
ONNX, load in kyc/modules/ocr/detector.py, run the pipeline - can be exercised
end to end on a laptop before any GPU time is spent on the real dataset. A
plumbing bug found here costs minutes; the same bug found after a Kaggle run
costs hours.

The layout mirrors the Iranian card closely enough that a nano model overfits
it in a handful of epochs, so a successful smoke run really does produce
detections rather than an empty output tensor.
"""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import cv2
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kyc.modules.ocr.profiles import get_profile  # noqa: E402
from scripts.prepare_dataset import write_data_yaml  # noqa: E402

CARD_W, CARD_H = 856, 540

# Fractional layout on the card, roughly where the real fields sit.
LAYOUT = {
    "img": (0.04, 0.22, 0.28, 0.78),
    "idnumber": (0.34, 0.20, 0.94, 0.31),
    "name": (0.34, 0.34, 0.80, 0.44),
    "lastname": (0.34, 0.47, 0.88, 0.57),
    "fathername": (0.34, 0.60, 0.78, 0.70),
    "birthday": (0.34, 0.73, 0.70, 0.83),
    "exp": (0.72, 0.73, 0.96, 0.83),
}


def draw_card(rng: random.Random) -> tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]:
    """Return a card image plus each field's box in fractional xyxy."""
    tint = rng.randint(215, 245)
    card = np.full((CARD_H, CARD_W, 3), (tint, tint - 4, tint - 10), np.uint8)

    # Background guilloche so the model has texture to ignore.
    for _ in range(60):
        y = rng.randint(0, CARD_H)
        cv2.line(card, (0, y), (CARD_W, y + rng.randint(-6, 6)), (tint - 18, tint - 22, tint - 26), 1)

    boxes: dict[str, tuple[float, float, float, float]] = {}
    for name, (fx1, fy1, fx2, fy2) in LAYOUT.items():
        jitter = rng.uniform(-0.008, 0.008)
        fx1, fx2 = fx1 + jitter, fx2 + jitter
        x1, y1 = int(fx1 * CARD_W), int(fy1 * CARD_H)
        x2, y2 = int(fx2 * CARD_W), int(fy2 * CARD_H)

        if name == "img":
            # A face-ish blob, so the portrait class is not just another text bar.
            cv2.rectangle(card, (x1, y1), (x2, y2), (190, 185, 180), -1)
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
            cv2.circle(card, (cx, cy - 20), (x2 - x1) // 3, (120, 110, 105), -1)
            cv2.ellipse(card, (cx, y2), ((x2 - x1) // 2, (y2 - y1) // 3), 0, 180, 360, (120, 110, 105), -1)
        else:
            # Text-like strokes of varying width and spacing.
            step = rng.randint(9, 14)
            for x in range(x1 + 4, x2 - 4, step):
                height = rng.randint(int((y2 - y1) * 0.4), int((y2 - y1) * 0.85))
                top = y1 + ((y2 - y1) - height) // 2
                cv2.line(card, (x, top), (x, top + height), (55, 50, 48), rng.randint(2, 4))

        boxes[name] = (fx1, fy1, fx2, fy2)

    boxes["idcard"] = (0.0, 0.0, 1.0, 1.0)
    return card, boxes


def photograph(
    card: np.ndarray,
    boxes: dict[str, tuple[float, float, float, float]],
    rng: random.Random,
) -> tuple[np.ndarray, dict[str, tuple[float, float, float, float]]]:
    """Put the card in a scene with perspective, so boxes move with it."""
    frame_w, frame_h = 1024, 768
    scene = np.full((frame_h, frame_w, 3), rng.randint(30, 90), np.uint8)

    scale = rng.uniform(0.62, 0.82)
    w, h = int(CARD_W * scale), int(CARD_H * scale)
    resized = cv2.resize(card, (w, h))
    ox = rng.randint(20, max(21, frame_w - w - 20))
    oy = rng.randint(20, max(21, frame_h - h - 20))
    tilt = rng.randint(-22, 22)

    src = np.array([[0, 0], [w - 1, 0], [w - 1, h - 1], [0, h - 1]], np.float32)
    dst = np.array(
        [
            [ox + max(0, tilt), oy],
            [ox + w - 1, oy + max(0, -tilt)],
            [ox + w - 1 - max(0, tilt), oy + h - 1],
            [ox, oy + h - 1 - max(0, -tilt)],
        ],
        np.float32,
    )
    matrix = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(resized, matrix, (frame_w, frame_h))
    mask = cv2.warpPerspective(np.full((h, w), 255, np.uint8), matrix, (frame_w, frame_h))
    scene[mask > 0] = warped[mask > 0]

    if rng.random() < 0.4:
        scene = cv2.GaussianBlur(scene, (3, 3), 0)

    moved: dict[str, tuple[float, float, float, float]] = {}
    for name, (fx1, fy1, fx2, fy2) in boxes.items():
        corners = np.array(
            [[[fx1 * w, fy1 * h]], [[fx2 * w, fy1 * h]], [[fx2 * w, fy2 * h]], [[fx1 * w, fy2 * h]]],
            np.float32,
        )
        projected = cv2.perspectiveTransform(corners, matrix).reshape(4, 2)
        x1, y1 = projected.min(axis=0)
        x2, y2 = projected.max(axis=0)
        moved[name] = (
            max(0.0, x1 / frame_w),
            max(0.0, y1 / frame_h),
            min(1.0, x2 / frame_w),
            min(1.0, y2 / frame_h),
        )
    return scene, moved


def to_yolo(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x1, y1, x2, y2 = box
    return (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", default=str(REPO_ROOT / ".data" / "synthetic"))
    parser.add_argument("--count", type=int, default=80)
    parser.add_argument("--val-split", type=float, default=0.2)
    parser.add_argument("--profile", default="ir_national_card_front")
    parser.add_argument("--seed", type=int, default=7)
    args = parser.parse_args()

    profile = get_profile(args.profile)
    class_names = profile.class_names
    index_of = {name: i for i, name in enumerate(class_names)}
    unknown = [n for n in {*LAYOUT, "idcard"} if n not in index_of]
    if unknown:
        raise SystemExit(f"Synthetic layout has classes the profile does not declare: {unknown}")

    out = Path(args.out)
    rng = random.Random(args.seed)
    n_val = int(args.count * args.val_split)

    for split in ("train", "val"):
        (out / split / "images").mkdir(parents=True, exist_ok=True)
        (out / split / "labels").mkdir(parents=True, exist_ok=True)

    for i in range(args.count):
        split = "val" if i < n_val else "train"
        card, boxes = draw_card(rng)
        scene, moved = photograph(card, boxes, rng)
        stem = f"card_{i:04d}"
        cv2.imwrite(str(out / split / "images" / f"{stem}.jpg"), scene)
        lines = []
        for name, box in moved.items():
            cx, cy, bw, bh = to_yolo(box)
            if bw <= 0 or bh <= 0:
                continue
            lines.append(f"{index_of[name]} {cx:.6f} {cy:.6f} {bw:.6f} {bh:.6f}")
        (out / split / "labels" / f"{stem}.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    write_data_yaml(out, class_names, ["train", "val"])
    print(f"[done] {out}")
    print(f"  train {args.count - n_val} images, val {n_val} images, {len(class_names)} classes")
    print(f"  data.yaml: {out / 'data.yaml'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
