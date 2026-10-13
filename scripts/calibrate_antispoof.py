"""Calibrate the anti-spoof thresholds on labelled selfies.

On the dev page (/dev/face-capture), after a selfie is captured, save it
with a label - real, phone_screen, monitor, print. Each file is named
`<label>__<time>.jpg` and is exactly the image anti-spoof judged. Then:

    python -m scripts.calibrate_antispoof --dir path/to/selfies [--csv out.csv]

For every image: the live score (same models, jitter and averaging as the
API) and the sharpness (face capture's denoised measure). Then, per candidate
threshold, the share of real selfies that pass and of attacks that get
through - and the sharpness range of real selfies, for the "retake" floor.
Nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import cv2
import numpy as np

from kyc.core.config import get_settings
from kyc.core.face.detector import FaceDetector
from kyc.modules.antispoof.pipeline import AntiSpoofPipeline
from kyc.modules.face_capture.quality import face_crop_gray, face_sharpness
from kyc.modules.face_capture.session import Capture

ATTACKS = ("monitor", "phone_screen", "print")
THRESHOLDS = (0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9)


def score_folder(folder: Path) -> tuple[list[dict], float]:
    s = get_settings()
    detector = FaceDetector.load(s.model_path(s.face_quality.detector_model), s.onnx_providers)
    configured_floor = s.antispoof.min_sharpness
    # Score every image regardless of sharpness: calibration needs the live
    # score even where the API would answer "retake".
    s.antispoof.min_sharpness = 0.0
    store: dict[str, Capture] = {}
    pipeline = AntiSpoofPipeline(s, capture_lookup=store.get)

    rows = []
    for path in sorted(folder.glob("*.jp*g")):
        label = path.name.split("__", 1)[0]
        img = cv2.imread(str(path))
        faces = detector.detect(img) if img is not None else []
        if not faces:
            print(f"skip (no face): {path.name}")
            continue
        sharp = face_sharpness(face_crop_gray(img, faces[0]))
        _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        store["c"] = Capture(
            capture_id="c", session_id="calibration", jpeg=buf.tobytes(), face_box=faces[0].box,
            kps=faces[0].kps, metrics={"sharpness": sharp}, score=1.0,
        )
        result = pipeline.verify("c")
        rows.append({
            "file": path.name,
            "label": label,
            "live": result.data["live_score"],
            "sharpness": round(sharp, 3),
            "per_model": [m["live_score"] for m in result.data["models"]],
            "face_height_ratio": round(faces[0].height / img.shape[0], 3),
        })
    return rows, configured_floor


def report(rows: list[dict], configured_floor: float) -> None:
    real = [r for r in rows if r["label"] == "real"]
    attacks = [r for r in rows if r["label"] in ATTACKS]
    counts = ", ".join(f"{lbl}={sum(r['label'] == lbl for r in rows)}" for lbl in ATTACKS)
    print()
    print(f"{len(rows)} selfies: {len(real)} real, {len(attacks)} attacks ({counts})")
    print()
    for label in ("real", *ATTACKS):
        group = [r for r in rows if r["label"] == label]
        if not group:
            continue
        live = np.array([r["live"] for r in group])
        sharp = np.array([r["sharpness"] for r in group])
        print(
            f"[{label:12s}] live: min {live.min():.3f} median {np.median(live):.3f} max {live.max():.3f}"
            f" | sharpness: min {sharp.min():.2f} median {np.median(sharp):.2f}"
        )
    if not real:
        return
    print()
    print("accept_above   real pass   attacks through")
    for t in THRESHOLDS:
        passed = sum(r["live"] >= t for r in real)
        through = sum(r["live"] >= t for r in attacks)
        line = f"   {t:.2f}        {passed:3d}/{len(real):<3d}"
        print(line + (f"     {through:3d}/{len(attacks)}" if attacks else ""))
    lowest = min(r["sharpness"] for r in real)
    print()
    print(f"real selfies' lowest sharpness: {lowest:.2f} "
          f"(the 'retake' floor must sit below it; current {configured_floor})")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--csv", type=Path)
    args = parser.parse_args()

    rows, configured_floor = score_folder(args.dir)
    if not rows:
        raise SystemExit(f"No labelled .jpg files in {args.dir}")
    report(rows, configured_floor)
    if args.csv:
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"per-selfie results written to {args.csv}")


if __name__ == "__main__":
    main()
