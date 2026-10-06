"""Calibrate the face-capture thresholds on labelled webcam frames.

Save frames from the dev page (/dev/face-capture, KYC_DEBUG=true) in
calibration mode; each file is named `<label>__<timestamp>.jpg`. Then:

    python -m scripts.calibrate_face_quality --dir path/to/frames [--csv out.csv]

For each label it reports how often the check that label should trigger
actually fired, how often "normal"/"headscarf" frames were wrongly blocked,
and the range of the metric behind each check - next to the same metric on
normal frames - so a threshold can be chosen between the two. Runs the real
models through the same pipeline code the API uses; nothing leaves the machine.
"""

from __future__ import annotations

import argparse
import csv
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from kyc.core.config import get_settings
from kyc.modules.face_capture.pipeline import FaceCapturePipeline

# label -> (check that should fire, metric behind it, how to read the metric)
EXPECTED = {
    "eyes_closed": ("eyes_closed", "eye_openness_min", "lower = more closed"),
    "mouth_open": ("mouth_open", "mouth_openness", "higher = more open"),
    "gaze_left": ("gaze_off_center", "gaze_offset_abs", "higher = further off-centre"),
    "gaze_right": ("gaze_off_center", "gaze_offset_abs", "higher = further off-centre"),
    "head_turned": ("head_turned", "yaw_abs", "degrees"),
    "head_up": ("head_turned", "pitch", "degrees, + = chin up"),
    "head_down": ("head_turned", "pitch", "degrees, - = chin down"),
    "head_tilted": ("head_tilted", "roll_abs", "degrees"),
    "mask": ("face_occluded", "clear_logit", "lower = more covered"),
    "sunglasses": ("face_occluded", "clear_logit", "lower = more covered"),
    "hand": ("face_occluded", "clear_logit", "lower = more covered"),
    "dark": ("face_too_dark", "brightness", "mean luma"),
    "bright": ("face_too_bright", "brightness", "mean luma"),
    "blurry": ("face_blurry", "sharpness", "lower = blurrier"),
    "far": ("face_too_small", "face_height_ratio", "face height / frame height"),
    "near": ("face_too_large", "face_height_ratio", "face height / frame height"),
}
SHOULD_PASS = {"normal", "headscarf"}


def derived(metrics: dict) -> dict:
    m = dict(metrics)
    if "eye_openness_left" in m:
        m["eye_openness_min"] = min(m["eye_openness_left"], m["eye_openness_right"])
    for key in ("gaze_offset", "yaw", "roll"):
        if key in m:
            m[f"{key}_abs"] = abs(m[key])
    return m


def describe(values: list[float]) -> str:
    if not values:
        return "-"
    a = np.array(values)
    p10, median = np.percentile(a, 10), np.median(a)
    return f"n={len(a):3d}  min {a.min():8.3f}  p10 {p10:8.3f}  median {median:8.3f}  max {a.max():8.3f}"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dir", required=True, type=Path)
    parser.add_argument("--csv", type=Path, help="Write every frame's metrics and codes here")
    args = parser.parse_args()

    pipeline = FaceCapturePipeline(get_settings())
    rows: list[dict] = []
    for path in sorted(args.dir.glob("*.jp*g")):
        label = path.name.split("__", 1)[0]
        img = cv2.imread(str(path))
        if img is None:
            print(f"skip unreadable {path.name}")
            continue
        a = pipeline.assess(img)
        errors = sorted(r.code for r in a.reasons if r.severity.value == "error")
        rows.append({"file": path.name, "label": label, "blocked": bool(errors), "errors": errors,
                     "all": sorted(r.code for r in a.reasons), **derived(a.metrics)})
    if not rows:
        raise SystemExit(f"No labelled .jpg files in {args.dir}")

    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    normal = [r for lbl in SHOULD_PASS for r in by_label.get(lbl, [])]

    print(f"\n{len(rows)} frames, labels: {', '.join(f'{k}={len(v)}' for k, v in sorted(by_label.items()))}\n")
    for label in sorted(SHOULD_PASS & by_label.keys()):
        group = by_label[label]
        blocked = [r for r in group if r["blocked"]]
        print(f"[{label}] should pass: {len(group) - len(blocked)}/{len(group)} passed")
        for code, n in Counter(c for r in blocked for c in r["errors"]).most_common():
            print(f"    wrongly blocked by {code}: {n}")

    for label in sorted(by_label.keys() - SHOULD_PASS):
        group = by_label[label]
        if label not in EXPECTED:
            print(f"\n[{label}] unknown label - expected one of {sorted(EXPECTED)}")
            continue
        code, metric, unit = EXPECTED[label]
        hits = sum(code in r["errors"] for r in group)
        print(f"\n[{label}] expected '{code}': fired on {hits}/{len(group)}")
        others = Counter(c for r in group for c in r["errors"] if c != code)
        if others:
            print(f"    other blocks: {dict(others)}")
        print(f"    {metric} ({unit})")
        print(f"      {label:>11s}: {describe([r[metric] for r in group if metric in r])}")
        print(f"      {'normal':>11s}: {describe([r[metric] for r in normal if metric in r])}")

    if args.csv:
        keys = sorted({k for r in rows for k in r})
        with args.csv.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=keys)
            writer.writeheader()
            for r in rows:
                writer.writerow({k: (";".join(v) if isinstance(v, list) else v) for k, v in r.items()})
        print(f"\nper-frame metrics written to {args.csv}")


if __name__ == "__main__":
    main()
