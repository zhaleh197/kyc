"""Train the ID-card field detector on a Kaggle GPU and export it to ONNX.

    python train_field_detector.py --data /kaggle/input/ir-card/data.yaml \
                                   --profile ir_national_card_front \
                                   --epochs 120 --out /kaggle/working/models

The only artifact that matters is the exported `.onnx`. Copy it into `models/`
in the repo and inference runs on CPU through onnxruntime - no ultralytics,
no torch, no Roboflow API key at serving time.

Class order is checked against the document profile before exporting. That
check exists because a mismatch is silent and catastrophic: the pipeline would
happily write the father's name into the national id field.
"""

from __future__ import annotations

import argparse
import contextlib
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data", required=True, help="Path to the YOLO data.yaml")
    parser.add_argument("--profile", default="ir_national_card_front", help="Document profile id to validate against")
    parser.add_argument("--model", default="yolov8n.pt", help="Base weights; n/s are enough for CPU serving")
    parser.add_argument("--epochs", type=int, default=120)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--device", default="0", help="'0' for the Kaggle GPU, 'cpu' to smoke-test")
    parser.add_argument("--opset", type=int, default=12)
    parser.add_argument("--out", default=str(REPO_ROOT / "models"), help="Where to copy the exported model")
    parser.add_argument("--skip-class-check", action="store_true", help="Export even if class order disagrees")
    return parser.parse_args()


def expected_classes(profile_id: str) -> tuple[str, ...]:
    from kyc.modules.ocr.profiles import get_profile

    return get_profile(profile_id).class_names


def check_class_order(trained: dict[int, str], expected: tuple[str, ...], strict: bool) -> None:
    """`trained` is ultralytics' {index: name} mapping from the dataset yaml."""
    actual = tuple(trained[i] for i in sorted(trained))
    if actual == expected:
        print(f"[ok] class order matches the profile: {actual}")
        return

    message = (
        "Class order mismatch between the dataset and the document profile.\n"
        f"  dataset : {actual}\n"
        f"  profile : {expected}\n"
        "Fix the `names:` list in data.yaml to match the profile's class_names, "
        "or update class_names in the profile module."
    )
    if strict:
        raise SystemExit("[fatal] " + message)
    print("[warn] " + message)


def main() -> None:
    args = parse_args()
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise SystemExit("ultralytics is not installed. On Kaggle: pip install ultralytics onnx onnxsim") from exc

    expected = expected_classes(args.profile)

    model = YOLO(args.model)
    model.train(
        data=args.data,
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        device=args.device,
        # No `project`: ultralytics resolves a relative project under its own
        # runs_dir and ends up with runs/detect/runs/<name>. Letting it use the
        # default gives the expected runs/detect/<name>.
        name=args.profile,
        exist_ok=True,
        # The card is rectified before detection, so aggressive geometric
        # augmentation teaches the model a distribution it will never see.
        degrees=5.0,
        translate=0.05,
        scale=0.15,
        shear=2.0,
        perspective=0.0005,
        fliplr=0.0,   # never mirror an ID card: the layout is not symmetric
        mosaic=0.3,
    )

    check_class_order(model.names, expected, strict=not args.skip_class_check)

    metrics = model.val()
    print(f"[val] mAP50={metrics.box.map50:.4f}  mAP50-95={metrics.box.map:.4f}")
    for index, name in sorted(model.names.items()):
        # Per-class AP is not always populated, depending on the split.
        with contextlib.suppress(IndexError, TypeError):
            print(f"       {name:<12} AP50={metrics.box.ap50[index]:.4f}")

    exported = model.export(
        format="onnx",
        opset=args.opset,
        imgsz=args.imgsz,
        simplify=True,
        dynamic=False,   # a static shape is measurably faster on CPU
        nms=False,       # NMS runs in kyc/modules/ocr/detector.py
    )

    from kyc.modules.ocr.profiles import get_profile

    destination = Path(args.out) / get_profile(args.profile).detector_model
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy(exported, destination)
    print(f"[done] exported -> {destination}")
    print("Copy it into models/ in the repo; no env change is needed.")


if __name__ == "__main__":
    main()
