"""Zip the synthetic name lines for upload as a private Kaggle dataset.

    python -m training.kaggle.pack_dataset            # -> .data/kaggle_upload/kyc-ocr-synth-fa-ckb.zip

Adds charset_current.txt: the character list inside the model the service
runs now, which the notebook checks PaddleOCR's dict against. Only the
generator's output goes in; nothing from a real card.
"""

from __future__ import annotations

import argparse
import sys
import zipfile
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def current_charset(model: Path) -> str:
    import onnxruntime as ort

    session = ort.InferenceSession(str(model), providers=["CPUExecutionProvider"])
    return session.get_modelmeta().custom_metadata_map["character"].rstrip("\n") + "\n"


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--src", type=Path, default=REPO_ROOT / ".data/ocr_synth")
    p.add_argument("--model", type=Path, default=REPO_ROOT / "models/ppocr_arabic_v5_rec.onnx")
    p.add_argument("--out", type=Path, default=REPO_ROOT / ".data/kaggle_upload/kyc-ocr-synth-fa-ckb.zip")
    args = p.parse_args()

    lists = {name: (args.src / name).read_text(encoding="utf-8").splitlines() for name in ("train.txt", "val.txt")}
    images = [line.split("\t", 1)[0] for rows in lists.values() for line in rows if line]
    missing = [rel for rel in images if not (args.src / rel).exists()]
    if missing:
        sys.exit(f"{len(missing)} listed images are missing, e.g. {missing[:3]}")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    # JPEGs do not compress further; storing them keeps packing fast.
    with zipfile.ZipFile(args.out, "w", zipfile.ZIP_STORED) as zf:
        for name in lists:
            zf.write(args.src / name, name)
        zf.writestr("charset_current.txt", current_charset(args.model))
        for i, rel in enumerate(images, 1):
            zf.write(args.src / rel, rel)
            if i % 50_000 == 0:
                print(f"  {i}/{len(images)}", flush=True)
    sizes = {k: len(v) for k, v in lists.items()}
    print(f"{args.out}  {args.out.stat().st_size / 2**20:.0f} MB  {sizes}")


if __name__ == "__main__":
    main()
