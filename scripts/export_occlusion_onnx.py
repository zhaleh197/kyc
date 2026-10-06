"""Export the prototype's face-occlusion MobileNetV3 to ONNX for serving.

    python -m scripts.export_occlusion_onnx

Needs torch + torchvision (dev only - serving uses onnxruntime). Verifies the
ONNX output against torch on random input before writing the manifest hash.
"""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="phase1_facedetect/obstruction_detector50.pt")
    parser.add_argument("--out", default="models/face_occlusion_mnv3.onnx")
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    model = torch.load(args.src, weights_only=False, map_location="cpu").eval()
    dummy = torch.zeros(1, 3, 224, 224)
    torch.onnx.export(
        model, dummy, args.out, input_names=["input"], output_names=["logit"],
        dynamic_axes={"input": {0: "n"}, "logit": {0: "n"}}, opset_version=17, dynamo=False,
    )

    x = np.random.default_rng(0).standard_normal((4, 3, 224, 224)).astype(np.float32)
    with torch.no_grad():
        expected = model(torch.from_numpy(x)).numpy()
    got = ort.InferenceSession(args.out, providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
    diff = float(np.abs(expected - got).max())
    if diff > 1e-3:
        raise SystemExit(f"ONNX output differs from torch by {diff}")
    digest = hashlib.sha256(Path(args.out).read_bytes()).hexdigest()
    print(f"wrote {args.out}  max |torch - onnx| = {diff:.2e}  sha256 {digest}")


if __name__ == "__main__":
    main()
