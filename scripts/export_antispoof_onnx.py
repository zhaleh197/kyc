"""Export the prototype's MiniFASNet anti-spoof models to ONNX for serving.

    python -m scripts.export_antispoof_onnx

Needs torch (dev only - serving uses onnxruntime). Source: Minivision's
Silent-Face-Anti-Spoofing (Apache-2.0), vendored in phase3-antispoof/.

Input contract, read from the upstream code rather than assumed: an 80x80
crop, BGR channel order as OpenCV decodes it, raw 0-255 floats - upstream's
custom ToTensor does NOT divide by 255. Output: 3 logits; softmax class 1 is
"real", classes 0 and 2 are two kinds of spoof (upstream test.py treats only
label 1 as real).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from collections import OrderedDict
from pathlib import Path

import numpy as np

PROTOTYPE = Path("phase3-antispoof")
MODELS = {
    # source file -> (exported name, class name)
    "2.7_80x80_MiniFASNetV2.pth": ("antispoof_minifasnet_v2_s2.7.onnx", "MiniFASNetV2"),
    "4_0_0_80x80_MiniFASNetV1SE.pth": ("antispoof_minifasnet_v1se_s4.0.onnx", "MiniFASNetV1SE"),
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="models")
    args = parser.parse_args()

    import onnxruntime as ort
    import torch

    sys.path.insert(0, str(PROTOTYPE))
    from src.model_lib import MiniFASNet  # noqa: E402 - vendored upstream code

    for src_name, (out_name, cls_name) in MODELS.items():
        model = getattr(MiniFASNet, cls_name)(conv6_kernel=(5, 5))  # (80+15)//16 = 5
        state = torch.load(PROTOTYPE / "resources" / "anti_spoof_models" / src_name, map_location="cpu")
        state = OrderedDict((k[7:] if k.startswith("module.") else k, v) for k, v in state.items())
        model.load_state_dict(state)
        model.eval()

        out = Path(args.out) / out_name
        torch.onnx.export(
            model, torch.zeros(1, 3, 80, 80), str(out), input_names=["input"], output_names=["logits"],
            dynamic_axes={"input": {0: "n"}, "logits": {0: "n"}}, opset_version=17, dynamo=False,
        )

        x = np.random.default_rng(0).uniform(0, 255, (4, 3, 80, 80)).astype(np.float32)
        with torch.no_grad():
            expected = model(torch.from_numpy(x)).numpy()
        got = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"]).run(None, {"input": x})[0]
        diff = float(np.abs(expected - got).max())
        if diff > 1e-3:
            raise SystemExit(f"{out_name}: ONNX differs from torch by {diff}")
        digest = hashlib.sha256(out.read_bytes()).hexdigest()
        print(f"{out_name}: max |torch - onnx| = {diff:.2e}  sha256 {digest}")


if __name__ == "__main__":
    main()
