"""Report which model artifacts are present and whether they fit the profiles.

    python -m scripts.check_models

Run this after dropping a freshly exported model into `models/`. It catches
the two mistakes that are otherwise silent until a real document fails: a
missing artifact, and a detector whose class count disagrees with the document
profile it is supposed to serve.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from kyc.core.config import get_settings  # noqa: E402
from kyc.modules.ocr.profiles import list_profiles  # noqa: E402
from kyc.modules.ocr.recognizers import CRNN_ARTIFACTS  # noqa: E402


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def describe(path: Path) -> str:
    if not path.exists():
        return "MISSING"
    return f"{path.stat().st_size / 1e6:7.1f} MB  sha256:{sha256(path)[:16]}"


def check_detector_shape(path: Path, class_count: int, providers: list[str]) -> str:
    """The YOLO head emits 4 box channels + one score per class (+32 mask
    coefficients for a segmentation export)."""
    try:
        import onnxruntime as ort
    except ImportError:
        return "onnxruntime not installed; shape not verified"

    try:
        session = ort.InferenceSession(str(path), providers=providers)
        shape = session.get_outputs()[0].shape
    except Exception as exc:  # noqa: BLE001 - report anything the runtime rejects
        return f"FAILED to load: {exc}"

    channels = next((d for d in shape if isinstance(d, int) and d not in (1,)), None)
    if channels is None:
        return f"output shape {shape} (dynamic; not verified)"
    expected_det = 4 + class_count
    expected_seg = expected_det + 32
    if channels in (expected_det, expected_seg):
        kind = "detection" if channels == expected_det else "segmentation"
        return f"OK  {kind} head, {class_count} classes, output {shape}"
    return (
        f"MISMATCH  output {shape} implies {channels - 4} classes, profile declares {class_count}. "
        "Retrain, or fix class_names in the profile."
    )


def main() -> int:
    settings = get_settings()
    print(f"models dir: {settings.models_dir}\n")
    problems = 0

    for profile in list_profiles():
        path = settings.model_path(profile.detector_model)
        print(f"[{profile.id}] {profile.detector_model}")
        print(f"    {describe(path)}")
        if path.exists():
            print(f"    {check_detector_shape(path, len(profile.class_names), settings.onnx_providers)}")
        else:
            problems += 1
        print()

    print("recognisers (optional; only needed when the backend is selected)")
    for backend_id, (model_file, charset_file) in CRNN_ARTIFACTS.items():
        model_path = settings.model_path(model_file)
        charset_path = settings.model_path(charset_file)
        print(f"  {backend_id}")
        print(f"    {model_file:<28} {describe(model_path)}")
        print(f"    {charset_file:<28} {describe(charset_path)}")
        if charset_path.exists():
            symbols = [s for s in charset_path.read_text(encoding="utf-8").splitlines() if s]
            print(f"    charset: {len(symbols)} symbols (+1 CTC blank)")

    active = {settings.ocr.text_backend, settings.ocr.digit_backend}
    print(f"\nactive backends: text={settings.ocr.text_backend} digits={settings.ocr.digit_backend}")
    for backend_id in active & set(CRNN_ARTIFACTS):
        model_file, charset_file = CRNN_ARTIFACTS[backend_id]
        if not settings.model_path(model_file).exists():
            print(f"  ERROR {backend_id} is selected but {model_file} is missing")
            problems += 1

    print("\n" + ("all required artifacts present" if problems == 0 else f"{problems} problem(s) found"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
