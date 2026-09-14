"""Central configuration.

Nothing in this project hard-codes a threshold. Every number that a
fraud analyst might want to tune lives here and is overridable through
environment variables (prefix `KYC_`) or a `.env` file.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

REPO_ROOT = Path(__file__).resolve().parents[2]


class OcrSettings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KYC_OCR_", env_file=".env", extra="ignore")

    # --- input gating -------------------------------------------------
    min_image_side: int = Field(400, description="Reject documents smaller than this on the short side")
    max_image_side: int = Field(2400, description="Downscale anything larger before processing")
    min_sharpness: float = Field(60.0, description="Laplacian variance floor for the rectified card")
    min_brightness: float = Field(45.0, description="Mean luma floor (0-255)")
    max_brightness: float = Field(215.0, description="Mean luma ceiling (0-255)")
    max_glare_ratio: float = Field(0.06, description="Max fraction of near-saturated pixels on the card")

    # --- geometry ------------------------------------------------------
    # Whether to flatten the card before running the detector.
    #
    # This must match how the detector was trained. Rectifying at serving time
    # while the model was trained on whole frames (or the reverse) shows it a
    # distribution it never saw. The classical rectifier only finds an outline
    # when the card has a visible border against its background; on the real
    # dataset it succeeded on 12% of images, so training there runs unrectified
    # and this stays off.
    rectify: bool = False

    # --- field detector ------------------------------------------------
    detector_conf: float = 0.35
    detector_iou: float = 0.45
    detector_imgsz: int = 640
    field_crop_padding: float = Field(0.06, description="Fraction of box size added around each field crop")

    # --- recognition ---------------------------------------------------
    # Measured on the held-out eval_text.tsv/eval_digits.tsv split (see
    # models/manifest.yaml): easyocr still edges out the fine-tuned CRNN on
    # names (57.1% vs 38.8% exact-match), but the digit model flips this
    # decisively (68.4% vs 7.0%) - easyocr's own allowlist does not actually
    # restrict its output (confirmed directly against the library, see
    # training/kaggle/README.md), while crnn_onnx.py's masking does. Keep
    # re-measuring before changing either default; this is not a permanent
    # verdict on either architecture, just what the current training data
    # supports today.
    text_backend: str = Field("easyocr", description="Backend id for free-text fields")
    digit_backend: str = Field("crnn_onnx_digits", description="Backend id for numeric fields")
    min_field_confidence: float = Field(0.40, description="Below this a field is reported but flagged low-confidence")
    upscale_small_crops_to: int = Field(64, description="Field crops shorter than this are upscaled before OCR")

    # --- decision ------------------------------------------------------
    review_on_low_confidence: bool = True


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="KYC_", env_file=".env", extra="ignore")

    app_name: str = "KYC API"
    debug: bool = False

    models_dir: Path = REPO_ROOT / "models"
    # Inference is CPU-only by design: models are trained on Kaggle GPUs,
    # exported to ONNX, and served here. Flip to a CUDA provider only if a
    # deployment actually has a GPU.
    onnx_providers: list[str] = Field(default_factory=lambda: ["CPUExecutionProvider"])
    onnx_intra_threads: int = 0  # 0 = let onnxruntime decide

    cors_origins: list[str] = Field(default_factory=list, description="Empty = same-origin only")
    max_upload_bytes: int = 12 * 1024 * 1024

    ocr: OcrSettings = Field(default_factory=OcrSettings)

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


@lru_cache
def get_settings() -> Settings:
    return Settings()
