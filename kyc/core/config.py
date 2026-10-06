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
    # 60 was calibrated assuming a rectified, cropped card (little background
    # in frame). With rectify=False (see below) this metric runs on the whole
    # photo instead, and most of that frame is smooth background/table, which
    # drags the measured variance down regardless of how sharp the card
    # itself is - real, clearly legible test photos measured 17-52, well
    # under the old default, and were rejected outright. Lowered to a floor
    # that still catches a genuinely out-of-focus frame without false-
    # rejecting normal photos in this operating mode. The correct long-term
    # fix is measuring sharpness on the detected card region specifically
    # (once the field detector has located it) rather than the whole scene -
    # not done yet, tracked as a follow-up.
    min_sharpness: float = Field(
        12.0, description="Laplacian variance floor, measured on the whole frame when rectify=false"
    )
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


class FaceQualitySettings(BaseSettings):
    """Face capture (selfie) gate. See specs/02-face-quality.md.

    Unless a comment says otherwise, the defaults below are first guesses
    checked only against ~15 still photos (phase*/test images), NOT against
    frames from the real browser capture path. Recalibrate them on webcam
    frames before relying on them (spec 02, Q6).
    """

    model_config = SettingsConfigDict(env_prefix="KYC_FACE_QUALITY_", env_file=".env", extra="ignore")

    detector_model: str = "buffalo_l_det_10g.onnx"
    landmark_model: str = "buffalo_l_2d106det.onnx"
    detector_conf: float = 0.5
    min_image_side: int = Field(240, description="Frames smaller than this on the short side are rejected")

    # --- framing -------------------------------------------------------
    min_face_height_ratio: float = Field(0.20, description="Face box height / frame height floor")
    max_face_height_ratio: float = Field(0.80, description="Face box height / frame height ceiling")
    max_center_offset: float = Field(
        0.25, description="Face centre distance from frame centre, as a fraction of frame size"
    )
    # A second face counts only if it is at least this fraction of the main
    # face's height - a poster far behind the user should not block capture.
    multiple_face_min_ratio: float = 0.35
    # Largest box scale (face box enlarged about its centre) that still fits
    # in the frame. Anti-spoof's MiniFASNet models look at the face at 2.7x
    # and 4.0x context; a close-up leaves no room for that. 0 disables the
    # check until spec 04 measures how much context it really needs.
    min_context_scale: float = 0.0

    # --- image quality on the face crop --------------------------------
    # Laplacian variance on the face crop resized to 256x256, so the number
    # does not depend on camera resolution. Test photos: 10-573 sharp, 3-47
    # after a sigma=3 blur - the ranges overlap; webcam calibration needed.
    min_sharpness: float = 12.0
    min_brightness: float = Field(50.0, description="Mean luma floor on the face crop (0-255)")
    max_brightness: float = Field(210.0, description="Mean luma ceiling on the face crop (0-255)")
    max_illumination_asymmetry: float = Field(
        0.5, description="|left half - right half| / mean luma on the face crop; above this is reported (info)"
    )

    # --- pose and eyes -------------------------------------------------
    # Normal test photos measured yaw -9..+9, pitch -19..+15, roll -6..+4
    # (one deliberately tilted photo: roll -21).
    max_yaw_deg: float = 20.0
    max_pitch_deg: float = 25.0
    max_roll_deg: float = 15.0
    # Eye aspect ratio; open eyes measured 0.24-0.35 on the test photos. No
    # closed-eye samples yet - the floor is the usual EAR rule of thumb.
    min_eye_openness: float = 0.15

    # --- session -------------------------------------------------------
    frames_required_ok: int = Field(2, description="Consecutive passing frames before the selfie is captured")
    session_ttl_s: int = 120
    max_frames_per_session: int = 600
    selfie_jpeg_quality: int = 95


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
    face_quality: FaceQualitySettings = Field(default_factory=FaceQualitySettings)

    def model_path(self, name: str) -> Path:
        return self.models_dir / name


@lru_cache
def get_settings() -> Settings:
    return Settings()
