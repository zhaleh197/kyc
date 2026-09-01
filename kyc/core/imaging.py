"""Image I/O and quality primitives shared by every module.

Kept dependency-light on purpose: numpy + opencv only, so this module
imports fine on a CPU-only box with no torch installed.
"""

from __future__ import annotations

import base64
import binascii

import cv2
import numpy as np

from .errors import ImageQualityError, InvalidInput

MAX_DECODE_PIXELS = 50_000_000  # guards against decompression bombs


def decode_image(data: bytes) -> np.ndarray:
    """bytes -> BGR uint8 array. Raises InvalidInput on anything undecodable."""
    if not data:
        raise InvalidInput("Empty image payload")
    buf = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise InvalidInput("Could not decode image; expected jpg/png/webp/bmp")
    if img.shape[0] * img.shape[1] > MAX_DECODE_PIXELS:
        raise InvalidInput("Image resolution exceeds the safety limit")
    return img


def decode_base64_image(payload: str) -> np.ndarray:
    """Accepts raw base64 or a full `data:image/jpeg;base64,...` URL."""
    if not payload:
        raise InvalidInput("Empty base64 payload")
    if payload.startswith("data:"):
        _, _, payload = payload.partition(",")
    try:
        raw = base64.b64decode(payload, validate=True)
    except (binascii.Error, ValueError) as exc:
        raise InvalidInput(f"Malformed base64: {exc}") from exc
    return decode_image(raw)


def encode_jpeg_base64(img: np.ndarray, quality: int = 90) -> str:
    ok, buf = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        raise InvalidInput("JPEG encoding failed")
    return base64.b64encode(buf.tobytes()).decode("ascii")


def limit_size(img: np.ndarray, max_side: int) -> np.ndarray:
    """Downscale so the long side is at most `max_side`. Never upscales."""
    h, w = img.shape[:2]
    longest = max(h, w)
    if longest <= max_side:
        return img
    scale = max_side / longest
    return cv2.resize(img, (int(round(w * scale)), int(round(h * scale))), interpolation=cv2.INTER_AREA)


def ensure_min_size(img: np.ndarray, min_side: int) -> None:
    h, w = img.shape[:2]
    if min(h, w) < min_side:
        raise ImageQualityError(
            f"Image is too small ({w}x{h}); the short side must be at least {min_side}px",
            details={"width": w, "height": h, "min_side": min_side},
        )


def sharpness(img: np.ndarray) -> float:
    """Variance of the Laplacian. Higher = sharper. Scale depends on resolution,
    so only compare values computed on similarly sized images."""
    gray = to_gray(img)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def brightness(img: np.ndarray) -> float:
    """Mean luma, 0-255."""
    return float(to_gray(img).mean())


def glare_ratio(img: np.ndarray, saturation_level: int = 250) -> float:
    """Fraction of near-blown-out pixels. High on a card photographed under a
    direct light source or behind plastic film, which is what kills OCR."""
    gray = to_gray(img)
    return float((gray >= saturation_level).mean())


def to_gray(img: np.ndarray) -> np.ndarray:
    if img.ndim == 2:
        return img
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def crop_with_padding(img: np.ndarray, box: tuple[int, int, int, int], pad_ratio: float) -> np.ndarray:
    """Crop x1,y1,x2,y2 with a relative margin, clamped to the image bounds."""
    h, w = img.shape[:2]
    x1, y1, x2, y2 = box
    pad_x = int(round((x2 - x1) * pad_ratio))
    pad_y = int(round((y2 - y1) * pad_ratio))
    x1 = max(0, x1 - pad_x)
    y1 = max(0, y1 - pad_y)
    x2 = min(w, x2 + pad_x)
    y2 = min(h, y2 + pad_y)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((1, 1, 3), np.uint8)
    return img[y1:y2, x1:x2].copy()


def upscale_to_height(img: np.ndarray, target_h: int) -> np.ndarray:
    """Text recognisers do badly on tiny crops; bump short lines up with cubic
    interpolation before handing them over."""
    h, w = img.shape[:2]
    if h >= target_h or h == 0:
        return img
    scale = target_h / h
    return cv2.resize(img, (max(1, int(round(w * scale))), target_h), interpolation=cv2.INTER_CUBIC)
