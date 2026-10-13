"""Face sharpness, shared by face capture, anti-spoof and liveness."""

from __future__ import annotations

import cv2
import numpy as np

from .detector import Face

CROP_SIZE = 256
# Sharpness is the Laplacian variance of the face crop AFTER a light blur.
# Raw Laplacian variance counts sensor noise as detail: a deliberately
# shaken phone selfie (indoor light, full-resolution still) measured 256.9 -
# 21x the floor - and 2.2x "sharper" than the video frames before it. With
# noise added to synthetic motion blur, the raw metric caught 40% of it; with
# this denoising, 83% (still < 0.7x the stream), no steady still flagged.
SHARPNESS_DENOISE_SIGMA = 1.2


def face_crop_gray(img: np.ndarray, face: Face) -> np.ndarray:
    """The face box as a CROP_SIZE x CROP_SIZE grayscale image, so measures do
    not depend on camera resolution."""
    x1, y1, x2, y2 = face.int_box()
    crop = img[max(0, y1) : max(0, y2), max(0, x1) : max(0, x2)]
    if crop.size == 0:
        return np.zeros((CROP_SIZE, CROP_SIZE), np.uint8)
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (CROP_SIZE, CROP_SIZE), interpolation=cv2.INTER_AREA)


def face_sharpness(gray_crop: np.ndarray) -> float:
    """Laplacian variance of a face crop after denoising (see
    SHARPNESS_DENOISE_SIGMA). `gray_crop` is face_crop_gray()'s output."""
    denoised = cv2.GaussianBlur(gray_crop, (0, 0), SHARPNESS_DENOISE_SIGMA)
    return float(cv2.Laplacian(denoised, cv2.CV_64F).var())
